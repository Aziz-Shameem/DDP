import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn as nn
from torch import rand
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from datasets import load_dataset
from helpers import get_2d_sincos_pos_embed, get_targets
from absl import app, flags
import optax

# import wavemix
from wavemix import Level1Waveblock, DWTForward
from pywt import Wavelet
from torch_dwt import dwt2,idwt2
import math
from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# from math_utils import get_2d_sincos_pos_embed, modulate
from jax._src import core
from jax._src import dtypes
from jax._src.nn.initializers import _compute_fans

def xavier_uniform_pytorchlike():
    def init(key, shape, dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        if len(shape) == 2: # Dense, [in, out]
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4: # Conv, [k, k, in, out]. Assumes patch-embed style conv.
            fan_in = shape[0] * shape[1] * shape[2]
            fan_out = shape[3]
        else:
            raise ValueError(f"Invalid shape {shape}")

        variance = 2 / (fan_in + fan_out)
        scale = jnp.sqrt(3 * variance)
        param = jax.random.uniform(key, shape, dtype, -1) * scale

        return param
    return init


class TrainConfig:
    def __init__(self, dtype):
        self.dtype = dtype
    def kern_init(self, name='default', zero=False):
        if zero or 'bias' in name:
            return nn.initializers.constant(0)
        return xavier_uniform_pytorchlike()
    def default_config(self):
        return {
            'kernel_init': self.kern_init(),
            'bias_init': self.kern_init('bias', zero=True),
            'dtype': self.dtype,
        }

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    tc: TrainConfig
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02), 
                     bias_init=self.tc.kern_init('time_bias'), dtype=self.tc.dtype)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02), 
                     bias_init=self.tc.kern_init('time_bias'))(x)
        return x
    
    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        # t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        embedding = embedding.astype(self.tc.dtype)
        return embedding
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    num_classes: int
    hidden_size: int
    tc: TrainConfig

    @nn.compact
    def __call__(self, labels):
        embedding_table = nn.Embed(self.num_classes + 1, self.hidden_size, 
                                   embedding_init=nn.initializers.normal(0.02), dtype=self.tc.dtype)
        embeddings = embedding_table(labels)
        return embeddings
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    patch_size: int
    hidden_size: int
    tc: TrainConfig
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        patch_tuple = (self.patch_size, self.patch_size)
        num_patches = (H // self.patch_size)
        x = nn.Conv(self.hidden_size, patch_tuple, patch_tuple, use_bias=self.bias, padding="VALID",
                     kernel_init=self.tc.kern_init('patch'), bias_init=self.tc.kern_init('patch_bias', zero=True),
                     dtype=self.tc.dtype)(x) # (B, P, P, hidden_size)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        return x
    
class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    mlp_dim: int
    tc: TrainConfig
    out_dim: Optional[int] = None
    dropout_rate: float = None
    train: bool = False

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim, **self.tc.default_config())(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=(not self.train))(x)
        output = nn.Dense(features=actual_out_dim, **self.tc.default_config())(x)
        output = nn.Dropout(rate=self.dropout_rate, deterministic=(not self.train))(output)
        return output
    
def modulate(x, shift, scale):
    # scale = jnp.clip(scale, -1, 1)
    return x * (1 + scale[:, None]) + shift[:, None]
    
################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    tc: TrainConfig
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    train: bool = False

    # @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
    @nn.compact
    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        c = nn.silu(c)
        c = nn.Dense(6 * self.hidden_size, **self.tc.default_config())(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        
        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False, dtype=self.tc.dtype)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        channels_per_head = self.hidden_size // self.num_heads
        k = nn.Dense(self.hidden_size, **self.tc.default_config())(x_modulated)
        q = nn.Dense(self.hidden_size, **self.tc.default_config())(x_modulated)
        v = nn.Dense(self.hidden_size, **self.tc.default_config())(x_modulated)
        k = jnp.reshape(k, (k.shape[0], k.shape[1], self.num_heads, channels_per_head))
        q = jnp.reshape(q, (q.shape[0], q.shape[1], self.num_heads, channels_per_head))
        v = jnp.reshape(v, (v.shape[0], v.shape[1], self.num_heads, channels_per_head))
        q = q / q.shape[3] # (1/d) scaling.
        w = jnp.einsum('bqhc,bkhc->bhqk', q, k) # [B, HW, HW, num_heads]
        w = w.astype(jnp.float32)
        w = nn.softmax(w, axis=-1)
        y = jnp.einsum('bhqk,bkhc->bqhc', w, v) # [B, HW, num_heads, channels_per_head]
        y = jnp.reshape(y, x.shape) # [B, H, W, C] (C = heads * channels_per_head)
        attn_x = nn.Dense(self.hidden_size, **self.tc.default_config())(y)
        x = x + (gate_msa[:, None] * attn_x)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False, dtype=self.tc.dtype)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio), tc=self.tc, 
                         dropout_rate=self.dropout, train=self.train)(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    patch_size: int
    out_channels: int
    hidden_size: int
    tc: TrainConfig

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=self.tc.kern_init(zero=True), 
                     bias_init=self.tc.kern_init('bias', zero=True), dtype=self.tc.dtype)(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = nn.LayerNorm(use_bias=False, use_scale=False, dtype=self.tc.dtype)(x)
        x = modulate(x, shift, scale)
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, 
                     kernel_init=self.tc.kern_init('final', zero=True), 
                     bias_init=self.tc.kern_init('final_bias', zero=True), dtype=self.tc.dtype)(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    out_channels: int
    class_dropout_prob: float
    num_classes: int
    ignore_dt: bool = False
    dropout: float = 0.0
    dtype: Dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, t, dt, y, train=False, return_activations=False):
        # (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        print("DiT: Input of shape", x.shape, "dtype", x.dtype)
        activations = {}

        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        num_patches1d = input_size // self.patch_size 
        num_patches = (num_patches1d) ** 2
        num_patches_side = input_size // self.patch_size
        tc = TrainConfig(dtype=self.dtype)

        if self.ignore_dt:
            dt = jnp.zeros_like(t)
        
        # pos_embed = self.param("pos_embed", get_2d_sincos_pos_embed, self.hidden_size, num_patches)
        # pos_embed = jax.lax.stop_gradient(pos_embed)
        pos_embed = get_2d_sincos_pos_embed(None, self.hidden_size, num_patches1d)
        x = PatchEmbed(self.patch_size, self.hidden_size, tc=tc)(x) # (B, num_patches, hidden_size)
        print("DiT: After patch embed, shape is", x.shape, "dtype", x.dtype)
        activations['patch_embed'] = x

        x = x + pos_embed
        x = x.astype(self.dtype)
        te = TimestepEmbedder(self.hidden_size, tc=tc)(t) # (B, hidden_size)
        dte = TimestepEmbedder(self.hidden_size, tc=tc)(dt) # (B, hidden_size)
        ye = LabelEmbedder(self.num_classes, self.hidden_size, tc=tc)(y) # (B, hidden_size)
        c = te + ye + dte
        
        activations['pos_embed'] = pos_embed
        activations['time_embed'] = te
        activations['dt_embed'] = dte
        activations['label_embed'] = ye
        activations['conditioning'] = c

        print("DiT: Patch Embed of shape", x.shape, "dtype", x.dtype)
        print("DiT: Conditioning of shape", c.shape, "dtype", c.dtype)
        for i in range(self.depth):
            x = DiTBlock(self.hidden_size, self.num_heads, tc, self.mlp_ratio, self.dropout, train)(x, c)
            activations[f'dit_block_{i}'] = x
        x = FinalLayer(self.patch_size, self.out_channels, self.hidden_size, tc)(x, c) # (B, num_patches, p*p*c)
        activations['final_layer'] = x
        # print("DiT: FinalLayer of shape", x.shape, "dtype", x.dtype)
        x = jnp.reshape(x, (batch_size, num_patches_side, num_patches_side, 
                            self.patch_size, self.patch_size, self.out_channels))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=int(num_patches_side), W=int(num_patches_side))
        assert x.shape == (batch_size, input_size, input_size, self.out_channels)

        t_discrete = jnp.floor(t * 256).astype(jnp.int32)
        logvars = nn.Embed(256, 1, embedding_init=nn.initializers.constant(0))(t_discrete) * 100

        if return_activations:
            return x, logvars, activations
        return x


##################################################################################
#                              Training Code                                     #
##################################################################################

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_integer('seed', 10, 'Random seed.') # Must be the same across all processes.
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 32, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')
flags.DEFINE_string('mode', 'train', 'train or inference.')

np.random.seed(FLAGS.seed)
print("Using devices", jax.local_devices())
device_count = len(jax.local_devices())
global_device_count = jax.device_count()
print("Device count", device_count)
print("Global device count", global_device_count)
local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
print("Global Batch: ", FLAGS.batch_size)
print("Node Batch: ", local_batch_size)
print("Device Batch:", local_batch_size // device_count)

dataset = get_dataset(FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit)
dataset_valid = get_dataset(FLAGS.dataset_name, local_batch_size, False, FLAGS.debug_overfit)
example_obs, example_labels = next(dataset)
example_obs = example_obs[:1]
example_obs_shape = example_obs.shape

dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'out_channels': example_obs_shape[-1],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
        'dropout': FLAGS.model['dropout'],
        'ignore_dt': False if (FLAGS.model['train_type'] in ('shortcut', 'livereflow')) else True,
    }
model_def = DiT(**dit_args)
lr_schedule = optax.warmup_cosine_decay_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'], FLAGS.max_steps)
adam = optax.adamw(learning_rate=lr_schedule, b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], weight_decay=FLAGS.model['weight_decay'])
tx = optax.chain(adam)

@partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
def update(train_state, train_state_teacher, images, labels, force_t=-1, force_dt=-1):
    new_rng, targets_key, dropout_key, perm_key = jax.random.split(train_state.rng, 4)
    info = {}

    id_perm = jax.random.permutation(perm_key, images.shape[0])
    images = images[id_perm]
    labels = labels[id_perm]
    labels = jnp.ones(labels.shape[0], dtype=jnp.int32) * FLAGS.model['num_classes']
    x_t, v_t, t, dt_base, labels, info = get_targets(FLAGS, targets_key, train_state, images, labels, force_t, force_dt)

    def loss_fn(grad_params):
        v_prime, logvars, activations = train_state.call_model(x_t, t, dt_base, labels, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
        mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
        loss = jnp.mean(mse_v)

        info = {
            'loss': loss,
            'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
            **{'activations/' + k : jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
        }

        if FLAGS.model['train_type'] == 'shortcut' or FLAGS.model['train_type'] == 'livereflow':
            bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
            info['loss_flow'] = jnp.mean(mse_v[bootstrap_size:])
            info['loss_bootstrap'] = jnp.mean(mse_v[:bootstrap_size])
        
        return loss, info
    
    grads, new_info = jax.grad(loss_fn, has_aux=True)(train_state.params)
    info = {**info, **new_info}
    updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    info['grad_norm'] = optax.global_norm(grads)
    info['update_norm'] = optax.global_norm(updates)
    info['param_norm'] = optax.global_norm(new_params)
    info['lr'] = lr_schedule(train_state.step)

    train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
    train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
    return train_state, info

###################################
# Train Loop
###################################
start_step = 1
for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                    smoothing=0.1,
                    dynamic_ncols=True):
    
    # Sample data.
    if not FLAGS.debug_overfit or i == 1:
        batch_images, batch_labels = shard_data(*next(dataset))

    # Train update.
    train_state, update_info = update(train_state, None, batch_images, batch_labels)
