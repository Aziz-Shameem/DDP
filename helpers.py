import tensorflow as tf
from tensorflow import keras

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from torch import rand
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from datasets import load_dataset

def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
    device  = None,
    output_type = "np",
):
    """
    Creates 2D sinusoidal positional embeddings.

    Args:
        embed_dim (`int`):
            The embedding dimension.
        grid_size (`int`):
            The size of the grid height and width.
        cls_token (`bool`, defaults to `False`):
            Whether or not to add a classification token.
        extra_tokens (`int`, defaults to `0`):
            The number of extra tokens to add.
        interpolation_scale (`float`, defaults to `1.0`):
            The scale of the interpolation.

    Returns:
        pos_embed (`torch.Tensor`):
            Shape is either `[grid_size * grid_size, embed_dim]` if not using cls_token, or `[1 + grid_size*grid_size,
            embed_dim]` if using cls_token
    """
    # if output_type == "np":
    #     deprecation_message = (
    #         "`get_2d_sincos_pos_embed` uses `torch` and supports `device`."
    #         " `from_numpy` is no longer required."
    #         "  Pass `output_type='pt' to use the new version now."
    #     )
    #     deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
    #     return get_2d_sincos_pos_embed_np(
    #         embed_dim=embed_dim,
    #         grid_size=grid_size,
    #         cls_token=cls_token,
    #         extra_tokens=extra_tokens,
    #         interpolation_scale=interpolation_scale,
    #         base_size=base_size,
    #     )
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        torch.arange(grid_size[0], device=device, dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], device=device, dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type=output_type)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.concat([torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, output_type="np"):
    r"""
    This function generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension.
        grid (`torch.Tensor`): Grid of positions with shape `(H * W,)`.

    Returns:
        `torch.Tensor`: The 2D sinusoidal positional embeddings with shape `(H * W, embed_dim)`
    """
    # if output_type == "np":
    #     deprecation_message = (
    #         "`get_2d_sincos_pos_embed_from_grid` uses `torch` and supports `device`."
    #         " `from_numpy` is no longer required."
    #         "  Pass `output_type='pt' to use the new version now."
    #     )
    #     deprecate("output_type=='np'", "0.33.0", deprecation_message, standard_warn=False)
    #     return get_2d_sincos_pos_embed_from_grid_np(
    #         embed_dim=embed_dim,
    #         grid=grid,
    #     )
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], output_type=output_type)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], output_type=output_type)  # (H*W, D/2)

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, output_type="np"):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`torch.Tensor`): 1D tensor of positions with shape `(M,)`

    Returns:
        `torch.Tensor`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    # if output_type == "np":
    #     deprecation_message = (
    #         "`get_1d_sincos_pos_embed_from_grid` uses `torch` and supports `device`."
    #         " `from_numpy` is no longer required."
    #         "  Pass `output_type='pt' to use the new version now."
    #     )
    #     deprecate("output_type=='np'", "0.34.0", deprecation_message, standard_warn=False)
    #     return get_1d_sincos_pos_embed_from_grid_np(embed_dim=embed_dim, pos=pos)
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # Sample t.
    t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)         # If force_t is not -1, then use force_t.
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]

    # Sample flow pairs x_t, v_t.
    if 'latent' in FLAGS.dataset_name:
        x_0 = images[..., :images.shape[-1] // 2]
        x_1 = images[..., images.shape[-1] // 2:]
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - 1e-5) * x_0
    else:
        x_1 = images
        x_0 = jax.random.normal(noise_key, images.shape)
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    return x_t, v_t, t, dt_base, labels_dropped, info