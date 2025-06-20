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

# device = 'cuda' # toggle here
xf1 = DWTForward(J=1, mode='zero', wave='db1') 

# running the complete experiment
def run(model, name, dataset_name='mnist', epochs=5, batchsize=64, timesteps=500, device='cuda', test=False) :
    batch_size = batchsize
    timesteps = timesteps

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name=='mnist' :
        dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        nchannels = 1
    elif dataset_name=='cifar10' :
        dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        nchannels = 3
    elif dataset_name=='tinyImageNet' :
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        ds = load_dataset("zh-plus/tiny-imagenet")
        dataset = TinyImageNet(ds, transforms=transform)
        nchannels = 3
    else :
        raise ValueError('dataset not supported')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define model and diffusion
    device = device
    print(f'{device=}')
    model.to(device)

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    nparams = sum([p.numel() for p in model.parameters()])/10**6

    # training 
    losslog = train(epochs, train_loader, model, gaussian_diffusion, optimizer, timesteps, device, test=test)
    
    # save model
    torch.save(model, f'{name}.pt')

    # visualize 
    imgs = gen_and_vis(gaussian_diffusion, model, dataset=dataset_name, show=False)

    # compute fid (500 images)
    if test : corpus = 2  
    else : corpus = 300
    temploader = torch.utils.data.DataLoader(dataset, batch_size=corpus, shuffle=False)
    i1 = next(iter(temploader))[0]
    if isinstance(i1, np.ndarray) : 
        i1 = torch.tensor(i1)
    if i1.dim()==3 : i1 = i1.unsqueeze(1)
    if dataset_name=='mnist' :
        gen_imgs = gaussian_diffusion.sample(model, 28, batch_size=corpus, channels=1)
        i2 = gen_imgs[-1].reshape(-1, 1, 28, 28)
        i2 = torch.tensor(i2)
    elif dataset_name=='cifar10' :
        gen_imgs = gaussian_diffusion.sample(model, 32, batch_size=corpus, channels=3)
        i2 = gen_imgs[-1].reshape(-1, 3, 32, 32)
        i2 = torch.tensor(i2)
        # i1 = i1.permute(0,3,1,2)
    elif dataset_name=='tinyImageNet' :
        gen_imgs = gaussian_diffusion.sample(model, 64, batch_size=corpus, channels=3)
        i2 = gen_imgs[-1].reshape(-1, 3, 64, 64)
        i2 = torch.tensor(i2)
        # i1 = i1.permute(0,3,1,2)
    fid = calculate_fid(i1, i2)

    # print summary 
    print('#'*35)
    print(' '*13 + 'SUMMARY' + ''*13)
    print('#'*35)
    print(f'Model : {name} on Dataset : {dataset_name}')
    print(f'Number of Parameters : {nparams} M')
    print(f'Train Epochs : {epochs}')
    print(f'FID : {fid}')
    print('Loss Progression')
    plt.plot(losslog); plt.title('Losslog'); plt.show()
    print('Sample Images Generated')
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    gs = fig.add_gridspec(8, 8)
    for n_row in range(8):
        for n_col in range(8):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            temp = (imgs[n_row, n_col]+1.0)/2
            if nchannels==1 : f_ax.imshow(temp.transpose((1,2,0)), cmap='gray')
            else : f_ax.imshow(temp.transpose((1,2,0)))
            f_ax.axis("off")
    plt.show()

    return nparams, fid, losslog, imgs, model

# FID Calculation
# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = np.resize(image, new_shape)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

# calculate frechet inception distance
def calculate_fid1(images1, images2, reshape=False):
    # loading model
    if not reshape : 
        inc_model = keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=images1.shape)
    else :
        inc_model = keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(150,150,3))
    # preprocessing
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    # resize images
    if reshape :
        images1 = scale_images(images1, (150,150,3))
        images2 = scale_images(images2, (150,150,3))

    images1 = keras.applications.inception_v3.preprocess_input(images1)
    images2 = keras.applications.inception_v3.preprocess_input(images2)
    # calculate activations
    act1 = inc_model.predict(images1)
    act2 = inc_model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False, dtype=np.csingle)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False, dtype=np.csingle)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = np.sqrt(sigma1*sigma2)
    # check and correct imaginary numbers from sqrt
    covmean = np.real(covmean)
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return np.real(fid)

def calculate_fid(images1, images2) :
    '''
    images1 : original dataset, shape (n,3,h,w)
    images1 : generated dataset, shape (n,3,h,w)
    '''
    # preprocess
    if images1.shape[1] == 1 :
        images1 = images1.repeat_interleave(3, dim=1)
        images2 = images2.repeat_interleave(3, dim=1)
    if images1.dtype != torch.uint8 :
        images1 = images1 + images1.min()
        images1 = (images1*255//images1.max()).type(torch.uint8)
    if images2.dtype != torch.uint8 :
        images2 = images2 + images2.min()
        images2 = (images2*255//images2.max()).type(torch.uint8)
    fid = FrechetInceptionDistance(feature=64)
    # generate two slightly overlapping image intensity distributions
    fid.update(images1, real=True)
    fid.update(images2, real=False)
    return fid.compute()

def report_fid(model, gaussian_diffusion, dataset, num_images) :
    i1 = dataset.data[:num_images]
    i2 = gaussian_diffusion.sample(model, 28, batch_size=num_images, channels=1)
    fid = calculate_fid(i1, i2)
    return fid

def train(epochs, train_loader, model, gaussian_diffusion, optimizer, timesteps, device, test=False) :
    losslog = []
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(tqdm(train_loader)):
            
            batch_size = images.shape[0]
            images = images.to(device)
            
            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            loss = gaussian_diffusion.train_losses(model, images, t)
            with torch.no_grad() : losslog.append(loss.item())
            
            if step % 100 == 0:
                tqdm.write(f"Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if test : break

    return losslog

def gen_and_vis(gaussian_diffusion, model, dataset='mnist', show=True) :
    nchannels = 1 if dataset=='mnist' else 3
    size = 28 if dataset=='mnist' else 32
    generated_images = gaussian_diffusion.sample(model, size, batch_size=64, channels=nchannels)
    imgs = generated_images[-1].reshape(8, 8, nchannels, size, size)

    if show :
        # generate new images
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        gs = fig.add_gridspec(8, 8)
        for n_row in range(8):
            for n_col in range(8):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                temp = (imgs[n_row, n_col]+1.0)/2
                if nchannels==1 : f_ax.imshow(temp.transpose((1,2,0)), cmap='gray')
                else : f_ax.imshow(temp.transpose((1,2,0)))
                f_ax.axis("off")
        plt.show()
    return imgs

# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# beta schedules
def linear_beta_schedule(timesteps):
    scale = 1 # 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# CLASSES

# for tiny image net
class TinyImageNet(Dataset) :
    def __init__(self, data, split='train', transforms=None) :
        assert split in ['train', 'test'], f'Split {split} not supported'
        self.data = data[split]
        self.transforms = transforms
    def __len__(self) :
        return len(self.data)
    def __getitem__(self, idx) :
        ret = self.transforms(np.array(self.data[idx]['image']))
        label = self.data[idx]['label']
        if ret.shape[-3]==1 :
            ret = torch.repeat_interleave(ret, repeats=3, dim=-3)
        return ret, label

class WaveMixSRV2Block(nn.Module):
    def __init__(
        self,
        *,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.5,
    ):
        super().__init__()
        
      
        self.feedforward = nn.Sequential(
                nn.Conv2d(final_dim, final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, ff_channel, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(ff_channel//4, final_dim,3, 1, 1),
                nn.BatchNorm2d(final_dim)
            
            )

        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
        
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.reduction(x)
        
        Y1, Yh = xf1(x)
        
        x = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        
        x = torch.cat((Y1,x), dim = 1)
        
        x = self.feedforward(x)
        
        return x

class WaveMixSRBlock(nn.Module):
    def __init__(
        self,
        *,
        depth,
        mult = 1,
        ff_channel = 16,
        final_dim = 32,
        time_dim = 32,
        in_channels = 1,
        dropout = 0.3,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        self.final = nn.Sequential(
            nn.Conv2d(final_dim,int(final_dim/2), 3, stride=1, padding=1),
            nn.Conv2d(int(final_dim/2), in_channels, 1)
        )
        self.path1 = nn.Sequential(
            nn.Upsample(scale_factor=1, mode='bicubic', align_corners = False),
            nn.Conv2d(in_channels, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )
        self.path2 = nn.Sequential(
            nn.Upsample(scale_factor=1, mode='bicubic', align_corners = False),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, final_dim),
            nn.SiLU(),
            nn.Linear(final_dim, final_dim),
        )

    def forward(self, img, t):
        t = self.time_mlp(timestep_embedding(t, self.time_dim))[:,:,None,None]
        img = self.path1(img) + t
        for attn in self.layers:
            img = attn(img) + img
        img = self.final(img)
        return img

class WaveMixSR(nn.Module):
    def __init__(
        self,
        *,
        depth,
        mult = 1,
        ff_channel = 16,
        final_dim = 32,
        time_dim = 32,
        in_channels = 1,
        repeat = 1,
        dropout = 0.3,
        device = 'cuda'
    ):
        super().__init__()
        self.modules1 = nn.ModuleList([WaveMixSRBlock(
                depth=depth, mult=mult, 
                ff_channel=ff_channel, final_dim=final_dim, in_channels=in_channels,
                time_dim=time_dim, dropout=dropout)
                for _ in range(repeat)]).to(device)

    def forward(self, img, t): 
        for i,mod in enumerate(self.modules1) :
            img = mod(img, t)
        return img
    
# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

# use GN for norm layer
def norm_layer(channels, short=False):
    if not short : return nn.GroupNorm(32, channels)
    return nn.GroupNorm(8, channels)

class ResidualBlock_Fixed(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels),
        )
        
        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_channels, out_channels),
            nn.SiLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(p=dropout),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout, shortnorm=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels, shortnorm),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            norm_layer(out_channels, shortnorm),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()   


    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class ResidualWaveBlock(TimestepBlock) :
    def __init__(self, in_channels, out_channels, time_channels, ff_channel=48, dropout=0.3, mult=2):
        super().__init__()
        self.feedforward = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(in_channels*mult, ff_channel, 1),
                nn.ConvTranspose2d(ff_channel, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels)
            
            )

        self.reduction = nn.Conv2d(in_channels, in_channels//4, 1)
        
        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_channels, in_channels),
            nn.SiLU()
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x, t) :
        """
        `x` has shape `[batch_size, in_channels, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        b, c, h, w = x.shape
        h = self.reduction(x) # (b, c/4, h, w)
        Y1, Yh = xf1.to(x.device)(h)
        h = Yh[0].flatten(1,2) # torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2))) # (10, 24*3, 8, 8)
        h = torch.cat((Y1,h), dim = 1) # (b, c, h, w)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.feedforward(h)
        return h + self.shortcut(x)

# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, shortnorm=False):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels, shortnorm)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

class AttentionBlock_Fixed(nn.Module):
    def __init__(self, channels, h, w, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.RMSNorm([channels, h, w])
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class WaveDownsample(nn.Module) :
    def __init__(self, mult = 2, channels=1, final_dim = 16, dropout = 0.5,) :
        super().__init__()
        self.feedforward = nn.Sequential(
                nn.Conv2d(final_dim, final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, final_dim, 1),
                nn.Conv2d(final_dim, channels, 3, 1, 1),
                nn.BatchNorm2d(channels)
            
            )

        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )
    def forward(self, x) :
        x = self.pre(x)
        b, c, h, w = x.shape
        x = self.reduction(x) # c/4, h, w
        Y1, Yh = xf1.to(x.device)(x)
        x = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        x = torch.cat((Y1,x), dim = 1) # c, h/2, w/2
        return self.feedforward(x)

class InverseWaveDownsample(nn.Module) :
    def __init__(self, mult = 2, channels=1, final_dim = 16, dropout = 0.5,) :
        super().__init__()
        self.feedforward = nn.Sequential(
                nn.Conv2d(channels, final_dim, 3, 1, 1),
                nn.Conv2d(final_dim, final_dim*mult, 1),
                nn.BatchNorm2d(final_dim*mult),
                nn.Conv2d(final_dim*mult, final_dim,1),
                nn.GELU(),
                nn.Dropout(dropout),
            
            )

        self.reduction = nn.Conv2d(final_dim, final_dim*4, 1)
        self.pre = nn.Sequential(
            nn.Conv2d(final_dim, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), channels, 3, 1, 1)
        )
    def forward(self, x) :
        x = self.feedforward(x)
        b, c, h, w = x.shape
        x = self.reduction(x) # 4*c, h, w
        temp = x.chunk(4, dim=1)
        coefs = torch.stack(temp).transpose(0,1)
        x = idwt2(coefs,"db1") # c,2*h,2*w
        return self.pre(x)
    
class WaveUpsample(nn.Module) :
    def __init__(self, mult = 2, channels=1, final_dim = 16, dropout = 0.5,) :
        super().__init__()
        self.feedforward = nn.Sequential(
                nn.Conv2d(final_dim, final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, final_dim, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(final_dim//4, channels,3, 1, 1),
                nn.BatchNorm2d(channels)
            
            )

        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
        self.pre = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners = False),
            nn.Conv2d(channels, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )

    def forward(self, x) :
        x = self.pre(x)
        b, c, h, w = x.shape
        x = self.reduction(x) # c/4, h, w
        Y1, Yh = xf1(x)
        x = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        x = torch.cat((Y1,x), dim = 1) # c, h/2, w/2
        return self.feedforward(x)
    
# The full UNet model with attention and timestep embedding
class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        waveResidual=False,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if not waveResidual:
                    layers = [
                        ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(ch, mult * model_channels, time_embed_dim, dropout=dropout)
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        
        # middle block
        if not waveResidual :
            self.middle_block = TimestepEmbedSequential(
                ResidualBlock(ch, ch, time_embed_dim, dropout),
                AttentionBlock(ch, num_heads=num_heads),
                ResidualBlock(ch, ch, time_embed_dim, dropout)
            )
        else :
            self.middle_block = TimestepEmbedSequential(
                ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout),
                AttentionBlock(ch, num_heads=num_heads),
                ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout)
            )
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                if not waveResidual :
                    layers = [
                        ResidualBlock(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout
                        )
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout=dropout
                        )
                    ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)
    

class UNetModel_Fixed(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        H=32, W=32,
        attention_resolutions=(8, 16),
        dropout=0,
        waveResidual=False,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if not waveResidual:
                    layers = [
                        ResidualBlock_Fixed(ch, mult * model_channels, time_embed_dim, dropout)
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(ch, mult * model_channels, time_embed_dim, dropout=dropout)
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock_Fixed(ch, h=H//ds, w=W//ds, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        
        # middle block
        if not waveResidual :
            self.middle_block = TimestepEmbedSequential(
                ResidualBlock_Fixed(ch, ch, time_embed_dim, dropout),
                AttentionBlock_Fixed(ch, h=H//ds, w=W//ds, num_heads=num_heads),
                ResidualBlock_Fixed(ch, ch, time_embed_dim, dropout)
            )
        else :
            self.middle_block = TimestepEmbedSequential(
                ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout),
                AttentionBlock(ch, num_heads=num_heads),
                ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout)
            )
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                if not waveResidual :
                    layers = [
                        ResidualBlock_Fixed(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout
                        )
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout=dropout
                        )
                    ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock_Fixed(ch, h=H//ds, w=W//ds, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)

# The full WaveUNet model with attention and timestep embedding
# down and up sampling blocks replaced
class WaveUNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        use_idwt=False,
        half = False,
        waveResidual = False,
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if not waveResidual:
                    layers = [
                        ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(ch, mult * model_channels, time_embed_dim, dropout=dropout)
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(WaveDownsample(channels=ch)))
                down_block_chans.append(ch)
                ds *= 2
        
        # middle block
        if not waveResidual :
            self.middle_block = TimestepEmbedSequential(
                ResidualBlock(ch, ch, time_embed_dim, dropout),
                AttentionBlock(ch, num_heads=num_heads),
                ResidualBlock(ch, ch, time_embed_dim, dropout)
            )
        else :
            self.middle_block = TimestepEmbedSequential(
                ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout),
                AttentionBlock(ch, num_heads=num_heads),
                ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout)
            )
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                if not waveResidual:
                    layers = [
                        ResidualBlock(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout
                        )
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout=dropout
                        )
                    ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    if not use_idwt : 
                        if not half : layers.append(WaveUpsample(channels=ch))
                        else : layers.append(Upsample(channels=ch, use_conv=conv_resample))
                    else : layers.append(InverseWaveDownsample(channels=ch))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)

# 3-path unet, ch : 16, 64, 256 (2-level dwt)
class UnetDWT2(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=16,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        shortnorm=True,
        conv_resample=True,
        att2wave=False,
        num_heads=4
    ):
        super().__init__()
        nlevels = 2
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.nlevels = nlevels
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # down blocks
        self.conv1 = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for lvl in range(nlevels+1) : 
            for _ in range(num_res_blocks) :
                ch = model_channels * (4**lvl)  
                layers = [ResidualBlock(ch, ch, time_embed_dim, dropout, shortnorm)]
                self.down_blocks.append(TimestepEmbedSequential(*layers))

        # middle block
        ch = model_channels * (4**nlevels)
        if att2wave :
            self.middle_block = TimestepEmbedSequential(
                ResidualBlock(ch, ch, time_embed_dim, dropout, shortnorm),
                Level1Waveblock(mult=1, final_dim=ch, ff_channel=16),
                ResidualBlock(ch, ch, time_embed_dim, dropout, shortnorm)
            )
        else :
            self.middle_block = TimestepEmbedSequential(
                ResidualBlock(ch, ch, time_embed_dim, dropout, shortnorm),
                AttentionBlock(ch, num_heads=num_heads, shortnorm=shortnorm),
                ResidualBlock(ch, ch, time_embed_dim, dropout, shortnorm)
            )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for lvl in range(nlevels, -1, -1) :
            for i in range(1+num_res_blocks) :
                ch = model_channels * (4**lvl)
                if i>0 or lvl==nlevels: layers = [ResidualBlock(ch*2, ch, time_embed_dim, dropout, shortnorm)]
                else : layers = [ResidualBlock(ch*5, ch, time_embed_dim, dropout, shortnorm)]
                self.up_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(
            norm_layer(ch, short=shortnorm),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward0(self, x, emb) :
        # top level
        h = x
        op = [h]
        for module in self.down_blocks[:2] :
            h = module(h, emb)
            op.append(h)
        return op
    def forward1(self, x, emb) :
        # mid level
        h = x
        op = [h]
        for module in self.down_blocks[2:4] :
            h = module(h, emb)
            op.append(h)
        return op
    def forward2(self, x, emb) :
        # bottom level
        h = x
        op = [h]
        for module in self.down_blocks[4:] :
            h = module(h, emb)
            op.append(h)
        return op
        
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.conv1(x)
        x1 = dwt2(x,"db1").flatten(1,2) # [N, C*4, H/2, W/2]
        x2 = dwt2(x1,"db1").flatten(1,2) # [N, C*16, H/4, W/4]
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        op0 = self.forward0(x, emb)
        op1 = self.forward1(x1, emb)
        op2 = self.forward2(x2, emb)
        # middle stage
        h = self.middle_block(op2[-1], emb)

        hs = [*op0, *op1, *op2] # stack for skip connections
        # up stage
        for module in self.up_blocks:
            if h.shape[-1] != hs[-1].shape[-1] :
                h = F.interpolate(h, scale_factor=2, mode="bicubic")
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)

# Attention (in middle_block) replaced with waveblock
class Att2Wave(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        ff_dim = 16,
        waveResidual=False,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        pixelShuffle=False,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if not waveResidual:
                    layers = [
                        ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(ch, mult * model_channels, time_embed_dim, dropout=dropout)
                    ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if not pixelShuffle : layers.append(Level1Waveblock(mult=1, final_dim=ch, ff_channel=ff_dim))
                    else : layers.append(WaveMixSRV2Block(mult=1, final_dim=ch, ff_channel=ff_dim))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        
        # middle block
        if not waveResidual :
            if not pixelShuffle :
                self.middle_block = TimestepEmbedSequential(
                    ResidualBlock(ch, ch, time_embed_dim, dropout),
                    Level1Waveblock(mult=1, final_dim=ch, ff_channel=ff_dim),
                    ResidualBlock(ch, ch, time_embed_dim, dropout)
                )
            else :
                self.middle_block = TimestepEmbedSequential(
                    ResidualBlock(ch, ch, time_embed_dim, dropout),
                    WaveMixSRV2Block(mult=1, final_dim=ch, ff_channel=ff_dim),
                    ResidualBlock(ch, ch, time_embed_dim, dropout)
                )
        else :
            if not pixelShuffle :
                self.middle_block = TimestepEmbedSequential(
                    ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout),
                    Level1Waveblock(mult=1, final_dim=ch, ff_channel=ff_dim),
                    ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout)
                )
            else :
                self.middle_block = TimestepEmbedSequential(
                    ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout),
                    WaveMixSRV2Block(mult=1, final_dim=ch, ff_channel=ff_dim),
                    ResidualWaveBlock(ch, ch, time_embed_dim, dropout=dropout)
                )
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                if not waveResidual :
                    layers = [
                        ResidualBlock(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout
                        )
                    ]
                else :
                    layers = [
                        ResidualWaveBlock(
                            ch + down_block_chans.pop(),
                            model_channels * mult,
                            time_embed_dim,
                            dropout=dropout
                        )
                    ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if not pixelShuffle :layers.append(Level1Waveblock(mult=1, final_dim=ch, ff_channel=ff_dim))
                    else : layers.append(WaveMixSRV2Block(mult=1, final_dim=ch, ff_channel=ff_dim))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear'
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        #self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
    
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    
    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image_size,
        batch_size=8,
        channels=3,
        ddim_timesteps=50,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. predict noise using model
            pred_noise = model(sample_img, t)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev
            
        return sample_img.cpu().numpy()
    
    # compute train losses
    def train_losses(self, model, x_start, t):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss
