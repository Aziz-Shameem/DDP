import argparse
import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore") 

import tensorflow as tf
from tensorflow import keras

## for FID calc
import tensorflow as tf
from tensorflow import keras

# FID Calculation

inc_model = keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(150,150,3))
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
def calculate_fid(model, images1, images2, verbose=True):
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    images1 = scale_images(images1, (150,150,3))
    images2 = scale_images(images2, (150,150,3))
    images1 = keras.applications.inception_v3.preprocess_input(images1)
    images2 = keras.applications.inception_v3.preprocess_input(images2)
    act1 = model.predict(images1, verbose=verbose)
    act2 = model.predict(images2, verbose=verbose)
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False, dtype=np.csingle)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False, dtype=np.csingle)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = np.sqrt(sigma1*sigma2)
    covmean = np.real(covmean)
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return np.real(fid)

## Custom Datasets

class cross_dataset(torch.utils.data.Dataset) :
    def __init__(self, num_points=10000, transforms=None, subset=0) :
        num_points = 10000
        filter = lambda x : x # if x>0.55 or x<0.45 else 0.5 + (x>0)*0.05
        X1 = [0.5]*num_points
        Y1 = [filter(x) for x in torch.rand(num_points)]
        X2 = [filter(x) for x in torch.rand(num_points)]
        Y2 = [0.5]*num_points
        data1 = [(x,y) for x,y in zip(X1,Y1)]
        data2 = [(x,y) for x,y in zip(X2,Y2)]
        self.arr = data1 + data2
        self.transforms = transforms
        if subset>0 : self.arr = self.arr[:subset]
    def __len__(self) :
        return len(self.arr)
    def __getitem__(self, idx) :
        Data = self.arr[idx]
        if self.transforms :
            Data = self.transforms(Data)
        return torch.tensor(Data)

class circle_dataset(torch.utils.data.Dataset) :
    def __init__(self, num_points=10000, R1=3/16, R2=8/16, transforms=None, subset=0) :
        theta1 = 2*torch.pi*torch.randn((num_points,))
        theta2 = 2*torch.pi*torch.randn((num_points,))
        X1 =  R1 * torch.cos(theta1) + 0.5
        X2 =  R2 * torch.cos(theta2) + 0.5
        Y1 =  R1 * torch.sin(theta1) + 0.5
        Y2 =  R2 * torch.sin(theta2) + 0.5
        data1 = [(x,y) for x,y in zip(X1,Y1)]
        data2 = [(x,y) for x,y in zip(X2,Y2)]
        self.arr = data1 + data2
        self.transforms = transforms
        if subset>0 : self.arr = self.arr[:subset]
    def __len__(self) :
        return len(self.arr)
    def __getitem__(self, idx) :
        Data = self.arr[idx]
        if self.transforms :
            Data = self.transforms(Data)
        return torch.tensor(Data).unsqueeze(0)

## VAE classes

class Encoder_mnist(nn.Module):
    """The encoder for VAE"""
    
    def __init__(self, image_size, input_dim, conv_dims, fc_dim, latent_dim):
        super().__init__()
        
        convs = []
        prev_dim = input_dim
        for conv_dim in conv_dims:
            convs.append(nn.Sequential(
                nn.Conv2d(prev_dim, conv_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ))
            prev_dim = conv_dim
        self.convs = nn.Sequential(*convs)
        
        prev_dim = (image_size // (2 ** len(conv_dims))) ** 2 * conv_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, fc_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(fc_dim, latent_dim)
        self.fc_log_var = nn.Linear(fc_dim, latent_dim)
                    
    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
class Decoder_mnist(nn.Module):
    """The decoder for VAE"""
    
    def __init__(self, latent_dim, image_size, conv_dims, output_dim):
        super().__init__()
        
        fc_dim = (image_size // (2 ** len(conv_dims))) ** 2 * conv_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU()
        )
        self.conv_size = image_size // (2 ** len(conv_dims))
        
        de_convs = []
        prev_dim = conv_dims[-1]
        for conv_dim in conv_dims[::-1]:
            de_convs.append(nn.Sequential(
                nn.ConvTranspose2d(prev_dim, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ))
            prev_dim = conv_dim
        self.de_convs = nn.Sequential(*de_convs)
        self.pred_layer = nn.Sequential(
            nn.Conv2d(prev_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.size(0), -1, self.conv_size, self.conv_size)
        x = self.de_convs(x)
        x = self.pred_layer(x)
        return x

class Encoder_2d(nn.Module):
    """The encoder for VAE"""
    
    def __init__(self, input_dim, fc_dim, latent_dim):
        super().__init__()
        
        self.fwd = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(fc_dim, latent_dim)
        self.fc_log_var = nn.Linear(fc_dim, latent_dim)
                    
    def forward(self, x):
        x = self.fwd(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
class Decoder_2d(nn.Module):
    """The decoder for VAE"""
    
    def __init__(self, latent_dim, fc_dim, output_dim):
        super().__init__()

        self.fwd = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fwd(x)

class VAE(nn.Module):
    """VAE"""

    def __init__(self, image_size, input_dim, conv_dims, fc_dim, latent_dim, mnist=True):
        super().__init__()
        if mnist :
            self.encoder = Encoder_mnist(image_size, input_dim, conv_dims, fc_dim, latent_dim)
            self.decoder = Decoder_mnist(latent_dim, image_size, conv_dims, input_dim)
        else :
            self.encoder = Encoder_2d(input_dim, fc_dim, latent_dim)
            self.decoder = Decoder_2d(latent_dim, fc_dim, input_dim)
        
    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var
    
    def compute_loss(self, x, recon, mu, log_var):
        """compute loss of VAE"""
        
        # KL loss
        kl_loss = (0.5*(log_var.exp() + mu ** 2 - 1 - log_var)).sum(1).mean()
        
        # recon loss
        recon_loss = F.binary_cross_entropy(recon, x, reduction="none").sum([1, 2, 3]).mean()
        
        return kl_loss + recon_loss

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mnist', help="One of ['mnist','cifar10','cross','circle']")
    parser.add_argument("--conv_dims", type=list, default=[5,6,7,8,9], help="conv dims for the encoder/decoder")
    parser.add_argument("--fc_dim", type=int, default=1024, help="dimension used in the fc layer")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent Dimension")
    parser.add_argument("--num_points", type=int, default=10000, help="Number of datapoint for cross/circle datasets")
    parser.add_argument("--R1", type=float, default=3/16, help="Inner radius in circle dataset")
    parser.add_argument("--R2", type=float, default=8/16, help="Outer radius in circle dataset")
    parser.add_argument("--batch_size", type=int, default=2056, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--device", type=int, default=-1, help="Device, -1 for cpu(default)")
    parser.add_argument("--use_pretrained", action='store_true', default=False, help="true if just inference is needed")
    parser.add_argument("--path", type=str, default="models/VAE_base_100d.pt", help='Path to the pretrained model')
    parser.add_argument("--savedir", type=str, default="VAE_generations", help='Save directory for example generations')
    parser.add_argument("--use_fid", action='store_true', default=False, help='Whether to calculate and report FID')
    args = parser.parse_args()
    
    transform=transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset == 'mnist' :
        image_size = 28
        dataset = datasets.MNIST('data', train=True, download=True,
                        transform=transform)
    elif args.dataset == 'cifar10' :
        image_size = 28
        dataset = datasets.CIFAR10('data', train=True, download=True,
                       transform=transform)
    elif args.dataset == 'cross' :
        image_size = 28
        dataset = cross_dataset(args.num_points)
    elif args.dataset == 'circle' :
        image_size = 28
        dataset = circle_dataset(args.num_points)
    else : 
        raise ValueError("Dataset must be one of ['mnist','cifar10','cross','circle']")

    ### HPARAMS ###
    conv_dims= args.conv_dims
    fc_dim = args.fc_dim
    latent_dim = args.latent_dim 
    batch_size = args.batch_size 
    epochs = args.epochs
    device = 'cpu' if args.device==-1 else f'cuda:{args.device}'
    print(f'{device=}')
    ################

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAE(image_size, 1, conv_dims, fc_dim, latent_dim).to(device)
    if args.use_pretrained : model = torch.load(args.path).to(device); print('Using pretrained...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print('Number of params', sum([p.numel() for p in model.parameters()]))

    ## Training
    if not args.use_pretrained :
        print('Training...')
        losslog = []
        for epoch in tqdm(range(epochs)):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                recon, mu, log_var = model(images)
                loss = model.compute_loss(images, recon, mu, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad() : losslog.append(loss.item())
            if epoch%5==0 : 
                tqdm.write(f'Epoch {epoch}, loss {loss.item()}')

    ## Inference
    n_cols, n_rows = 8, 8
    sample_zs = torch.randn(n_cols * n_rows, latent_dim)
    model.eval()
    with torch.no_grad(): 
        generated_imgs = model.decoder(sample_zs.cuda())
        generated_imgs = generated_imgs.cpu().numpy()
    generated_imgs = np.array(generated_imgs * 255, dtype=np.uint8).reshape(n_rows, n_cols, image_size, image_size)
    
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols)
    for n_col in range(n_cols):
        for n_row in range(n_rows):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(generated_imgs[n_row, n_col], cmap="gray")
            f_ax.axis("off")
    plt.savefig(args.savedir+'generations.png')

    if args.use_fid :
        test_dataset = datasets.MNIST(root='data_test', train=False, download=True, transform=transform)
        sample_zs = torch.randn(1000, latent_dim)
        with torch.no_grad():
            generated_imgs = model.decoder(sample_zs.cuda())
            generated_imgs = generated_imgs.cpu().numpy()
        images_from_vae = np.array(generated_imgs * 255, dtype=np.uint8).reshape(1000, 1, image_size, image_size)
        images_actual = test_dataset.data[:1000].unsqueeze(1).numpy()
        fid = calculate_fid(inc_model, images_from_vae, images_actual)
        print(f'FID ({images_actual.shape[0]} images from mnist test) : {fid:.4f}')
    