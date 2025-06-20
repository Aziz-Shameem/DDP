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
def calculate_fid(model, images1, images2):
    # preprocessing
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    # resize images
    images1 = scale_images(images1, (150,150,3))
    images2 = scale_images(images2, (150,150,3))

    images1 = keras.applications.inception_v3.preprocess_input(images1)
    images2 = keras.applications.inception_v3.preprocess_input(images2)
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
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

import sys,os, time
sys.path.append(os.path.abspath(os.getcwd()))
import math
import argparse
import logging
from abc import abstractmethod
import glob 
from torchvision.transforms import v2

from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Encoder(nn.Module):
    """The encoder for VAE"""
    
    def __init__(self, image_size, input_dim, conv_dims, fc_dim, latent_dim, dropout=0.25):
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
        self.do = nn.Dropout(p=dropout)
        prev_dim = (image_size // (2 ** len(conv_dims))) ** 2 * conv_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, fc_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(fc_dim, latent_dim)
        self.fc_log_var = nn.Linear(fc_dim, latent_dim)
                    
    def forward(self, x):
        x = self.do(self.convs(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
class Decoder(nn.Module):
    """The decoder for VAE"""
    
    def __init__(self, latent_dim, image_size, conv_dims, output_dim):
        super().__init__()
        
        fc_dim = (image_size // (2 ** len(conv_dims))) ** 2 * conv_dims[-1]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU()
        )
        self.conv_size1 = self.conv_size2 = image_size // (2 ** len(conv_dims))
        
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
        x = x.reshape(x.size(0), -1, self.conv_size1, self.conv_size2)
        x = self.de_convs(x)
        x = self.pred_layer(x)
        return x

class VAE(nn.Module):
    """VAE"""
    
    def __init__(self, image_size, input_dim, conv_dims, fc_dim, latent_dim, dropout):
        super().__init__()
        
        self.encoder = Encoder(image_size, input_dim, conv_dims, fc_dim, latent_dim, dropout=dropout)
        self.decoder = Decoder(latent_dim, image_size, conv_dims, input_dim)
        
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
        
        # kl_loss
        kl_loss = (0.5*(log_var.exp() + mu ** 2 - 1 - log_var)).sum(1).mean()
        # recon loss
        x = x.clip(max=1., min=0.)
        recon_loss = F.binary_cross_entropy(recon, x, reduction="none").sum([1, 2, 3]).mean()
        
        return kl_loss + recon_loss

class CelebA_Dataset(torch.utils.data.Dataset) :
    def __init__(self, data_path, transforms=None, subset=0) :
        self.paths = glob.glob(data_path + '/*.jpg')
        self.transforms = transforms
        if subset>0 : self.paths = self.paths[:subset]
        # self.data = []
        # for path in paths :
        #     self.data.append(plt.imread(path))
    def __len__(self) :
        return len(self.paths)
    def __getitem__(self, idx) :
        data = plt.imread(self.paths[idx])
        if self.transforms :
            data = self.transforms(data)
        return data

def add_sp_noise(image, p1=0.03, p2=-1) :
    # add pepper with prob p1
    # add salt with prov p2
    if p2==-1 : p2=p1
    image[torch.rand_like(image)<p1] = 0.
    image[torch.rand_like(image)<p2] = 1.
    return image

def mask(image, width=4) :
    x = torch.randint(0, image.shape[-1]-width+1, size=(1,))
    y = torch.randint(0, image.shape[-1]-width+1, size=(1,))
    image[x:x+width, y:y+width] = 0.
    return image

def train_epoch(model, loader, p1, p2) :
    losslog = []
    for images in tqdm(loader):
        images = images.to(device)
        recon, mu, log_var = model(images)
        if torch.randn(1)<=p1 : recon, mu, log_var = model(add_sp_noise(images.clone())) # denoising
        elif torch.randn(1)<=p2 : recon, mu, log_var = model(mask(images.clone())) # masking
        else : recon, mu, log_var = model(images)
        loss = model.compute_loss(images, recon, mu, log_var)
        with torch.no_grad() : losslog.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losslog

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=64, help="Image Size, -1 for original size")
    parser.add_argument("--conv_dims", type=list, default=[64, 128, 256], help="conv dims for the encoder/decoder")
    parser.add_argument("--fc_dim", type=int, default=1024, help="dimension used in the fc layer")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent Dimension")
    parser.add_argument("--epochs", type=int, default=101, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument("--aug", action='store_true', default=False, help="Whether to add augmentations")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout Probability")
    parser.add_argument("--p1", type=float, default=0., help="Prob of adding salt and pepper noise")
    parser.add_argument("--p2", type=float, default=0., help="Prob of Masking")
    parser.add_argument("--device", type=int, default=-1, help="Device, -1 for cpu(default)")
    parser.add_argument("--savedir", type=str, help='Save directory for example generations/losslog/model weights')
    parser.add_argument("--use_fid", action='store_true', default=False, help='Whether to calculate and report FID')
    parser.add_argument("--silent", action='store_true', default=False, help='Whether to not show progress bar')
    parser.add_argument("--generate", action='store_true', default=False, help='Whether to generate samples')
    parser.add_argument("--crop", action='store_true', default=False, help='Testing args')
    parser.add_argument("--test", action='store_true', default=False, help='Testing args')
    args = parser.parse_args()

    if args.test : 
        print(args)
        raise
    os.chdir('Experiments')
    if os.path.exists(args.savedir) : 
        print('Save Directory already exists')
        print('All files will be overwritten')
        time.sleep(5)
    print('Making save directory')
    os.makedirs(args.savedir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.savedir}/experiments.log",
        filemode="a+",
        level=logging.INFO,
        # format="%(levelname)s (%(asctime)s): %(message)s",
        # datefmt="%d/%m/%Y %I:%M:%S %p"
    )

    logging.info(f"Training a VAE on Celeb-A Dataset")
    logging.info(f"Running with args: {args}")
    if args.aug :
        if args.crop : 
            transform=v2.Compose([
                transforms.ToTensor(),
                v2.RandomAffine(degrees=5, translate=(0.2,0.2)),
                v2.RandomZoomOut(side_range=(1.,2.)),
                transforms.CenterCrop((args.image_size, args.image_size))
            ])
        else :
            transform=v2.Compose([
                transforms.ToTensor(),
                v2.RandomAffine(degrees=5, translate=(0.2,0.2)),
                v2.RandomZoomOut(side_range=(1.,2.)),
                transforms.Resize((args.image_size, args.image_size))
            ])
    else :
        if args.crop :
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((args.image_size, args.image_size))
            ])
        else : 
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((args.image_size, args.image_size))
            ])

    # use CelebA dataset
    dataset = CelebA_Dataset('../data_celebA/img_align_celeba', transforms=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = f'cuda:{args.device}' if torch.cuda.is_available() and args.device!=-1 else 'cpu'
    logging.info(f"Using device = {device}")
    model = VAE(args.image_size, 3, args.conv_dims, args.fc_dim, args.latent_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f'{device=}')
    print('params', sum([p.numel() for p in model.parameters()]))
    losslog = []
    # training loop
    logging.info("Training\n"+"*"*120+"\n"+"*"*120)
    for epoch in tqdm(range(args.epochs), disable = args.silent): 
        losses = train_epoch(model, train_loader, args.p1, args.p2)
        losslog.extend(losses)
        if epoch%1==0 : 
            tqdm.write(f'Epoch {epoch} loss {sum(losses)/len(losses):.4f}')
            logging.info(f"Epoch {epoch} loss {sum(losses)/len(losses):.4f}")
        if epoch%20==0 : torch.save(model, args.savedir+f'/model{epoch}.pt')

    logging.info("End of Training\n"+"*"*120+"\n"+"*"*120)

    
    torch.save(model, args.savedir+'/modelFinal.pt')
    plt.plot(losslog)
    plt.savefig(args.savedir+'/losslog.png')
    logging.info('Losslog saved as losslog.png')

    ## Inference
    if args.generate :
        n_cols, n_rows = 8, 8
        sample_zs = torch.randn(n_cols * n_rows, args.latent_dim)
        model.eval()
        with torch.no_grad(): 
            generated_imgs = model.decoder(sample_zs.to(device))
            generated_imgs = generated_imgs.cpu().numpy()
        generated_imgs = np.array(generated_imgs * 255, dtype=np.uint8).reshape(n_rows, n_cols, 3, args.image_size, args.image_size)
        
        fig = plt.figure(figsize=(8, 8), constrained_layout=True)
        gs = fig.add_gridspec(n_rows, n_cols)
        for n_col in range(n_cols):
            for n_row in range(n_rows):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow(generated_imgs[n_row, n_col].swapaxes(0,1).swapaxes(1,2))
                f_ax.axis("off")
        plt.savefig(args.savedir+'/generations.png')
        logging.info('Sample Generations saved as generations.png')

    print('END')
    logging.info("*"*120+"\n"+"*"*120)

