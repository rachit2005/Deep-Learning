# -*- coding: utf-8 -*-
"""variantional autoencoder.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rrBhZpe4_qb8yFuq-ch53IH_4syuv8Bi

variational autoencoder let us design complex generative models of data, and fit them to large datasets
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets

"""The latent space of GAN and VAE models is the hidden layer that contains the latent variables that are used to generate the outputs.

In a variational antoencoder (VAE), the output of the encoder, μ(mu) and σ(sigma), is a deterministic function of input data x
"""

class VariationalAutoencoder(nn.Module):
  def __init__(self , input_dim , z_dim = 20):
    super().__init__()
    #encoder
    self.img_2hid = nn.Linear(input_dim , 128)
    self.hid_2mu = nn.Linear(128 , z_dim)
    self.hid_2sigma = nn.Linear(128 , z_dim)
    #decoder
    self.z_2hid = nn.Linear(z_dim , 128)
    self.hid_2img = nn.Linear(128 , input_dim)

  def encoder(self , x):
    h = nn.ReLU()(self.img_2hid(x))

    mu = self.hid_2mu(h)
    sigma = self.hid_2sigma(h)
    return mu , sigma

  def decoder(self , z):
    h = nn.ReLU()(self.z_2hid(z))

    img = torch.sigmoid(self.hid_2img(h))
    return img


  def forward(self , x):
    mu , sigma = self.encoder(x)
    epsilon = torch.randn_like(sigma)
    z_reparametrized = mu + sigma * epsilon
    img_reconstructed = self.decoder(z_reparametrized)
    return img_reconstructed , mu , sigma

x = torch.randn(4 , 784) # an example image of (4, 1, 28,28)
vae = VariationalAutoencoder(input_dim = 784)
img_reconst , mu , sigma = vae(x)
print(img_reconst.shape)
print(mu.shape)
print(sigma.shape)

#configuration
IMPUT_DIM = 784
Z_DIM = 20
BATCH_SIZE = 32
EPOCHS = 10
lr = 3e-4

#dataset, dataloader
dataset = datasets.MNIST(root = 'data' , train = True , transform = transforms.ToTensor() , download = True)
dataloader = DataLoader(dataset = dataset , batch_size = BATCH_SIZE, shuffle = True)
model = VariationalAutoencoder(input_dim = IMPUT_DIM , z_dim = Z_DIM)
optimizer = torch.optim.Adam(params = model.parameters() , lr = lr)
loss_fun = nn.BCELoss()

for epoch in range(EPOCHS):
  for batch , (img , label) in enumerate(dataloader):
    img = img.view(-1 , IMPUT_DIM)
    img_reconst , mu , sigma = model(img)

    # compute loss
    recons_loss = loss_fun(img_reconst , img)
    kl_loss = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    loss = recons_loss + kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"epoch {epoch+1}.................")

def inference(digit , num_example = 1):
  images = []
  idx = 0

  # this will add all images from the dataset to this images list
  for x , y in dataset:
    if y == idx:
      images.append(x)
      idx+=1
    if idx == 10:
      break

  encoding_digit = []
  for d in range(10):
    with torch.inference_mode():
      img_to_sent = images[d].view(-1 , IMPUT_DIM) # reshaping the image
      mu , sigma = model.encoder(img_to_sent)
      encoding_digit.append((mu , sigma))

  # getting the mu , sigma of that particualr digit
  mu , sigma = encoding_digit[digit]
  for eg in range(num_example):
    epsilon = torch.randn_like(sigma)
    z = mu+sigma*epsilon
    out = model.decoder(z)
    out = out.view(-1,1,28,28)
    torchvision.utils.save_image(out , f"generarted_{digit}.png")

for idx in range(10):
  inference(idx)