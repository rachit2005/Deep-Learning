import torch
import torch.utils
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder , FakeData , MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


BATCH_SIZE = 32

minst = MNIST(root='data_mnist' , train=True , download=True , transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])) # dataset is formed
data_loader = DataLoader(dataset=minst, shuffle=True , batch_size = BATCH_SIZE)

# for image_batch , label_batch in data_loader:
#         print(image_batch.shape)
#         plt.imshow(image_batch[0][0], cmap="gray")
#         plt.show()
#         break

'''Making a Discriminator --> it will discriminate/determine whether the image is real or generated'''
# output --> 1 or 0 
from torch import nn
image_size = 784
hidden_size = 256

discriminant = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

'''Making a G to generate images '''
# output --> 28*28

LATENT_SIZE = 64
generator = nn.Sequential(
    nn.Linear(LATENT_SIZE, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())


import torch.nn.functional as F

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminant.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

def train_discriminator(images):
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(BATCH_SIZE, 1) # 32 rows of 1
    fake_labels = torch.zeros(BATCH_SIZE, 1) # 32 rows of 0 
        
    # Loss for real images
    outputs = discriminant(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Loss for fake images
    z = torch.randn(BATCH_SIZE, LATENT_SIZE) # generating a random noise of rows = 32 , columns = 64

    fake_images = generator(z)
    outputs = discriminant(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Combine losses
    d_loss = d_loss_real + d_loss_fake
    # Reset gradients
    reset_grad()
    # Compute gradients
    d_loss.backward()
    # Adjust the parameters using backprop
    d_optimizer.step()
    
    return d_loss, real_score, fake_score


def train_generator():
    # Generate fake images and calculate loss
    z = torch.randn(BATCH_SIZE, LATENT_SIZE) # generating a random noise

    fake_images = generator(z) # output = (28*28)
    labels = torch.ones(BATCH_SIZE, 1)
    g_loss = criterion(discriminant(fake_images), labels)

    # Backprop and optimize
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


from torchvision.utils import save_image 
from IPython.display import Image
from torchvision.utils import save_image
import os

sample_dir = 'generated_digits'
os.makedirs(sample_dir, exist_ok=True)

sample_vectors = torch.randn(BATCH_SIZE, LATENT_SIZE)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def save_fake_images(index):
    fake_images = generator(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)
    
# Before training
save_fake_images(0)
Image(os.path.join(sample_dir, 'fake_images-0000.png'))

num_epochs = 25
total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Load a batch & transform to vectors
        images = images.reshape(BATCH_SIZE, -1)
        
        # Train the discriminator and generator
        d_loss, real_score, fake_score = train_discriminator(images)
        g_loss, fake_images = train_generator()
        
        # Inspect the losses
        if (i+1) % 200 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
        
    # Sample and save images
    save_fake_images(epoch+1)

