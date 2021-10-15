"""09-05-2021: DCGAN implementation,
see https://arxiv.org/abs/1511.06434 for the paper."""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from deeplearning.model.init import init_normal
from deeplearning.trainer.trainer import GANTrainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import del_logs, timefunc

from dcgan1.model import Discriminator, Generator

# Hyperparameters
DATAPATH = 'files/'
TBLOGPATH = 'logs/dcgan1/'
batch_size = 128
num_epochs = 5
learning_rate = 2e-4
beta1 = 0.5
beta2 = 0.999
z_dim = 100
img_channels = 1
num_features = 8
IMG_SIZE = 64  # MNIST


D = Discriminator(img_channels=1)  # .to(device=device)
G = Generator(z_dim=z_dim, img_channels=1)  # .to(device=device)
D.apply(init_normal)
G.apply(init_normal)


# Test
def test():
    N, img_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    z = torch.rand(N, img_channels, H, W)
    D = Discriminator(img_channels)
    assert (D(z)).shape == (N, 1, 1, 1)
    print('nice')


# Data
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(img_channels)],
                         [0.5 for _ in range(img_channels)])
])
dataset = datasets.MNIST(root=DATAPATH, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Loss
criterion = nn.BCELoss()

# Optimizer
optim_D = optim.Adam(params=D.parameters(),
                     lr=learning_rate, betas=(beta1, beta2))
optim_G = optim.Adam(params=G.parameters(),
                     lr=learning_rate, betas=(beta1, beta2))


# Training
# @timefunc
def train():
    print('Start training.')
    fake_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/fake')
    real_writer = SummaryWriter(log_dir=f'{TBLOGPATH}/real')
    D.train()
    G.train()
    tb_step = 0

    for epoch in range(num_epochs):

        for batch_idx, (real_data, _) in enumerate(loader):
            current_batch_size = real_data.shape[0]
            x = real_data.to(device=device)
            # Random uniform
            z = torch.rand(current_batch_size, z_dim, 1, 1).to(device)
            fake_data = G(z)

            # Discriminator forward
            real_scores = D(x).reshape(-1)
            fake_scores = D(G(z)).reshape(-1)
            real_loss_D = criterion(real_scores, torch.ones_like(real_scores))
            fake_loss_D = criterion(fake_scores, torch.zeros_like(fake_scores))
            loss_D = (real_loss_D + fake_loss_D) / 2

            # Discriminator backward
            optim_D.zero_grad()
            loss_D.backward(retain_graph=True)  # retain_graph=True
            optim_D.step()

            # Generator forward
            fake_scores_updated = D(G(z))
            loss_G = criterion(fake_scores_updated,
                               torch.ones_like(fake_scores_updated))

            # Generator backward
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if batch_idx == 0:
                with torch.no_grad():
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}]'
                        f'\tLoss D: {loss_D:.6f}\tLoss G: {loss_G:.6f}')
                    fake_data = G(z).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
                    fake_grid = torchvision.utils.make_grid(
                        fake_data[:32], normalize=True)
                    fake_writer.add_image(
                        'Fake Img', fake_grid, global_step=tb_step)
                    real_data = real_data.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
                    real_grid = torchvision.utils.make_grid(
                        real_data[:32], normalize=True)
                    real_writer.add_image(
                        'Real Img', real_grid, global_step=tb_step)
                    tb_step += 1
    print('Finished training.')


train()


GT = GANTrainer(
    model_g=G,
    model_d=D,
    optim_g=optim_G,
    optim_d=optim_D,
    loader=loader,
    loss_fn=nn.BCELoss(),
    init_fn=init_normal,
    z_dim=z_dim,
    img_size=IMG_SIZE
)
