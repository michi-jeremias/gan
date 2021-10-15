import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, img_channels):
        super().__init__()
        self.img_channels = img_channels
        self.net = nn.Sequential(
            # 64 to 32
            nn.Conv2d(in_channels=self.img_channels, out_channels=128,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # 32 to 16
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=256),
            # 16 to 8
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features=512),
            # 8 to 4
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(1024),
            # 4 to 1
            nn.Conv2d(in_channels=1024, out_channels=1,
                      kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
        # self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels):
        super().__init__()
        self.img_channels = img_channels
        self.z_dim = z_dim
        # n_out = (n_in - 1)*stride-2*pad_in + kernel_size + pad_out
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.z_dim, out_channels=1024,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.ConvTranspose2d(in_channels=128, out_channels=self.img_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        # self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)
