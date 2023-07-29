from torch import nn

class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)
    

class Discriminator(nn.Module):

    def __init__(self, leak_slope = 0.2, p = 0.4):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leak_slope),
            nn.Dropout(p),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leak_slope),
            nn.Dropout(p),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(leak_slope),
            nn.Dropout(p),

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)