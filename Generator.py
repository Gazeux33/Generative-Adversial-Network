from torch import nn
import config


class GeneratorCelebA(nn.Module):
    def __init__(self, z_dim):
        super(GeneratorCelebA, self).__init__()
        self.z_dim = z_dim

        self.fwd = nn.Sequential(
            nn.Linear(self.z_dim, 512 * 4 * 4),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, config.NB_CHANNEL, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        x = self.fwd(inputs)
        x = x.view(-1, 512, 4, 4)
        x = self.conv(x)
        return x
