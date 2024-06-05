from torch import nn

import config


class CriticCelebA(nn.Module):
    def __init__(self):
        super(CriticCelebA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(config.NB_CHANNEL, 64, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(),

        )
        self.fwd = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512 * 4 * 4, 1)
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.fwd(x)
        return x
