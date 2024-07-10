import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import transforms

from WGAN_GP import WGAN_GP
from Generator import GeneratorCelebA
from Critic import CriticCelebA
import config

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
device = torch.device(device_name)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

data = torchvision.datasets.CelebA("data", transform=transform)

dataloader = DataLoader(data, batch_size=128, shuffle=True)


generator = GeneratorCelebA(config.Z_DIM).to(device)
critic = CriticCelebA().to(device)

critic_opt = torch.optim.Adam(critic.parameters(), lr=config.LEARNING_RATE)
generator_opt = torch.optim.Adam(generator.parameters(), lr=config.LEARNING_RATE)

wgan = WGAN_GP(
    z_dim=config.Z_DIM,
    critic=critic,
    generator=generator,
    critic_opt=critic_opt,
    generator_opt=generator_opt,
    batch_size=config.BATCH_SIZE,
    device=device,
    n_critic=config.N_CRITIC)

if __name__ == "__main__":
    wgan.train(dataloader, saving_freq=100,load=False)
