import torch
from torch import nn
import os
import config
import utils


class WGAN_GP(nn.Module):
    def __init__(self, z_dim, critic, generator, critic_opt, generator_opt, batch_size=128, n_critic=5, fixed_noise=True):
        super(WGAN_GP, self).__init__()
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.fixed_noise = fixed_noise
        if self.fixed_noise:
            self.static_random_noise = torch.randn(10, z_dim)

        self.critic = critic
        self.generator = generator
        self.critic_opt = critic_opt
        self.generator_opt = generator_opt

    def train(self, train, saving_freq=100, load=True):
        if load:
            self.load()
        history = {"critic_loss": [],
                   "generator_loss": []}

        nb_of_batches = len(train)

        self.critic.train()

        for epoch in range(config.EPOCHS):
            self.generator.train()

            critic_losses = []
            generator_losses = []

            for cnt, (real_data, _) in enumerate(iter(train)):
                current_batch_size = real_data.shape[0]
                random_noise = torch.randn((current_batch_size, self.z_dim)).to(real_data.device)
                self.critic_opt.zero_grad()

                pred_real = self.critic(real_data)
                pred_real_loss = -torch.mean(pred_real)

                fake_data = self.generator(random_noise)
                pred_fake = self.critic(fake_data.detach())
                pred_fake_loss = torch.mean(pred_fake)

                gradient_penalty = config.LAMBDA * self.gradient_penalty(real_data, fake_data)
                critic_loss = pred_real_loss + pred_fake_loss + gradient_penalty

                critic_loss.backward()
                self.critic_opt.step()

                critic_losses.append(critic_loss.item())

                if ((cnt + 1) % self.n_critic) == 0:
                    self.generator_opt.zero_grad()
                    fake_img = self.generator(random_noise)
                    pred_fake = self.critic(fake_img)
                    g_loss = -torch.mean(pred_fake)

                    g_loss.backward()
                    self.generator_opt.step()

                    generator_losses.append(g_loss.item())

                if ((cnt + 1) % saving_freq) == 0:
                    self.visualize(epoch + 1, cnt)
                    self.save(epoch, cnt)
                    utils.plot_history(history)

                prompt = f"epochs:{epoch + 1}/{config.EPOCHS} batch:{cnt}/{nb_of_batches} "
                if cnt > 10:
                    if critic_losses:
                        prompt += f"critic_loss:{round(critic_losses[-1], 3)} "
                    if generator_losses:
                        prompt += f"generator_loss:{round(generator_losses[-1], 3)}"
                print(prompt)

            # Stocker les moyennes des pertes pour chaque époque
            if critic_losses:
                history["critic_loss"].append(sum(critic_losses) / len(critic_losses))
            if generator_losses:
                history["generator_loss"].append(sum(generator_losses) / len(generator_losses))

        return history

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.shape[0]
        epsilon = torch.rand(batch_size, 1, 1, 1).expand_as(real_data).to(real_data.device)

        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated = interpolated.requires_grad_(True)

        critic_interpolated = self.critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(gradients.shape[0], -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()

    def save(self, epoch, cnt):
        torch.save(self.generator.state_dict(), os.path.join(config.MODEL_DIR, f'G_epoch{epoch}_batch{cnt}.pkl'))
        torch.save(self.critic.state_dict(), os.path.join(config.MODEL_DIR, f'C_epoch{epoch}_batch{cnt}.pkl'))
        print("Saving of Critic and Generator")

    def forward(self, inputs):
        return self.generator(inputs)

    def visualize(self, epoch, batch):
        if not self.fixed_noise:
            random_noise = torch.randn(10, self.z_dim).to(self.static_random_noise.device)
            out = self(random_noise)
        else:
            out = self(self.static_random_noise)
        utils.visualize_images(out, epoch, batch)

    def load(self):
        self.generator.load_state_dict(torch.load("models/GeneratorFace_epoch2_batch1249.pkl"))
        self.critic.load_state_dict(torch.load("models/CriticFace_epoch2_batch1249.pkl"))