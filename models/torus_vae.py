import torch
import numpy as np
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from models.utils import calc_latend_dim, tensor_prod, ind_comps, calc_latent_dim_sampling, tensor_prod_sampling
from models.resblocks import ResEncoder, ResDecoder


class TorusVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(TorusVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.bond_dim = kwargs['bond_dim']
        self.one = kwargs['one']
        self.n_s = kwargs['n_s']
        self.ct2d = kwargs.get('ConvTranspose', False)

        self.append_cosine = kwargs.get('append_cosine', False)
        self.resnet = kwargs.get('resnet', False)

        modules = []
        self.orig_channels = in_channels
        #
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]
        self.orig_channels = in_channels
        if hidden_dims is None:
            hidden_dims = [64, 128, 128, 256, 256, 512, 512, 512, 512]

        last_dim = hidden_dims[0]

        # # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size=3, stride=2, padding=1),
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim
        if self.resnet:
            self.encoder = ResEncoder(self.orig_channels, dim=64)
            self.fc_mu = nn.Linear(hidden_dims[-1] * 16, calc_latend_dim(self.bond_dim, self.n_s, self.one))
            self.fc_var = nn.Linear(hidden_dims[-1] * 16, calc_latend_dim(self.bond_dim, self.n_s, self.one))
            self.decoder = ResDecoder(self.orig_channels, dim=64)
            if self.append_cosine:
                self.decoder_input = nn.Linear((2 ** self.n_s) + self.n_s, hidden_dims[-1] * 16)
            else:
                self.decoder_input = nn.Linear(2 ** self.n_s, hidden_dims[-1] * 16)
        else:
            modules.append(nn.Conv2d(in_channels, out_channels=hidden_dims[0], kernel_size=3, padding=1))
            in_channels = hidden_dims[0]
            for i, h_dim in enumerate(hidden_dims[1:]):
                modules.append(
                    nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=1, padding=1),
                        nn.ReLU())
                )
                if i % 2 == 0 and i != 0:
                    modules.append(nn.AvgPool2d(kernel_size=2))
                in_channels = h_dim
            self.encoder = nn.Sequential(*modules)
            self.fc_mu = nn.Linear(hidden_dims[-1] * 64, calc_latend_dim(self.bond_dim, self.n_s, self.one))
            self.fc_var = nn.Linear(hidden_dims[-1] * 64, calc_latend_dim(self.bond_dim, self.n_s, self.one))

            # Build Decoder
            modules = []
            if self.append_cosine:
                self.decoder_input = nn.Linear((2 ** self.n_s) + self.n_s, hidden_dims[-1] * 64)
            else:
                self.decoder_input = nn.Linear(2 ** self.n_s, hidden_dims[-1] * 64)

            hidden_dims.reverse()

            # for i in range(len(hidden_dims) - 1):
            #     modules.append(
            #         nn.Sequential(
            #             nn.ConvTranspose2d(hidden_dims[i],
            #                                hidden_dims[i + 1],
            #                                kernel_size=3,
            #                                stride=2,
            #                                padding=1,
            #                                output_padding=1),
            #             nn.BatchNorm2d(hidden_dims[i + 1]),
            #             nn.LeakyReLU())
            #     )

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.BatchNorm2d(hidden_dims[i]),
                        nn.ReLU(),
                        nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, padding=1)
                    )
                )
                if i % 2 == 1 and i < (len(hidden_dims) - 2):
                    if self.ct2d:
                        modules.append(nn.ConvTranspose2d(hidden_dims[i + 1],
                                                          hidden_dims[i + 1],
                                                          kernel_size=3,
                                                          stride=2,
                                                          padding=1,
                                                          output_padding=1))
                    else:
                        modules.append(nn.Upsample(scale_factor=2, mode='nearest'))

            self.decoder = nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #     nn.ConvTranspose2d(hidden_dims[-1],
        #                        hidden_dims[-1],
        #                        kernel_size=3,
        #                        stride=2,
        #                        padding=1,
        #                        output_padding=1),
        #     nn.BatchNorm2d(hidden_dims[-1]),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(hidden_dims[-1], out_channels=3,
        #               kernel_size=3, padding=1),
        #     nn.Tanh())

        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(last_dim),
            nn.ReLU(),
            nn.Conv2d(last_dim, out_channels=self.orig_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # z_mu = self.tensor_prod(mu)
        # z_log_var = self.tensor_prod(log_var)

        return [mu, log_var]

    def decode(self, z: Tensor, sampling=False) -> Tensor:
        if sampling:
            z_prod = tensor_prod_sampling(z, self.bond_dim, self.n_s, self.append_cosine)
        else:
            z_prod = tensor_prod(z, self.bond_dim, self.n_s, self.append_cosine)

        result = self.decoder_input(z_prod)
        # result = result.view(-1, 512, 2, 2)
        if self.resnet:
            result = result.view(-1, 512, 4, 4)
        else:
            result = result.view(-1, 512, 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        if self.one:
            z1 = self.reparameterize(mu, log_var)
            z2 = self.reparameterize(mu, log_var)
            z = torch.stack([z1, z2], -1).flatten(1)[:, self.bond_dim:]
        else:
            z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, ind_comps(z, self.bond_dim, self.n_s)]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        mu_scale = mu[:, :self.bond_dim]
        std_scale = log_var[:, :self.bond_dim]
        mu_else = mu[:, self.bond_dim:]
        std_else = log_var[:, self.bond_dim:]
        kld_loss_scale_terms = -0.5 * torch.sum(1 + std_scale - (mu_scale - 1) ** 2 - std_scale.exp(), dim=1).mean(0)
        kld_loss = -0.5 * torch.sum(1 + std_else - mu_else ** 2 - std_else.exp(), dim=1).mean(0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * (kld_loss + kld_loss_scale_terms)
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * ((kld_loss + kld_loss_scale_terms) - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = 2 * np.pi * torch.rand(num_samples, calc_latent_dim_sampling(self.bond_dim, self.n_s))
        z[:, :self.bond_dim].normal_(0, 1)
        # z[:, :self.bond_dim] = 0

        z = z.to(current_device)

        samples = self.decode(z, sampling=True)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
