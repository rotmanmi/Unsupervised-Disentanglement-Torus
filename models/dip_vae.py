import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from models.resblocks import ResEncoder, ResDecoder

"""
From Disentagnlement lib pytorch https://github.com/amir-abdi/disentanglement-pytorch
"""


def dipvaei_loss_fn(w_dipvae, lambda_od, lambda_d, **kwargs):
    """
    Variational Inference of Disentangled Latent Concepts from Unlabeled Observations
    by Abhishek Kumar, Prasanna Sattigeri, Avinash Balakrishnan
    https://openreview.net/forum?id=H1kG7GZAW
    :param w_dipvae:
    :param lambda_od:
    :param lambda_d:
    :param kwargs:
    :return:
    """
    mu = kwargs['mu']

    cov_z_mean = covariance_z_mean(mu)
    cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z_mean, lambda_od, lambda_d)
    return cov_dip_regularizer * w_dipvae


def dipvaeii_loss_fn(w_dipvae, lambda_od, lambda_d, **kwargs):
    """
    Variational Inference of Disentangled Latent Concepts from Unlabeled Observations
    by Abhishek Kumar, Prasanna Sattigeri, Avinash Balakrishnan
    https://openreview.net/forum?id=H1kG7GZAW
    :param w_dipvae:
    :param lambda_od:
    :param lambda_d:
    :param kwargs:
    :return:
    """
    mu = kwargs['mu']
    logvar = kwargs['logvar']

    cov_z_mean = covariance_z_mean(mu)
    cov_enc = torch.diag(torch.exp(logvar))
    expectation_cov_enc = torch.mean(cov_enc, dim=0)
    cov_z = expectation_cov_enc + cov_z_mean
    cov_dip_regularizer = regularize_diag_off_diag_dip(cov_z, lambda_od, lambda_d)

    return cov_dip_regularizer * w_dipvae


def regularize_diag_off_diag_dip(covariance_matrix, lambda_od, lambda_d):
    """Compute on and off diagonal regularizers for DIP-VAE models.
    Penalize deviations of covariance_matrix from the identity matrix. Uses
    different weights for the deviations of the diagonal and off diagonal entries.
    Borrowed from https://github.com/google-research/disentanglement_lib/

    Args:
      covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
      lambda_od: Weight of penalty for off diagonal elements.
      lambda_d: Weight of penalty for diagonal elements.
    Returns:
      dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
    """
    covariance_matrix_diagonal = diag_part(covariance_matrix)
    # we can set diagonal to zero too... problem with the gradient?
    covariance_matrix_off_diagonal = covariance_matrix - torch.diag(covariance_matrix_diagonal)

    dip_regularizer = lambda_od * torch.sum(covariance_matrix_off_diagonal ** 2) + \
                      lambda_d * torch.sum((covariance_matrix_diagonal - 1) ** 2)

    return dip_regularizer


def diag_part(tensor):
    """
    return the diagonal elements of batched 2D square matrices
    :param tensor: batched 2D square matrix
    :return: diagonal elements of the matrix
    """
    assert len(tensor.shape) == 2, 'This is implemented for 2D matrices. Input shape is {}'.format(tensor.shape)
    assert tensor.shape[0] == tensor.shape[1], 'This only handles square matrices'
    return tensor[range(len(tensor)), range(len(tensor))]


def covariance_z_mean(z_mean):
    """Computes the covariance of z_mean.
    Borrowed from https://github.com/google-research/disentanglement_lib/
    Uses cov(z_mean) = E[z_mean*z_mean^T] - E[z_mean]E[z_mean]^T.
    Args:
      z_mean: Encoder mean, tensor of size [batch_size, num_latent].
    Returns:
      cov_z_mean: Covariance of encoder mean, tensor of size [num_latent, num_latent].
    """
    expectation_z_mean_z_mean_t = torch.mean(z_mean.unsqueeze(2) * z_mean.unsqueeze(1), dim=0)
    expectation_z_mean = torch.mean(z_mean, dim=0)
    cov_z_mean = expectation_z_mean_z_mean_t - \
                 (expectation_z_mean.unsqueeze(1) * expectation_z_mean.unsqueeze(0))
    return cov_z_mean


class DIPVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 lambda_diag: float = 10.,
                 lambda_offdiag: float = 5.,
                 **kwargs) -> None:
        super(DIPVAE, self).__init__()

        self.latent_dim = latent_dim
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag
        self.resnet = kwargs.get('resnet', False)

        modules = []


        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]
        self.orig_channels = in_channels
        if hidden_dims is None:
            hidden_dims = [64, 128, 128, 256, 256, 512, 512, 512, 512]
        last_dim = hidden_dims[0]
        # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size= 3, stride= 2, padding  = 1),
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim
        if self.resnet:
            self.encoder = ResEncoder(self.orig_channels, dim=64)
            self.fc_mu = nn.Linear(hidden_dims[-1] * 16, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1] * 16, latent_dim)
            self.decoder = ResDecoder(self.orig_channels, dim=64)
            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)
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
            self.fc_mu = nn.Linear(hidden_dims[-1] * 64, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1] * 64, latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 64)

            hidden_dims.reverse()

            # for i in range(len(hidden_dims) - 1):
            #     modules.append(
            #         nn.Sequential(
            #             nn.ConvTranspose2d(hidden_dims[i],
            #                                hidden_dims[i + 1],
            #                                kernel_size=3,
            #                                stride = 2,
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

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        if self.resnet:
            result = result.view(-1, 512, 4, 4)
        else:
            result = result.view(-1, 512, 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        dip_loss = dipvaeii_loss_fn(1, self.lambda_offdiag, self.lambda_diag, mu=mu, logvar=log_var)
        loss = recons_loss + kld_weight * kld_loss + dip_loss
        # # DIP Loss
        # centered_mu = mu - mu.mean(dim=1, keepdim=True)  # [B x D]
        # cov_mu = centered_mu.t().matmul(centered_mu).squeeze()  # [D X D]
        #
        # # Add Variance for DIP Loss II
        # cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1=0), dim=0)  # [D x D]
        # # For DIp Loss I
        # # cov_z = cov_mu
        #
        # cov_diag = torch.diag(cov_z)  # [D]
        # cov_offdiag = cov_z - torch.diag(cov_diag)  # [D x D]
        # dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
        #            self.lambda_diag * torch.sum((cov_diag - 1) ** 2)
        #
        # loss = recons_loss + kld_weight * kld_loss + dip_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'KLD': -kld_loss,
                'DIP_Loss': dip_loss}

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
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
