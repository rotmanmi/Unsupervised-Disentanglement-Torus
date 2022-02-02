import torch
import sys

s = 'abcdefghijklmnopqrstuvwxyz'


def get_einsum_str(n_s):
    return '...' + ',...'.join(s[:n_s]) + '->...' + s[:n_s]


def calc_latend_dim(bond_dim, n_s=3, one=False):
    if one:
        return (n_s + 1) * bond_dim
    else:
        return (2 * n_s + 1) * bond_dim


def calc_latent_dim_sampling(bond_dim, n_s=3):
    return (n_s + 1) * bond_dim


def tensor_prod_sampling(mu, bond_dim, n_s=3, append_cosine=False):
    scales = mu[:, :bond_dim]
    mus = mu[:, bond_dim:]

    # mus = mus.view(mus.shape[0], bond_dim, 7, 2)
    mus = mus.view(mus.shape[0], bond_dim, n_s)
    mus = torch.stack([torch.cos(mus), torch.sin(mus)], -1)
    # z_mu = (scales.unsqueeze(-1) * torch.einsum('abc,abd,abe->abcde', mus.unbind(2)).view(
    #     mu.shape[0], bond_dim, -1)).mean(1)

    z_mu = (torch.einsum(get_einsum_str(n_s), mus.unbind(2)).view(mu.shape[0], bond_dim, -1)).mean(1)
    if append_cosine:
        z_mu = torch.cat([z_mu, mus[..., 0].squeeze(1)], 1)
    return z_mu


def tensor_prod(mu, bond_dim, n_s=3, append_cosine=False):
    scales = mu[:, :bond_dim]
    mus = mu[:, bond_dim:]

    # mus = mus.view(mus.shape[0], bond_dim, 7, 2)
    mus = mus.view(mus.shape[0], bond_dim, n_s, 2)
    mus = mus / mus.norm(p=2, dim=-1, keepdim=True)
    # z_mu = (scales.unsqueeze(-1) * torch.einsum('abc,abd,abe->abcde', mus.unbind(2)).view(
    #     mu.shape[0], bond_dim, -1)).mean(1)

    z_mu = (torch.einsum(get_einsum_str(n_s), mus.unbind(2)).view(mu.shape[0], bond_dim, -1)).mean(1)
    if append_cosine:
        z_mu = torch.cat([z_mu, mus[..., 0].squeeze(1)], 1)

    return z_mu


def ind_comps(mu, bond_dim, n_s=3):
    scales = mu[:, :bond_dim]
    mus = mu[:, bond_dim:]

    # mus = mus.view(mus.shape[0], bond_dim, 7, 2)
    mus = mus.view(mus.shape[0], bond_dim, n_s, 2)
    mus = mus / mus.norm(p=2, dim=-1, keepdim=True)
    ind_mus = torch.atan2(mus[..., 0], mus[..., 1])

    return torch.cat([scales, ind_mus.flatten(1)], 1)
