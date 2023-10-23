import torch
from gpnpytorchtools import lossfunctions


def test_vaeloss():
    vae_loss = lossfunctions.VAELoss()
    x = torch.randn(10, 10)
    x_hat = torch.randn(10, 10)
    mu = torch.randn(10, 5)
    logvar = torch.randn(10, 5)
    vae_loss(x, x_hat, mu, logvar)


def test_ssvaerloss():
    ss_vae_loss = lossfunctions.SSVAERLoss()
    x = torch.randn(10, 10)
    x_hat = torch.randn(10, 10)
    z_mu = torch.randn(10, 6)
    z_logvar = torch.randn(10, 6)
    z_gen = torch.randn(10, 6)
    y_mu = torch.randn(10, 5)
    y_logvar = torch.randn(10, 5)
    y = torch.randn(10, 5)
    x_gen_hat = torch.randn(10, 10)
    ss_vae_loss(x, x_hat, z_mu, z_logvar, z_gen, y_mu, y_logvar, y, x_gen_hat)
