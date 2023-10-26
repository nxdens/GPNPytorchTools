import torch
from gpnpytorchtools import models


def test_vaer():
    x = torch.randn(10, 10)
    vaer = models.VAER(input_size=10)
    x_hat, z_mu, z_logvar, z, r_mu, r_logvar = vaer(x)
    torch.nn.MSELoss()(x_hat, x).backward()
    assert x_hat.shape == x.shape
    assert z_mu.shape == (10, 8)
    assert z_logvar.shape == (10, 8)
    assert z.shape == (10, 8)
    assert r_mu.shape == (10, 1)
    assert r_logvar.shape == (10, 1)


def test_vae():
    x = torch.randn(10, 10)
    latent_size = 5
    vae = models.VAE(
        input_size=10,
        layers=2,
        latent_dim=latent_size,
        activation=torch.nn.GELU(),
    )
    x_hat, mu, logvar, z_resample = vae(x)
    torch.nn.MSELoss()(x_hat, x).backward()
    assert x_hat.shape == x.shape
    assert mu.shape == (10, latent_size)
    assert logvar.shape == (10, latent_size)
    assert z_resample.shape == (10, latent_size)


def test_ssvaer():
    x = torch.randn(10, 10)
    latent_size = 5
    ssvaer = models.SSVAER(
        input_size=10,
        shared_layers=2,
        resample_layers=2,
        resample_size=6,
        regressor_layers=2,
        latent_size=latent_size,
        activation=torch.nn.GELU(),
    )
    (
        x_hat,
        z_mu,
        z_logvar,
        z_resample,
        y_mu,
        y_logvar,
        y_resample,
        dy_mu,
        z_gen,
        z_gen_resample,
        x_gen_hat,
    ) = ssvaer(x)

    assert x_hat.shape == x.shape
    assert z_mu.shape == (10, latent_size)
    assert z_logvar.shape == (10, latent_size)
    assert z_resample.shape == (10, latent_size)
    assert dy_mu.shape == (10, 1)
    assert z_gen.shape == (10, latent_size)
    assert z_gen_resample.shape == (10, latent_size)
    assert y_mu.shape == (10, 1)
    assert y_logvar.shape == (10, 1)
    assert y_resample.shape == (10, 1)
    assert x_gen_hat.shape == x.shape
