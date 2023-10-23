import torch
from torch import nn


class VAELoss(nn.Module):
    """Loss function for vae."""

    def __init__(self, reconstruction_loss=nn.MSELoss(reduction="sum")):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss

    def forward(self, x, x_hat, mu, logvar):
        """Loss function for VAE.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): reconstructed input data
            mu (torch.Tensor): predicted mean
            logvar (torch.Tensor): predicted log variance

        Returns:
            torch.Tensor: loss value
        """
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (
            reconstruction_loss + kl_divergence,
            reconstruction_loss,
            kl_divergence,
        )


class SSVAERLoss(nn.Module):
    """Loss function for SSVAER."""

    def __init__(self):
        super().__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="mean")

    #
    def forward(
        self, x, x_hat, z_mu, z_logvar, z_gen, y_mu, y_logvar, y, x_gen_hat
    ):
        """The `forward` method in the `SSVAERLoss` class calculates the loss for the SSVAER (Semi-
        Supervised Variational Autoencoder with Regression) model.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): reconstructed input data
            z_mu (torch.Tensor): latent mean
            z_logvar (torch.Tensor): latent log variance
            z_gen (torch.Tensor): generated latent vector
            y_mu (torch.Tensor): regression mean
            y_logvar (torch.Tensor): regression log variance
            y (torch.Tensor): regression target
            x_gen_hat (torch.Tensor): reconstructed from generated latent vector

        Returns:
            torch.Tensor: loss value for backprop
            torch.Tensor: reconstruction loss
            torch.Tensor: KL divergence loss
            torch.Tensor: label loss
        """
        reconstruction_loss = self.reconstruction_loss(x_hat, x)
        reconstruction_loss += self.reconstruction_loss(x_gen_hat, x)
        kld_loss = torch.mean(
            -0.5
            * (1 + z_logvar - (z_mu - z_gen).pow(2) - z_logvar.exp()).sum(
                dim=1
            )
        )
        label_loss = torch.mean(
            -0.5
            + 0.5 * torch.div((y_mu - y).pow(2), y_logvar.exp())
            + y_logvar
        )  # KL for label
        label_mse = torch.mean((y_mu - y).pow(2))
        return (
            reconstruction_loss + kld_loss + label_loss,
            reconstruction_loss,
            kld_loss,
            label_loss,
            label_mse,
        )
