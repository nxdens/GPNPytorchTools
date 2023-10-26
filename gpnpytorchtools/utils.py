import torch
import numpy as np
import logging
from lightning_utilities.core.rank_zero import (
    rank_prefixed_message,
    rank_zero_only,
)
from typing import Optional, Mapping


def generate_layer_sizes(input_size, output_size, layers, how="geomspace"):
    """Generates a list of layer sizes constrained by the input and output sizes.

    Args:
        input_size (int): first layer size
        output_size (int): last layer size
        layers (int): number of layers
        how (str, optional): _description_. Defaults to "geomspace". Must be one of 'logspace', 'linspace', 'geomspace'

    Raises:
        ValueError: Error raised if how is not one of 'logspace', 'linspace', 'geomspace'

    Returns:
        list: list of ints representing the layer sizes
    """
    if how == "logspace":
        output_power = np.floor(np.log2(output_size))
        input_power = np.ceil(np.log2(input_size))
        layer_sizes = np.logspace(
            input_power + 2, output_power, layers, base=2.0
        ).astype(int)
        layer_sizes[0] = input_size
        layer_sizes[-1] = output_size
    elif how == "linspace":
        layer_sizes = np.linspace(
            input_size, output_size, layers, endpoint=True
        ).astype(int)
    elif how == "geomspace":
        layer_sizes = np.geomspace(
            input_size, output_size, layers, endpoint=True
        ).astype(int)
    else:
        raise ValueError(
            "how must be one of 'logspace', 'linspace', 'geomspace'"
        )

    return list(layer_sizes)


def reparameterization(mu, logvar):
    """Reparametrization function for sampling gaussian distributions.

    Args:
        mu (torch.Tensor): mean parameter
        logvar (torch.Tensor): log variance parameter

    Returns:
        torch.Tensor: sampled latent vector
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def init_logger(filename="gpnpytorchtools.log"):
    logging.basicConfig(filename=filename, level=logging.DEBUG, filemode="w")
    logger = logging.getLogger(__name__)  # access logger by __name__
    return logger
