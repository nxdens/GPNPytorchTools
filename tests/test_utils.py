from math import log
from gpnpytorchtools import utils
import torch


def test_generate_layer_sizes():
    utils.generate_layer_sizes(1, 10, 10, how="logspace")
    utils.generate_layer_sizes(1, 10, 10, how="linspace")
    utils.generate_layer_sizes(1, 10, 10, how="geomspace")
    try:
        utils.generate_layer_sizes(1, 10, 10, how="wrong")
    except ValueError:
        pass


def test_reparameterization():
    mu = torch.randn(10, 10)
    log_var = torch.randn(10, 10)
    utils.reparameterization(mu, log_var)


def test_logger():
    logger = utils.init_logger(filename="test.log")
    logger.info("test")
