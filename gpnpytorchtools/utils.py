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


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)  # type: ignore
        self.rank_zero_only = rank_zero_only

    def log(
        self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
