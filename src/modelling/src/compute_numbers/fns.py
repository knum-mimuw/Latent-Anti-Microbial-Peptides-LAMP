"""Loss computation functions for use with LossManager."""

import torch
import torch.nn as nn


def kl_gaus_unitgauss(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between Gaussian posterior and unit Gaussian prior.

    KL(q(z|x) || p(z)) where:
    - q(z|x) ~ N(mean, std^2) is the posterior (encoder output)
    - p(z) ~ N(0, 1) is the unit Gaussian prior

    Formula: KL = 0.5 * sum(mean^2 + std^2 - 1 - 2*log_std)

    Args:
        mean: Latent mean [batch_size, latent_dim]
        log_std: Latent log standard deviation [batch_size, latent_dim]

    Returns:
        KL divergence (scalar tensor)
    """
    std = torch.exp(log_std)
    kl_per_sample = 0.5 * (mean.pow(2) + std.pow(2) - 1 - 2 * log_std).sum(dim=-1)
    return kl_per_sample.mean()  # Return mean over batch for loss
