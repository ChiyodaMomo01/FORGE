"""Full-rank low-frequency Gaussian: sampling and precision use one spectrum.

Modes below six dominate; a 5% spectral floor keeps the density nonsingular.
Local credit uses the exact spatial decomposition of the correlated Gaussian
log ratio. Pixels are never asserted to be independent.
"""

from __future__ import annotations

import torch

from ..core.sampling import smooth_noise, spectrum


def precision(value):
    spec = torch.as_tensor(spectrum(*value.shape[-2:]), device=value.device, dtype=value.dtype)
    return torch.fft.irfft2(
        torch.fft.rfft2(value, norm="ortho") / spec.square(),
        s=value.shape[-2:],
        norm="ortho",
    )


def statistics(mean, old, action, sigma, group_size=1, local=False):
    delta = mean.double() - old.double()
    noise = action.double() - old.double()
    pd = precision(delta)
    contribution = ((noise * pd - 0.5 * delta * pd) / sigma**2).sum(1)
    kl = (0.5 * delta * pd / sigma**2).sum(1)
    if local:
        return contribution, kl
    return contribution.reshape(-1, group_size, *contribution.shape[-2:]).sum(
        (1, 2, 3)
    ), kl.reshape(-1, group_size, *kl.shape[-2:]).sum((1, 2, 3))


__all__ = ["precision", "smooth_noise", "spectrum", "statistics"]
