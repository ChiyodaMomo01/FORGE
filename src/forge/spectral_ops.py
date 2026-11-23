"""Solver-free periodic Fourier restriction and 3/2 zero-padded multiplication.

Contains purely mathematical array operators on periodic 2D tensors with zero
imports from ``forge.cases`` and zero access to solver configuration or state.
"""

from __future__ import annotations

import torch


def spectral_restrict(value: torch.Tensor, n: int) -> torch.Tensor:
    """Restrict square periodic fields via Fourier truncation with zeroed Nyquist lines."""
    old = value.shape[-1]
    if value.shape[-2] != old or n < 2 or old % n or n % 2 or old % 2:
        raise ValueError("even integer square-grid restriction required")
    if old == n:
        fine = torch.fft.rfft2(value)
        half = n // 2
        fine = fine.clone()
        fine[..., half, :] = 0
        fine[..., :, -1] = 0
        return torch.fft.irfft2(fine, s=(n, n))
    fine = torch.fft.rfft2(value)
    half = n // 2
    reduced = torch.cat(
        (fine[..., :half, : half + 1], fine[..., -half:, : half + 1]), dim=-2
    ).clone()
    reduced[..., half, :] = 0
    reduced[..., :, -1] = 0
    reduced *= (n / old) ** 2
    return torch.fft.irfft2(reduced, s=(n, n))


def dealiased_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact 3/2 zero-padded periodic quadratic product; accesses no solver config."""
    if a.shape != b.shape or a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"matching square tensors required, got {a.shape} and {b.shape}")
    n = a.shape[-1]
    if n % 2:
        raise ValueError("even grid size required for 3/2 dealiasing")
    m = (3 * n) // 2
    half = n // 2

    def upsample(x: torch.Tensor) -> torch.Tensor:
        xh = torch.fft.rfft2(x).clone()
        xh[..., half, :] = 0
        xh[..., :, -1] = 0
        padded = xh.new_zeros((*xh.shape[:-2], m, m // 2 + 1))
        padded[..., :half, :half] = xh[..., :half, :half]
        padded[..., -half:, :half] = xh[..., -half:, :half]
        padded *= (m / n) ** 2
        return torch.fft.irfft2(padded, s=(m, m))

    prod_m = upsample(a) * upsample(b)
    ph = torch.fft.rfft2(prod_m)
    cropped = torch.cat((ph[..., :half, : half + 1], ph[..., -half:, : half + 1]), dim=-2).clone()
    cropped[..., half, :] = 0
    cropped[..., :, -1] = 0
    cropped *= (n / m) ** 2
    return torch.fft.irfft2(cropped, s=(n, n))
