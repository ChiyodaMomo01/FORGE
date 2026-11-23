"""Exact spatial restriction operators for the four 2D hybrid PDE benchmarks."""

from torch.nn.functional import avg_pool2d

from ..spectral_ops import spectral_restrict


def restrict(value, n, case):
    old = value.shape[-1]
    if value.shape[-2] != old or n < 2 or old % n:
        raise ValueError("integer square-grid restriction required")
    if case in ("cahn_hilliard", "gray_scott"):
        return avg_pool2d(value, old // n) if old != n else value.clone()
    if case in ("kolmogorov_ns", "mhd2d"):
        return spectral_restrict(value, n)
    raise ValueError(f"unsupported 2D restriction case: {case}")
