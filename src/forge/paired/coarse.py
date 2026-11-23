"""Declared coarse closure baselines for the four 2D hybrid PDE benchmarks."""

import torch

from ..spectral_ops import dealiased_mul


def coarse_value(case_name, x):
    if case_name == "cahn_hilliard":
        return x[:, :1].pow(3)
    if case_name == "gray_scott":
        return x[:, :1] * x[:, 1:2].square()
    if case_name == "kolmogorov_ns":
        omega = x[:, 0:1]
        u = x[:, 2:3]
        v = x[:, 3:4]
        return torch.cat((dealiased_mul(u, omega), dealiased_mul(v, omega)), dim=1)
    if case_name == "mhd2d":
        omega = x[:, 0:1]
        cur_j = x[:, 3:4]
        u = x[:, 4:5]
        v = x[:, 5:6]
        bx = x[:, 6:7]
        by = x[:, 7:8]
        g1 = dealiased_mul(u, omega) - dealiased_mul(bx, cur_j)
        g2 = dealiased_mul(v, omega) - dealiased_mul(by, cur_j)
        g3 = dealiased_mul(u, by) - dealiased_mul(v, bx)
        return torch.cat((g1, g2, g3), dim=1)
    raise ValueError(f"unsupported 2D public coarse G case: {case_name}")
