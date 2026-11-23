"""Benchmark physical cases for the four 2D hybrid PDE scenarios."""

from __future__ import annotations

import importlib

__all__ = [
    "CahnHilliardCase",
    "GrayScottCase",
    "KolmogorovNSCase",
    "MHD2DCase",
    "make_case",
]

_CASE_MODULES = {
    "CahnHilliardCase": ".reaction",
    "GrayScottCase": ".reaction",
    "KolmogorovNSCase": ".spectral2d",
    "MHD2DCase": ".spectral2d",
}


def __getattr__(name: str):
    if name in _CASE_MODULES:
        module = importlib.import_module(_CASE_MODULES[name], __name__)
        return getattr(module, name)
    if name in ("reaction", "spectral2d"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_case(name: str, config=None):
    if name in ("gray_scott", "cahn_hilliard"):
        reaction = importlib.import_module(".reaction", __name__)
        cls = reaction.GrayScottCase if name == "gray_scott" else reaction.CahnHilliardCase
        return cls(config)
    if name in ("kolmogorov_ns", "mhd2d"):
        spectral = importlib.import_module(".spectral2d", __name__)
        cls = spectral.KolmogorovNSCase if name == "kolmogorov_ns" else spectral.MHD2DCase
        return cls(config)
    raise ValueError(f"unknown benchmark case: {name}")
