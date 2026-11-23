"""Shared closure model serialization and parameter-vector helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from ..paired.model import FullComponent, assign, vector
from ..schemas import (
    ALL_2D_CASES,
    CANONICAL_CHECKPOINT_SEMANTICS,
    validate_checkpoint_semantics,
)
from .io import save_clean_json as save


def digest(x):
    return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()


def array(m):
    return vector(m).cpu().numpy()


def set_model(m, x):
    assign(m, torch.as_tensor(x, device=next(m.parameters()).device))


def actor(
    n=96,
    width=29,
    device="cpu",
    case_name="cahn_hilliard",
    statistics=None,
    modes=6,
    layers=4,
    radial_bins=None,
    bilinear_channels=None,
    head_bands=None,
    nyquist_gate=False,
    nyquist_cutoff=42.0,
    nyquist_tau=5e-4,
):
    if case_name not in ALL_2D_CASES:
        raise ValueError("unauthorized case")
    return (
        FullComponent(
            case_name,
            n=n,
            width=width,
            statistics=statistics,
            modes=modes,
            layers=layers,
            radial_bins=radial_bins,
            bilinear_channels=bilinear_channels,
            head_bands=head_bands,
            nyquist_gate=nyquist_gate,
            nyquist_cutoff=nyquist_cutoff,
            nyquist_tau=nyquist_tau,
        )
        .to(device)
        .eval()
    )


def load_model(path, device="cpu"):
    z = torch.load(path, map_location="cpu", weights_only=True)
    meta = z["metadata"]
    validate_checkpoint_semantics(meta.get("semantics"))
    has_v2_arch = "radial_bins" in meta or meta.get("arch_version") in (
        "radial_bilinear_bandsplit_v1",
        "radial_bilinear_bandsplit_nyquist_gate_v2",
    )
    has_nyquist_gate = bool(
        meta.get("nyquist_gate", False)
        or meta.get("arch_version") == "radial_bilinear_bandsplit_nyquist_gate_v2"
    )
    m = actor(
        meta["n"],
        meta["width"],
        device,
        meta["case_name"],
        modes=meta.get("modes", 6),
        layers=meta.get("layers", 4),
        radial_bins=meta.get("radial_bins", 14) if has_v2_arch else 0,
        bilinear_channels=meta.get("bilinear_channels", 16) if has_v2_arch else 0,
        head_bands=meta.get("head_bands", 6) if has_v2_arch else 1,
        nyquist_gate=has_nyquist_gate,
        nyquist_cutoff=meta.get("nyquist_cutoff", 42.0),
        nyquist_tau=meta.get("nyquist_tau", 5e-4),
    )
    m.load_state_dict(z["model"], strict=True)
    return m, meta


def checkpoint(path, m, metadata):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    cfg = m.network.config
    torch.save(
        {
            "model": m.state_dict(),
            "metadata": {
                "n": cfg.nx,
                "width": cfg.width,
                "modes": cfg.modes,
                "layers": cfg.layers,
                "radial_bins": cfg.radial_bins,
                "bilinear_channels": cfg.bilinear_channels,
                "head_bands": cfg.head_bands,
                "nyquist_gate": bool(cfg.nyquist_gate),
                "nyquist_cutoff": float(cfg.nyquist_cutoff),
                "nyquist_tau": float(cfg.nyquist_tau),
                "arch_version": (
                    "legacy_fno_v1"
                    if m.network.legacy
                    else "radial_bilinear_bandsplit_nyquist_gate_v2"
                    if cfg.nyquist_gate
                    else "radial_bilinear_bandsplit_v1"
                ),
                "case_name": m.case_name,
                "policy_hash": digest(array(m)),
                "semantics": CANONICAL_CHECKPOINT_SEMANTICS,
                **metadata,
            },
        },
        tmp,
    )
    tmp.replace(p)


def indices(m):
    h, b, s = [], [], []
    start = 0
    for name, p in m.named_parameters():
        stop = start + p.numel()
        (h if ".head." in name else b).extend(range(start, stop))
        s.append((name, slice(start, stop), p.shape))
        start = stop
    return np.asarray(h), np.asarray(b), s


def basis(m, seed, r=16, obs=None):
    h, b, _ = indices(m)
    rng = np.random.default_rng(seed)
    u = np.zeros((len(array(m)), r), np.float32)
    for ids in (h, b):
        q = np.linalg.qr(rng.normal(size=(len(ids), r)), mode="reduced")[0]
        u[ids] = q / np.sqrt(2)
    return u


def good(z):
    return (
        np.isfinite(z["mean_mse"])
        and not z.get("failures")
        and z.get("uncapped_nonfinite_outputs", 0) == 0
    )


def parallel_network(model):
    """Serial fallback; multi-device training dispatches explicit DDP jobs."""
    return model


__all__ = [
    "actor",
    "array",
    "basis",
    "checkpoint",
    "digest",
    "good",
    "indices",
    "load_model",
    "parallel_network",
    "save",
    "set_model",
]
