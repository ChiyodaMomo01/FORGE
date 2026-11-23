"""Deterministic window split construction, per-epoch sampling, and cold-start initialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .artifacts import atomic_torch_save
from .config import ForgePOM3DConfig
from .distributed import is_main_process
from .network import ForgePOMNet


def build_train_window_pool(cfg: ForgePOM3DConfig) -> list[tuple[int, int]]:
    pool: list[tuple[int, int]] = []
    for m in cfg.train_members:
        for d in range(cfg.train_start_day, cfg.train_end_day, cfg.train_stride_days):
            pool.append((m, d))
    return pool


def sample_epoch_windows(
    pool: list[tuple[int, int]],
    round_idx: int,
    batch_size: int,
    seed: int,
    stream_id: int,
) -> list[tuple[int, int]]:
    """Deterministically traverse `pool` via seeded per-epoch permutations without replacement."""
    n_pool = len(pool)
    if n_pool == 0 or batch_size <= 0:
        return []
    if batch_size >= n_pool:
        perm = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(stream_id), int(round_idx)])
        ).permutation(n_pool)
        return [pool[int(perm[i % n_pool])] for i in range(batch_size)]

    start_pos = (max(int(round_idx), 1) - 1) * batch_size
    end_pos = start_pos + batch_size
    perm_cache: dict[int, np.ndarray] = {}

    def _get_perm(ep: int) -> np.ndarray:
        if ep not in perm_cache:
            perm_cache[ep] = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(stream_id), int(ep)])
            ).permutation(n_pool)
        return perm_cache[ep]

    chosen_indices: list[int] = []
    chosen_set: set[int] = set()
    for pos in range(start_pos, end_pos):
        ep = pos // n_pool
        off = pos % n_pool
        perm = _get_perm(ep)
        idx = int(perm[off])
        if idx in chosen_set:
            for alt in perm:
                alt_i = int(alt)
                if alt_i not in chosen_set:
                    idx = alt_i
                    break
        chosen_indices.append(idx)
        chosen_set.add(idx)
    return [pool[i] for i in chosen_indices]


def build_val_windows(cfg: ForgePOM3DConfig) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for m in cfg.val_members:
        for d in range(cfg.val_start_day, cfg.val_end_day, cfg.val_stride_days):
            windows.append((m, d))
    return windows[:16]


def build_test_id_48_windows(cfg: ForgePOM3DConfig) -> list[tuple[int, int]]:
    """Construct the 48 strided 4-hour Test ID windows on `member=7` (`8000, 8020, ..., 8940`)."""
    m = int(cfg.eval_member)
    return [
        (m, int(d))
        for d in range(int(cfg.test_start_day), int(cfg.test_end_day), int(cfg.test_stride_days))
    ]


def build_test_id_continuity8_windows(cfg: ForgePOM3DConfig) -> list[tuple[int, int]]:
    """Construct the 8 continuity `eval_starts` 4-hour windows on `member=7` (`8000, 8064, ..., 8448`)."""
    m = int(cfg.eval_member)
    return [(m, int(d)) for d in cfg.eval_starts]


def build_fixed_24h_train_starts(
    train_pool: list[tuple[int, int]], count: int = 8
) -> list[tuple[int, int]]:
    """Select `count` deterministic evenly spaced training starts."""
    n = len(train_pool)
    if n <= count:
        return list(train_pool)
    step = max(n // count, 1)
    return [train_pool[(i * step) % n] for i in range(count)]


def ensure_common_zero_ckpt(
    cfg: ForgePOM3DConfig,
    rank: int,
    device: torch.device,
    explicit_ckpt: Path | None = None,
) -> Path:
    """Create or verify `common_zero.pt` with seeded random trunk and exact zero readout head."""
    zero_ckpt = (
        explicit_ckpt if explicit_ckpt is not None else (Path(cfg.output_root) / "common_zero.pt")
    )
    if not zero_ckpt.exists():
        torch.manual_seed(int(cfg.seed))
        model = ForgePOMNet(
            cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks
        ).to(device)
        model.reset_head_zero()
        if is_main_process(rank):
            atomic_torch_save(model.state_dict(), zero_ckpt)
        if dist.is_initialized():
            dist.barrier()

    state = torch.load(zero_ckpt, map_location="cpu", weights_only=True)
    head_max = max(
        float(state["head.weight"].abs().max().item()),
        float(state["head.bias"].abs().max().item()),
    )
    if head_max > 0.0:
        raise ValueError(
            f"common_zero.pt head is non-zero (max_abs={head_max}); illegal warm-start rejected"
        )
    return zero_ckpt


__all__ = [
    "build_fixed_24h_train_starts",
    "build_test_id_48_windows",
    "build_test_id_continuity8_windows",
    "build_train_window_pool",
    "build_val_windows",
    "ensure_common_zero_ckpt",
    "sample_epoch_windows",
]
