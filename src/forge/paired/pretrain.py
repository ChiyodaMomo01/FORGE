"""Network-only regression on frozen public G I/O. NEVER calls F/H.

Labels are constant actual G returns, decoded from declared train-only units.
There is no trajectory engine, client, numerical state, or native gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..core.io import file_digest, save
from ..core.model_serde import checkpoint
from ..networks import _radial_knots, _tent_partition
from ..schemas import (
    CANONICAL_RESIDUAL_STATISTICS_FILENAME,
    load_residual_statistics,
    validate_collection_protocol,
)
from .coarse import coarse_value
from .data import PairDataset, statistics
from .model import FullComponent


def periodic_differences(value, order):
    """Periodic finite differences at one grid-cell spacing in both directions."""
    if order == 1:
        return [torch.roll(value, -1, d) - value for d in (-2, -1)]
    if order == 2:
        return [torch.roll(value, -1, d) - 2 * value + torch.roll(value, 1, d) for d in (-2, -1)]
    raise ValueError("first or second differences only")


def _band_parseval_weights(n, h, knots, device, dtype):
    """Parseval spectral weights per band accounting for rfft2 Hermitian multiplicity."""
    kx = torch.fft.fftfreq(n, device=device, dtype=dtype) * n
    ky = torch.fft.rfftfreq(h, device=device, dtype=dtype) * h
    rho = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    tent = _tent_partition(rho, knots.to(device=device, dtype=dtype))
    mult = torch.full((n, h // 2 + 1), 2.0, device=device, dtype=dtype)
    mult[:, 0] = 1.0
    if h % 2 == 0:
        mult[:, -1] = 1.0
    return tent.square() * mult[None] / float(n * h)


def _band_parseval_energy(value, knots):
    """Per-(channel, band) spatial mean-square Parseval energy of a batch (B, C, N, H)."""
    n, h = value.shape[-2:]
    weights = _band_parseval_weights(n, h, knots, value.device, value.dtype)
    z_sq = torch.fft.rfft2(value, norm="ortho").abs().square()
    return torch.einsum("bcxy,mxy->cm", z_sq, weights) / max(len(value), 1)


def compute_train_band_statistics(train_data, model, device, distributed=False):
    """Compute per-band target variance and unresolved fraction from TRAIN pairs only."""
    n = train_data.n
    num_bands = model.network.config.head_bands
    c_out = model.action_channels
    kx = torch.fft.fftfreq(n, device=device, dtype=torch.float64) * n
    ky = torch.fft.rfftfreq(n, device=device, dtype=torch.float64) * n
    rho = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    rho_max = max(float(rho.max()), 1.0)
    knots = _radial_knots(num_bands, rho_max, device, torch.float64)

    res_energy = torch.zeros((c_out, num_bands), device=device, dtype=torch.float64)
    tot_energy = torch.zeros((c_out, num_bands), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.float64)
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    my_rank = dist.get_rank() if distributed and dist.is_initialized() else 0
    scale = (model.residual_coefficient * model.output_scale.to(device)).double()
    indices = range(my_rank, len(train_data), world)
    loader = DataLoader(
        torch.utils.data.Subset(train_data, list(indices)),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float64)
            y = y.to(device, dtype=torch.float64)
            base = coarse_value(model.case_name, x)
            r_norm = (y - base) / scale
            y_norm = y / scale
            bsz = float(len(x))
            res_energy += _band_parseval_energy(r_norm, knots) * bsz
            tot_energy += _band_parseval_energy(y_norm, knots) * bsz
            count += bsz
    if world > 1:
        dist.all_reduce(res_energy)
        dist.all_reduce(tot_energy)
        dist.all_reduce(count)
    res_var = (res_energy / count.clamp(min=1.0)).clamp(min=1e-12)
    tot_var = (tot_energy / count.clamp(min=1.0)).clamp(min=1e-12)
    unresolved_ratio = (res_energy / (tot_energy + 1e-30)).clamp(0.0, 1.0)
    return {
        "knots": knots.float().cpu(),
        "band_target_var": res_var.float().cpu(),
        "band_total_var": tot_var.float().cpu(),
        "band_unresolved_ratio": unresolved_ratio.float().cpu(),
    }


def regression_loss(prediction, target, band_stats=None):
    value = (prediction - target).square().mean()
    gradient = []
    for order in (1, 2):
        gradient.append(
            sum(
                (a - b).square().mean()
                for a, b in zip(
                    periodic_differences(prediction, order),
                    periodic_differences(target, order),
                    strict=True,
                )
            )
            / 2
        )
    loss = value + 0.1 * gradient[0] + 0.01 * gradient[1]
    terms = {
        "value_mse": float(value.detach()),
        "first_difference_mse": float(gradient[0].detach()),
        "second_difference_mse": float(gradient[1].detach()),
    }
    if band_stats is not None:
        knots = band_stats["knots"].to(prediction.device, prediction.dtype)
        unresolved = band_stats["band_unresolved_ratio"].to(prediction.device, prediction.dtype)
        resolved_fraction = (1.0 - unresolved).clamp(0.0, 1.0)
        if bool((resolved_fraction > 1e-3).any()):
            band_power = _band_parseval_energy(prediction, knots)
            band_var = band_stats["band_target_var"].to(prediction.device, prediction.dtype)
            prior_penalty = (resolved_fraction * band_power / band_var.clamp(min=1e-4)).mean()
            loss = loss + 0.05 * prior_penalty
            terms["band_prior_penalty"] = float(prior_penalty.detach())
    return loss, terms


def evaluate(model, loader, device, forward_fn=None, distributed=False):
    model.eval()
    run_model = model if forward_fn is None else forward_fn
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    my_rank = dist.get_rank() if distributed and dist.is_initialized() else 0
    total_err = torch.zeros((), device=device, dtype=torch.float64)
    total_cnt = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if world > 1 and (batch_idx % world) != my_rank:
                continue
            x, y = x.to(device), y.to(device)
            error = (run_model(x).double() - model.residual_target(x, y)).square().mean((1, 2, 3))
            if not torch.isfinite(error).all():
                raise FloatingPointError("nonfinite pair validation")
            total_err += error.sum()
            total_cnt += len(error)
    if world > 1:
        dist.all_reduce(total_err)
        dist.all_reduce(total_cnt)
    return float((total_err / total_cnt.clamp(min=1.0)).item())


def train(
    root,
    output,
    case_name,
    device="cpu",
    epochs=64,
    batch=8,
    patience=6,
    distributed=False,
):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    use_ddp = distributed or (dist.is_available() and dist.is_initialized())
    world_size = dist.get_world_size() if use_ddp else 1
    my_rank = dist.get_rank() if use_ddp else 0
    is_main = my_rank == 0
    out = Path(output)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "selected.pt").exists() or (out / "history.json").exists():
            raise FileExistsError("new pretraining attempt root required")
    if use_ddp:
        dist.barrier()
    verification = json.loads(
        (Path(root) / case_name / "restriction-audit.json").read_text(encoding="utf-8")
    )
    if verification["status"] != "passed" or verification["case"] != case_name:
        raise ValueError("verified stored-pair restriction required")
    if "protocol" in verification:
        validate_collection_protocol(verification.get("protocol"), case_name)
    train_data, validation_data = [PairDataset(root, case_name, s) for s in ("train", "validation")]
    stats = load_residual_statistics(Path(root) / case_name, case_name=case_name)
    if stats is None:
        stats = statistics(
            root,
            case_name,
            output=(out / CANONICAL_RESIDUAL_STATISTICS_FILENAME) if is_main else False,
        )
    model = FullComponent(case_name, train_data.n, 29, stats).to(device).eval()
    if use_ddp:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for b in model.buffers():
            dist.broadcast(b.data, src=0)
    band_stats = None
    if model.network.config.head_bands > 1:
        band_stats = compute_train_band_statistics(train_data, model, device, distributed=use_ddp)
        stats = {
            **stats,
            "band_knots": band_stats["knots"].tolist(),
            "band_target_var": band_stats["band_target_var"].tolist(),
            "band_unresolved_ratio": band_stats["band_unresolved_ratio"].tolist(),
        }
    forward_model = model
    loaders = [
        DataLoader(
            ds,
            batch_size=batch,
            shuffle=(i == 0),
            num_workers=0,
            generator=torch.Generator().manual_seed(12000 + i),
        )
        for i, ds in enumerate((train_data, validation_data))
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * len(loaders[0]), eta_min=1e-6
    )
    best = float("inf")
    stale = 0
    history = []
    updates = 0
    examples = 0
    begun = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if is_main:
        save(
            out / "plan.json",
            {
                "case": case_name,
                "epochs": epochs,
                "patience": patience,
                "batch": batch,
                "lr": 1e-4,
                "ddp_world_size": world_size,
                "selection": "lowest complete validation normalized residual MSE",
                "parameterization": "G_coarse+0.1*G_theta; G_theta=FNO*residual_RMS/0.1",
                "network": "full-spectrum radial-harmonic + dealiased bilinear periodic FNO, width29 layers4, post-GELU band-split head",
                "nyquist_gate": model.nyquist_gate,
                "nyquist_cutoff": model.nyquist_cutoff,
                "nyquist_tau": model.nyquist_tau,
                "parameters": sum(p.numel() for p in model.parameters()),
                "loss_weights": {
                    "value": 1.0,
                    "periodic_first_difference": 0.1,
                    "periodic_second_difference": 0.01,
                    "resolved_band_prior": 0.05,
                },
                "scheduler": "CosineAnnealingLR per update, eta_min=1e-6",
                "statistics": stats,
                "information": "frozen actual G pairs only; network-only backward; no native engine import/call",
            },
        )
    try:
        initial_validation = evaluate(model, loaders[1], device, forward_model, distributed=use_ddp)
        if is_main:
            save(
                out / "initial-validation.json",
                {
                    "pair_mse": initial_validation,
                    "role": "untrained regression diagnostic; checkpoint selection uses completed training epochs",
                },
            )
        for epoch in range(epochs):
            losses = []
            model.train()
            for x_full, y_full in loaders[0]:
                n_full = len(x_full)
                if use_ddp and n_full >= world_size:
                    x = x_full[my_rank::world_size].to(device).detach()
                    y = y_full[my_rank::world_size].to(device).detach()
                    shard_weight = float(len(x)) / float(n_full)
                else:
                    x = x_full.to(device).detach()
                    y = y_full.to(device).detach()
                    shard_weight = 1.0 / float(world_size)
                optimizer.zero_grad(set_to_none=True)
                target = model.residual_target(x, y).float().detach()
                loss, terms = regression_loss(forward_model(x), target, band_stats)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite regression update")
                (loss * shard_weight).backward()
                if x.grad is not None or y.grad is not None:
                    raise RuntimeError("data gradients forbidden")
                if use_ddp:
                    for p in model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                updates += 1
                examples += n_full
                batch_mse = torch.tensor(
                    terms["value_mse"] * shard_weight,
                    device=device,
                    dtype=torch.float64,
                )
                if use_ddp:
                    dist.all_reduce(batch_mse, op=dist.ReduceOp.SUM)
                losses.extend([float(batch_mse.item())] * n_full)
                if is_main and updates % 25 == 0:
                    save(
                        out / "progress.json",
                        {
                            "status": "training",
                            "epoch": epoch + 1,
                            "update": updates,
                            "examples": examples,
                            "latest_residual_mse": losses[-1],
                            "latest_total_loss": float(loss.detach()),
                            "learning_rate": schedule.get_last_lr()[0],
                            "wall_seconds": time.monotonic() - begun,
                        },
                    )
            val = evaluate(model, loaders[1], device, forward_model, distributed=use_ddp)
            row = {
                "epoch": epoch + 1,
                "updates": updates,
                "train_pair_mse": float(np.mean(losses)),
                "validation_pair_mse": val,
                "learning_rate": schedule.get_last_lr()[0],
            }
            history.append(row)
            if is_main:
                save(out / "history.json", history)
            if val < best:
                best = val
                stale = 0
                if is_main:
                    checkpoint(
                        out / "selected.pt",
                        model,
                        {
                            "method": "paired_pretrain",
                            "selected_epoch": epoch + 1,
                            "data_sha256": file_digest(train_data.root / "dataset.json"),
                            "statistics": stats,
                            "supervision": "frozen actual G_HR residual regression; no physics BPTT",
                        },
                    )
            else:
                stale += 1
            if stale >= patience:
                break
        if use_ddp:
            dist.barrier()
        result = {
            "status": "complete",
            "case": case_name,
            "epochs_completed": len(history),
            "planned_epochs": epochs,
            "selected_validation_pair_mse": best,
            "updates": updates,
            "train_examples": examples,
            "initial_validation_pair_mse": initial_validation,
            "parameterization": "G_coarse+0.1*G_theta",
            "residual_coefficient": 0.1,
            "parameters": sum(p.numel() for p in model.parameters()),
            "score_units": "train-only residual RMS; component regression, not deployment MSE",
            "selected_sha256": file_digest(out / "selected.pt"),
            "history": history,
            "trajectory_effect": "not measured here; requires separate public F/H evaluation",
            "native_calls": 0,
            "native_backward_calls": 0,
            "input_gradients": False,
        }
        if is_main:
            save(out / "result.json", result)
        return result
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF)
        if is_main:
            save(
                out / "resources.json",
                {
                    "pid": os.getpid(),
                    "wall_seconds": time.monotonic() - begun,
                    "cpu_seconds": after.ru_utime
                    + after.ru_stime
                    - before.ru_utime
                    - before.ru_stime,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                    if str(device).startswith("cuda")
                    else None,
                    "scope": "pretraining process only; no native F/H descendants",
                },
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=("cahn_hilliard", "gray_scott", "kolmogorov_ns", "mhd2d"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.root, args.output, args.case, args.device)


if __name__ == "__main__":
    main()
