"""Privileged trajectory AD, persistent one-process/one-GPU DDP."""

import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..learning.common import array, checkpoint, digest, load_model, save
from ..learning.distributed import rank, size, wrap
from ..learning.distributed_jobs import close_client, query, setup
from ..learning.schema import epoch_batches
from ..runtime.data import WindowBank
from ..runtime.engine import case_for
from .trajectory import (
    _snapshot_body_vector,
    _summarize_grad_norms,
    _summarize_parameters,
    _TrajectoryShard,
)


def train(e, output, recipe, folder_name="AD", reference=None):
    root = Path(reference) if reference is not None else Path(output)
    folder = Path(output) / folder_name
    bank = WindowBank(root, recipe)
    model, meta = load_model(root / "common.pt", e.device)
    if model.case_name != recipe.case_name:
        raise ValueError("cross-case checkpoint rejected")
    initial_body = _snapshot_body_vector(model)
    start_hash = digest(array(model))
    if rank() == 0:
        save(
            folder / "initialization.json",
            {
                "policy_hash": start_hash,
                "shared_pretrained": meta.get("method") == "paired_pretrain",
                "initialization": meta.get("method"),
                "source": "common.pt",
            },
        )
    case = case_for(bank, "train")
    trajectory = wrap(_TrajectoryShard(case, model, bank.info["field_std"], recipe.sample_dt))
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.ad_lr)
    groups = len(bank.maps["train"])
    batches_per_epoch = math.ceil(groups / recipe.gradient_batch_groups)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, recipe.ad_epochs * batches_per_epoch, eta_min=recipe.ad_lr * 0.1
    )
    setup(
        e,
        {
            "root": str(root),
            "folder": str(folder),
            "recipe": recipe,
            "checkpoint": str(root / "common.pt"),
        },
    )

    def evaluate(label):
        value = query(
            e,
            {
                "theta": array(model),
                "split": "validation",
                "trials": None,
                "kwargs": {"label": label},
            },
        )
        small = e.broadcast(
            {
                "mean_mse": value["mean_mse"],
                "failures": value["failures"],
                "uncapped_nonfinite_outputs": value["uncapped_nonfinite_outputs"],
                "cost": {
                    "window_attempts": int(value.get("cost", {}).get("window_attempts", 0)),
                    "output_attempts": int(value.get("cost", {}).get("output_attempts", 0)),
                },
            }
            if rank() == 0
            else None
        )
        return value if rank() == 0 else small

    val = evaluate("shared-pretrained-validation")
    best, chosen, stale, plateau = val["mean_mse"], 0, 0, float("inf")
    history = []
    init_val_win = int(val.get("cost", {}).get("window_attempts", 0))
    init_val_out = int(val.get("cost", {}).get("output_attempts", 0))
    cost = {
        k: 0
        for k in (
            "optimizer_updates",
            "window_attempts",
            "validation_window_attempts",
            "validation_output_attempts",
            "training_and_selection_window_attempts",
            "post_selection_validation_window_attempts",
            "output_targets",
            "planned_internal_steps_for_window_attempts",
            "learner_G_forward_examples_including_recompute",
            "loss_backward_calls",
            "backward_window_attempts",
            "nonfinite_updates",
        )
    }
    cost["validation_window_attempts"] = init_val_win
    cost["validation_output_attempts"] = init_val_out
    cost["training_and_selection_window_attempts"] = init_val_win
    if rank() == 0:
        checkpoint(
            folder / "selected.pt",
            model,
            {"method": "AD", "selected_epoch": 0, "shared_pretrained_hash": start_hash},
        )
        checkpoint(
            folder / "checkpoints" / "best-epoch-000.pt",
            model,
            {
                "method": "AD",
                "selected_epoch": 0,
                "validation_mse": best,
                "window_attempts": 0,
                "validation_window_attempts": init_val_win,
                "training_and_selection_window_attempts": init_val_win,
                "shared_pretrained_hash": start_hash,
            },
        )
    forwards = [0]

    def count(module, args):
        forwards[0] += len(args[0])

    hook = model.register_forward_pre_hook(count)
    start_epoch = 0
    resume_batch_offset = 0
    resume_running = []
    resume_state_path = os.environ.get("FORGE_RESUME_AD_STATE", "").strip()
    if resume_state_path:
        rp = Path(resume_state_path)
        st = torch.load(rp, map_location=e.device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        schedule.load_state_dict(st["schedule"])
        cost = dict(st["cost"])
        history = list(st["history"])
        best = float(st["best"])
        chosen = int(st["chosen"])
        start_epoch = int(st["epoch"])
        resume_batch_offset = int(st.get("batch_offset", 0))
        resume_running = [float(x) for x in st.get("running_losses", [])]
        stale, plateau = 0, float("inf")
        for h_item in history:
            v_mse = float(h_item["validation"]["mean_mse"])
            if v_mse < plateau * (1 - recipe.ad_plateau_relative):
                plateau, stale = v_mse, 0
            else:
                stale += 1
        if rank() == 0:
            src_sel = rp.parent / "selected.pt"
            if src_sel.is_file():
                shutil.copy2(src_sel, folder / "selected.pt")
            src_ckpts = rp.parent / "checkpoints"
            if src_ckpts.is_dir():
                (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
                for ck in src_ckpts.glob("*.pt"):
                    shutil.copy2(ck, folder / "checkpoints" / ck.name)
            save(folder / "history.json", history)
        dist.barrier()
    begun = time.monotonic()
    try:
        for epoch in range(start_epoch, recipe.ad_epochs):
            model.eval()
            running = list(resume_running) if epoch == start_epoch else []
            grad_norms = []
            for batch_index, ids in enumerate(
                epoch_batches(groups, epoch, recipe.seed, recipe.gradient_batch_groups)
            ):
                if epoch == start_epoch and batch_index < resume_batch_offset:
                    continue
                n = 4 * len(ids)
                if n % size():
                    raise ValueError(
                        "AD global batch must divide ranks; never silently change batch"
                    )
                positions = range(rank(), n, size())
                x, y, _ = bank.cohort("train", ids, e.device, positions=positions)
                optimizer.zero_grad(set_to_none=True)
                forwards[0] = 0
                loss = trajectory(x, y)
                loss.backward()
                cost["loss_backward_calls"] += 1
                cost["window_attempts"] += n
                cost["backward_window_attempts"] += n
                cost["output_targets"] += n * recipe.horizon
                cost["planned_internal_steps_for_window_attempts"] += (
                    n
                    * recipe.horizon
                    * max(
                        1,
                        math.ceil(recipe.sample_dt / case.default_dt(x.shape[-1]) - 1e-10),
                    )
                )
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                except BaseException:
                    cost["nonfinite_updates"] += 1
                    cost["training_and_selection_window_attempts"] = (
                        cost["window_attempts"] + cost["validation_window_attempts"]
                    )
                    if rank() == 0:
                        save(
                            folder / "progress.json",
                            {
                                "status": "failed",
                                "epoch": epoch + 1,
                                "epochs": recipe.ad_epochs,
                                "update": cost["optimizer_updates"],
                                "planned_updates": recipe.ad_epochs * batches_per_epoch,
                                "cost": cost,
                                "execution": "DDP",
                                "world_size": size(),
                            },
                        )
                    raise
                grad_norms.append(float(grad_norm.detach()))
                optimizer.step()
                schedule.step()
                metrics = torch.tensor(
                    [float(loss.detach()) / size(), forwards[0]],
                    dtype=torch.float64,
                    device=e.device,
                )
                dist.all_reduce(metrics)
                running.append(float(metrics[0]))
                cost["optimizer_updates"] += 1
                cost["learner_G_forward_examples_including_recompute"] += int(metrics[1])
                if rank() == 0 and cost["optimizer_updates"] % 10 == 0:
                    save(
                        folder / "progress.json",
                        {
                            "status": "training",
                            "epoch": epoch + 1,
                            "epochs": recipe.ad_epochs,
                            "update": cost["optimizer_updates"],
                            "planned_updates": recipe.ad_epochs * batches_per_epoch,
                            "latest_loss": running[-1],
                            "cost": cost,
                            "execution": "DDP",
                            "world_size": size(),
                        },
                    )
                if rank() == 0 and cost["optimizer_updates"] % 100 == 0:
                    tmp = folder / "training-state.tmp"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "schedule": schedule.state_dict(),
                            "cost": cost,
                            "history": history,
                            "best": best,
                            "chosen": chosen,
                            "epoch": epoch,
                            "batch_offset": batch_index + 1,
                            "running_losses": running,
                        },
                        tmp,
                    )
                    tmp.replace(folder / "training-state.pt")
            val = evaluate(f"epoch-{epoch + 1}-validation")
            val_win = int(val.get("cost", {}).get("window_attempts", 0))
            val_out = int(val.get("cost", {}).get("output_attempts", 0))
            cost["validation_window_attempts"] += val_win
            cost["validation_output_attempts"] += val_out
            cost["training_and_selection_window_attempts"] = (
                cost["window_attempts"] + cost["validation_window_attempts"]
            )
            if not val["failures"] and val["mean_mse"] < best:
                best, chosen = val["mean_mse"], epoch + 1
                if rank() == 0:
                    checkpoint(
                        folder / "selected.pt",
                        model,
                        {
                            "method": "AD",
                            "selected_epoch": chosen,
                            "shared_pretrained_hash": start_hash,
                        },
                    )
                    checkpoint(
                        folder / "checkpoints" / f"best-epoch-{chosen:03d}.pt",
                        model,
                        {
                            "method": "AD",
                            "selected_epoch": chosen,
                            "validation_mse": best,
                            "window_attempts": cost["window_attempts"],
                            "validation_window_attempts": cost["validation_window_attempts"],
                            "training_and_selection_window_attempts": cost[
                                "training_and_selection_window_attempts"
                            ],
                            "shared_pretrained_hash": start_hash,
                        },
                    )
            history.append(
                {
                    "epoch": epoch + 1,
                    "updates": cost["optimizer_updates"],
                    "window_attempts": cost["window_attempts"],
                    "validation_window_attempts": cost["validation_window_attempts"],
                    "training_and_selection_window_attempts": cost[
                        "training_and_selection_window_attempts"
                    ],
                    "training_batch_mean": float(np.mean(running)),
                    "grad_norm_pre_clip": _summarize_grad_norms(grad_norms, 1.0),
                    "parameters": _summarize_parameters(model, initial_body),
                    "lr_end_of_epoch": float(schedule.get_last_lr()[0]),
                    "validation": val,
                    "seconds": time.monotonic() - begun,
                }
            )
            if rank() == 0:
                checkpoint(
                    folder / "latest.pt",
                    model,
                    {
                        "method": "AD",
                        "epoch": epoch + 1,
                        "shared_pretrained_hash": start_hash,
                    },
                )
                save(folder / "history.json", history)
            if val["mean_mse"] < plateau * (1 - recipe.ad_plateau_relative):
                plateau, stale = val["mean_mse"], 0
            else:
                stale += 1
            if recipe.ad_patience and stale >= recipe.ad_patience:
                break
    except BaseException:
        close_client(e, None)
        raise
    finally:
        hook.remove()
    dist.barrier()
    selected, _ = load_model(folder / "selected.pt", e.device)
    model.load_state_dict(selected.state_dict())
    val = evaluate("final-validation")
    cost["post_selection_validation_window_attempts"] = int(
        val.get("cost", {}).get("window_attempts", 0)
    )
    cost["training_and_selection_window_attempts"] = (
        cost["window_attempts"] + cost["validation_window_attempts"]
    )
    result = {
        "status": "complete",
        "method": "AD",
        "selected_epoch": chosen,
        "selected_hash": digest(array(selected)),
        "start_hash": start_hash,
        "epochs_completed": len(history),
        "planned_epochs": recipe.ad_epochs,
        "early_stopped": len(history) < recipe.ad_epochs,
        "history": history,
        "evaluations": {"validation": val},
        "cost": cost,
        "seconds": time.monotonic() - begun,
        "execution": "DDP",
        "world_size": size(),
        "supervision": "PRIVILEGED native derivatives; same case-specific common and unchanged global batches",
    }
    from ..core.io import sanitize_persisted_paths

    result = sanitize_persisted_paths(result, folder)
    if rank() == 0:
        save(folder / "result.json", result)
    close_client(e, None)
    return result
