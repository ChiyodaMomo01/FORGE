"""Differentiable Solver BPTT (AD) reference trajectory trainer."""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..learning.common import array, checkpoint, digest, load_model, save
from ..learning.schema import RECIPE, epoch_batches
from ..runtime.data import WindowBank
from ..runtime.engine import case_for, trajectory_loss


class _TrajectoryShard(torch.nn.Module):
    def __init__(self, case, model, scale, dt):
        super().__init__()
        self.case = case
        self.model = model
        self.scale = scale
        self.dt = dt

    def forward(self, initial, target):
        return trajectory_loss(self.case, initial, target, self.model, self.scale, self.dt, True)


def _snapshot_body_vector(model):
    body = [
        p.detach().double().reshape(-1).cpu()
        for n, p in model.named_parameters()
        if not n.startswith("network.head.")
    ]
    return torch.cat(body) if body else torch.zeros(0, dtype=torch.float64)


def _summarize_parameters(model, initial_body):
    head_parts = [
        p.detach().double().reshape(-1).cpu()
        for n, p in model.named_parameters()
        if n.startswith("network.head.")
    ]
    head_vec = torch.cat(head_parts) if head_parts else torch.zeros(0, dtype=torch.float64)
    body_vec = _snapshot_body_vector(model)
    return {
        "head_l2": float(torch.linalg.vector_norm(head_vec)) if len(head_vec) else 0.0,
        "head_max_abs": float(head_vec.abs().max()) if len(head_vec) else 0.0,
        "body_delta_l2": float(torch.linalg.vector_norm(body_vec - initial_body))
        if len(body_vec)
        else 0.0,
    }


def _summarize_grad_norms(grad_norms, clip_value=1.0):
    if not grad_norms:
        return {"median": 0.0, "max": 0.0, "fraction_clipped": 0.0}
    arr = np.asarray(grad_norms, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "fraction_clipped": float(np.mean(arr > clip_value)),
    }


def train(output, device, finetune=True, recipe=RECIPE, folder_name="AD", reference=None):
    if not finetune:
        raise ValueError("Solver BPTT (AD) reference trainer requires finetune=True")
    from ..learning.numeric import configure

    actor_numeric = configure(recipe)
    from ..learning.client import ShardedClient as Client

    root = Path(reference) if reference is not None else Path(output)
    folder = Path(output) / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        raise FileExistsError("completed run cannot be overwritten")
    bank = WindowBank(root, recipe)
    torch.manual_seed(recipe.seed)
    model, metadata = load_model(root / "common.pt", device)
    if model.case_name != recipe.case_name:
        raise ValueError("cross-case checkpoint rejected")
    initial_body = _snapshot_body_vector(model)
    start_hash = digest(array(model))
    save(
        folder / "initialization.json",
        {
            "policy_hash": start_hash,
            "shared_pretrained": metadata.get("method") == "paired_pretrain",
            "initialization": metadata.get("method"),
            "source": "common.pt"
            if finetune
            else "random zero-output network, supervised pretraining only",
        },
    )
    case = case_for(bank, "train")
    groups = len(bank.maps["train"])
    epochs = recipe.ad_epochs
    world_size = (
        int(os.environ.get("FORGE_DDP_WORLD_SIZE", "1")) if str(device).startswith("cuda") else 1
    )
    use_multi_gpu = (
        str(device).startswith("cuda")
        and world_size > 1
        and torch.cuda.device_count() >= world_size
    )
    shard_cases = [case_for(bank, "train") for _ in range(world_size)] if use_multi_gpu else [case]
    lr = recipe.ad_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batches_per_epoch = math.ceil(groups / recipe.gradient_batch_groups)
    updates = epochs * batches_per_epoch
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=lr * 0.1)
    history = []
    best = float("inf")
    chosen = None
    cost = {
        "optimizer_updates": 0,
        "window_attempts": 0,
        "output_targets": 0,
        "planned_internal_steps_for_window_attempts": 0,
        "learner_G_forward_examples_including_recompute": 0,
        "loss_backward_calls": 0,
        "backward_window_attempts": 0,
        "nonfinite_updates": 0,
    }
    begun = time.monotonic()
    # Validation uses the same detached CPU runtime as every black-box method.
    with Client(
        root,
        folder,
        device,
        recipe,
        checkpoint_path=root / "common.pt" if finetune else None,
    ) as oracle:
        if finetune:
            val = oracle.score(
                array(model), "validation", None, label="shared-pretrained-validation"
            )
            best = val["mean_mse"]
            chosen = 0
            if not (folder / "training-state.pt").exists():
                checkpoint(
                    folder / "selected.pt",
                    model,
                    {
                        "method": "AD",
                        "selected_epoch": 0,
                        "shared_pretrained_hash": start_hash,
                    },
                )
        start_epoch = 0
        state_path = folder / "training-state.pt"
        if state_path.exists():
            prior = torch.load(state_path, map_location=device, weights_only=False)
            model.load_state_dict(prior["model"])
            optimizer.load_state_dict(prior["optimizer"])
            schedule.load_state_dict(prior["schedule"])
            cost = prior["cost"]
            history = prior["history"]
            best = prior["best"]
            chosen = prior["chosen"]
            start_epoch = prior["epoch"]
        plateau_best = float("inf")
        stale_epochs = 0
        for row in history:
            value = row["validation"]["mean_mse"]
            if value < plateau_best * (1 - recipe.ad_plateau_relative):
                plateau_best = value
                stale_epochs = 0
            else:
                stale_epochs += 1

        def count_forward(module, args):
            cost["learner_G_forward_examples_including_recompute"] += len(args[0])

        hook = model.register_forward_pre_hook(count_forward)
        try:
            for epoch in range(start_epoch, epochs):
                running = (
                    list(prior.get("running_losses", []))
                    if state_path.exists() and epoch == start_epoch and prior["batch_offset"]
                    else []
                )
                grad_norms = []
                model.eval()
                batches = epoch_batches(groups, epoch, recipe.seed, recipe.gradient_batch_groups)
                offset = (
                    prior["batch_offset"] if state_path.exists() and epoch == start_epoch else 0
                )
                for batch_index, ids in enumerate(batches):
                    if batch_index < offset:
                        continue
                    x, y, rows = bank.cohort("train", ids, device)
                    cost["window_attempts"] += len(x)
                    cost["output_targets"] += len(x) * recipe.horizon
                    cost["planned_internal_steps_for_window_attempts"] += (
                        len(x)
                        * recipe.horizon
                        * max(
                            1,
                            math.ceil(recipe.sample_dt / case.default_dt(x.shape[-1]) - 1e-10),
                        )
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if use_multi_gpu and len(x) >= world_size and len(x) % world_size == 0:
                        devices = [torch.device(f"cuda:{r}") for r in range(world_size)]
                        replicas = torch.nn.parallel.replicate(model, devices)
                        shards = [
                            _TrajectoryShard(
                                shard_cases[r],
                                replicas[r],
                                bank.info["field_std"],
                                recipe.sample_dt,
                            )
                            for r in range(world_size)
                        ]
                        xs = x.chunk(world_size, dim=0)
                        ys = y.chunk(world_size, dim=0)
                        inputs = [
                            (xs[r].to(devices[r]), ys[r].to(devices[r])) for r in range(world_size)
                        ]
                        losses = torch.nn.parallel.parallel_apply(shards, inputs, devices=devices)
                        loss = sum(loss.to(device) for loss in losses) / len(losses)
                        recompute_factor = (
                            2
                            * recipe.horizon
                            * max(
                                1,
                                math.ceil(recipe.sample_dt / case.default_dt(x.shape[-1]) - 1e-10),
                            )
                            * (
                                4
                                if recipe.case_name == "gray_scott"
                                else 8
                                if recipe.case_name in ("kolmogorov_ns", "mhd2d")
                                else 1
                            )
                        )
                        cost["learner_G_forward_examples_including_recompute"] += (
                            len(x) * (world_size - 1) / world_size * recompute_factor
                        )
                    else:
                        loss = trajectory_loss(
                            case,
                            x,
                            y,
                            model,
                            bank.info["field_std"],
                            recipe.sample_dt,
                            True,
                        )
                    loss.backward()
                    cost["backward_window_attempts"] += len(x)
                    cost["loss_backward_calls"] += 1
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0, error_if_nonfinite=True
                        )
                    except BaseException:
                        cost["nonfinite_updates"] += 1
                        save(
                            folder / "progress.json",
                            {
                                "status": "failed",
                                "epoch": epoch + 1,
                                "epochs": epochs,
                                "update": cost["optimizer_updates"],
                                "planned_updates": updates,
                                "cost": cost,
                            },
                        )
                        raise
                    grad_norms.append(float(grad_norm.detach()))
                    optimizer.step()
                    schedule.step()
                    cost["optimizer_updates"] += 1
                    running.append(float(loss.detach()))
                    if cost["optimizer_updates"] % 100 == 0:
                        tmp = state_path.with_suffix(".tmp")
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
                        tmp.replace(state_path)
                    if cost["optimizer_updates"] % 10 == 0:
                        save(
                            folder / "progress.json",
                            {
                                "status": "training",
                                "epoch": epoch + 1,
                                "epochs": epochs,
                                "update": cost["optimizer_updates"],
                                "planned_updates": updates,
                                "latest_loss": running[-1],
                                "cost": cost,
                            },
                        )
                checkpoint(
                    folder / "latest.pt",
                    model,
                    {
                        "method": "AD",
                        "epoch": epoch + 1,
                        "shared_pretrained_hash": start_hash if finetune else None,
                    },
                )
                val = oracle.score(
                    array(model),
                    "validation",
                    None,
                    label=f"epoch-{epoch + 1}-validation",
                )
                row = {
                    "epoch": epoch + 1,
                    "updates": cost["optimizer_updates"],
                    "training_batch_mean": float(np.mean(running)),
                    "grad_norm_pre_clip": _summarize_grad_norms(grad_norms, 1.0),
                    "parameters": _summarize_parameters(model, initial_body),
                    "lr_end_of_epoch": float(schedule.get_last_lr()[0]),
                    "validation": val,
                    "seconds": time.monotonic() - begun,
                }
                if not val["failures"] and val["mean_mse"] < best:
                    best = val["mean_mse"]
                    chosen = epoch + 1
                    checkpoint(
                        folder / "selected.pt",
                        model,
                        {
                            "method": "AD",
                            "selected_epoch": chosen,
                            "shared_pretrained_hash": start_hash if finetune else None,
                        },
                    )
                history.append(row)
                save(folder / "history.json", history)
                tmp = state_path.with_suffix(".tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "schedule": schedule.state_dict(),
                        "cost": cost,
                        "history": history,
                        "best": best,
                        "chosen": chosen,
                        "epoch": epoch + 1,
                        "batch_offset": 0,
                        "running_losses": [],
                    },
                    tmp,
                )
                tmp.replace(state_path)
                if val["mean_mse"] < plateau_best * (1 - recipe.ad_plateau_relative):
                    plateau_best = val["mean_mse"]
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if finetune and recipe.ad_patience and stale_epochs >= recipe.ad_patience:
                    break
        finally:
            hook.remove()
        if chosen is None:
            raise RuntimeError("no finite trained pretraining checkpoint")
        selected, meta = load_model(folder / "selected.pt", device)
        evaluations = {
            s: oracle.score(array(selected), s, None, label="final-" + s)
            for s in (("validation",) if recipe.defer_test_id else ("validation", "test_id"))
        }
        result = {
            "status": "complete",
            "method": "AD",
            "selected_epoch": chosen,
            "selected_hash": digest(array(selected)),
            "start_hash": start_hash,
            "epochs_completed": history[-1]["epoch"],
            "planned_epochs": epochs,
            "early_stopped": history[-1]["epoch"] < epochs,
            "history": history,
            "evaluations": evaluations,
            "cost": cost,
            "seconds": time.monotonic() - begun,
            "actor_numeric": actor_numeric,
            "supervision": "PRIVILEGED additional native derivative information; independent optimization from shared common",
        }
        from ..core.io import sanitize_persisted_paths

        result = sanitize_persisted_paths(result, folder)
        save(folder / "result.json", result)
        return result
