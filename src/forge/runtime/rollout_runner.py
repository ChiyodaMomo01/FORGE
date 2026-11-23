"""Session execution, public trace/event recording, and window batch scoring."""

from __future__ import annotations

import resource
import time
from pathlib import Path

import numpy as np
import torch

from ..actor_service import BatchedActor
from ..core.model_serde import digest, save
from ..core.sampling import coherent_probe_schedule, smooth_noise, spatial_cost
from ..paired.model import prepare_observation, project_action
from ..sandbox import start_session
from .engine import case_for


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def make_actor_service(
    model,
    mode: str,
    workers: int,
    external_actor: bool,
    connection,
) -> BatchedActor | None:
    if mode == "network":

        def raw(obs):
            if external_actor:
                connection.send({"op": "actor_batch", "observations": obs.numpy()})
                response = connection.recv()
                if response.get("op") != "actor_result":
                    raise RuntimeError("public actor response required")
                return torch.from_numpy(response["values"])
            prepared = prepare_observation(obs, next(model.parameters()).dtype)
            return model.network(prepared)

        return BatchedActor(
            model,
            maximum_requests=workers,
            instantaneous=raw,
        )
    if mode != "pure_physics":
        raise ValueError("invalid full-G actor mode")
    return None


def execute_trial_rollout(
    trial: int,
    *,
    bank,
    split: str,
    recipe,
    model,
    device: str,
    mode: str,
    exploration: str,
    request: dict,
    service: BatchedActor | None,
    sessions,
    out: Path,
) -> dict:
    x, y, ids = bank.cohort(split, [trial])
    case = case_for(bank, split)
    target = y.numpy()
    initial = case.state_from_fields(x, time=ids[0]["start"] * recipe.sample_dt)
    parent, process = (sessions.start if sessions else start_session)(
        case,
        initial,
        recipe.sample_dt,
        recipe.horizon,
        active_model=mode != "pure_physics",
        float32_observations=True,
        full_g=True,
        residual_g=True,
    )
    rng = [
        np.random.default_rng(np.random.SeedSequence([request["seed"], trial, q])) for q in range(4)
    ]
    reservoir_rng = [
        np.random.default_rng(np.random.SeedSequence([request["seed"], trial, q, 991]))
        for q in range(4)
    ]
    joint_rng = np.random.default_rng(np.random.SeedSequence([request["seed"], trial, 772]))
    probe_schedule = None
    probe_interval = None
    interval_innovation = None
    retained = [[] for _ in range(4)]
    counts = np.zeros(4, dtype=int)
    interval = -1
    attempts = 0
    reason = None
    curves = np.full((4, recipe.horizon), 400.0, np.float64)
    uncapped = np.full_like(curves, np.inf)
    costfields = (
        np.full(
            (
                4,
                recipe.horizon,
                model.network.config.nx,
                model.network.config.nx,
            ),
            400.0,
            np.float32,
        )
        if request["capacity"]
        else None
    )
    residuals = (
        np.zeros(
            (
                4,
                recipe.horizon,
                recipe.field_channels,
                model.network.config.nx,
                model.network.config.nx,
            ),
            np.float32,
        )
        if request["residual"]
        else None
    )
    traced = (
        {
            k: []
            for k in (
                "observations",
                "actions",
                "innovations",
                "intervals",
                "fields",
                "output_indices",
                "output_after_sequence",
            )
        }
        if request.get("trace")
        else None
    )
    try:
        while True:
            if not parent.poll(180):
                raise TimeoutError("window native process timeout")
            msg = parent.recv()
            tag = msg[0]
            if tag == "attempt":
                interval = int(msg[1])
                attempts += 4
            elif tag == "observation":
                obs = torch.as_tensor(msg[1], device=device)
                if service is None:
                    raise RuntimeError("pure physics must not request a learned G action")
                mean = service(obs).numpy()
                base_action = project_action(recipe.case_name, torch.from_numpy(mean)).numpy()
                if request.get("coherent"):
                    if probe_schedule is None:
                        probe_schedule = coherent_probe_schedule(
                            rng,
                            request["seed"],
                            recipe.horizon,
                            mean.shape[1:],
                            request["sigma"],
                        )
                    if probe_interval != interval:
                        interval_innovation = probe_schedule[interval]
                        probe_interval = interval
                    raw_innovation = interval_innovation
                elif exploration == "independent":
                    raw_innovation = (
                        float(request["sigma"])
                        * np.stack(
                            [g.standard_normal(size=mean.shape[1:], dtype=np.float64) for g in rng]
                        ).astype(mean.dtype, copy=False)
                        if request["sigma"]
                        else np.zeros_like(mean)
                    )
                else:
                    raw_innovation = (
                        request["sigma"] * np.stack([smooth_noise(g, mean.shape[1:]) for g in rng])
                        if request["sigma"]
                        else np.zeros_like(mean)
                    )
                latent = (mean + raw_innovation).astype(mean.dtype, copy=False)
                exec_innovation = (
                    project_action(recipe.case_name, torch.from_numpy(raw_innovation))
                    .numpy()
                    .astype(np.float32)
                )
                action = (base_action + exec_innovation).astype(np.float32)
                innovation = exec_innovation
                if traced is not None:
                    traced["observations"].append(np.array(msg[1], dtype=np.float32, copy=True))
                    traced["innovations"].append(innovation.copy())
                    traced["actions"].append(action.copy())
                    traced["intervals"].append(interval)
                group_count = int(counts[0])
                cap = request["capacity"]
                group_index = (
                    group_count
                    if group_count < cap
                    else int(joint_rng.integers(group_count + 1))
                    if cap and request.get("joint_actions")
                    else None
                )
                for q in range(4):
                    count = counts[q]
                    counts[q] += 1
                    cap = request["capacity"]
                    if not cap:
                        continue
                    index = (
                        group_index
                        if request.get("joint_actions")
                        else int(count)
                        if count < cap
                        else int(reservoir_rng[q].integers(count + 1))
                    )
                    if index < cap:
                        event = {
                            "obs": msg[1][q].astype(np.float32),
                            "mean": mean[q].copy(),
                            "latent": latent[q].copy(),
                            "action": action[q].copy(),
                            "interval": interval,
                            "sequence": int(count),
                        }
                        if count < cap:
                            retained[q].append(event)
                        else:
                            retained[q][index] = event
                physical = (
                    model.residual_coefficient * action * model.output_scale.detach().cpu().numpy()
                )
                parent.send(("action", physical))
            elif tag == "fields":
                _, k, field = msg
                if not np.isfinite(field).all():
                    reason = "nonfinite public output"
                    parent.send(("stop",))
                    continue
                if traced is not None:
                    traced["fields"].append(field.astype(np.float32))
                    traced["output_indices"].append(int(k))
                    traced["output_after_sequence"].append(len(traced["observations"]) - 1)
                diff = (field - target[:, k]) / np.asarray(bank.info["field_std"]).reshape(
                    1, -1, 1, 1
                )
                uncapped[:, k] = np.square(diff).mean((1, 2, 3))
                curves[:, k] = np.minimum(uncapped[:, k], 200.0)
                if residuals is not None:
                    residuals[:, k] = diff
                if costfields is not None:
                    costfields[:, k] = spatial_cost(field, target[:, k], bank.info["field_std"])
                parent.send(("continue",))
            elif tag == "complete":
                break
            elif tag == "error":
                reason = "native rollout error"
                break
            else:
                raise RuntimeError("unknown public native message")
    except (OSError, EOFError, TimeoutError, RuntimeError) as e:
        reason = type(e).__name__
    finally:
        parent.close()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()

    event_path = None
    trace_path = None
    if traced is not None:
        trace_folder = out / f"trace-{trial:05d}"
        trace_folder.mkdir()
        for key, values in traced.items():
            value = np.stack(values) if values else np.array([], dtype=np.float32)
            np.save(trace_folder / (key + ".npy"), value)
        np.save(trace_folder / "reference.npy", target.astype(np.float32))
        save(
            trace_folder / "trace.json",
            {
                "schema": "public-response-v1",
                "case_name": recipe.case_name,
                "split": split,
                "trial": trial,
                "identities": ids,
                "callback_count": len(traced["observations"]),
                "output_count": len(traced["fields"]),
                "complete": reason is None and len(traced["fields"]) == recipe.horizon,
                "failure": reason,
                "policy_hash": digest(request["theta"]),
                "sigma": request["sigma"],
                "seed": request["seed"],
                "feedback": "policy recomputed on each actual public observation; innovations are independent seeded additive action probes",
                "coherent_interval": request.get("coherent", False),
                "exploration": exploration,
                "innovation_law": (
                    "20x20 orthogonal DCT-II (fit) / DCT-IV (held-out) closed-loop additive action probes (same reset, fixed policy G recomputed on perturbed observations; executed post-projection additive innovation recorded, DC included)"
                    if request.get("coherent")
                    else "fresh per-callback conditionally independent spatial Gaussian; antithetic signs share seed"
                    if exploration == "independent"
                    else "fresh per-callback full-rank low-frequency Gaussian; antithetic signs share seed"
                ),
                "field_std": bank.info["field_std"],
                "array_axes": "callback/output,original four-parent member,channel,y,x",
                "reference_axes": "original four-parent member,output,channel,y,x",
                "source": "observations/actions/fields copied only from original public messages; adjacent callback observations are paired by sequence, no extra observe or hidden state",
                "reference_role": "public HR reference for G task loss only; never an E response label",
                "terminal_observation": "not observed after final action; terminal fields remain available",
                "storage": "float32 public arrays; integer message indices; original score remains float64",
            },
        )
        trace_path = str((trace_folder / "trace.json").resolve())
    if request["capacity"]:
        p = out / f"trial-{trial:05d}"
        p.mkdir()
        all_events = [(q, e) for q in range(4) for e in retained[q]]
        if all_events:
            for key in ("obs", "mean", "latent", "action"):
                np.save(
                    p / (key + ".npy"),
                    np.stack([e[key] for _, e in all_events]),
                )
            np.save(
                p / "window.npy",
                np.asarray([q for q, _ in all_events], np.int64),
            )
            np.save(
                p / "interval.npy",
                np.asarray([e["interval"] for _, e in all_events], np.int64),
            )
            np.save(
                p / "sequence.npy",
                np.asarray([e["sequence"] for _, e in all_events], np.int64),
            )
            np.save(p / "costs.npy", costfields)
            save(
                p / "events.json",
                {
                    "trial": trial,
                    "count": len(all_events),
                    "per_window_total_events": counts,
                    "retained_per_window": [len(z) for z in retained],
                    "sigma": request["sigma"],
                    "exploration": exploration,
                    "joint_actions": request.get("joint_actions", False),
                    "source": "only original public attempt/observation/action/fields messages; no extra observe",
                },
            )
            event_path = str((p / "events.json").resolve())
    return {
        "trial": trial,
        "identities": ids,
        "mean_mse": float(curves.mean()),
        "per_window": curves.mean(1),
        "curves": curves,
        "uncapped": uncapped,
        "reason": reason,
        "events": event_path,
        "trace": trace_path,
        "callback_counts": counts,
        "output_attempts": attempts,
        "residual": residuals,
        "persistent_solver_cpu_seconds": getattr(process, "cpu_seconds", 0.0),
    }


def summarize_rollout_batch(
    rows: list[dict],
    *,
    out: Path,
    split: str,
    trials: list[int],
    mode: str,
    exploration: str,
    request: dict,
    started: float,
    before_cpu: float,
    sessions,
    service: BatchedActor | None,
    device: str,
) -> tuple[dict, dict]:
    curves = np.concatenate([r["curves"] for r in rows])
    np.save(out / "curves.npy", curves.astype(np.float32))
    uncapped = np.concatenate([r["uncapped"] for r in rows])
    np.save(out / "uncapped-curves.npy", uncapped)
    residual_path = None
    if request["residual"] and not any(r["reason"] for r in rows):
        residual_path = str((out / "residual.npy").resolve())
        np.save(residual_path, np.concatenate([r["residual"] for r in rows]))
    cost = {
        "window_attempts": 4 * len(rows),
        "output_attempts": sum(r["output_attempts"] for r in rows),
        "actual_component_window_calls": int(sum(r["callback_counts"].sum() for r in rows)),
        "wall_seconds": time.monotonic() - started,
        "cpu_seconds": cpu_seconds()
        - before_cpu
        + sum(r["persistent_solver_cpu_seconds"] for r in rows),
        "persistent_solver_cpu_seconds": sum(r["persistent_solver_cpu_seconds"] for r in rows),
        "persistent_solver_pids": [p.pid for _, p in sessions.slots if p.is_alive()]
        if sessions
        else [],
        "bytes_written": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        "actor_service": service.cost if service else None,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else None,
    }
    result = {
        "mean_mse": float(curves.mean()),
        "uncapped_mean_mse": float(uncapped.mean()) if np.isfinite(uncapped).all() else None,
        "uncapped_nonfinite_outputs": int((~np.isfinite(uncapped)).sum()),
        "per_window": curves.mean(1),
        "groups": [
            {
                "trial": r["trial"],
                "identities": r["identities"],
                "mean_mse": r["mean_mse"],
            }
            for r in rows
        ],
        "failures": [{"trial": r["trial"], "reason": r["reason"]} for r in rows if r["reason"]],
        "event_paths": [r["events"] for r in rows if r["events"]],
        "curves_path": str((out / "curves.npy").resolve()),
        "residual_path": residual_path,
        "trace_paths": [r["trace"] for r in rows if r["trace"]],
        "split": split,
        "trials": trials,
        "mode": mode,
        "label": request["label"],
        "policy_hash": digest(request["theta"]),
        "sigma": request["sigma"],
        "exploration": exploration,
        "seed": request["seed"],
        "cost": cost,
    }
    return result, cost
