"""Public numerical transport: no physical operators, data or private state."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..paired.model import prepare_observation
from .common import digest, set_model
from .distributed import engine
from .numeric import configure
from .schema import ACTIVE_SPLITS, RECIPE_BY_CASE


def resolve_recipe(root, recipe=None):
    """Infer omitted recipes from public manifests; explicit recipes take precedence."""
    if recipe is not None:
        return recipe
    for filename, key in (
        ("window-manifest.json", "case_name"),
        ("common.json", "case"),
    ):
        path = Path(root) / filename
        if path.is_file():
            case = json.loads(path.read_text(encoding="utf-8")).get(key)
            if case in RECIPE_BY_CASE:
                return RECIPE_BY_CASE[case]
    raise ValueError("a supported case recipe or public case manifest is required")


class Client:
    def __init__(
        self,
        root,
        folder,
        device="cpu",
        recipe=None,
        checkpoint_path=None,
        external_model=None,
    ):
        from ..runtime.bootstrap import serve

        recipe = resolve_recipe(root, recipe)
        self.actor_numeric = configure(recipe)
        if external_model is not None and getattr(recipe, "actor_fp64", False):
            external_model = external_model.double()
        self.external_model = external_model
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.server = context.Process(
            target=serve,
            args=(
                child,
                str(root),
                str(folder),
                "external" if external_model is not None else device,
                recipe,
                str(checkpoint_path) if checkpoint_path else None,
            ),
        )
        self.server.start()
        child.close()
        self.cache = {}
        try:
            if not self.connection.poll(180):
                raise TimeoutError("public service startup timed out")
            self.info = self.connection.recv()
            if "error" in self.info:
                raise RuntimeError(self.info["error"])
        except BaseException:
            self.close()
            raise

    def score(
        self,
        theta,
        split="train",
        trials=None,
        *,
        capacity=0,
        sigma=0.0,
        seed=0,
        label="score",
        residual=False,
        mode="network",
        sealed=False,
        joint_actions=False,
        trace=False,
        coherent=False,
        exploration="correlated",
    ):
        if split not in ACTIVE_SPLITS:
            raise ValueError("only train/validation/frozen Test ID supported")
        if exploration not in ("correlated", "independent"):
            raise ValueError("unsupported public exploration law")
        if split != "train" and (
            capacity or sigma or coherent or residual or exploration != "correlated"
        ):
            raise ValueError("heldouts are score-only deterministic")
        if trace and (split != "train" or mode != "network" or sealed):
            raise ValueError("continuous public traces are train network only")
        key = (
            digest(theta),
            split,
            tuple(trials) if trials is not None else None,
            mode,
        )
        cacheable = not any((trace, residual, sealed, capacity, sigma))
        if cacheable and key in self.cache:
            original = self.cache[key]
            return {
                **original,
                "cache_hit": True,
                "original_cost": original["cost"],
                "cost": {
                    "window_attempts": 0,
                    "wall_seconds": 0.0,
                    "cpu_seconds": 0.0,
                    "bytes_written": 0,
                },
            }
        if self.external_model is not None:
            set_model(self.external_model, theta)
        self.connection.send(
            {
                "op": "rollout",
                "theta": np.asarray(theta, np.float32),
                "split": split,
                "trials": trials,
                "capacity": capacity,
                "sigma": sigma,
                "seed": seed,
                "label": label,
                "residual": residual,
                "mode": mode,
                "sealed": sealed,
                "joint_actions": joint_actions,
                "trace": trace,
                "coherent": coherent,
                "exploration": exploration,
            }
        )
        result = self.connection.recv()
        while result.get("op") == "actor_batch":
            param = next(self.external_model.parameters())
            device, dtype = param.device, param.dtype
            with torch.no_grad():
                obs = torch.as_tensor(result["observations"], device=device)
                values = self.external_model.network(prepare_observation(obs, dtype))
            self.connection.send({"op": "actor_result", "values": values.cpu().numpy()})
            result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        if cacheable:
            self.cache[key] = result
        return result

    def close(self):
        if self.connection:
            try:
                self.connection.send({"op": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.connection.close()
            self.connection = None
        self.server.join(15)
        if self.server.is_alive():
            self.server.terminate()
            self.server.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ShardedClient:
    """One shared policy, public task shards across the leased GPUs."""

    def __new__(cls, *args, **kwargs):
        if engine() is not None:
            return DistributedClient(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        root,
        folder,
        device="cpu",
        recipe=None,
        checkpoint_path=None,
        *,
        world_size: int | None = None,
    ):
        recipe = resolve_recipe(root, recipe)
        self.clients = []
        resolved_world = (
            int(world_size)
            if world_size is not None
            else int(os.environ.get("FORGE_DDP_WORLD_SIZE", "1"))
        )
        count = resolved_world if device.startswith("cuda") else 1
        try:
            for shard_rank in range(count):
                self.clients.append(
                    Client(
                        root,
                        Path(folder) / f"gpu-{shard_rank}",
                        f"cuda:{shard_rank}" if device.startswith("cuda") else device,
                        recipe,
                        checkpoint_path,
                    )
                )
        except BaseException:
            self.close()
            raise
        self.info = self.clients[0].info
        if any(c.info["initial_hash"] != self.info["initial_hash"] for c in self.clients):
            raise ValueError("shared center mismatch")
        self.pool = ThreadPoolExecutor(max_workers=count)

    def score(self, theta, split="train", trials=None, **kwargs):
        selected = list(range(self.info["trials"][split])) if trials is None else list(trials)
        tasks = [(c, selected[i :: len(self.clients)]) for i, c in enumerate(self.clients)]
        futures = [
            self.pool.submit(c.score, theta, split, part, **kwargs) for c, part in tasks if part
        ]
        rows = [f.result() for f in futures]
        return merge_scores(rows, selected, kwargs)

    def close(self):
        if hasattr(self, "pool"):
            self.pool.shutdown(wait=True)
        for client in self.clients:
            client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def merge_scores(rows, selected, kwargs):
    per_trial = {
        trial: x["per_window"][4 * j : 4 * j + 4]
        for x in rows
        for j, trial in enumerate(x["trials"])
    }
    per_trace = {
        trial: x["trace_paths"][j]
        for x in rows
        if len(x.get("trace_paths", ())) == len(x["trials"])
        for j, trial in enumerate(x["trials"])
    }
    costs = {
        key: sum(x["cost"].get(key, 0) for x in rows)
        for key in (
            "window_attempts",
            "output_attempts",
            "actual_component_window_calls",
            "bytes_written",
            "cpu_seconds",
        )
    }
    if rows and rows[0].get("policy_hash") is not None:
        p_hash = rows[0]["policy_hash"]
        if any(x.get("policy_hash") != p_hash for x in rows):
            raise ValueError("shard policy_hash mismatch across workers")
        parity = {
            "verified": True,
            "ranks": len(rows),
            "policy_hash": p_hash,
        }
    else:
        parity = None
    return {
        **rows[0],
        "trials": selected,
        "policy_hash_parity": parity,
        "mean_mse": float(
            np.average(
                [x["mean_mse"] for x in rows],
                weights=[len(x["trials"]) for x in rows],
            )
        ),
        "uncapped_mean_mse": float(
            np.average(
                [x["uncapped_mean_mse"] for x in rows],
                weights=[len(x["trials"]) for x in rows],
            )
        )
        if all(x.get("uncapped_mean_mse") is not None for x in rows)
        else None,
        "per_window": [value for trial in selected for value in per_trial[trial]],
        "cache_hit": all(x.get("cache_hit", False) for x in rows),
        "groups": sorted([g for x in rows for g in x["groups"]], key=lambda x: x["trial"]),
        "failures": [f for x in rows for f in x["failures"]],
        "event_paths": [p for x in rows for p in x["event_paths"]],
        "trace_paths": (
            [per_trace[trial] for trial in selected]
            if len(per_trace) == len(selected)
            else [p for x in rows for p in x["trace_paths"]]
        ),
        "residual_path": None,
        "residual_paths": [x["residual_path"] for x in rows] if kwargs.get("residual") else [],
        "uncapped_nonfinite_outputs": sum(x["uncapped_nonfinite_outputs"] for x in rows),
        "shards": rows,
        "cost": {
            **costs,
            "wall_seconds": max(x["cost"]["wall_seconds"] for x in rows),
            "scope": "disjoint service processes; inclusive stage CPU is authoritative, do not add query subsets",
        },
    }


class DistributedClient:
    """One public service per rank; coordinator sees only public records."""

    def __init__(
        self,
        root,
        folder,
        device="cpu",
        recipe=None,
        checkpoint_path=None,
        *,
        world_size: int | None = None,
    ):
        del device, world_size
        recipe = resolve_recipe(root, recipe)
        self.engine = engine()
        assert self.engine is not None
        self.info = self.engine.call(
            "setup",
            {
                "root": str(root),
                "folder": str(folder),
                "recipe": recipe,
                "checkpoint": str(checkpoint_path or Path(root) / "common.pt"),
            },
        )

    def score(self, theta, split="train", trials=None, **kwargs):
        return self.engine.call(
            "query",
            {"theta": theta, "split": split, "trials": trials, "kwargs": kwargs},
        )

    def close(self):
        self.engine.call("close_client", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *args):
        if exc_type is None:
            self.close()
