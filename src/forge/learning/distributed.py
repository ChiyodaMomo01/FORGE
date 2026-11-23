"""Persistent one-process/one-device execution for public-data computation.

Rank zero controls the frozen algorithm. Other ranks execute bounded collective
jobs, not independent experiments. Native capabilities remain in CPU services.
"""

from __future__ import annotations

import gc
import os
import resource
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel

from .common import save
from .events import close_record_cache

_engine: Engine | None = None
_default_handlers: dict[str, Callable[[Engine, Any], Any]] = {}


def set_handlers(handlers: dict[str, Callable[[Engine, Any], Any]]) -> None:
    global _default_handlers
    _default_handlers = handlers


def engine() -> Engine | None:
    return _engine


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def wrap(model):
    return DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()] if next(model.parameters()).is_cuda else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def full_prediction(model, x):
    """Shard examples, preserve joint PPO objectives and their exact weighting."""
    count = len(x)
    width = (count + size() - 1) // size()
    indices = torch.arange(rank() * width, (rank() + 1) * width, device=x.device)
    local = x[indices.clamp(max=count - 1)]
    prediction = model(local)
    return torch.cat(all_gather(prediction), dim=0)[:count]


class Engine:
    def __init__(
        self,
        folder: str | Path,
        backend: str | None = None,
        handlers: dict[str, Callable[[Engine, Any], Any]] | None = None,
    ):
        global _engine
        self.device = (
            torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend or ("nccl" if self.device.type == "cuda" else "gloo"),
            timeout=timedelta(minutes=30),
        )
        self.control = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
        self.folder = Path(folder)
        self.handlers: dict[str, Callable[[Engine, Any], Any]] = (
            handlers if handlers is not None else _default_handlers
        )
        self.state: dict[str, Any] = {}
        self.timings: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self.cpu_started = resource.getrusage(resource.RUSAGE_SELF)
        _engine = self
        save(
            self.folder / f"rank-{rank()}-identity.json",
            {
                "rank": rank(),
                "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
                "world_size": size(),
                "pid": os.getpid(),
                "device": str(self.device),
                "backend": dist.get_backend(),
                "started_unix": time.time(),
            },
        )

    def broadcast(self, value):
        values = [value if rank() == 0 else None]
        dist.broadcast_object_list(values, src=0, group=self.control)
        return values[0]

    def gather(self, value):
        values = [None] * size() if rank() == 0 else None
        dist.gather_object(value, values, dst=0, group=self.control)
        return values

    def call(self, operation: str, payload: Any = None):
        if rank() != 0:
            raise RuntimeError("only coordinator may dispatch collective work")
        self.broadcast((operation, payload))
        return self.execute(operation, payload)

    def execute(self, operation: str, payload: Any):
        start = time.monotonic()
        save(
            self.folder / f"rank-{rank()}-phase.json",
            {
                "operation": operation,
                "started_unix": time.time(),
                "completed_jobs": len(self.timings),
            },
        )
        try:
            result = self.handlers[operation](self, payload)
        finally:
            if operation in (
                "response",
                "project",
                "ppo",
                "closure_rl",
                "emulator_step",
                "residual",
            ):
                close_record_cache()
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        self.timings.append({"operation": operation, "seconds": time.monotonic() - start})
        return result

    def serve(self):
        while True:
            operation, payload = self.broadcast(None)
            if operation == "close":
                break
            self.execute(operation, payload)

    def close(self, notify: bool = True):
        global _engine
        if rank() == 0 and notify:
            self.broadcast(("close", None))
        for value in self.state.values():
            if hasattr(value, "close"):
                value.close()
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        save(
            self.folder / f"rank-{rank()}-resources.json",
            {
                "rank": rank(),
                "world_size": size(),
                "pid": os.getpid(),
                "device": str(self.device),
                "backend": dist.get_backend(),
                "wall_seconds": time.monotonic() - self.started,
                "cpu_seconds": cpu.ru_utime
                + cpu.ru_stime
                - self.cpu_started.ru_utime
                - self.cpu_started.ru_stime,
                "jobs": self.timings,
                "scope": "rank CPU is a subset of worker inclusive cost; GPU only this rank's device",
            },
        )
        _engine = None
        dist.destroy_process_group()
