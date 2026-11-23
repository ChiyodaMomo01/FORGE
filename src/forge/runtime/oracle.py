"""Trusted numerical service: public stage I/O for all four 2D cases."""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch

from ..core.contracts import configure
from ..core.model_serde import actor, load_model, save, set_model
from ..sandbox import SessionPool
from .data import WindowBank
from .protocol import build_handshake_info, validate_rollout_request
from .rollout_runner import (
    cpu_seconds,
    execute_trial_rollout,
    make_actor_service,
    summarize_rollout_batch,
)


def serve(connection, root, folder, device, recipe, checkpoint_path):
    external_actor = device == "external"
    device = "cpu" if external_actor else device
    sessions = None
    torch.set_num_threads(1)
    actor_numeric = configure(recipe)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        raw_mem_frac = os.environ.get("FORGE_CUDA_MEMORY_FRACTION", "").strip()
        if raw_mem_frac:
            torch.cuda.set_per_process_memory_fraction(float(raw_mem_frac))
    folder = Path(folder) / "oracle"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        bank = WindowBank(root, recipe)
        model = (
            load_model(checkpoint_path, device)[0]
            if checkpoint_path
            else actor(
                n=bank.info["splits"]["train"]["grid"],
                device=device,
                case_name=recipe.case_name,
            )
        )
        connection.send(build_handshake_info(model, bank, recipe, checkpoint_path, actor_numeric))
        if getattr(recipe, "actor_fp64", False):
            model = model.double()
        serial = max([int(x.name.split("-")[1]) for x in folder.glob("query-*")] + [0])
        while True:
            request = connection.recv()
            if request["op"] == "close":
                break
            split, trials, exploration = validate_rollout_request(request, bank, root, recipe)
            serial += 1
            out = folder / f"query-{serial:06d}"
            out.mkdir()
            set_model(model, request["theta"])
            mode = request["mode"]
            if external_actor and sessions is None:
                sessions = SessionPool()
            workers = min(sessions.capacity if sessions else 8, len(trials))
            started = time.monotonic()
            before_cpu = cpu_seconds()
            service = make_actor_service(model, mode, workers, external_actor, connection)
            runner = partial(
                execute_trial_rollout,
                bank=bank,
                split=split,
                recipe=recipe,
                model=model,
                device=device,
                mode=mode,
                exploration=exploration,
                request=request,
                service=service,
                sessions=sessions,
                out=out,
            )
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(runner, trials))
            finally:
                if service:
                    service.close()
            result, cost = summarize_rollout_batch(
                rows,
                out=out,
                split=split,
                trials=trials,
                mode=mode,
                exploration=exploration,
                request=request,
                started=started,
                before_cpu=before_cpu,
                sessions=sessions,
                service=service,
                device=device,
            )
            save(out / "public.json", result)
            public_payload = json.loads((out / "public.json").read_text(encoding="utf-8"))
            with (folder / "ledger.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(public_payload) + "\n")
            connection.send(
                {"sealed_manifest": str((out / "public.json").resolve()), "cost": cost}
                if request.get("sealed")
                else public_payload
            )
    except BaseException as e:
        traceback.print_exc()
        try:
            connection.send({"error": type(e).__name__ + ": trusted oracle failed; see worker log"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if sessions is not None:
            sessions.close()
        connection.close()
