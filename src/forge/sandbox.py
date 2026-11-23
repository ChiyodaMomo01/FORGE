"""A real CPU process boundary: only observations/actions/physical fields cross.

The native process never receives network parameters or exposes AD operations.
The matching differentiable engine is a separate, additional-information track.
"""

import multiprocessing as mp
import os
import queue
import resource

import torch

from .cases import make_case
from .state import advance, to_device


def _worker(
    pipe,
    name,
    config,
    state,
    dt,
    steps,
    active_model,
    interval_observation=False,
    float32_observations=False,
    full_g=False,
    residual_g=False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    case = make_case(name, {**config, "differentiable": False})
    state = to_device(state, "cpu", torch.float64)

    def component(obs):
        # Only an explicitly float32 Component may use the smaller transport.
        # Classical closures and double-precision actors retain float64 input.
        value = obs.detach().to(dtype=torch.float32) if float32_observations else obs.detach()
        pipe.send(("observation", value.cpu().numpy()))
        tag, value = pipe.recv()
        if tag != "action":
            raise RuntimeError("CPU physics accepts detached action messages only")
        correction = torch.as_tensor(value, dtype=obs.dtype)
        if residual_g:
            from .paired.coarse import coarse_value

            return coarse_value(name, obs) + correction
        return correction

    if full_g:
        component.output_semantics = "full-g-v1"

    try:
        with torch.inference_mode():
            for t in range(steps):
                start_observation = case.observe(state).detach() if interval_observation else None
                if start_observation is not None and float32_observations:
                    start_observation = start_observation.float()
                pipe.send(
                    (
                        "attempt",
                        t,
                        start_observation.numpy() if start_observation is not None else None,
                    )
                )
                state = advance(case, state, dt, component if active_model else None)
                pipe.send(("fields", t, case.fields(state).detach().numpy()))
                response = pipe.recv()
                if response[0] == "stop":
                    break
                if response[0] != "continue":
                    raise RuntimeError("invalid native continuation message")
            pipe.send(("complete",))
    except BaseException as e:
        pipe.send(("error", type(e).__name__, str(e)))
    finally:
        pipe.close()


def _configuration(case):
    from dataclasses import asdict, is_dataclass

    return asdict(case.config) if is_dataclass(case.config) else dict(case.config)


def start_session(case, state, dt, steps, active_model=True, **options):
    """Oracle-side adapter: exact execution configuration never forms training features."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(
        target=_worker,
        args=(
            child,
            case.name,
            _configuration(case),
            to_device(state, "cpu"),
            dt,
            steps,
            active_model,
        ),
        kwargs=options,
    )
    proc.start()
    child.close()
    return parent, proc


def _session_loop(control):
    """Import torch once; construct a fresh native case/state for every reset."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    while True:
        request = control.recv()
        if request is None:
            break
        pipe, args, options = request
        before = resource.getrusage(resource.RUSAGE_SELF)
        _worker(pipe, *args, **options)
        after = resource.getrusage(resource.RUSAGE_SELF)
        control.send(
            (
                "finished",
                after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            )
        )
    control.close()


class _Lease:
    def __init__(self, pool, slot):
        self.pool, self.slot, self.active = pool, slot, True
        self.cpu_seconds = 0.0

    def join(self, timeout=None):
        if self.active and self.slot[0].poll(timeout):
            result = self.slot[0].recv()
            if result[0] != "finished":
                raise RuntimeError("invalid native worker completion")
            self.cpu_seconds = result[1]
            self.active = False
            self.pool.available.put(self.slot)

    def is_alive(self):
        return self.active

    def terminate(self):
        self.slot[1].terminate()
        self.slot[1].join()
        self.active = False
        self.pool.available.put(self.pool.spawn())


class SessionPool:
    """Persistent isolated CPU solvers; each lease is still a fresh public reset."""

    def __init__(self, capacity=5):
        self.capacity = capacity
        self.context = mp.get_context("spawn")
        self.available = queue.Queue()
        self.slots = []
        for _ in range(capacity):
            self.available.put(self.spawn())

    def spawn(self):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_session_loop, args=(child,))
        process.start()
        child.close()
        slot = parent, process
        self.slots.append(slot)
        return slot

    def start(self, case, state, dt, steps, active_model=True, **options):
        slot = self.available.get()
        parent, child = self.context.Pipe()
        slot[0].send(
            (
                child,
                (
                    case.name,
                    _configuration(case),
                    to_device(state, "cpu"),
                    dt,
                    steps,
                    active_model,
                ),
                options,
            )
        )
        child.close()
        return parent, _Lease(self, slot)

    def close(self):
        for pipe, process in self.slots:
            if process.is_alive():
                try:
                    pipe.send(None)
                except (BrokenPipeError, EOFError):
                    pass
        for pipe, process in self.slots:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()
