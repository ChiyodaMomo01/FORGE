"""Microbatch independent native requests for a stateless, read-only actor."""

import queue
import threading
import time
from concurrent.futures import Future

import torch


class BatchedActor:
    """Stateless FNO actors preserve per-example semantics when requests are joined.

    This service never combines physical states or time-step decisions. Each
    native CPU process still advances its original batch of four parents.
    Stateful sessions may share this service only for their instantaneous
    network evaluations; their memory is never combined here.
    """

    def __init__(
        self,
        model,
        maximum_requests=8,
        gather_seconds=0.001,
        instantaneous=None,
    ):
        if model.training:
            raise ValueError("native batching requires an eval-mode actor")
        self.model = model
        self.instantaneous = model if instantaneous is None else instantaneous
        self.maximum_requests = maximum_requests
        self.gather_seconds = gather_seconds
        self.requests = queue.Queue()
        self.cost = {
            "actor_forward_batches": 0,
            "actor_observation_fields": 0,
            "actor_service_seconds": 0.0,
        }
        self.thread = threading.Thread(target=self._serve, name="native-actor-batches", daemon=True)
        self.thread.start()

    def __call__(self, observation):
        result = Future()
        self.requests.put((observation, result))
        return result.result()

    def _serve(self):
        while True:
            first = self.requests.get()
            if first is None:
                return
            items = [first]
            deadline = time.monotonic() + self.gather_seconds
            while len(items) < self.maximum_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    raise RuntimeError("close actor service only after native requests finish")
                items.append(item)
            started = time.monotonic()
            try:
                counts = [x.shape[0] for x, _ in items]
                with torch.no_grad():
                    value = self.instantaneous(torch.cat([x for x, _ in items]))
                self.cost["actor_forward_batches"] += 1
                self.cost["actor_observation_fields"] += sum(counts)
                # Transfer once for the combined batch. Returning CPU tensors
                # avoids one CUDA synchronization/transfer per RPC consumer.
                values = value.detach().cpu().split(counts)
                for (_, result), output in zip(items, values, strict=True):
                    result.set_result(output)
            except BaseException as error:
                for _, result in items:
                    result.set_exception(error)
            finally:
                self.cost["actor_service_seconds"] += time.monotonic() - started

    def close(self):
        self.requests.put(None)
        self.thread.join()
