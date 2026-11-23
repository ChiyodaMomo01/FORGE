"""Only public recorded observations, Gaussian actions and H output costs."""

import errno
import json
import threading
from collections import OrderedDict

import numpy as np

from ..core.io import load_array, resolve_archived_path

_MAX_MAPPINGS = 256
_MAPPINGS = OrderedDict()
_MAPPING_LOCK = threading.RLock()


def close_record_cache():
    with _MAPPING_LOCK:
        for value in _MAPPINGS.values():
            getattr(value, "_mmap", None) is not None and value._mmap.close()
        _MAPPINGS.clear()


class RecordedArray:
    """Read-only recorded field; slices own their data after cache eviction."""

    def __init__(self, path):
        self.path = resolve_archived_path(path)

    def __getitem__(self, index):
        key = str(self.path)
        with _MAPPING_LOCK:
            value = _MAPPINGS.pop(key, None)
            if value is None:
                while len(_MAPPINGS) >= _MAX_MAPPINGS:
                    _, old = _MAPPINGS.popitem(last=False)
                    getattr(old, "_mmap", None) is not None and old._mmap.close()
                while True:
                    try:
                        value = load_array(self.path, mmap_mode="r", allow_pickle=False)
                        break
                    except OSError as error:
                        # The process may already own IPC/CUDA/test descriptors.
                        # A fixed map count alone cannot bound its remaining
                        # descriptor budget. Evict only our own mappings.
                        if error.errno != errno.EMFILE or not _MAPPINGS:
                            raise
                        _, old = _MAPPINGS.popitem(last=False)
                        getattr(old, "_mmap", None) is not None and old._mmap.close()
            _MAPPINGS[key] = value
            return np.array(value[index], copy=True)

    def __array__(self, dtype=None, copy=None):
        # Full-array consumers (e.g. PPO returns) receive owned storage too.
        value = load_array(self.path, allow_pickle=False)
        return value.astype(dtype, copy=False) if dtype is not None else value


def records(score):
    out = []
    for path in score.get("event_paths", []):
        resolved = resolve_archived_path(path)
        p = resolved.parent
        meta = json.loads(resolved.read_text())
        z = {
            k: RecordedArray(p / (k + ".npy")) for k in ("obs", "mean", "latent", "action", "costs")
        }
        z.update(
            {k: load_array(p / (k + ".npy"), allow_pickle=False) for k in ("window", "interval")}
        )
        if (p / "sequence.npy").exists():
            z["sequence"] = load_array(p / "sequence.npy", allow_pickle=False)
        out.append((meta, z))
    return out


def anchors(score, count=16):
    obs = []
    for _meta, z in records(score):
        for q in range(4):
            ids = np.flatnonzero(z["window"] == q)
            if len(ids):
                obs.append(np.asarray(z["obs"][ids[0]]).copy())
    if not obs:
        raise RuntimeError("no public event anchors")
    ix = np.linspace(0, len(obs) - 1, min(count, len(obs)), dtype=int)
    return np.asarray([obs[i] for i in ix], np.float32)
