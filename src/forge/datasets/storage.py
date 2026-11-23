"""Dataset checkpoint and publication path utilities."""

import json
from pathlib import Path

import torch

from ..config import save_json


def save_state(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def paths(root, task):
    key = Path(task["case"]) / task["split"] / task["parent"]
    return (
        root / key.parent / (key.name + ".json"),
        root / ".work" / key,
    )


def halt_path(root, task):
    """Return the per-case halt marker path for reference production."""
    revision = "-r" + str(task["reference_revision"]) if "reference_revision" in task else ""
    return root / ("HALT-" + task["case"] + revision + ".json")


def complete(root, task):
    meta, _ = paths(root, task)
    return (
        meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("status") == "complete"
    )


def initialize_manifest(root, campaign):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != campaign:
            raise ValueError("existing dataset belongs to a different campaign")
    else:
        save_json(path, campaign)
    return True
