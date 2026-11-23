"""Preparation, regression, frozen-reference evaluation and post-training stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .core.io import file_digest, sanitize_persisted_paths, save
from .schemas import load_residual_statistics


def execute(args, recipe):
    root = Path(args.output)
    reference = Path(args.reference) if args.reference else root
    if args.reference and args.method not in (
        "ES",
        "EKI",
        "PPO_global",
        "PPO_local",
        "emulator",
        "closure_rl",
        "main",
        "AD",
    ):
        raise ValueError("--reference is only supported for post-training methods")
    if args.method == "prepare":
        from .collect.audit import audit
        from .runtime.data import prepare

        existing = Path(args.pairs) / recipe.case_name / "restriction-audit.json"
        if (
            not existing.exists()
            or json.loads(existing.read_text(encoding="utf-8")).get("status") != "passed"
        ):
            audit(args.pairs, recipe.case_name, args.original)
        return prepare(args.original, args.pairs, root, recipe)
    if args.method == "physics_start":
        import torch

        from .core.model_serde import array, checkpoint, digest
        from .paired.data import statistics
        from .paired.model import FullComponent

        stats = load_residual_statistics(
            Path(args.pairs) / recipe.case_name, case_name=recipe.case_name
        )
        if stats is None:
            stats = statistics(args.pairs, recipe.case_name, root / "statistics.json")
        torch.manual_seed(recipe.seed)
        model = FullComponent(
            recipe.case_name,
            n=json.loads(
                (Path(args.pairs) / recipe.case_name / "dataset.json").read_text(encoding="utf-8")
            )["lr_grid"],
            statistics=stats,
        ).eval()
        if (root / "common.pt").exists():
            raise FileExistsError("common already frozen")
        checkpoint(
            root / "common.pt",
            model,
            {
                "method": "physics_start",
                "statistics": stats,
                "supervised_updates": 0,
                "seed": recipe.seed,
            },
        )
        record = {
            "status": "complete",
            "case": recipe.case_name,
            "sha256": file_digest(root / "common.pt"),
            "policy_hash": digest(array(model)),
            "source": "random trunk and exact zero residual head",
            "initialization": "physics_start",
            "supervised_updates": 0,
            "statistics": stats,
            "frozen": True,
        }
        save(root / "common.json", record)
        return record
    if args.method == "pretrain":
        from .learning.distributed import engine

        payload = {
            "root": str(args.pairs),
            "output": str(root / "pretrain"),
            "case_name": recipe.case_name,
            "device": args.device,
            "epochs": recipe.pretrain_epochs,
            "patience": recipe.pretrain_patience,
        }
        if engine() is not None:
            return engine().call("pretrain", payload)
        from .paired.pretrain import train

        return train(**payload)
    if args.method == "freeze_common":
        from .core.model_serde import array, digest, load_model

        result = json.loads((root / "pretrain/result.json").read_text(encoding="utf-8"))
        if (
            result["status"] != "complete"
            or result["native_calls"]
            or result["native_backward_calls"]
        ):
            raise ValueError("pure paired regression provenance required")
        if result["case"] != recipe.case_name:
            raise ValueError("cross-case common rejected")
        source = root / "pretrain/selected.pt"
        if file_digest(source) != result["selected_sha256"]:
            raise ValueError("selected weight changed")
        if (root / "common.pt").exists():
            raise FileExistsError("common already frozen")
        shutil.copy2(source, root / "common.pt")
        model, meta = load_model(root / "common.pt")
        record = {
            "status": "complete",
            "case": recipe.case_name,
            "sha256": file_digest(root / "common.pt"),
            "policy_hash": digest(array(model)),
            "source": "pretrain/selected.pt",
            "frozen": True,
            "statistics": meta["statistics"],
        }
        save(root / "common.json", record)
        return record
    if args.method == "validate_gate":
        from .execution import validation_gate

        common_res = json.loads((root / "common/result.json").read_text(encoding="utf-8"))[
            "evaluations"
        ]["validation"]
        phys_res = json.loads((root / "pure_physics/result.json").read_text(encoding="utf-8"))[
            "evaluations"
        ]["validation"]
        common_meta = json.loads((root / "common.json").read_text(encoding="utf-8"))
        actual_common_sha = file_digest(root / "common.pt")
        if common_meta.get("sha256") != actual_common_sha:
            raise ValueError("frozen common changed")
        init_mode = common_meta.get(
            "initialization",
            "physics_start" if common_meta.get("method") == "physics_start" else "paired_pretrain",
        )
        gate = validation_gate(
            common_res,
            phys_res,
            actual_common_sha,
            case_name=recipe.case_name,
            initialization=init_mode,
        )
        save(root / "validation-gate.json", gate)
        if gate.get("status") != "passed" or not gate.get("post_training_authorized_by_gate"):
            raise ValueError(f"validation gate did not pass for {recipe.case_name}: {gate}")
        return gate
    if args.method not in ("common", "pure_physics"):
        gate = json.loads((reference / "validation-gate.json").read_text(encoding="utf-8"))
        if (
            gate.get("status") != "passed"
            or not gate.get("post_training_authorized_by_gate")
            or gate.get("common_sha256") != file_digest(reference / "common.pt")
        ):
            raise ValueError("post-training denied until complete common validation gate passes")
    if args.method == "AD":
        from .learning.distributed import engine

        ad_folder = (
            getattr(args, "resolved_folder_name", None)
            or getattr(args, "folder_name", None)
            or (f"AD-{args.ad_tag}" if getattr(args, "ad_tag", None) else "AD")
        )
        options = {"reference": str(reference)} if args.reference else {}
        if engine() is not None:
            return engine().call(
                "ad",
                {
                    "output": str(root),
                    "recipe": recipe,
                    "folder_name": ad_folder,
                    **options,
                },
            )
        from .privileged.trajectory import train

        return train(root, args.device, True, recipe, folder_name=ad_folder, **options)
    from .core.model_serde import array, digest, load_model
    from .learning.client import ShardedClient as Client

    model, _ = load_model(reference / "common.pt", args.device)
    if model.case_name != recipe.case_name:
        raise ValueError("cross-case common rejected")
    if (
        file_digest(reference / "common.pt")
        != json.loads((reference / "common.json").read_text(encoding="utf-8"))["sha256"]
    ):
        raise ValueError("frozen common changed")
    folder = root / getattr(args, "resolved_folder_name", args.method)
    with Client(
        reference,
        folder,
        args.device,
        recipe,
        checkpoint_path=reference / "common.pt",
    ) as client:
        if args.method in ("pure_physics", "common"):
            mode = "pure_physics" if args.method == "pure_physics" else "network"
            scores = {
                s: client.score(array(model), s, None, mode=mode, label=args.method + "-" + s)
                for s in ("train", "validation")
            }
            result = {
                "status": "complete",
                "method": args.method,
                "policy_hash": digest(array(model)),
                "evaluations": {"validation": scores["validation"]},
                "training": scores["train"],
                "information": "original LR true G, no neural replacement"
                if mode == "pure_physics"
                else "shared common policy; initialization in common.json",
                "test_id_deferred": True,
            }
            result = sanitize_persisted_paths(result, folder)
            save(folder / "result.json", result)
            return result
        from .learning.study import Study

        study = Study(
            client,
            folder,
            args.method,
            args.device,
            recipe,
            reference=reference if args.reference else None,
        )
        if args.method in ("ES", "EKI"):
            from .learning.value_methods import run_ensemble

            return run_ensemble(study)
        if args.method in ("PPO_global", "PPO_local"):
            from .learning.ppo_joint import run

            return run(study)
        if args.method == "closure_rl":
            from .learning.closure_rl import run

            return run(study)
        if args.method == "emulator":
            from .learning.emulator import run

            return run(study)
        if args.method == "main":
            from .learning.function_main import run

            return run(study, resume_from=getattr(args, "resume_from_attempt", None))
    raise ValueError("unsupported training method")
