"""Common 2D ablation flags and recipe mapping."""

from dataclasses import replace


def add_arguments(parser):
    parser.add_argument("--enable-stage-one", action="store_true")
    parser.add_argument("--stage-one-only", action="store_true")
    parser.add_argument("--stage-two-only", action="store_true")
    parser.add_argument("--stage-two-no-modulation", action="store_true")
    parser.add_argument("--stage-two-one-step-only", action="store_true")
    parser.add_argument(
        "--projection-l2-only",
        "--stage-two-l2-projection",
        dest="projection_l2_only",
        action="store_true",
    )
    parser.add_argument(
        "--projection-head-only",
        "--stage-two-head-only",
        dest="projection_head_only",
        action="store_true",
    )


def apply_arguments(recipe, args):
    flags = {
        "disable_stage_two": args.stage_one_only,
        "disable_stage_one": args.stage_two_only,
        "stage2_no_modulation": args.stage_two_no_modulation,
        "stage2_one_step_only": args.stage_two_one_step_only,
        "stage2_l2_projection": args.projection_l2_only,
        "stage2_head_only": args.projection_head_only,
    }
    enable_s1 = bool(getattr(args, "enable_stage_one", False))
    if (any(flags.values()) or enable_s1) and args.method != "main":
        raise ValueError("Stage I/II ablation flags are only valid for --method main")
    if args.stage_one_only and args.stage_two_only:
        raise ValueError("Stage I and Stage II cannot both be disabled")
    if enable_s1 and args.stage_two_only:
        raise ValueError("--enable-stage-one and --stage-two-only cannot both be set")
    updates = {key: True for key, value in flags.items() if value}
    if args.stage_one_only or enable_s1:
        updates["disable_stage_one"] = False
    if args.stage_two_only:
        updates["disable_stage_two"] = False
    result = replace(recipe, **updates)
    if result.disable_stage_one and result.disable_stage_two:
        raise ValueError("Stage I and Stage II cannot both be disabled")
    if result.disable_stage_two and any(
        (
            result.stage2_no_modulation,
            result.stage2_one_step_only,
            result.stage2_l2_projection,
            result.stage2_head_only,
        )
    ):
        raise ValueError("Stage II ablations require Stage II to be enabled")
    return result


def variant(recipe):
    labels = [
        ("stage_I_only", recipe.disable_stage_two),
        ("stage_II_only", recipe.disable_stage_one),
        ("no_modulation", recipe.stage2_no_modulation),
        ("one_step_only", recipe.stage2_one_step_only),
        ("l2_projection", recipe.stage2_l2_projection),
        ("head_only_projection", recipe.stage2_head_only),
    ]
    return "+".join(label for label, active in labels if active) or "full"
