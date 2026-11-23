"""Black-box and white-box reference baseline methods for 3D Double-Gyre POM."""

from .ad_reference import (
    run_ad_reference,
    run_stage8_whitebox_ad,
    verify_whitebox_ad_parity_and_gradient,
)
from .closure_rl import (
    WetColumnSpatialCritic3D,
    compute_wet_column_ppo_stats,
    evaluate_closure_rl_batch_stats_ddp,
    run_closure_rl_baseline,
    run_stage7_closure_rl,
)
from .eki import compute_streamed_eki_ensemble_update, run_eki_baseline
from .emulator import (
    NeuralEmulator3D,
    evaluate_emulator_unseen_3d,
    run_emulator_baseline,
    run_stage6_emulator,
)
from .es import run_es_baseline
from .physics import run_physics_baseline
from .ppo import compute_joint_gaussian_ppo_stats, run_ppo_baseline

__all__ = [
    "NeuralEmulator3D",
    "WetColumnSpatialCritic3D",
    "compute_joint_gaussian_ppo_stats",
    "compute_streamed_eki_ensemble_update",
    "compute_wet_column_ppo_stats",
    "evaluate_closure_rl_batch_stats_ddp",
    "evaluate_emulator_unseen_3d",
    "run_ad_reference",
    "run_closure_rl_baseline",
    "run_eki_baseline",
    "run_emulator_baseline",
    "run_es_baseline",
    "run_physics_baseline",
    "run_ppo_baseline",
    "run_stage6_emulator",
    "run_stage7_closure_rl",
    "run_stage8_whitebox_ad",
    "verify_whitebox_ad_parity_and_gradient",
]
