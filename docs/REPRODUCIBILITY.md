# Reproducibility

## Installation and Lint Verification

```bash
python -m pip install -e '.[dev,pom3d]'
python -m ruff check --no-cache src scripts
python -m ruff format --check --no-cache src scripts
```

Configuration precedence across both 2D and 3D runners is strictly **CLI flags > environment variables > JSON profile overrides > recipe defaults**. Unknown override keys or conflicting metadata in JSON profiles raise an explicit `ValueError`.

## 2D Data Production and Shared Initialization

### 1. Produce Reference Trajectories and Replay Pairs

```bash
export FORGE_ORIGINAL_ROOT=/path/to/original
export FORGE_PAIRS_ROOT=/path/to/pairs

python scripts/produce_2d_benchmarks.py \
  --campaign-config configs/benchmarks_reaction_2d.json \
  --original-root "$FORGE_ORIGINAL_ROOT" --pairs-root "$FORGE_PAIRS_ROOT" \
  --replay-workers 8 --device cpu

python scripts/produce_2d_benchmarks.py \
  --campaign-config configs/benchmarks_turbulence_2d.json \
  --original-root "$FORGE_ORIGINAL_ROOT" --pairs-root "$FORGE_PAIRS_ROOT" \
  --replay-workers 4 --replay-devices 0,1,2,3 --device cuda:0
```

### 2. Freeze the Shared Starting Checkpoint (`common.pt`)

- **Zero-output cold start ($h_0 = 0$; canonical for `gray_scott`, `kolmogorov_ns`, `mhd2d`, and optional cold-start control for `cahn_hilliard`):**

```bash
export FORGE_CASE=gray_scott
export FORGE_OUTPUT_ROOT=/path/to/gs_suite

for stage in prepare physics_start pure_physics common validate_gate; do
  python -m forge.run --case "$FORGE_CASE" --method "$stage" --device cpu
done
```

- **Offline pair-supervised initialization ($h_0 \neq 0$; canonical for `cahn_hilliard`):**

```bash
export FORGE_CASE=cahn_hilliard
export FORGE_OUTPUT_ROOT=/path/to/ch_suite

python -m forge.run --case "$FORGE_CASE" --method prepare --device cpu
python -m forge.run --case "$FORGE_CASE" --method pretrain --device cuda:0
for stage in freeze_common pure_physics common validate_gate; do
  python -m forge.run --case "$FORGE_CASE" --method "$stage" --device cpu
done
```

## 2D Training, Baselines, Ablations, and Evaluation

### Supported Methods and Controls

`python -m forge.run` supports `--case {cahn_hilliard,gray_scott,kolmogorov_ns,mhd2d}` and `--method {prepare,physics_start,pretrain,freeze_common,pure_physics,common,validate_gate,ES,EKI,PPO_global,PPO_local,emulator,closure_rl,main,AD}`.

| Category | CLI Flags | Description |
| --- | --- | --- |
| Stage selection | `--stage-one-only`, `--stage-two-only` | Isolate Stage I or Stage II |
| Stage II response | `--stage-two-no-modulation`, `--stage-two-one-step-only` | Disable spatial context modulation or multi-step rollout refinement |
| Stage II projection | `--projection-l2-only`, `--projection-head-only` | Replace $H^2$ Sobolev loss with $L^2$ or freeze the hidden backbone |
| Stage I geometry | `--body-basis-mode`, `--body-compensation-rank`, `--stage-one-trust-mode` | Backbone perturbation basis, SVD compensation rank (`4` or `0`), and trust cap |
| Regularization & PPO | `--damping-mode`, `--head-weight-penalty-eta`, `--ppo-lr` | Gauss-Newton damping, readout penalty, and explicit PPO learning-rate override |
| White-box `Solver BPTT` | `--ad-lr`, `--ad-epochs`, `--ad-tag` | Converged AD reference (`--ad-lr 3e-5 --ad-epochs 128` on CH/GS; `--ad-lr 3e-3 --ad-epochs 128` on NS/MHD) |

### Distributed Training and Frozen Test Evaluation

```bash
export FORGE_REFERENCE_ROOT=/path/to/ch_suite
export FORGE_OUTPUT_ROOT=/path/to/ch_main
FORGE_METHOD=main NPROC_PER_NODE=8 bash scripts/train_2d_ddp.sh --max-rounds 100
```

Run converged white-box `Solver BPTT` (`AD`) from the same frozen `common.pt` and window manifest:

```bash
# Reaction-diffusion cases (cahn_hilliard, gray_scott): lr = 3e-5, epochs = 128
# Spectral turbulence cases (kolmogorov_ns, mhd2d): lr = 3e-3, epochs = 128
python -m forge.run --case "$FORGE_CASE" --method AD \
  --output "$FORGE_REFERENCE_ROOT" --device cuda:0 \
  --ad-lr 3e-5 --ad-epochs 128 --ad-tag converged
```

Evaluate a frozen checkpoint on held-out `test_id` windows:

```bash
python scripts/eval_2d_test_id.py \
  --case "$FORGE_CASE" --reference-root "$FORGE_REFERENCE_ROOT" \
  --checkpoint "$FORGE_OUTPUT_ROOT/main/selected.pt" \
  --output-json "$FORGE_OUTPUT_ROOT/test_id_eval.json" --device cuda --world-size 8
```

## 3D POM Training, Baselines, and 1-48 Hour Rollouts

### Four-Hour Closed-Loop Optimization (`pure_4h`)

```bash
export FORGE_POM3D_DATASET_ROOT=/path/to/pom3d_dataset
export FORGE_POM3D_OUTPUT_ROOT=/path/to/pom3d_run
FORGE_POM3D_STAGE=forge_main NPROC_PER_NODE=8 bash scripts/train_pom3d_ddp.sh
```

Supported `--stages` targets include `physics`, `forge_main`, `es`, `eki`, `ppo`, `emulator`, `closure_rl`, `ad_ref`, and `eval_test_4h` (as well as `forge_stage1_only` and `forge_ablation`). Single-factor ablations are controlled via `--ablation-variant {none,no_stage1,no_body_comp,no_kobs,intra_hour_only,action_l2,head_only_stage2}` or `--stage-one-only`.

### Evaluate Continuous 1-48 Hour Forecast Generalization

```bash
python scripts/eval_pom3d_rollout.py \
  --dataset-root "$FORGE_POM3D_DATASET_ROOT" \
  --checkpoint "$FORGE_POM3D_OUTPUT_ROOT/stage2_forge_main/best_model.pt" \
  --split test --hours 1 2 4 24 48 --device cuda:0 \
  --output-json "$FORGE_POM3D_OUTPUT_ROOT/rollout_test.json"
```

Pass `--mode pure_physics` without `--checkpoint` to evaluate the unparameterized coarse ocean solver.
