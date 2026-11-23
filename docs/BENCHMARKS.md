# Benchmarks

## Four Periodic 2D Benchmarks

Parent trajectories are partitioned into disjoint **train**, **validation**, and **test** (`test_id`) splits prior to extracting 20-interval windows at stride 10:

| Case | Reference $\to$ coarse grid | Public input $\to$ action channels | Coarse integrator ($H$) | Callbacks / window | Train / validation / test windows |
| --- | --- | --- | --- | --- | --- |
| **Cahn-Hilliard** (`cahn_hilliard`) | $384^2 \to 96^2$ ($4\times$ cell avg.) | $2 \to 1$ | Stabilized semi-implicit Fourier Euler ($S=2.0$) | `100` | `2,352` / `784` / `784` (`48 / 16 / 16` parents) |
| **Gray-Scott** (`gray_scott`) | $768^2 \to 96^2$ ($8\times$ cell avg.) | $4 \to 1$ | Strang exact diffusion + 2-stage midpoint reaction | `480` | `4,752` / `1,584` / `1,584` (`48 / 16 / 16` parents) |
| **Kolmogorov NS** (`kolmogorov_ns`) | $256^2 \to 64^2$ (spectral trunc.) | $6 \to 2$ | 4-stage Cox-Matthews ETD-RK4 | `160` | `360` / `120` / `120` (`24 / 8 / 8` parents) |
| **Guide-Field MHD** (`mhd2d`) | $256^2 \to 64^2$ (spectral trunc.) | $8 \to 3$ | Coupled $2\times 2$ matrix ETD-RK4 ($B_0 = 0.5$) | `160` | `360` / `120` / `120` (`24 / 8 / 8` parents) |

### Initialization and Evaluation Protocol

- **Shared initialization ($\theta_0$):** **Gray-Scott**, **Kolmogorov NS**, and **Guide-Field MHD** initialize from zero-output cold start ($h_0 = 0$, via `physics_start`), whereas **Cahn-Hilliard** canonically initializes from a shared offline pair-supervised checkpoint ($h_0 \neq 0$, via `pretrain` and `freeze_common`; cold-start $h_0 = 0$ is also supported).
- **Closure architectures:** Reaction-diffusion cases use a 4-layer periodic FNO (`width=29, modes=6`, `489,057` parameters). Turbulence cases enable full-spectrum radial-angular wavenumber shells (`radial_bins=14`), $2\times$-dealiased quadratic bilinear interactions, and 6-band spectral readouts (`490,027` and `490,057` parameters).
- **Model selection and held-out evaluation:** Training runs up to 100 rounds with full validation every 2 rounds (plateau stopping after 5 checks below $0.2\%$ relative improvement). Checkpoints are selected strictly by lowest validation MSE; held-out `test_id` windows are unlocked only after freezing selected policy hashes, supporting both 20-step ($1\times$) and 80/160-step ($4\times/8\times$) rollout evaluation.

## 3D Ocean Circulation (POM Double Gyre)

The embedded semi-implicit **POM** solver (`src/forge_pom3d/ocean/`) discretizes hydrostatic $\sigma$-coordinate primitive equations on an Arakawa C-grid of **90 $\times$ 82 $\times$ 16** points (**15 active $\sigma$-layers**, resting depth $2500\text{ m}$), coupling a spectral-preconditioned 2D Helmholtz conjugate-gradient free-surface solver ($\Delta t_{\text{3D}} = 600\text{ s}$), Mellor-Yamada 2.5 vertical turbulence, tridiagonal Thomas vertical diffusion, and mode-preserving Robert-Asselin filtering ($\alpha_{\text{RA}} = 0.15$).

### Four-Hour Optimization (`pure_4h`) and 48-Hour Generalization

- **Interface and conservation:** Once per simulated hour ($4$ callbacks / $24$ internal steps / $12$ twenty-minute scored segments per 4h window), `observe_F` exports **161 channels** ($101$ instantaneous/lagged state and geometry fields + $60$ explicit predictor tendencies), and an 8-block 3D dilated SE-ResNet (`4,271,172` parameters, initialized at $h_0 = 0$) predicts **60 baroclinic channels**. Wet-volume orthogonal projection removes column-mean momentum and layer-mean tracer increments prior to implicit vertical diffusion.
- **Data splits:** Reference trajectories come from an 8-member, 9,000-day $4\times$-refined ($354 \times 322 \times 16\sigma$, $0.0625^\circ$) eddy-resolving ensemble coarse-grained onto $90 \times 82 \times 16\sigma$: members **0-5** (days `1,000-6,980`, 20-day stride) provide `1,800` training windows, member **6** provides `16` validation windows (days `7,000-7,900`) plus eight 48h validation rollouts, and member **7** provides `48` held-out test trajectories (days `8,000-8,940`, 20-day stride) for both 4h trajectory MSE and continuous **1-48 hour** rollout NRMSE evaluation.
