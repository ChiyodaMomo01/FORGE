# Architecture

## Public Solver Contract

Let $q_t$ denote the coarse solver state at callback $t$, $x_t$ the public observation, and $a_t$ the executed closure action. With feature extractor $F$, baseline coarse formula $g_0$ ($g_0 \equiv 0$ in 3D POM), learner-owned neural closure $G_\theta$ parameterized by $\theta = (h, b)$, fixed physical scaling $S$, conservation projector $P$, and host time integrator $H$, each hybrid step executes

$$
x_t = F(q_t), \qquad a_t = g_0(x_t) + P S G_\theta(x_t), \qquad q_{t+1} = H(q_t, a_t).
$$

Within each benchmark, all methods start from a frozen checkpoint $\theta_0$ (`common.pt` in 2D; `common_zero.pt` in 3D POM):
- **Zero-output cold start ($h_0 = 0$):** default on 2D Gray-Scott, 2D Kolmogorov NS, 2D Guide-Field MHD, and 3D POM (`physics_start` sets the linear readout $h_0 = 0$ so $G_{\theta_0}(x) \equiv 0$; `validate_gate` checks rollout equality with unparameterized coarse physics).
- **Offline pair-supervised initialization ($h_0 \neq 0$):** canonical on 2D Cahn-Hilliard (`pretrain` followed by `freeze_common` fits coarse-grained input-output pairs without solver derivatives; zero-output cold start is also supported).

## Acyclic Package Boundaries

### 2D Package (`src/forge/`)

| Layer / Subpackage | Responsibility | Allowed Dependencies |
| --- | --- | --- |
| `forge.core` / foundational modules (`config`, `schemas`, `spectral_ops`, `networks`, `diagnostics`) | Shared recipes, canonical schema contracts, array/model serialization, deterministic digests, network architectures, spectral primitives | Standard library, NumPy, PyTorch |
| `forge.cases`, `forge.datasets`, `forge.paired`, `forge.production`, `forge.collect` | Four 2D PDE discretizations, dataset checks/storage, offline pair regression, trajectory production, and restriction audits | Foundational layer |
| `forge.runtime`, `forge.actor_service`, `forge.sandbox` | Window bank loading, process-isolated solver execution, request validation, and `test_id` freeze gate | Foundational layer, `forge.cases`, `forge.paired` (never `forge.learning`) |
| `forge.learning` | Two-stage FORGE (`subspace`, `geometry`, `function_main`, `tangent/*`, `projection`, `closed_loop`) and forward-only baselines (`study`, `ppo_joint`, `closure_rl`, `emulator`) | Foundational layer and public arrays only; communicates with `forge.runtime` strictly over an OS subprocess pipe via `Client` |
| `forge.privileged` | Serial and distributed white-box `Solver BPTT` (`AD`) references | Foundational layer, `forge.runtime`, and differentiable `forge.cases` |
| `forge.execution`, `forge.stages`, `forge.run` | CLI normalization, stage dispatch, and validation gate | All layers above |

### 3D Package (`src/forge_pom3d/`)

| Layer / Subpackage | Responsibility | Allowed Dependencies |
| --- | --- | --- |
| `forge_pom3d.config`, `forge_pom3d.ocean` | Configuration schema and self-contained semi-implicit 3D POM ocean dynamical core (`semi_implicit`, `hybrid`, `io`, `contracts`) | Standard library, NumPy, PyTorch, optional `netCDF4` / `triton` |
| `forge_pom3d.{distributed,artifacts,packing,sampling,network,physics_env}` | DDP collectives, non-self-referential source/config digests, wet-mask volume packing, deterministic window sampling, 3D dilated ResNet, and `BlackBoxProvider3D` | Foundational layer and `forge_pom3d.ocean` |
| `forge_pom3d.evaluation` | Four-hour validation/test evaluation and `Pure4hSelectionTracker` | Infrastructure layer |
| `forge_pom3d.training` (`stage1`, `response_operator`, `stage2`, `trainer`) & `forge_pom3d.baselines` (`physics`, `es`, `eki`, `ppo`, `emulator`, `closure_rl`, `ad_reference`) | Parallel algorithm implementations with zero cross-baseline imports and zero upward imports into CLI/runner | Infrastructure and evaluation layers |
| `forge_pom3d.{runner,cli,__main__}` | Stage dispatch and pure argument/profile parsing | Algorithm and lower layers |

## Two-Stage Optimization Mechanics

### Stage I: Parameter-Space Perturbation and Readout Drift Compensation

Stage I builds $M \ll p$ joint readout--backbone directions $u_m = (\Delta h_m, \Delta b_m)$ using zero solver calls: in 2D ($M = 16$: $8$ readout + $8$ backbone), forward-mode network Jacobians orthonormalize the first-order output tangents $v_m = \frac{\partial G_\theta}{\partial \theta} u_m$ on cached public observations (pairing backbone modes with whitened readout modes at cold start $h_0 = 0$); in 3D POM ($M = 8$), `build_mixed_head_body_basis` constructs QR-orthonormalized mixed readout--backbone directions in parameter space and computes their cached output tangents for drift compensation. Before querying the solver at finite step coordinates, a rank-$4$ truncated-SVD readout adjustment (wet-weighted in 3D POM) cancels nonlinear trunk feature drift while preserving the exact first-order Jacobian at zero step size. Antithetic $\pm \rho_m u_m$ solver rollouts form the trajectory Jacobian Gram matrix and curvature-scaled Gauss-Newton candidates, screened on actual rollout loss.

### Stage II: Causal Output-Response Identification and Full-Network Projection

Stage II injects paired $\pm \delta a_t$ orthogonal output-space perturbations under identical initial states (DCT-II fitting probes and disjoint DCT-IV validation probes in 2D; wet-projected Walsh-Hadamard harmonic probes in 3D) to identify a structured causal linear response operator $\delta y \approx \mathcal{R} \delta a$:
- **2D shared periodic recurrence:** combines Hermitian radial wavenumber-shell multipliers, odd directional derivative multipliers, polynomial high-wavenumber tails, and spatial context modulations under bisection step-gain clamping $\|A_t\|_2 \le \gamma_{\text{step}}$.
- **3D bounded-basin operator:** fits 30 causal hour-to-segment response blocks $\mathcal{R}_{jk}$ (column-wise vertical coupling, self-adjoint zero-flux wet-Laplacian diffusion, and basin harmonic modes) alongside 6 causal hourly observation-shift blocks $K_{\text{obs}}$ at zero additional solver calls.

After passing held-out probe verification ($\text{RRE} \le 0.50$, $R^2 \ge 0.50$, $\cos \ge 0.80$), trust-region conjugate gradients solve $(\mathcal{R}^* \mathcal{R} + \lambda I) \Delta a = -\mathcal{R}^* r$ for target actions $a_t^\star = a_t + \Delta a_t$. All neural parameters $\theta = (h, b)$ are updated offline via discrete $H^2$ Sobolev variable projection (exact closed-form readout solve, probe-calibrated Adam trunk update, and variable-projection line search) in 2D or replay-weighted optimization through $\mathcal{R}$ and $K_{\text{obs}}$ in 3D, followed by forward rollout acceptance.
