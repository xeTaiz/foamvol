# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Build

**Python environment:** `micromamba run -n radfoam`

**CUDA env vars required for building:**
```bash
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
export CUDA_HOME=/usr/local/cuda-12.4
```

**Build (after modifying CUDA/C++ in `src/` or `torch_bindings/`):**
```bash
micromamba run -n radfoam pip install -e .
# or with CMake for faster iteration:
cd build && make install
```

**Run training:**
```bash
micromamba run -n radfoam python train.py -c configs/r2_1m.yaml
micromamba run -n radfoam python train.py -c configs/r2_1m.yaml --experiment_name my_run
```

**Batch training across datasets:**
```bash
micromamba run -n radfoam python train_all.py -c configs/r2_1m.yaml --name my_run
micromamba run -n radfoam python train_all.py -c configs/r2_1m.yaml --name my_run --worker 1 --of 4
micromamba run -n radfoam python train_all.py --name my_run --summarize
micromamba run -n radfoam python train_all.py --list --data-root r2_data
```

**Evaluate / benchmark:**
```bash
micromamba run -n radfoam python test.py -c output/<run>/config.yaml
micromamba run -n radfoam python benchmark.py -c output/<run>/config.yaml
```

**Run a sweep:**
```bash
micromamba run -n radfoam python sweep_24_smooth.py
micromamba run -n radfoam python sweep_24_smooth.py --runs V01 V02
micromamba run -n radfoam python sweep_24_smooth.py --list
micromamba run -n radfoam python sweep_24_smooth.py --summarize
```

**Output:** All runs land in `output/<experiment_name>/` containing `config.yaml`, `model.pt`, `metrics.txt`, and TensorBoard events.

---

## Architecture

### The Voronoi Foam Representation

The model is a **Voronoi foam** (Delaunay triangulation of point cloud): each point owns a Voronoi cell with a learned scalar density. Rendering integrates density along rays using Beer-Lambert absorption through the tetrahedral mesh.

The core data structures live in `CTScene` (`radfoam_model/scene.py`):
- `self.primal_points` — [N, 3] point positions (learnable)
- `self.density` — [N, 1] raw density (softplus-activated: `activation_scale * softplus(raw, β=10)`)
- `self.point_adjacency` / `self.point_adjacency_offsets` — CSR neighbor graph from Delaunay mesh
- `self._cached_cell_radius` — farthest neighbor distance (Voronoi cell size proxy)

The Delaunay triangulation is rebuilt/updated via `model.update_triangulation()`. The C++/CUDA kernels are accessed through `import radfoam` (compiled from `src/` + `torch_bindings/`).

### Forward Pass

`train.py` → `CTScene.forward()` → `TraceRays` (`radfoam_model/render.py`)

`TraceRays` is a custom `torch.autograd.Function` that:
1. Calls `pipeline.trace_forward()` (CUDA) — ray-cell intersection, density accumulation
2. Returns projections [N_rays], per-cell contributions, hit counts
3. On backward: calls `pipeline.trace_backward()` — propagates `dL/d(density)` and `dL/d(points)`

`CUDADensityPipeline` (in `src/tracing/pipeline.cu`) handles the actual line-integral kernels.

### Training Phases

The training loop in `train.py` has distinct phases:

| Phase | Iterations | What happens |
|-------|-----------|--------------|
| Densification | `densify_from` → `densify_until` (1k–6k) | Points added every N iters; `prune_and_densify()` grows mesh from `init_points` to `final_points` |
| Regularization | configurable starts | TV, voxel/neighbor variance, BF all have `*_start` params |
| Freeze geometry | `freeze_points` (9.5k) | Point positions frozen; only density optimized |
| Interpolation | `interpolation_start` (9k) | Switches to IDW density evaluation (smoother output) |
| End of densify | `densify_until` | Final standalone prune; interpolation sigma prepared |

### Densification (`prune_and_densify`)

Each densification step adds `(densify_factor - 1) * N_current` new points. The interval auto-adjusts: `next_step ∝ N_current * (until - from) / (final - init)` so that pruning (which reduces N_current) naturally increases step frequency.

New points are sampled with three strategies (budget split by fractions):
- **Gradient** (`gradient_fraction=0.4`): weighted by accumulated ray error × cell radius
- **IDW** (`idw_fraction=0.3`): placed along edges with high bilateral prediction error
- **Entropy** (`entropy_fraction=0.3`): weighted by neighbor density distribution entropy

Pruning removes cells with `low_contrib` or `tiny_radius`, plus optional redundancy pruning.

### Interpolation Mode

At `interpolation_start`, density evaluation switches from constant per-cell to IDW over neighbors:
```
mu_query = Σ w_j * mu_j / Σ w_j
w_j = exp(-d²/σ²) * exp(-Δμ²/σ_v²)   [spatial × bilateral weight]
```
`per_cell_sigma=True`: σ = `interp_sigma_scale * cell_radius_i` (adaptive to cell size)
`per_neighbor_sigma=True`: bilateral weight uses the neighbor's own sigma

---

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training loop, all logging, densification orchestration |
| `radfoam_model/scene.py` | `CTScene`: all model methods (densification, regularization, pruning, checkpointing) |
| `radfoam_model/render.py` | `TraceRays`: differentiable ray tracing autograd function |
| `configs/__init__.py` | All hyperparameter classes (`PipelineParams`, `ModelParams`, `OptimizationParams`, `DatasetParams`) |
| `data_loader/` | `DataHandler` for R2-Gaussian (CT cone-beam) and Mip-NeRF 360 datasets |
| `vis_foam.py` | 2D slice visualization, IDW diagnostics, DRR rendering |
| `voxelize.py` | Export trained model to regular 3D numpy grid |
| `train_all.py` | Batch train + summarize across multiple datasets |
| `src/tracing/pipeline.cu` | CUDA kernels: CT line-integral forward/backward |
| `torch_bindings/pipeline_bindings.cpp` | Python bindings: `create_ct_pipeline` |

---

## Config Parameters Reference

All parameters are CLI args (via `configargparse`) and YAML keys. Config classes are in `configs/__init__.py`.

### PipelineParams (training procedure)

| Param | Default | Description |
|-------|---------|-------------|
| `iterations` | 10000 | Total training steps |
| `rays_per_batch` | 2M | Rays sampled per step |
| `loss_type` | `l2` | `l1` or `l2` reconstruction loss |
| `densify_from` | 1000 | First densification step |
| `densify_until` | 6000 | Last densification step |
| `densify_factor` | 1.15 | Growth factor per step (adds 15% of current N) |
| `gradient_fraction` | 0.4 | Budget fraction for gradient-weighted new points |
| `idw_fraction` | 0.3 | Budget fraction for IDW-edge new points |
| `entropy_fraction` | 0.3 | Budget fraction for entropy-weighted new points |
| `redundancy_threshold` | 0.01 | IDW error threshold (× p95 density) for pruning candidates |
| `redundancy_cap` | 0.05 | Max fraction of cells pruned per step (constant) |
| `redundancy_cap_init` | 0.0 | Adaptive cap: start value (0 = use `redundancy_cap`) |
| `redundancy_cap_final` | 0.0 | Adaptive cap: end value (schedule over densify range) |
| `prune_variance_criterion` | False | Use neighborhood variance × radius as pruning score instead of IDW |
| `prune_hops` | 1 | K-hop neighborhood depth for variance pruning |
| `interpolation_start` | 9000 | Switch to IDW interpolation at this step (-1 = off) |
| `interp_sigma_scale` | 0.7 | IDW spatial sigma scale (× cell radius if `per_cell_sigma`) |
| `interp_sigma_v` | 0.35 | IDW bilateral value sigma |
| `per_cell_sigma` | False | Adaptive spatial sigma (σ = scale × cell_radius_i) |
| `per_neighbor_sigma` | False | Use neighbor's sigma for bilateral weight |
| `high_error_fraction` | 0.0 | Extra rays toward high-residual views (0 = off) |
| `high_error_power` | 1.0 | Power scaling on error weights (1=linear, 2=quadratic) |
| `high_error_start` | -1 | Activation step (-1 = densify_from) |
| `targeted_fraction` | 0.0 | Extra rays toward small/under-visited cells (0 = off) |
| `targeted_start` | -1 | Activation step (-1 = densify_from) |
| `bf_start` | -1 | Bilateral filter start (-1 = off) |
| `bf_until` | 6000 | Bilateral filter end |
| `bf_period` | 10 | Apply BF every N steps |
| `bf_sigma_init/final` | 2.0/0.3 | Spatial sigma schedule (× cell radius) |
| `bf_sigma_v_init/final` | 10.0/0.1 | Value sigma schedule (high = Gaussian blur, low = bilateral) |

### ModelParams (scene representation)

| Param | Default | Description |
|-------|---------|-------------|
| `init_points` | 32000 | Starting point count |
| `final_points` | 128000 | Target point count |
| `activation_scale` | 1.0 | Multiplier on softplus density output |
| `init_scale` | 1.05 | Half-extent of initialization volume |
| `init_type` | `random` | `random` or `regular` point initialization |
| `init_density` | 0.0 | Initial raw density value (before softplus) |
| `init_volume_path` | `""` | Path to `.npy` FDK volume for density init (empty = disabled) |
| `init_points_file` | `""` | Path to checkpoint `.pt` to load points from |

### OptimizationParams (learning)

| Param | Default | Description |
|-------|---------|-------------|
| `points_lr_init/final` | 2e-4/5e-6 | Position LR schedule (exponential decay) |
| `density_lr_init/final` | 5e-2/1e-3 | Density LR schedule |
| `freeze_points` | 9500 | Freeze point positions at this step |
| `density_grad_clip` | 1.0 | Gradient clipping on density params |
| `tv_weight` | 1e-4 | Charbonnier TV weight (0 = off) |
| `tv_start` | 5000 | TV activation step |
| `tv_epsilon` | 1e-4 | Charbonnier smoothing ε |
| `tv_area_weighted` | False | Weight TV by cell area |
| `tv_border` | False | Evaluate TV at Voronoi borders (uses gradient field if available) |
| `tv_anneal` | False | Anneal TV weight to 0 over freeze_points |
| `tv_on_raw` | False | Apply TV to raw (pre-activation) density |
| `voxel_var_weight` | 0.0 | Bilateral voxel-grid variance loss weight |
| `voxel_var_resolution` | 32 | Voxel grid resolution per axis |
| `voxel_var_start` | 0 | Activation step |
| `neighbor_var_weight` | 0.0 | Graph neighbor variance loss weight |
| `neighbor_var_hops` | 1 | K-hop neighborhood depth (1 = immediate neighbors) |
| `neighbor_var_start` | 0 | Activation step |
| `var_sigma_v_init` | 0.2 | Bilateral value sigma at `densify_from` (large = plain smoothing) |
| `var_sigma_v_final` | 0.2 | Bilateral value sigma at `densify_until` (small = edge-preserving) |
| `gradient_start` | -1 | Enable per-cell linear gradient field (-1 = off) |
| `gradient_max_slope` | 5.0 | Maximum slope in physical density units |
| `gradient_freeze_points` | 500 | Freeze gradient field N steps after activation |
| `gaussian_start` | -1 | Enable Gaussian splat parameters (-1 = off) |

### DatasetParams

| Param | Default | Description |
|-------|---------|-------------|
| `dataset` | `r2_gaussian` | Dataset type (`r2_gaussian`, `mipnerf360`, `db`) |
| `data_path` | — | Root path of the dataset |

---

## Dataset Format (R2-Gaussian / CT)

Each dataset directory must contain:
```
<dataset>/
  proj_train/proj_train_XXXX.npy   # training projection images
  proj_test/proj_test_XXXX.npy     # test projection images
  meta_data.json                    # scanner geometry, angle lists
  vol_gt.npy                        # ground-truth volume (256³ float32, X,Y,Z order)
  traditional/fdk/ct_pred.npy       # FDK reconstruction (optional, for init_volume_path)
```

`meta_data.json` contains the scanner configuration (SID, SDD, detector size, angles). Used by `DataHandler` to set up ray geometry.

---

## Features Summary

### Regularization
- **TV on edges** (`tv_weight > 0`): Charbonnier (smooth L1) on density differences across Voronoi edges. `tv_border=True` evaluates at face midpoints using gradient fields.
- **Voxel variance** (`voxel_var_weight > 0`): Random-offset voxel grid; bilateral variance within each voxel. Controlled by `var_sigma_v_init/final` schedule.
- **Neighbor variance** (`neighbor_var_weight > 0`): Graph-based; each cell penalized for deviating from k-hop neighborhood mean. Uses same `var_sigma_v` schedule. Naturally adaptive to cell size. `neighbor_var_hops=2` gives broader smoothing.
- **Sigma annealing** (`var_sigma_v_init != var_sigma_v_final`): Linear schedule from `init` (strong plain smoothing) to `final` (edge-preserving). Applied to both voxel and neighbor variance. Schedule runs from `densify_from` to `densify_until`.
- **Bilateral filter** (`bf_start >= 0`): Direct in-place density smoothing during densification. Spatial and value sigmas anneal independently.

### Ray Sampling
- **High-error sampling** (`high_error_fraction > 0`): Extra rays allocated to views/pixels with large residuals (biases training toward poorly-reconstructed regions).
- **Targeted sampling** (`targeted_fraction > 0`): Extra rays weighted by 1/cell_size toward small cells (helps cells that are rarely visited).
- Both sampling modes activate at `densify_from` by default.

### Pruning
- **Basic pruning** (always active): Remove cells with `point_contribution < 0.01` or `cell_radius < 1e-4`.
- **IDW redundancy pruning** (`redundancy_cap > 0`, `prune_variance_criterion=False`): Prune cells with IDW leave-one-out error below `redundancy_threshold × p95_density`. Independent-set filter prevents adjacent removals.
- **Variance-based pruning** (`prune_variance_criterion=True`): Score = `neighborhood_variance × max(radius, p10_radius)`. No threshold — purely cap-based. Targets smooth + small cells; large empty-space cells protected by large radius.
- **Adaptive cap** (`redundancy_cap_init/final`): Linearly schedules cap from high (early = more cycles) to low (late = accumulate cells). Auto-increases densification frequency.

### Density Initialization
- **FDK init** (`init_volume_path`): Grid-samples FDK `.npy` volume at initial point positions via `F.grid_sample`. Applies `softplus_inv` to set raw density params. Volume coordinate convention: flip (x,y,z)→(z,y,x) for grid_sample since volume is stored (X,Y,Z)=(D,H,W). Negatives in FDK are Ram-Lak ramp filter artifacts — clamped to 1e-6 before inversion.

### Interpolation
- **IDW mode** (after `interpolation_start`): Switches from constant per-cell density to bilateral IDW over neighbors. `per_cell_sigma=True` is the best setting (adaptive σ per cell).
- **Per-cell gradient** (`gradient_start >= 0`): Adds learned gradient vector per cell; density evaluated as `μ_i + tanh(g)·Δx * max_slope`. `tv_border=True` uses this for richer regularization.
- **Gaussian splats** (`gaussian_start >= 0`): Adds Gaussian density peaks per cell.

---

## Sweep Scripts

All sweeps follow the same pattern:
- `ALL_RUNS` dict: `{name: config_dict}`
- `base_config(**kwargs)` generates full config with defaults
- `--list` prints run names and key params
- `--runs ID1 ID2` selects specific runs
- `--summarize` collects existing `metrics.txt` files into `summary.csv`
- Results in `output/sweep<N>_<name>/`

| Script | Sweep # | What it tests |
|--------|---------|---------------|
| `sweep_scaling.py` | 17 | Cell count (512k–4M) |
| `sweep_he_tv.py` | 18 | High-error fraction × voxel TV weight |
| `sweep_vvar.py` | 19 | Voxel variance weight, resolution, sigma_v |
| `sweep_vvar2.py` | 20 | Resolution 48/128, sigma gaps, BF combos |
| `sweep_fdk.py` | 21 | FDK init vs. uniform, with/without vvar |
| `sweep_75view.py` | 22 | Anti-grittiness on 75-view: fewer cells, pruning, stronger vvar, FDK |
| `sweep_targeting.py` | 23 | high_error_fraction × targeted_fraction |
| `sweep_24_smooth.py` | 24 | Graph neighbor variance + adaptive pruning |

**Current best config for 75-view CT (from sweeps 22/23):**
- `final_points=512000`, `voxel_var_weight=1e-3`, `high_error_fraction=0.2`, `targeted_fraction=0.1`

---

## CUDA/C++ Backend

`radfoam` Python module wraps compiled CUDA kernels from `src/`:
- `radfoam.Triangulation` — Delaunay triangulation with incremental updates
- `radfoam.farthest_neighbor(points, adj, offsets)` → `(farthest_idx, cell_radius)`
- `radfoam.nn(query, aabb_tree, points)` → nearest neighbor indices
- `create_ct_pipeline(points, adj, offsets)` → `CUDADensityPipeline`
  - `.trace_forward(rays, ...)` → projections, contributions, hit_count
  - `.trace_backward(rays, grad, ...)` → gradients

`CUDADensityPipeline` computes: `μ = softplus(raw) * activation_scale`, optionally with gradient: `μ = max(0, μ_base + max_slope * tanh(g) · Δx)`

Adjacency in CSR format: `point_adjacency` = column indices (E,), `point_adjacency_offsets` = row pointers (N+1,).

**Eigen note:** Vec3f uses `g[0]`, `g[1]`, `g[2]` (not `.x/.y/.z`); element-wise multiply via `.cwiseProduct()`.
