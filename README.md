# Frozen-PINN PAT simulation framework

This is the consolidated simulation code for propagation-aware PAT control with a Frozen-PINN optical propagation model.

## Project structure

```text
PAT_framework_final/
├── config/
│   └── settings.py
├── system_model/
│   ├── optical_system.py
│   └── turbulence.py
├── solver/
│   ├── frozen_pinn.py
│   ├── frozen_pinn_torch.py
│   ├── differentiable_ssfm.py
│   ├── baselines.py
│   └── optimization.py
├── simulation/
│   ├── runner.py
│   ├── plotting.py
│   └── results.py
├── experiments/
│   ├── compare_pat.py
│   ├── runtime_scaling.py
│   └── validate_framework.py
└── results/
```

## Important modeling convention

`--intervals` changes only the simulated duration. It does **not** change the target speed.

The target is defined in physical time:

```text
t_k = k * control_interval
x_target(t) = A_x sin(2 pi t / T_target)
y_target(t) = A_y sin(4 pi t / T_target + phase_y)
```

Default values are:

```text
control interval = 0.01 s
trajectory period = 2.0 s
```

Therefore one complete default target period contains 200 PAT intervals. For a full figure-eight trajectory:

```bash
python experiments/compare_pat.py --objective centroid --intervals 200
```

Changing `--intervals 8` to `--intervals 200` no longer changes the underlying trajectory velocity.

## Physical FSM constraints vs numerical optimization

The physical actuator limit and the numerical optimizer are separated.

Physical constraints:

```text
--theta-max
--theta-slew-rate
--theta-slew-max        # optional per-interval override
--control-interval
```

By default,

```text
theta_slew_per_interval = theta_slew_rate * control_interval
```

Numerical settings:

```text
--max-opt-iters
--optimizer-step
--line-search-steps
--optimizer-min-step
--gradient-tol
--objective-rel-tol
--objective-abs-tol
```

Increasing `--max-opt-iters` cannot increase the physically reachable FSM angle.

## Main comparison

Power:

```bash
python experiments/compare_pat.py \
  --objective power \
  --intervals 100 \
  --device mps
```

Coupling efficiency:

```bash
python experiments/compare_pat.py \
  --objective coupling \
  --intervals 100 \
  --device mps
```

Centroid:

```bash
python experiments/compare_pat.py \
  --objective centroid \
  --intervals 200 \
  --device mps \
  --gif \
  --fps 10
```

On CUDA:

```bash
python experiments/compare_pat.py --objective power --device cuda --dtype float64
```

On CPU:

```bash
python experiments/compare_pat.py --objective power --device cpu --dtype float64
```

### Same-device comparison

The default `ssfm_oracle` is the differentiable torch SSFM oracle. `Frozen-PINN` and `Diff-SSFM oracle` use the same `--device` and requested `--dtype`.

On Apple MPS, both automatically use float32 internally.

The legacy NumPy finite-difference SSFM oracle remains available as:

```text
ssfm_oracle_cpu
```

## Frozen-PINN defaults

The stable default uses a higher-capacity frozen basis because centroid moments are more sensitive to small field errors than received power:

```text
hidden width      = 1000
collocation side  = 24
boundary side     = 16
SVD cutoff        = 1e-6
pinv rcond        = 1e-6
RK4 steps         = 64
```

The optimized torch implementation uses:

1. frozen tanh neural features,
2. the boundary-compliant layer,
3. truncated SVD,
4. a reduced transition matrix `T_k` computed once per channel interval,
5. reduced-space power/coupling/centroid objectives.

During steering optimization it does not reconstruct the full receiver grid.

## Centroid visualization

With:

```bash
python experiments/compare_pat.py --objective centroid --gif --fps 10
```

`centroid_tracking.gif` is generated.

Its layout reproduces the original tracking visualization:

- left: actual SSFM receiver-plane intensity,
- white circle: camera centroid ROI,
- white `+`: target Rx center,
- red `x`: actual centroid,
- cyan open circle: Frozen-PINN predicted centroid,
- right: target, actual, and Frozen-PINN predicted centroid trajectories.

The status bar also shows physical time, FSM angle, actual target error, and Frozen-PINN prediction-vs-actual centroid error.

## Results

Each run writes:

```text
results/<experiment>/results.json
results/<experiment>/objective_comparison.png
results/<experiment>/runtime_comparison.png
```

For centroid:

```text
results/<experiment>/centroid_trajectories.png
results/<experiment>/centroid_tracking.gif   # when --gif is enabled
```

The JSON contains physical time, target position, FSM command, actual metric, actual centroid, predicted centroid when available, optimizer diagnostics, and timing diagnostics.

## Repeated runs

```bash
python experiments/compare_pat.py \
  --objective power \
  --repeat 20 \
  --intervals 100
```

By default only the physical turbulence seed changes between repetitions. The Frozen-PINN random feature basis remains fixed. Use:

```text
--vary-frozen-seed
```

only for a separate basis-initialization robustness ablation.

## Runtime scaling

Repeated-query latency:

```bash
python experiments/runtime_scaling.py \
  --objective power \
  --device mps \
  --grid-sizes 64 128 256 512 \
  --query-counts 1 2 4 8 \
  --ssfm-steps 32
```

The runtime experiment separately reports:

- offline/static setup,
- interval-specific model preparation,
- repeated objective-gradient query latency,
- total online interval latency.

Outputs include:

```text
runtime_scaling.json
runtime_vs_grid.png
runtime_vs_queries.png
interval_total_vs_grid.png
interval_total_vs_queries.png
quality_vs_latency.png
```

## Validation

Before running paper-scale experiments:

```bash
python experiments/validate_framework.py
```

The validation suite checks:

- target motion is independent of `num_intervals`,
- the target closes after one physical period,
- torch Frozen-PINN matches the SciPy reference objective/gradient,
- reduced-space centroid equals centroid computed from the reconstructed Frozen-PINN field.

## Dependencies

```bash
pip install numpy scipy matplotlib pillow torch
```


## Focused performance plots

`No control` is still evaluated and stored in JSON, but it no longer determines the y-axis of the main performance graph.

Each run saves:

```text
objective_comparison.png          # controlled methods only, zoomed
objective_comparison_all.png      # includes No control
```

For repeated experiments:

```text
aggregate_objective_comparison.png       # controlled methods only, zoomed
aggregate_objective_comparison_all.png   # includes No control
```

Scalar-objective GIFs also focus on controlled methods. `No control` remains available in JSON and the `_all.png` graphs.
