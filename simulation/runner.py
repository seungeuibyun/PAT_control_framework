from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np

from config.settings import ExperimentConfig
from system_model.optical_system import OpticalPATSystem
from solver.frozen_pinn import FrozenPINNSolver
from solver.frozen_pinn_torch import TorchFrozenPINNSolver, resolve_torch_device
from solver.baselines import make_baseline
from simulation.results import save_results_json
from simulation.plotting import (
    plot_objective_comparison,
    plot_runtime_comparison,
    plot_centroid_trajectories,
    save_comparison_gif,
    plot_aggregate_objective,
)


def _solver_summary(objective, records):
    display_values = np.asarray([r["display_metric"] for r in records], dtype=float)
    runtime = np.asarray([r["runtime_sec"] for r in records], dtype=float)
    summary = {
        "mean_runtime_sec": float(np.mean(runtime)),
        "total_runtime_sec": float(np.sum(runtime)),
    }
    if objective == "centroid":
        summary["mean_centroid_error_mm"] = float(np.mean(display_values))
        summary["max_centroid_error_mm"] = float(np.max(display_values))
    else:
        summary["mean_objective"] = float(np.mean(display_values))
        summary["min_objective"] = float(np.min(display_values))
        summary["max_objective"] = float(np.max(display_values))
    return summary


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def run_comparison(
    cfg: ExperimentConfig,
    solver_names,
    output_root: str | Path,
    make_gif: bool = False,
    gif_fps: int = 3,
):
    system = OpticalPATSystem(cfg)
    solvers = []
    for name in solver_names:
        key = name.lower().strip()
        if key in {"frozen_pinn", "frozen-pinn", "frozen pinn"}:
            backend = cfg.frozen_pinn.backend.lower()

            if backend == "auto":
                # The optimized torch Frozen-PINN supports CPU/CUDA/MPS.
                # Use it on every device for consistent runtime comparisons.
                backend = "torch"

            if backend == "torch":
                solvers.append(
                    TorchFrozenPINNSolver(
                        system,
                        cfg,
                        device=cfg.frozen_pinn.device,
                        dtype=cfg.frozen_pinn.dtype,
                        rk_steps=cfg.frozen_pinn.rk_steps,
                    )
                )
            elif backend == "scipy":
                solvers.append(
                    FrozenPINNSolver(
                        system,
                        cfg,
                    )
                )
            else:
                raise ValueError(
                    "Frozen-PINN backend must be auto, scipy, or torch"
                )
        else:
            solvers.append(make_baseline(name, system, cfg))

    output_dir = Path(output_root) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_results = {}

    for solver in solvers:
        print(f"\n=== {solver.name} ===")
        theta_prev = np.zeros(2, dtype=float)
        history = []
        records = []

        for k in range(cfg.tracking.num_intervals):
            target = system.target_position(k)
            solved = solver.solve(
                interval=k,
                target=target,
                theta_prev=theta_prev,
                history=history,
            )
            theta = np.asarray(solved["theta"], dtype=float)
            U_truth = system.ssfm(theta, k)
            objective_value = system.metric(U_truth, target, cfg.tracking.objective)
            display_value = system.display_metric(U_truth, target, cfg.tracking.objective)
            actual_centroid = system.centroid(U_truth, target)

            diagnostics = solved.get("diagnostics", {})
            record = {
                "interval": k,
                "time_sec": system.time_at_interval(k),
                "target": target,
                "theta": theta,
                "objective_value": objective_value,
                "display_metric": display_value,
                "actual_centroid": actual_centroid,
                "runtime_sec": solved["runtime_sec"],
                "predicted_metric": solved.get("predicted_metric"),
                "diagnostics": diagnostics,
            }

            if "predicted_centroid" in diagnostics:
                record["predicted_centroid"] = np.asarray(
                    diagnostics["predicted_centroid"], dtype=float
                )
            records.append(record)
            history.append(record)
            theta_prev = theta

            metric_text = (
                f"centroid error={display_value:.4f} mm"
                if cfg.tracking.objective == "centroid"
                else f"metric={display_value:.6g}"
            )
            print(
                f"[{k+1:02d}/{cfg.tracking.num_intervals:02d}] "
                f"theta=({theta[0]*1e6:+.2f}, {theta[1]*1e6:+.2f}) urad, "
                f"{metric_text}, runtime={solved['runtime_sec']:.3f}s"
            )

        solver_results[solver.name] = {
            "records": records,
            "summary": _solver_summary(cfg.tracking.objective, records),
        }

    json_path = save_results_json(output_dir, cfg, solver_results)
    plot_paths = [
        plot_objective_comparison(output_dir, cfg.tracking.objective, solver_results),
        plot_runtime_comparison(output_dir, solver_results),
    ]
    if cfg.tracking.objective == "centroid":
        plot_paths.append(plot_centroid_trajectories(output_dir, solver_results))

    gif_path = None
    if make_gif:
        gif_path = save_comparison_gif(
            output_dir,
            cfg.tracking.objective,
            solver_results,
            fps=gif_fps,
            system=system,
        )

    return {
        "output_dir": output_dir,
        "json_path": json_path,
        "plot_paths": plot_paths,
        "gif_path": gif_path,
        "solver_results": solver_results,
    }


def run_repeated_comparison(
    cfg: ExperimentConfig,
    solver_names,
    output_root: str | Path,
    repeats: int = 1,
    make_gif: bool = False,
    gif_fps: int = 3,
    vary_frozen_seed: bool = False,
):
    repeats = max(int(repeats), 1)

    if repeats == 1:
        return {
            "runs": [
                run_comparison(
                    cfg=cfg,
                    solver_names=solver_names,
                    output_root=output_root,
                    make_gif=make_gif,
                    gif_fps=gif_fps,
                )
            ],
            "aggregate_json_path": None,
            "aggregate_plot_path": None,
        }

    base_name = cfg.name
    base_output_dir = Path(output_root) / base_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for repeat_idx in range(repeats):
        print(f"\n\n######## REPEAT {repeat_idx+1}/{repeats} ########")
        run_cfg = deepcopy(cfg)
        run_cfg.turbulence.seed = cfg.turbulence.seed + repeat_idx

        if vary_frozen_seed:
            run_cfg.frozen_pinn.seed = cfg.frozen_pinn.seed + repeat_idx
        else:
            run_cfg.frozen_pinn.seed = cfg.frozen_pinn.seed

        run_cfg.name = f"{base_name}/run_{repeat_idx:03d}"
        result = run_comparison(
            cfg=run_cfg,
            solver_names=solver_names,
            output_root=output_root,
            make_gif=make_gif,
            gif_fps=gif_fps,
        )
        runs.append(result)

    aggregate = {}
    first_solver_results = runs[0]["solver_results"]
    for solver_name in first_solver_results:
        per_run_values = []
        per_run_runtime = []
        for run in runs:
            records = run["solver_results"][solver_name]["records"]
            per_run_values.append([float(r["display_metric"]) for r in records])
            per_run_runtime.append([float(r["runtime_sec"]) for r in records])
        arr = np.asarray(per_run_values, dtype=float)
        rt = np.asarray(per_run_runtime, dtype=float)
        aggregate[solver_name] = {
            "mean_by_interval": np.mean(arr, axis=0),
            "std_by_interval": np.std(arr, axis=0),
            "overall_mean": float(np.mean(arr)),
            "overall_std": float(np.std(arr)),
            "mean_runtime_sec": float(np.mean(rt)),
            "std_runtime_sec": float(np.std(rt)),
        }

    aggregate_payload = {
        "experiment_name": base_name,
        "objective": cfg.tracking.objective,
        "repeats": repeats,
        "base_config": cfg.to_dict(),
        "solvers": aggregate,
        "run_directories": [str(run["output_dir"]) for run in runs],
    }

    aggregate_json_path = base_output_dir / "aggregate_results.json"
    aggregate_json_path.write_text(
        json.dumps(_jsonable(aggregate_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    aggregate_plot_path = plot_aggregate_objective(
        base_output_dir,
        cfg.tracking.objective,
        aggregate,
    )

    return {
        "runs": runs,
        "aggregate_json_path": aggregate_json_path,
        "aggregate_plot_path": aggregate_plot_path,
    }
