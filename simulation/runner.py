from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
from config.settings import ExperimentConfig
from system_model.optical_system import OpticalPATSystem
from solver.frozen_pinn import FrozenPINNSolver
from solver.baselines import make_baseline
from simulation.results import save_results_json, _jsonable
from simulation.plotting import (plot_objective_comparison, plot_runtime_comparison,
    plot_centroid_trajectories, plot_receiver_snapshots, save_comparison_gif, plot_aggregate_objective)


def build_solvers(system, cfg, names):
    solvers = []
    for name in names:
        if name == "frozen_pinn":
            if cfg.frozen_pinn.backend in {"torch", "auto"}:
                from solver.frozen_pinn_torch import TorchFrozenPINNSolver
                solver = TorchFrozenPINNSolver(system, cfg)
            elif cfg.frozen_pinn.backend == "scipy":
                solver = FrozenPINNSolver(system, cfg)
            else:
                raise ValueError("frozen backend must be scipy, torch or auto")
        else:
            solver = make_baseline(name, system, cfg)
        solvers.append(solver)
    if not solvers or len({s.name for s in solvers}) != len(solvers):
        raise ValueError("Select at least one solver, without duplicates")
    return solvers


def _solver_summary(records):
    power = np.array([r["communication_power_w"] for r in records])
    runtime = np.array([r["runtime_sec"] for r in records])
    return dict(mean_objective=float(power.mean()), min_objective=float(power.min()),
        max_objective=float(power.max()), mean_power_w=float(power.mean()),
        mean_runtime_sec=float(runtime.mean()), total_runtime_sec=float(runtime.sum()),
        mean_psd_error_um=float(np.nanmean([r["psd_error_um"] for r in records])))


def run_comparison(cfg: ExperimentConfig, solver_names, output_root, make_gif=True, gif_fps=8):
    cfg.validate()
    output_dir = Path(output_root) / cfg.name
    if (output_dir / "results.json").exists():
        raise FileExistsError(f"Results already exist: {output_dir}. Select a new --name.")
    system = OpticalPATSystem(cfg)
    solvers = build_solvers(system, cfg, solver_names)
    results = {s.name: {"records": []} for s in solvers}
    previous = {s.name: np.asarray(cfg.tracking.initial_theta, dtype=float) for s in solvers}
    gif_frames = [] if make_gif else None
    gif_focus = "Frozen-PINN" if "Frozen-PINN" in results else next(iter(results))
    for k in range(cfg.tracking.num_intervals):
        # Plant simulation is outside controller latency. Every model-based
        # controller separately pays for its own atmospheric prediction.
        truth = system.atmospheric_field(k)
        reduced = system.reduce_field(truth)
        for solver in solvers:
            prev = previous[solver.name]
            measured, valid = system.measure_psd(reduced, prev, k)
            records = results[solver.name]["records"]
            solved = solver.solve(k, system.reference_centroid, prev, records,
                                  measurement=measured, measurement_valid=valid)
            theta = np.asarray(solved["theta"])
            # Check physical feasibility independently of the optimizer.
            if (np.any(abs(theta) > np.asarray(cfg.tracking.theta_max) + 1e-12)
                or np.any(abs(theta-prev) > np.asarray(cfg.tracking.theta_slew_max) + 1e-12)
                or not np.allclose(theta / cfg.tracking.theta_quantization,
                                   np.round(theta / cfg.tracking.theta_quantization), atol=1e-8, rtol=0)):
                raise AssertionError(f"Infeasible command from {solver.name}")
            power = system.power(system.detector_field(reduced, theta))
            sensing = system.sensing_vector(reduced, theta)
            diagnostics = solved.get("diagnostics", {})
            if isinstance(solver, FrozenPINNSolver):
                predicted = solver._reduced
                diagnostics["receiver_field_relative_error"] = float(
                    np.linalg.norm(predicted-reduced) / max(np.linalg.norm(reduced), 1e-30))
                diagnostics["receiver_intensity_relative_error"] = float(
                    np.linalg.norm(abs(predicted)**2-abs(reduced)**2)
                    / max(np.linalg.norm(abs(reduced)**2), 1e-30))
                diagnostics["power_prediction_relative_error"] = float(
                    abs(solved["predicted_metric"]-power) / max(power, 1e-30))
            record = dict(interval=k, time_sec=system.time_at_interval(k),
                target=system.reference_centroid.copy(), theta=theta, theta_previous=prev,
                tx_angular_error=system.tx_angle(k), objective_value=power, display_metric=power,
                communication_power_w=power, actual_centroid=sensing[:2], psd_power_w=sensing[2],
                psd_measurement=measured, psd_measurement_valid=valid,
                psd_error_um=float(np.linalg.norm(sensing[:2]-system.reference_centroid)*1e6),
                runtime_sec=solved["runtime_sec"], predicted_metric=solved.get("predicted_metric"),
                diagnostics=diagnostics)
            if "predicted_centroid" in diagnostics:
                record["predicted_centroid"] = diagnostics["predicted_centroid"]
            records.append(record)
            previous[solver.name] = theta
            print(f"[{k+1:03d}/{cfg.tracking.num_intervals:03d}] {solver.name:21s} "
                  f"P_D={power*1e3:.6f} mW  theta=({theta[0]*1e6:+.1f},{theta[1]*1e6:+.1f}) urad "
                  f"runtime={solved['runtime_sec']:.4f}s", flush=True)
        if make_gif:
            # Reuse the plant field already computed for this interval. GIF
            # rendering never repeats the expensive atmospheric propagation.
            command = results[gif_focus]["records"][-1]["theta"]
            gif_frames.append((abs(system.detector_field(reduced, command, view=True))**2).astype(np.float32))
    for payload in results.values():
        payload["summary"] = _solver_summary(payload["records"])
        approximation = [r["diagnostics"]["power_prediction_relative_error"]
                         for r in payload["records"] if "power_prediction_relative_error" in r["diagnostics"]]
        if approximation:
            payload["summary"]["mean_power_prediction_relative_error"] = float(np.mean(approximation))
            payload["summary"]["max_power_prediction_relative_error"] = float(np.max(approximation))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"objective": "P_D,k: communication-detector collected power [W]",
        "calibration": system.calibration(),
        "runtime_scope": "controller atmospheric prediction + receiver queries + command selection; excludes simulated PSD acquisition, plant truth and plotting",
        "offline_basis_setup_excluded": True,
        "offline_operator_setup_excluded": True,
        "discretization": {"ssfm_grid_shape": [cfg.optical.grid_size]*2,
            "pinn_feature_count": cfg.frozen_pinn.hidden_width,
            "pinn_collocation_shape": [cfg.frozen_pinn.collocation_side]*2},
        "atmosphere": "HV profile, finite random Fourier modified-von-Karman Markov layers",
        "resolution_status": "Table II nominal setting; convergence still requires validation" if cfg.preset == "paper" else "reduced-resolution demonstration, not a paper-scale reproduction"}
    json_path = save_results_json(output_dir, cfg, results, metadata=metadata)
    plot_paths, gif_path = _render_outputs(output_dir, results, system, make_gif, gif_fps,
                                         reduced=reduced, frames=gif_frames)
    return dict(output_dir=output_dir, json_path=json_path, plot_paths=plot_paths,
                gif_path=gif_path, solver_results=results)


def _render_outputs(output_dir, results, system, make_gif, gif_fps, *, reduced=None, frames=None):
    print("Rendering figures" + (" and detector XY GIF..." if make_gif else "..."), flush=True)
    plot_paths = [plot_objective_comparison(output_dir, "power", results),
                  output_dir / "objective_comparison_all.png",
                  plot_runtime_comparison(output_dir, results),
                  plot_centroid_trajectories(output_dir, results),
                  plot_receiver_snapshots(output_dir, results, system, reduced=reduced)]
    if "No control" in results:
        plot_paths.append(output_dir / "power_gain_comparison.png")
    gif_path = save_comparison_gif(output_dir, "power", results, fps=gif_fps,
                                   system=system, frames=frames) if make_gif else None
    manifest = dict(plots=[p.name for p in plot_paths], gif_requested=make_gif,
                    gif=gif_path.name if gif_path else None,
                    gif_frames=len(next(iter(results.values()))["records"]) if gif_path else 0,
                    fps=gif_fps if gif_path else None)
    (output_dir/"output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return plot_paths, gif_path


def replot_results(path, make_gif=True, gif_fps=8):
    """Replay saved commands/channels; numerical results and timings are untouched."""
    path = Path(path)
    json_path = path/"results.json" if path.is_dir() else path
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig.from_dict(payload["config"])
    system = OpticalPATSystem(cfg)
    results = payload["solvers"]
    counts = {len(p["records"]) for p in results.values()}
    if counts != {cfg.tracking.num_intervals}:
        raise ValueError("Saved record counts do not match the experiment config")
    plots, gif = _render_outputs(json_path.parent, results, system, make_gif, gif_fps)
    return dict(output_dir=json_path.parent, json_path=json_path, plot_paths=plots, gif_path=gif)


def run_repeated_comparison(cfg, solver_names, output_root, repeats=1, make_gif=True,
                            gif_fps=8, vary_frozen_seed=False, feature_seeds=1):
    if repeats < 1 or feature_seeds < 1:
        raise ValueError("repeats and feature_seeds must be positive")
    base = Path(output_root) / cfg.name
    if (base / "results.json").exists() or (base / "aggregate_results.json").exists():
        raise FileExistsError(f"Existing results at {base}; choose a new --name")
    total = repeats * feature_seeds
    if total == 1:
        return dict(runs=[run_comparison(cfg, solver_names, output_root, make_gif, gif_fps)],
                    aggregate_json_path=None, aggregate_plot_path=None)
    for idx in range(total):
        if (base / f"run_{idx:03d}" / "results.json").exists():
            raise FileExistsError(f"Existing run at {base}; choose a new --name")
    runs = []
    # Cartesian product: same channel realizations across independent feature
    # seeds, rather than confounding both seeds along a diagonal.
    for channel in range(repeats):
        for feature in range(feature_seeds):
            run_cfg = deepcopy(cfg)
            run_cfg.turbulence.seed += channel
            run_cfg.frozen_pinn.seed += feature + (channel if vary_frozen_seed else 0)
            run_cfg.name = f"{cfg.name}/run_{len(runs):03d}"
            runs.append(run_comparison(run_cfg, solver_names, output_root, make_gif, gif_fps))
    aggregate = {}
    for name in runs[0]["solver_results"]:
        power = np.asarray([[r["display_metric"] for r in run["solver_results"][name]["records"]] for run in runs])
        runtime = np.asarray([[r["runtime_sec"] for r in run["solver_results"][name]["records"]] for run in runs])
        aggregate[name] = dict(mean_by_interval=power.mean(axis=0), std_by_interval=power.std(axis=0),
            overall_mean=float(power.mean()), overall_std=float(power.std()),
            mean_runtime_sec=float(runtime.mean()), std_runtime_sec=float(runtime.std()))
    path = base / "aggregate_results.json"
    path.write_text(json.dumps(_jsonable(dict(experiment_name=cfg.name, objective="power",
        repeats=repeats, feature_seeds=feature_seeds, independent_channel_seeds=repeats,
        feature_seed_policy="cartesian" if not vary_frozen_seed else "shifted_cartesian",
        uncertainty="descriptive SD across channel/feature runs; feature seeds share channels",
        base_config=cfg.to_dict(), solvers=aggregate, run_directories=[str(r["output_dir"]) for r in runs])),
        indent=2, allow_nan=False), encoding="utf-8")
    return dict(runs=runs, aggregate_json_path=path,
                aggregate_plot_path=plot_aggregate_objective(base, "power", aggregate))
