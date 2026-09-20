#!/usr/bin/env python3
"""
Runtime-scaling benchmark for PAT propagation/control solvers.

This script answers the reviewer-style question:

    "Why not just use SSFM inside the PAT optimizer?"

It compares three objective-gradient query engines:

    frozen_pinn
        Frozen-PINN reduced propagation + steering sensitivity ODE.

    ssfm_fd
        Strong cached NumPy SSFM with central finite differences in theta.
        One objective-gradient query requires 1 base + 4 perturbed propagations.

    ssfm_autodiff
        Differentiable torch SSFM.
        One objective-gradient query uses one full-grid SSFM graph and backward.

Two sweeps are generated:

    1) grid-size sweep at a fixed number of optimization queries
    2) optimization-query sweep at a fixed grid size

The benchmark excludes one-time solver construction / neural-basis construction
from online latency and stores those setup times separately.

Examples
--------

Quick CPU run:

    python experiments/runtime_scaling.py \
        --objective power \
        --grid-sizes 64 128 256 \
        --query-counts 1 4 8 \
        --grid-for-query-sweep 128 \
        --ssfm-steps 32 \
        --repeats 3

Include larger grids:

    python experiments/runtime_scaling.py \
        --grid-sizes 64 128 256 512 \
        --query-counts 1 2 4 8 16 \
        --device auto \
        --dtype float64

Results
-------

    results/runtime_scaling/<name>/runtime_scaling.json
    results/runtime_scaling/<name>/runtime_vs_grid.png
    results/runtime_scaling/<name>/runtime_vs_queries.png
    results/runtime_scaling/<name>/quality_vs_latency.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ExperimentConfig,
    OpticalConfig,
    TurbulenceConfig,
    FrozenPINNConfig,
    BaselineConfig,
    TrackingConfig,
)

from system_model.optical_system import OpticalPATSystem
from solver.frozen_pinn_torch import TorchFrozenPINNSolver, resolve_torch_device
from solver.differentiable_ssfm import DifferentiableSSFMSolver


# =============================================================================
# Strong cached NumPy SSFM engine
# =============================================================================

class CachedNumpySSFM:
    """Cached SSFM propagation for a fixed atmospheric interval.

    Phase screens are prepared once and reused across steering queries, making
    this baseline stronger than repeatedly reconstructing Delta n for every
    finite-difference propagation.
    """

    def __init__(
        self,
        system: OpticalPATSystem,
        cfg: ExperimentConfig,
        *,
        dtype: str = "float64",
    ):
        self.system = system
        self.cfg = cfg

        if dtype == "float64":
            self.real_dtype = np.float64
            self.complex_dtype = np.complex128
        elif dtype == "float32":
            self.real_dtype = np.float32
            self.complex_dtype = np.complex64
        else:
            raise ValueError("dtype must be float32 or float64")

        self.X = np.asarray(
            system.X,
            dtype=self.real_dtype,
        )

        self.Y = np.asarray(
            system.Y,
            dtype=self.real_dtype,
        )

        self.gaussian = np.asarray(
            system.gaussian_beam(
                system.X,
                system.Y,
            ),
            dtype=self.real_dtype,
        )

        self.H_diff = np.asarray(
            system.H_diff,
            dtype=self.complex_dtype,
        )

        self._phase_cache = {}

    def prepare_interval(self, interval: int):
        if interval in self._phase_cache:
            return

        oc = self.cfg.optical
        tc = self.cfg.tracking

        phase_list = []

        for iz in range(tc.ssfm_steps):
            z = (iz + 0.5) * self.system.dz_truth

            dn = self.system.turbulence.eval(
                self.system.X,
                self.system.Y,
                z,
                interval,
            )

            phase = np.exp(
                (
                    1j * oc.k0 * dn
                    - oc.attenuation / 2.0
                )
                * self.system.dz_truth
                / 2.0
            )

            phase_list.append(
                np.asarray(
                    phase,
                    dtype=self.complex_dtype,
                )
            )

        self._phase_cache[interval] = phase_list

    def propagate(self, theta, interval: int):
        self.prepare_interval(interval)

        oc = self.cfg.optical

        U = (
            self.gaussian
            * np.exp(
                1j
                * oc.k0
                * (
                    theta[0] * self.X
                    + theta[1] * self.Y
                )
            )
        ).astype(self.complex_dtype)

        for half in self._phase_cache[interval]:
            U *= half
            U = np.fft.ifft2(
                np.fft.fft2(U) * self.H_diff
            ).astype(self.complex_dtype)
            U *= half

        return U

    def objective(
        self,
        theta,
        interval: int,
        target,
    ) -> float:
        U = self.propagate(theta, interval)

        return self.system.metric(
            U,
            target,
            self.cfg.tracking.objective,
        )

    def objective_and_fd_gradient(
        self,
        theta,
        interval: int,
        target,
        fd_step: float,
    ):
        base = self.objective(
            theta,
            interval,
            target,
        )

        grad = np.zeros(2, dtype=float)

        for d in range(2):
            step = np.zeros(2, dtype=float)
            step[d] = fd_step

            fp = self.objective(
                theta + step,
                interval,
                target,
            )

            fm = self.objective(
                theta - step,
                interval,
                target,
            )

            grad[d] = (
                fp - fm
            ) / (2.0 * fd_step)

        return base, grad


# =============================================================================
# Method adapters
# =============================================================================

class FrozenPINNQueryEngine:
    name = "Frozen-PINN"

    def __init__(
        self,
        system,
        cfg,
        *,
        device,
        dtype,
        rk_steps,
    ):
        t0 = time.perf_counter()
        self.solver = TorchFrozenPINNSolver(
            system, cfg, device=device, dtype=dtype, rk_steps=rk_steps
        )
        self.solver._sync()
        self.offline_setup_time_sec = time.perf_counter() - t0

        prep0 = time.perf_counter()
        self.solver.prepare_interval(0)
        self.solver._sync()
        self.interval_prepare_time_sec = time.perf_counter() - prep0
        self.setup_time_sec = self.offline_setup_time_sec + self.interval_prepare_time_sec

        self.system = system
        self.cfg = cfg

    def objective_and_gradient(
        self,
        theta,
        interval,
        target,
    ):
        return self.solver.objective_and_gradient(
            theta,
            interval,
            target,
        )

    def prepare_target(self, target):
        self.solver._sync()
        t0 = time.perf_counter()

        self.solver.prepare_target(
            target,
            self.cfg.tracking.objective,
        )

        self.solver._sync()

        target_prepare = time.perf_counter() - t0
        self.interval_prepare_time_sec += target_prepare
        self.setup_time_sec += target_prepare

    def synchronize(self):
        self.solver._sync()


class SSFMFDQueryEngine:
    name = "SSFM finite difference"

    def __init__(
        self,
        system,
        cfg,
        *,
        dtype,
        fd_step,
    ):
        t0 = time.perf_counter()
        self.engine = CachedNumpySSFM(system, cfg, dtype=dtype)
        self.offline_setup_time_sec = time.perf_counter() - t0

        prep0 = time.perf_counter()
        self.engine.prepare_interval(0)
        self.interval_prepare_time_sec = time.perf_counter() - prep0
        self.setup_time_sec = self.offline_setup_time_sec + self.interval_prepare_time_sec
        self.fd_step = float(fd_step)

    def objective_and_gradient(
        self,
        theta,
        interval,
        target,
    ):
        return self.engine.objective_and_fd_gradient(
            theta,
            interval,
            target,
            self.fd_step,
        )


class SSFMAutodiffQueryEngine:
    name = "Differentiable SSFM"

    def __init__(
        self,
        system,
        cfg,
        *,
        device,
        dtype,
    ):
        t0 = time.perf_counter()
        self.engine = DifferentiableSSFMSolver(
            system, cfg, device=device, dtype=dtype,
            num_iterations=1, line_search_steps=0
        )
        self.engine._sync()
        self.offline_setup_time_sec = time.perf_counter() - t0

        prep0 = time.perf_counter()
        self.engine.prepare_interval(0)
        self.engine._sync()
        self.interval_prepare_time_sec = time.perf_counter() - prep0
        self.setup_time_sec = self.offline_setup_time_sec + self.interval_prepare_time_sec

    def objective_and_gradient(
        self,
        theta,
        interval,
        target,
    ):
        return self.engine.objective_and_gradient(
            theta,
            interval,
            target,
        )

    def synchronize(self):
        self.engine._sync()


# =============================================================================
# benchmark helpers
# =============================================================================

METHOD_KEYS = {
    "frozen_pinn": "Frozen-PINN",
    "ssfm_fd": "SSFM finite difference",
    "ssfm_autodiff": "Differentiable SSFM",
}


def effective_ssfm_dtype(device: str, dtype: str) -> str:
    """Return the actual SSFM precision used by the benchmark.

    MPS is always forced to float32/complex64.
    """
    if device.lower() == "mps":
        return "float32"

    if device.lower() == "auto":
        try:
            import torch
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and not torch.cuda.is_available()
            ):
                return "float32"
        except Exception:
            pass

    return dtype


def make_config(args, grid_size: int) -> ExperimentConfig:
    return ExperimentConfig(
        name=args.name,
        optical=OpticalConfig(
            wavelength=args.wavelength,
            transmit_power=args.transmit_power,
            beam_waist=args.beam_waist,
            propagation_distance=args.distance,
            grid_size=grid_size,
            aperture_radius=args.aperture_radius,
            smf_mode_waist=args.smf_mode_waist,
        ),
        turbulence=TurbulenceConfig(
            delta_n_rms=args.delta_n_rms,
            num_modes=args.turbulence_modes,
            seed=args.seed,
        ),
        frozen_pinn=FrozenPINNConfig(
            sampler=args.frozen_sampler,
            backend="torch",
            device=(
                args.device
                if args.frozen_device == "same"
                else args.frozen_device
            ),
            dtype=(
                args.dtype
                if args.frozen_dtype == "same"
                else args.frozen_dtype
            ),
            rk_steps=args.frozen_rk_steps,
            hidden_width=args.hidden_width,
            collocation_side=args.collocation_side,
            boundary_side=args.boundary_side,
            svd_cutoff=args.svd_cutoff,
            pinv_rcond=args.pinv_rcond,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            num_opt_iterations=1,
            theta_step_max=args.theta_step,
            theta_max=args.theta_max,
            seed=args.seed,
        ),
        baselines=BaselineConfig(
            theta_max=args.theta_max,
            oracle_fd_step=args.fd_step,
            oracle_theta_step_max=args.theta_step,
        ),
        tracking=TrackingConfig(
            objective=args.objective,
            num_intervals=max(args.interval + 1, 2),
            target_x_amplitude=args.target_x_amplitude,
            target_y_amplitude=args.target_y_amplitude,
            ssfm_steps=args.ssfm_steps,
        ),
    )


def build_engines(
    system,
    cfg,
    args,
):
    engines = {}

    ssfm_dtype = effective_ssfm_dtype(
        args.device,
        args.dtype,
    )

    frozen_device = (
        args.device
        if args.frozen_device == "same"
        else args.frozen_device
    )

    frozen_dtype = (
        ssfm_dtype
        if args.frozen_dtype == "same"
        else args.frozen_dtype
    )

    # MPS is always float32 in the torch Frozen-PINN solver.
    if frozen_device == "mps":
        frozen_dtype = "float32"

    if frozen_device == "auto":
        resolved_frozen = resolve_torch_device("auto")

        if resolved_frozen.type == "mps":
            frozen_dtype = "float32"

    for method in args.methods:
        if method == "frozen_pinn":
            engines[method] = FrozenPINNQueryEngine(
                system,
                cfg,
                device=frozen_device,
                dtype=frozen_dtype,
                rk_steps=args.frozen_rk_steps,
            )

        elif method == "ssfm_fd":
            engines[method] = SSFMFDQueryEngine(
                system,
                cfg,
                dtype=ssfm_dtype,
                fd_step=args.fd_step,
            )

        elif method == "ssfm_autodiff":
            engines[method] = SSFMAutodiffQueryEngine(
                system,
                cfg,
                device=args.device,
                dtype=ssfm_dtype,
            )

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

    return engines


def normalized_update(
    theta,
    grad,
    *,
    step_size,
    theta_max,
):
    norm = np.linalg.norm(grad)

    if not np.isfinite(norm) or norm < 1e-30:
        return theta.copy()

    return np.clip(
        theta
        + step_size
        * grad
        / norm,
        -theta_max,
        theta_max,
    )


def run_query_loop(
    engine,
    query_count: int,
    *,
    interval: int,
    target,
    theta_step: float,
    theta_max: float,
):
    theta = np.zeros(2, dtype=float)
    metric = np.nan

    if hasattr(engine, "synchronize"):
        engine.synchronize()

    t0 = time.perf_counter()

    for _ in range(query_count):
        metric, grad = (
            engine.objective_and_gradient(
                theta,
                interval,
                target,
            )
        )

        theta = normalized_update(
            theta,
            grad,
            step_size=theta_step,
            theta_max=theta_max,
        )

    if hasattr(engine, "synchronize"):
        engine.synchronize()

    elapsed = time.perf_counter() - t0

    return {
        "elapsed_sec": float(elapsed),
        "theta": theta,
        "predicted_metric": float(metric),
    }


def truth_display_metric(
    system,
    theta,
    *,
    interval,
    target,
    objective,
):
    U = system.ssfm(
        theta,
        interval,
    )

    return float(
        system.display_metric(
            U,
            target,
            objective,
        )
    )


def benchmark_point(
    args,
    *,
    grid_size: int,
    query_count: int,
):
    cfg = make_config(
        args,
        grid_size,
    )

    system = OpticalPATSystem(cfg)
    target = system.target_position(
        args.interval
    )

    engines = build_engines(
        system,
        cfg,
        args,
    )

    # Frozen-PINN receiver operators are target-specific but can be prepared
    # once per PAT interval, just like SSFM channel phase screens.  Exclude
    # this one-time preparation from repeated steering-query latency.
    for engine in engines.values():
        if hasattr(engine, "prepare_target"):
            engine.prepare_target(target)

    point = {
        "grid_size": int(grid_size),
        "ssfm_steps": int(args.ssfm_steps),
        "query_count": int(query_count),
        "objective": args.objective,
        "methods": {},
    }

    for method_key, engine in engines.items():
        # Warmup is not included in timing.
        for _ in range(args.warmup):
            run_query_loop(
                engine,
                1,
                interval=args.interval,
                target=target,
                theta_step=args.theta_step,
                theta_max=args.theta_max,
            )

        timings = []
        final_thetas = []
        predicted_metrics = []

        for _ in range(args.repeats):
            result = run_query_loop(
                engine,
                query_count,
                interval=args.interval,
                target=target,
                theta_step=args.theta_step,
                theta_max=args.theta_max,
            )

            timings.append(
                result["elapsed_sec"]
            )

            final_thetas.append(
                result["theta"]
            )

            predicted_metrics.append(
                result["predicted_metric"]
            )

        timings = np.asarray(
            timings,
            dtype=float,
        )

        # Use the median-run theta for an independent SSFM truth quality check.
        median_index = int(
            np.argsort(timings)[
                len(timings) // 2
            ]
        )

        theta_eval = final_thetas[
            median_index
        ]

        truth_metric = truth_display_metric(
            system,
            theta_eval,
            interval=args.interval,
            target=target,
            objective=args.objective,
        )

        point["methods"][method_key] = {
            "label": METHOD_KEYS[method_key],
            "setup_time_sec": float(engine.setup_time_sec),
            "offline_setup_time_sec": float(getattr(engine, "offline_setup_time_sec", 0.0)),
            "interval_prepare_time_sec": float(getattr(engine, "interval_prepare_time_sec", 0.0)),
            "runtime_mean_sec": float(
                np.mean(timings)
            ),
            "runtime_std_sec": float(
                np.std(timings)
            ),
            "runtime_median_sec": float(
                np.median(timings)
            ),
            "runtime_min_sec": float(
                np.min(timings)
            ),
            "runtime_max_sec": float(
                np.max(timings)
            ),
            "runtime_samples_sec": timings.tolist(),
            "runtime_per_query_mean_sec": float(
                np.mean(timings)
                / max(query_count, 1)
            ),
            "online_interval_total_mean_sec": float(
                getattr(engine, "interval_prepare_time_sec", 0.0) + np.mean(timings)
            ),
            "theta_eval": np.asarray(
                theta_eval
            ).tolist(),
            "predicted_metric_median_run": float(
                predicted_metrics[
                    median_index
                ]
            ),
            "truth_display_metric": truth_metric,
        }

    return point


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(v)
            for v in value
        ]

    return value


# =============================================================================
# plots
# =============================================================================

def plot_runtime_vs_grid(
    output_dir: Path,
    grid_results,
):
    plt.figure(figsize=(7.5, 4.7))

    methods = list(
        grid_results[0]["methods"].keys()
    )

    for method in methods:
        x = np.asarray(
            [
                p["grid_size"]
                for p in grid_results
            ],
            dtype=float,
        )

        y = np.asarray(
            [
                p["methods"][method]
                ["runtime_mean_sec"]
                for p in grid_results
            ],
            dtype=float,
        )

        yerr = np.asarray(
            [
                p["methods"][method]
                ["runtime_std_sec"]
                for p in grid_results
            ],
            dtype=float,
        )

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=METHOD_KEYS[method],
        )

    plt.xlabel("SSFM transverse grid size N")
    plt.ylabel("Online latency [s]")
    plt.title("PAT optimization-query latency vs grid size")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    path = (
        output_dir
        / "runtime_vs_grid.png"
    )

    plt.savefig(path, dpi=190)
    plt.close()

    return path


def plot_runtime_vs_queries(
    output_dir: Path,
    query_results,
):
    plt.figure(figsize=(7.5, 4.7))

    methods = list(
        query_results[0]["methods"].keys()
    )

    for method in methods:
        x = np.asarray(
            [
                p["query_count"]
                for p in query_results
            ],
            dtype=float,
        )

        y = np.asarray(
            [
                p["methods"][method]
                ["runtime_mean_sec"]
                for p in query_results
            ],
            dtype=float,
        )

        yerr = np.asarray(
            [
                p["methods"][method]
                ["runtime_std_sec"]
                for p in query_results
            ],
            dtype=float,
        )

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=METHOD_KEYS[method],
        )

    plt.xlabel("Number of objective-gradient queries")
    plt.ylabel("Online latency [s]")
    plt.title("Repeated PAT-query latency")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()

    path = (
        output_dir
        / "runtime_vs_queries.png"
    )

    plt.savefig(path, dpi=190)
    plt.close()

    return path


def plot_interval_total_vs_grid(
    output_dir: Path,
    grid_results,
):
    plt.figure(figsize=(7.5, 4.7))
    methods = list(grid_results[0]["methods"].keys())
    for method in methods:
        x = np.asarray([p["grid_size"] for p in grid_results], dtype=float)
        y = np.asarray([p["methods"][method]["online_interval_total_mean_sec"] for p in grid_results], dtype=float)
        plt.plot(x, y, marker="o", label=METHOD_KEYS[method])
    plt.xlabel("SSFM transverse grid size N")
    plt.ylabel("Total online interval latency [s]")
    plt.title("Interval preparation + PAT-query latency")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "interval_total_vs_grid.png"
    plt.savefig(path, dpi=190)
    plt.close()
    return path


def plot_interval_total_vs_queries(
    output_dir: Path,
    query_results,
):
    plt.figure(figsize=(7.5, 4.7))
    methods = list(query_results[0]["methods"].keys())
    for method in methods:
        x = np.asarray([p["query_count"] for p in query_results], dtype=float)
        y = np.asarray([p["methods"][method]["online_interval_total_mean_sec"] for p in query_results], dtype=float)
        plt.plot(x, y, marker="o", label=METHOD_KEYS[method])
    plt.xlabel("Number of objective-gradient queries")
    plt.ylabel("Total online interval latency [s]")
    plt.title("Interval preparation + repeated PAT-query latency")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "interval_total_vs_queries.png"
    plt.savefig(path, dpi=190)
    plt.close()
    return path


def plot_quality_vs_latency(
    output_dir: Path,
    query_results,
    objective: str,
):
    plt.figure(figsize=(7.5, 4.7))

    methods = list(
        query_results[0]["methods"].keys()
    )

    for method in methods:
        latency = np.asarray(
            [
                p["methods"][method]
                ["runtime_mean_sec"]
                for p in query_results
            ],
            dtype=float,
        )

        quality = np.asarray(
            [
                p["methods"][method]
                ["truth_display_metric"]
                for p in query_results
            ],
            dtype=float,
        )

        plt.plot(
            latency,
            quality,
            marker="o",
            label=METHOD_KEYS[method],
        )

    plt.xlabel("Online latency [s]")

    if objective == "power":
        plt.ylabel("Actual receive power")
    elif objective == "coupling":
        plt.ylabel("Actual SMF coupling efficiency")
    else:
        plt.ylabel("Actual centroid error [mm]")

    plt.title("Control quality vs online latency")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()

    path = (
        output_dir
        / "quality_vs_latency.png"
    )

    plt.savefig(path, dpi=190)
    plt.close()

    return path


# =============================================================================
# crossover summary
# =============================================================================

def crossover_summary(
    grid_results,
    query_results,
):
    summary = {}

    def first_faster(
        results,
        x_key,
        competitor,
    ):
        for point in results:
            if (
                "frozen_pinn"
                not in point["methods"]
                or competitor
                not in point["methods"]
            ):
                continue

            frozen = (
                point["methods"]["frozen_pinn"]
                ["runtime_mean_sec"]
            )

            other = (
                point["methods"][competitor]
                ["runtime_mean_sec"]
            )

            if frozen < other:
                return {
                    x_key: point[x_key],
                    "frozen_runtime_sec": frozen,
                    "competitor_runtime_sec": other,
                    "speedup": other / frozen,
                }

        return None

    summary[
        "first_grid_frozen_faster_than_autodiff"
    ] = first_faster(
        grid_results,
        "grid_size",
        "ssfm_autodiff",
    )

    summary[
        "first_grid_frozen_faster_than_fd"
    ] = first_faster(
        grid_results,
        "grid_size",
        "ssfm_fd",
    )

    summary[
        "first_query_count_frozen_faster_than_autodiff"
    ] = first_faster(
        query_results,
        "query_count",
        "ssfm_autodiff",
    )

    summary[
        "first_query_count_frozen_faster_than_fd"
    ] = first_faster(
        query_results,
        "query_count",
        "ssfm_fd",
    )

    return summary


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Benchmark Frozen-PINN vs finite-difference SSFM "
            "and differentiable SSFM for repeated PAT queries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--name",
        default="runtime_scaling",
    )

    p.add_argument(
        "--objective",
        choices=[
            "power",
            "coupling",
            "centroid",
        ],
        default="power",
    )

    p.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "frozen_pinn",
            "ssfm_fd",
            "ssfm_autodiff",
        ],
        default=[
            "frozen_pinn",
            "ssfm_fd",
            "ssfm_autodiff",
        ],
    )

    p.add_argument(
        "--grid-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256],
    )

    p.add_argument(
        "--query-counts",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )

    p.add_argument(
        "--grid-for-query-sweep",
        type=int,
        default=128,
    )

    p.add_argument(
        "--queries-for-grid-sweep",
        type=int,
        default=4,
    )

    p.add_argument(
        "--ssfm-steps",
        type=int,
        default=32,
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=1,
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    p.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
        default="auto",
    )

    p.add_argument(
        "--dtype",
        choices=[
            "float64",
            "float32",
        ],
        default="float64",
        help=(
            "SSFM precision. MPS automatically uses float32/complex64."
        ),
    )

    p.add_argument(
        "--frozen-device",
        choices=[
            "same",
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
        default="same",
        help=(
            "Frozen-PINN torch device. 'same' uses --device, so both "
            "Frozen-PINN and differentiable SSFM run on the same accelerator."
        ),
    )

    p.add_argument(
        "--frozen-dtype",
        choices=[
            "same",
            "float64",
            "float32",
        ],
        default="same",
        help=(
            "Frozen-PINN precision. 'same' matches the effective SSFM precision. "
            "MPS is always forced to float32."
        ),
    )

    p.add_argument(
        "--frozen-rk-steps",
        type=int,
        default=64,
        help="Fixed RK4 steps for torch Frozen-PINN propagation.",
    )

    p.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help=(
            "If >0, set torch CPU thread count. "
            "0 leaves the PyTorch default unchanged."
        ),
    )

    # optical
    p.add_argument(
        "--wavelength",
        type=float,
        default=1550e-9,
    )

    p.add_argument(
        "--distance",
        type=float,
        default=20.0,
    )

    p.add_argument(
        "--transmit-power",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--beam-waist",
        type=float,
        default=2.5e-3,
    )

    p.add_argument(
        "--aperture-radius",
        type=float,
        default=2.4e-3,
    )

    p.add_argument(
        "--smf-mode-waist",
        type=float,
        default=1.7e-3,
    )

    # turbulence
    p.add_argument(
        "--delta-n-rms",
        type=float,
        default=8e-9,
    )

    p.add_argument(
        "--turbulence-modes",
        type=int,
        default=10,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    p.add_argument(
        "--interval",
        type=int,
        default=1,
    )

    # PAT update
    p.add_argument(
        "--theta-step",
        type=float,
        default=30e-6,
    )

    p.add_argument(
        "--theta-max",
        type=float,
        default=250e-6,
    )

    p.add_argument(
        "--fd-step",
        type=float,
        default=2e-6,
    )

    # Frozen-PINN
    p.add_argument(
        "--frozen-sampler",
        choices=["elm", "swim"],
        default="elm",
    )

    p.add_argument(
        "--hidden-width",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--collocation-side",
        type=int,
        default=24,
    )

    p.add_argument(
        "--boundary-side",
        type=int,
        default=16,
    )

    p.add_argument(
        "--svd-cutoff",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--pinv-rcond",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--ode-rtol",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--ode-atol",
        type=float,
        default=1e-8,
    )

    # target
    p.add_argument(
        "--target-x-amplitude",
        type=float,
        default=2.4e-3,
    )

    p.add_argument(
        "--target-y-amplitude",
        type=float,
        default=1.8e-3,
    )

    p.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT
            / "results"
            / "runtime_scaling"
        ),
    )

    return p


def main():
    args = build_parser().parse_args()

    if args.torch_threads > 0:
        import torch
        torch.set_num_threads(
            args.torch_threads
        )

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be >= 1"
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup must be >= 0"
        )

    if any(v < 1 for v in args.grid_sizes):
        raise ValueError(
            "all --grid-sizes must be >= 1"
        )

    if any(v < 1 for v in args.query_counts):
        raise ValueError(
            "all --query-counts must be >= 1"
        )

    output_dir = (
        Path(args.output_root)
        / args.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Runtime benchmark")
    print(f"  objective          : {args.objective}")
    print(f"  methods            : {', '.join(args.methods)}")
    print(f"  grid sizes         : {args.grid_sizes}")
    print(f"  query counts       : {args.query_counts}")
    print(f"  SSFM steps         : {args.ssfm_steps}")
    print(f"  repeats            : {args.repeats}")
    effective_dtype = effective_ssfm_dtype(
        args.device,
        args.dtype,
    )

    print(f"  device             : {args.device}")
    print(f"  requested dtype    : {args.dtype}")
    print(f"  effective SSFM dtype: {effective_dtype}")
    print(f"  Frozen device      : {args.frozen_device}")
    print(f"  Frozen dtype       : {args.frozen_dtype}")
    print(f"  Frozen RK4 steps   : {args.frozen_rk_steps}")

    grid_results = []

    for grid in args.grid_sizes:
        print(
            f"\n[grid sweep] N={grid}, "
            f"queries={args.queries_for_grid_sweep}"
        )

        point = benchmark_point(
            args,
            grid_size=grid,
            query_count=args.queries_for_grid_sweep,
        )

        grid_results.append(point)

        for method, payload in point["methods"].items():
            print(
                f"  {METHOD_KEYS[method]:24s} "
                f"{payload['runtime_mean_sec']:.6f} s "
                f"(± {payload['runtime_std_sec']:.6f})"
            )

    query_results = []

    for count in args.query_counts:
        print(
            f"\n[query sweep] N={args.grid_for_query_sweep}, "
            f"queries={count}"
        )

        point = benchmark_point(
            args,
            grid_size=args.grid_for_query_sweep,
            query_count=count,
        )

        query_results.append(point)

        for method, payload in point["methods"].items():
            print(
                f"  {METHOD_KEYS[method]:24s} "
                f"{payload['runtime_mean_sec']:.6f} s "
                f"(± {payload['runtime_std_sec']:.6f})"
            )

    crossovers = crossover_summary(
        grid_results,
        query_results,
    )

    payload = {
        "name": args.name,
        "objective": args.objective,
        "methods": args.methods,
        "device": args.device,
        "requested_dtype": args.dtype,
        "effective_ssfm_dtype": effective_dtype,
        "frozen_device": args.frozen_device,
        "frozen_dtype": args.frozen_dtype,
        "frozen_rk_steps": args.frozen_rk_steps,
        "ssfm_steps": args.ssfm_steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "grid_sweep": grid_results,
        "query_sweep": query_results,
        "crossovers": crossovers,
        "benchmark_note": (
            "Repeated-query latency excludes interval preparation, while "
            "online_interval_total_mean_sec includes interval-specific transition/"
            "phase-screen/receiver-operator preparation. Offline basis/static tensor "
            "setup is reported separately. "
            "SSFM finite differences use one base plus four perturbed "
            "full-grid propagations per objective-gradient query. "
            "Differentiable SSFM uses one full-grid forward/backward graph. "
            "Frozen-PINN precomputes the interval state-transition matrix "
            "and target-specific reduced receiver operators once, then each "
            "steering query uses only reduced matrix applications and quadratic "
            "forms on the selected CPU/CUDA/MPS device."
        ),
    }

    json_path = (
        output_dir
        / "runtime_scaling.json"
    )

    json_path.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plot_paths = [
        plot_runtime_vs_grid(output_dir, grid_results),
        plot_runtime_vs_queries(output_dir, query_results),
        plot_interval_total_vs_grid(output_dir, grid_results),
        plot_interval_total_vs_queries(output_dir, query_results),
        plot_quality_vs_latency(
            output_dir,
            query_results,
            args.objective,
        ),
    ]

    print("\nCrossovers")
    print(
        json.dumps(
            crossovers,
            indent=2,
        )
    )

    print("\nSaved")
    print(f"  JSON : {json_path}")

    for path in plot_paths:
        print(f"  plot : {path}")


if __name__ == "__main__":
    main()
