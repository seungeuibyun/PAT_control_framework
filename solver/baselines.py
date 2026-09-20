from __future__ import annotations

import time
import numpy as np

from config.settings import ExperimentConfig
from system_model.optical_system import OpticalPATSystem
from solver.optimization import project_pat_command, projected_gradient_ascent
from solver.differentiable_ssfm import DifferentiableSSFMSolver


class NoControlBaseline:
    name = "No control"

    def __init__(self, system: OpticalPATSystem, cfg: ExperimentConfig):
        self.system = system
        self.cfg = cfg

    def solve(self, interval, target, theta_prev, history):
        return {
            "theta": np.zeros(2, dtype=float),
            "runtime_sec": 0.0,
            "predicted_metric": None,
            "diagnostics": {},
        }


class PIDBaseline:
    """Conventional geometric PID using the measured beam centroid."""
    name = "PID"

    def __init__(self, system: OpticalPATSystem, cfg: ExperimentConfig):
        self.system = system
        self.cfg = cfg
        self.integral = np.zeros(2, dtype=float)
        self.prev_error = np.zeros(2, dtype=float)

    def solve(self, interval, target, theta_prev, history):
        bc = self.cfg.baselines
        oc = self.cfg.optical
        t0 = time.perf_counter()
        U = self.system.ssfm(theta_prev, interval)
        p = self.system.centroid(U, target)
        error = np.asarray(target) - p
        dt = max(self.cfg.tracking.control_interval_sec, 1e-30)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        delta_theta = (
            bc.pid_kp * error + bc.pid_ki * self.integral + bc.pid_kd * derivative
        ) / oc.propagation_distance
        theta = project_pat_command(
            np.asarray(theta_prev) + delta_theta,
            theta_prev,
            self.cfg.tracking,
        )
        self.prev_error = error
        return {
            "theta": theta,
            "runtime_sec": time.perf_counter() - t0,
            "predicted_metric": None,
            "diagnostics": {
                "measured_centroid_x": float(p[0]),
                "measured_centroid_y": float(p[1]),
            },
        }


class LinearMPCBaseline:
    """One-step geometric MPC under p_next ≈ p + L delta_theta."""
    name = "Linear MPC"

    def __init__(self, system: OpticalPATSystem, cfg: ExperimentConfig):
        self.system = system
        self.cfg = cfg

    def solve(self, interval, target, theta_prev, history):
        bc = self.cfg.baselines
        oc = self.cfg.optical
        t0 = time.perf_counter()
        U = self.system.ssfm(theta_prev, interval)
        p = self.system.centroid(U, target)
        error = np.asarray(target) - p
        gain = oc.propagation_distance / (oc.propagation_distance**2 + bc.mpc_rho)
        delta_theta = gain * error
        theta = project_pat_command(
            np.asarray(theta_prev) + delta_theta,
            theta_prev,
            self.cfg.tracking,
        )
        return {
            "theta": theta,
            "runtime_sec": time.perf_counter() - t0,
            "predicted_metric": None,
            "diagnostics": {
                "measured_centroid_x": float(p[0]),
                "measured_centroid_y": float(p[1]),
            },
        }



class DifferentiableSSFMOracleBaseline:
    """Full-grid differentiable SSFM using the same projected PAT optimizer.

    This is the default `ssfm_oracle` baseline.  It runs on the device selected
    by BaselineConfig.ssfm_device and uses torch autograd for dQ/dtheta.
    """

    name = "Diff-SSFM oracle"

    def __init__(
        self,
        system: OpticalPATSystem,
        cfg: ExperimentConfig,
    ):
        self.system = system
        self.cfg = cfg

        self.engine = DifferentiableSSFMSolver(
            system,
            cfg,
            device=cfg.baselines.ssfm_device,
            dtype=cfg.baselines.ssfm_dtype,
            num_iterations=1,
            line_search_steps=0,
        )

    def solve(
        self,
        interval,
        target,
        theta_prev,
        history,
    ):
        self.engine._sync()
        total_t0 = time.perf_counter()
        prep_start = total_t0

        # Phase-screen preparation is an online interval update and is included
        # in total controller latency.
        self.engine.prepare_interval(interval)

        self.engine._sync()
        phase_prepare_time = time.perf_counter() - prep_start
        query_t0 = time.perf_counter()

        def objective_and_gradient(theta):
            return self.engine.objective_and_gradient(
                theta,
                interval,
                target,
            )

        def objective_only(theta):
            return self.engine.objective_only(
                theta,
                interval,
                target,
            )

        initial_theta = None
        if self.cfg.tracking.objective == "centroid":
            initial_theta = np.asarray(target, dtype=float) / max(self.system.cfg.optical.propagation_distance, 1e-30)

        theta, metric, opt_diag = (
            projected_gradient_ascent(
                theta_prev,
                objective_and_gradient,
                objective_only,
                self.cfg.tracking,
                initial_theta=initial_theta,
            )
        )

        self.engine._sync()
        query_runtime = time.perf_counter() - query_t0
        total_runtime = time.perf_counter() - total_t0

        diagnostics = {
            "device": str(
                self.engine.device
            ),
            "requested_dtype": (
                self.engine.requested_dtype
            ),
            "effective_dtype": (
                self.engine.effective_dtype
            ),
            "phase_prepare_time_sec": float(
                phase_prepare_time
            ),
            "gradient_backend": "torch_autograd",
            "query_runtime_sec": float(query_runtime),
            "online_total_runtime_sec": float(total_runtime),
        }

        diagnostics.update(
            opt_diag
        )

        return {
            "theta": theta,
            "runtime_sec": total_runtime,
            "predicted_metric": metric,
            "diagnostics": diagnostics,
        }


class NumPySSFMOracleBaseline:
    """Strong numerical-physics baseline using SSFM inside the same optimizer."""
    name = "SSFM oracle (CPU-FD)"

    def __init__(self, system: OpticalPATSystem, cfg: ExperimentConfig):
        self.system = system
        self.cfg = cfg

    def _objective(self, theta, interval, target):
        U = self.system.ssfm(theta, interval)

        return self.system.metric(
            U,
            target,
            self.cfg.tracking.objective,
        )

    def _finite_difference_gradient(self, theta, interval, target):
        h = self.cfg.baselines.oracle_fd_step

        grad = np.zeros(2, dtype=float)

        for d in range(2):
            step = np.zeros(2)
            step[d] = h

            fp = self._objective(
                theta + step,
                interval,
                target,
            )

            fm = self._objective(
                theta - step,
                interval,
                target,
            )

            grad[d] = (
                fp - fm
            ) / (2.0 * h)

        return grad

    def solve(self, interval, target, theta_prev, history):
        t0 = time.perf_counter()

        def objective_and_gradient(theta):
            metric = self._objective(
                theta,
                interval,
                target,
            )

            grad = self._finite_difference_gradient(
                theta,
                interval,
                target,
            )

            return metric, grad

        def objective_only(theta):
            return self._objective(
                theta,
                interval,
                target,
            )

        initial_theta = None
        if self.cfg.tracking.objective == "centroid":
            initial_theta = np.asarray(target, dtype=float) / max(self.system.cfg.optical.propagation_distance, 1e-30)

        theta, metric, opt_diag = (
            projected_gradient_ascent(
                theta_prev,
                objective_and_gradient,
                objective_only,
                self.cfg.tracking,
                initial_theta=initial_theta,
            )
        )

        return {
            "theta": theta,
            "runtime_sec": time.perf_counter() - t0,
            "predicted_metric": metric,
            "diagnostics": opt_diag,
        }


def make_baseline(name: str, system: OpticalPATSystem, cfg: ExperimentConfig):
    key = name.lower().strip()
    if key in {"none", "no_control", "no control"}:
        return NoControlBaseline(system, cfg)
    if key == "pid":
        return PIDBaseline(system, cfg)
    if key in {"mpc", "linear_mpc", "linear mpc"}:
        return LinearMPCBaseline(system, cfg)
    if key in {
        "ssfm_oracle",
        "oracle",
        "ssfm oracle",
        "diff_ssfm",
        "diff-ssfm",
        "differentiable_ssfm",
    }:
        return DifferentiableSSFMOracleBaseline(
            system,
            cfg,
        )

    if key in {
        "ssfm_oracle_cpu",
        "ssfm_cpu",
        "numpy_ssfm_oracle",
    }:
        return NumPySSFMOracleBaseline(
            system,
            cfg,
        )

    raise ValueError(f"Unknown baseline: {name}")
