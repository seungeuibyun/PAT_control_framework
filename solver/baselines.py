"""PSD-only controllers and SSFM model-based power baselines."""
from __future__ import annotations
import time
import numpy as np
from solver.optimization import quantize_pat_command, projected_gradient_ascent, sensing_initial_command


class NoControlBaseline:
    name = "No control"
    def __init__(self, system, cfg):
        self.system, self.cfg = system, cfg
    def solve(self, interval, target, theta_prev, history, **kwargs):
        return dict(theta=np.asarray(self.cfg.tracking.initial_theta), runtime_sec=0.0,
                    predicted_metric=None, diagnostics={})


class PIDBaseline:
    name = "PID"
    def __init__(self, system, cfg):
        self.system, self.cfg = system, cfg
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

    def solve(self, interval, target, theta_prev, history, measurement=None, measurement_valid=True):
        start = time.perf_counter()
        if not measurement_valid:
            return dict(theta=np.asarray(theta_prev), runtime_sec=time.perf_counter()-start,
                        predicted_metric=None, diagnostics={"sensing_valid": False})
        b, t = self.cfg.baselines, self.cfg.tracking
        error = np.linalg.pinv(self.system.psd_jacobian) @ (self.system.reference_centroid - measurement[:2])
        trial_integral = self.integral + error * t.control_interval_sec
        delta = b.pid_kp * error + b.pid_ki * trial_integral + b.pid_kd * (error - self.prev_error) / t.control_interval_sec
        raw = np.asarray(theta_prev) + delta
        theta = quantize_pat_command(raw, theta_prev, t)
        # Conditional integration prevents windup while range/slew saturated.
        unsaturated = abs(raw - theta) <= t.theta_quantization / 2 + 1e-14
        self.integral[unsaturated] = trial_integral[unsaturated]
        self.prev_error = error
        return dict(theta=theta, runtime_sec=time.perf_counter()-start, predicted_metric=None,
                    diagnostics={"sensing_valid": True})


class LinearMPCBaseline:
    name = "Linear MPC"
    def __init__(self, system, cfg):
        self.system, self.cfg = system, cfg
    def solve(self, interval, target, theta_prev, history, measurement=None, measurement_valid=True):
        start = time.perf_counter()
        delta = np.zeros(2)
        if measurement_valid:
            delta = np.linalg.pinv(self.system.psd_jacobian) @ (self.system.reference_centroid - measurement[:2])
            delta /= 1 + self.cfg.baselines.mpc_rho
        theta = quantize_pat_command(np.asarray(theta_prev) + delta, theta_prev, self.cfg.tracking)
        return dict(theta=theta, runtime_sec=time.perf_counter()-start, predicted_metric=None,
                    diagnostics={"sensing_valid": measurement_valid})


class NumPySSFMOracleBaseline:
    name = "SSFM oracle (CPU-FD)"
    def __init__(self, system, cfg):
        self.system, self.cfg = system, cfg
        self._interval = None
    def prepare_interval(self, interval):
        if self._interval != interval:
            self._reduced = self.system.reduce_field(self.system.atmospheric_field(interval, cache=False))
            self._interval = interval
    def objective_only(self, theta, interval, target=None):
        self.prepare_interval(interval)
        return self.system.power(self.system.detector_field(self._reduced, theta))
    def objective_and_gradient(self, theta, interval, target=None):
        h = self.cfg.baselines.oracle_fd_step
        grad = [(self.objective_only(theta + h * axis, interval)
                 - self.objective_only(theta - h * axis, interval)) / (2*h) for axis in np.eye(2)]
        return self.objective_only(theta, interval), np.asarray(grad)
    def solve(self, interval, target, theta_prev, history, measurement=None, measurement_valid=True):
        start = time.perf_counter()
        self.prepare_interval(interval)
        prep = time.perf_counter() - start
        initial = sensing_initial_command(self.system, theta_prev, measurement, measurement_valid)
        theta, value, diag = projected_gradient_ascent(theta_prev,
            lambda c: self.objective_and_gradient(c, interval),
            lambda c: self.objective_only(c, interval), self.cfg.tracking, initial)
        diag.update(atmosphere_prepare_time_sec=prep, atmosphere_integrations_this_interval=1,
                    atmosphere_depends_on_fsm=False, gradient_backend="receiver_finite_difference",
                    initial_command=initial)
        return dict(theta=theta, predicted_metric=value, runtime_sec=time.perf_counter()-start, diagnostics=diag)


def make_baseline(name, system, cfg):
    if name in {"none", "no_control"}:
        return NoControlBaseline(system, cfg)
    if name == "pid":
        return PIDBaseline(system, cfg)
    if name in {"linear_mpc", "mpc"}:
        return LinearMPCBaseline(system, cfg)
    if name in {"ssfm_oracle_cpu", "ssfm_cpu"}:
        return NumPySSFMOracleBaseline(system, cfg)
    if name in {"ssfm_oracle", "diff_ssfm", "oracle"}:
        from solver.differentiable_ssfm import DifferentiableSSFMSolver
        return DifferentiableSSFMSolver(system, cfg)
    raise ValueError(f"Unknown baseline: {name}")
