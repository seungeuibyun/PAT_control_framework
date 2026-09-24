"""Eqs. (43)-(47): box projection, PSD initialization and feasible quantization."""
from __future__ import annotations
import numpy as np


def command_bounds(theta_prev, tracking_cfg):
    previous = np.asarray(theta_prev, dtype=float)
    limit = np.broadcast_to(tracking_cfg.theta_max, (2,))
    slew = np.broadcast_to(tracking_cfg.theta_slew_max, (2,))
    lower, upper = np.maximum(-limit, previous - slew), np.minimum(limit, previous + slew)
    if np.any(lower > upper):
        raise ValueError("Empty physical command set")
    return lower, upper


def project_pat_command(theta, theta_prev, tracking_cfg):
    lower, upper = command_bounds(theta_prev, tracking_cfg)
    return np.clip(np.asarray(theta, dtype=float), lower, upper)


def quantize_pat_command(theta, theta_prev, tracking_cfg):
    lower, upper = command_bounds(theta_prev, tracking_cfg)
    q = tracking_cfg.theta_quantization
    lo, hi = np.ceil(lower / q - 1e-10), np.floor(upper / q + 1e-10)
    if np.any(lo > hi):
        raise ValueError("No quantized command satisfies the range/slew constraints")
    return np.clip(np.rint(np.asarray(theta) / q), lo, hi) * q


def sensing_initial_command(system, theta_prev, measurement, valid):
    if not valid:
        return np.asarray(theta_prev, dtype=float).copy()
    delta = np.linalg.pinv(system.psd_jacobian) @ (system.reference_centroid - measurement[:2])
    return project_pat_command(np.asarray(theta_prev) + delta, theta_prev, system.cfg.tracking)


def projected_gradient_ascent(theta_prev, objective_and_gradient, objective_only,
                              tracking_cfg, initial_theta=None):
    t = tracking_cfg
    previous = np.asarray(theta_prev, dtype=float)
    theta = project_pat_command(previous if initial_theta is None else initial_theta, previous, t)
    metric, grad = objective_and_gradient(theta)
    if not np.isfinite(metric):
        raise FloatingPointError("Nonfinite initial detector power")
    def penalized(value, command):
        return float(value) - 0.5 * t.control_regularization * np.sum((command - previous)**2)
    obj = penalized(metric, theta)
    reason, accepted_steps, final_norm, final_step = "max_iterations", 0, None, 0.0
    trace = [float(metric)]
    for _ in range(t.max_opt_iterations):
        g = np.asarray(grad) - t.control_regularization * (theta - previous)
        norm = float(np.linalg.norm(g))
        final_norm = norm if np.isfinite(norm) else None
        if not np.isfinite(norm):
            raise FloatingPointError("Nonfinite receiver power gradient")
        if norm <= t.gradient_norm_tol:
            reason = "gradient_tol"
            break
        step = t.optimizer_step
        accepted = False
        for _ in range(t.line_search_steps):
            candidate = project_pat_command(theta + step * g / norm, previous, t)
            final_step = float(np.linalg.norm(candidate - theta))
            if final_step > t.optimizer_min_step:
                candidate_metric = objective_only(candidate)
                candidate_obj = penalized(candidate_metric, candidate)
                if np.isfinite(candidate_obj) and candidate_obj >= obj:
                    accepted = True
                    break
            step *= t.line_search_shrink
        if not accepted:
            reason = "line_search_failed"
            break
        improvement = candidate_obj - obj
        tolerance = t.objective_abs_tol + t.objective_rel_tol * max(abs(obj), abs(candidate_obj))
        theta, metric, obj = candidate, candidate_metric, candidate_obj
        accepted_steps += 1
        trace.append(float(metric))
        if improvement <= tolerance:
            reason = "objective_tol"
            break
        metric, grad = objective_and_gradient(theta)
    continuous = theta.copy()
    theta = quantize_pat_command(continuous, previous, t)
    metric = objective_only(theta)  # prediction must describe the applied command
    return theta, float(metric), {
        "optimizer_iterations": accepted_steps, "optimizer_stop_reason": reason,
        "final_gradient_norm": final_norm, "final_numerical_step_norm": final_step,
        "continuous_command": continuous, "continuous_power_trace_w": trace,
        "physical_command_change_norm": float(np.linalg.norm(theta - previous)),
        "physical_command_change_by_axis": np.abs(theta - previous),
        "theta_slew_max": np.broadcast_to(t.theta_slew_max, (2,)),
        "theta_quantization": t.theta_quantization,
    }
