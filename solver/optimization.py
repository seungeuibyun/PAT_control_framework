from __future__ import annotations

import numpy as np


def project_pat_command(theta, theta_prev, tracking_cfg):
    """Project a numerical optimizer iterate onto the physical FSM set.

    Two constraints are enforced independently of the number of optimizer
    iterations:

        |theta_d| <= theta_max
        ||theta - theta_prev||_2 <= theta_slew_rate * T_c

    The second one is the physical per-control-interval slew constraint,
    derived from a slew rate rather than the optimizer iteration count.
    """
    theta = np.asarray(theta, dtype=float).copy()
    theta_prev = np.asarray(theta_prev, dtype=float)

    theta = np.clip(
        theta,
        -tracking_cfg.theta_max,
        tracking_cfg.theta_max,
    )

    delta = theta - theta_prev
    delta_norm = np.linalg.norm(delta)

    if (
        np.isfinite(tracking_cfg.theta_slew_max)
        and tracking_cfg.theta_slew_max > 0
        and delta_norm > tracking_cfg.theta_slew_max
    ):
        delta *= (
            tracking_cfg.theta_slew_max
            / max(delta_norm, 1e-30)
        )

        theta = theta_prev + delta

        theta = np.clip(
            theta,
            -tracking_cfg.theta_max,
            tracking_cfg.theta_max,
        )

    return theta


def projected_gradient_ascent(
    theta_prev,
    objective_and_gradient,
    objective_only,
    tracking_cfg,
    initial_theta=None,
):
    """Shared PAT optimizer used by Frozen-PINN and SSFM oracle.

    Numerical optimizer settings are deliberately separated from physical FSM
    constraints. Increasing max_opt_iterations therefore does not increase the
    physically reachable angular change in one PAT interval.
    """
    theta_prev = np.asarray(
        theta_prev,
        dtype=float,
    )

    if initial_theta is None:
        theta = project_pat_command(
            theta_prev,
            theta_prev,
            tracking_cfg,
        )
    else:
        theta = project_pat_command(
            np.asarray(initial_theta, dtype=float),
            theta_prev,
            tracking_cfg,
        )

    metric, grad = objective_and_gradient(theta)

    regularization = float(
        tracking_cfg.control_regularization
    )

    def penalized(metric_value, theta_value):
        return (
            float(metric_value)
            - 0.5
            * regularization
            * np.sum(
                (
                    np.asarray(theta_value)
                    - theta_prev
                )
                ** 2
            )
        )

    obj = penalized(
        metric,
        theta,
    )

    reason = "max_iterations"
    accepted_steps = 0
    final_grad_norm = np.nan
    final_step_norm = 0.0

    for iteration in range(
        int(tracking_cfg.max_opt_iterations)
    ):
        grad_obj = (
            np.asarray(grad, dtype=float)
            - regularization
            * (theta - theta_prev)
        )

        grad_norm = float(
            np.linalg.norm(grad_obj)
        )

        final_grad_norm = grad_norm

        if not np.isfinite(grad_norm):
            reason = "nonfinite_gradient"
            break

        if grad_norm <= tracking_cfg.gradient_norm_tol:
            reason = "gradient_tol"
            break

        direction = (
            grad_obj
            / max(grad_norm, 1e-30)
        )

        numerical_step = float(
            tracking_cfg.optimizer_step
        )

        accepted = False
        candidate = theta
        candidate_metric = metric
        candidate_obj = obj
        actual_step_norm = 0.0

        for _ in range(
            int(tracking_cfg.line_search_steps)
        ):
            raw_candidate = (
                theta
                + numerical_step
                * direction
            )

            candidate = project_pat_command(
                raw_candidate,
                theta_prev,
                tracking_cfg,
            )

            actual_step_norm = float(
                np.linalg.norm(
                    candidate - theta
                )
            )

            if (
                actual_step_norm
                <= tracking_cfg.optimizer_min_step
            ):
                numerical_step *= 0.5
                continue

            candidate_metric = objective_only(
                candidate
            )

            candidate_obj = penalized(
                candidate_metric,
                candidate,
            )

            if candidate_obj >= obj:
                accepted = True
                break

            numerical_step *= 0.5

        if not accepted:
            reason = "line_search_failed"
            break

        improvement = float(
            candidate_obj - obj
        )

        theta = candidate
        metric = candidate_metric
        obj = candidate_obj
        accepted_steps += 1
        final_step_norm = actual_step_norm

        tolerance = (
            float(
                tracking_cfg.objective_abs_tol
            )
            + float(
                tracking_cfg.objective_rel_tol
            )
            * max(
                abs(obj),
                abs(obj - improvement),
                1e-12,
            )
        )

        if improvement <= tolerance:
            reason = "objective_tol"
            break

        # Refresh gradient only after accepting the iterate.
        metric, grad = objective_and_gradient(
            theta
        )

    else:
        iteration = (
            int(tracking_cfg.max_opt_iterations)
            - 1
        )

    diagnostics = {
        "optimizer_iterations": int(
            accepted_steps
        ),
        "optimizer_stop_reason": reason,
        "final_gradient_norm": (
            None
            if not np.isfinite(final_grad_norm)
            else float(final_grad_norm)
        ),
        "final_numerical_step_norm": float(
            final_step_norm
        ),
        "physical_command_change_norm": float(
            np.linalg.norm(
                theta - theta_prev
            )
        ),
        "theta_slew_max": float(
            tracking_cfg.theta_slew_max
        ),
    }

    return (
        theta,
        float(metric),
        diagnostics,
    )
