from __future__ import annotations

import time
import numpy as np
from scipy.integrate import solve_ivp

from config.settings import ExperimentConfig
from system_model.optical_system import OpticalPATSystem
from solver.optimization import projected_gradient_ascent


class FrozenPINNBasis:
    """Frozen-PINN spatial neural representation."""

    def __init__(self, system: OpticalPATSystem, cfg: ExperimentConfig):
        self.system = system
        self.cfg = cfg
        oc = cfg.optical
        fc = cfg.frozen_pinn

        xc = np.linspace(-oc.half_width, oc.half_width, fc.collocation_side)
        yc = np.linspace(-oc.half_width, oc.half_width, fc.collocation_side)
        Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
        self.xc = Xc.ravel()
        self.yc = Yc.ravel()
        self.xy_phys = np.column_stack([self.xc, self.yc])
        self.xy = self.xy_phys / oc.half_width

        xb = np.linspace(-oc.half_width, oc.half_width, fc.boundary_side)
        yb = np.linspace(-oc.half_width, oc.half_width, fc.boundary_side)
        boundary_xy = np.vstack([
            np.column_stack([xb, np.full_like(xb, -oc.half_width)]),
            np.column_stack([xb, np.full_like(xb,  oc.half_width)]),
            np.column_stack([np.full_like(yb, -oc.half_width), yb]),
            np.column_stack([np.full_like(yb,  oc.half_width), yb]),
        ])
        self.xy_boundary_phys = np.unique(boundary_xy, axis=0)
        self.xy_boundary = self.xy_boundary_phys / oc.half_width

        rng = np.random.default_rng(fc.seed + 1234)
        if fc.sampler == "elm":
            self.W, self.b = self._sample_elm(rng)
        elif fc.sampler == "swim":
            self.W, self.b = self._sample_swim(rng)
        else:
            raise ValueError("sampler must be elm or swim")

        raw = self._raw_features(self.xy)
        raw_lap = self._raw_laplacian(self.xy)
        raw_boundary = self._raw_features(self.xy_boundary)

        # boundary-compliant layer
        _, s_b, Vh_b = np.linalg.svd(raw_boundary.T, full_matrices=True)
        if len(s_b) > 0:
            tol_b = max(raw_boundary.T.shape) * np.finfo(float).eps * s_b[0]
            rank_b = int(np.sum(s_b > tol_b))
        else:
            rank_b = 0
        A = Vh_b[rank_b:, :]
        if A.shape[0] == 0:
            raise RuntimeError(
                "Boundary-compliant layer has zero width. Increase hidden_width or reduce boundary_side."
            )

        boundary_compliant = A @ raw
        V, s, _ = np.linalg.svd(boundary_compliant, full_matrices=False)
        if s[0] <= 0:
            raise RuntimeError("Frozen-PINN feature matrix has zero rank.")
        keep = s >= fc.svd_cutoff * s[0]
        if not np.any(keep):
            keep[0] = True
        self.singular_values = s
        self.Ar = V[:, keep].T @ A

        self.B = (self.Ar @ raw).astype(np.complex128)
        self.B_lap = (self.Ar @ raw_lap).astype(np.complex128)
        self.R = self.B.shape[0]
        self.B_plus = np.linalg.pinv(self.B, rcond=fc.pinv_rcond)
        self.boundary_residual = float(np.max(np.abs(self.Ar @ raw_boundary)))

        xy_eval = np.column_stack([system.X.ravel(), system.Y.ravel()])
        self.B_eval = self.basis_at(xy_eval)

    def _sample_elm(self, rng):
        fc = self.cfg.frozen_pinn
        W = rng.standard_normal((fc.hidden_width, 2))
        b = rng.uniform(-fc.elm_bias_range, fc.elm_bias_range, fc.hidden_width)
        return W, b

    def _sample_swim(self, rng):
        fc = self.cfg.frozen_pinn
        n = self.xy.shape[0]
        i1 = rng.integers(0, n, fc.hidden_width)
        i2 = rng.integers(0, n, fc.hidden_width)
        same = i1 == i2
        while np.any(same):
            i2[same] = rng.integers(0, n, np.sum(same))
            same = i1 == i2
        x1 = self.xy[i1]
        x2 = self.xy[i2]
        d = x2 - x1
        norm2 = np.sum(d * d, axis=1)
        bad = norm2 < 1e-10
        while np.any(bad):
            i2[bad] = rng.integers(0, n, np.sum(bad))
            x2[bad] = self.xy[i2[bad]]
            d = x2 - x1
            norm2 = np.sum(d * d, axis=1)
            bad = norm2 < 1e-10
        s = np.arctanh(0.5)
        W = 2.0 * s * d / norm2[:, None]
        b = -s - np.sum(W * x1, axis=1)
        return W, b

    def _raw_features(self, xy_normalized):
        a = self.W @ xy_normalized.T + self.b[:, None]
        return np.tanh(a)

    def _raw_laplacian(self, xy_normalized):
        oc = self.cfg.optical
        H = self._raw_features(xy_normalized)
        sigma_second = -2.0 * H * (1.0 - H**2)
        w_phys_sq = np.sum(self.W**2, axis=1) / oc.half_width**2
        return sigma_second * w_phys_sq[:, None]

    def basis_at(self, xy_physical):
        oc = self.cfg.optical
        xy_n = np.asarray(xy_physical) / oc.half_width
        raw = self._raw_features(xy_n)
        return (self.Ar @ raw).astype(np.complex128)

    def initial_coefficients(self, field_values):
        values = np.asarray(field_values, dtype=np.complex128).reshape(-1)
        return self.B_plus.T @ values

    def G(self, delta_n_colloc):
        oc = self.cfg.optical
        dn = np.asarray(delta_n_colloc).reshape(-1)
        L_basis = (
            (1j / (2.0 * oc.k0)) * self.B_lap.T
            + 1j * oc.k0 * dn[:, None] * self.B.T
            - (oc.attenuation / 2.0) * self.B.T
        )
        return self.B_plus.T @ L_basis


class FrozenPINNSolver:
    name = "Frozen-PINN"

    def __init__(self, system: OpticalPATSystem, cfg: ExperimentConfig):
        self.system = system
        self.cfg = cfg
        self.basis = FrozenPINNBasis(system, cfg)

        b0 = self.system.gaussian_beam(
            self.basis.xc,
            self.basis.yc,
        ).astype(np.complex128)
        c0 = self.basis.initial_coefficients(b0)
        U0_hat = (self.basis.B_eval.T @ c0).reshape(self.system.X.shape)
        U0_true = self.system.gaussian_beam(
            self.system.X,
            self.system.Y,
        ).astype(np.complex128)

        dA = self.system.dx * self.system.dy
        self.initial_field_relative_error = float(
            np.linalg.norm(U0_hat - U0_true)
            / max(np.linalg.norm(U0_true), 1e-30)
        )
        P_hat = float(np.sum(np.abs(U0_hat)**2) * dA)
        P_true = float(np.sum(np.abs(U0_true)**2) * dA)
        self.initial_power_ratio = P_hat / max(P_true, 1e-30)

        if self.initial_field_relative_error > 0.25:
            print(
                "[Frozen-PINN warning] Poor initial-field representation: "
                f"relative error={self.initial_field_relative_error:.3f}, "
                f"power ratio={self.initial_power_ratio:.3f}."
            )

    def _initial_state(self, theta):
        oc = self.cfg.optical
        b = self.system.gaussian_beam(self.basis.xc, self.basis.yc) * np.exp(
            1j * oc.k0 * (theta[0] * self.basis.xc + theta[1] * self.basis.yc)
        )
        c0 = self.basis.initial_coefficients(b)
        bx = 1j * oc.k0 * self.basis.xc * b
        by = 1j * oc.k0 * self.basis.yc * b
        sx0 = self.basis.initial_coefficients(bx)
        sy0 = self.basis.initial_coefficients(by)
        return np.column_stack([c0, sx0, sy0])

    def propagate(self, theta, interval):
        fc = self.cfg.frozen_pinn
        oc = self.cfg.optical
        S0 = self._initial_state(theta)
        R = self.basis.R

        def rhs(z, y):
            S = y.reshape(R, 3)
            dn = self.system.turbulence.eval(
                self.basis.xc, self.basis.yc, z, interval
            )
            G = self.basis.G(dn)
            return (G @ S).reshape(-1)

        sol = solve_ivp(
            rhs,
            (0.0, oc.propagation_distance),
            S0.reshape(-1),
            method="RK45",
            rtol=fc.ode_rtol,
            atol=fc.ode_atol,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        SL = sol.y[:, -1].reshape(R, 3)
        U = (self.basis.B_eval.T @ SL[:, 0]).reshape(self.system.X.shape)
        dUx = (self.basis.B_eval.T @ SL[:, 1]).reshape(self.system.X.shape)
        dUy = (self.basis.B_eval.T @ SL[:, 2]).reshape(self.system.X.shape)
        return U, dUx, dUy

    def _power_gradient(self, U, dUx, dUy, target):
        A = self.system.aperture_mask(target)
        dA = self.system.dx * self.system.dy
        Q = np.sum(A * np.abs(U) ** 2) * dA
        gx = 2.0 * np.real(np.sum(A * np.conj(U) * dUx) * dA)
        gy = 2.0 * np.real(np.sum(A * np.conj(U) * dUy) * dA)
        return float(Q), np.array([gx, gy])

    def _coupling_gradient(self, U, dUx, dUy, target):
        psi = self.system.smf_mode(target)
        dA = self.system.dx * self.system.dy
        a = np.sum(U * np.conj(psi)) * dA
        B = np.sum(np.abs(U) ** 2) * dA
        C = np.sum(np.abs(psi) ** 2) * dA
        Q = np.abs(a) ** 2 / max(B * C, 1e-30)
        grad = []
        for dU in (dUx, dUy):
            da = np.sum(dU * np.conj(psi)) * dA
            dB = 2.0 * np.real(np.sum(np.conj(U) * dU) * dA)
            dQ = (
                2.0 * np.real(np.conj(a) * da) * B - np.abs(a) ** 2 * dB
            ) / max(B**2 * C, 1e-30)
            grad.append(dQ)
        return float(np.real(Q)), np.asarray(grad)

    def _centroid_gradient(self, U, dUx, dUy, target):
        W = self.system.centroid_roi_mask(target)
        dA = self.system.dx * self.system.dy
        I = np.abs(U) ** 2
        D = max(np.sum(W * I) * dA, 1e-30)
        Nx = np.sum(W * self.system.X * I) * dA
        Ny = np.sum(W * self.system.Y * I) * dA
        p = np.array([Nx / D, Ny / D])
        J = np.zeros((2, 2), dtype=float)
        for j, dU in enumerate((dUx, dUy)):
            dI = 2.0 * np.real(np.conj(U) * dU)
            dD = np.sum(W * dI) * dA
            dNx = np.sum(W * self.system.X * dI) * dA
            dNy = np.sum(W * self.system.Y * dI) * dA
            J[0, j] = (dNx * D - Nx * dD) / D**2
            J[1, j] = (dNy * D - Ny * dD) / D**2
        e = p - np.asarray(target)
        Q = -np.dot(e, e)
        grad = -2.0 * J.T @ e
        return float(Q), grad

    def _metric_gradient(self, U, dUx, dUy, target):
        objective = self.cfg.tracking.objective
        if objective == "power":
            return self._power_gradient(U, dUx, dUy, target)
        if objective == "coupling":
            return self._coupling_gradient(U, dUx, dUy, target)
        if objective == "centroid":
            return self._centroid_gradient(U, dUx, dUy, target)
        raise ValueError("objective must be power, coupling, or centroid")

    def solve(self, interval: int, target, theta_prev, history):
        t0 = time.perf_counter()

        def objective_and_gradient(theta):
            U, dUx, dUy = self.propagate(
                theta,
                interval,
            )
            return self._metric_gradient(
                U,
                dUx,
                dUy,
                target,
            )

        def objective_only(theta):
            U, _, _ = self.propagate(
                theta,
                interval,
            )
            return self.system.metric(
                U,
                target,
                self.cfg.tracking.objective,
            )

        initial_theta = None
        if self.cfg.tracking.objective == "centroid":
            initial_theta = np.asarray(target, dtype=float) / max(self.system.cfg.optical.propagation_distance, 1e-30)

        theta, predicted_metric, opt_diag = (
            projected_gradient_ascent(
                theta_prev,
                objective_and_gradient,
                objective_only,
                self.cfg.tracking,
                initial_theta=initial_theta,
            )
        )

        runtime = time.perf_counter() - t0

        diagnostics = {
            "basis_rank": int(self.basis.R),
            "boundary_residual": self.basis.boundary_residual,
            "initial_field_relative_error": self.initial_field_relative_error,
            "initial_power_ratio": self.initial_power_ratio,
        }

        diagnostics.update(opt_diag)

        if self.cfg.tracking.objective == "centroid":
            U_pred, _, _ = self.propagate(theta, interval)
            diagnostics["predicted_centroid"] = self.system.centroid(
                U_pred, target
            ).tolist()

        return {
            "theta": theta,
            "predicted_metric": predicted_metric,
            "runtime_sec": runtime,
            "diagnostics": diagnostics,
        }

