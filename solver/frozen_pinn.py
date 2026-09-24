"""Frozen tanh features and one atmospheric coefficient evolution per interval.

A_boundary is a fixed Gaussian window followed by a Dirichlet sine projection.
It is linear, satisfies the boundary everywhere (not just at sampled points),
and has an analytic Laplacian. SVD whitening only changes coefficient units.
The matrix-free RHS is exactly the collocation least-squares G(z)c, Eq. (35).
"""
from __future__ import annotations
import time
import numpy as np
from scipy.fft import dstn, idstn
from scipy.integrate import solve_ivp
from solver.optimization import projected_gradient_ascent, sensing_initial_command


class FrozenPINNBasis:
    def __init__(self, system, cfg):
        self.system, self.cfg = system, cfg
        o, f = cfg.optical, cfg.frozen_pinn
        self.n, self.s = f.collocation_side, f.spectral_side
        self.H = o.half_width
        self.axis = -self.H + 2 * self.H * np.arange(1, self.n + 1) / (self.n + 1)
        self.X, self.Y = np.meshgrid(self.axis, self.axis)
        self.xc, self.yc = self.X.ravel(), self.Y.ravel()
        rng = np.random.default_rng(f.seed + 1234)
        direction = rng.normal(size=(f.hidden_width, 2))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        scales = np.exp(rng.uniform(np.log(5.0), np.log(f.feature_scale_max), f.hidden_width))
        self.W = direction * scales[:, None]
        # Multi-scale ridge locations are an explicit implementation choice;
        # the PDF does not prescribe a distribution for w_m or b_m.
        self.b = rng.uniform(-2.0, 2.0, f.hidden_width)
        modal_features = np.empty((self.s**2, f.hidden_width))
        for start in range(0, f.hidden_width, 32):
            stop = min(start + 32, f.hidden_width)
            raw = np.tanh(self.W[start:stop, 0, None, None] * self.X / self.H
                          + self.W[start:stop, 1, None, None] * self.Y / self.H
                          + self.b[start:stop, None, None])
            raw *= np.exp(-(self.X**2 + self.Y**2) / (f.boundary_envelope_fraction * self.H)**2)
            modal = dstn(raw, type=1, axes=(-2, -1), norm="ortho") * (2 * self.H / (self.n + 1))
            modal_features[:, start:stop] = modal[:, :self.s, :self.s].reshape(stop - start, -1).T
        Q, singular, _ = np.linalg.svd(modal_features, full_matrices=False)
        keep = singular > f.svd_cutoff * singular[0]
        self.Q = Q[:, keep].copy()
        self.singular_values = singular
        self.R = self.Q.shape[1]
        modes = np.arange(1, self.s + 1) * np.pi / (2 * self.H)
        self.lap_eigenvalues = -(modes[:, None]**2 + modes[None, :]**2).ravel()
        self.laplacian = self.Q.T @ (self.lap_eigenvalues[:, None] * self.Q)
        self.boundary_residual = 0.0

    def to_modal(self, field):
        modes = dstn(np.asarray(field).reshape(self.n, self.n), type=1, norm="ortho")
        return (modes[:self.s, :self.s] * (2 * self.H / (self.n + 1))).ravel()

    def from_modal(self, modes):
        padded = np.zeros((self.n, self.n), dtype=np.asarray(modes).dtype)
        padded[:self.s, :self.s] = np.asarray(modes).reshape(self.s, self.s)
        return idstn(padded, type=1, norm="ortho") * ((self.n + 1) / (2 * self.H))

    def initial_coefficients(self, values):
        return self._multiply(self.Q.T, self.to_modal(values))

    @staticmethod
    def _multiply(matrix, value):
        # Avoid repeatedly converting a large real matrix to complex inside
        # BLAS on every RK45 RHS evaluation.
        return matrix @ value.real + 1j * (matrix @ value.imag)

    def reconstruct(self, c, x, y):
        modes = np.arange(1, self.s + 1) * np.pi / (2 * self.H)
        sx = np.sin(np.outer(np.asarray(x) + self.H, modes)) / np.sqrt(self.H)
        sy = np.sin(np.outer(np.asarray(y) + self.H, modes)) / np.sqrt(self.H)
        return sy @ self._multiply(self.Q, c).reshape(self.s, self.s) @ sx.T

    def rhs(self, c, dn):
        o = self.cfg.optical
        field = self.from_modal(self._multiply(self.Q, c))
        potential = self._multiply(self.Q.T, self.to_modal(dn * field))
        return 1j / (2 * o.k0) * self._multiply(self.laplacian, c) + 1j * o.k0 * potential - o.attenuation / 2 * c


class FrozenPINNSolver:
    name = "Frozen-PINN"

    def __init__(self, system, cfg):
        self.system, self.cfg = system, cfg
        start = time.perf_counter()
        self.basis = FrozenPINNBasis(system, cfg)
        self.basis_setup_time_sec = time.perf_counter() - start
        self._interval = None
        self._reduced = None
        self.atmosphere_integrations = 0
        self.initial_field_relative_error = None
        self.interval_diagnostics = {}

    def _integrate_coefficients(self, c, interval):
        """CPU reference for the collocation least-squares evolution."""
        b, f = self.basis, self.cfg.frozen_pinn
        nfev = 0
        start = time.perf_counter()
        for left, right in zip(self.system.turbulence.edges[:-1], self.system.turbulence.edges[1:]):
            dn = self.system.turbulence.eval(b.X, b.Y, (left + right) / 2, interval)
            sol = solve_ivp(lambda z, state: b.rhs(state, dn), (left, right), c,
                            method="RK45", rtol=f.ode_rtol, atol=f.ode_atol, t_eval=[right])
            if not sol.success or not np.all(np.isfinite(sol.y)):
                raise RuntimeError(f"Frozen-PINN RK45 failed: {sol.message}")
            c = sol.y[:, -1]
            nfev += sol.nfev
        return c, dict(ode_nfev=nfev, atmosphere_backend="scipy_cpu_matrix_free_RK45",
                       coefficient_evolution_time_sec=time.perf_counter()-start,
                       operator_setup_time_sec=0.0)

    def prepare_interval(self, interval):
        if self._interval == interval:
            return
        start = time.perf_counter()
        b, o, f = self.basis, self.cfg.optical, self.cfg.frozen_pinn
        u0 = self.system.tx_field(interval, b.X, b.Y)
        c = b.initial_coefficients(u0)
        # Evaluate fit error on an independent grid, not the fit samples.
        check_axis = np.linspace(-o.half_width, o.half_width, max(128, 2 * b.n), endpoint=False)
        X, Y = np.meshgrid(check_axis, check_axis)
        truth = self.system.tx_field(interval, X, Y)
        error = np.linalg.norm(b.reconstruct(c, check_axis, check_axis) - truth) / np.linalg.norm(truth)
        self.initial_field_relative_error = float(error)
        initial_norm = np.linalg.norm(c)
        c, integration = self._integrate_coefficients(c, interval)
        # Evaluate the frozen field directly at reducer preimage coordinates;
        # no giant [receiver_pixels x rank] matrix is formed.
        axis = self.system.rx / o.reducer_magnification
        field = b.reconstruct(c, axis, axis)
        reduced = (np.sqrt(o.reducer_transmission) / abs(o.reducer_magnification)
                   * field * self.system.reduced_mask
                   * np.exp(1j * o.reducer_phase_curvature * (self.system.RX**2 + self.system.RY**2)))
        self._interval, self._reduced, self._coefficients = interval, reduced, c
        self.atmosphere_integrations += 1
        self.interval_diagnostics = {
            "basis_rank": b.R, "boundary_residual": b.boundary_residual,
            "basis_setup_time_sec": self.basis_setup_time_sec,
            "initial_field_relative_error": float(error),
            "coefficient_power_ratio": float(np.linalg.norm(c)**2 / max(initial_norm**2, 1e-30)),
            "expected_extinction_ratio": float(10**(-o.path_loss_db / 10)),
            "atmosphere_prepare_time_sec": time.perf_counter() - start,
            "atmosphere_integrations_this_interval": 1, **integration,
            "ode_solver": "scipy_RK45", "boundary_transform": "fixed_Dirichlet_sine_projection",
            "atmosphere_depends_on_fsm": False,
        }

    def objective_and_gradient(self, theta, interval, target=None):
        self.prepare_interval(interval)
        return self.system.power_and_gradient(self._reduced, theta)

    def objective_only(self, theta, interval, target=None):
        self.prepare_interval(interval)
        return self.system.power(self.system.detector_field(self._reduced, theta))

    def reconstruct_field(self, theta, interval):
        self.prepare_interval(interval)
        return self.system.detector_field(self._reduced, theta)

    def solve(self, interval, target, theta_prev, history, measurement=None, measurement_valid=True):
        start = time.perf_counter()
        self.prepare_interval(interval)
        initial = sensing_initial_command(self.system, theta_prev, measurement, measurement_valid)
        query_start = time.perf_counter()
        theta, value, diag = projected_gradient_ascent(theta_prev,
            lambda command: self.objective_and_gradient(command, interval),
            lambda command: self.objective_only(command, interval), self.cfg.tracking, initial)
        query_time = time.perf_counter() - query_start
        predicted_centroid = self.system.sensing_vector(self._reduced, theta)[:2]
        diag.update(self.interval_diagnostics)
        diag.update(predicted_centroid=predicted_centroid, initial_command=initial,
                    sensing_valid=measurement_valid, query_runtime_sec=query_time,
                    gradient_backend="analytic_receiver_sensitivity", device="cpu", effective_dtype="float64")
        return dict(theta=theta, predicted_metric=value, runtime_sec=time.perf_counter() - start,
                    diagnostics=diag)
