"""Layerwise GPU Galerkin assembly, CPU float64 RK45, Torch receiver queries.

P.T @ diag(delta_n) @ P is the same collocation least-squares potential as
Q.T @ DST(delta_n * IDST(Q @ c)). Assemble it once per Markov layer, not at
every RHS evaluation. Spatial basis construction is an offline cost.
"""
import time
import numpy as np
import torch
from scipy.fft import idstn
from scipy.integrate import solve_ivp
from solver.frozen_pinn import FrozenPINNSolver
from solver.differentiable_ssfm import TorchReceiver, _resolve_device

resolve_torch_device = _resolve_device


class TorchFrozenPINNSolver(FrozenPINNSolver):
    def __init__(self, system, cfg, *, device=None, dtype=None):
        super().__init__(system, cfg)
        self.engine = TorchReceiver(system, device or cfg.frozen_pinn.device, dtype or cfg.frozen_pinn.dtype)
        self._torch_interval = None
        mode = cfg.frozen_pinn.operator_backend
        self.projected_operator = mode == "projected" or (mode == "auto" and self.engine.device.type != "cpu")
        self.operator_setup_time_sec = 0.0
        if self.projected_operator:
            start = time.perf_counter()
            b = self.basis
            # Discrete orthonormal basis; continuous reconstruction scale cancels
            # between the forward and inverse DST in the reference RHS.
            host_dtype = np.float32 if self.engine.effective_dtype == "float32" else np.float64
            spatial = np.empty((b.n*b.n, b.R), dtype=host_dtype)
            for first in range(0, b.R, 32):
                last = min(first+32, b.R)
                pad = np.zeros((last-first, b.n, b.n))
                pad[:, :b.s, :b.s] = b.Q[:, first:last].T.reshape(-1, b.s, b.s)
                spatial[:, first:last] = idstn(pad, type=1, axes=(-2, -1), norm="ortho").reshape(last-first, -1).T
            self.spatial_basis = torch.as_tensor(spatial, device=self.engine.device)
            self.engine._sync()
            self.operator_setup_time_sec = time.perf_counter()-start

    def _layer_generator(self, dn):
        b, o = self.basis, self.cfg.optical
        v = self.engine.tensor(np.asarray(dn).ravel())
        potential = (self.spatial_basis.T @ (v[:, None]*self.spatial_basis)).cpu().numpy().astype(np.float64)
        # Roundoff in GPU GEMM must not introduce an anti-Hermitian potential.
        potential = (potential+potential.T)*0.5
        return b.laplacian/(2*o.k0) + o.k0*potential

    def _integrate_coefficients(self, c, interval):
        if not self.projected_operator:
            return super()._integrate_coefficients(c, interval)
        b, f, o = self.basis, self.cfg.frozen_pinn, self.cfg.optical
        start = time.perf_counter()
        assembly_time = ode_time = channel_time = 0.0
        nfev = 0
        for left, right in zip(self.system.turbulence.edges[:-1], self.system.turbulence.edges[1:]):
            tick = time.perf_counter()
            dn = self.system.turbulence.eval(b.X, b.Y, (left+right)/2, interval)
            channel_time += time.perf_counter()-tick
            tick = time.perf_counter()
            generator = self._layer_generator(dn)
            assembly_time += time.perf_counter()-tick
            tick = time.perf_counter()
            sol = solve_ivp(lambda z, state: 1j*b._multiply(generator, state)-o.attenuation/2*state,
                            (left, right), c, method="RK45", rtol=f.ode_rtol,
                            atol=f.ode_atol, t_eval=[right])
            if not sol.success or not np.all(np.isfinite(sol.y)):
                raise RuntimeError(f"Frozen-PINN RK45 failed: {sol.message}")
            c = sol.y[:, -1]
            nfev += sol.nfev
            ode_time += time.perf_counter()-tick
        return c, dict(ode_nfev=nfev,
            atmosphere_backend=f"{self.engine.device.type}_projected_scipy_cpu_RK45",
            operator_dtype=self.engine.effective_dtype, ode_dtype="complex128",
            operator_setup_time_sec=self.operator_setup_time_sec,
            coefficient_evolution_time_sec=time.perf_counter()-start,
            operator_assembly_time_sec=assembly_time, ode_integration_time_sec=ode_time,
            channel_sampling_time_sec=channel_time)

    def prepare_interval(self, interval):
        super().prepare_interval(interval)
        if self._torch_interval != interval:
            self.engine.set_field(self._reduced)
            self._torch_interval = interval

    def objective_and_gradient(self, theta, interval, target=None):
        self.prepare_interval(interval)
        return self.engine.objective_and_gradient(theta)

    def objective_only(self, theta, interval, target=None):
        self.prepare_interval(interval)
        return self.engine.objective_only(theta)

    def solve(self, *args, **kwargs):
        result = super().solve(*args, **kwargs)
        result["diagnostics"].update(device=str(self.engine.device),
            requested_dtype=self.engine.requested_dtype,
            receiver_complex_backend=self.engine.complex_backend,
            effective_dtype=self.engine.effective_dtype,
            gradient_backend="torch_receiver_autograd")
        return result
