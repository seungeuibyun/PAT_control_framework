from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from config.settings import ExperimentConfig
from system_model.optical_system import OpticalPATSystem


def _resolve_device(device: str) -> torch.device:
    key = device.lower()

    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if key == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if key == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

    return torch.device(key)


class DifferentiableSSFMSolver:
    """Differentiable split-step Fourier PAT solver.

    This is a strong SSFM baseline:
      * the propagation operator is evaluated directly on the full spatial grid,
      * torch autograd obtains dQ/dtheta from one forward/backward graph,
      * no finite-difference steering perturbations are required.

    It is useful both as a controller baseline and as the main runtime competitor
    against Frozen-PINN.
    """

    name = "Differentiable SSFM"

    def __init__(
        self,
        system: OpticalPATSystem,
        cfg: ExperimentConfig,
        *,
        device: str = "auto",
        dtype: str = "float64",
        num_iterations: int | None = None,
        theta_step_max: float | None = None,
        line_search_steps: int = 0,
    ):
        self.system = system
        self.cfg = cfg
        self.device = _resolve_device(device)

        # Apple MPS does not support the same float64/complex128 path used
        # by CPU/CUDA here.  Force float32/complex64 automatically on MPS,
        # regardless of the CLI dtype argument.
        if self.device.type == "mps":
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64
            self.requested_dtype = dtype
            self.effective_dtype = "float32"

            if dtype != "float32":
                print(
                    "[Differentiable SSFM] MPS detected: "
                    "forcing float32/complex64."
                )

        else:
            if dtype == "float64":
                self.real_dtype = torch.float64
                self.complex_dtype = torch.complex128
                self.effective_dtype = "float64"
            elif dtype == "float32":
                self.real_dtype = torch.float32
                self.complex_dtype = torch.complex64
                self.effective_dtype = "float32"
            else:
                raise ValueError("dtype must be float32 or float64")

            self.requested_dtype = dtype

        self.num_iterations = (
            int(num_iterations)
            if num_iterations is not None
            else int(cfg.baselines.oracle_num_iterations)
        )

        self.theta_step_max = (
            float(theta_step_max)
            if theta_step_max is not None
            else float(cfg.baselines.oracle_theta_step_max)
        )

        self.line_search_steps = int(line_search_steps)
        self.theta_max = float(cfg.baselines.theta_max)

        self._phase_cache: Dict[int, torch.Tensor] = {}

        self._build_static_tensors()

    # -------------------------------------------------------------------------
    # setup / timing helpers
    # -------------------------------------------------------------------------

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def _build_static_tensors(self):
        oc = self.cfg.optical
        tc = self.cfg.tracking

        self.X = torch.as_tensor(
            self.system.X,
            dtype=self.real_dtype,
            device=self.device,
        )

        self.Y = torch.as_tensor(
            self.system.Y,
            dtype=self.real_dtype,
            device=self.device,
        )

        self.gaussian = torch.as_tensor(
            self.system.gaussian_beam(
                self.system.X,
                self.system.Y,
            ),
            dtype=self.real_dtype,
            device=self.device,
        )

        fx = np.fft.fftfreq(oc.grid_size, d=self.system.dx)
        fy = np.fft.fftfreq(oc.grid_size, d=self.system.dy)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")

        kx = 2.0 * np.pi * FX
        ky = 2.0 * np.pi * FY

        H_np = np.exp(
            -1j
            * (kx**2 + ky**2)
            * self.system.dz_truth
            / (2.0 * oc.k0)
        )

        self.H_diff = torch.as_tensor(
            H_np,
            dtype=self.complex_dtype,
            device=self.device,
        )

        self.dA = float(self.system.dx * self.system.dy)
        self.dz = float(self.system.dz_truth)
        self.k0 = float(oc.k0)
        self.alpha = float(oc.attenuation)
        self.steps = int(tc.ssfm_steps)

    def prepare_interval(self, interval: int):
        """Cache SSFM half-step phase screens for a fixed atmospheric interval.

        This makes the baseline deliberately strong: channel-state evaluation is
        performed once and the same phase screens are reused across all steering
        queries inside the PAT optimizer.
        """
        if interval in self._phase_cache:
            return

        phase_list = []

        for iz in range(self.steps):
            z = (iz + 0.5) * self.dz

            dn = self.system.turbulence.eval(
                self.system.X,
                self.system.Y,
                z,
                interval,
            )

            phase_np = np.exp(
                (
                    1j * self.k0 * dn
                    - self.alpha / 2.0
                )
                * self.dz
                / 2.0
            )

            phase_list.append(
                torch.as_tensor(
                    phase_np,
                    dtype=self.complex_dtype,
                    device=self.device,
                )
            )

        self._phase_cache[interval] = torch.stack(
            phase_list,
            dim=0,
        )

        self._sync()

    # -------------------------------------------------------------------------
    # propagation / objective
    # -------------------------------------------------------------------------

    def propagate(self, theta: torch.Tensor, interval: int) -> torch.Tensor:
        self.prepare_interval(interval)

        phase_steer = torch.exp(
            1j
            * self.k0
            * (
                theta[0] * self.X
                + theta[1] * self.Y
            )
        )

        U = self.gaussian.to(self.complex_dtype) * phase_steer
        phase_stack = self._phase_cache[interval]

        for iz in range(self.steps):
            half = phase_stack[iz]

            U = U * half
            U = torch.fft.ifft2(
                torch.fft.fft2(U) * self.H_diff
            )
            U = U * half

        return U

    def _objective_tensor(
        self,
        U: torch.Tensor,
        target,
    ) -> torch.Tensor:
        objective = self.cfg.tracking.objective

        if objective == "power":
            mask = torch.as_tensor(
                self.system.aperture_mask(target),
                dtype=self.real_dtype,
                device=self.device,
            )

            return torch.sum(
                mask * torch.abs(U) ** 2
            ) * self.dA

        if objective == "coupling":
            psi = torch.as_tensor(
                self.system.smf_mode(target),
                dtype=self.complex_dtype,
                device=self.device,
            )

            inner = torch.sum(
                U * torch.conj(psi)
            ) * self.dA

            field_power = torch.sum(
                torch.abs(U) ** 2
            ) * self.dA

            mode_power = torch.sum(
                torch.abs(psi) ** 2
            ) * self.dA

            return (
                torch.abs(inner) ** 2
                / torch.clamp(
                    field_power * mode_power,
                    min=1e-30,
                )
            )

        if objective == "centroid":
            roi = torch.as_tensor(
                self.system.centroid_roi_mask(target),
                dtype=self.real_dtype,
                device=self.device,
            )

            intensity = torch.abs(U) ** 2

            denom = torch.sum(
                roi * intensity
            ) * self.dA

            denom = torch.clamp(
                denom,
                min=1e-30,
            )

            px = (
                torch.sum(
                    roi * self.X * intensity
                )
                * self.dA
                / denom
            )

            py = (
                torch.sum(
                    roi * self.Y * intensity
                )
                * self.dA
                / denom
            )

            tx = torch.as_tensor(
                float(target[0]),
                dtype=self.real_dtype,
                device=self.device,
            )

            ty = torch.as_tensor(
                float(target[1]),
                dtype=self.real_dtype,
                device=self.device,
            )

            return -(
                (px - tx) ** 2
                + (py - ty) ** 2
            )

        raise ValueError(
            "objective must be power, coupling, or centroid"
        )

    def objective_and_gradient(
        self,
        theta_np,
        interval: int,
        target,
    ):
        theta = torch.tensor(
            np.asarray(theta_np, dtype=float),
            dtype=self.real_dtype,
            device=self.device,
            requires_grad=True,
        )

        U = self.propagate(theta, interval)
        Q = self._objective_tensor(U, target)

        grad = torch.autograd.grad(
            Q,
            theta,
            create_graph=False,
            retain_graph=False,
        )[0]

        self._sync()

        return (
            float(Q.detach().cpu().item()),
            grad.detach().cpu().numpy().astype(float),
        )

    def objective_only(
        self,
        theta_np,
        interval: int,
        target,
    ) -> float:
        with torch.no_grad():
            theta = torch.as_tensor(
                np.asarray(theta_np, dtype=float),
                dtype=self.real_dtype,
                device=self.device,
            )

            U = self.propagate(theta, interval)
            Q = self._objective_tensor(U, target)

        self._sync()

        return float(Q.detach().cpu().item())

    # -------------------------------------------------------------------------
    # controller
    # -------------------------------------------------------------------------

    def solve(
        self,
        interval: int,
        target,
        theta_prev,
        history,
    ):
        theta = np.asarray(
            theta_prev,
            dtype=float,
        ).copy()

        self.prepare_interval(interval)
        self._sync()

        t0 = time.perf_counter()
        metric = None

        for _ in range(self.num_iterations):
            metric, grad = self.objective_and_gradient(
                theta,
                interval,
                target,
            )

            gnorm = np.linalg.norm(grad)

            if not np.isfinite(gnorm) or gnorm < 1e-14:
                break

            step = (
                self.theta_step_max
                * grad
                / max(gnorm, 1e-30)
            )

            cand = np.clip(
                theta + step,
                -self.theta_max,
                self.theta_max,
            )

            # Optional line search.  A zero value isolates one
            # forward/backward query per optimization iteration.
            if self.line_search_steps > 0:
                accepted = False
                local_step = step.copy()

                for _ in range(self.line_search_steps):
                    cand = np.clip(
                        theta + local_step,
                        -self.theta_max,
                        self.theta_max,
                    )

                    cand_metric = self.objective_only(
                        cand,
                        interval,
                        target,
                    )

                    if cand_metric >= metric:
                        metric = cand_metric
                        accepted = True
                        break

                    local_step *= 0.5

                if not accepted:
                    break

            theta = cand

        self._sync()
        runtime = time.perf_counter() - t0

        return {
            "theta": theta,
            "runtime_sec": runtime,
            "predicted_metric": metric,
            "diagnostics": {
                "device": str(self.device),
                "requested_dtype": self.requested_dtype,
                "effective_dtype": self.effective_dtype,
                "torch_dtype": str(self.real_dtype),
            },
        }
