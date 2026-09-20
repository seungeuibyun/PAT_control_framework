from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch

from config.settings import ExperimentConfig
from system_model.optical_system import OpticalPATSystem
from solver.frozen_pinn import FrozenPINNBasis
from solver.optimization import projected_gradient_ascent


def resolve_torch_device(device: str) -> torch.device:
    key = device.lower()

    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    if key == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    if key == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available.")

    return torch.device(key)


def resolve_real_dtype(
    device: torch.device,
    dtype: str,
):
    """Return torch dtype and effective precision.

    The optimized Frozen-PINN uses a real/imaginary block representation.
    Apple MPS is therefore supported without complex matrix multiplication.
    MPS is always forced to float32.
    """
    if device.type == "mps":
        if dtype != "float32":
            print(
                "[Frozen-PINN] MPS detected: "
                "forcing float32 real/imag block arithmetic."
            )

        return torch.float32, "float32"

    if dtype == "float64":
        return torch.float64, "float64"

    if dtype == "float32":
        return torch.float32, "float32"

    raise ValueError("dtype must be float32 or float64")


class TorchFrozenPINNSolver:
    """Reduced-space torch Frozen-PINN PAT controller.

    This implementation exploits two structures that are specific to the
    proposed PAT formulation.

    1. Interval transition precomputation
       ----------------------------------
       For a fixed atmospheric state in PAT interval k,

           dc/dz = G_k(z)c

       is linear and G_k(z) is independent of the steering angle.  Therefore
       one state-transition matrix T_k is precomputed once per interval:

           c(L)   = T_k c(0)
           s_x(L) = T_k s_x(0)
           s_y(L) = T_k s_y(0).

       Steering queries no longer integrate RK4 along z.

    2. Reduced-space receiver objectives
       ---------------------------------
       The receiver field is

           U = Phi^T c.

       Power, coupling efficiency, and centroid moments can therefore be
       written as low-dimensional quadratic forms in c.  During PAT
       optimization the N x N receiver field is never reconstructed.

    The full field is reconstructed only by reconstruct_field(), which is
    intended for visualization/diagnostics and is not used in the online
    objective/gradient path.

    Complex reduced dynamics are represented by real and imaginary blocks, so
    the same implementation runs on CPU, CUDA, and Apple MPS.
    """

    name = "Frozen-PINN"

    def __init__(
        self,
        system: OpticalPATSystem,
        cfg: ExperimentConfig,
        *,
        device: str = "auto",
        dtype: str = "float64",
        rk_steps: int = 32,
    ):
        self.system = system
        self.cfg = cfg

        self.device = resolve_torch_device(device)
        self.real_dtype, self.effective_dtype = resolve_real_dtype(
            self.device,
            dtype,
        )

        self.requested_dtype = dtype
        self.rk_steps = int(rk_steps)

        if self.rk_steps < 1:
            raise ValueError("rk_steps must be >= 1")

        # Frozen neural features / boundary layer / SVD are constructed once.
        t0 = time.perf_counter()
        self.basis = FrozenPINNBasis(system, cfg)
        self.basis_setup_time_sec = time.perf_counter() - t0

        self._build_torch_tensors()

        # interval -> state transition matrix
        self._transition_cache: Dict[int, Dict[str, object]] = {}

        # (objective, target_x, target_y) -> reduced receiver operators
        self._target_cache: Dict[Tuple[object, ...], Dict[str, object]] = {}

        # Coupling denominator operator, independent of receiver center.
        self._global_mass_matrix = None
        self._global_mass_prepare_time_sec = None

        self._build_diagnostics()

    # ------------------------------------------------------------------
    # device / tensor helpers
    # ------------------------------------------------------------------

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def _tensor(self, array):
        return torch.as_tensor(
            np.asarray(array),
            dtype=self.real_dtype,
            device=self.device,
        )

    def _build_torch_tensors(self):
        oc = self.cfg.optical

        # Frozen basis matrices are real for the current tanh architecture.
        self.B = self._tensor(
            np.real(self.basis.B)
        )                                           # R x Nc

        self.B_lap = self._tensor(
            np.real(self.basis.B_lap)
        )                                           # R x Nc

        self.B_plus = self._tensor(
            np.real(self.basis.B_plus)
        )                                           # Nc x R

        self.B_eval = self._tensor(
            np.real(self.basis.B_eval)
        )                                           # R x Ngrid

        self.xc = self._tensor(self.basis.xc)
        self.yc = self._tensor(self.basis.yc)

        self.gaussian_col = self._tensor(
            self.system.gaussian_beam(
                self.basis.xc,
                self.basis.yc,
            )
        )

        self.X_flat = self._tensor(
            self.system.X.reshape(-1)
        )

        self.Y_flat = self._tensor(
            self.system.Y.reshape(-1)
        )

        self.k0 = float(oc.k0)
        self.alpha = float(oc.attenuation)
        self.L = float(oc.propagation_distance)
        self.dA = float(
            self.system.dx
            * self.system.dy
        )

        self.R = int(self.basis.R)
        self.Nc = int(self.basis.B.shape[1])

        # Constant real part of G.
        if self.alpha == 0.0:
            self.zero_real_operator = True

            self.G_real_const = torch.zeros(
                (self.R, self.R),
                dtype=self.real_dtype,
                device=self.device,
            )
        else:
            self.zero_real_operator = False

            self.G_real_const = (
                self.B_plus.T
                @ (
                    -(self.alpha / 2.0)
                    * self.B.T
                )
            )

        # Constant diffraction part of Im(G).
        self.G_imag_diffraction = (
            self.B_plus.T
            @ (
                (1.0 / (2.0 * self.k0))
                * self.B_lap.T
            )
        )

    # ------------------------------------------------------------------
    # reduced operator and state-transition precomputation
    # ------------------------------------------------------------------

    def _G_parts_at_z(
        self,
        z: float,
        interval: int,
    ):
        """Return real and imaginary parts of G_k(z) on the selected device."""
        dn_np = self.system.turbulence.eval(
            self.basis.xc,
            self.basis.yc,
            z,
            interval,
        )

        dn = self._tensor(dn_np)

        # B_plus^T diag(dn) B^T without constructing diag(dn).
        turbulence_imag = (
            (
                self.B_plus.T
                * dn.unsqueeze(0)
            )
            @ self.B.T
        ) * self.k0

        G_i = (
            self.G_imag_diffraction
            + turbulence_imag
        )

        return self.G_real_const, G_i

    def _rhs(
        self,
        G_r,
        G_i,
        S_r,
        S_i,
    ):
        # Atmospheric attenuation is zero in the current main experiments,
        # so avoid two unnecessary matrix multiplications.
        if self.zero_real_operator:
            return (
                -(G_i @ S_i),
                G_i @ S_r,
            )

        return (
            G_r @ S_r - G_i @ S_i,
            G_r @ S_i + G_i @ S_r,
        )

    def _rk4_step(
        self,
        S_r,
        S_i,
        G0_r,
        G0_i,
        Gm_r,
        Gm_i,
        G1_r,
        G1_i,
        dz,
    ):
        k1_r, k1_i = self._rhs(
            G0_r,
            G0_i,
            S_r,
            S_i,
        )

        k2_r, k2_i = self._rhs(
            Gm_r,
            Gm_i,
            S_r + 0.5 * dz * k1_r,
            S_i + 0.5 * dz * k1_i,
        )

        k3_r, k3_i = self._rhs(
            Gm_r,
            Gm_i,
            S_r + 0.5 * dz * k2_r,
            S_i + 0.5 * dz * k2_i,
        )

        k4_r, k4_i = self._rhs(
            G1_r,
            G1_i,
            S_r + dz * k3_r,
            S_i + dz * k3_i,
        )

        S_r = S_r + (
            dz / 6.0
        ) * (
            k1_r
            + 2.0 * k2_r
            + 2.0 * k3_r
            + k4_r
        )

        S_i = S_i + (
            dz / 6.0
        ) * (
            k1_i
            + 2.0 * k2_i
            + 2.0 * k3_i
            + k4_i
        )

        return S_r, S_i

    def prepare_interval(
        self,
        interval: int,
    ):
        """Precompute T_k once for the current atmospheric interval."""
        if interval in self._transition_cache:
            return

        self._sync()
        t0 = time.perf_counter()

        dz = self.L / self.rk_steps

        # T(0)=I.  Propagate all basis vectors together.
        T_r = torch.eye(
            self.R,
            dtype=self.real_dtype,
            device=self.device,
        )

        T_i = torch.zeros(
            (self.R, self.R),
            dtype=self.real_dtype,
            device=self.device,
        )

        # Reuse G(z_i+dz) as the next step's G(z_{i+1}).
        G0_r, G0_i = self._G_parts_at_z(
            0.0,
            interval,
        )

        for step in range(self.rk_steps):
            z0 = step * dz
            zm = z0 + 0.5 * dz
            z1 = z0 + dz

            Gm_r, Gm_i = self._G_parts_at_z(
                zm,
                interval,
            )

            G1_r, G1_i = self._G_parts_at_z(
                z1,
                interval,
            )

            T_r, T_i = self._rk4_step(
                T_r,
                T_i,
                G0_r,
                G0_i,
                Gm_r,
                Gm_i,
                G1_r,
                G1_i,
                dz,
            )

            G0_r, G0_i = G1_r, G1_i

        self._sync()
        elapsed = time.perf_counter() - t0

        self._transition_cache[interval] = {
            "T_r": T_r,
            "T_i": T_i,
            "prepare_time_sec": float(elapsed),
        }

    # ------------------------------------------------------------------
    # steering-dependent initial reduced state
    # ------------------------------------------------------------------

    def _initial_state(
        self,
        theta_np,
    ):
        theta = self._tensor(
            np.asarray(
                theta_np,
                dtype=float,
            )
        )

        phase = self.k0 * (
            theta[0] * self.xc
            + theta[1] * self.yc
        )

        b_r = (
            self.gaussian_col
            * torch.cos(phase)
        )

        b_i = (
            self.gaussian_col
            * torch.sin(phase)
        )

        # c(0)
        c_r = self.B_plus.T @ b_r
        c_i = self.B_plus.T @ b_i

        # s_x(0) = Phi^+ j k0 x b
        sx_b_r = (
            -self.k0
            * self.xc
            * b_i
        )

        sx_b_i = (
            self.k0
            * self.xc
            * b_r
        )

        # s_y(0) = Phi^+ j k0 y b
        sy_b_r = (
            -self.k0
            * self.yc
            * b_i
        )

        sy_b_i = (
            self.k0
            * self.yc
            * b_r
        )

        sx_r = self.B_plus.T @ sx_b_r
        sx_i = self.B_plus.T @ sx_b_i

        sy_r = self.B_plus.T @ sy_b_r
        sy_i = self.B_plus.T @ sy_b_i

        S_r = torch.stack(
            [c_r, sx_r, sy_r],
            dim=1,
        )

        S_i = torch.stack(
            [c_i, sx_i, sy_i],
            dim=1,
        )

        return S_r, S_i

    def _initial_coefficients_only(
        self,
        theta_np,
    ):
        """Steering-dependent c(0) without constructing sensitivities."""
        theta = self._tensor(
            np.asarray(
                theta_np,
                dtype=float,
            )
        )

        phase = self.k0 * (
            theta[0] * self.xc
            + theta[1] * self.yc
        )

        b_r = (
            self.gaussian_col
            * torch.cos(phase)
        )

        b_i = (
            self.gaussian_col
            * torch.sin(phase)
        )

        return (
            self.B_plus.T @ b_r,
            self.B_plus.T @ b_i,
        )

    def propagate_reduced(
        self,
        theta_np,
        interval: int,
    ):
        """Apply precomputed T_k to [c(0), s_x(0), s_y(0)]."""
        self.prepare_interval(interval)

        cache = self._transition_cache[interval]

        T_r = cache["T_r"]
        T_i = cache["T_i"]

        S0_r, S0_i = self._initial_state(
            theta_np
        )

        SL_r = (
            T_r @ S0_r
            - T_i @ S0_i
        )

        SL_i = (
            T_r @ S0_i
            + T_i @ S0_r
        )

        return SL_r, SL_i

    def propagate_coefficients(
        self,
        theta_np,
        interval: int,
    ):
        """Propagate only c(L), used by line-search objective evaluations."""
        self.prepare_interval(interval)

        cache = self._transition_cache[interval]

        T_r = cache["T_r"]
        T_i = cache["T_i"]

        c0_r, c0_i = (
            self._initial_coefficients_only(
                theta_np
            )
        )

        cL_r = (
            T_r @ c0_r
            - T_i @ c0_i
        )

        cL_i = (
            T_r @ c0_i
            + T_i @ c0_r
        )

        return cL_r, cL_i

    # ------------------------------------------------------------------
    # reduced receiver operators
    # ------------------------------------------------------------------

    @staticmethod
    def _target_key(
        objective: str,
        target,
    ):
        target = np.asarray(
            target,
            dtype=float,
        )

        return (
            objective,
            round(float(target[0]), 15),
            round(float(target[1]), 15),
        )

    def _weighted_gram(
        self,
        indices_np,
        weights_np=None,
    ):
        """Phi W Phi^T over selected receiver-grid points."""
        indices = torch.as_tensor(
            np.asarray(indices_np),
            dtype=torch.long,
            device=self.device,
        )

        Phi = self.B_eval[:, indices]

        if weights_np is None:
            return (
                Phi @ Phi.T
            ) * self.dA

        weights = self._tensor(
            np.asarray(weights_np)
        )

        return (
            (
                Phi
                * weights.unsqueeze(0)
            )
            @ Phi.T
        ) * self.dA

    def _ensure_global_mass_matrix(self):
        if self._global_mass_matrix is not None:
            return

        self._sync()
        t0 = time.perf_counter()

        self._global_mass_matrix = (
            self.B_eval
            @ self.B_eval.T
        ) * self.dA

        self._sync()
        self._global_mass_prepare_time_sec = float(
            time.perf_counter() - t0
        )

    def prepare_target(
        self,
        target,
        objective: str | None = None,
    ):
        """Precompute the reduced receiver objective for one target position."""
        objective = (
            objective
            or self.cfg.tracking.objective
        )

        key = self._target_key(
            objective,
            target,
        )

        if key in self._target_cache:
            return

        self._sync()
        t0 = time.perf_counter()

        if objective == "power":
            mask = self.system.aperture_mask(
                target
            ).reshape(-1)

            idx = np.flatnonzero(
                mask > 0.5
            )

            M = self._weighted_gram(
                idx
            )

            payload = {
                "M": M,
            }

        elif objective == "coupling":
            self._ensure_global_mass_matrix()

            psi_np = self.system.smf_mode(
                target
            ).reshape(-1)

            psi = self._tensor(psi_np)

            g = (
                self.B_eval @ psi
            ) * self.dA

            mode_power = float(
                np.sum(
                    np.abs(psi_np) ** 2
                )
                * self.dA
            )

            payload = {
                "g": g,
                "M_total": self._global_mass_matrix,
                "mode_power": mode_power,
            }

        elif objective == "centroid":
            roi = self.system.centroid_roi_mask(
                target
            ).reshape(-1)

            idx = np.flatnonzero(
                roi > 0.5
            )

            x_sub = (
                self.system.X.reshape(-1)[idx]
            )

            y_sub = (
                self.system.Y.reshape(-1)[idx]
            )

            M0 = self._weighted_gram(
                idx
            )

            Mx = self._weighted_gram(
                idx,
                x_sub,
            )

            My = self._weighted_gram(
                idx,
                y_sub,
            )

            payload = {
                "M0": M0,
                "Mx": Mx,
                "My": My,
                "target": self._tensor(
                    np.asarray(
                        target,
                        dtype=float,
                    )
                ),
            }

        else:
            raise ValueError(
                "objective must be power, coupling, or centroid"
            )

        self._sync()
        elapsed = time.perf_counter() - t0

        payload["prepare_time_sec"] = float(
            elapsed
        )

        self._target_cache[key] = payload

    # ------------------------------------------------------------------
    # reduced objective algebra
    # ------------------------------------------------------------------

    @staticmethod
    def _quadratic_real_complex(
        M,
        a_r,
        a_i,
    ):
        """a^H M a for real symmetric M and complex a."""
        return (
            torch.dot(
                a_r,
                M @ a_r,
            )
            + torch.dot(
                a_i,
                M @ a_i,
            )
        )

    @staticmethod
    def _quadratic_derivative(
        M,
        c_r,
        c_i,
        s_r,
        s_i,
    ):
        """2 Re{s^H M c} for real symmetric M."""
        Mc_r = M @ c_r
        Mc_i = M @ c_i

        return 2.0 * (
            torch.dot(
                s_r,
                Mc_r,
            )
            + torch.dot(
                s_i,
                Mc_i,
            )
        )

    def objective_only(
        self,
        theta_np,
        interval: int,
        target,
    ):
        """Reduced scalar objective without steering-sensitivity propagation."""
        objective = (
            self.cfg.tracking.objective
        )

        self.prepare_target(
            target,
            objective,
        )

        key = self._target_key(
            objective,
            target,
        )

        target_ops = self._target_cache[
            key
        ]

        c_r, c_i = (
            self.propagate_coefficients(
                theta_np,
                interval,
            )
        )

        if objective == "power":
            Q = self._quadratic_real_complex(
                target_ops["M"],
                c_r,
                c_i,
            )

        elif objective == "coupling":
            g = target_ops["g"]
            M_total = target_ops[
                "M_total"
            ]
            C = target_ops[
                "mode_power"
            ]

            a_r = torch.dot(
                g,
                c_r,
            )

            a_i = torch.dot(
                g,
                c_i,
            )

            numerator = (
                a_r * a_r
                + a_i * a_i
            )

            B = self._quadratic_real_complex(
                M_total,
                c_r,
                c_i,
            )

            Q = (
                numerator
                / torch.clamp(
                    B * C,
                    min=1e-30,
                )
            )

        elif objective == "centroid":
            D = self._quadratic_real_complex(
                target_ops["M0"],
                c_r,
                c_i,
            )

            D = torch.clamp(
                D,
                min=1e-30,
            )

            Nx = self._quadratic_real_complex(
                target_ops["Mx"],
                c_r,
                c_i,
            )

            Ny = self._quadratic_real_complex(
                target_ops["My"],
                c_r,
                c_i,
            )

            target_t = target_ops[
                "target"
            ]

            px = Nx / D
            py = Ny / D

            Q = -(
                (px - target_t[0]) ** 2
                + (py - target_t[1]) ** 2
            )

        else:
            raise ValueError(
                "objective must be power, coupling, or centroid"
            )

        self._sync()

        return float(
            Q.detach().cpu().item()
        )

    def predicted_centroid(
        self,
        theta_np,
        interval: int,
        target,
    ):
        """Frozen-PINN predicted centroid in the same camera ROI as evaluation."""
        self.prepare_target(target, "centroid")
        ops = self._target_cache[self._target_key("centroid", target)]
        c_r, c_i = self.propagate_coefficients(theta_np, interval)
        D = self._quadratic_real_complex(ops["M0"], c_r, c_i)
        D = torch.clamp(D, min=1e-30)
        Nx = self._quadratic_real_complex(ops["Mx"], c_r, c_i)
        Ny = self._quadratic_real_complex(ops["My"], c_r, c_i)
        return np.array([
            float((Nx / D).detach().cpu().item()),
            float((Ny / D).detach().cpu().item()),
        ], dtype=float)

    def objective_and_gradient(
        self,
        theta_np,
        interval: int,
        target,
    ):
        objective = (
            self.cfg.tracking.objective
        )

        self.prepare_target(
            target,
            objective,
        )

        key = self._target_key(
            objective,
            target,
        )

        target_ops = self._target_cache[
            key
        ]

        S_r, S_i = self.propagate_reduced(
            theta_np,
            interval,
        )

        c_r = S_r[:, 0]
        c_i = S_i[:, 0]

        sx_r = S_r[:, 1]
        sx_i = S_i[:, 1]

        sy_r = S_r[:, 2]
        sy_i = S_i[:, 2]

        if objective == "power":
            M = target_ops["M"]

            Q = self._quadratic_real_complex(
                M,
                c_r,
                c_i,
            )

            gx = self._quadratic_derivative(
                M,
                c_r,
                c_i,
                sx_r,
                sx_i,
            )

            gy = self._quadratic_derivative(
                M,
                c_r,
                c_i,
                sy_r,
                sy_i,
            )

            grad = torch.stack(
                [gx, gy]
            )

        elif objective == "coupling":
            g = target_ops["g"]
            M_total = target_ops[
                "M_total"
            ]
            C = target_ops[
                "mode_power"
            ]

            a_r = torch.dot(
                g,
                c_r,
            )

            a_i = torch.dot(
                g,
                c_i,
            )

            numerator = (
                a_r * a_r
                + a_i * a_i
            )

            B = self._quadratic_real_complex(
                M_total,
                c_r,
                c_i,
            )

            denominator = torch.clamp(
                B * C,
                min=1e-30,
            )

            Q = (
                numerator
                / denominator
            )

            grad_list = []

            for s_r, s_i in (
                (sx_r, sx_i),
                (sy_r, sy_i),
            ):
                da_r = torch.dot(
                    g,
                    s_r,
                )

                da_i = torch.dot(
                    g,
                    s_i,
                )

                d_numerator = 2.0 * (
                    a_r * da_r
                    + a_i * da_i
                )

                dB = self._quadratic_derivative(
                    M_total,
                    c_r,
                    c_i,
                    s_r,
                    s_i,
                )

                dQ = (
                    d_numerator * B
                    - numerator * dB
                ) / torch.clamp(
                    B * B * C,
                    min=1e-30,
                )

                grad_list.append(
                    dQ
                )

            grad = torch.stack(
                grad_list
            )

        elif objective == "centroid":
            M0 = target_ops["M0"]
            Mx = target_ops["Mx"]
            My = target_ops["My"]
            target_t = target_ops[
                "target"
            ]

            D = self._quadratic_real_complex(
                M0,
                c_r,
                c_i,
            )

            D = torch.clamp(
                D,
                min=1e-30,
            )

            Nx = self._quadratic_real_complex(
                Mx,
                c_r,
                c_i,
            )

            Ny = self._quadratic_real_complex(
                My,
                c_r,
                c_i,
            )

            px = Nx / D
            py = Ny / D

            ex = px - target_t[0]
            ey = py - target_t[1]

            Q = -(
                ex * ex
                + ey * ey
            )

            grad_list = []

            for s_r, s_i in (
                (sx_r, sx_i),
                (sy_r, sy_i),
            ):
                dD = self._quadratic_derivative(
                    M0,
                    c_r,
                    c_i,
                    s_r,
                    s_i,
                )

                dNx = self._quadratic_derivative(
                    Mx,
                    c_r,
                    c_i,
                    s_r,
                    s_i,
                )

                dNy = self._quadratic_derivative(
                    My,
                    c_r,
                    c_i,
                    s_r,
                    s_i,
                )

                dpx = (
                    dNx * D
                    - Nx * dD
                ) / (
                    D * D
                )

                dpy = (
                    dNy * D
                    - Ny * dD
                ) / (
                    D * D
                )

                dQ = -2.0 * (
                    ex * dpx
                    + ey * dpy
                )

                grad_list.append(
                    dQ
                )

            grad = torch.stack(
                grad_list
            )

        else:
            raise ValueError(
                "objective must be power, coupling, or centroid"
            )

        self._sync()

        return (
            float(
                Q.detach().cpu().item()
            ),
            grad.detach().cpu().numpy().astype(
                float
            ),
        )

    # ------------------------------------------------------------------
    # optional full-field reconstruction (not used in online optimization)
    # ------------------------------------------------------------------

    def reconstruct_field(
        self,
        theta_np,
        interval: int,
    ):
        S_r, S_i = self.propagate_reduced(
            theta_np,
            interval,
        )

        U_r = (
            self.B_eval.T
            @ S_r[:, 0]
        ).reshape(
            self.system.X.shape
        )

        U_i = (
            self.B_eval.T
            @ S_i[:, 0]
        ).reshape(
            self.system.X.shape
        )

        return (
            U_r.detach().cpu().numpy()
            + 1j
            * U_i.detach().cpu().numpy()
        )

    # ------------------------------------------------------------------
    # controller
    # ------------------------------------------------------------------

    def solve(
        self,
        interval: int,
        target,
        theta_prev,
        history,
    ):
        self._sync()
        total_t0 = time.perf_counter()

        # These are online interval updates because the atmospheric state and
        # receiver position can change at every PAT interval.
        self.prepare_interval(interval)
        self.prepare_target(target, self.cfg.tracking.objective)

        self._sync()
        query_t0 = time.perf_counter()

        def objective_and_gradient(theta):
            return self.objective_and_gradient(
                theta,
                interval,
                target,
            )

        def objective_only(theta):
            return self.objective_only(
                theta,
                interval,
                target,
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

        self._sync()

        query_runtime = time.perf_counter() - query_t0
        total_runtime = time.perf_counter() - total_t0

        transition_info = (
            self._transition_cache[
                interval
            ]
        )

        target_key = self._target_key(
            self.cfg.tracking.objective,
            target,
        )

        target_info = (
            self._target_cache[
                target_key
            ]
        )

        diagnostics = {
            "device": str(self.device),
            "requested_dtype": self.requested_dtype,
            "effective_dtype": self.effective_dtype,
            "rk_steps": self.rk_steps,
            "basis_rank": int(self.basis.R),
            "boundary_residual": float(
                self.basis.boundary_residual
            ),
            "basis_setup_time_sec": float(
                self.basis_setup_time_sec
            ),
            "transition_prepare_time_sec": float(
                transition_info[
                    "prepare_time_sec"
                ]
            ),
            "target_prepare_time_sec": float(
                target_info[
                    "prepare_time_sec"
                ]
            ),
            "initial_field_relative_error": float(
                self.initial_field_relative_error
            ),
            "online_full_grid_reconstruction": False,
        }

        diagnostics.update(opt_diag)
        diagnostics["query_runtime_sec"] = float(query_runtime)
        diagnostics["online_total_runtime_sec"] = float(total_runtime)

        if self.cfg.tracking.objective == "centroid":
            diagnostics["predicted_centroid"] = self.predicted_centroid(
                theta, interval, target
            ).tolist()

        return {
            "theta": theta,
            "runtime_sec": total_runtime,
            "predicted_metric": predicted_metric,
            "diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def _build_diagnostics(self):
        zero = np.zeros(
            2,
            dtype=float,
        )

        S_r, S_i = self._initial_state(
            zero
        )

        U0_r = (
            self.B_eval.T
            @ S_r[:, 0]
        ).reshape(
            self.system.X.shape
        )

        U0_i = (
            self.B_eval.T
            @ S_i[:, 0]
        ).reshape(
            self.system.X.shape
        )

        U0_hat = (
            U0_r.detach().cpu().numpy()
            + 1j
            * U0_i.detach().cpu().numpy()
        )

        U0_true = (
            self.system.gaussian_beam(
                self.system.X,
                self.system.Y,
            )
        )

        self.initial_field_relative_error = float(
            np.linalg.norm(
                U0_hat - U0_true
            )
            / max(
                np.linalg.norm(
                    U0_true
                ),
                1e-30,
            )
        )
