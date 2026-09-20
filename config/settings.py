from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass
class OpticalConfig:
    wavelength: float = 1550e-9
    transmit_power: float = 1.0
    beam_waist: float = 2.5e-3
    propagation_distance: float = 20.0
    attenuation: float = 0.0
    half_width: float = 12e-3
    grid_size: int = 64
    aperture_radius: float = 2.4e-3
    smf_mode_waist: float = 1.7e-3
    centroid_roi_radius: float = 4.5e-3

    @property
    def k0(self) -> float:
        import numpy as np
        return 2.0 * np.pi / self.wavelength


@dataclass
class TurbulenceConfig:
    delta_n_rms: float = 8e-9
    num_modes: int = 10
    shift_per_interval: float = 0.40
    seed: int = 7


@dataclass
class FrozenPINNConfig:
    sampler: str = "elm"          # elm | swim
    backend: str = "auto"         # auto | scipy | torch
    device: str = "auto"          # auto | cpu | cuda | mps
    dtype: str = "float64"        # MPS automatically uses float32
    rk_steps: int = 64
    hidden_width: int = 1000
    collocation_side: int = 24
    boundary_side: int = 16
    elm_bias_range: float = 2.0
    svd_cutoff: float = 1e-6
    pinv_rcond: float = 1e-6
    ode_rtol: float = 1e-5
    ode_atol: float = 1e-8
    num_opt_iterations: int = 5
    theta_step_max: float = 30e-6
    theta_max: float = 250e-6
    line_search_steps: int = 7
    lambda_theta: float = 0.0
    seed: int = 7


@dataclass
class BaselineConfig:
    theta_max: float = 250e-6

    # Common device/precision for the differentiable SSFM oracle.
    ssfm_device: str = "auto"      # auto | cpu | cuda | mps
    ssfm_dtype: str = "float64"    # MPS automatically uses float32

    pid_kp: float = 0.80
    pid_ki: float = 8.0
    pid_kd: float = 1e-3
    mpc_rho: float = 0.15
    mpc_theta_step_max: float = 40e-6

    # Retained for the optional legacy NumPy finite-difference oracle.
    oracle_num_iterations: int = 5
    oracle_fd_step: float = 2e-6
    oracle_theta_step_max: float = 30e-6
    oracle_line_search_steps: int = 6


@dataclass
class TrackingConfig:
    objective: str = "power"       # power | coupling | centroid
    num_intervals: int = 8

    # Physical time. num_intervals controls only simulation duration; it does
    # not change the target speed or trajectory period.
    control_interval_sec: float = 0.01
    target_period_sec: float = 2.0
    target_x_amplitude: float = 2.4e-3
    target_y_amplitude: float = 1.8e-3
    target_y_phase: float = 0.0

    ssfm_steps: int = 24

    # Physical FSM constraints.  Per-interval slew is derived from the physical
    # slew rate and control interval, unless an explicit override is provided.
    theta_max: float = 250e-6
    theta_slew_rate: float = 18e-3
    theta_slew_max_override: float | None = None

    @property
    def theta_slew_max(self) -> float:
        if self.theta_slew_max_override is not None:
            return float(self.theta_slew_max_override)
        return float(self.theta_slew_rate * self.control_interval_sec)

    # Shared numerical optimizer used by Frozen-PINN and SSFM oracle.
    max_opt_iterations: int = 20
    optimizer_step: float = 40e-6
    line_search_steps: int = 10
    optimizer_min_step: float = 1e-9
    gradient_norm_tol: float = 1e-12
    objective_rel_tol: float = 1e-6
    objective_abs_tol: float = 1e-12
    control_regularization: float = 0.0


@dataclass
class ExperimentConfig:
    name: str = "pat_comparison"
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    frozen_pinn: FrozenPINNConfig = field(default_factory=FrozenPINNConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
