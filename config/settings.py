"""SI units. Defaults follow PAT (1).pdf Tables I-II; demo reduces numerics only."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math
import numpy as np


@dataclass
class OpticalConfig:
    wavelength: float = 1550e-9
    transmit_power: float = 10e-3
    beam_waist: float = 50e-3
    propagation_distance: float = 20e3
    path_loss_db: float = 1.0
    half_width: float = 1.0
    grid_size: int = 1024
    tx_displacement: tuple = (0.0, 0.0)
    tx_angular_std: float = 5e-6
    entrance_diameter: float = 100e-3
    reducer_magnification: float = 0.1
    reducer_transmission: float = 0.97
    reducer_phase_curvature: float = 0.0  # phi_G = coefficient * |r|^2
    receiver_grid_size: int = 1024
    receiver_half_width: float = 6e-3
    focal_length: float = 200e-3
    fsm_to_lens: float = 50e-3
    lens_to_splitter: float = 50e-3
    splitter_to_psd: float = 150e-3
    splitter_to_detector: float = 150e-3
    splitter_transmission: float = 0.62
    splitter_reflection: float = 0.27
    splitter_phase: float = 0.0
    fsm_calibration: tuple = ((2.0, 0.0), (0.0, 2.0))
    psd_width: float = 10e-3
    psd_center: tuple = (0.0, 0.0)
    detector_center: tuple = (0.0, 0.0)
    aperture_radius: float = 75e-6
    detector_sampling: float = 2e-6
    detector_view_half_width: float = 450e-6
    psd_padding: int = 2
    psd_position_noise_std: float = 5e-6
    psd_power_noise_std: float = 0.0  # not specified in Table I
    psd_min_power: float = 1e-12

    @property
    def k0(self):
        return 2 * math.pi / self.wavelength

    @property
    def attenuation(self):
        return self.path_loss_db * math.log(10) / (10 * self.propagation_distance)


@dataclass
class TurbulenceConfig:
    ground_cn2: float = 1.7e-14
    hv_wind_speed: float = 21.0
    outer_scale: float = 10.0
    inner_scale: float = 5e-3
    strength_scale: float = 1.0
    num_modes: int = 128
    layer_thickness: float = 50.0
    seed: int = 7


@dataclass
class FrozenPINNConfig:
    backend: str = "auto"
    operator_backend: str = "auto"  # GPU layer projection; CPU matrix-free reference
    device: str = "cpu"
    dtype: str = "float64"
    hidden_width: int = 1024
    collocation_side: int = 128
    spectral_side: int = 48  # bandwidth of fixed boundary transform
    feature_scale_max: float = 80.0
    boundary_envelope_fraction: float = 0.35
    svd_cutoff: float = 1e-6
    ode_rtol: float = 1e-6
    ode_atol: float = 1e-8
    seed: int = 7


@dataclass
class BaselineConfig:
    ssfm_device: str = "cpu"
    ssfm_dtype: str = "float64"
    pid_kp: float = 0.80
    pid_ki: float = 8.0
    pid_kd: float = 1e-3
    mpc_rho: float = 0.15  # penalty in calibrated angular coordinates
    oracle_fd_step: float = 0.1e-6


@dataclass
class TrackingConfig:
    objective: str = "power"
    num_intervals: int = 20
    control_interval_sec: float = 0.01
    disturbance: str = "iid"  # temporal law unspecified in PDF
    target_period_sec: float = 0.5  # optional periodic Tx angular disturbance
    ssfm_steps: int = 400
    theta_max: float = 1e-3
    theta_slew_rate: float = 10e-3
    theta_slew_max_override: float | None = None
    theta_quantization: float = 1e-6
    initial_theta: tuple = (0.0, 0.0)
    max_opt_iterations: int = 20
    optimizer_step: float = 50e-6
    line_search_shrink: float = 0.5
    line_search_steps: int = 10
    optimizer_min_step: float = 1e-9
    gradient_norm_tol: float = 1e-12
    objective_rel_tol: float = 1e-6
    objective_abs_tol: float = 1e-12
    control_regularization: float = 0.0

    @property
    def theta_slew_max(self):
        if self.theta_slew_max_override is not None:
            return self.theta_slew_max_override
        return np.asarray(self.theta_slew_rate) * self.control_interval_sec


@dataclass
class ExperimentConfig:
    name: str = "paper_power_comparison"
    preset: str = "paper"
    model_version: str = "receiver_fsm_pdf_v1"
    reference_document: str = "PAT (1).pdf, Sections II-IV, Tables I-II"
    channel_realizations: int = 100
    feature_seeds: int = 5
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    frozen_pinn: FrozenPINNConfig = field(default_factory=FrozenPINNConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Load current-model results without reinterpreting legacy experiments."""
        if data.get("model_version") != "receiver_fsm_pdf_v1":
            raise ValueError("Replot requires receiver_fsm_pdf_v1 results; legacy physics cannot be replayed with this model")
        values = dict(data)
        for key, constructor in (("optical", OpticalConfig), ("turbulence", TurbulenceConfig),
                                 ("frozen_pinn", FrozenPINNConfig), ("baselines", BaselineConfig),
                                 ("tracking", TrackingConfig)):
            values[key] = constructor(**values[key])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self):
        o, t, f, a = self.optical, self.tracking, self.frozen_pinn, self.turbulence
        if f.operator_backend not in {"auto", "matrix_free", "projected"}:
            raise ValueError("operator_backend must be auto, matrix_free or projected")
        if t.objective != "power":
            raise ValueError("The PDF objective is communication-detector power (power).")
        positive = dict(wavelength=o.wavelength, transmit_power=o.transmit_power,
            beam_waist=o.beam_waist, distance=o.propagation_distance,
            half_width=o.half_width, focal_length=o.focal_length,
            entrance_diameter=o.entrance_diameter, detector_radius=o.aperture_radius,
            sampling=o.detector_sampling, psd_width=o.psd_width,
            control_interval=t.control_interval_sec, quantization=t.theta_quantization,
            layer_thickness=a.layer_thickness, inner_scale=a.inner_scale,
            outer_scale=a.outer_scale, period=t.target_period_sec,
            optimizer_step=t.optimizer_step, ode_rtol=f.ode_rtol, ode_atol=f.ode_atol)
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        counts = dict(grid_size=o.grid_size, receiver_grid_size=o.receiver_grid_size,
            collocation_side=f.collocation_side, spectral_side=f.spectral_side,
            hidden_width=f.hidden_width, ssfm_steps=t.ssfm_steps,
            num_intervals=t.num_intervals, num_modes=a.num_modes,
            max_opt_iterations=t.max_opt_iterations,
            line_search_steps=t.line_search_steps, psd_padding=o.psd_padding)
        for name, value in counts.items():
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if f.spectral_side > f.collocation_side or not 0 < f.svd_cutoff < 1:
            raise ValueError("Require spectral_side <= collocation_side and 0 < svd_cutoff < 1")
        if f.feature_scale_max < 5 or f.boundary_envelope_fraction <= 0:
            raise ValueError("Require feature_scale_max >= 5 and a positive boundary envelope")
        if not 0 < abs(o.reducer_magnification) < 1 or not 0 < o.reducer_transmission <= 1:
            raise ValueError("Invalid beam-reducer magnification/transmission")
        if min(o.splitter_reflection, o.splitter_transmission) <= 0 or o.splitter_reflection + o.splitter_transmission > 1:
            raise ValueError("Require positive T_BS, R_BS with T_BS + R_BS <= 1")
        if min(o.path_loss_db, o.tx_angular_std, o.psd_position_noise_std,
               o.psd_power_noise_std, a.strength_scale, a.ground_cn2, a.hv_wind_speed) < 0:
            raise ValueError("Loss, noise and turbulence parameters must be nonnegative")
        if t.disturbance not in {"iid", "periodic", "zero"}:
            raise ValueError("disturbance must be iid, periodic or zero")
        if not 0 < t.line_search_shrink < 1:
            raise ValueError("line_search_shrink must lie in (0,1)")
        if np.any(np.asarray(t.theta_max) <= 0) or np.any(np.asarray(t.theta_slew_max) < 0):
            raise ValueError("Steering range must be positive and slew nonnegative")
        initial = np.asarray(t.initial_theta)
        if np.any(abs(initial) > t.theta_max) or not np.allclose(initial / t.theta_quantization, np.round(initial / t.theta_quantization), atol=1e-9, rtol=0):
            raise ValueError("Initial command must satisfy steering range and quantization")
        if np.linalg.matrix_rank(o.fsm_calibration) < 2:
            raise ValueError("FSM calibration must have rank two")
        for length in (o.splitter_to_detector, o.splitter_to_psd):
            if not np.isclose(o.lens_to_splitter + length, o.focal_length):
                raise ValueError("Table I implementation requires both detectors in the back focal plane")
        if o.receiver_half_width < abs(o.reducer_magnification) * o.entrance_diameter / 2:
            raise ValueError("Reduced aperture must fit the receiver grid")
        support = o.wavelength * o.focal_length / (4 * o.receiver_half_width / o.receiver_grid_size)
        if support < o.psd_width / 2 + max(abs(np.asarray(o.psd_center))):
            raise ValueError("Increase receiver_grid_size to cover the complete PSD without Fourier aliasing")


def make_preset(name="paper"):
    cfg = ExperimentConfig(preset=name)
    if name == "demo":
        cfg.name = "pdf_power_demo"
        cfg.optical.grid_size = 256
        cfg.optical.receiver_grid_size = 512
        cfg.frozen_pinn.hidden_width = 1024
        cfg.frozen_pinn.collocation_side = 96
        cfg.turbulence.num_modes = 32
        cfg.turbulence.layer_thickness = 500.0
        cfg.tracking.ssfm_steps = 40
        cfg.channel_realizations = cfg.feature_seeds = 1
    elif name != "paper":
        raise ValueError("preset must be demo or paper")
    return cfg
