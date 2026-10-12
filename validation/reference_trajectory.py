"""Replay the reference repository's periodic angular law in PSD coordinates.

The upstream figure eight is a transmitter-angle disturbance, not a moving
PSD reference. For receiver commissioning only, map that normalized temporal
law to a measured PSD reference using one common scale for both axes. Actual
hardware advances after measured arrival, so its time scale is different.
"""
from __future__ import annotations

import numpy as np

# receiver_pinn resolves and imports the pinned upstream code directory.
from receiver_pinn import CalibrationError
from config.settings import make_preset
from system_model.optical_system import OpticalPATSystem


def github_periodic_reference(amplitude_mm=.35):
    if not np.isfinite(amplitude_mm) or not 0 < amplitude_mm <= .35:
        raise CalibrationError('Use a common PSD amplitude at or below 0.35 mm')
    cfg = make_preset('demo')
    cfg.tracking.disturbance = 'periodic'
    period = cfg.tracking.target_period_sec
    interval = cfg.tracking.control_interval_sec
    count = int(round(period/interval))
    if not np.isclose(count*interval, period) or cfg.optical.tx_angular_std <= 0:
        raise CalibrationError('The upstream periodic time grid is not replayable')
    # Evaluate the original method without allocating or running atmosphere/
    # detector simulation. tx_angle/time_at_interval depend only on cfg.
    angular_law = OpticalPATSystem.__new__(OpticalPATSystem)
    angular_law.cfg = cfg
    angles = np.array([angular_law.tx_angle(k) for k in range(count+1)])
    source_amplitude = float(np.sqrt(2)*cfg.optical.tx_angular_std)
    scale = amplitude_mm/source_amplitude
    positions = angles*scale
    provenance = {
        'reference_repository': 'https://github.com/seungeuibyun/PAT_control_framework',
        'reference_commit': '1192e7e92b885bfcb72e1c712948c8624cd3b3b5',
        'function': 'system_model.optical_system.OpticalPATSystem.tx_angle',
        'upstream_disturbance_mode': 'periodic',
        'upstream_role': 'Transmitter angular disturbance; communication detector target stays fixed',
        'hardware_role': 'Scaled PSD position reference for receiver commissioning',
        'upstream_period_sec': period, 'upstream_sample_interval_sec': interval,
        'upstream_angular_amplitude_rad': source_amplitude,
        'upstream_angle_samples_rad': angles.tolist(),
        'common_coordinate_scale_mm_per_rad': scale,
        'PSD_half_amplitude_mm': amplitude_mm, 'same_scale_for_both_axes': True,
        'PSD_reference_samples_mm': positions.tolist(),
        'hardware_timing': 'Advance only after actual PSD arrival; original 0.5 s timing is not reproduced',
        'upstream_full_power_optimization_experiment_reproduced': False,
    }
    return positions, provenance


class GitHubPeriodicTrajectory:
    def __init__(self, amplitude_mm=.35, hardware_period_sec=12.):
        self.points, self.provenance = github_periodic_reference(amplitude_mm)
        if not np.isfinite(hardware_period_sec) or hardware_period_sec < 8:
            raise CalibrationError('Continuous commissioning requires a period of at least 8 seconds')
        cfg = make_preset('demo')
        cfg.tracking.disturbance = 'periodic'
        self.law = OpticalPATSystem.__new__(OpticalPATSystem)
        self.law.cfg = cfg
        self.period = float(hardware_period_sec)
        self.scale = self.provenance['common_coordinate_scale_mm_per_rad']
        self.provenance.update(hardware_period_sec=self.period,
            time_scale_factor=self.period/self.provenance['upstream_period_sec'],
            hardware_timing='Continuous phase from a monotonic clock; common time dilation on both axes')

    def position(self, elapsed_sec):
        simulated_time = elapsed_sec/self.period*self.provenance['upstream_period_sec']
        interval = simulated_time/self.provenance['upstream_sample_interval_sec']
        return self.scale*self.law.tx_angle(interval)
