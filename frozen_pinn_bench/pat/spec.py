"""Load and validate config/bench.json.

Derived quantities (beam radius at the FSM, FSM poles, sample counts) are
computed here once so every module uses the same numbers.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT/'config/bench.json'


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in value.items()})
    return value


def _steps(seconds, dt, name):
    steps = seconds/dt
    if abs(steps-round(steps)) > 1e-9 or round(steps) < 1:
        raise ValueError(f'{name} must be a positive multiple of the simulation step')
    return int(round(steps))


class BenchSpec:
    def __init__(self, raw):
        self.raw = raw
        for key in ('optics', 'fsm', 'psd', 'calibration', 'simulation', 'controllers', 'evaluation'):
            setattr(self, key, _ns(raw[key]))
        self.families = raw['disturbance']['families']
        o, f, p, s = self.optics, self.fsm, self.psd, self.simulation
        self.dt = s.dt_s
        self.k0 = 2*math.pi/o.wavelength_m
        # Collimator waist x expander, then vacuum propagation to the FSM.
        w0 = o.collimated_waist_diameter_m/2*o.expander_magnification
        zr = math.pi*w0**2/o.wavelength_m
        z = o.tx_to_fsm_m
        self.beam_radius_at_fsm = w0*math.sqrt(1+(z/zr)**2)
        self.beam_curvature_at_fsm = math.inf if z == 0 else z*(1+(zr/z)**2)
        self.rayleigh_range = zr
        self.focal_length = o.lens_focal_length_m
        # Receiver geometry. 'focal' (legacy/pat_bench_v2): PSD/PDA in the lens back focal plane, both levers = f.
        # 'direct' (as built): Tx expander -> link -> Rx reducer, no focusing lens; PSD/PDA see the beam
        # position directly at their distances from the FSM, and a Tx translation D shifts the
        # detector-plane beam by D / reducer magnification without any angle.
        receiver = raw.get('receiver', {'mode': 'focal'})
        self.receiver = _ns(receiver)
        self.mode = receiver['mode']
        if self.mode == 'focal':
            self.psd_lever = self.pda_lever = self.focal_length
            self.pda_capture_peak = 1.0
        elif self.mode == 'direct':
            r = self.receiver
            self.psd_lever, self.pda_lever = r.fsm_to_psd_m, r.fsm_to_pda_m
            self.reducer = r.reducer_magnification
            self.beam_radius_at_fsm = r.beam_radius_at_detectors_m
            self.beam_curvature_at_fsm = math.inf
            self.pda_capture_peak = 1-math.exp(-2*(o.pda_radius_m/r.beam_radius_at_detectors_m)**2)
            self.rail = _ns(raw['rail'])
        else:
            raise ValueError(f'Unknown receiver mode {self.mode}')
        self.fsm_gain = np.asarray(o.fsm_optical_gain, dtype=float)
        self.omega = 2*np.pi*np.asarray(f.bandwidth_hz, dtype=float)
        # Low-frequency group delay of the second-order closed loop: 2 zeta / omega_n.
        self.fsm_group_delay = 2*f.damping/self.omega
        self.lpf_tau = 1/(2*np.pi*p.analog_lpf_hz)
        self.tick_steps = _steps(f.command_period_s, self.dt, 'command_period_s')
        self.latency_steps = _steps(f.command_latency_s, self.dt, 'command_latency_s')
        self.frame_steps = _steps(p.frame_period_s, self.dt, 'frame_period_s')
        self.capture_steps = _steps(p.capture_after_query_s, self.dt, 'capture_after_query_s')
        self.delivery_steps = _steps(p.delivery_after_query_s, self.dt, 'delivery_after_query_s')
        self.assumed_capture_steps = _steps(self.controllers.assumed_capture_after_query_s, self.dt,
                                            'assumed_capture_after_query_s')
        self.warmup_steps = int(round(s.warmup_s/self.dt))
        self.psd_lsb = o.psd_width_m/2**p.adc_bits
        self.validate()

    def validate(self):
        o, f, p = self.optics, self.fsm, self.psd
        if self.fsm_gain.shape != (2, 2) or abs(np.linalg.det(self.fsm_gain)) < 1e-6:
            raise ValueError('FSM optical gain must be an invertible 2x2 matrix')
        if not 0 < f.damping < 1.5 or np.any(self.omega <= 0):
            raise ValueError('Invalid FSM loop model')
        if self.delivery_steps < self.capture_steps or self.delivery_steps != self.frame_steps:
            raise ValueError('PSD must deliver one frame per query, after capture, at the 50 Hz cadence')
        if p.frame_period_s < 1/50 - 1e-12:
            raise ValueError('CONEX-PSD USB rate is limited to 50 Hz')
        if self.frame_steps % self.tick_steps:
            raise ValueError('PSD deliveries must coincide with command ticks')
        if self.beam_radius_at_fsm*2 > o.fsm_mirror_diameter_m*math.cos(math.radians(o.fsm_aoi_deg)):
            raise ValueError('Beam footprint exceeds the projected MR-15-30 aperture')
        if not (o.splitter_reflection > 0 and o.splitter_transmission > 0
                and o.splitter_reflection+o.splitter_transmission <= 1):
            raise ValueError('Invalid splitter')
        peak_v = o.power_at_fsm_w*o.splitter_transmission*o.pda_branch_nd_transmission \
            * self.pda_capture_peak*o.pda_responsivity_a_per_w*o.pda_gain_v_per_a
        if peak_v >= o.pda_max_output_v:
            raise ValueError(f'PDA015C2 saturates ({peak_v:.2f} V); add attenuation')
        self.pda_peak_voltage = peak_v
        if self.mode == 'direct':
            r = self.rail
            if min(r.travel_m, r.max_speed_m_s, r.max_accel_m_s2) <= 0:
                raise ValueError('Rail limits must be positive')
            if self.reducer <= 1 or min(self.psd_lever, self.pda_lever) <= 0:
                raise ValueError('Invalid direct-receiver geometry')

    def episodes(self, name):
        cfg = self.raw['evaluation']['episodes'][name]
        return cfg['family'], [cfg['seed0']+i for i in range(cfg['count'])], cfg['duration_s']

    def variant(self, **changes):
        """Copy with dotted-path overrides, e.g. variant(**{'fsm.jitter_rms_rad': 5e-6})."""
        raw = json.loads(json.dumps(self.raw))
        for path, value in changes.items():
            node = raw
            *parents, leaf = path.split('.')
            for key in parents:
                node = node[key]
            if leaf not in node:
                raise KeyError(path)
            node[leaf] = value
        return BenchSpec(raw)


def load_spec(path=DEFAULT_PATH):
    return BenchSpec(json.loads(Path(path).read_text()))
