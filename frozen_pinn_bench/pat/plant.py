"""Batched bench loop on a 0.5 ms grid.

Per step:  PSD frame delivery -> controller tick (every 2 ms) -> command applied
after the USB latency -> MR-15-30 closed-loop dynamics + jitter + static offset
-> focal spot = f (C_F theta + alpha) -> 175 Hz PSD filter -> capture/quantize.

Every controller sees only the delivered PSD frames, its own commands, the
nominal models and the calibrated target. Disturbance, true C_F, offset,
jitter and noise stay in EpisodeSet and are used only by the plant and metrics.
Episodes are generated from seeds, so all controllers see identical noise.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter

from pat.disturbance import make_episode


@dataclass
class EpisodeSet:
    family: str
    seeds: list
    duration: float
    alpha: np.ndarray
    params: list
    fsm_offset: np.ndarray
    fsm_jitter: np.ndarray
    psd_noise: np.ndarray
    gain_true: np.ndarray
    gain_model: np.ndarray
    target: np.ndarray

    @property
    def count(self):
        return len(self.seeds)


def _closed_loop_noise(rng, n, dt, omega, zeta, rms):
    freqs = 2*np.pi*np.fft.rfftfreq(n, dt)
    gain = omega**4/((omega**2-freqs**2)**2+(2*zeta*omega*freqs)**2)
    spectrum = np.sqrt(gain)*(rng.standard_normal(freqs.size)+1j*rng.standard_normal(freqs.size))
    spectrum[0] = 0
    series = np.fft.irfft(spectrum, n)
    return series*rms/np.std(series) if rms > 0 else np.zeros(n)


def build_episodes(spec, family_name, seeds, duration):
    """family_name may be a list: episodes then cycle through the families by seed order."""
    names = family_name if isinstance(family_name, (list, tuple)) else [family_name]
    n = int(round(duration/spec.dt))
    frames = n//spec.frame_steps+2
    f, c = spec.fsm, spec.calibration
    alphas, params, offsets, jitters, noises, gains, targets = [], [], [], [], [], [], []
    for i, seed in enumerate(seeds):
        family = spec.families[names[i % len(names)]]
        alpha, p = make_episode(family, seed, duration, spec.dt, spec)
        rng = np.random.default_rng([seed, 1])
        offsets.append(rng.normal(0, f.static_offset_rms_rad, 2))
        jitters.append(np.stack([_closed_loop_noise(rng, n, spec.dt, spec.omega[a], f.damping, f.jitter_rms_rad)
                                 for a in range(2)], axis=1))
        noises.append(rng.normal(0, spec.psd.position_noise_rms_m, (frames, 2)))
        error = np.diag(rng.normal(0, c.fsm_gain_error_rms, 2))
        error[0, 1], error[1, 0] = rng.normal(0, c.fsm_coupling_error_rms, 2)
        gains.append((np.eye(2)+error)@spec.fsm_gain)
        targets.append(rng.normal(0, c.target_error_rms_m, 2))
        alphas.append(alpha)
        params.append(p)
    return EpisodeSet(family_name, list(seeds), duration, np.stack(alphas), params, np.stack(offsets),
                      np.stack(jitters), np.stack(noises), np.stack(gains),
                      np.repeat(spec.fsm_gain[None], len(seeds), axis=0), np.stack(targets))


def fsm_discretization(spec):
    """Exact ZOH discretization of each axis: theta'' = w^2 (u - theta) - 2 zeta w theta'."""
    ads, bds = [], []
    for omega in spec.omega:
        a = np.array([[0.0, 1.0], [-omega**2, -2*spec.fsm.damping*omega]])
        b = np.array([0.0, omega**2])
        ad = expm(a*spec.dt)
        ads.append(ad)
        bds.append(np.linalg.solve(a, (ad-np.eye(2))@b))
    return np.stack(ads), np.stack(bds)


@dataclass
class LoopContext:
    """What a controller may know: nominal models, calibration and timing only."""
    spec: object
    batch: int
    gain_model: np.ndarray
    target: np.ndarray


@dataclass
class PowerModel:
    """Normalized-power response evaluated on the PDA-plane offset ('pda') or on the
    PSD-plane residual scaled to another receiver ('psd', e.g. PDF focal-plane equivalent)."""
    name: str
    response: object
    source: str = 'pda'
    scale: float = 1.0


def as_power_models(models):
    return [m if isinstance(m, PowerModel) else PowerModel(f'power_{i}', m) for i, m in enumerate(models)]


class Metrics:
    """Accumulates residual/power statistics after warmup without storing all traces."""
    def __init__(self, batch, models, outage_fraction):
        self.models = as_power_models(models)
        self.outage_fraction = outage_fraction
        self.sum_sq = np.zeros((batch, 2))
        self.count = 0
        self.power = np.zeros((batch, len(self.models)))
        self.outage = np.zeros((batch, len(self.models)))
        self.peak = np.zeros(batch)
        self._chunk = []
        self._pda = []

    def add(self, residual, pda_offset=None):
        self._chunk.append(residual)
        self._pda.append(residual if pda_offset is None else pda_offset)
        if len(self._chunk) >= 2000:
            self.flush()

    def flush(self):
        if not self._chunk:
            return
        r = np.stack(self._chunk, axis=1)
        pda = np.stack(self._pda, axis=1)
        self._chunk, self._pda = [], []
        self.sum_sq += np.sum(r**2, axis=1)
        self.count += r.shape[1]
        self.peak = np.maximum(self.peak, np.max(np.hypot(r[..., 0], r[..., 1]), axis=1))
        for i, model in enumerate(self.models):
            power = model.response.normalized_power(pda if model.source == 'pda' else r*model.scale)
            self.power[:, i] += power.sum(axis=1)
            self.outage[:, i] += (power < self.outage_fraction).sum(axis=1)

    def result(self):
        self.flush()
        if self.count == 0:
            nan = np.full(self.sum_sq.shape[0], np.nan)
            return dict(rms_focal_m=nan, rms_focal_axis_m=np.full_like(self.sum_sq, np.nan), peak_focal_m=nan,
                        mean_power=np.full_like(self.power, np.nan), outage=np.full_like(self.outage, np.nan))
        return dict(rms_focal_m=np.sqrt(self.sum_sq.sum(axis=1)/self.count),
                    rms_focal_axis_m=np.sqrt(self.sum_sq/self.count),
                    peak_focal_m=self.peak, mean_power=self.power/self.count,
                    outage=self.outage/self.count)


def open_loop_frames(spec, episodes, capture_steps=None):
    """PSD frames with the FSM held at zero (bench: record while injecting, no control).
    Readings are PSD-plane positions; divide by spec.psd_lever for FSM-plane angle units.

    Vectorized form of simulate(HoldController); returns (E, F, 2) readings and capture steps.
    """
    capture = spec.capture_steps if capture_steps is None else capture_steps
    theta = episodes.fsm_offset[:, None]+episodes.fsm_jitter
    focal = spec.psd_lever*(np.einsum('eij,etj->eti', episodes.gain_true, theta)+episodes.alpha)
    g = 1-np.exp(-spec.dt/spec.lpf_tau)
    filtered, _ = lfilter([g], [1, -(1-g)], focal, axis=1, zi=((1-g)*focal[:, :1]))
    steps = np.arange(capture, focal.shape[1], spec.frame_steps)
    frames = np.arange(len(steps))
    # Frames are delivered one period after the query, so the last capture may be undelivered.
    delivered = steps+spec.frame_steps-capture < focal.shape[1]
    steps, frames = steps[delivered], frames[delivered]
    readings = filtered[:, steps]+episodes.psd_noise[:, frames]
    return np.round(readings/spec.psd_lsb)*spec.psd_lsb, steps


def simulate(spec, episodes, controller, optics_list, *, batch_index=None, trace_index=(),
             capture_steps=None, keep_residual=False):
    """Run one controller on episodes[batch_index]. capture_steps overrides the true PSD capture."""
    idx = np.arange(episodes.count) if batch_index is None else np.asarray(batch_index)
    batch, steps = len(idx), episodes.alpha.shape[1]
    f, f_pda = spec.psd_lever, spec.pda_lever
    ad, bd = fsm_discretization(spec)
    state = np.zeros((batch, 2, 2))
    command = np.zeros((batch, 2))
    pending = []
    alpha, jitter = episodes.alpha, episodes.fsm_jitter
    offset, gain = episodes.fsm_offset[idx], episodes.gain_true[idx]
    lpf_gain = 1-np.exp(-spec.dt/spec.lpf_tau)
    capture = spec.capture_steps if capture_steps is None else capture_steps
    context = LoopContext(spec, batch, episodes.gain_model[idx].copy(), episodes.target[idx].copy())
    controller.reset(context)
    metrics = Metrics(batch, optics_list, spec.raw['evaluation']['outage_fraction_of_aligned_power'])
    traces = {int(i): dict(focal=[], command=[], frames=[]) for i in trace_index}
    residual = np.zeros((batch, steps, 2), dtype=np.float32) if keep_residual else None
    frame = None
    lpf = None
    software_range = spec.fsm.software_range_rad
    tick_time = 0.0
    ticks = 0
    for n in range(steps):
        m, phase = divmod(n, spec.frame_steps)
        if phase == 0 and frame is not None:
            reading, frame_index = frame
            controller.on_frame(reading, frame_index*spec.frame_steps+spec.assumed_capture_steps, n)
            for i in traces:
                traces[i]['frames'].append((n, frame_index, reading[i].copy()))
            frame = None
        if n % spec.tick_steps == 0:
            start = time.perf_counter()
            proposed = np.clip(controller.tick(n), -software_range, software_range)
            tick_time += time.perf_counter()-start
            ticks += 1
            pending.append((n+spec.latency_steps, proposed))
        while pending and pending[0][0] <= n:
            command = pending.pop(0)[1]
        theta = state[:, :, 0]+offset+jitter[idx, n]
        deflection = np.einsum('bij,bj->bi', gain, theta)
        focal = f*(deflection+alpha[idx, n])
        # Same plane as the PSD in the focal mode; separate lever for the direct (as-built) receiver.
        pda_offset = focal if f_pda == f else f_pda*deflection+f*alpha[idx, n]
        lpf = focal.copy() if lpf is None else lpf+lpf_gain*(focal-lpf)
        if phase == capture:
            reading = lpf+episodes.psd_noise[idx, m]
            reading = np.round(reading/spec.psd_lsb)*spec.psd_lsb
            frame = (reading, m)
        if n >= spec.warmup_steps:
            metrics.add(focal, pda_offset)
        if residual is not None:
            residual[:, n] = focal
        for i in traces:
            traces[i]['focal'].append(focal[i])
            traces[i]['command'].append(command[i])
        state = np.einsum('aij,baj->bai', ad, state)+command[:, :, None]*bd[None]
    result = metrics.result()
    result['tick_time_per_episode_s'] = tick_time/max(ticks, 1)/batch
    result['tick_time_batch_s'] = tick_time/max(ticks, 1)
    for i in traces:
        traces[i] = {k: (np.asarray(v) if k != 'frames' else v) for k, v in traces[i].items()}
    result['traces'] = traces
    result['residual'] = residual
    return result
