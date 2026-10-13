"""Declared bench disturbances in the FSM plane (optical angle, rad).

Families of kind 'rail' (config/bench.json) are Tx translations on the motorized rail,
optionally with structural lines injected into the FSM command; see make_rail_episode.
Angle families (legacy/pat_bench_v2, focal mode) have three parts per episode,
all hidden from the controllers:
  sway       narrowband resonance 0.2-1 Hz (platform / coarse-gimbal residual)
  lines      1-2 structural vibration lines with slow frequency drift and AM,
             each along a random direction (so the two axes are correlated)
  broadband  tilt-like coloured noise, S(f) ~ (f^2+f0^2)^(-1/3) (1+(f/fc)^2)^(-8/3)
These are test inputs, not measured HAPS data. Parameters are drawn per
episode from the family ranges in config/bench.json.
"""
import numpy as np


def _uniform(rng, bounds):
    low, high = bounds
    return float(rng.uniform(low, high))


def sample_parameters(family, rng):
    p = dict(axes=[])
    for _ in range(2):
        p['axes'].append(dict(sway_hz=_uniform(rng, family['sway_hz']),
                              sway_damping=_uniform(rng, family['sway_damping']),
                              sway_rms=_uniform(rng, family['sway_rms_rad']),
                              broadband_rms=_uniform(rng, family['broadband_rms_rad']),
                              broadband_corner_hz=_uniform(rng, family['broadband_corner_hz'])))
    low, high = family['lines']
    p['lines'] = []
    for _ in range(int(rng.integers(low, high+1))):
        line = dict(hz=_uniform(rng, family['line_hz']), amp=_uniform(rng, family['line_amp_rad']),
                    drift=_uniform(rng, family['line_drift_frac']),
                    drift_period=_uniform(rng, family['line_drift_period_s']),
                    am=_uniform(rng, family['line_am_depth']), am_period=_uniform(rng, family['line_am_period_s']),
                    direction=float(rng.uniform(0, np.pi)), phases=rng.uniform(0, 2*np.pi, 3).tolist())
        if family.get('switch'):
            line['switch_time_frac'] = family['switch']['time_frac']
            line['switch_hz'] = _uniform(rng, family['switch']['to_hz'])
        p['lines'].append(line)
    return p


def _coloured(rng, n, dt, shape, rms):
    freqs = np.fft.rfftfreq(n, dt)
    amplitude = np.sqrt(shape(freqs))
    amplitude[0] = 0.0
    spectrum = amplitude*(rng.standard_normal(freqs.size)+1j*rng.standard_normal(freqs.size))
    series = np.fft.irfft(spectrum, n)
    return series*rms/np.std(series)


def synthesize(params, duration, dt, rng):
    n = int(round(duration/dt))
    t = np.arange(n)*dt
    out = np.zeros((n, 2))
    for axis, a in enumerate(params['axes']):
        fs, zeta = a['sway_hz'], a['sway_damping']
        out[:, axis] += _coloured(rng, n, dt, lambda f: 1/((fs**2-f**2)**2+(2*zeta*fs*f)**2+1e-12), a['sway_rms'])
        fc = a['broadband_corner_hz']
        out[:, axis] += _coloured(rng, n, dt, lambda f: (f**2+0.1**2)**(-1/3)*(1+(f/fc)**2)**(-8/3),
                                  a['broadband_rms'])
    _add_lines(out, params['lines'], t, dt, duration)
    return out


def _add_lines(out, lines, t, dt, duration):
    for line in lines:
        ph = line['phases']
        hz = line['hz']*(1+line['drift']*np.sin(2*np.pi*t/line['drift_period']+ph[1]))
        if 'switch_hz' in line:
            ratio = line['switch_hz']/line['hz']
            hz = np.where(t >= line['switch_time_frac']*duration, hz*ratio, hz)
        phase = 2*np.pi*np.cumsum(hz)*dt+ph[0]
        amp = line['amp']*(1+line['am']*np.sin(2*np.pi*t/line['am_period']+ph[2]))
        value = amp*np.sin(phase)
        out[:, 0] += value*np.cos(line['direction'])
        out[:, 1] += value*np.sin(line['direction'])


def _sample_lines(spec_lines, rng):
    low, high = spec_lines['lines']
    lines = []
    for _ in range(int(rng.integers(low, high+1))):
        lines.append(dict(hz=_uniform(rng, spec_lines['line_hz']), amp=_uniform(rng, spec_lines['line_amp_rad']),
                          drift=_uniform(rng, spec_lines['line_drift_frac']),
                          drift_period=_uniform(rng, spec_lines['line_drift_period_s']),
                          am=_uniform(rng, spec_lines['line_am_depth']), am_period=_uniform(rng, spec_lines['line_am_period_s']),
                          direction=float(rng.uniform(0, np.pi)), phases=rng.uniform(0, 2*np.pi, 3).tolist()))
    return lines


def _sweep(rng, n, dt, amplitude, speed, accel):
    """Back-and-forth constant-speed travel with acceleration-limited reversals."""
    x, v = float(rng.uniform(-amplitude, amplitude)), 0.0
    direction = 1.0 if rng.uniform() < 0.5 else -1.0
    out = np.empty(n)
    for i in range(n):
        if direction*x+v*v/(2*accel) >= amplitude:
            direction = -direction
        target = direction*speed
        v += np.clip(target-v, -accel*dt, accel*dt)
        x += v*dt
        out[i] = x
    return out


def _derivative_peaks(x, dt):
    v = np.diff(x)/dt
    a = np.diff(v)/dt
    return float(np.max(abs(v))), float(np.max(abs(a)))


def make_rail_episode(family, seed, duration, dt, spec):
    """Tx translation on the rail (x only), within the stage travel, speed and acceleration limits.

    Returned as the equivalent FSM-plane angle alpha = D / (reducer * PSD lever), so the
    detector-plane beam shift D / reducer equals psd_lever * alpha, as in the focal model.
    """
    rng = np.random.default_rng(seed)
    rail = spec.rail
    n = int(round(duration/dt))
    p = dict(kind='rail')
    drift = sway = sweep = np.zeros(n)
    # Slow components are band-limited (8th-order roll-off at 2 Hz): an OU drift alone would
    # demand unbounded stage speed and acceleration at the 0.5 ms simulation step.
    slow_band = lambda f: 1/(1+(f/2.0)**8)
    if family.get('drift_rms_m'):
        fc = p['drift_corner_hz'] = _uniform(rng, family['drift_corner_hz'])
        drift = _coloured(rng, n, dt, lambda f: slow_band(f)/(f**2+fc**2), _uniform(rng, family['drift_rms_m']))
    if family.get('sway_rms_m'):
        fs, zeta = p['sway_hz'], p['sway_damping'] = (_uniform(rng, family['sway_hz']),
                                                       _uniform(rng, family['sway_damping']))
        sway = _coloured(rng, n, dt, lambda f: slow_band(f)/((fs**2-f**2)**2+(2*zeta*fs*f)**2+1e-12),
                         _uniform(rng, family['sway_rms_m']))
    if family.get('sweep'):
        sw = family['sweep']
        p['sweep_amplitude_m'] = _uniform(rng, sw['amplitude_m'])
        p['sweep_speed_m_s'] = min(_uniform(rng, sw['speed_m_s']), rail.max_speed_m_s)
        sweep = _sweep(rng, n, dt, p['sweep_amplitude_m'], p['sweep_speed_m_s'],
                       sw.get('accel_fraction', 0.5)*rail.max_accel_m_s2)
    low, high = p['jitter_low_hz'], p['jitter_high_hz'] = (_uniform(rng, family['jitter_low_hz']),
                                                          _uniform(rng, family['jitter_high_hz']))
    jitter = _coloured(rng, n, dt, lambda f: (f/low)**2/(1+(f/low)**2)/(1+(f/high)**8),
                       _uniform(rng, family['jitter_rms_m']))
    # Jitter dominates acceleration: keep it within half the stage limit.
    _, accel = _derivative_peaks(jitter, dt)
    jitter_scale = min(1.0, 0.5*rail.max_accel_m_s2/accel)
    jitter = jitter*jitter_scale
    slow = drift+sway+sweep
    room = rail.travel_m-np.max(abs(jitter))
    slow_scale = min(1.0, room/max(np.max(abs(slow)), 1e-15))
    position = slow*slow_scale+jitter
    speed, accel = _derivative_peaks(position, dt)
    if speed > rail.max_speed_m_s or accel > rail.max_accel_m_s2 or np.max(abs(position)) > rail.travel_m+1e-12:
        raise ValueError(f'Rail profile for seed {seed} exceeds the stage limits: '
                         f'{np.max(abs(position))*1e3:.2f} mm, {speed*1e3:.1f} mm/s, {accel:.3f} m/s^2')
    p.update(jitter_scale=jitter_scale, slow_scale=slow_scale,
             rms_m=float(np.std(position)), jitter_rms_m=float(np.std(jitter)),
             peak_m=float(np.max(abs(position))), peak_speed_m_s=speed, peak_accel_m_s2=accel)
    alpha = np.zeros((n, 2))
    alpha[:, 0] = position/(spec.reducer*spec.psd_lever)
    if family.get('offset_m'):
        # Constant rail offset at start (acquisition demo): the beam begins off target.
        alpha[:, 0] += family['offset_m']/(spec.reducer*spec.psd_lever)
        p['offset_m'] = family['offset_m']
    if family.get('injected_lines'):
        # Structural vibration injected in software into the FSM command and hidden from all
        # controllers; already in FSM-plane angle units (as in the focal-mode bench, legacy/pat_bench_v2).
        inj = family['injected_lines']
        lines = _sample_lines(inj, np.random.default_rng([seed, 7]))
        scale = inj.get('amplitude_scale', 1.0)
        for line in lines:
            line['amp'] *= scale
        _add_lines(alpha, lines, np.arange(n)*dt, dt, duration)
        p['injected_lines'] = lines
    return alpha, p


def make_episode(family, seed, duration, dt, spec=None):
    if family.get('kind') == 'rail':
        return make_rail_episode(family, seed, duration, dt, spec)
    rng = np.random.default_rng(seed)
    params = sample_parameters(family, rng)
    return synthesize(params, duration, dt, rng), params
