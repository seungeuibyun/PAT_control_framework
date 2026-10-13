"""Communication metrics from pointing residuals (HAPS-equivalent IM/DD receiver).

The bench residual at the PSD plane maps to the focal-plane offset of the HAPS receiver
(lever = focal length), whose capture function eta(s) gives the normalized received power.
With signal-independent (thermal/NEP-limited) detection, the electrical SNR of IM/DD scales as
power squared:  SNR(t) = SNR_al * eta(t)^2, where SNR_al is the aligned SNR.
  BER_M(g) = 4/log2(M) (1 - 1/sqrt(M)) Q(sqrt(3 g / (M-1)))      (Gray square M-QAM)
  outage   = Pr{SNR(t) < g_th(64)},  g_th: SNR where BER_64 reaches the FEC threshold
  rate     = R_s * payload fraction * E[max{log2 M : SNR(t) >= g_th(M)}]
A symbol-level simulation with per-frame pilot gain estimates is provided for the figures.
"""
import numpy as np
from scipy.special import erfc

from pat.optics import GaussianFocalResponse
from pat.qam import constellation, modulate, demodulate


def q_function(x):
    return 0.5*erfc(x/np.sqrt(2))


def ber_qam(order, snr):
    return 4/np.log2(order)*(1-1/np.sqrt(order))*q_function(np.sqrt(3*snr/(order-1)))


def threshold_snr(order, target_ber):
    grid = np.linspace(-5, 50, 55001)
    ber = ber_qam(order, 10**(grid/10))
    return float(10**(grid[np.argmax(ber <= target_ber)]/10))


class HAPSReceiver:
    """HAPS receiver (PDF Table I optics) with a given communication-detector radius."""
    def __init__(self, cfg, detector_radius, lever):
        h = cfg['haps_receiver']
        self.response = GaussianFocalResponse(h['wavelength_m'], h['focal_length_m'], h['pupil_radius_m'],
                                              h['aperture_radius_m'], detector_radius)
        self.scale = h['focal_length_m']/lever
        self.radius = detector_radius

    def eta(self, residual):
        return self.response.normalized_power(np.asarray(residual)*self.scale)


class LinkModel:
    def __init__(self, cfg):
        c = cfg['comm']
        self.cfg = c
        self.fec = c['fec_ber']
        self.orders = list(c['qam_orders'])
        self.thresholds = {m: threshold_snr(m, self.fec) for m in self.orders}
        self.snr_grid_db = np.arange(c['aligned_snr_db'][0], c['aligned_snr_db'][1]+1e-9, c['aligned_snr_db'][2])
        self.operating_snr = self.thresholds[64]*10**(c['operating_margin_db']/10)
        self.rate_scale = c['symbol_rate_baud']*c['payload_fraction']

    def metrics(self, eta, snr_aligned):
        """eta: (..., T) normalized power; returns BER_64, outage_64, rate [bit/s] averaged over time."""
        snr = snr_aligned*np.asarray(eta)**2
        bits = np.zeros_like(snr)
        for m in self.orders:
            bits = np.where(snr >= self.thresholds[m], np.log2(m), bits)
        return dict(ber64=float(ber_qam(64, snr).mean()), outage64=float((snr < self.thresholds[64]).mean()),
                    rate_bps=float(bits.mean()*self.rate_scale))

    def curves(self, eta):
        out = {k: [] for k in ('ber64', 'outage64', 'rate_bps')}
        for snr_db in self.snr_grid_db:
            m = self.metrics(eta, 10**(snr_db/10))
            for k in out:
                out[k].append(m[k])
        return {k: np.asarray(v) for k, v in out.items()}


def simulate_symbols(eta_of_t, times, snr_aligned, frame_s, pilots, rng):
    """64-QAM symbols at the given times; gain estimated from the pilots at each frame start.

    Returns transmitted, raw received (aligned constellation has unit energy) and equalized symbols.
    """
    times = np.asarray(times)
    bits = rng.integers(0, 2, (len(times), 6), dtype=np.uint8)
    tx = modulate(bits)
    sigma = np.sqrt(1/(2*snr_aligned))
    gain = eta_of_t(times)
    raw = gain*tx+sigma*(rng.standard_normal(len(times))+1j*rng.standard_normal(len(times)))
    frame_start = np.floor(times/frame_s)*frame_s
    pilot_gain = eta_of_t(frame_start)
    estimate = pilot_gain+sigma/np.sqrt(pilots)*(rng.standard_normal(len(times))+1j*rng.standard_normal(len(times)))
    equalized = raw/np.where(abs(estimate) > 1e-6, estimate, 1e-6)
    errors = np.count_nonzero(demodulate(equalized) != bits, axis=1)
    return dict(tx=tx, raw=raw, equalized=equalized, bit_errors=errors, gain=gain)


__all__ = ['HAPSReceiver', 'LinkModel', 'simulate_symbols', 'ber_qam', 'threshold_snr', 'constellation']
