"""Layered HV / modified-von-Karman channel shared by both propagators.

Finite random Fourier quadrature of the phase-screen PSD:
0.023 r0^(-5/3) exp(-(f/fm)^2)/(f^2+f0^2)^(11/6),
r0^(-5/3) = 0.423 k0^2 integral(Cn2 dz).
Dividing phase by k0*dz gives piecewise-constant delta_n. This is a
Markov layer approximation, not a measured 3D atmosphere. Increasing modes,
spatial resolution and refining layers requires a convergence study.
"""
from __future__ import annotations
import numpy as np
from config.settings import TurbulenceConfig


def hufnagel_valley(h, ground_cn2=1.7e-14, wind_speed=21.0):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    return (0.00594 * (wind_speed / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000)
            + 2.7e-16 * np.exp(-h / 1500) + ground_cn2 * np.exp(-h / 100))


class HVVonKarman:
    def __init__(self, cfg: TurbulenceConfig, distance: float):
        self.cfg, self.distance = cfg, distance
        self.edges = np.linspace(0, distance, int(np.ceil(distance / cfg.layer_thickness)) + 1)
        self.width = self.edges[1] - self.edges[0]
        rng = np.random.default_rng(cfg.seed + 100)
        f0, fm = 1 / cfg.outer_scale, 5.92 / (2 * np.pi * cfg.inner_scale)
        low, high = f0 / 100, fm * 3
        # Stratified log-frequency quadrature; includes large-scale tip/tilt.
        u = (np.arange(cfg.num_modes) + rng.random(cfg.num_modes)) / cfg.num_modes
        f = low * (high / low)**u
        pdf = 1 / (f * np.log(high / low))
        angle = rng.uniform(0, 2 * np.pi, cfg.num_modes)
        self.kx, self.ky = 2 * np.pi * f * np.cos(angle), 2 * np.pi * f * np.sin(angle)
        psd = 0.023 * 0.423 * np.exp(-(f / fm)**2) / (f*f + f0*f0)**(11/6)
        self.amplitude = np.sqrt(2 * psd * 2 * np.pi * f / pdf / cfg.num_modes)
        points, weights = np.polynomial.legendre.leggauss(16)
        z = (self.edges[:-1, None] + self.width * (points + 1) / 2)
        self.cn2 = (hufnagel_valley(distance - z, cfg.ground_cn2, cfg.hv_wind_speed)
                    @ weights / 2 * cfg.strength_scale)
        self._realization_key = None
        self._coefficients = None
        self._xy_cache = {}

    def _coeff(self, interval, layer):
        if self._realization_key != interval:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, interval, 991]))
            phases = rng.uniform(0, 2 * np.pi, (len(self.cn2), self.cfg.num_modes))
            self._coefficients = (np.exp(1j * phases) * self.amplitude[None, :]
                                  * np.sqrt(self.cn2[:, None] / self.width))
            self._realization_key = interval
        return self._coefficients[layer]

    def eval(self, x, y, z: float, interval: int):
        x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        if self.cfg.strength_scale == 0:
            return np.zeros_like(x, dtype=float)
        layer = min(int(max(z, 0) / self.width), len(self.cn2) - 1)
        coeff = self._coeff(interval, layer)
        if x.ndim == 2 and np.all(x == x[0:1, :]) and np.all(y == y[:, 0:1]):
            key = (x.shape, x[0].tobytes(), y[:, 0].tobytes())
            if key not in self._xy_cache:
                self._xy_cache[key] = (np.exp(1j * self.kx[:, None] * x[0]),
                                      np.exp(1j * y[:, 0:1] * self.ky[None, :]))
            ex, ey = self._xy_cache[key]
            return ((ey * coeff) @ ex).real
        phase = self.kx[:, None] * x.ravel() + self.ky[:, None] * y.ravel()
        return (coeff @ np.exp(1j * phase)).real.reshape(x.shape)
