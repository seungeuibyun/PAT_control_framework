from __future__ import annotations

import numpy as np
from config.settings import TurbulenceConfig


class SmoothDeltaN:
    """Deterministic controlled volumetric refractive-index fluctuation."""

    def __init__(self, cfg: TurbulenceConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed + 100)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, cfg.num_modes)
        self.weight = rng.normal(size=cfg.num_modes)
        period_xy = rng.uniform(8e-3, 30e-3, size=(cfg.num_modes, 2))
        sign_xy = rng.choice([-1.0, 1.0], size=(cfg.num_modes, 2))
        self.kx = sign_xy[:, 0] * 2.0 * np.pi / period_xy[:, 0]
        self.ky = sign_xy[:, 1] * 2.0 * np.pi / period_xy[:, 1]
        period_z = rng.uniform(8.0, 40.0, cfg.num_modes)
        sign_z = rng.choice([-1.0, 1.0], size=cfg.num_modes)
        self.kz = sign_z * 2.0 * np.pi / period_z
        raw_rms = np.sqrt(0.5 * np.sum(self.weight**2))
        self.weight = self.weight / max(raw_rms, 1e-12) * cfg.delta_n_rms

    def eval(self, x, y, z: float, interval: int):
        x = np.asarray(x)
        y = np.asarray(y)
        out = np.zeros(np.broadcast(x, y).shape, dtype=float)
        shift = interval * self.cfg.shift_per_interval
        for a, kx, ky, kz, ph in zip(self.weight, self.kx, self.ky, self.kz, self.phase):
            out += a * np.cos(kx * x + ky * y + kz * z + ph + shift)
        return out
