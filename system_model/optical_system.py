from __future__ import annotations

import numpy as np
from config.settings import ExperimentConfig
from system_model.turbulence import SmoothDeltaN


class OpticalPATSystem:
    """Physical optical PAT system and independent SSFM truth model."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        oc = cfg.optical
        tc = cfg.tracking
        self.turbulence = SmoothDeltaN(cfg.turbulence)
        self.x = np.linspace(-oc.half_width, oc.half_width, oc.grid_size, endpoint=False)
        self.y = np.linspace(-oc.half_width, oc.half_width, oc.grid_size, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")
        self.dz_truth = oc.propagation_distance / tc.ssfm_steps
        fx = np.fft.fftfreq(oc.grid_size, d=self.dx)
        fy = np.fft.fftfreq(oc.grid_size, d=self.dy)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        kx = 2.0 * np.pi * FX
        ky = 2.0 * np.pi * FY
        self.H_diff = np.exp(-1j * (kx**2 + ky**2) * self.dz_truth / (2.0 * oc.k0))

    def gaussian_beam(self, X, Y):
        oc = self.cfg.optical
        r2 = np.asarray(X) ** 2 + np.asarray(Y) ** 2
        return np.sqrt(2.0 * oc.transmit_power / (np.pi * oc.beam_waist**2)) * np.exp(-r2 / oc.beam_waist**2)

    def tx_field(self, theta, X, Y):
        oc = self.cfg.optical
        return self.gaussian_beam(X, Y) * np.exp(1j * oc.k0 * (theta[0] * X + theta[1] * Y))

    def time_at_interval(self, interval: int) -> float:
        return float(interval * self.cfg.tracking.control_interval_sec)

    def target_position(self, interval: int):
        tc = self.cfg.tracking
        t = self.time_at_interval(interval)
        tau = 2.0 * np.pi * t / max(tc.target_period_sec, 1e-30)
        return np.array([
            tc.target_x_amplitude * np.sin(tau),
            tc.target_y_amplitude * np.sin(2.0 * tau + tc.target_y_phase),
        ], dtype=float)

    def aperture_mask(self, center):
        oc = self.cfg.optical
        cx, cy = center
        return (((self.X - cx) ** 2 + (self.Y - cy) ** 2) <= oc.aperture_radius**2).astype(float)

    def centroid_roi_mask(self, center):
        oc = self.cfg.optical
        cx, cy = center
        return (((self.X - cx) ** 2 + (self.Y - cy) ** 2) <= oc.centroid_roi_radius**2).astype(float)

    def smf_mode(self, center):
        oc = self.cfg.optical
        cx, cy = center
        psi = np.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2) / oc.smf_mode_waist**2)
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx * self.dy)
        return psi / max(norm, 1e-30)

    def power(self, U, target):
        A = self.aperture_mask(target)
        return float(np.sum(A * np.abs(U) ** 2) * self.dx * self.dy)

    def coupling(self, U, target):
        psi = self.smf_mode(target)
        dA = self.dx * self.dy
        inner = np.sum(U * np.conj(psi)) * dA
        field_power = np.sum(np.abs(U) ** 2) * dA
        mode_power = np.sum(np.abs(psi) ** 2) * dA
        return float(np.abs(inner) ** 2 / max(field_power * mode_power, 1e-30))

    def centroid(self, U, target):
        W = self.centroid_roi_mask(target)
        I = np.abs(U) ** 2
        dA = self.dx * self.dy
        denom = max(np.sum(W * I) * dA, 1e-30)
        px = np.sum(W * self.X * I) * dA / denom
        py = np.sum(W * self.Y * I) * dA / denom
        return np.array([px, py], dtype=float)

    def metric(self, U, target, objective: str | None = None):
        objective = objective or self.cfg.tracking.objective
        if objective == "power":
            return self.power(U, target)
        if objective == "coupling":
            return self.coupling(U, target)
        if objective == "centroid":
            p = self.centroid(U, target)
            e = p - np.asarray(target)
            return float(-np.dot(e, e))
        raise ValueError("objective must be power, coupling, or centroid")

    def display_metric(self, U, target, objective: str | None = None):
        objective = objective or self.cfg.tracking.objective
        if objective == "power":
            return self.power(U, target)
        if objective == "coupling":
            return self.coupling(U, target)
        if objective == "centroid":
            p = self.centroid(U, target)
            return float(np.linalg.norm(p - target) * 1e3)
        raise ValueError("objective must be power, coupling, or centroid")

    def ssfm(self, theta, interval: int):
        oc = self.cfg.optical
        tc = self.cfg.tracking
        U = self.tx_field(theta, self.X, self.Y).astype(np.complex128)
        for iz in range(tc.ssfm_steps):
            z = (iz + 0.5) * self.dz_truth
            dn = self.turbulence.eval(self.X, self.Y, z, interval)
            half = np.exp((1j * oc.k0 * dn - oc.attenuation / 2.0) * self.dz_truth / 2.0)
            U *= half
            U = np.fft.ifft2(np.fft.fft2(U) * self.H_diff)
            U *= half
        return U
