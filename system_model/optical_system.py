"""Eqs. (1)-(20): atmosphere -> aperture/reducer -> FSM -> lens -> branches.

The back-focal-plane Collins integral evaluates P_l2 L_f P_l1 followed
by P_lD/P_lP as one scaled Fourier transform. This avoids applying a
micrometre detector mask on the metre-scale atmospheric grid.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates
from config.settings import ExperimentConfig
from system_model.turbulence import HVVonKarman


class OpticalPATSystem:
    def __init__(self, cfg: ExperimentConfig):
        cfg.validate()
        self.cfg = cfg
        o = cfg.optical
        self.turbulence = HVVonKarman(cfg.turbulence, o.propagation_distance)
        self.x = np.linspace(-o.half_width, o.half_width, o.grid_size, endpoint=False)
        self.y = self.x.copy()
        self.dx = self.dy = 2 * o.half_width / o.grid_size
        self.X, self.Y = np.meshgrid(self.x, self.y)
        freq = 2 * np.pi * np.fft.fftfreq(o.grid_size, self.dx)
        self.k_squared = freq[:, None]**2 + freq[None, :]**2
        # Include all channel discontinuities even when the SSFM grid is coarser.
        self.z_edges = np.unique(np.concatenate((self.turbulence.edges,
            np.linspace(0, o.propagation_distance, cfg.tracking.ssfm_steps + 1))))
        self.rx = np.linspace(-o.receiver_half_width, o.receiver_half_width,
                              o.receiver_grid_size, endpoint=False)
        self.RX, self.RY = np.meshgrid(self.rx, self.rx)
        self.rdx = 2 * o.receiver_half_width / o.receiver_grid_size
        self.CF = np.asarray(o.fsm_calibration, dtype=float)
        self.psd_jacobian = o.focal_length * self.CF
        self.reduced_mask = (self.RX**2 + self.RY**2 <=
                             (abs(o.reducer_magnification) * o.entrance_diameter / 2)**2)
        self._interp_coordinates = np.asarray([
            (self.RY / o.reducer_magnification + o.half_width) / self.dx,
            (self.RX / o.reducer_magnification + o.half_width) / self.dx])
        # Fine quadrature only around the fixed communication aperture.
        q = np.arange(-int(np.ceil(o.aperture_radius / o.detector_sampling)),
                      int(np.ceil(o.aperture_radius / o.detector_sampling)) + 1) * o.detector_sampling
        self.detector_x = q + o.detector_center[0]
        self.detector_y = q + o.detector_center[1]
        self.DX, self.DY = np.meshgrid(self.detector_x, self.detector_y)
        self.detector_mask = (self.DX - o.detector_center[0])**2 + (self.DY - o.detector_center[1])**2 <= o.aperture_radius**2
        self.detector_operators = self._fourier_matrices(self.detector_x, self.detector_y)
        npad = o.psd_padding * o.receiver_grid_size
        self.psd_x = np.fft.fftshift(np.fft.fftfreq(npad, self.rdx)) * o.wavelength * o.focal_length
        self.PX, self.PY = np.meshgrid(self.psd_x, self.psd_x)
        self.psd_dx = self.psd_x[1] - self.psd_x[0]
        self.psd_mask = ((abs(self.PX - o.psd_center[0]) <= o.psd_width / 2)
                         & (abs(self.PY - o.psd_center[1]) <= o.psd_width / 2))
        self._truth_key = None
        self._truth_field = None
        self.atmosphere_propagations = 0
        # Ideal, aligned back-focal-plane calibration. Offset is specified in
        # detector coordinates; arbitrary measured calibration can replace this.
        self.reference_centroid = np.asarray(o.detector_center, dtype=float)

    def time_at_interval(self, interval):
        return interval * self.cfg.tracking.control_interval_sec

    def target_position(self, interval):
        return self.reference_centroid.copy()

    def tx_angle(self, interval):
        o, t = self.cfg.optical, self.cfg.tracking
        if t.disturbance == "zero":
            return np.zeros(2)
        if t.disturbance == "periodic":
            phase = 2 * np.pi * self.time_at_interval(interval) / t.target_period_sec
            return np.sqrt(2) * o.tx_angular_std * np.array([np.sin(phase), np.sin(2 * phase)])
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.turbulence.seed, interval, 17]))
        return rng.normal(0, o.tx_angular_std, 2)

    def gaussian_beam(self, X, Y):
        o = self.cfg.optical
        r2 = (np.asarray(X) - o.tx_displacement[0])**2 + (np.asarray(Y) - o.tx_displacement[1])**2
        return np.sqrt(2 * o.transmit_power / (np.pi * o.beam_waist**2)) * np.exp(-r2 / o.beam_waist**2)

    def tx_field(self, interval, X=None, Y=None):
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        alpha = self.tx_angle(interval)
        return self.gaussian_beam(X, Y) * np.exp(1j * self.cfg.optical.k0 * (alpha[0] * X + alpha[1] * Y))

    def atmospheric_field(self, interval, *, cache=True):
        if cache and self._truth_key == interval:
            return self._truth_field
        o = self.cfg.optical
        U = self.tx_field(interval).astype(complex)
        for left, right in zip(self.z_edges[:-1], self.z_edges[1:]):
            dz = right - left
            dn = self.turbulence.eval(self.X, self.Y, (left + right) / 2, interval)
            half = np.exp((1j * o.k0 * dn - o.attenuation / 2) * dz / 2)
            H = np.exp(-1j * self.k_squared * dz / (2 * o.k0))
            U = half * np.fft.ifft2(np.fft.fft2(half * U) * H)
        self.atmosphere_propagations += 1
        if cache:
            self._truth_key, self._truth_field = interval, U
        return U

    def reduce_field(self, U_rx):
        o = self.cfg.optical
        # Sample U_rx(r/mG), then apply exactly transformed entrance aperture.
        sampled = (map_coordinates(U_rx.real, self._interp_coordinates, order=3, mode="constant")
                   + 1j * map_coordinates(U_rx.imag, self._interp_coordinates, order=3, mode="constant"))
        return (np.sqrt(o.reducer_transmission) / abs(o.reducer_magnification)
                * sampled * self.reduced_mask
                * np.exp(1j * o.reducer_phase_curvature * (self.RX**2 + self.RY**2)))

    def fsm_field(self, reduced, theta):
        angle = self.CF @ np.asarray(theta)
        return reduced * np.exp(1j * self.cfg.optical.k0 * (angle[0] * self.RX + angle[1] * self.RY))

    def _fourier_matrices(self, x, y):
        o = self.cfg.optical
        ex = np.exp(-2j * np.pi * np.outer(x, self.rx) / (o.wavelength * o.focal_length))
        ey = np.exp(-2j * np.pi * np.outer(y, self.rx) / (o.wavelength * o.focal_length))
        return ey, ex

    def _output_phase(self, X, Y):
        o = self.cfg.optical
        # ABCD: A=0, B=f, D=1-l1/f. Global propagation phase omitted.
        return np.exp(1j * o.k0 * (1 - o.fsm_to_lens / o.focal_length)
                      * (X*X + Y*Y) / (2 * o.focal_length))

    def detector_field(self, reduced, theta, *, view=False):
        o = self.cfg.optical
        U = self.fsm_field(reduced, theta)
        if view:
            n = int(np.ceil(o.detector_view_half_width / o.detector_sampling))
            v = np.arange(-n, n + 1) * o.detector_sampling
            x, y = v + o.detector_center[0], v + o.detector_center[1]
            X, Y = np.meshgrid(x, y)
            ey, ex = self._fourier_matrices(x, y)
        else:
            X, Y = self.DX, self.DY
            ey, ex = self.detector_operators
        prefactor = np.sqrt(o.splitter_transmission) * self.rdx**2 / (1j * o.wavelength * o.focal_length)
        return prefactor * self._output_phase(X, Y) * (ey @ U @ ex.T)

    def power(self, U, target=None):
        return float(np.sum(self.detector_mask * abs(U)**2) * self.cfg.optical.detector_sampling**2)

    def metric(self, U, target=None, objective=None):
        if objective not in (None, "power"):
            raise ValueError("Only the PDF power objective is supported")
        return self.power(U)

    display_metric = metric

    def power_and_gradient(self, reduced, theta):
        o = self.cfg.optical
        F = self.fsm_field(reduced, theta)
        ey, ex = self.detector_operators
        scale = np.sqrt(o.splitter_transmission) * self.rdx**2 / (o.wavelength * o.focal_length)
        U = scale * (ey @ F @ ex.T)  # common output phase cancels in power
        gradients = []
        for a in range(2):
            dF = 1j * o.k0 * (self.CF[0, a] * self.RX + self.CF[1, a] * self.RY) * F
            dU = scale * (ey @ dF @ ex.T)
            gradients.append(2 * np.real(np.sum(self.detector_mask * U.conj() * dU)) * o.detector_sampling**2)
        return self.power(U), np.asarray(gradients)

    def sensing_vector(self, reduced, theta):
        o = self.cfg.optical
        F = self.fsm_field(reduced, theta) * np.exp(1j * o.splitter_phase)
        n = o.receiver_grid_size
        nfull = o.psd_padding * n
        before = (nfull - n) // 2
        F = np.pad(F, ((before, nfull - n - before),) * 2)
        P = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(F)))
        P *= np.sqrt(o.splitter_reflection) * self.rdx**2 / (o.wavelength * o.focal_length)
        I = abs(P)**2 * self.psd_mask
        power = float(I.sum() * self.psd_dx**2)
        if power <= o.psd_min_power:
            return np.array([np.nan, np.nan, power])
        return np.array([(I * self.PX).sum() / I.sum(), (I * self.PY).sum() / I.sum(), power])

    def measure_psd(self, reduced_truth, theta_prev, interval):
        o = self.cfg.optical
        sensing = self.sensing_vector(reduced_truth, theta_prev)
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.turbulence.seed, interval, 37]))
        noise = rng.normal(size=3) * [o.psd_position_noise_std, o.psd_position_noise_std, o.psd_power_noise_std]
        measured = sensing + noise
        valid = bool(np.all(np.isfinite(measured)) and measured[2] > o.psd_min_power
                     and np.all(abs(measured[:2] - o.psd_center) <= o.psd_width / 2))
        return measured, valid

    def ssfm(self, theta, interval):
        return self.detector_field(self.reduce_field(self.atmospheric_field(interval)), theta)

    def calibration(self):
        return {"reference_centroid_m": self.reference_centroid,
                "jacobian_m_per_rad": self.psd_jacobian,
                "source": "ideal aligned back-focal-plane calibration; replace with measurements for hardware"}
