"""Bench receiver optics: Gaussian beam -> MR-15-30 -> f = 200 mm lens -> PSD/PDA.

The Tx path is vacuum over a few metres (Rayleigh range ~39 m), so the field
at the FSM is the expanded Gaussian with its small curvature. The focal field
is the scaled Fourier transform of the clipped pupil field. In the paraxial
regime an FSM tilt translates this spot without changing its shape, so the
PDA response is the cross-correlation of the spot with the 75 um disk.
"""
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class BenchOptics:
    def __init__(self, spec):
        o = spec.optics
        n, dx = o.pupil_grid, o.pupil_sampling_m
        x = (np.arange(n)-n//2)*dx
        X, Y = np.meshgrid(x, x)
        w, curvature = spec.beam_radius_at_fsm, spec.beam_curvature_at_fsm
        field = np.exp(-(X**2+Y**2)/w**2).astype(complex)
        if math.isfinite(curvature):
            field *= np.exp(-1j*spec.k0*(X**2+Y**2)/(2*curvature))
        # Mirror aperture projected onto the beam (foreshortened along x) and lens aperture.
        a_mirror = o.fsm_mirror_diameter_m/2
        b_mirror = a_mirror*math.cos(math.radians(o.fsm_aoi_deg))
        pupil = ((X/b_mirror)**2+(Y/a_mirror)**2 <= 1) & (X**2+Y**2 <= (o.lens_clear_aperture_m/2)**2)
        unclipped = np.sum(abs(field)**2)
        field = field*pupil
        self.clip_transmission = float(np.sum(abs(field)**2)/unclipped)
        # Back focal plane: U_f(u) ~ FT{U_p}(u/(lambda f)), sampling lambda f/(n dx).
        focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        self.focal_dx = o.wavelength_m*spec.focal_length/(n*dx)
        intensity = abs(focal)**2
        intensity *= self.clip_transmission/intensity.sum()
        # Crop around the spot; beyond the half-width the spot carries no power.
        half = int(math.ceil(o.response_half_width_m/self.focal_dx))
        c = n//2
        crop = intensity[c-half:c+half+1, c-half:c+half+1]
        self.lost_outside_crop = float(self.clip_transmission-crop.sum())
        axis = (np.arange(crop.shape[0])-half)*self.focal_dx
        self.axis = axis
        self.spot = crop
        self.spot_radius_1e2 = o.wavelength_m*spec.focal_length/(math.pi*w)
        self.beam_radius_at_fsm = w
        # P(s) = sum_r disk(r) I(r - s): correlation of the spot with the detector disk.
        disk = (axis[None, :]**2+axis[:, None]**2 <= o.pda_radius_m**2).astype(float)
        size = 2*crop.shape[0]
        spectrum = np.fft.rfft2(crop[::-1, ::-1], (size, size))*np.fft.rfft2(disk, (size, size))
        corr = np.fft.irfft2(spectrum, (size, size))[half:half+crop.shape[0], half:half+crop.shape[0]]
        corr = np.clip(corr, 0, None)
        self.pda_fraction_table = corr
        self.aligned_fraction = float(corr[half, half])
        self._interp = RegularGridInterpolator((axis, axis), corr, bounds_error=False, fill_value=0.0)
        self.splitter_reflection = o.splitter_reflection
        self.pda_transmission = o.splitter_transmission*o.pda_branch_nd_transmission
        self.power_at_fsm = o.power_at_fsm_w

    def pda_fraction(self, offset):
        """Fraction of FSM-plane power on the PDA disk for focal offsets (..., 2) [m, (x, y)]."""
        offset = np.asarray(offset, dtype=float)
        # Table is indexed (y, x).
        points = np.stack([offset[..., 1], offset[..., 0]], axis=-1).reshape(-1, 2)
        return self._interp(points).reshape(offset.shape[:-1])

    def normalized_power(self, offset):
        return self.pda_fraction(offset)/self.aligned_fraction

    def pda_power_w(self, offset):
        return self.power_at_fsm*self.pda_transmission*self.pda_fraction(offset)

    def summary(self):
        return dict(beam_radius_at_fsm_m=self.beam_radius_at_fsm, focal_sampling_m=self.focal_dx,
                    clip_transmission=self.clip_transmission, aligned_pda_fraction=self.aligned_fraction,
                    lost_outside_response_window=self.lost_outside_crop,
                    gaussian_focal_radius_1e2_m=self.spot_radius_1e2)


class DirectBeamResponse:
    """Unfocused Gaussian beam (1/e^2 radius w) on a circular detector of radius a.

    P(s) = int_0^a (4r/w^2) exp(-2(r-s)^2/w^2) i0e(4rs/w^2) dr, the exact offset-disk integral.
    """
    def __init__(self, beam_radius, detector_radius, samples=4001):
        from scipy.special import i0e
        w, a = beam_radius, detector_radius
        self.beam_radius, self.detector_radius = w, a
        self.offsets = np.linspace(0, a+4*w, samples)
        r = np.linspace(0, a, 801)
        s = self.offsets[:, None]
        integrand = 4*r/w**2*np.exp(-2*(r-s)**2/w**2)*i0e(4*r*s/w**2)
        self.table = np.trapezoid(integrand, r, axis=1)
        self.aligned_fraction = float(self.table[0])

    def pda_fraction(self, offset):
        offset = np.asarray(offset, dtype=float)
        return np.interp(np.hypot(offset[..., 0], offset[..., 1]), self.offsets, self.table, right=0.0)

    def normalized_power(self, offset):
        return self.pda_fraction(offset)/self.aligned_fraction

    def summary(self):
        return dict(kind='direct_gaussian_beam', beam_radius_m=self.beam_radius,
                    detector_radius_m=self.detector_radius, aligned_fraction=self.aligned_fraction,
                    half_power_offset_m=float(np.interp(0.5, self.table[::-1]/self.aligned_fraction,
                                                        self.offsets[::-1])))


class GaussianFocalResponse:
    """Back-focal-plane detector response for a clipped Gaussian pupil (e.g. the PDF receiver).

    Pupil: Gaussian radius w clipped by a circular aperture of radius a, focused by focal length f.
    """
    def __init__(self, wavelength, focal_length, pupil_radius, aperture_radius, detector_radius,
                 grid=2048, sampling=40e-6, half_width=500e-6):
        x = (np.arange(grid)-grid//2)*sampling
        X, Y = np.meshgrid(x, x)
        field = np.exp(-(X**2+Y**2)/pupil_radius**2)*(X**2+Y**2 <= aperture_radius**2)
        intensity = abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field))))**2
        intensity /= intensity.sum()
        self.focal_dx = wavelength*focal_length/(grid*sampling)
        half = int(math.ceil(half_width/self.focal_dx))
        c = grid//2
        crop = intensity[c-half:c+half+1, c-half:c+half+1]
        axis = (np.arange(crop.shape[0])-half)*self.focal_dx
        disk = (axis[None, :]**2+axis[:, None]**2 <= detector_radius**2).astype(float)
        size = 2*crop.shape[0]
        spectrum = np.fft.rfft2(crop[::-1, ::-1], (size, size))*np.fft.rfft2(disk, (size, size))
        corr = np.clip(np.fft.irfft2(spectrum, (size, size))[half:half+crop.shape[0], half:half+crop.shape[0]], 0, None)
        self.aligned_fraction = float(corr[half, half])
        self._interp = RegularGridInterpolator((axis, axis), corr, bounds_error=False, fill_value=0.0)
        self.spot, self.axis = crop, axis          # focal-plane intensity (sums to 1), for figures
        self.params = dict(kind='gaussian_focal', wavelength_m=wavelength, focal_length_m=focal_length,
                           pupil_radius_m=pupil_radius, aperture_radius_m=aperture_radius,
                           detector_radius_m=detector_radius, focal_sampling_m=self.focal_dx)

    def normalized_power(self, offset):
        offset = np.asarray(offset, dtype=float)
        points = np.stack([offset[..., 1], offset[..., 0]], axis=-1).reshape(-1, 2)
        return self._interp(points).reshape(offset.shape[:-1])/self.aligned_fraction

    def summary(self):
        return dict(self.params, aligned_fraction=self.aligned_fraction)
