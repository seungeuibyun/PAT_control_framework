"""As-built bench (config/bench.json): Tx rail translation, Rx reducer, direct position sensing."""
from pathlib import Path
import unittest

import numpy as np

from pat.controllers import HoldController, OracleController
from pat.disturbance import _derivative_peaks
from pat.optics import DirectBeamResponse, GaussianFocalResponse
from pat.plant import PowerModel, build_episodes, open_loop_frames, simulate
from pat.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT/'config/bench.json'
FOCAL = ROOT/'config/focal_mode.json'


def ideal(spec, **extra):
    return spec.variant(**{'fsm.jitter_rms_rad': 0.0, 'fsm.static_offset_rms_rad': 0.0,
                           'psd.position_noise_rms_m': 0.0, 'calibration.fsm_gain_error_rms': 0.0,
                           'calibration.fsm_coupling_error_rms': 0.0, 'calibration.target_error_rms_m': 0.0,
                           **extra})


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(BENCH)

    def test_geometry_and_regression_of_focal_mode(self):
        s = self.spec
        self.assertEqual((s.mode, s.psd_lever, s.pda_lever, s.reducer), ('direct', 0.2, 0.2, 10.0))
        self.assertLess(s.pda_peak_voltage, s.optics.pda_max_output_v)
        v2 = load_spec(FOCAL)
        self.assertEqual((v2.mode, v2.psd_lever, v2.pda_lever), ('focal', 0.2, 0.2))

    def test_rail_profiles_respect_stage_limits(self):
        r = self.spec.rail
        for family in ('station', 'sweep', 'fast_jitter'):
            episodes = build_episodes(self.spec, family, range(500, 512), 10.0)
            position = episodes.alpha[:, :, 0]*self.spec.reducer*self.spec.psd_lever
            self.assertLessEqual(np.max(abs(position)), r.travel_m+1e-12)
            for row in position:
                speed, accel = _derivative_peaks(row, self.spec.dt)
                self.assertLessEqual(speed, r.max_speed_m_s)
                self.assertLessEqual(accel, r.max_accel_m_s2)
            np.testing.assert_array_equal(episodes.alpha[:, :, 1], 0)

    def test_translation_is_seen_as_position_and_corrected_by_tilt(self):
        spec = ideal(self.spec)
        rail = 1e-3
        episodes = build_episodes(spec, 'station', [7], 3.0)
        episodes.alpha[:] = [rail/(spec.reducer*spec.psd_lever), 0.0]
        frames, _ = open_loop_frames(spec, episodes)
        # Reducer: 1 mm at the Tx -> 100 um on the PSD, quantized to the 12-bit LSB.
        self.assertAlmostEqual(frames[0, -1, 0], rail/spec.reducer, delta=spec.psd_lsb)
        result = simulate(spec, episodes, OracleController(episodes), [], trace_index=(0,))
        command = result['traces'][0]['command'][-1]
        # theta_x = D / (2 x reducer x lever) with optical gain 2 on the in-plane axis.
        self.assertAlmostEqual(-command[0], rail/(2*spec.reducer*spec.psd_lever), delta=1e-9)
        self.assertLess(np.max(abs(result['traces'][0]['focal'][-100:])), 1e-9)

    def test_pda_lever_mismatch_shifts_the_pda_spot(self):
        spec = ideal(self.spec, **{'receiver.fsm_to_pda_m': 0.22})
        rail = 3e-3
        episodes = build_episodes(spec, 'station', [8], 4.0)
        episodes.alpha[:] = [rail/(spec.reducer*spec.psd_lever), 0.0]
        response = DirectBeamResponse(spec.receiver.beam_radius_at_detectors_m, spec.optics.pda_radius_m)
        result = simulate(spec, episodes, OracleController(episodes), [PowerModel('pda', response)])
        expected_offset = rail/spec.reducer*(1-0.22/0.2)
        self.assertAlmostEqual(result['mean_power'][0, 0], float(response.normalized_power([expected_offset, 0])),
                               delta=2e-3)

    def test_direct_beam_response_matches_brute_force(self):
        w, a = 0.55e-3, 75e-6
        response = DirectBeamResponse(w, a)
        r, th = np.linspace(0, a, 500), np.linspace(0, 2*np.pi, 500)
        R, T = np.meshgrid(r, th)
        for offset in (0, 150e-6, 400e-6):
            intensity = 2/(np.pi*w**2)*np.exp(-2*((R*np.cos(T)-offset)**2+(R*np.sin(T))**2)/w**2)
            brute = np.trapezoid(np.trapezoid(intensity*R, r, axis=1), th)
            self.assertAlmostEqual(float(response.pda_fraction([offset, 0])), brute, delta=1e-5)

    def test_pdf_equivalent_focal_response_is_pointing_sensitive(self):
        g = GaussianFocalResponse(1.55e-6, 0.2, 0.02036, 0.005, 75e-6)
        self.assertAlmostEqual(float(g.normalized_power([0, 0])), 1.0)
        self.assertGreater(float(g.normalized_power([30e-6, 0])), 0.97)
        self.assertLess(float(g.normalized_power([80e-6, 0])), 0.5)

    def test_hold_sees_the_rail(self):
        episodes = build_episodes(self.spec, 'sweep', [9], 6.0)
        result = simulate(self.spec, episodes, HoldController(), [], trace_index=(0,))
        rail = episodes.alpha[0, :, 0]*self.spec.reducer*self.spec.psd_lever
        detector = result['traces'][0]['focal'][:, 0]
        # Without control the detector-plane x follows rail/10 (plus FSM offset and jitter).
        self.assertGreater(np.corrcoef(detector, rail)[0, 1], 0.99)


if __name__ == '__main__':
    unittest.main()
