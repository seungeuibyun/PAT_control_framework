"""Focal-mode bench (config/focal_mode.json, from legacy/pat_bench_v2): loop timing, information boundary, predictors.

The loop, controllers and predictors are shared with the as-built direct mode; the focal
config keeps the lens-focused geometry where the lever equals the focal length.
"""
import dataclasses
from pathlib import Path
import unittest

import numpy as np

from pat.predictor import ContextModel, replay_online
from pat.controllers import NominalPlant, PIDController, PredictiveController
from pat.optics import BenchOptics
from pat.plant import LoopContext, build_episodes, fsm_discretization, open_loop_frames, simulate
from pat.spec import load_spec

FOCAL = Path(__file__).resolve().parents[1]/'config/focal_mode.json'


def ideal(spec):
    """No unknown hardware errors: isolates the nominal-model logic."""
    return spec.variant(**{'fsm.jitter_rms_rad': 0.0, 'fsm.static_offset_rms_rad': 0.0,
                           'psd.position_noise_rms_m': 0.0, 'calibration.fsm_gain_error_rms': 0.0,
                           'calibration.fsm_coupling_error_rms': 0.0, 'calibration.target_error_rms_m': 0.0})


class RecordingPredictor(PredictiveController):
    """Issues arbitrary commands and records its pseudo-open-loop reconstruction."""
    def __init__(self, model):
        super().__init__(model, 'recording')
        self.rng = np.random.default_rng(3)
        self.recorded = []

    def on_frame(self, reading, capture_step, now):
        self.plant.advance(now+1)
        self.recorded.append(reading/self.f-self.plant.contribution(capture_step))

    def tick(self, now):
        self.plant.advance(now+1)
        command = self.rng.normal(0, 200e-6, (self.plant.state.shape[0], 2))
        self.plant.issue(now, command)
        return command


class FocalModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(FOCAL)

    def test_equipment_derived_values(self):
        s = self.spec
        self.assertAlmostEqual(s.beam_radius_at_fsm, 4.355e-3, delta=0.01e-3)
        self.assertLess(s.pda_peak_voltage, s.optics.pda_max_output_v)
        self.assertGreaterEqual(s.psd.frame_period_s, 1/50)
        self.assertEqual((s.capture_steps, s.delivery_steps, s.tick_steps, s.latency_steps), (20, 40, 4, 2))
        with self.assertRaises(ValueError):
            s.variant(**{'psd.frame_period_s': 0.01, 'psd.delivery_after_query_s': 0.01})
        with self.assertRaises(ValueError):
            s.variant(**{'optics.pda_branch_nd_transmission': 1.0})

    def test_optics_matches_analytic_gaussian(self):
        spec = self.spec.variant(**{'optics.fsm_mirror_diameter_m': 0.1, 'optics.lens_clear_aperture_m': 0.1})
        optics = BenchOptics(spec)
        wf, a = optics.spot_radius_1e2, spec.optics.pda_radius_m
        r, th = np.linspace(0, a, 500), np.linspace(0, 2*np.pi, 500)
        R, T = np.meshgrid(r, th)
        for offset in (0, 40e-6, 60e-6, 80e-6):
            intensity = 2/(np.pi*wf**2)*np.exp(-2*((R*np.cos(T)-offset)**2+(R*np.sin(T))**2)/wf**2)
            analytic = np.trapezoid(np.trapezoid(intensity*R, r, axis=1), th)
            for point in ([offset, 0], [0, offset]):
                self.assertAlmostEqual(float(optics.pda_fraction(point)), analytic, delta=3e-3)

    def test_fsm_settling_matches_datasheet(self):
        ad, bd = fsm_discretization(self.spec)
        for axis, datasheet in enumerate((2.2e-3, 2.0e-3)):
            x, out = np.zeros(2), []
            for _ in range(200):
                out.append(x[0])
                x = ad[axis]@x+bd[axis]
            outside = np.nonzero(abs(np.asarray(out)-1) > 0.05)[0]
            settling = (outside[-1]+1)*self.spec.dt
            self.assertAlmostEqual(settling, datasheet, delta=0.5e-3)

    def test_open_loop_frames_equal_hold_loop(self):
        episodes = build_episodes(self.spec, 'train', [11, 12], 3.0)
        frames, _ = open_loop_frames(self.spec, episodes)
        from pat.controllers import HoldController
        result = simulate(self.spec, episodes, HoldController(), [], trace_index=(0, 1))
        for i in (0, 1):
            delivered = np.array([f[2] for f in result['traces'][i]['frames']])
            np.testing.assert_allclose(delivered, frames[i, :len(delivered)], rtol=0, atol=1e-15)

    def test_pseudo_open_loop_reconstruction_is_exact_with_ideal_hardware(self):
        spec = ideal(self.spec)
        episodes = build_episodes(spec, 'train', [21, 22], 2.0)
        model = ContextModel(4, 2)
        model.sigma_diff = model.sigma_abs = 1.0
        model.beta = np.zeros((model.size, 2*model.horizon))
        controller = RecordingPredictor(model)
        simulate(spec, episodes, controller, [])
        recorded = np.stack(controller.recorded, axis=1)
        expected = open_loop_frames(spec, episodes)[0]/spec.psd_lever
        # Commands differ from the hold run, yet the reconstruction recovers the same disturbance frames.
        np.testing.assert_allclose(recorded, expected[:, :recorded.shape[1]], rtol=0, atol=2*spec.psd_lsb/spec.psd_lever)

    def test_controllers_receive_only_nominal_information(self):
        fields = {f.name for f in dataclasses.fields(LoopContext)}
        self.assertEqual(fields, {'spec', 'batch', 'gain_model', 'target'})
        episodes = build_episodes(self.spec, 'train', [31], 1.0)
        np.testing.assert_array_equal(episodes.gain_model[0], self.spec.fsm_gain)
        self.assertFalse(np.allclose(episodes.gain_true[0], self.spec.fsm_gain))

    def test_pid_removes_constant_offset(self):
        spec = ideal(self.spec)
        episodes = build_episodes(spec, 'train', [41], 4.0)
        episodes.alpha[:] = np.array([300e-6, -200e-6])
        result = simulate(spec, episodes, PIDController(0.1, 20.0, 0.0, 0.01), [], trace_index=(0,))
        # The PSD reading is quantized (12 bit over 10 mm), so the loop settles within half an LSB.
        self.assertLessEqual(np.max(abs(result['traces'][0]['focal'][-200:])), spec.psd_lsb/2+1e-12)

    def test_linear_context_predicts_a_known_tone_and_online_refit_adapts(self):
        dt, rng = 0.02, np.random.default_rng(5)
        tone = lambda hz, n: [np.stack([np.sin(2*np.pi*hz*np.arange(n)*dt+p) for p in rng.uniform(0, 6, 2)], 1)*1e-4
                              for _ in range(6)]
        model = ContextModel(6, 3).fit(tone(8.0, 600), tone(8.0, 600), [1e-8])
        self.assertLess(np.sqrt(model.fit_report['validation_mse_h1_h2']), 1e-7)
        offline = replay_online(model, tone(15.0, 600), 200, 25, None)
        adapted = replay_online(model, tone(15.0, 600), 200, 25, 1e-3)
        self.assertLess(adapted, 0.1*offline)

    def test_nominal_plant_buffer_guards_old_requests(self):
        plant = NominalPlant(self.spec, 1, self.spec.fsm_gain[None], buffer=8)
        plant.advance(20)
        with self.assertRaises(ValueError):
            plant.contribution(5)


if __name__ == '__main__':
    unittest.main()
