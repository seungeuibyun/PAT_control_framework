"""HAPS-equivalent link metrics and the injected-vibration rail family."""
import json
from pathlib import Path
import unittest

import numpy as np

from pat.comm import HAPSReceiver, LinkModel, ber_qam, simulate_symbols, threshold_snr
from pat.plant import build_episodes
from pat.spec import load_spec


class CommTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(Path(__file__).resolve().parents[1]/'config/bench.json')
        cls.ev = cls.spec.raw['evaluation']

    def test_thresholds_and_ber(self):
        th = {m: threshold_snr(m, 3.8e-3) for m in (4, 16, 64, 256)}
        self.assertAlmostEqual(ber_qam(64, th[64]), 3.8e-3, delta=1e-5)
        self.assertTrue(th[4] < th[16] < th[64] < th[256])
        self.assertAlmostEqual(10*np.log10(th[64]), 21.1, delta=0.1)

    def test_receivers_and_operating_point(self):
        r75 = HAPSReceiver(self.ev['comm'], 75e-6, self.spec.psd_lever)
        r25 = HAPSReceiver(self.ev['comm'], 25e-6, self.spec.psd_lever)
        self.assertAlmostEqual(float(r75.eta([0.0, 0.0])), 1.0)
        self.assertGreater(float(r75.eta([30e-6, 0])), float(r25.eta([30e-6, 0])))
        link = LinkModel(self.ev)
        aligned = link.metrics(np.ones((2, 10)), link.operating_snr)
        self.assertEqual(aligned['outage64'], 0.0)
        self.assertAlmostEqual(aligned['rate_bps'], 6*link.rate_scale)
        faded = link.metrics(np.full((2, 10), 0.5), link.operating_snr)
        self.assertEqual(faded['outage64'], 1.0)

    def test_symbol_simulation_is_error_free_at_high_snr(self):
        rng = np.random.default_rng(0)
        sym = simulate_symbols(lambda t: np.ones_like(t), np.linspace(0, 0.01, 2000), 10**4, 5e-4, 128, rng)
        self.assertEqual(int(sym['bit_errors'].sum()), 0)

    def test_injected_vibration_is_deterministic_and_two_axis(self):
        a = build_episodes(self.spec, 'vibration', [11, 12], 3.0)
        b = build_episodes(self.spec, 'vibration', [11, 12], 3.0)
        np.testing.assert_array_equal(a.alpha, b.alpha)
        station = build_episodes(self.spec, 'station', [11, 12], 3.0)
        self.assertGreater(np.max(abs(a.alpha-station.alpha)), 1e-5)


if __name__ == '__main__':
    unittest.main()
