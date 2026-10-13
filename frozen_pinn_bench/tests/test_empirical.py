"""Empirical branch: bench-log loading, analysis, and deployable-model round trip."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pat.comm import HAPSReceiver, LinkModel
from pat.empirical import analyse, load_session
from pat.predictor import ContextModel, load_model, save_model
from pat.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def write_run(folder, name, info, arrays):
    np.savez(folder/'runs'/f'{name}.npz', **arrays)
    (folder/'runs'/f'{name}.json').write_text(json.dumps(info))


def frames(offset_m, seconds=4.0, jitter_m=0.0, seed=0):
    t = np.arange(0, seconds, 0.02)+0.013
    rng = np.random.default_rng(seed)
    xy = np.tile(offset_m, (len(t), 1))+jitter_m*rng.standard_normal((len(t), 2))
    return dict(frame_t_s=t, psd_xy_m=xy, command_t_s=np.arange(0, seconds, 0.002),
                command_rad=np.zeros((int(round(seconds/0.002)), 2)))


class EmpiricalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(ROOT/'config/bench.json')
        ev = cls.spec.raw['evaluation']
        cls.link = LinkModel(ev)
        cls.receivers = {'r25': HAPSReceiver(ev['comm'], 25e-6, cls.spec.psd_lever)}

    def session(self, root):
        folder = Path(root)
        (folder/'runs').mkdir()
        target = [1e-4, -2e-4]
        (folder/'session.json').write_text(json.dumps(dict(calibration=dict(p_tar_m=target), warmup_s=1.0)))
        return folder, np.asarray(target)

    def test_residual_is_reading_minus_target_after_warmup(self):
        with tempfile.TemporaryDirectory() as root:
            folder, target = self.session(root)
            # 3-4-5 offset from the target -> 5 um RMS; PID runs twice, Frozen-PINN once.
            for seed in (0, 1):
                write_run(folder, f'station_s{seed}__pid', dict(family='station', controller='pid'),
                          frames(target+[3e-6, 4e-6]))
            write_run(folder, 'station_s0__frozen_pinn', dict(family='station', controller='frozen_pinn'),
                      frames(target+[0.0, 1e-6]))
            write_run(folder, 'acquisition_s0__pid', dict(family='acquisition', controller='pid'),
                      frames(target+[2e-3, 0.0], seconds=1.0))
            meta, runs = load_session(folder)
            summary, arrays = analyse(meta, runs, self.spec, self.link, self.receivers)
        station = summary['scenarios']['station']
        self.assertAlmostEqual(station['pid']['rms_residual_um'], 5.0, places=6)
        self.assertAlmostEqual(station['frozen_pinn']['rms_residual_um'], 1.0, places=6)
        self.assertEqual(station['pid']['runs'], 2)
        t = frames(target)['frame_t_s']
        self.assertEqual(station['pid']['frames'], 2*np.count_nonzero(t-t[0] >= 1.0))
        self.assertNotIn('acquisition', summary['scenarios'])
        np.testing.assert_allclose(arrays['demo/pid/residual_m'][:, 0], 2e-3)
        comm = summary['comm']['station']
        self.assertGreater(comm['frozen_pinn']['r25']['mean_power'], comm['pid']['r25']['mean_power'])
        self.assertEqual(len(arrays['comm/station/pid/r25/ber64']), len(arrays['comm/snr_db']))

    def test_rejects_malformed_logs(self):
        with tempfile.TemporaryDirectory() as root:
            folder, target = self.session(root)
            bad = frames(target)
            del bad['command_rad']
            write_run(folder, 'station_s0__pid', dict(family='station', controller='pid'), bad)
            with self.assertRaises(ValueError):
                load_session(folder)
        with tempfile.TemporaryDirectory() as root:
            folder, target = self.session(root)
            bad = frames(target)
            bad['frame_t_s'] = bad['frame_t_s'][::-1]
            write_run(folder, 'station_s0__pid', dict(family='station', controller='pid'), bad)
            with self.assertRaises(ValueError):
                load_session(folder)

    def test_model_round_trip(self):
        rng = np.random.default_rng(4)
        seqs = [np.cumsum(rng.standard_normal((300, 2)), axis=0)*1e-6 for _ in range(6)]
        model = ContextModel(6, 2, 16, seed=1).fit(seqs[:4], seqs[4:], [1e-6, 1e-3], candidates=2000)
        with tempfile.TemporaryDirectory() as root:
            save_model(model, Path(root)/'m.npz')
            loaded = load_model(Path(root)/'m.npz')
        u, _ = model.samples(seqs[4:])
        np.testing.assert_array_equal(model.features(u)@model.beta, loaded.features(u)@loaded.beta)


if __name__ == '__main__':
    unittest.main()
