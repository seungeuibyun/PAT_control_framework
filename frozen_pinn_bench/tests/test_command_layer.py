"""Frozen-PINN command layer: FSM-ODE residual solution and closed-loop behaviour."""
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import expm

from pat.command_layer import PINNCommandLayer
from pat.spec import load_spec


class PINNCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(Path(__file__).resolve().parents[1]/'config/bench.json')
        cls.cfg = dict(cls.spec.raw['controllers']['pinn_command'], hold_ticks=3, point_ticks=3)

    def exact(self, theta0, rate0, u, tau, axis):
        w, z = self.spec.omega[axis], self.spec.fsm.damping
        a = np.array([[0, 1], [-w*w, -2*z*w]])
        b = np.array([0, w*w])
        return np.array([(expm(a*t)@[theta0, rate0]+np.linalg.solve(a, (expm(a*t)-np.eye(2))@b)*u)[0] for t in tau])

    def test_angle_trajectory_satisfies_the_fsm_dynamics(self):
        layer = PINNCommandLayer(self.spec, self.spec.fsm_gain, self.cfg)
        rng = np.random.default_rng(1)
        for _ in range(10):
            theta0, rate0 = rng.normal(0, 2e-4, (1, 2)), rng.normal(0, 0.1, (1, 2))
            d_hat = rng.normal(0, 5e-5, (1, 1, 2))+rng.normal(0, 1e-6, (1, layer.n_point, 2))
            trajectory, u = layer.trajectory(theta0, rate0, theta0, np.zeros((1, 2)), d_hat)
            for axis in range(2):
                exact = self.exact(theta0[0, axis], rate0[0, axis], u[0, axis], layer.tau, axis)
                self.assertLess(np.max(abs(trajectory[0, :, axis]-exact)), 1e-3*np.ptp(exact)+1e-12)

    def test_static_disturbance_is_cancelled_at_rest(self):
        layer = PINNCommandLayer(self.spec, self.spec.fsm_gain, self.cfg)
        d = np.array([[40e-6, -25e-6]])
        expected = -np.linalg.solve(self.spec.psd_lever*self.spec.fsm_gain, d[0])
        state = np.array([expected])
        u = layer.command(state, np.zeros((1, 2)), state, np.zeros((1, 2)),
                          np.repeat(d[:, None, :], layer.n_point, axis=1))
        np.testing.assert_allclose(u[0], expected, rtol=1e-6, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
