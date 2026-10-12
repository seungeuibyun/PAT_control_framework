"""Receiver-only Frozen-PINN adapter for measured PSD/PD/FSM calibration.

Reuse the reference repository's frozen tanh/sine basis and constrained
optimizer. Fit the local Gaussian receiving-amplitude equation
    grad(E) + 0.5 K (p - mu) E = 0
with measured PD voltage and its measured dark offset. This is a local
receiver coupling model, not an inferred 20 km complex atmospheric field.
PSD coordinates are millimetres. Commands use either measured current units
or the mirror's verified internal normalized XY position servo.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parent
# validation/ lives directly inside PAT_control_framework. The numerical
# primitives are imported from the existing repository, not copied here.
sys.path.insert(0, str(ROOT.parent))
from config.settings import FrozenPINNConfig
from solver.frozen_pinn import FrozenPINNBasis
from solver.optimization import projected_gradient_ascent, project_pat_command


class CalibrationError(RuntimeError):
    pass


class ReceiverPINN:
    def __init__(self, record, modal_basis, coefficients):
        self.record = record
        self.Q = np.asarray(modal_basis)
        self.coefficients = np.asarray(coefficients)
        self.s = record['spectral_side']
        self.half_width = record['normalized_half_width']
        self.center = np.asarray(record['PD_peak_PSD_mm'])
        self.scale = record['coordinate_scale_mm']
        self.peak = record['PD_peak_above_dark_V']
        self.modes = np.arange(1, self.s + 1) * np.pi / (2 * self.half_width)
        self.modal = (self.Q @ self.coefficients).reshape(self.s, self.s)
        self.support = Delaunay(np.asarray(record['training_PSD_mm']))

    def amplitude_features(self, positions):
        q = (np.asarray(positions).reshape(-1, 2) - self.center) / self.scale
        sx = np.sin((q[:, 0, None] + self.half_width) * self.modes) / np.sqrt(self.half_width)
        sy = np.sin((q[:, 1, None] + self.half_width) * self.modes) / np.sqrt(self.half_width)
        dx = np.cos((q[:, 0, None] + self.half_width) * self.modes) * self.modes / np.sqrt(self.half_width)
        dy = np.cos((q[:, 1, None] + self.half_width) * self.modes) * self.modes / np.sqrt(self.half_width)
        modal = self.Q.reshape(self.s, self.s, -1)
        feature = lambda x, y: np.einsum('ni,nj,jir->nr', x, y, modal, optimize=True)
        return feature(sx, sy), feature(dx, sy), feature(sx, dy)

    def contains(self, p):
        return bool(self.support.find_simplex(np.asarray(p)) >= 0)

    def voltage_gradient(self, p):
        q = (np.asarray(p) - self.center) / self.scale
        sx = np.sin((q[0] + self.half_width) * self.modes) / np.sqrt(self.half_width)
        sy = np.sin((q[1] + self.half_width) * self.modes) / np.sqrt(self.half_width)
        dx = np.cos((q[0] + self.half_width) * self.modes) * self.modes / np.sqrt(self.half_width)
        dy = np.cos((q[1] + self.half_width) * self.modes) * self.modes / np.sqrt(self.half_width)
        amp = float(sy @ self.modal @ sx)
        grad = np.array([sy @ self.modal @ dx, dy @ self.modal @ sx]) / self.scale
        return self.peak * amp**2, 2 * self.peak * amp * grad

    def save(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        np.savez_compressed(path.with_suffix('.npz'), modal_basis=self.Q,
                            coefficients=self.coefficients)

    @classmethod
    def load(cls, path):
        path = Path(path)
        record = json.loads(path.read_text(encoding='utf-8'))
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
            return cls(record, arrays['modal_basis'], arrays['coefficients'])


def train_receiver_pinn(dataset):
    rows = dataset['samples']
    if not dataset.get('dark_verified') or len(rows) < 24:
        raise CalibrationError('A verified dark baseline and at least 24 real calibration samples are required')
    p = np.array([r['PSD_mm'] for r in rows], float)
    q = np.array([r['FSM_feedback_xy'] for r in rows], float)
    current = np.array([r['FSM_current_mA'] for r in rows], float)
    voltage = np.array([r['PD_V'] for r in rows], float) - dataset['PD_dark_V']
    if not np.all(np.isfinite(np.column_stack((p, q, current, voltage)))) or np.any(voltage <= .03):
        raise CalibrationError('Nonfinite or indistinguishable PD optical signals')
    if np.any(np.ptp(p, axis=0) < .05):
        raise CalibrationError('Calibration did not excite both PSD axes sufficiently')
    p0, q0, c0 = p.mean(axis=0), q.mean(axis=0), current.mean(axis=0)
    geometry = np.linalg.lstsq(np.column_stack((q-q0, np.ones(len(q)))), p, rcond=None)[0]
    G = geometry[:2].T
    geometry_rms = float(np.sqrt(np.mean((np.column_stack((q-q0, np.ones(len(q)))) @ geometry-p)**2)))
    actuator_mode = dataset.get('actuator_mode', 'manual_current')
    if actuator_mode not in ('manual_current', 'device_xy_servo'):
        raise CalibrationError('Unknown measured actuator mode')
    actuator = np.linalg.lstsq(current-c0, q-q0, rcond=None)[0].T
    command_J = G if actuator_mode == 'device_xy_servo' else G @ actuator
    if (np.linalg.cond(G) > 50 or np.linalg.cond(command_J) > 100
            or geometry_rms > .03 or np.any(np.ptp(q, axis=0) < .0002)):
        raise CalibrationError('The measured two-axis geometry or actuator response is unreliable')
    z = p-p0
    design = np.column_stack((np.ones(len(p)), z, z[:, 0]**2,
                              z[:, 0]*z[:, 1], z[:, 1]**2))
    if np.linalg.matrix_rank(design) < 6:
        raise CalibrationError('The PD receiving profile is not observable in two dimensions')
    weights = np.linalg.lstsq(design, np.log(voltage), rcond=None)[0]
    K = -np.array([[2*weights[3], weights[4]], [weights[4], 2*weights[5]]])
    if np.linalg.eigvalsh(K).min() <= .001:
        raise CalibrationError('Measurements do not establish a unique local PD receiving peak')
    offset = np.linalg.solve(K, weights[1:3])
    center = p0 + offset
    peak = float(np.exp(weights[0] + .5*offset @ K @ offset))
    hull = Delaunay(p)
    if hull.find_simplex(center) < 0:
        raise CalibrationError('Predicted PD maximum lies outside the measured calibration region')
    gaussian_rms = float(np.sqrt(np.mean((np.exp(design @ weights)-voltage)**2))/peak)
    if gaussian_rms > .08:
        raise CalibrationError('Measured PD response is not compatible with the local Gaussian coupling model')
    scale = float(1 / np.sqrt(np.linalg.eigvalsh(K).min()))
    f = FrozenPINNConfig(hidden_width=80, collocation_side=48, spectral_side=24,
                         feature_scale_max=16, boundary_envelope_fraction=.65, seed=17)
    half_width = 5.0
    basis = FrozenPINNBasis(None, SimpleNamespace(
        optical=SimpleNamespace(half_width=half_width), frozen_pinn=f))
    record = {'state': 'training', 'model': 'measured_receiver_Gaussian_coupling_Frozen_PINN',
        'scope': 'Local receiver coupling; no source-stage truth or atmospheric field reconstruction',
        'reference_commit': dataset['reference_commit'], 'simulation': bool(dataset.get('simulation')),
        'spectral_side': f.spectral_side, 'normalized_half_width': half_width,
        'coordinate_scale_mm': scale, 'PD_peak_PSD_mm': center.tolist(),
        'PD_peak_above_dark_V': peak, 'PD_dark_V': dataset['PD_dark_V'],
        'gaussian_precision_per_mm2': K.tolist(), 'training_PSD_mm': p.tolist(),
        'feedback_to_PSD_mm_per_xy': G.tolist(), 'actuator_mode': actuator_mode,
        'command_unit': 'normalized_mirror_xy' if actuator_mode == 'device_xy_servo' else 'mA',
        'command_to_PSD_mm': command_J.tolist(),
        'calibration_current_center_mA': c0.tolist(), 'calibration_feedback_center': q0.tolist(),
        'geometry_fit_rms_mm': geometry_rms, 'gaussian_profile_relative_rms': gaussian_rms,
        'source_current_mA': dataset.get('source_current_mA'),
        'hardware': dataset.get('hardware'), 'dark_verified': True,
        'physical_center_coincidence_verified': False,
        'neural_primitive': 'PINN_reference.solver.frozen_pinn.FrozenPINNBasis',
        'optimizer': 'PINN_reference.solver.optimization.projected_gradient_ascent',
        'physics_equation': 'grad(E) + 0.5 K (p-mu) E = 0; Gaussian overlap amplitude at a small receiver',
        'training_sample_count': len(rows)}
    model = ReceiverPINN(record, basis.Q, np.zeros(basis.R))
    if actuator_mode == 'manual_current':
        record['incremental_current_to_PSD_mm_per_mA'] = command_J.tolist()
    a, _, _ = model.amplitude_features(p)
    axis = np.linspace(-2, 2, 13)
    cx, cy = np.meshgrid(axis, axis)
    physical_points = center + scale*np.column_stack((cx.ravel(), cy.ravel()))
    value, dx, dy = model.amplitude_features(physical_points)
    normalized_precision = K*scale**2
    drift = .5 * ((physical_points-center)/scale) @ normalized_precision.T
    physics = np.vstack((dx + drift[:, 0, None]*value,
                          dy + drift[:, 1, None]*value))
    # Independently withhold every fifth measured point from the neural fit.
    validation = np.arange(len(p)) % 5 == 0
    train = ~validation
    matrices = np.vstack((a[train]/np.sqrt(train.sum()),
                           .20*physics/np.sqrt(len(physics)), 1e-5*np.eye(basis.R)))
    targets = np.concatenate((np.sqrt(voltage[train]/peak)/np.sqrt(train.sum()),
                              np.zeros(len(physics)+basis.R)))
    coeff = np.linalg.lstsq(matrices, targets, rcond=1e-10)[0]
    model = ReceiverPINN(record, basis.Q, coeff)
    prediction = np.array([model.voltage_gradient(v)[0] for v in p])
    val_error = float(np.sqrt(np.mean((prediction[validation]-voltage[validation])**2))/peak)
    physics_error = float(np.sqrt(np.mean((physics @ coeff)**2)))
    if val_error > .10 or physics_error > .12:
        raise CalibrationError('Frozen-PINN failed independent measurements or physics residual checks')
    record.update(state='validated_local_receiver_PINN',
                  withheld_PD_relative_rms=val_error, physics_residual_rms=physics_error,
                  basis_rank=int(basis.R))
    return model


class ReceiverController:
    def __init__(self, model):
        if model.record['state'] != 'validated_local_receiver_PINN':
            raise CalibrationError('Receiver PINN is not validated')
        self.model = model
        self.actuator_mode = model.record.get('actuator_mode', 'manual_current')
        self.J = np.asarray(model.record.get('command_to_PSD_mm',
            model.record.get('incremental_current_to_PSD_mm_per_mA')))
        self.reference = model.center.copy()
        xy = self.actuator_mode == 'device_xy_servo'
        self.tracking = SimpleNamespace(theta_max=.25 if xy else 50., theta_slew_max=.0008 if xy else 2.,
            theta_quantization=1e-6 if xy else .01, max_opt_iterations=8, optimizer_step=.0003 if xy else .8,
            line_search_shrink=.5, line_search_steps=8, optimizer_min_step=1e-8 if xy else 1e-6,
            gradient_norm_tol=1e-8, objective_rel_tol=1e-6, objective_abs_tol=1e-7,
            control_regularization=.0001)

    def next_command(self, measured_PSD_mm, previous_command):
        p, previous = np.asarray(measured_PSD_mm), np.asarray(previous_command)
        if not self.model.contains(p):
            raise CalibrationError('Actual PSD position is outside the measured PINN region')
        def objective(command):
            predicted = p + self.J @ (np.asarray(command)-previous)
            if not self.model.contains(predicted):
                return -1e6, np.zeros(2)
            power, gradient = self.model.voltage_gradient(predicted)
            return power, self.J.T @ gradient
        initial = project_pat_command(previous + np.linalg.pinv(self.J) @
                                      (self.reference-p), previous, self.tracking)
        if objective(initial)[0] < 0:
            initial = previous
        command, power, diagnostics = projected_gradient_ascent(previous,
            objective, lambda v: objective(v)[0], self.tracking, initial)
        diagnostics['continuous_voltage_trace_V'] = diagnostics.pop('continuous_power_trace_w')
        return command, {'predicted_PD_above_dark_V': power,
                          'predicted_PSD_mm': (p + self.J @ (command-previous)).tolist(),
                          'PSD_error_mm': (self.reference-p).tolist(),
                          'optimizer_stop_reason': diagnostics['optimizer_stop_reason']}
