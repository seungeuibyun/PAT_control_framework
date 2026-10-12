"""Measured, local ray-geometry Frozen-PINN for the FSM/PSD position loop.

Reuse the upstream frozen tanh/sine neural basis and constrained optimizer.
The two network outputs are PSD x/y in mm, given actual mirror x/y feedback.
The local first-order reflected-ray constraint is dp/dq = G, with scale and
rotation identified from real mirror/PSD samples. No PD profile, transmitter
position or simulated atmospheric field is needed for this commissioning step.
This does not implement the upstream full atmospheric propagation equation.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import Delaunay

from receiver_pinn import (CalibrationError, FrozenPINNBasis, FrozenPINNConfig,
                           projected_gradient_ascent, project_pat_command)


class PSDPositionPINN:
    def __init__(self, record, modal_basis, coefficients):
        self.record = record
        self.Q = np.asarray(modal_basis, float)
        self.coefficients = np.asarray(coefficients, float)
        self.s = record['spectral_side']
        self.half_width = record['normalized_half_width']
        self.center = np.asarray(record['feedback_center_xy'], float)
        self.qscale = float(record['feedback_coordinate_scale'])
        self.pcenter = np.asarray(record['PSD_coordinate_center_mm'], float)
        self.pscale = float(record['PSD_coordinate_scale_mm'])
        self.modes = np.arange(1, self.s + 1) * np.pi / (2 * self.half_width)
        self.support = Delaunay(np.asarray(record['measured_feedback_xy'], float))

    def features(self, positions):
        z = (np.asarray(positions, float).reshape(-1, 2)-self.center)/self.qscale
        h, modes = self.half_width, self.modes
        sx = np.sin((z[:, 0, None]+h)*modes)/np.sqrt(h)
        sy = np.sin((z[:, 1, None]+h)*modes)/np.sqrt(h)
        dx = np.cos((z[:, 0, None]+h)*modes)*modes/np.sqrt(h)
        dy = np.cos((z[:, 1, None]+h)*modes)*modes/np.sqrt(h)
        modal = self.Q.reshape(self.s, self.s, -1)
        def feature(x, y):
            return np.einsum('ni,nj,jir->nr', x, y, modal, optimize=True)
        return feature(sx, sy), feature(dx, sy), feature(sx, dy)

    def contains(self, command):
        q = np.asarray(command, float)
        return q.shape == (2,) and np.all(np.isfinite(q)) and bool(
            self.support.find_simplex(q, tol=1e-10) >= 0)

    def position_and_jacobian(self, command):
        a, dx, dy = self.features(command)
        value = self.pcenter+self.pscale*(a@self.coefficients)[0]
        jacobian = self.pscale/self.qscale*np.column_stack((
            (dx@self.coefficients)[0], (dy@self.coefficients)[0]))
        if not np.all(np.isfinite(np.r_[value, jacobian.ravel()])):
            raise CalibrationError('Nonfinite neural ray prediction')
        return value, jacobian

    def save(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        np.savez_compressed(path.with_suffix('.npz'), modal_basis=self.Q,
                            coefficients=self.coefficients)

    @classmethod
    def load(cls, path):
        path = Path(path)
        record = json.loads(path.read_text(encoding='utf-8'))
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
            return cls(record, arrays['modal_basis'], arrays['coefficients'])


def train_psd_position_pinn(dataset):
    rows = dataset['center_calibration_samples']
    if dataset.get('state') != 'real_PSD_FSM_closed_loop_verified' or len(rows) < 25:
        raise CalibrationError('A verified, two-axis center calibration with at least 25 samples is required')
    q = np.array([r['FSM_feedback_xy'] for r in rows], float)
    p = np.array([r['PSD_mm'] for r in rows], float)
    signal = np.array([r['PSD_percent'] for r in rows], float)
    if not np.all(np.isfinite(np.column_stack((q, p, signal)))) or np.any(
            (signal < 2) | (signal > 90)) or np.any(abs(p) > 4.5):
        raise CalibrationError('Calibration includes missing, clipped or overloaded PSD data')
    if np.min(np.ptp(q, axis=0)) < .003 or np.min(np.ptp(p, axis=0)) < .3:
        raise CalibrationError('Both mirror and PSD axes must have been measured')
    q0, p0 = q.mean(0), p.mean(0)
    # Withhold interleaved measured points before fitting either G or the NN.
    validation = np.arange(len(q)) % 5 == 2
    train = ~validation
    design = np.column_stack((q-q0, np.ones(len(q))))
    fit = np.linalg.lstsq(design[train], p[train], rcond=None)[0]
    G = fit[:2].T
    ray_rms = float(np.sqrt(np.mean((design@fit-p)**2)))
    if np.linalg.cond(G) > 10 or ray_rms > .03:
        raise CalibrationError('Measured local ray geometry is unreliable')
    qscale = float(np.max(np.ptp(q, axis=0))/2)
    pscale = float(max(np.max(np.ptp(p, axis=0))/2, .1))
    cfg = FrozenPINNConfig(hidden_width=80, collocation_side=48, spectral_side=24,
                          feature_scale_max=16, boundary_envelope_fraction=.65, seed=23)
    basis = FrozenPINNBasis(None, SimpleNamespace(
        optical=SimpleNamespace(half_width=5.), frozen_pinn=cfg))
    record = {
        'state': 'training', 'model': 'measured_local_ray_geometry_Frozen_PINN',
        'scope': 'Local receiver position control, before physical PD alignment',
        'reference_commit': dataset.get('reference_commit',
            '1192e7e92b885bfcb72e1c712948c8624cd3b3b5'),
        'simulation': bool(dataset.get('simulation', False)),
        'source_current_mA': dataset.get('source', {}).get('current_mA'),
        'spectral_side': cfg.spectral_side, 'normalized_half_width': 5.,
        'feedback_center_xy': q0.tolist(), 'feedback_coordinate_scale': qscale,
        'PSD_coordinate_center_mm': p0.tolist(), 'PSD_coordinate_scale_mm': pscale,
        'measured_feedback_xy': q.tolist(), 'measured_PSD_mm': p.tolist(),
        'feedback_to_PSD_mm_per_xy': G.tolist(), 'linear_geometry_rms_mm': ray_rms,
        'training_sample_count': int(train.sum()), 'withheld_sample_count': int(validation.sum()),
        'withheld_sample_indices': np.where(validation)[0].tolist(),
        'neural_primitive': 'PINN_reference.solver.frozen_pinn.FrozenPINNBasis',
        'optimizer': 'PINN_reference.solver.optimization.projected_gradient_ascent',
        'physics_equation': 'd(PSD_position_mm)/d(mirror_feedback_xy) - measured_local_ray_Jacobian = 0',
        'physics_scope': 'First-order reflected-ray approximation; measured scale, signs and axis rotation',
        'actuator_mode': 'device_xy_servo', 'command_unit': 'normalized_mirror_xy',
        'external_PAT_PID_or_KF_used': False, 'source_stage_position_used': False,
        'PD_input_required': False, 'physical_center_coincidence_verified': False,
        'absolute_PSD_coordinate_accuracy_verified': False,
    }
    model = PSDPositionPINN(record, basis.Q, np.zeros((basis.R, 2)))
    a, _, _ = model.features(q)
    # Physics collocation lies wholly inside the actual measured command hull.
    support = model.support
    qaxis = np.linspace(-.85, .85, 9)
    cx, cy = np.meshgrid(qaxis, qaxis)
    collocation = q0+qscale*np.column_stack((cx.ravel(), cy.ravel()))
    collocation = collocation[support.find_simplex(collocation) >= 0]
    _, dx, dy = model.features(collocation)
    normalized_G = G*qscale/pscale
    weight = .35
    matrices = np.vstack((a[train]/np.sqrt(train.sum()),
        weight*dx/np.sqrt(len(dx)), weight*dy/np.sqrt(len(dy)), 1e-5*np.eye(basis.R)))
    targets = np.vstack(((p[train]-p0)/pscale/np.sqrt(train.sum()),
        weight*np.tile(normalized_G[:, 0], (len(dx), 1))/np.sqrt(len(dx)),
        weight*np.tile(normalized_G[:, 1], (len(dy), 1))/np.sqrt(len(dy)),
        np.zeros((basis.R, 2))))
    coeff = np.linalg.lstsq(matrices, targets, rcond=1e-10)[0]
    model = PSDPositionPINN(record, basis.Q, coeff)
    prediction = p0+pscale*(a@coeff)
    val_rms = float(np.sqrt(np.mean((prediction[validation]-p[validation])**2)))
    derivative_rms = float(np.sqrt(np.mean(np.vstack((
        dx@coeff-normalized_G[:, 0], dy@coeff-normalized_G[:, 1]))**2)))
    if not np.isfinite(val_rms+derivative_rms) or val_rms > .03 or derivative_rms > .05:
        raise CalibrationError('Neural ray model failed withheld measurements or physics residual checks')
    record.update(state='validated_local_PSD_position_PINN', withheld_PSD_rms_mm=val_rms,
                  normalized_physics_derivative_rms=derivative_rms, basis_rank=int(basis.R))
    return model


class PSDPositionController:
    def __init__(self, model):
        if model.record.get('state') != 'validated_local_PSD_position_PINN':
            raise CalibrationError('The PSD position PINN is not validated')
        self.model = model
        self.reference = np.zeros(2)
        self.tracking = SimpleNamespace(theta_max=.25, theta_slew_max=.0008,
            theta_quantization=1e-6, max_opt_iterations=8, optimizer_step=.0003,
            line_search_shrink=.5, line_search_steps=10, optimizer_min_step=1e-8,
            gradient_norm_tol=1e-7, objective_rel_tol=1e-6, objective_abs_tol=1e-9,
            control_regularization=.0001)

    def next_command(self, measured_PSD_mm, actual_feedback_xy, previous_target_xy=None):
        p, previous = np.asarray(measured_PSD_mm, float), np.asarray(actual_feedback_xy, float)
        prior_command = previous if previous_target_xy is None else np.asarray(previous_target_xy, float)
        reference = np.asarray(self.reference, float)
        if (not np.all(np.isfinite(np.r_[p, previous, prior_command, reference]))
                or not self.model.contains(previous) or not self.model.contains(prior_command)):
            raise CalibrationError('Actual mirror position is outside the measured neural calibration hull')
        baseline, J = self.model.position_and_jacobian(previous)
        if np.linalg.cond(J) > 10:
            raise CalibrationError('The neural position sensitivity is ill-conditioned')
        def objective(command):
            if not self.model.contains(command):
                return -1e6, np.zeros(2)
            predicted, jacobian = self.model.position_and_jacobian(command)
            error = p+predicted-baseline-reference
            return -.5*float(error@error), -jacobian.T@error
        initial = project_pat_command(previous+.8*np.linalg.solve(J, reference-p),
                                      prior_command, self.tracking)
        if not self.model.contains(initial):
            initial = previous
        command, value, diag = projected_gradient_ascent(prior_command, objective,
            lambda v: objective(v)[0], self.tracking, initial)
        if not self.model.contains(command) or not np.isfinite(value) or value < -1e5:
            raise CalibrationError('Proposed neural command lies outside the actual calibration hull')
        predicted = p+self.model.position_and_jacobian(command)[0]-baseline
        if np.linalg.norm(reference-p) > .04 and np.linalg.norm(command-previous) < 2e-6:
            raise CalibrationError('The requested PSD correction is not reachable in the measured region')
        return command, {'predicted_PSD_mm': predicted.tolist(),
            'reference_PSD_mm': reference.tolist(), 'PSD_error_mm': (reference-p).tolist(),
            'command_change_from_previous_target_xy': (command-prior_command).tolist(),
            'optimizer_iterations': diag['optimizer_iterations'],
            'optimizer_stop_reason': diag['optimizer_stop_reason']}
