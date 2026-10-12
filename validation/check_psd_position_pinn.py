"""Numerical verification of neural sensitivities, limits and receiver-only tracking."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from psd_position_pinn import (CalibrationError, PSDPositionPINN,
                               PSDPositionController, train_psd_position_pinn)


def main():
    axis = np.linspace(-.003, .003, 5)
    q0 = np.array([-.01, .053])
    G = np.array([[245., 11.], [-7., -162.]])
    rows = []
    for j, y in enumerate(axis):
        for x in axis if j % 2 == 0 else axis[::-1]:
            q = q0+[x, y]
            rows.append({'FSM_feedback_xy': q.tolist(), 'PSD_mm': (G@(q-q0)).tolist(),
                         'PSD_percent': 20})
    fixture = {'state': 'real_PSD_FSM_closed_loop_verified', 'simulation': True,
        'source': {'current_mA': 32}, 'center_calibration_samples': rows}
    model = train_psd_position_pinn(fixture)
    derivative_error = []
    for offset in np.random.default_rng(34).uniform(-.0018, .0018, (8, 2)):
        q = q0+offset
        _, gradient = model.position_and_jacobian(q)
        step = 1e-7
        finite_difference = np.column_stack([
            (model.position_and_jacobian(q+np.eye(2)[i]*step)[0]
             -model.position_and_jacobian(q-np.eye(2)[i]*step)[0])/(2*step)
            for i in range(2)])
        error = np.max(abs(gradient-finite_difference))
        assert error < 1e-4, error
        derivative_error.append(float(error))
    controller = PSDPositionController(model)
    q, p = q0.copy(), np.array([.38, -.25])
    initial_error = float(np.linalg.norm(p))
    for i in range(20):
        previous = q.copy()
        command, _ = controller.next_command(p, q)
        assert np.max(abs(command-previous)) <= .0008001
        assert np.max(abs(command)) <= .25 and model.contains(command)
        p += G@(command-q)
        q = command
        if i == 9:
            # Unannounced incident-beam displacement; controller never sees it.
            p += [.13, .09]
    final_error = float(np.linalg.norm(p))
    assert final_error < .002, final_error
    try:
        controller.next_command([0, 0], q0+[.02, 0])
    except CalibrationError:
        pass
    else:
        raise AssertionError('Unmeasured mirror commands were accepted')
    # Serialization must preserve the neural outputs and gradients exactly.
    folder = Path(__file__).resolve().parent/'calibration'
    fixture_path = folder/'psd_position_pinn_numerical_fixture.json'
    model.save(fixture_path)
    restored = PSDPositionPINN.load(fixture_path)
    for q in (q0, q0+[.001, -.001]):
        for left, right in zip(model.position_and_jacobian(q), restored.position_and_jacobian(q)):
            assert np.array_equal(left, right)
    result = {'state': 'passed', 'simulation': True,
        'withheld_PSD_rms_mm': model.record['withheld_PSD_rms_mm'],
        'maximum_gradient_finite_difference_error': max(derivative_error),
        'initial_error_mm': initial_error, 'final_error_after_unobserved_disturbance_mm': final_error,
        'source_stage_position_supplied_to_controller': False,
        'command_slew_and_measured_hull_verified': True, 'serialization_verified': True}
    (folder/'psd_position_pinn_numerical_verification.json').write_text(
        json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
