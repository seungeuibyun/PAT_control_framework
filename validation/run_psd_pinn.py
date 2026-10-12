"""Commission or hold the real PSD-centered FSM loop with the local ray PINN.

PD alignment is deliberately not an input or interlock for this position loop.
Fresh PSD and actual FSM feedback, bounded measured support and GUI Stop are
required throughout. The device's internal mirror position servo remains on.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from psd_position_pinn import CalibrationError, PSDPositionPINN, PSDPositionController
from reference_trajectory import github_periodic_reference

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'calibration'))
from xy_motion import XYMotion
from laser_termite import TermiteLaser
from find_pd_maximum import SearchCancelled
from fsm_reacquisition import FSMReacquisition


def validate_state(state, model):
    p, f = state['psd'], state['fsm']
    for name, part in (('PSD', p), ('FSM', f)):
        age = part.get('age_s')
        if not part.get('connected') or age is None or age > .5:
            raise CalibrationError(name+' readout is unavailable or stale')
    if (not p.get('position_available') or not 2 <= p.get('relative_percent', -1) <= 90
            or p.get('edge_margin_mm', 0) < .5):
        raise CalibrationError('PSD position is absent, overloaded or outside the usable interior')
    if f.get('fault') or f.get('operation_mode') != 2 or f.get('device_status', -1) & ~16:
        raise CalibrationError('FSM mode or device status differs from calibration')
    if max(map(abs, f.get('actual_current_mA', [float('inf')]*2))) > 90:
        raise CalibrationError('Actual mirror coil current exceeds its verified limit')
    q = f.get('feedback_xy')
    target = f.get('target_xy')
    if target is None or not all(math.isfinite(v) and abs(v) <= .25 for v in target):
        raise CalibrationError('Actual normalized XY setpoint is invalid')
    if not model.contains(q) or not model.contains(target):
        raise CalibrationError('Mirror pose is outside the actual measured neural calibration hull')
    return np.array([p['x_mm'], p['y_mm']]), np.array(q)


def run(args):
    folder = ROOT/'calibration'
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    output = folder/('PSD_PINN_'+timestamp+'.json')
    history = output.with_suffix('.jsonl')
    report = {'started': dt.datetime.now().astimezone().isoformat(), 'state': 'preflight',
        'armed': False, 'loop_active': False, 'sample_count': 0, 'samples': [],
        'settled_reference_samples': [], 'requested_seconds': args.seconds,
        'trajectory': args.trajectory,
        'repeat_trajectory': args.repeat_trajectory, 'completed_trajectory_batches': 0,
        'settled_reference_count': 0,
        'model_path': str(Path(args.model).resolve()), 'history_path': str(history),
        'position_controller': 'local_ray_geometry_Frozen_PINN',
        'neural_prediction_and_derivative_used_for_commands': True,
        'external_PAT_PID_or_KF_used': False, 'internal_mirror_position_servo': True,
        'source_stage_position_used': False, 'PD_used_to_control_position': False,
        'physical_PD_PSD_center_coincidence_verified': False,
        'laser_or_stage_set_commands_issued': False,
        'reacquisition_test_requested':args.reacquisition_test,'PAT_phase':'preflight'}
    motion, published_active = XYMotion(), False
    report.update(search_id=motion.search_id,search_client=motion.client)
    last_saved = 0.
    def save():
        nonlocal last_saved
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        (folder/'PSD_PINN_latest.json').write_text(
            json.dumps({'report': str(output), 'state': report['state'],
                'loop_active': report['loop_active'], 'sample_count': report['sample_count'],
                'latest': report['samples'][-1] if report['samples'] else None},
                indent=2, allow_nan=False)+'\n', encoding='utf-8')
        last_saved = time.monotonic()
    def status(active):
        sample=report['samples'][-1] if report['samples'] else None
        motion.broker.post('/internal/status', {'electrical_dark': True,
            'fsm_mapping': True, 'pd_reference': False,
            'PINN_active': active, 'PINN_deferred_by_user': False,
            'PSD_feedback_loop_active': active, 'PSD_feedback_controller': report['position_controller'],
            'PSD_trajectory': args.trajectory if active else 'hold',
            'PSD_PINN_artifact': output.name,'PSD_PINN_model_artifact':Path(args.model).name,
            'PSD_PINN_state':report['state'],'PSD_PINN_sample_count':report['sample_count'],
            'PAT_phase':report['PAT_phase'],
            'PSD_PINN_updated_time':time.time(),
            'PSD_PINN_reference_mm':sample['reference_PSD_mm'] if sample else [0.,0.],
            'PSD_PINN_error_norm_mm':sample['PSD_error_norm_mm'] if sample else None,
            'PD_acquisition_threshold_V':report.get('reacquisition',{}).get('PD_detection_threshold_V'),
            'acquisition_PD_seen_count':report.get('reacquisition',{}).get('PD_seen_count',0),
            'activity':'PINN PSD/FSM closed loop running' if active else
                'Receiver FSM acquisition: '+report['PAT_phase'] if report['PAT_phase']!='stopped' else
                'PINN trial ended; FSM holds its last pose'})
        motion.broker.post('/internal/status', {'PSD_reference_path_mm':
            np.round(report.get('PSD_reference_path_mm', []), 5).tolist() if active else []})
    try:
        model = PSDPositionPINN.load(args.model)
        if model.record.get('simulation'):
            raise CalibrationError('A simulated neural model cannot drive physical devices')
        checks = json.loads((folder/'psd_position_pinn_numerical_verification.json').read_text())
        if checks.get('state') != 'passed':
            raise CalibrationError('Numerical neural sensitivity and control checks have not passed')
        controller = PSDPositionController(model)
        source = TermiteLaser().initial
        if source['current_mA'] != model.record.get('source_current_mA'):
            raise CalibrationError('Source setting differs from the measured calibration')
        report['initial_state']=motion.state()
        validate_state(report['initial_state'], model)
        motion.start()
        report.update(armed=True, loop_active=False, state='physical_PSD_PINN_loop_starting',
                      source=source, neural_withheld_rms_mm=model.record['withheld_PSD_rms_mm'])
        motion.broker.post('/internal/status',dict(raw_trajectory_active=False,
            raw_trajectory_epoch=report['started'],raw_trajectory_origin_xy=model.center.tolist(),
            raw_trajectory_amplitude_xy=[.035,.035] if args.reacquisition_test else [model.qscale,model.qscale],
            raw_trajectory_shape='hold',raw_trajectory_statistics=None,
            raw_trajectory_started_time=None,PSD_reference_path_mm=[]))
        if args.reacquisition_test:
            anchor=list(motion.state()['fsm']['target_xy'])
            conversion=motion.state()['calibration'].get('PD_voltage_conversion',{})
            if not math.isfinite(conversion.get('dark_voltage_V',float('nan'))):
                raise CalibrationError('Verified electrical PD dark baseline is required for acquisition')
            def acquisition_progress(value):
                report['reacquisition']=value;report['PAT_phase']=value['state']
                report['state']='physical_receiver_acquisition_running'
                save();status(False)
            acquisition=FSMReacquisition(motion,model,conversion['dark_voltage_V'],acquisition_progress)
            try:
                report['reacquisition']=acquisition.run(anchor)
            except SearchCancelled:
                raise
            except Exception:
                # A failed scan returns to its actual prior receiver pose; a
                # monitor Stop cancels the lease and prevents this return.
                motion.command(anchor)
                raise
            if TermiteLaser().initial['current_mA']!=model.record['source_current_mA']:
                raise CalibrationError('Source current changed during receiver acquisition')
        report.update(loop_active=True,state='physical_PSD_PINN_loop_running',PAT_phase='tracking')
        save()
        targets = []
        if args.trajectory == 'github-periodic':
            points, provenance = github_periodic_reference(args.github_amplitude_mm)
            report['trajectory_provenance'] = provenance
            report['PSD_reference_path_mm'] = points.tolist()
            for _ in range(args.circle_cycles):
                targets.extend(points)
        elif args.trajectory == 'figure-eight':
            for cycle in range(args.circle_cycles):
                for angle in np.linspace(0, 2*np.pi, 32, endpoint=False):
                    targets.append(np.array([args.figure_eight_width_mm*np.sin(angle),
                        args.figure_eight_height_mm*np.sin(2*angle)]))
            targets.append(np.zeros(2))
        elif args.circle_radius_mm:
            for cycle in range(args.circle_cycles):
                for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                    targets.append(args.circle_radius_mm*np.array([np.cos(angle), np.sin(angle)]))
            targets.append(np.zeros(2))
        status(True); published_active = True; save()
        index, target_iterations = 0, 0
        started = last_source_check = last_print = time.monotonic()
        previous_error, worsening = None, 0
        squared_error_sum, maximum_settled_error = 0., 0.
        while args.seconds == 0 or time.monotonic()-started < args.seconds:
            state = motion.state()
            position, feedback = validate_state(state, model)
            reference = targets[index] if index < len(targets) else np.zeros(2)
            controller.reference = reference
            error = float(np.linalg.norm(reference-position))
            worsening = worsening+1 if previous_error is not None and error > previous_error+.04 else 0
            if worsening >= 3:
                raise CalibrationError('Actual PSD error increased through three successive corrections')
            target_iterations += 1
            if index < len(targets) and target_iterations > 30:
                raise CalibrationError('The actual neural position loop failed to reach its test reference')
            target = np.array(state['fsm']['target_xy'])
            command, prediction = controller.next_command(position, feedback, target)
            # Avoid chasing readout noise inside the verified center deadband.
            if error <= .015:
                command = target
                prediction['command_held_within_deadband'] = True
                prediction['predicted_PSD_mm'] = (position+
                    model.position_and_jacobian(target)[0]-
                    model.position_and_jacobian(feedback)[0]).tolist()
            row = {'time': dt.datetime.now().astimezone().isoformat(),
                'elapsed_s': time.monotonic()-started, 'PSD_mm': position.tolist(),
                'PSD_relative_percent': state['psd']['relative_percent'],
                'FSM_feedback_xy': feedback.tolist(), 'previous_target_xy': target.tolist(),
                'next_xy': command.tolist(), 'FSM_actual_current_mA': state['fsm']['actual_current_mA'],
                'PSD_error_norm_mm': error,
                'PD_V_observation_only': state.get('pd', {}).get('mean_voltage_V'), **prediction}
            report['samples'].append(row);report['sample_count'] += 1
            if len(report['samples']) > 2048:
                report['samples'] = report['samples'][-2048:]
            with history.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            # Refresh the automatic lease even while holding an unchanged XY target.
            motion.command(command.tolist())
            if time.monotonic()-last_print > 3:
                print(json.dumps({'samples': report['sample_count'], 'reference_mm': reference.tolist(),
                    'PSD_mm': position.tolist(), 'error_mm': error,
                    'PD_V_observation_only': row['PD_V_observation_only']}), flush=True)
                last_print = time.monotonic()
            if time.monotonic()-last_saved > 2:
                save();status(True)
            previous_error = error
            if index < len(targets) and error <= .025:
                report['settled_reference_samples'].append(row)
                report['settled_reference_samples'] = report['settled_reference_samples'][-256:]
                report['settled_reference_count'] += 1
                squared_error_sum += error**2
                maximum_settled_error = max(maximum_settled_error, error)
                report['reference_tracking_RMS_mm'] = float(np.sqrt(
                    squared_error_sum/report['settled_reference_count']))
                report['maximum_settled_error_mm'] = maximum_settled_error
                index += 1;target_iterations = 0;previous_error = None;worsening = 0
                if index == len(targets):
                    report['completed_trajectory_batches'] += 1
                    report['physical_trajectory_tracking_verified'] = True
                    if args.repeat_trajectory:
                        index = 0
            if time.monotonic()-last_source_check > 30:
                if TermiteLaser().initial['current_mA'] != model.record['source_current_mA']:
                    raise CalibrationError('Source current changed during PSD position control')
                last_source_check = time.monotonic()
            time.sleep(.1)
        if index < len(targets) and not args.repeat_trajectory:
            raise CalibrationError('Trial duration ended before all physical test references were verified')
        settled = report['settled_reference_samples']
        errors = [r['PSD_error_norm_mm'] for r in settled]
        report.update(state='physical_PSD_PINN_trial_completed',
            reference_tracking_RMS_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            maximum_settled_error_mm=max(errors) if errors else None,
            final_center_error_mm=report['samples'][-1]['PSD_error_norm_mm'])
    except SearchCancelled as exc:
        report.update(state='cancelled', error=str(exc))
    except (Exception, KeyboardInterrupt) as exc:
        report.update(state='stopped_by_interlock' if report['armed'] else 'not_armed', error=str(exc))
    finally:
        report['loop_active'] = False
        report['PAT_phase']='stopped'
        if motion.active:
            # Never automatically resume a lease cancelled by the monitor's Stop.
            try:
                report['handoff'] = motion.close()
            except Exception as exc:
                report['handoff_error'] = str(exc)
        try:
            report['final_state'] = motion.broker.state()
            if report['armed']:
                status(False)
        except Exception as exc:
            report['status_error'] = str(exc)
        report['finished'] = dt.datetime.now().astimezone().isoformat()
        save()
        print(json.dumps({k: report.get(k) for k in ('state', 'sample_count', 'error',
            'reference_tracking_RMS_mm', 'maximum_settled_error_mm', 'final_center_error_mm')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(ROOT/'calibration'/'psd_position_pinn_model.json'))
    parser.add_argument('--seconds', type=float, default=30., help='0 continues until monitor Stop or a guard')
    parser.add_argument('--trajectory', choices=('hold', 'circle', 'figure-eight', 'github-periodic'), default='hold')
    parser.add_argument('--circle-radius-mm', type=float, default=0.)
    parser.add_argument('--circle-cycles', type=int, default=1)
    parser.add_argument('--figure-eight-width-mm', type=float, default=.6,
                        help='Horizontal half-width of the figure eight on the actual PSD')
    parser.add_argument('--figure-eight-height-mm', type=float, default=.28,
                        help='Vertical half-height of the figure eight on the actual PSD')
    parser.add_argument('--repeat-trajectory', action='store_true',
                        help='Repeat the measured reference pattern until duration or monitor Stop')
    parser.add_argument('--reacquisition-test',action='store_true',
                        help='Mispoint once, verify actual PD/PSD loss, scan, then hand the real captured beam to PINN')
    parser.add_argument('--github-amplitude-mm', type=float, default=.35,
                        help='Common x/y scale for the upstream periodic angular law')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 <= args.seconds <= 3600:
        parser.error('Duration must be finite and between 0 and 3600 seconds')
    if not math.isfinite(args.circle_radius_mm) or not 0 <= args.circle_radius_mm <= .3:
        parser.error('Circle radius must be between 0 and 0.3 mm in the verified center region')
    if not 1 <= args.circle_cycles <= 3:
        parser.error('Use 1 to 3 verification circles')
    if (not math.isfinite(args.figure_eight_width_mm) or not 0 < args.figure_eight_width_mm <= .6
            or not math.isfinite(args.figure_eight_height_mm) or not 0 < args.figure_eight_height_mm <= .28):
        parser.error('Figure eight must stay inside the measured region: half-width <= 0.6 mm, half-height <= 0.28 mm')
    if not math.isfinite(args.github_amplitude_mm) or not 0 < args.github_amplitude_mm <= .35:
        parser.error('Upstream periodic trajectory amplitude must be in (0, 0.35] mm on both PSD axes')
    run(args)
