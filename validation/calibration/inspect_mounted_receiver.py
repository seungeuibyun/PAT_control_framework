"""Bounded physical acquisition after mounting both GBE10-C assemblies.

Move the verified, referenced source stage only +/-0.5 mm, return it, then
search with measured FSM feedback. No source emission commands are sent.
Raw PSD replies and actual mirror/stage feedback are retained independently.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import serial

from shared_acquisition import Broker
from xy_motion import XYMotion


FOLDER = Path(__file__).resolve().parent
MICROSTEP_MM = 0.00009921875


def main(stage_radius_mm=.5, stage_only=False, positive_only=False, fsm_only=False, raster=False):
    if not .1 <= stage_radius_mm <= 5.0:
        raise ValueError('Diagnostic stage radius must be 0.1 to 5 mm')
    broker = Broker(FOLDER / 'live_gui_connection.json')
    motion = XYMotion()
    stage, initial_stage, held = None, None, False
    held_stage_offset = None
    initial = broker.state()
    original_current = initial['fsm']['target_current_mA']
    report = {'started': datetime.now().astimezone().isoformat(),
              'state': 'measuring', 'source_commands_issued': False,
              'stage_points': [], 'FSM_points': [], 'PINN_used': False,
              'PSD_coordinates': 'Unmodified native 1GP, mm',
              'stage_scale_mm_per_native_unit': MICROSTEP_MM,
              'new_geometry_calibrated': False, 'stage_radius_mm': stage_radius_mm,
              'stage_only': stage_only, 'positive_only': positive_only,
              'fsm_only': fsm_only, 'raster': raster}
    output = FOLDER / ('mounted_receiver_acquisition_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def stage_query(command):
        stage.reset_input_buffer()
        stage.write((command + '\r\n').encode('ascii'))
        stage.flush()
        reply = stage.read_until(b'\n').decode('ascii').strip()
        parts = reply.split()
        if len(parts) < 6 or parts[:3] != ['@01', '0', 'OK']:
            raise RuntimeError('Unexpected Zaber reply: ' + reply)
        if parts[4].startswith(('F', 'W')):
            raise RuntimeError('Zaber fault/reference warning: ' + reply)
        return parts, reply

    def stage_move(offset_mm):
        if abs(offset_mm) > stage_radius_mm + 1e-6:
            raise ValueError('Stage probe exceeds its local window')
        target = initial_stage + round(offset_mm / MICROSTEP_MM)
        current = int(stage_query('/1 get pos')[0][-1])
        delta = target - current
        if abs(delta * MICROSTEP_MM) > 2 * stage_radius_mm + .01:
            raise RuntimeError('Stage changed outside this local probe')
        if delta:
            # 1 mm/s, ~4.85 mm/s^2; per-move limits preserve saved settings.
            report.setdefault('stage_commands', []).append(stage_query(f'/1 move rel {delta} 16513 8')[1])
        deadline = time.monotonic() + abs(delta * MICROSTEP_MM) + 4
        while time.monotonic() < deadline:
            state = broker.state()
            if not state['fsm'].get('connected') or state['fsm'].get('fault'):
                stage_query('/1 stop')
                raise RuntimeError('Receiver feedback unavailable during stage motion')
            parts, _ = stage_query('/1 get pos')
            if parts[3] == 'IDLE':
                if abs(int(parts[-1]) - target) > 2:
                    raise RuntimeError('Stage failed position readback')
                time.sleep(.15)
                return int(parts[-1])
            time.sleep(.08)
        stage_query('/1 stop')
        raise RuntimeError('Stage probe did not finish')

    def sample(label):
        rows = []
        for _ in range(7):
            reply = broker.post('/internal/psd', {'command': '1GP'})['reply']
            values = [float(v) for v in reply[3:].split(',')]
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise RuntimeError('Invalid native PSD reply')
            rows.append({'time_s': time.time(), 'reply': reply, 'values': values})
            time.sleep(.035)
        pd = []
        while len(pd) < 3:
            try:
                value = broker.post('/internal/pd', {})['voltage_V']
                scale = broker.state()['pd']['scale_V_per_div']
                if abs(value) > 3 * scale:
                    raise RuntimeError('PD range insufficient')
                pd.append(value)
                time.sleep(.05)
            except RuntimeError as exc:
                if 'unavailable' not in str(exc) and 'range insufficient' not in str(exc):
                    raise
                scale = broker.state()['pd']['scale_V_per_div']
                larger = next((s for s in [.02, .05, .1, .2, .5, 1, 2, 5] if s > scale * 1.01), None)
                if larger is None:
                    raise
                broker.post('/internal/scope', {'command': f':CHANnel1:SCALe {larger}'})
        if max(pd) >= 8.0:
            raise RuntimeError('PD output approaches amplifier saturation')
        data = np.array([r['values'] for r in rows])
        state = broker.state()
        point = {'label': label, 'time': datetime.now().astimezone().isoformat(),
                 'PSD_mm': data[:, :2].mean(0).tolist(),
                 'PSD_std_mm': data[:, :2].std(0).tolist(),
                 'PSD_relative_percent': float(np.median(data[:, 2])),
                 'PSD_raw_readings': rows, 'PD_samples_V': pd,
                 'PD_mean_V': statistics.mean(pd),
                 'FSM_feedback_xy': state['fsm']['feedback_xy'],
                 'FSM_target_xy': state['fsm'].get('target_xy'),
                 'actual_coil_current_mA': state['fsm']['actual_current_mA'],
                 'scope_scale_V_per_div': state['pd']['scale_V_per_div'],
                 'usable_spot': bool(np.min(data[:, 2]) >= 1 and np.max(data[:, 2]) < 98
                                     and np.max(np.abs(data[:, :2])) < 4.65
                                     and np.max(data[:, :2].std(0)) < .08)}
        print(json.dumps({k: point[k] for k in ['label', 'PSD_mm', 'PSD_relative_percent',
                                               'PD_mean_V', 'usable_spot']}), flush=True)
        return point

    try:
        broker.post('/internal/status', {'activity': '새 GBE 구성: Zaber·FSM 미세 정렬 측정',
            'PINN_active': False, 'PSD_feedback_loop_active': False,
            'raw_trajectory_active': False, 'pd_reference': False, 'fsm_mapping': False})
        if not fsm_only:
            stage = serial.Serial('COM6', 115200, timeout=.7, write_timeout=.7)
            if int(stage_query('/1 get deviceid')[0][-1]) != 50946:
                raise RuntimeError('COM6 is not the verified X-LSQ450A-E01C')
            if int(stage_query('/1 get resolution')[0][-1]) != 64:
                raise RuntimeError('Microstep resolution differs from the measured unit conversion')
            initial_stage = int(stage_query('/1 get pos')[0][-1])
            low = int(stage_query('/1 get limit.min')[0][-1])
            high = int(stage_query('/1 get limit.max')[0][-1])
            margin = math.ceil(stage_radius_mm / MICROSTEP_MM) + 100
            if not low + margin < initial_stage < high - margin:
                raise RuntimeError('Insufficient stage margin for the local probe')
            report['initial_stage_native'] = initial_stage
            report['initial_stage_nominal_mm'] = initial_stage * MICROSTEP_MM
        report['initial_FSM_state'] = initial['fsm']
        save()
        stage_offsets = [0., stage_radius_mm, -stage_radius_mm, 0.]
        if stage_radius_mm > .5:
            stage_offsets = [0., .5, 1., stage_radius_mm, 0., -.5, -1., -stage_radius_mm, 0.]
        if positive_only:
            stage_offsets = sorted(set([0., min(2., stage_radius_mm),
                                        min(3., stage_radius_mm), stage_radius_mm])) + [0.]
        if fsm_only:
            stage_offsets = []
        for offset in stage_offsets:
            native = stage_move(offset)
            point = sample(f'Zaber_{offset:+g}mm')
            point.update(stage_position_native=native,
                         stage_offset_mm=(native-initial_stage)*MICROSTEP_MM)
            report['stage_points'].append(point)
            save()
        if stage_only:
            baseline = report['stage_points'][0]
            candidate = max(report['stage_points'], key=lambda p: p['PD_mean_V'])
            uncertainty = math.sqrt(statistics.variance(candidate['PD_samples_V']) +
                                    statistics.variance(baseline['PD_samples_V']))
            margin_V = max(.0005, 5 * uncertainty)
            if candidate['PD_mean_V'] - baseline['PD_mean_V'] > margin_V:
                stage_move(candidate['stage_offset_mm'])
                verify = sample('verify_best_stage_coupling')
                if verify['PD_mean_V'] - baseline['PD_mean_V'] > margin_V:
                    held_stage_offset = candidate['stage_offset_mm']
                    report['held_stage_offset_mm'] = held_stage_offset
                    report['held_stage_verify'] = verify
                    report['state'] = 'improved_stage_coupling_held'
            if held_stage_offset is None:
                report['state'] = 'no_significant_PD_improvement_in_stage_probe'
            return
        motion.start()
        report['FSM_origin_xy'] = motion.origin
        broker.post('/internal/status', {'FSM_trajectory_origin_xy': motion.origin,
                                         'activity': 'FSM 소범위 탐색: PSD 유효 빔 위치 확인'})
        origin = np.array(motion.origin)
        offsets = [(0., 0.)] + [(0., y) for y in [.004, .01, .018, .028, .04, .055]]
        offsets += [(0., 0.)] + [(0., y) for y in [-.008, -.02, -.035, -.05]]
        offsets += [(0., 0.), (.012, 0.), (.025, 0.), (.04, 0.), (0., 0.),
                    (-.012, 0.), (-.025, 0.), (-.04, 0.), (0., 0.)]
        if raster:
            offsets = [(0., 0.)]
            for i, y in enumerate([-.04, -.02, 0., .02, .04]):
                xs = [-.04, -.02, 0., .02, .04]
                if i % 2:
                    xs.reverse()
                offsets.extend((x, y) for x in xs if (x, y) != (0., 0.))
            offsets += [(0., 0.), (.065, 0.), (.045, .045), (0., .065),
                        (-.045, .045), (-.065, 0.), (-.045, -.045),
                        (0., -.065), (.045, -.045), (0., 0.)]
        for dx, dy in offsets:
            target = origin + [dx, dy]
            current = np.array(motion.state()['fsm']['target_xy'])
            count = max(1, math.ceil(np.linalg.norm(target-current)/.035))
            for i in range(1, count+1):
                motion.command((current + (target-current)*i/count).tolist())
            point = sample(f'FSM_delta_{dx:+g}_{dy:+g}')
            point['command_offset_xy'] = [dx, dy]
            report['FSM_points'].append(point)
            save()
            if point['usable_spot']:
                held = True
                report['state'] = 'usable_PSD_spot_acquired'
                report['held_point'] = point
                break
        if not held:
            report['state'] = 'no_usable_spot_in_bounded_probe'
    except BaseException as exc:
        report.update(state='aborted', error=repr(exc))
    finally:
        if stage is not None:
            try:
                stage_query('/1 stop')
                if initial_stage is not None:
                    stage_move(held_stage_offset if held_stage_offset is not None else 0.)
            except Exception as exc:
                report['stage_restore_error'] = str(exc)
            stage.close()
        if motion.active:
            try:
                if not held:
                    motion.command(motion.origin)
                    motion._post('current_mode')
                    motion._post('set', x_mA=original_current[0], y_mA=original_current[1])
                report['handoff'] = motion.close()
            except Exception as exc:
                report['FSM_restore_error'] = str(exc)
                try:
                    motion.broker.post('/internal/fsm', {'action': 'stop', 'client_id': 'mounted-receiver-emergency', 'sequence': 1})
                except Exception as stop_exc:
                    report['stop_error'] = str(stop_exc)
        report['finished'] = datetime.now().astimezone().isoformat()
        report['final_state'] = {k: broker.state()[k] for k in ['psd', 'fsm', 'pd']}
        save()
        print(json.dumps({'state': report['state'], 'error': report.get('error'),
                          'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-radius-mm', type=float, default=.5)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--positive-only', action='store_true')
    parser.add_argument('--fsm-only', action='store_true')
    parser.add_argument('--raster', action='store_true')
    args = parser.parse_args()
    main(args.stage_radius_mm, args.stage_only, args.positive_only, args.fsm_only, args.raster)
