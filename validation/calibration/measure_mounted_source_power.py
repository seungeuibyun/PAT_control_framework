"""Measure fresh powered-detector dark readings and a requested 32->34 mA trial.

The fixed source drive and TEC protections are preserved. PSD offsets may be
updated only from stable measurements with verified zero source current.
No FSM or Zaber commands are issued. Source restoration is verified on error.
"""
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

import numpy as np

from laser_termite import TermiteLaser
from shared_acquisition import Broker


def main():
    folder = Path(__file__).resolve().parent
    path = folder / ('mounted_power_comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    broker = Broker(folder / 'live_gui_connection.json')
    laser = TermiteLaser(authorised_alignment_ceiling_mA=34)
    original_current = laser.initial['current_mA']
    original_offsets = broker.post('/internal/psd', {'command': '1OF?'})['reply'][3:]
    report = {'started': datetime.now().astimezone().isoformat(), 'state': 'measuring',
              'original_source': laser.initial, 'user_requested_small_power_increase': True,
              'FSM_commands_issued': False, 'Zaber_commands_issued': False,
              'dark_condition': 'Verified zero laser drive; powered PD and PSD',
              'original_PSD_offsets': original_offsets, 'points': []}
    source_held, offsets_verified, offsets_changed = False, False, False

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def readings(label, count=24):
        rows = []
        for i in range(count):
            raw = broker.post('/internal/psd', {'command': '1RA?'})['reply']
            gp = broker.post('/internal/psd', {'command': '1GP'})['reply']
            pd = broker.post('/internal/pd', {})['voltage_V']
            rows.append({'time_s': time.time(), 'RA_reply': raw, 'GP_reply': gp,
                         'RA_V': [float(v) for v in raw[3:].split(',')],
                         'GP': [float(v) for v in gp[3:].split(',')], 'PD_V': pd})
            time.sleep(.045)
        electrodes = np.array([r['RA_V'] for r in rows])
        positions = np.array([r['GP'] for r in rows])
        result = {'label': label, 'rows': rows,
                  'PD_mean_V': statistics.mean(r['PD_V'] for r in rows),
                  'PD_std_V': statistics.stdev(r['PD_V'] for r in rows),
                  'electrode_mean_V': electrodes.mean(0).tolist(),
                  'electrode_std_V': electrodes.std(0).tolist(),
                  'electrode_half_run_drift_V': (electrodes[count//2:].mean(0)-electrodes[:count//2].mean(0)).tolist(),
                  'PSD_power_percent_median': float(np.median(positions[:, 2])),
                  'PSD_raw_position_mean_mm': positions[:, :2].mean(0).tolist()}
        report['points'].append(result)
        save()
        print(json.dumps({k: result[k] for k in ['label', 'PD_mean_V',
              'PD_std_V', 'PSD_power_percent_median', 'PSD_raw_position_mean_mm']}), flush=True)
        return result

    try:
        if original_current != 32:
            raise RuntimeError('Initial source must be the freshly verified 32 mA setting')
        broker.post('/internal/status', {'activity': 'PD 전원 켠 상태: 암전 기준 및 레이저 소폭 증강 비교',
                                         'PINN_active': False, 'PSD_feedback_loop_active': False})
        report['source_off'] = laser.set_current(0)
        time.sleep(.7)
        dark = readings('source_0mA_powered_detector_dark')
        means, std, drift = [np.array(dark[k]) for k in
                           ('electrode_mean_V', 'electrode_std_V', 'electrode_half_run_drift_V')]
        if (max(abs(means)) > .3 or max(std) > .008 or max(abs(drift)) > .012
                or dark['PD_std_V'] > .001 or not 0 <= dark['PD_mean_V'] < .1):
            raise RuntimeError('Fresh dark readings are not stable electrical offsets')
        if dark['PSD_power_percent_median'] > 2:
            raise RuntimeError('Unexpected large PSD signal with zero source current')
        command = '1OF' + ','.join(f'{v:.6f}' for v in means)
        offsets_changed = True
        broker.post('/internal/psd', {'command': command})
        readback = broker.post('/internal/psd', {'command': '1OF?'})['reply']
        report['PSD_offset_readback'] = readback
        if max(abs(np.array([float(v) for v in readback[3:].split(',')])-means)) > .0062:
            raise RuntimeError('Fresh electrical offset readback failed')
        corrected = readings('source_0mA_corrected_dark', 12)
        if corrected['PSD_power_percent_median'] > 1:
            raise RuntimeError('Corrected dark PSD signal is not near zero')
        offsets_verified = True
        report['source_32mA'] = laser.set_current(32)
        time.sleep(.7)
        low = readings('source_32mA')
        report['source_34mA'] = laser.set_current(34)
        if report['source_34mA']['after']['power_dBm'] is None or report['source_34mA']['after']['power_dBm'] > 5.0:
            raise RuntimeError('Increased source power exceeds the small diagnostic trial')
        time.sleep(.7)
        high = readings('source_34mA')
        if high['PSD_power_percent_median'] >= 98 or high['PD_mean_V'] >= 8:
            raise RuntimeError('Increased power approaches detector full scale')
        report['PD_dark_mean_V'] = dark['PD_mean_V']
        report['PD_dark_std_V'] = dark['PD_std_V']
        report['PD_net_32mA_V'] = low['PD_mean_V'] - dark['PD_mean_V']
        report['PD_net_34mA_V'] = high['PD_mean_V'] - dark['PD_mean_V']
        report['PD_source_power_increase_response_V'] = high['PD_mean_V'] - low['PD_mean_V']
        conversion_path = folder / 'PD_readout_conversion.json'
        conversion = json.loads(conversion_path.read_text(encoding='utf-8'))
        conversion.update(dark_voltage_V=dark['PD_mean_V'], dark_artifact=path.name,
                          updated=datetime.now().astimezone().isoformat(),
                          absolute_optical_power_calibrated=False)
        conversion_path.write_text(json.dumps(conversion, indent=2) + '\n', encoding='utf-8')
        display_keys = ['model', 'wavelength_nm', 'responsivity_A_per_W',
                        'gain_high_z_V_per_A', 'gain_50ohm_V_per_A', 'default_load',
                        'physical_load_verified', 'dark_voltage_V', 'dark_artifact',
                        'absolute_optical_power_calibrated', 'power_display_kind']
        broker.post('/internal/status', {'PD_voltage_conversion': {k:conversion[k] for k in display_keys},
            'electrical_dark': True, 'PD_dark_measured': True,
            'activity': '레이저 34 mA 비교 완료 · 새 광학 경로 정렬 진행'})
        source_held = True
        report['state'] = 'powered_detector_dark_verified_34mA_trial_measured'
        report['held_source_current_mA'] = 34
    except BaseException as exc:
        report.update(state='aborted', error=repr(exc))
    finally:
        if offsets_changed and not offsets_verified:
            try:
                broker.post('/internal/psd', {'command': '1OF' + original_offsets})
            except Exception as exc:
                report['offset_restore_error'] = str(exc)
        if not source_held:
            try:
                report['source_restored'] = laser.set_current(original_current)
            except Exception as exc:
                report['source_restore_error'] = str(exc)
        report['finished'] = datetime.now().astimezone().isoformat()
        save()
        print(json.dumps({'state': report['state'], 'error': report.get('error'),
                          'source_held_mA': report.get('held_source_current_mA', original_current),
                          'file': str(path)}), flush=True)


if __name__ == '__main__':
    main()
