"""Measure electrical dark offsets with verified zero laser drive current.

This is an explicit source-off measurement, not a claim of physical beam
blockage by the user. Restore the measured attenuated source setting.
FSM commands, detector gains and TEC settings are preserved.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import time

import numpy as np

from laser_termite import TermiteLaser
from shared_acquisition import Broker


def main():
    folder = Path(__file__).resolve().parent
    broker = Broker(folder / 'live_gui_connection.json')
    laser = TermiteLaser()
    original_current = laser.initial['current_mA']
    initial = broker.state()
    result = {'started': dt.datetime.now().astimezone().isoformat(),
              'state': 'in_progress', 'dark_condition': 'verified_zero_source_drive_current',
              'user_physical_blockage_claimed': False, 'original_source': laser.initial,
              'initial_readout': initial, 'FSM_commands_issued': False,
              'TEC_commands_issued': False, 'gain_commands_issued': False,
              'raw_samples_V': [], 'PD_dark_samples_V': []}
    path = folder / 'source_off_dark_calibration.json'
    original_offsets = broker.post('/internal/psd', {'command': '1OF?'})['reply'][3:]
    result['original_offsets'] = original_offsets
    changed = False
    scope_range_changed = False

    def save():
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    try:
        if original_current != 32:
            raise RuntimeError('Expected the newly measured 32 mA source setting')
        result['source_off'] = laser.set_current(0)
        # The previous 5 V/div range quantizes small DC offsets too coarsely.
        # Restore a suitable illuminated range before restoring emission.
        broker.post('/internal/scope', {'command': ':CHANnel1:SCALe 0.05'})
        scope_range_changed = True
        time.sleep(.8)
        s = broker.state()
        if s['psd']['relative_percent'] > 1 or abs(s['pd']['mean_voltage_V']) > .06:
            raise RuntimeError('Measured source-off signal is not sufficiently dark')
        for i in range(60):
            raw = broker.post('/internal/psd', {'command': '1RA?'})['reply']
            vals = [float(x) for x in raw[3:].split(',')]
            if len(vals) != 4 or not np.all(np.isfinite(vals)):
                raise RuntimeError('Invalid raw dark electrodes')
            result['raw_samples_V'].append(vals)
            if i % 6 == 0:
                result['PD_dark_samples_V'].append(
                    broker.post('/internal/pd', {})['voltage_V'])
            time.sleep(.025)
        data = np.asarray(result['raw_samples_V'])
        means = data.mean(axis=0)
        stds = data.std(axis=0, ddof=1)
        drift = data[30:].mean(axis=0) - data[:30].mean(axis=0)
        result.update(electrode_dark_mean_V=means.tolist(),
                      electrode_dark_std_V=stds.tolist(),
                      electrode_dark_drift_V=drift.tolist(),
                      PD_dark_mean_V=float(np.mean(result['PD_dark_samples_V'])),
                      PD_dark_std_V=float(np.std(result['PD_dark_samples_V'], ddof=1)))
        if max(abs(means)) > .2 or max(stds) > .008 or max(abs(drift)) > .01:
            raise RuntimeError('Dark electrode readings fail magnitude, stability or drift checks')
        if result['PD_dark_std_V'] > .01:
            raise RuntimeError('Dark PD voltage is unstable')
        command = '1OF' + ','.join(f'{v:.6f}' for v in means)
        result['offset_command'] = command
        changed = True
        broker.post('/internal/psd', {'command': command})
        actual = broker.post('/internal/psd', {'command': '1OF?'})['reply']
        result['offset_readback'] = actual
        # This controller rounds electrode offsets to its observed 12.21 mV
        # ADC voltage increment. Verify within half an increment, then verify
        # the actual corrected dark reading independently.
        result['offset_readback_tolerance_V'] = .0062
        if max(abs(np.array([float(x) for x in actual[3:].split(',')]) - means)) > .0062:
            raise RuntimeError('Electrical offset readback mismatch')
        result['corrected_dark_samples'] = []
        for _ in range(15):
            raw = broker.post('/internal/psd', {'command': '1GP'})['reply']
            result['corrected_dark_samples'].append([float(x) for x in raw[3:].split(',')])
            time.sleep(.04)
        if max(abs(v[2]) for v in result['corrected_dark_samples']) > 1:
            raise RuntimeError('Corrected dark PSD power is not near zero')
        result['state'] = 'source_off_electrical_dark_calibration_verified'
    except Exception as exc:
        result.update(state='aborted', error=str(exc))
        if changed:
            try:
                broker.post('/internal/psd', {'command': '1OF' + original_offsets})
                result['offset_restore_readback'] = broker.post('/internal/psd', {'command': '1OF?'})['reply']
            except Exception as restore:
                result['offset_restore_error'] = str(restore)
    finally:
        if scope_range_changed:
            try:
                result['illuminated_scope_range_V_per_div'] = float(
                    broker.post('/internal/scope', {'command': ':CHANnel1:SCALe 2'})['reply'])
            except Exception as restore:
                result['scope_range_restore_error'] = str(restore)
        try:
            result['source_restored'] = laser.set_current(original_current)
            time.sleep(.7)
            result['restored_readout'] = broker.state()
        except Exception as restore:
            result['source_restore_error'] = str(restore)
        result['finished'] = dt.datetime.now().astimezone().isoformat()
        save()
        print(json.dumps({'state': result['state'], 'error': result.get('error'),
            'PD_dark_mean_V': result.get('PD_dark_mean_V'),
            'source_restore_error': result.get('source_restore_error'), 'saved': str(path)}), flush=True)


if __name__ == '__main__':
    main()
