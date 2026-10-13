"""Measured bench logs -> the same metrics as the analytical branch.

Session layout (docs/empirical_protocol.md):
  <session>/session.json            calibration (p_tar), notes, synthetic flag
  <session>/runs/<run>.json         family, controller, seed, duration
  <session>/runs/<run>.npz          frame_t_s (K,), psd_xy_m (K,2), command_t_s (C,), command_rad (C,2)
                                    optional: rail_t_s, rail_m, injection_t_s, injection_rad, psd_power,
                                    usrp_t_s, usrp_evm, usrp_ber
Residuals are PSD readings relative to the calibrated target (50 Hz, PSD noise included).
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

REQUIRED = ('frame_t_s', 'psd_xy_m', 'command_t_s', 'command_rad')
ORDER = ['station', 'sweep', 'fast_jitter', 'fast_sweep', 'vibration', 'acquisition']


def load_session(folder):
    folder = Path(folder)
    meta = json.loads((folder/'session.json').read_text())
    runs = []
    for path in sorted((folder/'runs').glob('*.npz')):
        info = json.loads(path.with_suffix('.json').read_text())
        arrays = dict(np.load(path))
        missing = [k for k in REQUIRED if k not in arrays]
        if missing:
            raise ValueError(f'{path.name}: missing {missing}')
        if arrays['psd_xy_m'].shape != (len(arrays['frame_t_s']), 2):
            raise ValueError(f'{path.name}: psd_xy_m must be (K, 2) matching frame_t_s')
        if np.any(np.diff(arrays['frame_t_s']) <= 0):
            raise ValueError(f'{path.name}: frame_t_s must increase')
        runs.append(dict(info, name=path.stem, arrays=arrays))
    if not runs:
        raise ValueError(f'No runs in {folder/"runs"}')
    return meta, runs


def analyse(meta, runs, spec, link, receivers):
    target = np.asarray(meta['calibration']['p_tar_m'], dtype=float)
    warm = meta.get('warmup_s', spec.raw['simulation']['warmup_s'])
    groups = {}
    for run in runs:
        groups.setdefault((run['family'], run['controller']), []).append(run)
    summary = dict(scenarios={}, comm={}, usrp={}, runs=[])
    arrays = {}
    rank = lambda key: (ORDER.index(key[0]) if key[0] in ORDER else len(ORDER), key)
    for (family, controller), members in sorted(groups.items(), key=lambda item: rank(item[0])):
        a0 = members[0]['arrays']
        t0, r0 = a0['frame_t_s']-a0['frame_t_s'][0], a0['psd_xy_m']-target
        if family == 'acquisition':
            arrays[f'demo/{controller}/t_s'], arrays[f'demo/{controller}/residual_m'] = t0, r0
            continue
        pooled, rms, spectra = [], [], []
        for run in members:
            a = run['arrays']
            t = a['frame_t_s']-a['frame_t_s'][0]
            residual = (a['psd_xy_m']-target)[t >= warm]
            pooled.append(residual)
            value = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
            rms.append(value)
            summary['runs'].append(dict(run=run['name'], family=family, controller=controller, rms_residual_um=value*1e6))
            fs = 1/np.median(np.diff(t))
            f, p = welch(residual[:, 0], fs=fs, nperseg=min(256, len(residual)))
            spectra.append(p)
        summary['scenarios'].setdefault(family, {})[controller] = dict(
            rms_residual_um=float(np.mean(rms)*1e6), runs=len(members), frames=int(sum(len(p) for p in pooled)))
        arrays[f'spectra/{family}/{controller}/f'] = f
        arrays[f'spectra/{family}/{controller}/amp'] = np.sqrt(np.mean(spectra, axis=0))
        arrays[f'{family}/{controller}/t_s'] = t0
        arrays[f'{family}/{controller}/residual_m'] = r0
        if 'rail_m' in a0 and f'{family}/rail_mm' not in arrays:
            arrays[f'{family}/t_s'] = a0['rail_t_s']-a0['rail_t_s'][0]
            arrays[f'{family}/rail_mm'] = a0['rail_m']*1e3
        residual = np.concatenate(pooled)[None]
        summary['comm'].setdefault(family, {})[controller] = {}
        for key, receiver in receivers.items():
            eta = receiver.eta(residual)
            summary['comm'][family][controller][key] = dict(link.metrics(eta, link.operating_snr), mean_power=float(eta.mean()))
            for metric, values in link.curves(eta).items():
                arrays[f'comm/{family}/{controller}/{key}/{metric}'] = values
        usrp = [run['arrays'] for run in members if 'usrp_evm' in run['arrays']]
        if usrp:
            summary['usrp'].setdefault(controller, dict(evm=[], ber=[]))
            summary['usrp'][controller]['evm'] += [float(np.mean(u['usrp_evm'])) for u in usrp]
            summary['usrp'][controller]['ber'] += [float(np.mean(u['usrp_ber'])) for u in usrp]
    for controller, values in summary['usrp'].items():
        summary['usrp'][controller] = dict(evm=float(np.mean(values['evm'])), ber=float(np.mean(values['ber'])))
    arrays['comm/snr_db'] = link.snr_grid_db
    for metric, values in link.curves(np.ones((1, 1))).items():
        arrays[f'comm/ideal/{metric}'] = values
    return summary, arrays
