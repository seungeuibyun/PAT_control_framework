#!/usr/bin/env python3
"""Analytical branch: Frozen-PINN pointing control on the bench model (config/bench.json).

Pipeline: open-loop PSD recordings (FSM held) -> fit the Frozen-PINN predictor and its linear
special case (one linear solve) -> choose the online refit and the command-layer horizon on
validation profiles -> tune PID per disturbance family -> evaluate held-out profiles,
sensitivity runs, severity sweeps and the acquisition demo -> HAPS-equivalent link metrics
-> figures. Nothing is tuned on test profiles.

Output: results/analytical/<name>/  experiment.json, summary.json, data.npz,
        models/ (deployable predictor and command layer), figures/ (+ FIGURES.md).
Runs are never overwritten.
"""
import argparse
import copy
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pat.predictor import ContextModel, replay_online, save_model
from pat.controllers import HoldController, OracleController, PIDController, PredictiveController
from pat.optics import BenchOptics, DirectBeamResponse, GaussianFocalResponse
from pat.plant import PowerModel, build_episodes, open_loop_frames, simulate
from pat.spec import BenchSpec, load_spec
from pat.comm import HAPSReceiver, LinkModel
from pat.command_layer import PINNCommandLayer

ORDER = ['hold', 'pid', 'linear_offline', 'linear_online', 'frozen_offline', 'frozen_online', 'frozen_pinn', 'oracle']


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def power_models(spec):
    """Power metrics declared in the config (evaluation.power_metrics)."""
    models = []
    for m in spec.raw['evaluation']['power_metrics']:
        kind = m['type']
        if kind == 'bench_focal':
            radius = m.get('detector_radius_m', spec.optics.pda_radius_m)
            response = BenchOptics(spec.variant(**{'optics.pda_radius_m': radius}))
        elif kind == 'direct_beam':
            response = DirectBeamResponse(spec.receiver.beam_radius_at_detectors_m,
                                          m.get('detector_radius_m', spec.optics.pda_radius_m))
        elif kind == 'gaussian_focal':
            response = GaussianFocalResponse(m['wavelength_m'], m['focal_length_m'], m['pupil_radius_m'],
                                             m['aperture_radius_m'], m['detector_radius_m'])
        else:
            raise ValueError(f'Unknown power metric type {kind}')
        scale = m['focal_length_m']/spec.psd_lever if m.get('source') == 'psd' else 1.0
        models.append(PowerModel(m['name'], response, m.get('source', 'pda'), scale))
    return models


def sequences(spec, episodes):
    readings, _ = open_loop_frames(spec, episodes)
    return list(readings/spec.psd_lever)


def episodes_for(spec, name, quick):
    family, seeds, duration = spec.episodes(name)
    if quick:
        seeds, duration = seeds[:max(2, len(seeds)//8)], min(duration, 10.0)
    return build_episodes(spec, family, seeds, duration)


def tune_pid(spec, episodes, models):
    g = spec.controllers.pid_grid
    combos = list(product(g.kp, g.ki_per_s, g.kd_s))
    count = episodes.count
    gains = np.repeat(np.asarray(combos), count, axis=0)
    controller = PIDController(gains[:, 0], gains[:, 1], gains[:, 2], g.derivative_filter_s)
    result = simulate(spec, episodes, controller, models[:1], batch_index=np.tile(np.arange(count), len(combos)))
    rms = result['rms_focal_m'].reshape(len(combos), count).mean(axis=1)
    best = int(np.argmin(rms))
    kp, ki, kd = combos[best]
    edge = dict(kp=kp in (min(g.kp), max(g.kp)), ki=ki in (min(g.ki_per_s), max(g.ki_per_s)),
                kd=kd in (min(g.kd_s), max(g.kd_s)))
    return dict(kp=kp, ki_per_s=ki, kd_s=kd, derivative_filter_s=g.derivative_filter_s,
                mean_rms_residual_m=float(rms[best]), on_grid_edge=edge,
                grid=[dict(kp=c[0], ki_per_s=c[1], kd_s=c[2], mean_rms_residual_m=float(r)) for c, r in zip(combos, rms)])


def select_online(model, validation, windows, priors, every):
    scores = {}
    for window, prior in product(windows, priors):
        scores[(window, prior)] = replay_online(model, validation, window, every, prior)
    (window, prior), mse = min(scores.items(), key=lambda item: item[1])
    offline = replay_online(model, validation, max(windows), every, None)
    return dict(window=window, prior=prior, every=every, validation_rmse_h1_h2_rad=float(np.sqrt(mse)),
                offline_validation_rmse_h1_h2_rad=float(np.sqrt(offline)),
                grid=[dict(window=w, prior=p, rmse_rad=float(np.sqrt(v))) for (w, p), v in scores.items()])


def online_cfg(online, key):
    cfg = online[key]
    return dict(window=cfg['window'], every=cfg['every'], prior=cfg['prior'])


def select_command_layer(spec, model, online, models_power):
    """Choose the Frozen-PINN command horizon (hold, scored ticks) by closed loop on validation seeds."""
    c = dict(spec.raw['controllers']['pinn_command'])
    episodes = build_episodes(spec, spec.episodes('validation')[0],
                              [c['selection_seed0']+i for i in range(c['selection_episodes'])], c['selection_duration_s'])
    scores = {}
    lead = simulate(spec, episodes, PredictiveController(model, 'lead', online_cfg(online, 'frozen')), models_power[:1])
    for hold, point in c['horizon_grid']:
        cfg = dict(c, hold_ticks=hold, point_ticks=point)
        r = simulate(spec, episodes, PredictiveController(model, 'pinn', online_cfg(online, 'frozen'), 'pinn', cfg),
                     models_power[:1])
        scores[(hold, point)] = float(r['rms_focal_m'].mean())
    (hold, point), best = min(scores.items(), key=lambda item: item[1])
    layer = dict(c, hold_ticks=hold, point_ticks=point)
    return layer, dict(hold_ticks=hold, point_ticks=point, validation_rms_m=best,
                       lead_validation_rms_m=float(lead['rms_focal_m'].mean()),
                       grid=[dict(hold_ticks=h, point_ticks=q, rms_m=v) for (h, q), v in scores.items()])


def controllers_for(episodes, pid, models, online, layer):
    out = {'hold': HoldController(),
           'pid': PIDController(pid['kp'], pid['ki_per_s'], pid['kd_s'], pid['derivative_filter_s']),
           'oracle': OracleController(episodes)}
    for key, model in models.items():
        out[f'{key}_offline'] = PredictiveController(model, f'{key}_offline')
        out[f'{key}_online'] = PredictiveController(model, f'{key}_online', online_cfg(online, key))
    # Full method: Frozen-PINN predictor with online refit + FSM-dynamics-residual command layer.
    out['frozen_pinn'] = PredictiveController(models['frozen'], 'frozen_pinn', online_cfg(online, 'frozen'), 'pinn', layer)
    return out


def summarize(spec, results, models):
    pid = results['pid']
    m_g = spec.raw['evaluation'].get('pdf_receiver_reducer', 0.1)
    table = {}
    for name in ORDER:
        if name not in results:
            continue
        r = results[name]
        rms = r['rms_focal_m']
        row = dict(rms_residual_um=float(rms.mean()*1e6), rms_residual_um_std=float(rms.std()*1e6),
                   rms_x_um=float(r['rms_focal_axis_m'][:, 0].mean()*1e6),
                   rms_y_um=float(r['rms_focal_axis_m'][:, 1].mean()*1e6),
                   rms_fsm_plane_urad=float(rms.mean()/spec.psd_lever*1e6),
                   rms_pdf_aperture_urad=float(rms.mean()/spec.psd_lever*m_g*1e6),
                   peak_residual_um=float(r['peak_focal_m'].mean()*1e6),
                   tick_ms_per_episode=float(r['tick_time_per_episode_s']*1e3),
                   per_episode_rms_residual_um=(rms*1e6).round(3).tolist())
        for i, model in enumerate(models):
            row[f'power_{model.name}'] = float(r['mean_power'][:, i].mean())
            row[f'outage_{model.name}'] = float(r['outage'][:, i].mean())
        if name != 'pid':
            ratio = rms/pid['rms_focal_m']
            row.update(rms_ratio_vs_pid=float(ratio.mean()), episodes_better_than_pid=int(np.sum(ratio < 1)),
                       episodes=int(len(ratio)))
            for i, model in enumerate(models):
                row[f'power_gain_vs_pid_{model.name}'] = float(r['mean_power'][:, i].mean()-pid['mean_power'][:, i].mean())
        table[name] = row
    return table


def run_scenario(spec, episodes, controllers, models, *, capture_steps=None, keep=True):
    results = {}
    for name in ORDER:
        if name not in controllers:
            continue
        results[name] = simulate(spec, episodes, controllers[name], models, trace_index=(0,),
                                 capture_steps=capture_steps, keep_residual=keep)
    return results


def spectra(spec, results, names):
    out = {}
    start = spec.warmup_steps
    for name in names:
        residual = results[name]['residual'][:, start:].astype(float)
        f, p = welch(residual, fs=1/spec.dt, nperseg=8192, axis=1)
        out[name] = p.mean(axis=0)
    return f, out


def rolling_rms(spec, results, names, window_s=0.5):
    """Episode-averaged RMS residual in consecutive windows (shows adaptation over time)."""
    step = int(round(window_s/spec.dt))
    out = {}
    for name in names:
        r = results[name]['residual'].astype(float)
        blocks = r.shape[1]//step
        sq = np.sum(r[:, :blocks*step]**2, axis=2).reshape(r.shape[0], blocks, step).mean(axis=2)
        out[name] = np.sqrt(sq.mean(axis=0))
    return (np.arange(blocks)+0.5)*window_s, out


COMM_NAMES = ['hold', 'pid', 'linear_online', 'frozen_online', 'frozen_pinn', 'oracle']


def comm_eval(spec, results, link, receivers):
    """HAPS-equivalent BER / outage / rate from the residuals (after warm-up)."""
    summary, arrays = {}, {}
    start = spec.warmup_steps
    for name in COMM_NAMES:
        if name not in results or results[name]['residual'] is None:
            continue
        residual = results[name]['residual'][:, start:].astype(float)
        summary[name] = {}
        for key, receiver in receivers.items():
            eta = receiver.eta(residual)
            summary[name][key] = dict(link.metrics(eta, link.operating_snr), mean_power=float(eta.mean()))
            for metric, values in link.curves(eta).items():
                arrays[f'{name}/{key}/{metric}'] = values
    return summary, arrays


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='config/bench.json')
    parser.add_argument('--name', default='run_000')
    parser.add_argument('--quick', action='store_true', help='small smoke run; not for reporting')
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    spec = load_spec(ROOT/args.config)
    out = ROOT/'results'/'analytical'/(args.name+('_quick' if args.quick else ''))
    if out.exists():
        raise SystemExit(f'{out} exists; choose a new --name')
    out.mkdir(parents=True)
    started = time.time()
    cfg = spec.controllers
    models_power = power_models(spec)
    log = lambda message: print(f'[{time.time()-started:7.1f}s] {message}', flush=True)

    train, validation = episodes_for(spec, 'train', args.quick), episodes_for(spec, 'validation', args.quick)
    train_seq, val_seq = sequences(spec, train), sequences(spec, validation)
    log(f'{spec.raw["id"]}: training {train.count} x {train.duration:g} s, validation {validation.count} x {validation.duration:g} s')

    models = {'linear': ContextModel(cfg.history_frames, cfg.horizon_frames, 0, cfg.seed),
              'frozen': ContextModel(cfg.history_frames, cfg.horizon_frames, cfg.frozen_features, cfg.seed)}
    online = {}
    for key, model in models.items():
        model.fit(train_seq, val_seq, cfg.ridge_grid, cfg.swim_candidate_pairs)
        windows = sorted({cfg.online_window_frames//2, cfg.online_window_frames})
        online[key] = select_online(model, val_seq, windows, cfg.online_prior_grid, cfg.online_refit_every_frames)
        log(f'{key}: ridge {model.fit_report["ridge"]:g}, offline/online validation RMSE '
            f'{online[key]["offline_validation_rmse_h1_h2_rad"]*1e6:.1f}/{online[key]["validation_rmse_h1_h2_rad"]*1e6:.1f} urad (FSM plane)')

    tuning = episodes_for(spec, 'train', args.quick)
    tuning = build_episodes(spec, tuning.family, tuning.seeds[:cfg.pid_tuning_episodes], tuning.duration)
    pid = tune_pid(spec, tuning, models_power)
    log(f'PID tuned: kp {pid["kp"]}, ki {pid["ki_per_s"]}, kd {pid["kd_s"]}, edge {pid["on_grid_edge"]}')
    layer, layer_report = select_command_layer(spec, models['frozen'], online, models_power)
    log(f'command layer: hold {layer["hold_ticks"]}, scored {layer["point_ticks"]} ticks, validation RMS '
        f'{layer_report["validation_rms_m"]*1e6:.2f} um (lead approximation {layer_report["lead_validation_rms_m"]*1e6:.2f} um)')

    ev = spec.raw['evaluation']
    per_scenario = ev.get('pid_tuning', {}).get('mode') == 'per_scenario'
    pid_by_family = {}

    def pid_for(family):
        """PID gains for a test family: re-tuned on its own training seeds when per_scenario."""
        if not per_scenario or isinstance(family, (list, tuple)):
            return pid
        if family not in pid_by_family:
            _, seeds, duration = spec.episodes('train')
            if args.quick:
                duration = min(duration, 10.0)
            tuned = tune_pid(spec, build_episodes(spec, family, seeds[:cfg.pid_tuning_episodes], duration), models_power)
            pid_by_family[family] = tuned
            log(f'PID tuned for {family}: kp {tuned["kp"]}, ki {tuned["ki_per_s"]}, kd {tuned["kd_s"]}, edge {tuned["on_grid_edge"]}')
        return pid_by_family[family]

    comm_cfg = ev.get('comm')
    link = receivers = None
    if comm_cfg:
        link = LinkModel(ev)
        receivers = {f'r{int(round(r*1e6))}': HAPSReceiver(comm_cfg, r, spec.psd_lever)
                     for r in comm_cfg['detector_radii_m']}
    comm_summary, comm_arrays = {}, {}

    scenarios = [k for k in ev['episodes'] if k.startswith('test')]
    summary, extras, traces = {}, {}, {}
    names_plot = ['hold', 'pid', 'linear_offline', 'linear_online', 'frozen_offline', 'frozen_online', 'frozen_pinn', 'oracle']
    for scenario in scenarios:
        episodes = episodes_for(spec, scenario, args.quick)
        scenario_pid = pid_for(ev['episodes'][scenario]['family'])
        results = run_scenario(spec, episodes, controllers_for(episodes, scenario_pid, models, online, layer), models_power)
        summary[scenario] = summarize(spec, results, models_power)
        if link:
            comm_summary[scenario], arrays_s = comm_eval(spec, results, link, receivers)
            comm_arrays.update({f'comm/{scenario}/{k}': v for k, v in arrays_s.items()})
            op = comm_summary[scenario]
            log(f'{scenario} comm (r25 outage / rate Mb/s): ' + ', '.join(
                f'{k} {v["r25"]["outage64"]*100:.1f}% / {v["r25"]["rate_bps"]/1e6:.1f}' for k, v in op.items()))
        f, s = spectra(spec, results, names_plot)
        t, rolling = rolling_rms(spec, results, names_plot)
        extras[scenario] = dict(frequency_hz=f, spectra=s, rolling_time_s=t, rolling=rolling)
        traces[scenario] = dict(alpha_rad=episodes.alpha[0], params=episodes.params[0],
                                controllers={n: dict(residual_m=results[n]['traces'][0]['focal'],
                                                     command_rad=results[n]['traces'][0]['command'])
                                             for n in names_plot})
        log(f'{scenario}: ' + ', '.join(f'{k} {v["rms_residual_um"]:.1f}' for k, v in summary[scenario].items()))

    sensitivity = {}
    for label, sens in spec.raw['evaluation'].get('sensitivity', {}).items():
        variant = spec.variant(**sens.get('overrides', {})) if sens.get('overrides') else spec
        family, seeds, duration = variant.episodes(sens.get('episodes', 'test_id'))
        if args.quick:
            seeds, duration = seeds[:2], 10.0
        episodes = build_episodes(variant, family, seeds, duration)
        capture = (int(round(sens['true_capture_after_query_s']/variant.dt))
                   if 'true_capture_after_query_s' in sens else None)
        variant_layer, reselected = layer, None
        if any(key.startswith('fsm.') for key in sens.get('overrides', {})):
            # The command horizon is a design choice tied to the actuator dynamics: re-select it on
            # validation seeds for the changed actuator (as after identifying a new stage on the bench).
            variant_layer, reselected = select_command_layer(variant, models['frozen'], online, power_models(variant))
        results = run_scenario(variant, episodes, controllers_for(episodes, pid_for(family), models, online, variant_layer),
                               power_models(variant), capture_steps=capture, keep=False)
        sensitivity[label] = dict(description=sens.get('description', ''), command_layer_reselected=reselected,
                                  results=summarize(variant, results, power_models(variant)))
        log(f'sensitivity {label}: ' + ', '.join(f'{k} {v["rms_residual_um"]:.1f}' for k, v in sensitivity[label]['results'].items()))

    severity = {}
    sev = ev.get('severity')
    if sev and not args.quick:
        raw = copy.deepcopy(spec.raw)
        fams = raw['disturbance']['families']
        cases = []
        for i, speed in enumerate(sev['sweep_speeds_m_s']):
            f = copy.deepcopy(fams['fast_sweep'])
            f['sweep'].update(speed_m_s=[speed, speed], amplitude_m=[3e-3, 3e-3])
            fams[f'sev_sweep_{i}'] = f
            cases.append(('sweep', speed, f'sev_sweep_{i}', 'fast_sweep'))
        for j, scale in enumerate(sev['vibration_scales']):
            f = copy.deepcopy(fams['vibration'])
            f['injected_lines']['amplitude_scale'] = scale
            fams[f'sev_vib_{j}'] = f
            cases.append(('vibration', scale, f'sev_vib_{j}', 'vibration'))
        sev_spec = BenchSpec(raw)
        seeds = [sev['seed0']+i for i in range(sev['episodes'])]
        for kind, value, fam_name, tuned_family in cases:
            episodes = build_episodes(sev_spec, fam_name, seeds, sev['duration_s'])
            ctrls = controllers_for(episodes, pid_for(tuned_family), models, online, layer)
            row = {}
            for name in ('pid', 'frozen_pinn', 'oracle'):
                r = simulate(sev_spec, episodes, ctrls[name], models_power[:1], keep_residual=bool(link))
                entry = dict(rms_residual_um=float(r['rms_focal_m'].mean()*1e6))
                if link:
                    residual = r['residual'][:, sev_spec.warmup_steps:].astype(float)
                    for key, receiver in receivers.items():
                        entry[key] = link.metrics(receiver.eta(residual), link.operating_snr)
                row[name] = entry
            severity.setdefault(kind, []).append(dict(value=value, results=row))
        log('severity sweeps: ' + '; '.join(f'{k} {len(v)} points' for k, v in severity.items()))

    demo_arrays = {}
    demo = ev.get('demo')
    if demo:
        episodes = build_episodes(spec, demo['family'], [demo['seed']], demo['duration_s'])
        ctrls = controllers_for(episodes, pid_for('station'), models, online, layer)
        for name in ('hold', 'pid', 'frozen_pinn', 'oracle'):
            r = simulate(spec, episodes, ctrls[name], models_power[:1], trace_index=(0,))
            demo_arrays[f'demo/{name}/residual_m'] = r['traces'][0]['focal']
            demo_arrays[f'demo/{name}/command_rad'] = r['traces'][0]['command']
        demo_arrays['demo/alpha_rad'] = episodes.alpha[0]
        log('acquisition demo done')

    derived = dict(mode=spec.mode, psd_lever_m=spec.psd_lever, pda_lever_m=spec.pda_lever,
                   beam_radius_at_fsm_m=spec.beam_radius_at_fsm, fsm_group_delay_s=spec.fsm_group_delay,
                   psd_lpf_tau_s=spec.lpf_tau, pda_peak_voltage_v=spec.pda_peak_voltage, psd_lsb_m=spec.psd_lsb)
    if spec.mode == 'direct':
        derived.update(reducer=spec.reducer, rail_m_per_fsm_plane_rad=spec.reducer*spec.psd_lever)
    experiment = dict(id=spec.raw['id'], name=args.name, quick=args.quick, git_commit=git_commit(),
                      python=platform.python_version(), numpy=np.__version__, started_unix=started,
                      elapsed_s=time.time()-started, config=spec.raw, derived=derived,
                      power_metrics={m.name: dict(source=m.source, scale=m.scale, **m.response.summary())
                                     for m in models_power},
                      context_models={k: m.fit_report for k, m in models.items()}, online_selection=online,
                      pid_tuning=pid, pid_tuning_per_family=pid_by_family, command_layer=layer_report,
                      comm_link=(dict(operating_snr_db=10*np.log10(link.operating_snr),
                                      thresholds_db={m: 10*np.log10(v) for m, v in link.thresholds.items()},
                                      receivers={k: dict(radius_m=r.radius, **r.response.summary()) for k, r in receivers.items()})
                                 if link else None),
                      information=dict(
                          all_controllers='Delivered PSD frames (x, y) with assumed capture time, own commands, nominal C_F and FSM model, calibrated target',
                          predictors_offline='Fitted on open-loop training recordings (FSM held), labels = future PSD frames',
                          predictors_online="Additionally refit on the current episode's own past frames",
                          pid_tuning='Closed-loop grid search on training episodes (RMS PSD-plane residual)',
                          oracle='True disturbance, offset and C_F; bound only',
                          hidden='Disturbance, true C_F, FSM offset/jitter, PSD noise'))
    (out/'experiment.json').write_text(json.dumps(jsonable(experiment), indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(jsonable(dict(scenarios=summary, sensitivity=sensitivity, comm=comm_summary,
                                                             severity=severity)), indent=2)+'\n')
    arrays = {}
    for scenario, group in traces.items():
        arrays[f'{scenario}/alpha_rad'] = group['alpha_rad']
        for name, item in group['controllers'].items():
            for field, value in item.items():
                arrays[f'{scenario}/{name}/{field}'] = value
        e = extras[scenario]
        arrays[f'spectra/{scenario}/frequency_hz'] = e['frequency_hz']
        arrays[f'rolling/{scenario}/time_s'] = e['rolling_time_s']
        for name in names_plot:
            arrays[f'spectra/{scenario}/{name}'] = e['spectra'][name]
            arrays[f'rolling/{scenario}/{name}'] = e['rolling'][name]
    arrays.update(comm_arrays)
    arrays.update(demo_arrays)
    if link:
        arrays['comm/snr_db'] = link.snr_grid_db
        for metric, values in link.curves(np.ones((1, 1))).items():
            arrays[f'comm/ideal/{metric}'] = values
    np.savez_compressed(out/'data.npz', **arrays)
    (out/'models').mkdir()
    for key, model in models.items():
        save_model(model, out/'models'/f'{key}.npz')
    command = PINNCommandLayer(spec, spec.fsm_gain, layer)
    np.savez(out/'models'/'command_layer.npz', K_u=command.K_u, tau=command.tau, config=json.dumps(jsonable(layer)))
    (out/'models'/'controllers.json').write_text(json.dumps(jsonable(dict(
        pid_per_family=pid_by_family or {'all': pid}, online_refit=online, command_layer=layer)), indent=2)+'\n')
    (out/'trace_params.json').write_text(json.dumps(jsonable({k: v['params'] for k, v in traces.items()}), indent=2)+'\n')
    if not args.no_figures:
        sys.path.insert(0, str(ROOT/'experiments'))
        from make_figures import analytical_figures
        analytical_figures(out)
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
