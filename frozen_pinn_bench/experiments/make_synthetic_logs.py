#!/usr/bin/env python3
"""Dry run of the empirical branch: write bench-format logs from the bench model.

  python experiments/make_synthetic_logs.py --analytical results/analytical/<run> \
         --session data/empirical/synthetic_dryrun

Uses the predictor, command layer and PID gains saved by the analytical run, simulates the
listed profiles, and records only what the bench would log: delivered PSD frames, issued
commands, rail position and per-frame USRP metrics of the unfocused bench detector.
The session is marked synthetic so that every empirical figure is labelled as a dry run.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pat.comm import LinkModel, ber_qam
from pat.controllers import PIDController, PredictiveController
from pat.disturbance import _add_lines
from pat.optics import DirectBeamResponse
from pat.plant import build_episodes, simulate
from pat.predictor import load_model
from pat.spec import BenchSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analytical', required=True)
    parser.add_argument('--session', default='data/empirical/synthetic_dryrun')
    parser.add_argument('--families', nargs='+', default=['station', 'fast_sweep', 'vibration'])
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--duration', type=float, default=20.0)
    args = parser.parse_args()
    run = ROOT/args.analytical
    config = json.loads((run/'experiment.json').read_text())['config']
    spec = BenchSpec(config)
    ctrl = json.loads((run/'models'/'controllers.json').read_text())
    model = load_model(run/'models'/'frozen.npz')
    online = ctrl['online_refit']['frozen']
    online = dict(window=online['window'], every=online['every'], prior=online['prior'])
    session = ROOT/args.session
    if session.exists():
        raise SystemExit(f'{session} exists; choose a new --session')
    (session/'runs').mkdir(parents=True)
    link = LinkModel(config['evaluation'])
    bench_pda = DirectBeamResponse(spec.receiver.beam_radius_at_detectors_m, spec.optics.pda_radius_m)
    usrp_frame = 0.01
    plan = [(f, args.duration, 9100+i*10+k) for i, f in enumerate(args.families) for k in range(args.runs)]
    plan.append(('acquisition', 3.0, 9000))
    targets = []
    for family, duration, seed in plan:
        episodes = build_episodes(spec, family, [seed], duration)
        per_family = ctrl['pid_per_family']
        gains = per_family.get(family) or per_family.get('all') or per_family['station']
        controllers = {'pid': PIDController(gains['kp'], gains['ki_per_s'], gains['kd_s'], gains['derivative_filter_s']),
                       'frozen_pinn': PredictiveController(model, 'frozen_pinn', online, 'pinn', ctrl['command_layer'])}
        targets.append(episodes.target[0])
        # Split the hidden disturbance into the rail motion and the vibration injected into the FSM command.
        injected = np.zeros_like(episodes.alpha[0])
        if episodes.params[0].get('injected_lines'):
            _add_lines(injected, episodes.params[0]['injected_lines'], np.arange(len(injected))*spec.dt, spec.dt, duration)
        rail_m = (episodes.alpha[0, :, 0]-injected[:, 0])*spec.reducer*spec.psd_lever
        for name, controller in controllers.items():
            result = simulate(spec, episodes, controller, [], trace_index=(0,))
            trace = result['traces'][0]
            frames = trace['frames']
            focal = trace['focal']
            steps = np.arange(len(focal))
            ticks = steps[::spec.tick_steps]
            eta = bench_pda.normalized_power(focal)
            per = int(round(usrp_frame/spec.dt))
            blocks = len(eta)//per
            snr = link.operating_snr*eta[:blocks*per].reshape(blocks, per)**2
            arrays = dict(frame_t_s=np.array([n*spec.dt for n, _, _ in frames]),
                          psd_xy_m=np.array([reading for _, _, reading in frames]),
                          command_t_s=ticks*spec.dt, command_rad=trace['command'][ticks],
                          rail_t_s=steps[::2]*spec.dt,
                          rail_m=rail_m[::2],
                          usrp_t_s=(np.arange(blocks)+0.5)*usrp_frame,
                          usrp_evm=np.sqrt(1/snr).mean(axis=1), usrp_ber=ber_qam(64, snr).mean(axis=1))
            if injected.any():
                arrays.update(injection_t_s=steps[::2]*spec.dt,
                              injection_rad=injected[::2]@np.linalg.inv(spec.fsm_gain).T)
            stem = f'{family}_s{seed}__{name}'
            np.savez(session/'runs'/f'{stem}.npz', **arrays)
            (session/'runs'/f'{stem}.json').write_text(json.dumps(dict(family=family, controller=name, seed=seed,
                                                                       duration_s=duration), indent=2)+'\n')
            print(f'{stem}: {len(frames)} frames')
    meta = dict(synthetic=True, source=str(run.relative_to(ROOT)),
                note='Generated by the bench model; replace with measured logs in the same format.',
                calibration=dict(p_tar_m=np.mean(targets, axis=0).tolist(), J_F=(spec.psd_lever*spec.fsm_gain).tolist()),
                warmup_s=config['simulation']['warmup_s'])
    (session/'session.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(f'wrote {session}')


if __name__ == '__main__':
    main()
