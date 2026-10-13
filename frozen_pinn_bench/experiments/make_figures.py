#!/usr/bin/env python3
"""Figures for one result folder, split into the two branches.

  python experiments/make_figures.py --analytical results/analytical/<run>
  python experiments/make_figures.py --empirical  results/empirical/<session>

Writes <folder>/figures/A** or E** as PNG, PDF and EPS (--formats to change), figures/FIGURES.md
(Korean captions) and RESULTS.md (tables).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pat import figures as F
from pat.comm import HAPSReceiver, LinkModel
from pat.optics import DirectBeamResponse
from pat.spec import BenchSpec

SCENARIOS = [('test_id', 'Station keeping'), ('test_sweep', 'Sweep 1-5 m/s'), ('test_fast_jitter', 'Fast jitter'),
             ('test_fast_sweep', 'Fast motion 6-14 m/s'), ('test_vibration', 'Injected vibration')]
LINK_SCENARIOS = [('test_id', 'Station keeping'), ('test_fast_sweep', 'Fast lateral motion, 6-14 m/s'),
                  ('test_vibration', 'Injected 6-18 Hz vibration')]
SUMMARY_NAMES = ['pid', 'frozen_offline', 'frozen_online', 'linear_online', 'frozen_pinn', 'oracle']


def _context(config):
    spec = BenchSpec(config)
    ev = config['evaluation']
    link = LinkModel(ev)
    receivers = {f'r{int(round(r*1e6))}': HAPSReceiver(ev['comm'], r, spec.psd_lever) for r in ev['comm']['detector_radii_m']}
    return spec, ev, link, receivers


def _curves(data, scenarios, names=('pid', 'frozen_pinn', 'oracle')):
    out = {}
    for sc, _ in scenarios:
        out[sc] = {}
        for name in names:
            if f'comm/{sc}/{name}/r75/ber64' in data:
                out[sc][name] = {rk: {m: data[f'comm/{sc}/{name}/{rk}/{m}'] for m in ('ber64', 'outage64', 'rate_bps')}
                                 for rk, _ in F.DETECTORS}
    return out


def _capture_curves(spec, receivers):
    offsets = np.linspace(0, 400e-6, 801)
    points = np.stack([offsets, np.zeros_like(offsets)], axis=1)
    bench = DirectBeamResponse(spec.receiver.beam_radius_at_detectors_m, spec.optics.pda_radius_m)
    return {'HAPS receiver, 75 µm detector': (offsets, receivers['r75'].eta(points), dict(color=F.BLUE, lw=1.6)),
            'HAPS receiver, 25 µm detector': (offsets, receivers['r25'].eta(points), dict(color=F.AQUA, lw=1.6)),
            'Bench detector (unfocused 0.55 mm beam)': (offsets, bench.normalized_power(points),
                                                        dict(color=F.INK2, lw=1.3, ls='--'))}


def _reversal_window(t, rail_mm, warm, half=0.6):
    smooth = np.convolve(np.gradient(rail_mm), np.ones(200)/200, mode='same')
    start = int(np.searchsorted(t, warm))
    flips = np.nonzero(np.diff(np.sign(smooth[start:])) != 0)[0]
    centre = t[start+flips[0]] if len(flips) else t[len(t)//2]
    return (max(centre-half, warm), centre+half)


def analytical_figures(run_dir, formats=F.FORMATS):
    run_dir = Path(run_dir)
    data = np.load(run_dir/'data.npz')
    summary = json.loads((run_dir/'summary.json').read_text())
    config = json.loads((run_dir/'experiment.json').read_text())['config']
    spec, ev, link, receivers = _context(config)
    dt, lever = spec.dt, spec.psd_lever
    rail = lambda sc: data[f'{sc}/alpha_rad'][:, 0]*spec.reducer*lever*1e3
    idx = F.FigureIndex(run_dir/'figures', 'Analytical figures (bench model + HAPS-equivalent link)', formats=formats)

    idx.save(F.fig_capture_function(_capture_curves(spec, receivers)), 'A01_capture_function.png',
             '검출기 오프셋에 따른 수집 전력. HAPS 수신기(초점면)에서 75 µm 검출기는 ±40 µm까지 평평하고, '
             '25 µm 고속 검출기는 수 µm부터 전력이 줄어든다. 벤치 검출기(렌즈 없는 0.55 mm 빔)는 수백 µm에서야 줄어든다.')
    profiles = [(label, np.arange(len(data[f'{sc}/alpha_rad']))*dt, rail(sc)) for sc, label in SCENARIOS]
    idx.save(F.fig_rail_profiles(profiles, spec.rail.travel_m*1e3), 'A02_disturbance_profiles.png',
             '시험 교란군별 첫 에피소드의 레일 위치(진동군은 FSM에 주입한 성분을 레일 등가로 포함). 20 km HAPS 횡변위 = 레일 × 1000.')
    table = {sc: {n: v['rms_residual_um'] for n, v in summary['scenarios'][sc].items()} for sc, _ in SCENARIOS}
    idx.save(F.fig_pointing_summary(SCENARIOS, table, SUMMARY_NAMES), 'A03_pointing_summary.png',
             '시험 세트별 RMS 지향 잔차(검출기면 µm = HAPS 초점면 µm). 빗금은 ablation: 온라인 재적합 제거, '
             'FSM-ODE 명령층 제거. 주황은 선형 예측기(M = 0).')
    if summary.get('severity'):
        lines = config['disturbance']['families']['vibration']['injected_lines']
        idx.save(F.fig_rms_vs_severity(summary['severity'], np.mean(lines['line_amp_rad'])*lever*1e6),
                 'A04_rms_vs_severity.png', 'HAPS 횡속도와 주입 진동 진폭에 따른 RMS 잔차. PID는 교란 세기에 따라 커지고 '
                 'Frozen-PINN은 지터 하한 근처를 유지한다.')
    panels = []
    for sc, label in LINK_SCENARIOS:
        f = data[f'spectra/{sc}/frequency_hz']
        panels.append((label, {n: (f, np.sqrt(data[f'spectra/{sc}/{n}'][:, 0])) for n in ('hold', 'pid', 'frozen_pinn', 'oracle')}))
    idx.save(F.fig_spectra(panels), 'A05_residual_spectra.png',
             '잔차 x축 스펙트럼. PID는 저주파 추종 지연과 진동선이 남고, Frozen-PINN은 25 Hz 이하 성분을 상쇄한다. '
             '점선(oracle)은 MR-15-30 지터 하한.')
    t = np.arange(len(data['test_fast_sweep/alpha_rad']))*dt
    window = _reversal_window(t, rail('test_fast_sweep'), spec.raw['simulation']['warmup_s'])
    residuals = {n: data[f'test_fast_sweep/{n}/residual_m'] for n in ('pid', 'frozen_pinn')}
    idx.save(F.fig_tracking_window(t, rail('test_fast_sweep'), residuals, window, 'fast lateral motion (rail reversal)'),
             'A06_tracking_window.png', '빠른 이동 중 레일 반전 구간의 잔차. PID는 등속 구간에서 뒤처지고 반전에서 크게 벗어난다.')
    curves = _curves(data, LINK_SCENARIOS)
    ideal = {m: data[f'comm/ideal/{m}'] for m in ('ber64', 'rate_bps')}
    idx.save(F.fig_link_curves(data['comm/snr_db'], curves, ideal, LINK_SCENARIOS, 'ber64', link), 'A07_ber_vs_snr.png',
             'HAPS 등가 링크의 64-QAM BER 대 정렬 SNR. 위: 75 µm, 아래: 25 µm 검출기.')
    idx.save(F.fig_link_curves(data['comm/snr_db'], curves, ideal, LINK_SCENARIOS, 'rate_bps', link), 'A08_rate_vs_snr.png',
             '적응 QAM(4/16/64/256) 데이터율 대 정렬 SNR(10 Mbaud, 페이로드 82.4%).')
    idx.save(F.fig_outage(summary['comm'], SCENARIOS), 'A09_outage.png',
             '동작점(64-QAM FEC 임계값 + 3 dB)에서의 outage. 75 µm는 진동 주입에서만, 25 µm는 모든 교란에서 차이가 난다.')
    if 'demo/pid/residual_m' in data:
        traces = {n: F.Trace(np.arange(len(data[f'demo/{n}/residual_m']))*dt, data[f'demo/{n}/residual_m'], receivers['r75'])
                  for n in ('pid', 'frozen_pinn')}
        idx.save(F.fig_acquisition(traces, link, ev['comm']), 'A10_acquisition.png',
                 '2 mm 어긋난 상태에서 시작한 획득 과정과 constellation(보정 전, 75 µm). Frozen-PINN은 첫 PSD 프레임(20 ms) '
                 '직후 한 번에 맞추고 PID는 40-70 ms에 맞춘다.', dpi=150)
    for k, (sc, label) in enumerate([('test_fast_sweep', 'Fast lateral motion, 6-14 m/s'),
                                     ('test_vibration', 'Injected 6-18 Hz vibration')]):
        traces = {n: F.Trace(np.arange(len(data[f'{sc}/{n}/residual_m']))*dt, data[f'{sc}/{n}/residual_m'], receivers['r25'])
                  for n in ('pid', 'frozen_pinn')}
        idx.save(F.fig_dashboard(traces, link, ev['comm'], 10.0, label), f'A{11+k}_dashboard_{sc[5:]}.png',
                 f'{label}: HAPS 초점면 스폿 영상과 검출기 원(점선 75 µm, 실선 25 µm), 최근 1초 궤적, 최근 200 ms constellation, '
                 '25 µm 수신 전력과 BER 시계열(t = 10 s).', dpi=140)
    idx.write()
    analytical_tables(run_dir)
    return idx


EMPIRICAL_LABELS = {'station': 'Station keeping', 'sweep': 'Sweep 1-5 m/s', 'fast_jitter': 'Fast jitter',
                    'fast_sweep': 'Fast motion 6-14 m/s', 'vibration': 'Injected vibration'}


TABLE_NAMES = [('hold', 'Hold'), ('pid', 'PID'), ('linear_online', 'Linear (M=0) + refit'),
               ('frozen_offline', 'Frozen-PINN, no refit'), ('frozen_online', 'Frozen-PINN, lead command'),
               ('frozen_pinn', '**Frozen-PINN**'), ('oracle', 'Oracle (jitter floor)')]


def _md(header, rows):
    lines = ['| '+' | '.join(header)+' |', '|'+'---|'*len(header)]
    return '\n'.join(lines+['| '+' | '.join(str(c) for c in row)+' |' for row in rows])


def _pct(x):
    return f'{100*x:.2f}%' if x < 0.1 else f'{100*x:.1f}%'


def analytical_tables(run_dir):
    """RESULTS.md: the numbers behind the A figures, generated from summary.json."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir/'summary.json').read_text())
    sc = summary['scenarios']
    out = [f'# Analytical results: {run_dir.name}', '',
           'Generated by `experiments/make_figures.py` from `summary.json`. Bench model, 16 held-out episodes x 20 s '
           'per test set (first 2 s excluded). Residual = detector-plane offset at 0.5 ms resolution '
           '(= HAPS focal-plane offset; PDF aperture angle = residual / 2 in urad).', '',
           '## RMS pointing residual [um]', '']
    rows = []
    for name, label in TABLE_NAMES:
        if name in sc['test_id']:
            rows.append([label]+[f"{sc[k][name]['rms_residual_um']:.1f}" for k, _ in SCENARIOS])
    rows.append(['Frozen-PINN better than PID (episodes)']+[f"{sc[k]['frozen_pinn']['episodes_better_than_pid']}/{sc[k]['frozen_pinn']['episodes']}"
                                                        for k, _ in SCENARIOS])
    out += [_md(['Controller']+[label for _, label in SCENARIOS], rows), '']
    if summary.get('comm'):
        out += ['## HAPS-equivalent link at the operating point (64-QAM FEC threshold + 3 dB)', '']
        rows = []
        for k, label in SCENARIOS:
            for name in ('pid', 'linear_online', 'frozen_pinn', 'oracle'):
                c = summary['comm'][k][name]
                rows.append([label if name == 'pid' else '', dict(TABLE_NAMES)[name],
                             _pct(c['r75']['outage64']), f"{c['r75']['ber64']:.1e}",
                             _pct(c['r25']['outage64']), f"{c['r25']['ber64']:.1e}", f"{c['r25']['rate_bps']/1e6:.1f}"])
        out += [_md(['Scenario', 'Controller', 'Outage 75 um', 'BER 75 um', 'Outage 25 um', 'BER 25 um', 'Rate 25 um [Mb/s]'], rows), '']
    if summary.get('severity'):
        out += ['## Severity sweeps (RMS um / 25 um rate Mb/s)', '']
        for key, unit, scale in (('sweep', 'HAPS speed [m/s]', 1e3), ('vibration', 'vibration amplitude (x nominal 60-200 urad)', 1)):
            rows = [[f"{p['value']*scale:g}"]+[f"{p['results'][n]['rms_residual_um']:.1f} / {p['results'][n]['r25']['rate_bps']/1e6:.1f}"
                                              for n in ('pid', 'frozen_pinn', 'oracle')] for p in summary['severity'][key]]
            out += [_md([unit, 'PID', 'Frozen-PINN', 'Oracle'], rows), '']
    if summary.get('sensitivity'):
        out += ['## Sensitivity (RMS um)', '']
        rows = []
        for key, item in summary['sensitivity'].items():
            r = item['results']
            rows.append([item.get('description', key)]+[f"{r[n]['rms_residual_um']:.1f}" for n in ('pid', 'frozen_online', 'frozen_pinn', 'oracle')])
        out += [_md(['Condition', 'PID', 'Frozen-PINN, lead command', 'Frozen-PINN', 'Oracle'], rows), '']
    (run_dir/'RESULTS.md').write_text('\n'.join(out))


def empirical_tables(result_dir):
    result_dir = Path(result_dir)
    summary = json.loads((result_dir/'summary.json').read_text())
    out = [f'# Empirical results: {result_dir.name}', '']
    if summary.get('banner'):
        out += [f"**{summary['banner']}**", '']
    out += ['RMS residual from PSD frames (50 Hz, PSD noise included), relative to the calibrated target.', '']
    names = sorted({n for v in summary['scenarios'].values() for n in v})
    rows = [[EMPIRICAL_LABELS.get(k, k)]+[f"{v[n]['rms_residual_um']:.1f} ({v[n]['runs']})" if n in v else '-' for n in names]
            for k, v in summary['scenarios'].items()]
    out += [_md(['Scenario']+[f'{n} [um] (runs)' for n in names], rows), '']
    if summary.get('comm'):
        out += ['## HAPS-equivalent link from the measured residual (operating point)', '']
        rows = [[EMPIRICAL_LABELS.get(k, k), n, _pct(c['r75']['outage64']), _pct(c['r25']['outage64']), f"{c['r25']['rate_bps']/1e6:.1f}"]
                for k, v in summary['comm'].items() for n, c in v.items()]
        out += [_md(['Scenario', 'Controller', 'Outage 75 um', 'Outage 25 um', 'Rate 25 um [Mb/s]'], rows), '']
    if summary.get('usrp'):
        rows = [[n, f"{u['evm']*100:.1f}%", f"{u['ber']:.1e}"] for n, u in summary['usrp'].items()]
        out += ['## Bench USRP link (unfocused bench detector)', '', _md(['Controller', 'EVM', 'BER'], rows), '']
    (result_dir/'RESULTS.md').write_text('\n'.join(out))


def empirical_figures(result_dir, formats=F.FORMATS):
    result_dir = Path(result_dir)
    data = np.load(result_dir/'data.npz')
    summary = json.loads((result_dir/'summary.json').read_text())
    config = summary['config']
    spec, ev, link, receivers = _context(config)
    banner = summary.get('banner')
    scenarios = [(sc, EMPIRICAL_LABELS.get(sc, sc)) for sc in summary['scenarios']]
    idx = F.FigureIndex(result_dir/'figures', 'Empirical figures (measured bench logs)', banner, formats)
    profiles = [(label, data[f'{sc}/t_s'], data[f'{sc}/rail_mm']) for sc, label in scenarios if f'{sc}/rail_mm' in data]
    if profiles:
        note = 'Rail encoder position (injected vibration not included). HAPS lateral position at 20 km = rail × 1000.'
        idx.save(F.fig_rail_profiles(profiles, spec.rail.travel_m*1e3, banner, note), 'E01_disturbance_profiles.png',
                 '측정한 레일 엔코더 위치(각 프로파일의 첫 실행). FSM 명령에 주입한 진동은 포함하지 않는다.')
    table = {sc: {n: v['rms_residual_um'] for n, v in summary['scenarios'][sc].items()} for sc, _ in scenarios}
    names = [n for n in ('pid', 'frozen_pinn') if any(n in table[sc] for sc in table)]
    idx.save(F.fig_pointing_summary(scenarios, table, names, banner, 'RMS residual from PSD frames [µm]'),
             'E02_pointing_summary.png', 'PSD 프레임(50 Hz)으로 측정한 RMS 잔차. PSD 잡음을 포함하며 프레임 사이의 움직임은 보이지 않는다.')
    panels = [(label, {n: (data[f'spectra/{sc}/{n}/f'], data[f'spectra/{sc}/{n}/amp']) for n in names
                       if f'spectra/{sc}/{n}/f' in data}) for sc, label in scenarios]
    idx.save(F.fig_spectra(panels[:3], banner=banner), 'E03_residual_spectra.png', '측정 잔차 스펙트럼(PSD 프레임 기준, 25 Hz까지).')
    link_scenarios = [(sc, label) for sc, label in scenarios if f'comm/{sc}/pid/r75/ber64' in data][:3]
    if link_scenarios:
        curves = _curves(data, link_scenarios, names)
        ideal = {m: data[f'comm/ideal/{m}'] for m in ('ber64', 'rate_bps')}
        idx.save(F.fig_link_curves(data['comm/snr_db'], curves, ideal, link_scenarios, 'ber64', link, banner),
                 'E04_ber_vs_snr.png', '측정 잔차를 HAPS 수신기에 넣은 64-QAM BER 대 SNR.')
        idx.save(F.fig_link_curves(data['comm/snr_db'], curves, ideal, link_scenarios, 'rate_bps', link, banner),
                 'E05_rate_vs_snr.png', '측정 잔차를 HAPS 수신기에 넣은 적응 QAM 데이터율 대 SNR.')
        idx.save(F.fig_outage(summary['comm'], scenarios, banner), 'E06_outage.png', '측정 잔차 기반 동작점 outage.')
    if 'demo/pid/residual_m' in data:
        traces = {n: F.Trace(data[f'demo/{n}/t_s'], data[f'demo/{n}/residual_m'], receivers['r75'], hold=True)
                  for n in names if f'demo/{n}/t_s' in data}
        note = ('Measured PSD frames (50 Hz), each held until the next frame; symbols from the HAPS-equivalent 75 µm '
                'detector at FEC threshold + 3 dB, before gain correction.')
        idx.save(F.fig_acquisition(traces, link, ev['comm'], banner, note, 'Time since the first PSD frame [ms] (log)'),
                 'E07_acquisition.png', '측정한 획득 과정과 그에 대응하는 HAPS 등가 constellation. PSD 프레임 사이 값은 '
                 '보간하지 않고 직전 프레임 값을 유지한다.', dpi=150)
    if summary.get('usrp'):
        idx.save(F.fig_usrp(summary['usrp'], banner), 'E08_bench_usrp_link.png',
                 '벤치에서 USRP로 직접 측정한 EVM/BER. 렌즈 없는 벤치 검출기라 지향 잔차에 둔감할 것으로 예상된다.')
    idx.write()
    empirical_tables(result_dir)
    return idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--analytical')
    group.add_argument('--empirical')
    parser.add_argument('--formats', nargs='+', default=list(F.FORMATS), choices=['png', 'pdf', 'eps', 'svg'])
    args = parser.parse_args()
    if args.analytical:
        idx = analytical_figures(ROOT/args.analytical, args.formats)
    else:
        idx = empirical_figures(ROOT/args.empirical, args.formats)
    print(f'{len(idx.items)} figures -> {idx.folder}')


if __name__ == '__main__':
    main()
