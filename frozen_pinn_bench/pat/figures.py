"""Analytical and empirical figures in one style, numbered, with an index (FIGURES.md).

Both branches call the same figure functions with plain arrays:
  analytical  results/analytical/<run>/   bench-model simulation + HAPS-equivalent link
  empirical   results/empirical/<session>/ measured bench logs + HAPS-equivalent link
Colour = controller family (validated all-pairs: blue/aqua/orange); references are neutral.
Every figure is written as PNG (preview), PDF (pdflatex) and EPS (latex/dvips, IEEE submission).
EPS has no transparency, so scatter colours are pre-blended with the surface instead of using
alpha, and dense scatters are rasterized (text and axes stay vector); all formats look the same.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['hatch.linewidth'] = 0.8
# Embed TrueType (Type 42) fonts: editable text, and no Type 3 fonts (flagged by IEEE PDF eXpress).
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np
from scipy.ndimage import shift as image_shift

from pat.comm import ber_qam, simulate_symbols
from pat.qam import constellation

SURFACE, INK, INK2, GRID, BAND, IDEAL = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#f0efec', '#b5b4af'
BLUE, AQUA, ORANGE = '#2a78d6', '#1baf7a', '#eb6834'
STYLE = {
    'hold': dict(color='#9a9994', label='No control', lw=1.0),
    'pid': dict(color=BLUE, label='PID', lw=1.5),
    'frozen_pinn': dict(color=AQUA, label='Frozen-PINN', lw=1.6),
    'frozen_online': dict(color=AQUA, label='Frozen-PINN w/o FSM-ODE command', lw=1.2, ls='-.'),
    'frozen_offline': dict(color=AQUA, label='Frozen-PINN w/o online refit', lw=1.2, ls=':'),
    'linear_online': dict(color=ORANGE, label='Linear predictor (M = 0)', lw=1.3),
    'linear_offline': dict(color=ORANGE, label='Linear, w/o online refit', lw=1.2, ls=':'),
    'oracle': dict(color=INK2, label='Oracle bound', lw=1.2, ls='--'),
}
FORMATS = ('png', 'pdf', 'eps')
DETECTORS = [('r75', '75 µm detector (PDA015C2 class)'), ('r25', '25 µm detector (high-speed)')]


def _label(name):
    return STYLE[name]['label']


def _blend(colour, weight):
    """Opaque colour equal to `colour` at opacity `weight` over the surface (EPS has no alpha)."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    return tuple(weight*rgb+(1-weight)*np.array(matplotlib.colors.to_rgb(SURFACE)))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#bdbcb7')
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def new_figure(*shape, **kwargs):
    fig, axes = plt.subplots(*shape, facecolor=SURFACE, **kwargs)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax)
    return fig, axes


def heading(fig, text, sub, banner=None):
    fig.text(0.01, 0.985, text, ha='left', va='top', fontsize=11.5, color=INK, weight='bold')
    if banner:
        # Short tag on the title line; the full banner sentence goes into FIGURES.md.
        fig.text(0.99, 0.985, banner.split(' - ')[0], ha='right', va='top', fontsize=8.5, color=INK, weight='bold',
                 bbox=dict(facecolor=BAND, edgecolor=INK2, boxstyle='round,pad=0.3', linewidth=0.6))
    fig.text(0.01, 0.945, sub, ha='left', va='top', fontsize=8.5, color=INK2)


class FigureIndex:
    """Saves each figure in every format and writes FIGURES.md next to the figures."""
    def __init__(self, folder, title, banner=None, formats=FORMATS):
        self.folder, self.title, self.banner, self.items = Path(folder), title, banner, []
        self.formats = tuple(formats)
        self.folder.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name, caption, dpi=160):
        stem = Path(name).stem
        for ext in self.formats:
            # dpi sets the PNG resolution and that of rasterized parts in PDF/EPS.
            fig.savefig(self.folder/f'{stem}.{ext}', dpi=dpi, facecolor=SURFACE)
        plt.close(fig)
        self.items.append((stem, caption))

    def write(self):
        lines = [f'# {self.title}', '']
        if self.banner:
            lines += [f'> {self.banner}', '']
        for stem, caption in self.items:
            files = ' · '.join(f'[{ext.upper()}]({stem}.{ext})' for ext in self.formats)
            preview = [f'![{stem}]({stem}.png)', ''] if 'png' in self.formats else []
            lines += [f'## {stem}', '', caption, '', files, '']+preview
        (self.folder/'FIGURES.md').write_text('\n'.join(lines))


# ---------------------------------------------------------------- shared figure functions

def fig_capture_function(curves, banner=None):
    fig, ax = new_figure(1, 1, figsize=(8, 4.2))
    for label, (offsets, eta, style) in curves.items():
        ax.plot(offsets*1e6, eta, label=label, **style)
    ax.axhline(0.5, color=INK2, lw=0.7, ls=':')
    ax.text(2, 0.52, 'half power', fontsize=7.5, color=INK2)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('Beam offset on the communication detector [µm]', fontsize=8.5)
    ax.set_ylabel('Collected power [aligned = 1]', fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    heading(fig, 'Why pointing matters only for small detectors',
            'HAPS receiver (focused, PDF Table I) for two detector sizes, and the unfocused bench detector.', banner)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


RAIL_NOTE = ('Rail position; vibration families also carry software-injected FSM-plane motion. '
             'HAPS lateral position at 20 km = rail × 1000.')


def fig_rail_profiles(profiles, travel_mm, banner=None, note=RAIL_NOTE):
    fig, axes = new_figure(len(profiles), 1, figsize=(8, 1.45*len(profiles)+1.2), sharex=True)
    for ax, (label, t, pos_mm) in zip(np.atleast_1d(axes), profiles):
        ax.plot(t, pos_mm, color=INK2, lw=1.0)
        ax.set_ylim(-travel_mm*1.05, travel_mm*1.05)
        ax.set_ylabel('mm', fontsize=8)
        ax.text(0.01, 0.92, label, transform=ax.transAxes, ha='left', va='top', fontsize=8, color=INK,
                bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5), zorder=5)
    np.atleast_1d(axes)[-1].set_xlabel('Time [s]', fontsize=8.5)
    heading(fig, 'Disturbance profiles (first episode of each test family)', note, banner)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def fig_pointing_summary(scenarios, table, names, banner=None, ylabel='RMS residual at the detector [µm]'):
    fig, ax = new_figure(1, 1, figsize=(10, 4.6))
    width = 0.8/len(names)
    x = np.arange(len(scenarios))
    for i, name in enumerate(names):
        values = [table[s].get(name, np.nan) for s, _ in scenarios]
        pos = x+(i-(len(names)-1)/2)*width
        colour = STYLE[name]['color']
        ablation = name in ('frozen_offline', 'frozen_online', 'linear_offline')
        ax.bar(pos, values, width*0.86, color=SURFACE if ablation else colour, edgecolor=colour if ablation else SURFACE,
               hatch={'frozen_offline': '////', 'frozen_online': '....', 'linear_offline': '////'}.get(name),
               linewidth=1.0 if ablation else 0.8, label=_label(name))
        if name in ('pid', 'frozen_pinn'):
            for p, v in zip(pos, values):
                ax.text(p, v, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, color=INK)
    ax.set_xticks(x, [label for _, label in scenarios], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=7.5, frameon=False)
    sub = 'Same profiles and sensor information for every controller'
    if any(n in ('frozen_offline', 'frozen_online', 'linear_offline') for n in names):
        sub += '; hatched bars are ablations'
    heading(fig, 'Pointing residual on held-out profiles', sub+'.', banner)
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    return fig


def fig_rms_vs_severity(severity, vibration_um_per_scale, banner=None):
    fig, axes = new_figure(1, 2, figsize=(11, 4.4))
    panels = (('sweep', 'HAPS lateral speed [m/s]  (= rail speed [mm/s])', lambda v: v*1e3),
              ('vibration', 'Injected vibration, mean line amplitude at the detector [µm]', lambda v: v*vibration_um_per_scale))
    for ax, (kind, xlabel, xmap) in zip(axes, panels):
        rows = severity[kind]
        xs = [xmap(r['value']) for r in rows]
        for name in ('pid', 'frozen_pinn', 'oracle'):
            ax.plot(xs, [r['results'][name]['rms_residual_um'] for r in rows], marker='o', ms=4,
                    **dict(STYLE[name], label=_label(name)))
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel('RMS residual [µm]', fontsize=8.5)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc='upper left', fontsize=8, frameon=False)
    heading(fig, 'Pointing residual versus disturbance severity',
            'Six held-out episodes per point; PID re-tuned for each disturbance family.', banner)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_spectra(panels, nyquist_hz=25.0, banner=None):
    fig, axes = new_figure(1, len(panels), figsize=(5.6*len(panels), 4.3), sharey=True)
    for ax, (label, curves) in zip(np.atleast_1d(axes), panels):
        for name, (f, amp) in curves.items():
            keep = f > 0
            ax.loglog(f[keep], amp[keep]*1e6, **dict(STYLE[name], label=_label(name)))
        ax.axvline(nyquist_hz, color=INK2, lw=0.8, ls=':')
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel('Frequency [Hz]', fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel('Residual x spectrum [µm/√Hz]', fontsize=8.5)
    np.atleast_1d(axes)[0].legend(fontsize=7.5, frameon=False, loc='lower left')
    heading(fig, 'Where the residual lives', 'Dotted line: PSD Nyquist frequency (25 Hz). '
            'The oracle line is the mirror jitter floor.', banner)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_tracking_window(t, rail_mm, residuals, window, label, banner=None):
    lo, hi = np.searchsorted(t, window)
    fig, (top, bottom) = new_figure(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3]))
    top.plot(t[lo:hi], rail_mm[lo:hi], color=INK2, lw=1.2)
    top.set_ylabel('Rail position [mm]', fontsize=8.5)
    for name, xy in residuals.items():
        bottom.plot(t[lo:hi], xy[lo:hi, 0]*1e6, **dict(STYLE[name], label=_label(name)))
    bottom.set_ylabel('Residual x [µm]', fontsize=8.5)
    bottom.set_xlabel('Time [s]', fontsize=8.5)
    bottom.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=8, frameon=False, ncol=3)
    heading(fig, f'Residual during {label}', 'PSD frames every 20 ms; Frozen-PINN commands every 2 ms.', banner)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def fig_link_curves(snr_db, curves, ideal, scenarios, metric, link, banner=None):
    scale = 1e-6 if metric == 'rate_bps' else 1.0
    ylabel = {'ber64': '64-QAM BER', 'rate_bps': 'Data rate [Mb/s]'}[metric]
    fig, axes = new_figure(2, len(scenarios), figsize=(3.85*len(scenarios), 6.8), sharex=True)
    op_db = 10*np.log10(link.operating_snr)
    for i, (rk, rlabel) in enumerate(DETECTORS):
        for j, (sc, slabel) in enumerate(scenarios):
            ax = axes[i, j]
            ax.plot(snr_db, ideal[metric]*scale, color=IDEAL, lw=1.4, ls=':', label='No pointing error')
            for name in ('pid', 'frozen_pinn', 'oracle'):
                if name in curves[sc]:
                    ax.plot(snr_db, curves[sc][name][rk][metric]*scale, **dict(STYLE[name], label=_label(name)))
            ax.axvline(op_db, color=INK2, lw=0.7, ls=':')
            if metric == 'ber64':
                ax.set_yscale('log')
                ax.set_ylim(1e-6, 0.5)
                ax.axhline(link.fec, color=INK2, lw=0.7, ls='--')
            if i == 0:
                ax.set_title(slabel, fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f'{rlabel}\n{ylabel}', fontsize=8.5)
            if i == 1:
                ax.set_xlabel('Aligned SNR [dB]', fontsize=8.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=8, frameon=False)
    what = 'bit-error rate' if metric == 'ber64' else 'adaptive-QAM data rate (10 Mbaud, 82.4% payload)'
    heading(fig, f'HAPS-equivalent {what} versus link SNR',
            'Dotted vertical: operating point (FEC threshold + 3 dB); dashed horizontal: HD-FEC 3.8e-3. '
            'SNR scales with received power squared (IM/DD).', banner)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    return fig


def fig_outage(op, scenarios, banner=None):
    fig, axes = new_figure(1, 2, figsize=(11, 4.6))
    names = ('pid', 'frozen_pinn', 'oracle')
    width = 0.26
    x = np.arange(len(scenarios))
    for ax, (rk, rlabel) in zip(axes, DETECTORS):
        for i, name in enumerate(names):
            values = [op[sc][name][rk]['outage64']*100 if name in op[sc] else np.nan for sc, _ in scenarios]
            pos = x+(i-1)*width
            ax.bar(pos, values, width*0.88, color=STYLE[name]['color'], edgecolor=SURFACE, linewidth=0.8, label=_label(name))
            if name != 'oracle':
                for p, v in zip(pos, values):
                    if np.isfinite(v):
                        ax.text(p, v, f'{v:.1f}' if v >= 0.1 else f'{v:.2f}', ha='center', va='bottom', fontsize=6.5, color=INK)
        ax.set_xticks(x, [label for _, label in scenarios], fontsize=7.5)
        ax.set_ylabel('Outage [% of time]', fontsize=8.5)
        ax.set_title(rlabel, fontsize=9, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    heading(fig, '64-QAM outage at the operating point', 'Fraction of time below the HD-FEC threshold, '
            'HAPS-equivalent receiver, aligned SNR = threshold + 3 dB. Lower is better.', banner)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    return fig


class Trace:
    """Residual time series on a receiver. hold=True for sampled data (PSD frames): the value is held
    until the next sample instead of being interpolated, so nothing is invented between frames."""
    def __init__(self, t, residual, receiver, hold=False):
        self.t, self.residual = np.asarray(t), np.asarray(residual, dtype=float)
        self.receiver, self.hold = receiver, hold
        self.eta = receiver.eta(self.residual)

    def eta_of_t(self, times):
        if self.hold:
            return self.eta[np.clip(np.searchsorted(self.t, times, side='right')-1, 0, len(self.t)-1)]
        return np.interp(times, self.t, self.eta)


def _scatter(ax, symbols, colour, head, limit=1.6):
    ideal = constellation()
    ax.scatter(ideal.real, ideal.imag, marker='+', color=INK2, s=16, linewidths=0.7, zorder=3)
    ax.scatter(np.clip(symbols.real, -limit, limit), np.clip(symbols.imag, -limit, limit), s=4,
               color=_blend(colour, 0.6), linewidths=0, rasterized=True)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(head, fontsize=8.5, color=INK)
    ax.tick_params(labelsize=7)


ACQ_NOTE = ('Received 64-QAM symbols before gain correction (aligned = unit energy), HAPS-equivalent 75 µm detector, '
            'aligned SNR = FEC threshold + 3 dB.')


def fig_acquisition(traces, link, comm, banner=None, note=ACQ_NOTE, xlabel='Time since start [ms] (log)'):
    windows = [(0, 10), (10, 25), (25, 40), (40, 70), (70, 150), (150, 400), (1000, 3000)]
    names = [n for n in ('pid', 'frozen_pinn') if n in traces]
    fig = plt.figure(figsize=(14, 7.8), facecolor=SURFACE)
    gs = GridSpec(1+len(names), len(windows), figure=fig, height_ratios=[1.0]+[1.15]*len(names), hspace=0.45, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    style_axes(ax)
    for name in names:
        tr = traces[name]
        ax.plot(np.maximum(tr.t*1e3, 0.5), np.hypot(tr.residual[:, 0], tr.residual[:, 1])*1e6,
                drawstyle='steps-post' if tr.hold else 'default', **dict(STYLE[name], label=_label(name)))
    ax.axhline(75, color=INK2, lw=0.8, ls='--')
    ax.text(0.55, 78, 'detector radius 75 µm', fontsize=7.5, color=INK2, va='bottom')
    for i, (a, b) in enumerate(windows):
        ax.axvspan(max(a, 0.5), b, color=BAND if i % 2 == 0 else SURFACE, zorder=0)
        ax.text(np.sqrt(max(a, 0.5)*b), 228, chr(65+i), ha='center', fontsize=8, color=INK2)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 3000)
    ax.set_ylim(0, 245)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel('Spot offset [µm]', fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc='center right')
    rng = np.random.default_rng(5)
    for row, name in enumerate(names, start=1):
        tr = traces[name]
        for col, (a, b) in enumerate(windows):
            ax = fig.add_subplot(gs[row, col])
            style_axes(ax)
            times = np.sort(rng.uniform(a*1e-3, b*1e-3, 500))
            sym = simulate_symbols(tr.eta_of_t, times, link.operating_snr, comm['frame_s'], comm['pilots'], rng)
            _scatter(ax, sym['raw'], STYLE[name]['color'], f'{chr(65+col)}: {a}-{b} ms\npower {tr.eta_of_t(times).mean():.2f}')
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(_label(name), fontsize=9, color=INK)
    heading(fig, 'Constellation as PAT control brings the beam onto the detector (start 2 mm off target)', note, banner)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.03)
    return fig


def _spot(receiver, offset, half=150e-6):
    r = receiver.response
    pixels = np.asarray(offset)*receiver.scale/r.focal_dx
    image = image_shift(r.spot, (pixels[1], pixels[0]), order=1, mode='constant')
    keep = abs(r.axis) <= half
    return image[np.ix_(keep, keep)], r.axis[keep]


def _windowed(t, values, window_s):
    dt = t[1]-t[0]
    step = max(1, int(round(window_s/dt)))
    blocks = len(values)//step
    return (np.arange(blocks)+0.5)*step*dt+t[0], values[:blocks*step].reshape(blocks, step).mean(axis=1)


def fig_dashboard(traces, link, comm, t_now, label, window=0.2, banner=None):
    names = ('pid', 'frozen_pinn')
    fig = plt.figure(figsize=(13.5, 9.4), facecolor=SURFACE)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.15, 1.0, 0.8], hspace=0.55, wspace=0.38)
    rng = np.random.default_rng(int(t_now*100))
    norm = LogNorm(vmin=1e-4, vmax=1)
    stats = {}
    for col, name in enumerate(names):
        tr = traces[name]
        i_now = int(np.searchsorted(tr.t, t_now))
        lo = int(np.searchsorted(tr.t, t_now-1.0))
        w0 = int(np.searchsorted(tr.t, t_now-window))
        ax = fig.add_subplot(gs[0, col])
        style_axes(ax)
        image, axis = _spot(tr.receiver, tr.residual[i_now])
        ax.imshow(np.maximum(image/image.max(), 1e-4), origin='lower', cmap='magma', norm=norm,
                  extent=[axis[0]*1e6, axis[-1]*1e6, axis[0]*1e6, axis[-1]*1e6])
        for radius, ls in ((75, '--'), (25, '-')):
            ax.add_patch(Circle((0, 0), radius, fill=False, ec='white', lw=0.9, ls=ls))
        path = tr.residual[lo:i_now+1]*1e6
        ax.plot(path[:, 0], path[:, 1], color='#7fe0c4', lw=0.6)
        ax.set_title(f'{_label(name)}: HAPS focal plane', fontsize=9, color=INK)
        ax.set_xlabel('x [µm]', fontsize=8)
        ax.set_ylabel('y [µm]', fontsize=8)
        times = np.sort(rng.uniform(max(tr.t[0], t_now-window), t_now, 900))
        sym = simulate_symbols(tr.eta_of_t, times, link.operating_snr, comm['frame_s'], comm['pilots'], rng)
        ber = sym['bit_errors'].sum()/(6*len(sym['tx']))
        snr = link.operating_snr*tr.eta[w0:i_now+1]**2
        stats[name] = dict(sym=sym, ber=ber, rms=np.sqrt(np.mean(np.sum(tr.residual[w0:i_now+1]**2, axis=1)))*1e6,
                           power=tr.eta[w0:i_now+1].mean(), outage=np.mean(snr < link.thresholds[64])*100)
        ax = fig.add_subplot(gs[1, col])
        style_axes(ax)
        _scatter(ax, sym['equalized'], STYLE[name]['color'], f'{_label(name)}: after pilot gain correction\n'
                 f'last {window*1e3:.0f} ms, BER {ber:.1e}')
    ax = fig.add_subplot(gs[0, 2:])
    style_axes(ax)
    for name in names:
        tr = traces[name]
        lo, hi = np.searchsorted(tr.t, [t_now-1.0, t_now])
        ax.plot(tr.residual[lo:hi+1, 0]*1e6, tr.residual[lo:hi+1, 1]*1e6, color=STYLE[name]['color'], lw=0.8,
                label=_label(name))
    for radius, ls in ((75, '--'), (25, '-')):
        ax.add_patch(Circle((0, 0), radius, fill=False, ec=INK2, lw=0.8, ls=ls))
    ax.plot([0], [0], marker='+', color=INK, ms=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x [µm]', fontsize=8)
    ax.set_ylabel('y [µm]', fontsize=8)
    ax.set_title('Spot path over the last 1 s (dashed: 75 µm, solid: 25 µm detector)', fontsize=9, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    ax = fig.add_subplot(gs[1, 2])
    style_axes(ax)
    for name in names:
        raw = stats[name]['sym']['raw']
        ax.scatter(raw.real, raw.imag, s=4, color=_blend(STYLE[name]['color'], 0.6), linewidths=0,
                   label=_label(name), rasterized=True)
    ideal = constellation()
    ax.scatter(ideal.real, ideal.imag, marker='+', color=INK2, s=16, linewidths=0.7, zorder=3)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Before gain correction\n(aligned constellation = unit energy)', fontsize=8.5, color=INK)
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    lines = [f't = {t_now:.2f} s', label, '25 µm detector, aligned SNR', f'{10*np.log10(link.operating_snr):.1f} dB (FEC + 3 dB)', '']
    for name in names:
        s = stats[name]
        lines += [f'{_label(name)}  (last {window*1e3:.0f} ms)', f'  RMS offset  {s["rms"]:6.1f} µm',
                  f'  mean power  {s["power"]:6.3f}', f'  outage      {s["outage"]:5.1f} %', f'  BER         {s["ber"]:.1e}', '']
    ax.text(0, 1, '\n'.join(lines), va='top', ha='left', family='monospace', fontsize=8.5, color=INK)
    ax = fig.add_subplot(gs[2, :2])
    style_axes(ax)
    for name in names:
        tb, power = _windowed(traces[name].t, traces[name].eta, 0.1)
        ax.plot(tb, power, **dict(STYLE[name], label=_label(name)))
    ax.axvline(t_now, color=INK2, lw=0.8, ls='--')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Time [s]', fontsize=8)
    ax.set_ylabel('Received power (25 µm),\n100 ms mean', fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc='lower right', ncol=2)
    ax = fig.add_subplot(gs[2, 2:])
    style_axes(ax)
    for name in names:
        tr = traces[name]
        tb, ber = _windowed(tr.t, ber_qam(64, link.operating_snr*tr.eta**2), 0.1)
        ax.semilogy(tb, np.maximum(ber, 1e-8), **dict(STYLE[name], label=_label(name)))
    ax.axhline(link.fec, color=INK2, lw=0.7, ls='--')
    ax.axvline(t_now, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('Time [s]', fontsize=8)
    ax.set_ylabel('64-QAM BER, 100 ms windows', fontsize=8)
    heading(fig, f'PAT + 64-QAM link  |  {label}  |  t = {t_now:.2f} s',
            'Spot: HAPS-receiver focal plane (log intensity) shifted by the residual. Symbols: 64-QAM at the operating '
            'SNR, pilot block every 0.5 ms.', banner)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.06)
    return fig


def fig_usrp(usrp, banner=None):
    """Bench-measured USRP link per controller (empirical branch)."""
    fig, axes = new_figure(1, 2, figsize=(10, 4.2))
    names = [n for n in ('pid', 'frozen_pinn') if n in usrp]
    for ax, key, ylabel in zip(axes, ('evm', 'ber'), ('EVM [%]', '64-QAM BER')):
        values = [usrp[n][key]*(100 if key == 'evm' else 1) for n in names]
        ax.bar(range(len(names)), values, 0.6, color=[STYLE[n]['color'] for n in names], edgecolor=SURFACE)
        ax.set_xticks(range(len(names)), [_label(n) for n in names], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}' if key == 'evm' else f'{v:.2e}', ha='center', va='bottom', fontsize=7.5, color=INK)
    heading(fig, 'Bench-measured USRP link', 'Unfocused bench detector: expected to be insensitive to the pointing '
            'residual; the HAPS-equivalent figures carry the pointing effect.', banner)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')
