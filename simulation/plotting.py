"""Measured power, paired improvements and receiver-plane visualizations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

COLORS = {"Frozen-PINN": "#2166AC", "PID": "#E68613", "Linear MPC": "#4D9A75",
          "Diff-SSFM oracle": "#A45D92", "SSFM oracle (CPU-FD)": "#A45D92", "No control": "#9B9FA5"}
LABELS = {"Diff-SSFM oracle": "SSFM oracle", "SSFM oracle (CPU-FD)": "SSFM oracle (FD)"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "font.family": "DejaVu Sans"})


def _color(name):
    return COLORS.get(name, "#555555")


def _label(name):
    return LABELS.get(name, name)


def _controlled_solver_results(results):
    controlled = {k: v for k, v in results.items() if k != "No control"}
    return controlled or results


def _style(ax, *, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9CDD2")
    ax.tick_params(color="#C9CDD2", labelcolor="#42464C", length=3)
    ax.grid(axis=grid, color="#E6E8EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def _legend(fig, names, *, y=0.96):
    handles = [Line2D([0], [0], color=_color(n), lw=2,
                     ls="--" if n == "No control" else "-", label=_label(n)) for n in names]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y),
               ncol=min(len(names), 5), frameon=False, fontsize=9, handlelength=2)


def _save(fig, path, *, legend=False):
    fig.tight_layout(rect=(0, 0.02, 1, 0.86 if legend else 0.92), pad=1.5)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _times(records):
    return np.array([r.get("time_sec", r["interval"]) for r in records], dtype=float)


def _values(payload):
    return np.asarray([r["display_metric"] for r in payload["records"]], dtype=float)


def _gains(results):
    if "No control" not in results:
        return {}
    baseline = _values(results["No control"])
    gains = {}
    for name, payload in _controlled_solver_results(results).items():
        values = _values(payload)
        gains[name] = np.divide(values-baseline, baseline, out=np.full_like(values, np.nan), where=baseline>0)*100
    return gains


def _draw_power(ax, results):
    # No markers on every interval: the time series remains legible at 100+ points.
    for name, payload in results.items():
        records = payload["records"]
        ax.plot(_times(records), _values(payload)*1e3, color=_color(name),
                lw=1.25, ls="--" if name == "No control" else "-", alpha=.9,
                marker="o" if len(records) == 1 else None, ms=4)
    ax.set(xlabel="Time [s]", ylabel="Collected power [mW]", ylim=(0, None))
    _style(ax)


def _draw_gain_distribution(ax, gains):
    for name, values in gains.items():
        values = np.sort(values[np.isfinite(values)])
        if len(values):
            ax.step(values, np.arange(1, len(values)+1)/len(values), where="post",
                    color=_color(name), lw=1.8)
            if len(values) == 1:
                ax.plot(values, [1], "o", color=_color(name), ms=4)
    ax.axvline(0, color="#A9ADB2", lw=.8, ls="--")
    ax.set(xlabel="Power change from No control [%]", ylabel="Fraction of intervals",
           ylim=(0, 1.04), title="Controller differences on the same channel")
    _style(ax)


def plot_objective_comparison(output_dir, objective, solver_results):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gains = _gains(solver_results)
    if gains:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.3, 1]})
        _draw_power(axes[0], solver_results)
        axes[0].set_title("Absolute power: channel variation is shared")
        _draw_gain_distribution(axes[1], gains)
    else:
        fig, ax = plt.subplots(figsize=(8, 4.7))
        _draw_power(ax, solver_results)
    fig.suptitle("Communication-detector power", x=.06, ha="left", fontsize=14, fontweight="medium")
    _legend(fig, solver_results, y=.92)
    _save(fig, output_dir/"objective_comparison.png", legend=True)
    fig, ax = plt.subplots(figsize=(9, 4.7))
    _draw_power(ax, solver_results)
    fig.suptitle("Collected power · all controllers", x=.07, ha="left", fontsize=14)
    _legend(fig, solver_results, y=.92)
    _save(fig, output_dir/"objective_comparison_all.png", legend=True)
    if gains:
        fig, (ax, dist) = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.35, 1]})
        first = next(iter(solver_results.values()))["records"]
        for name, values in gains.items():
            ax.plot(_times(first), values, lw=1, color=_color(name), alpha=.85,
                    marker="o" if len(first) == 1 else None, ms=4)
        ax.axhline(0, color="#A9ADB2", lw=.8, ls="--")
        ax.set(xlabel="Time [s]", ylabel="Power change from No control [%]", title="Per interval · no smoothing")
        _style(ax)
        names = list(gains)
        data = [gains[n][np.isfinite(gains[n])] for n in names]
        boxes = dist.boxplot(data, vert=False, patch_artist=True, widths=.45, showmeans=True,
            showfliers=False, meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="#333333", markersize=4),
            medianprops=dict(color="#333333", linewidth=1.1), whiskerprops=dict(color="#9B9FA5"), capprops=dict(color="#9B9FA5"))
        for patch, name in zip(boxes["boxes"], names):
            patch.set(facecolor=_color(name), edgecolor=_color(name), alpha=.65)
        dist.set_yticks(np.arange(1, len(names)+1), [_label(n) for n in names])
        dist.invert_yaxis()
        dist.axvline(0, color="#A9ADB2", lw=.8, ls="--")
        dist.set(xlabel="Power change [%]", title="Distribution · dot = mean")
        _style(dist, grid="x")
        # Keep categorical tick positions after applying the numeric-axis theme.
        dist.set_yticks(np.arange(1, len(names)+1), [_label(n) for n in names])
        fig.suptitle("Receiver steering contribution", x=.06, ha="left", fontsize=14)
        _legend(fig, gains, y=.92)
        _save(fig, output_dir/"power_gain_comparison.png", legend=True)
    return output_dir/"objective_comparison.png"


def plot_runtime_comparison(output_dir, solver_results):
    names = list(solver_results)
    totals, prep, queries = [], [], []
    for name in names:
        records = solver_results[name]["records"]
        totals.append(np.mean([r["runtime_sec"] for r in records])*1e3)
        prep.append(np.mean([r.get("diagnostics", {}).get("atmosphere_prepare_time_sec", 0) for r in records])*1e3)
        queries.append(np.mean([r.get("diagnostics", {}).get("query_runtime_sec", 0) for r in records])*1e3)
    totals, prep, queries = map(np.asarray, (totals, prep, queries))
    # Older files may have independently sampled clocks; do not draw negative overhead.
    prep = np.minimum(prep, totals)
    queries = np.minimum(queries, np.maximum(0, totals-prep))
    other = np.maximum(0, totals-prep-queries)
    fig, (ax, parts) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(names))
    positive = totals[totals>0]
    floor = min(positive.min()/3, .01) if len(positive) else .001
    upper = max(positive.max()*6, 30) if len(positive) else 30
    for i, (name, value) in enumerate(zip(names, totals)):
        if value > 0:
            ax.barh(i, value, color=_color(name), height=.5)
            label = f"{value:,.0f}" if value >= 100 else f"{value:.3g}"
            ax.text(value*1.15, i, f"{label} ms", va="center", fontsize=9)
        else:
            ax.text(floor*1.3, i, "Held command (0)", va="center", color="#777777", fontsize=9)
    ax.set_xscale("log")
    ax.set(xlim=(floor, upper), xlabel="Mean online runtime [ms] · log scale", title="Atmosphere prediction + control")
    _style(ax, grid="x")
    ax.set_yticks(y, [_label(n) for n in names]); ax.set_ylim(len(names)-.5, -.5)
    starts = np.zeros(len(names))
    for values, label, color in [(prep, "Atmosphere", "#527AA3"), (queries, "Steering queries", "#72A68D"), (other, "Other control", "#C9CDD2")]:
        width = np.divide(values, totals, out=np.zeros_like(totals), where=totals>0)*100
        parts.barh(y, width, left=starts, height=.5, label=label, color=color)
        starts += width
    parts.set(xlim=(0, 100), xlabel="Fraction of online runtime [%]", title="Where the time goes")
    _style(parts, grid="x"); parts.set_yticks(y, []); parts.set_ylim(len(names)-.5, -.5)
    fig.legend(*parts.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.72, .9),
               ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Runtime breakdown", x=.06, ha="left", fontsize=14)
    fig.text(.06, .02, "Offline basis setup, simulated sensor acquisition and truth evaluation are excluded.", fontsize=8, color="#666666")
    return _save(fig, Path(output_dir)/"runtime_comparison.png", legend=True)


def plot_centroid_trajectories(output_dir, solver_results):
    names = list(solver_results)
    cols = min(3, len(names)); rows = int(np.ceil(len(names)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3.3*rows), squeeze=False)
    all_points = [np.asarray([r["actual_centroid"] for r in v["records"]], dtype=float)*1e6 for v in solver_results.values()]
    finite = np.concatenate(all_points).ravel(); finite = finite[np.isfinite(finite)]
    half = max(5., np.max(abs(finite))*1.2) if len(finite) else 5.
    reference = np.asarray(next(iter(solver_results.values()))["records"][0]["target"])*1e6
    for ax, name, points in zip(axes.flat, names, all_points):
        ax.scatter(points[:, 0], points[:, 1], s=14, color=_color(name), alpha=.55, linewidths=0)
        ax.plot(*reference, marker="+", ms=11, mew=1.5, color="#333333")
        ax.set(title=_label(name), xlabel="PSD x [µm]", ylabel="PSD y [µm]", xlim=(-half, half), ylim=(-half, half))
        _style(ax, grid="both"); ax.set_aspect("equal")
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle("PSD positions · one point per interval", x=.06, ha="left", fontsize=14)
    fig.text(.06, .02, "+ Calibrated reference. PSD position is a diagnostic; the objective is communication power.", fontsize=8, color="#666666")
    return _save(fig, Path(output_dir)/"centroid_trajectories.png")


def _view_extent(system):
    o = system.cfg.optical
    half = np.ceil(o.detector_view_half_width/o.detector_sampling)*o.detector_sampling
    return [(o.detector_center[0]-half)*1e6, (o.detector_center[0]+half)*1e6,
            (o.detector_center[1]-half)*1e6, (o.detector_center[1]+half)*1e6]


def _detector_circle(ax, system):
    o = system.cfg.optical
    ax.add_patch(Circle(np.asarray(o.detector_center)*1e6, o.aperture_radius*1e6,
                        fill=False, ec="white", lw=1.2))
    ax.set(xlabel="Detector x [µm]", ylabel="Detector y [µm]")
    ax.tick_params(length=2, labelsize=8)


def _view_limits(system, images):
    o = system.cfg.optical; extent = _view_extent(system)
    center = np.asarray(o.detector_center)*1e6; half = 2*o.aperture_radius*1e6
    for intensity in images:
        y, x = np.unravel_index(np.argmax(intensity), intensity.shape)
        peak = [extent[0]+x*(extent[1]-extent[0])/(intensity.shape[1]-1),
                extent[2]+y*(extent[3]-extent[2])/(intensity.shape[0]-1)]
        half = max(half, max(abs(np.asarray(peak)-center))+o.aperture_radius*1e6)
    half = min(half, (extent[1]-extent[0])/2)
    return (center[0]-half, center[0]+half), (center[1]-half, center[1]+half)


def plot_receiver_snapshots(output_dir, solver_results, system, reduced=None):
    names = list(solver_results)
    records = next(iter(solver_results.values()))["records"]
    k = int(records[-1]["interval"])
    if reduced is None:
        reduced = system.reduce_field(system.atmospheric_field(k))
    images = [abs(system.detector_field(reduced, solver_results[n]["records"][-1]["theta"], view=True))**2 for n in names]
    cols = min(3, len(names)); rows = int(np.ceil(len(names)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6*cols, 3.3*rows), squeeze=False, layout="constrained")
    maximum = max(float(im.max()) for im in images); xlim, ylim = _view_limits(system, images)
    for ax, name, intensity in zip(axes.flat, names, images):
        im = ax.imshow(intensity, extent=_view_extent(system), origin="lower", vmin=0, vmax=maximum, cmap="magma")
        _detector_circle(ax, system); ax.set(xlim=xlim, ylim=ylim)
        power = solver_results[name]["records"][-1]["communication_power_w"]
        ax.set_title(f"{_label(name)}\n{power*1e3:.5f} mW", fontsize=10)
    for ax in list(axes.flat)[len(names):]: ax.set_visible(False)
    fig.colorbar(im, ax=list(axes.flat)[:len(names)], label="Irradiance [W/m²]", shrink=.8, pad=.025)
    fig.suptitle(f"Communication detector · interval {k} · circle = active area", fontsize=13)
    path = Path(output_dir)/"receiver_xy_comparison.png"; fig.savefig(path, dpi=180); plt.close(fig)
    return path


def save_comparison_gif(output_dir, objective, solver_results, fps=8, system=None, frames=None):
    if system is None: raise ValueError("Receiver XY animation requires the system model")
    if fps < 1: raise ValueError("GIF fps must be positive")
    focus = "Frozen-PINN" if "Frozen-PINN" in solver_results else next(iter(solver_results))
    records = solver_results[focus]["records"]
    images = frames
    if images is None:
        images = []
        for i, r in enumerate(records):
            reduced = system.reduce_field(system.atmospheric_field(r["interval"]))
            images.append((abs(system.detector_field(reduced, r["theta"], view=True))**2).astype(np.float32))
            if (i+1) % 20 == 0: print(f"GIF fields: {i+1}/{len(records)}", flush=True)
    if len(images) != len(records): raise ValueError("GIF frame/record count differs")
    fig = plt.figure(figsize=(11.6, 5.1))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], hspace=.48, wspace=.35)
    left = fig.add_subplot(grid[:, 0]); power_ax = fig.add_subplot(grid[0, 1]); gain_ax = fig.add_subplot(grid[1, 1])
    maximum = max(max(float(v.max()) for v in images), 1e-30)
    # A fixed, labelled logarithmic scale keeps low-power channels visible
    # without normalizing away the physical power variation between frames.
    im = left.imshow(images[0], origin="lower", extent=_view_extent(system), cmap="magma",
                     norm=LogNorm(vmin=maximum*1e-4, vmax=maximum))
    _detector_circle(left, system); xlim, ylim = _view_limits(system, images); left.set(xlim=xlim, ylim=ylim)
    fig.colorbar(im, ax=left, fraction=.045, pad=.03, label="Irradiance [W/m²] · log scale")
    left.set_title(f"{focus} · detector plane", fontsize=11)
    gains = _gains(solver_results); times = _times(records)
    lines = {}; gain_lines = {}
    ymax = max(float(_values(v).max())*1e3 for v in solver_results.values())
    for name in solver_results:
        lines[name], = power_ax.plot([], [], lw=1.25, color=_color(name), ls="--" if name=="No control" else "-",
                                    marker="o", ms=3, markevery=[-1])
        if name in gains:
            gain_lines[name], = gain_ax.plot([], [], lw=1.2, color=_color(name), marker="o", ms=3, markevery=[-1])
    end = times[-1] if len(times)>1 else times[0]+system.cfg.tracking.control_interval_sec
    power_ax.set(xlim=(times[0], end), ylim=(0, max(ymax*1.08, 1e-9)), ylabel="Power [mW]", title="Actual collected power")
    if gains:
        vals = np.concatenate(list(gains.values())); vals = vals[np.isfinite(vals)]
        low, high = (float(vals.min()), float(vals.max())) if len(vals) else (-1., 1.)
        pad = max((high-low)*.12, .05)
        gain_ax.set(ylim=(min(low-pad, 0), max(high+pad, 0)))
    gain_ax.axhline(0, color="#A9ADB2", lw=.8, ls="--")
    gain_ax.set(xlim=(times[0], end), xlabel="Time [s]", ylabel="Δpower [%]", title="Relative to No control")
    _style(power_ax); _style(gain_ax)
    _legend(fig, solver_results, y=.99)
    status = fig.text(.5, .025, "", ha="center", fontsize=9, color="#444444")
    def update(frame):
        r = records[frame]; im.set_data(images[frame])
        for name, payload in solver_results.items():
            lines[name].set_data(times[:frame+1], _values(payload)[:frame+1]*1e3)
            if name in gains: gain_lines[name].set_data(times[:frame+1], gains[name][:frame+1])
        theta = np.asarray(r["theta"])*1e6
        status.set_text(f"t = {r['time_sec']:.2f} s   |   interval {r['interval']}   |   FSM ({theta[0]:+.0f}, {theta[1]:+.0f}) µrad   |   P = {r['communication_power_w']*1e3:.5f} mW")
        return (im, status, *lines.values(), *gain_lines.values())
    fig.subplots_adjust(left=.07, right=.97, top=.84, bottom=.17)
    animation = FuncAnimation(fig, update, frames=len(records), interval=1000/fps, blit=False)
    path = Path(output_dir)/"comparison.gif"
    temporary = path.with_name("comparison.partial.gif")
    animation.save(temporary, writer=PillowWriter(fps=fps))
    temporary.replace(path)
    plt.close(fig)
    return path


def plot_aggregate_objective(output_dir, objective, aggregate_results):
    for filename, results in [("aggregate_objective_comparison.png", _controlled_solver_results(aggregate_results)),
                              ("aggregate_objective_comparison_all.png", aggregate_results)]:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for name, payload in results.items():
            mean = np.asarray(payload["mean_by_interval"])*1e3; std = np.asarray(payload["std_by_interval"])*1e3
            k = np.arange(len(mean)); ax.plot(k, mean, lw=1.5, color=_color(name))
            ax.fill_between(k, np.maximum(0, mean-std), mean+std, color=_color(name), alpha=.08)
        ax.set(xlabel="PAT interval", ylabel="Collected power [mW]"); _style(ax)
        fig.suptitle("Repeated experiments · mean ± descriptive SD", x=.07, ha="left", fontsize=13)
        _legend(fig, results, y=.92); _save(fig, Path(output_dir)/filename, legend=True)
    return Path(output_dir)/"aggregate_objective_comparison.png"
