from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle


def _objective_label(objective: str) -> tuple[str, str]:
    if objective == "power":
        return "Receive power", "PAT receive-power comparison"
    if objective == "coupling":
        return "SMF coupling efficiency", "PAT coupling-efficiency comparison"
    return "Centroid tracking error [mm]", "PAT centroid-tracking comparison"


def _is_no_control(solver_name: str) -> bool:
    key = solver_name.lower().replace("-", " ").replace("_", " ").strip()
    return key in {"no control", "none"}


def _controlled_solver_results(solver_results):
    controlled = {
        name: payload
        for name, payload in solver_results.items()
        if not _is_no_control(name)
    }
    return controlled if controlled else solver_results


def _set_metric_ylim(ax, values, objective: str):
    finite = np.asarray(
        [v for v in values if np.isfinite(v)],
        dtype=float,
    )
    if finite.size == 0:
        return

    ymin = float(np.min(finite))
    ymax = float(np.max(finite))

    if abs(ymax - ymin) < 1e-15:
        pad = max(abs(ymax), 1.0) * 0.05
    else:
        pad = 0.12 * (ymax - ymin)

    lower = max(0.0, ymin - pad) if objective == "centroid" and ymin >= 0 else ymin - pad
    ax.set_ylim(lower, ymax + pad)


def _draw_objective_curves(ax, objective: str, solver_results):
    values_for_limits = []

    for solver_name, payload in solver_results.items():
        records = payload["records"]
        k = np.arange(len(records))
        values = np.asarray(
            [r["display_metric"] for r in records],
            dtype=float,
        )

        kwargs = {}
        if _is_no_control(solver_name):
            kwargs.update(
                linestyle="--",
                alpha=0.65,
                linewidth=1.6,
            )

        ax.plot(
            k,
            values,
            marker="o",
            label=solver_name,
            **kwargs,
        )
        values_for_limits.extend(values.tolist())

    ylabel, title = _objective_label(objective)
    ax.set_xlabel("PAT control interval k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return values_for_limits


def plot_objective_comparison(output_dir: Path, objective: str, solver_results):
    """Save a controller-focused graph plus a complete graph with No control."""
    output_dir.mkdir(parents=True, exist_ok=True)

    controlled = _controlled_solver_results(solver_results)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    values = _draw_objective_curves(ax, objective, controlled)
    _set_metric_ylim(ax, values, objective)
    fig.tight_layout()
    path = output_dir / "objective_comparison.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    _draw_objective_curves(ax, objective, solver_results)
    fig.tight_layout()
    all_path = output_dir / "objective_comparison_all.png"
    fig.savefig(all_path, dpi=190)
    plt.close(fig)

    return path


def plot_runtime_comparison(output_dir: Path, solver_results):
    names = []
    runtime = []
    for solver_name, payload in solver_results.items():
        records = payload["records"]
        names.append(solver_name)
        runtime.append(np.mean([r["runtime_sec"] for r in records]))
    x = np.arange(len(names))
    plt.figure(figsize=(7.4, 4.5))
    plt.bar(x, runtime)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Mean control runtime per interval [s]")
    plt.title("Online control runtime")
    plt.tight_layout()
    path = output_dir / "runtime_comparison.png"
    plt.savefig(path, dpi=190)
    plt.close()
    return path


def plot_centroid_trajectories(output_dir: Path, solver_results):
    plt.figure(figsize=(6.2, 6.0))
    first_payload = next(iter(solver_results.values()))
    target = np.asarray([r["target"] for r in first_payload["records"]])
    plt.plot(
        target[:, 0] * 1e3,
        target[:, 1] * 1e3,
        linestyle="--",
        marker=".",
        label="Rx target",
    )
    for solver_name, payload in solver_results.items():
        centroid = np.asarray([r["actual_centroid"] for r in payload["records"]])
        plt.plot(
            centroid[:, 0] * 1e3,
            centroid[:, 1] * 1e3,
            marker="o",
            label=solver_name,
        )

        if solver_name == "Frozen-PINN" and all(
            "predicted_centroid" in r for r in payload["records"]
        ):
            pred = np.asarray([r["predicted_centroid"] for r in payload["records"]])
            plt.plot(
                pred[:, 0] * 1e3,
                pred[:, 1] * 1e3,
                linestyle=":",
                marker=".",
                label="Frozen-PINN predicted",
            )
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title("2D centroid trajectories")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "centroid_trajectories.png"
    plt.savefig(path, dpi=190)
    plt.close()
    return path


def save_centroid_tracking_gifs(output_dir: Path, solver_results, system, fps: int = 3):
    """Create the centroid GIF in the original two-panel tracking style.

    Left: actual SSFM intensity map for the Frozen-PINN command, target Rx
    center, actual centroid, and Frozen-PINN predicted centroid.

    Right: target / actual / Frozen-PINN predicted centroid trajectories.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "Frozen-PINN" in solver_results:
        payload = solver_results["Frozen-PINN"]
    else:
        payload = next(iter(solver_results.values()))

    records = payload["records"]
    n_intervals = len(records)

    fields = []
    vmax = 0.0
    for r in records:
        U = system.ssfm(
            np.asarray(r["theta"], dtype=float),
            int(r["interval"]),
        )
        I = np.abs(U) ** 2
        fields.append(I)
        vmax = max(vmax, float(np.max(I)))

    extent = [
        system.x[0] * 1e3,
        system.x[-1] * 1e3,
        system.y[0] * 1e3,
        system.y[-1] * 1e3,
    ]

    fig, (ax_im, ax_tr) = plt.subplots(1, 2, figsize=(12.0, 5.4))

    im = ax_im.imshow(
        fields[0],
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=vmax if vmax > 0 else None,
    )
    plt.colorbar(im, ax=ax_im, fraction=0.046, pad=0.04, label="Optical intensity")
    ax_im.set_xlabel("x [mm]")
    ax_im.set_ylabel("y [mm]")
    ax_im.set_title("Actual receiver-plane field: objective=centroid")

    first_target = np.asarray(records[0]["target"], dtype=float) * 1e3
    roi_radius_mm = system.cfg.optical.centroid_roi_radius * 1e3
    roi_circle = Circle(
        (first_target[0], first_target[1]),
        roi_radius_mm,
        fill=False,
        edgecolor="white",
        linewidth=2.0,
    )
    ax_im.add_patch(roi_circle)

    target_im, = ax_im.plot(
        [first_target[0]], [first_target[1]],
        marker="+", markersize=14, markeredgewidth=2.5,
        color="white", linestyle="None", label="target Rx center",
    )
    actual_im, = ax_im.plot(
        [], [], marker="x", markersize=12, markeredgewidth=2.8,
        color="red", linestyle="None", label="actual beam centroid",
    )
    pred_im, = ax_im.plot(
        [], [], marker="o", markersize=8, markerfacecolor="none",
        markeredgewidth=2.2, color="cyan", linestyle="None",
        label="predicted beam centroid",
    )
    ax_im.legend(loc="upper right", fontsize=8)

    target_path = np.asarray([r["target"] for r in records], dtype=float) * 1e3
    ax_tr.plot(
        target_path[:, 0], target_path[:, 1],
        linestyle="--", marker=".", label="target",
    )
    actual_line, = ax_tr.plot([], [], marker="x", label="actual")
    pred_line, = ax_tr.plot(
        [], [], marker="o", markerfacecolor="none", label="Frozen-PINN"
    )
    ax_tr.set_xlim(-system.cfg.optical.half_width * 1e3, system.cfg.optical.half_width * 1e3)
    ax_tr.set_ylim(-system.cfg.optical.half_width * 1e3, system.cfg.optical.half_width * 1e3)
    ax_tr.set_aspect("equal")
    ax_tr.set_xlabel("x [mm]")
    ax_tr.set_ylabel("y [mm]")
    ax_tr.set_title("Objective-consistent 2D tracking\nbeam centroid")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend(loc="upper right")

    status = fig.text(0.5, 0.025, "", ha="center")

    def update(frame):
        r = records[frame]
        im.set_data(fields[frame])

        target = np.asarray(r["target"], dtype=float)
        actual = np.asarray(r["actual_centroid"], dtype=float)
        predicted = np.asarray(
            r.get("predicted_centroid", r["actual_centroid"]),
            dtype=float,
        )

        target_mm = target * 1e3
        actual_mm = actual * 1e3
        predicted_mm = predicted * 1e3

        roi_circle.center = (target_mm[0], target_mm[1])
        target_im.set_data([target_mm[0]], [target_mm[1]])
        actual_im.set_data([actual_mm[0]], [actual_mm[1]])
        pred_im.set_data([predicted_mm[0]], [predicted_mm[1]])

        actual_hist = np.asarray(
            [records[i]["actual_centroid"] for i in range(frame + 1)],
            dtype=float,
        ) * 1e3
        predicted_hist = np.asarray(
            [records[i].get("predicted_centroid", records[i]["actual_centroid"]) for i in range(frame + 1)],
            dtype=float,
        ) * 1e3

        actual_line.set_data(actual_hist[:, 0], actual_hist[:, 1])
        pred_line.set_data(predicted_hist[:, 0], predicted_hist[:, 1])

        theta = np.asarray(r["theta"], dtype=float)
        target_error = np.linalg.norm(actual - target) * 1e3
        model_error = np.linalg.norm(predicted - actual) * 1e3
        t_sec = float(r.get("time_sec", frame * system.cfg.tracking.control_interval_sec))

        status.set_text(
            f"t={t_sec:.3f} s   k={frame}   "
            f"theta=({theta[0]*1e6:+.1f},{theta[1]*1e6:+.1f}) urad   "
            f"target error={target_error:.3f} mm   "
            f"prediction-vs-actual centroid error={model_error:.3f} mm"
        )

        return (
            im, roi_circle, target_im, actual_im, pred_im,
            actual_line, pred_line, status,
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=n_intervals,
        interval=1000 / max(int(fps), 1),
        blit=False,
        repeat=True,
    )
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.14, top=0.90, wspace=0.27)

    path = output_dir / "centroid_tracking.gif"
    ani.save(path, writer=PillowWriter(fps=max(int(fps), 1)))
    plt.close(fig)

    return path


def save_comparison_gif(output_dir: Path, objective: str, solver_results, fps: int = 3, system=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if objective == "centroid" and system is not None:
        return save_centroid_tracking_gifs(output_dir, solver_results, system, fps=fps)

    gif_results = _controlled_solver_results(solver_results)
    n_intervals = max(len(payload["records"]) for payload in gif_results.values())

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ylabel, title = _objective_label(objective)
    ax.set_xlabel("PAT control interval k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    all_values = []
    for payload in gif_results.values():
        all_values.extend(float(r["display_metric"]) for r in payload["records"])
    finite = np.asarray([v for v in all_values if np.isfinite(v)], dtype=float)
    if finite.size:
        ymin, ymax = float(np.min(finite)), float(np.max(finite))
        pad = max(abs(ymax), 1.0) * 0.05 if abs(ymax - ymin) < 1e-15 else 0.10 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlim(-0.25, max(n_intervals - 1, 1) + 0.25)
    ax.grid(True, alpha=0.3)

    lines = {}
    for solver_name in gif_results:
        line, = ax.plot([], [], marker="o", label=solver_name)
        lines[solver_name] = line
    ax.legend(loc="best")
    status = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    def update(frame):
        for solver_name, payload in gif_results.items():
            records = payload["records"]
            upto = min(frame + 1, len(records))
            x = np.arange(upto)
            y = np.asarray([records[i]["display_metric"] for i in range(upto)], dtype=float)
            lines[solver_name].set_data(x, y)
        status.set_text(f"interval {frame + 1}/{n_intervals}")
        return tuple(lines.values()) + (status,)

    animation = FuncAnimation(
        fig,
        update,
        frames=n_intervals,
        interval=1000 / max(int(fps), 1),
        blit=False,
        repeat=True,
    )
    path = output_dir / "comparison.gif"
    animation.save(path, writer=PillowWriter(fps=max(int(fps), 1)))
    plt.close(fig)
    return path

def _draw_aggregate_curves(ax, objective: str, aggregate_results):
    values_for_limits = []

    for solver_name, payload in aggregate_results.items():
        mean = np.asarray(payload["mean_by_interval"], dtype=float)
        std = np.asarray(payload["std_by_interval"], dtype=float)
        k = np.arange(len(mean))

        kwargs = {}
        if _is_no_control(solver_name):
            kwargs.update(
                linestyle="--",
                alpha=0.65,
                linewidth=1.6,
            )

        ax.plot(
            k,
            mean,
            marker="o",
            label=solver_name,
            **kwargs,
        )
        ax.fill_between(
            k,
            mean - std,
            mean + std,
            alpha=0.16 if not _is_no_control(solver_name) else 0.08,
        )
        values_for_limits.extend((mean - std).tolist())
        values_for_limits.extend((mean + std).tolist())

    ylabel, title = _objective_label(objective)
    ax.set_xlabel("PAT control interval k")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (mean ± std)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return values_for_limits


def plot_aggregate_objective(output_dir: Path, objective: str, aggregate_results):
    """Mean±std controller-focused graph plus complete all-method graph."""
    output_dir.mkdir(parents=True, exist_ok=True)

    controlled = _controlled_solver_results(aggregate_results)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    values = _draw_aggregate_curves(ax, objective, controlled)
    _set_metric_ylim(ax, values, objective)
    fig.tight_layout()
    path = output_dir / "aggregate_objective_comparison.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    _draw_aggregate_curves(ax, objective, aggregate_results)
    fig.tight_layout()
    all_path = output_dir / "aggregate_objective_comparison_all.png"
    fig.savefig(all_path, dpi=190)
    plt.close(fig)

    return path

