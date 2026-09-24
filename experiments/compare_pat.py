#!/usr/bin/env python3
"""Run receiver-FSM PAT experiments from PAT (1).pdf."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import make_preset
from simulation.runner import run_repeated_comparison, replot_results

DEFAULT_SOLVERS = ["frozen_pinn", "pid", "linear_mpc", "ssfm_oracle", "no_control"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preset", choices=["demo", "paper"], default="paper")
    p.add_argument("--replot", type=Path, help="Regenerate figures/GIF from an existing result directory or results.json; keep numerical results")
    p.add_argument("--name", help="New result directory; existing results are never overwritten")
    p.add_argument("--objective", choices=["power"], default="power", help="PDF Eq. (20), collected communication power")
    p.add_argument("--solvers", nargs="+", choices=DEFAULT_SOLVERS+["ssfm_oracle_cpu"], default=DEFAULT_SOLVERS)
    p.add_argument("--intervals", type=int)
    p.add_argument("--repeat", "--repeats", dest="repeats", type=int, default=1)
    p.add_argument("--feature-seeds", type=int, default=1, help="Cartesian feature-seed repetitions for each channel seed")
    p.add_argument("--paper-ensemble", action="store_true", help="100 channel seeds x 5 feature seeds (expensive); implies --preset paper")
    p.add_argument("--vary-frozen-seed", action="store_true", help="Legacy option: also shift feature seed with channel seed")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--feature-seed", type=int, default=7)
    p.add_argument("--gif", action=argparse.BooleanOptionalAction, default=True, help="Save detector XY animation; --no-gif disables it")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--output-root", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu", help="Receiver and PINN layer-projection device; auto selects CUDA, MPS, then CPU. RK45 and SSFM propagation use CPU")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64", help="Torch receiver precision; MPS uses float32. RK45 remains float64")
    p.add_argument("--frozen-backend", choices=["scipy", "torch", "auto"], default="auto", help="auto/torch follows --device for receiver queries; scipy explicitly uses CPU")
    p.add_argument("--frozen-operator", choices=["auto", "matrix_free", "projected"], default="auto", help="auto assembles the Galerkin operator once per layer on GPU; matrix_free retains the CPU reference RHS")
    p.add_argument("--disturbance", choices=["iid", "periodic", "zero"], default="iid")
    # Unspecified overrides inherit the selected preset.
    fields = {
        "optical": {"grid-size": int, "receiver-grid-size": int, "distance": float,
            "transmit-power": float, "beam-waist": float, "half-width": float,
            "tx-angular-std": float, "path-loss-db": float, "entrance-diameter": float,
            "reducer-magnification": float, "reducer-transmission": float,
            "focal-length": float, "aperture-radius": float, "detector-sampling": float,
            "psd-position-noise-std": float},
        "turbulence": {"ground-cn2": float, "hv-wind-speed": float, "outer-scale": float,
            "inner-scale": float, "turbulence-strength": float, "turbulence-modes": int,
            "layer-thickness": float},
        "Frozen-PINN": {"hidden-width": int, "collocation-side": int, "spectral-side": int,
            "feature-scale-max": float, "svd-cutoff": float, "ode-rtol": float, "ode-atol": float},
        "steering": {"control-interval": float, "target-period": float, "ssfm-steps": int,
            "theta-max": float, "theta-slew-rate": float, "theta-slew-max": float,
            "theta-quantization": float, "max-opt-iters": int, "optimizer-step": float,
            "line-search-steps": int, "line-search-shrink": float},
        "baselines": {"pid-kp": float, "pid-ki": float, "pid-kd": float, "mpc-rho": float,
            "oracle-fd-step": float}}
    for group, args in fields.items():
        g = p.add_argument_group(group)
        for name, kind in args.items():
            g.add_argument("--"+name, type=kind)
    p.add_argument("--detector-offset", type=float, nargs=2, metavar=("X_M", "Y_M"))
    return p


def make_config(args):
    cfg = make_preset("paper" if args.paper_ensemble else args.preset)
    cfg.name = args.name or cfg.name
    cfg.turbulence.seed = args.seed
    cfg.frozen_pinn.seed = args.feature_seed
    cfg.frozen_pinn.backend = args.frozen_backend
    cfg.frozen_pinn.operator_backend = args.frozen_operator
    cfg.frozen_pinn.device = cfg.baselines.ssfm_device = args.device
    cfg.frozen_pinn.dtype = cfg.baselines.ssfm_dtype = args.dtype
    cfg.tracking.disturbance = args.disturbance
    mapping = {
        "optical": dict(grid_size="grid_size", receiver_grid_size="receiver_grid_size",
            distance="propagation_distance", transmit_power="transmit_power", beam_waist="beam_waist",
            half_width="half_width", tx_angular_std="tx_angular_std", path_loss_db="path_loss_db",
            entrance_diameter="entrance_diameter", reducer_magnification="reducer_magnification",
            reducer_transmission="reducer_transmission", focal_length="focal_length",
            aperture_radius="aperture_radius", detector_sampling="detector_sampling",
            psd_position_noise_std="psd_position_noise_std"),
        "turbulence": dict(ground_cn2="ground_cn2", hv_wind_speed="hv_wind_speed",
            outer_scale="outer_scale", inner_scale="inner_scale", turbulence_strength="strength_scale",
            turbulence_modes="num_modes", layer_thickness="layer_thickness"),
        "frozen_pinn": {x:x for x in ["hidden_width", "collocation_side", "spectral_side",
            "feature_scale_max", "svd_cutoff", "ode_rtol", "ode_atol"]},
        "tracking": dict(intervals="num_intervals", control_interval="control_interval_sec",
            target_period="target_period_sec", ssfm_steps="ssfm_steps", theta_max="theta_max",
            theta_slew_rate="theta_slew_rate", theta_slew_max="theta_slew_max_override",
            theta_quantization="theta_quantization", max_opt_iters="max_opt_iterations",
            optimizer_step="optimizer_step", line_search_steps="line_search_steps",
            line_search_shrink="line_search_shrink"),
        "baselines": {x:x for x in ["pid_kp", "pid_ki", "pid_kd", "mpc_rho", "oracle_fd_step"]}}
    for group, fields in mapping.items():
        for arg, field in fields.items():
            value = getattr(args, arg)
            if value is not None:
                setattr(getattr(cfg, group), field, value)
    if args.focal_length is not None:
        cfg.optical.splitter_to_psd = cfg.optical.splitter_to_detector = args.focal_length-cfg.optical.lens_to_splitter
    if args.detector_offset is not None:
        cfg.optical.detector_center = tuple(args.detector_offset)
    cfg.validate()
    return cfg


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.repeats, args.feature_seeds, args.fps) < 1:
        parser.error("repeats, feature-seeds and fps must be positive")
    if args.replot:
        result = replot_results(args.replot, make_gif=args.gif, gif_fps=args.fps)
        print(f"Figures regenerated: {result['output_dir']}")
        if result['gif_path']:
            print(f"GIF: {result['gif_path']}")
        return
    try:
        cfg = make_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    repeats = cfg.channel_realizations if args.paper_ensemble else args.repeats
    features = cfg.feature_seeds if args.paper_ensemble else args.feature_seeds
    print(f"{cfg.name}: preset={cfg.preset}, objective=P_D [W], receiver-side FSM")
    print(f"{cfg.tracking.num_intervals} intervals x {repeats} channel seeds x {features} feature seeds")
    print(f"Atmosphere: SSFM grid={cfg.optical.grid_size} x {cfg.optical.grid_size} spatial cells; PINN M={cfg.frozen_pinn.hidden_width} features, Nc={cfg.frozen_pinn.collocation_side} x {cfg.frozen_pinn.collocation_side} collocation points; RK45")
    print(f"FSM: {cfg.tracking.theta_max*1e6:g} urad range, {cfg.tracking.theta_slew_max*1e6:g} urad/axis/interval, q={cfg.tracking.theta_quantization*1e6:g} urad")
    print("Demo presets and explicit numerical overrides require convergence validation before paper claims.")
    result = run_repeated_comparison(cfg, args.solvers, Path(args.output_root), repeats,
        args.gif, args.fps, args.vary_frozen_seed, features)
    for run in result["runs"]:
        print(f"Saved: {run['output_dir']}")
        if run['gif_path']:
            print(f"GIF: {run['gif_path']}")
    if result["aggregate_json_path"]:
        print(f"Aggregate: {result['aggregate_json_path']}")


if __name__ == "__main__":
    main()
