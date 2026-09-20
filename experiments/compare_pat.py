#!/usr/bin/env python3
"""CLI-driven PAT solver comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ExperimentConfig,
    OpticalConfig,
    TurbulenceConfig,
    FrozenPINNConfig,
    BaselineConfig,
    TrackingConfig,
)
from simulation.runner import run_repeated_comparison


DEFAULT_SOLVERS = [
    "frozen_pinn",
    "pid",
    "linear_mpc",
    "ssfm_oracle",
    "no_control",
]


def build_parser():
    p = argparse.ArgumentParser(
        description="Compare Frozen-PINN PAT against baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("experiment")
    g.add_argument("--name", default=None, help="Result folder name.")
    g.add_argument("--objective", choices=["power", "coupling", "centroid"], default="power")
    g.add_argument(
        "--solvers",
        nargs="+",
        default=DEFAULT_SOLVERS,
        choices=[
            "frozen_pinn",
            "pid",
            "linear_mpc",
            "ssfm_oracle",
            "ssfm_oracle_cpu",
            "no_control",
        ],
    )
    g.add_argument("--intervals", type=int, default=8, help="PAT control intervals.")
    g.add_argument(
        "--repeat",
        "--repeats",
        dest="repeats",
        type=int,
        default=1,
        help="Independent repetitions with incremented seeds.",
    )
    g.add_argument("--seed", type=int, default=7, help="Base random seed.")
    g.add_argument(
        "--vary-frozen-seed",
        action="store_true",
        help="Also vary the Frozen-PINN random-feature seed across repeats.",
    )

    g = p.add_argument_group("output")
    g.add_argument("--gif", action="store_true", help="Save comparison.gif for each run.")
    g.add_argument("--fps", type=int, default=3, help="GIF frames per second.")
    g.add_argument("--output-root", default=str(PROJECT_ROOT / "results"))

    g = p.add_argument_group("compute")
    g.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help=(
            "Common accelerator for Frozen-PINN and the default "
            "differentiable SSFM oracle."
        ),
    )
    g.add_argument(
        "--dtype",
        choices=["float64", "float32"],
        default="float64",
        help=(
            "Requested precision for both Frozen-PINN and differentiable SSFM. "
            "MPS automatically forces float32."
        ),
    )

    g = p.add_argument_group("optical system")
    g.add_argument("--grid-size", type=int, default=48)
    g.add_argument("--distance", type=float, default=20.0)
    g.add_argument("--transmit-power", type=float, default=1.0)
    g.add_argument("--beam-waist", type=float, default=2.5e-3)
    g.add_argument("--aperture-radius", type=float, default=2.4e-3)
    g.add_argument("--smf-mode-waist", type=float, default=1.7e-3)

    g = p.add_argument_group("turbulence")
    g.add_argument("--delta-n-rms", type=float, default=8e-9)
    g.add_argument("--turbulence-modes", type=int, default=10)

    g = p.add_argument_group("Frozen-PINN")
    g.add_argument("--frozen-sampler", choices=["elm", "swim"], default="elm")
    g.add_argument(
        "--frozen-backend",
        choices=["auto", "scipy", "torch"],
        default="auto",
        help=(
            "Frozen-PINN propagation backend. "
            "auto uses the optimized torch backend on CPU/CUDA/MPS."
        ),
    )
    g.add_argument(
        "--frozen-device",
        choices=["same", "auto", "cpu", "cuda", "mps"],
        default="same",
        help=(
            "Frozen-PINN device override. 'same' uses --device, "
            "which is recommended for fair comparison."
        ),
    )
    g.add_argument(
        "--frozen-dtype",
        choices=["same", "float64", "float32"],
        default="same",
        help=(
            "Frozen-PINN precision override. 'same' uses --dtype. "
            "MPS automatically forces float32."
        ),
    )
    g.add_argument(
        "--frozen-rk-steps",
        type=int,
        default=64,
        help="Fixed RK4 steps for torch Frozen-PINN propagation.",
    )
    g.add_argument("--hidden-width", type=int, default=1000)
    g.add_argument("--collocation-side", type=int, default=24)
    g.add_argument("--boundary-side", type=int, default=16)
    g.add_argument("--svd-cutoff", type=float, default=1e-6)
    g.add_argument("--pinv-rcond", type=float, default=1e-6)
    g.add_argument("--ode-rtol", type=float, default=1e-5)
    g.add_argument("--ode-atol", type=float, default=1e-8)

    g = p.add_argument_group("PAT optimization")
    g.add_argument(
        "--max-opt-iters",
        "--opt-iters",
        "--frozen-opt-iters",
        "--oracle-opt-iters",
        dest="max_opt_iters",
        type=int,
        default=20,
        help=(
            "Maximum numerical optimizer iterations. "
            "This does NOT change the physical FSM slew limit."
        ),
    )
    g.add_argument(
        "--optimizer-step",
        "--theta-step-max",
        dest="optimizer_step",
        type=float,
        default=40e-6,
        help="Initial numerical line-search step [rad].",
    )
    g.add_argument(
        "--theta-max",
        type=float,
        default=250e-6,
        help="Absolute physical FSM deflection limit [rad].",
    )
    g.add_argument(
        "--theta-slew-rate",
        type=float,
        default=18e-3,
        help="Physical FSM slew-rate limit [rad/s].",
    )
    g.add_argument(
        "--theta-slew-max",
        type=float,
        default=None,
        help=(
            "Optional per-interval slew override [rad]. If omitted, the limit "
            "is --theta-slew-rate times --control-interval."
        ),
    )
    g.add_argument(
        "--line-search-steps",
        type=int,
        default=10,
    )
    g.add_argument(
        "--optimizer-min-step",
        type=float,
        default=1e-9,
    )
    g.add_argument(
        "--gradient-tol",
        type=float,
        default=1e-12,
    )
    g.add_argument(
        "--objective-rel-tol",
        type=float,
        default=1e-6,
    )
    g.add_argument(
        "--objective-abs-tol",
        type=float,
        default=1e-12,
    )
    g.add_argument(
        "--control-regularization",
        type=float,
        default=0.0,
    )

    g = p.add_argument_group("baselines")
    g.add_argument("--pid-kp", type=float, default=0.80)
    g.add_argument("--pid-ki", type=float, default=8.0)
    g.add_argument("--pid-kd", type=float, default=1e-3)
    g.add_argument("--mpc-rho", type=float, default=0.15)
    g.add_argument("--oracle-fd-step", type=float, default=2e-6)

    g = p.add_argument_group("simulation")
    g.add_argument("--ssfm-steps", type=int, default=20)
    g.add_argument(
        "--control-interval",
        type=float,
        default=0.01,
        help="Physical PAT control interval T_c [s].",
    )
    g.add_argument(
        "--target-period",
        type=float,
        default=2.0,
        help="Physical target-trajectory period [s].",
    )
    g.add_argument("--target-x-amplitude", type=float, default=2.4e-3)
    g.add_argument("--target-y-amplitude", type=float, default=1.8e-3)
    g.add_argument("--target-y-phase", type=float, default=0.0)

    return p


def make_config(args):
    name = args.name if args.name is not None else f"{args.objective}_comparison"

    frozen_device = (
        args.device
        if args.frozen_device == "same"
        else args.frozen_device
    )

    frozen_dtype = (
        args.dtype
        if args.frozen_dtype == "same"
        else args.frozen_dtype
    )

    return ExperimentConfig(
        name=name,
        optical=OpticalConfig(
            transmit_power=args.transmit_power,
            beam_waist=args.beam_waist,
            propagation_distance=args.distance,
            grid_size=args.grid_size,
            aperture_radius=args.aperture_radius,
            smf_mode_waist=args.smf_mode_waist,
        ),
        turbulence=TurbulenceConfig(
            delta_n_rms=args.delta_n_rms,
            num_modes=args.turbulence_modes,
            seed=args.seed,
        ),
        frozen_pinn=FrozenPINNConfig(
            sampler=args.frozen_sampler,
            backend=args.frozen_backend,
            device=frozen_device,
            dtype=frozen_dtype,
            rk_steps=args.frozen_rk_steps,
            hidden_width=args.hidden_width,
            collocation_side=args.collocation_side,
            boundary_side=args.boundary_side,
            svd_cutoff=args.svd_cutoff,
            pinv_rcond=args.pinv_rcond,
            num_opt_iterations=args.max_opt_iters,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            theta_step_max=args.optimizer_step,
            theta_max=args.theta_max,
            line_search_steps=args.line_search_steps,
            lambda_theta=args.control_regularization,
            seed=args.seed,
        ),
        baselines=BaselineConfig(
            ssfm_device=args.device,
            ssfm_dtype=args.dtype,
            pid_kp=args.pid_kp,
            pid_ki=args.pid_ki,
            pid_kd=args.pid_kd,
            mpc_rho=args.mpc_rho,
            theta_max=args.theta_max,
            oracle_num_iterations=args.max_opt_iters,
            oracle_fd_step=args.oracle_fd_step,
            oracle_theta_step_max=args.optimizer_step,
            oracle_line_search_steps=args.line_search_steps,
        ),
        tracking=TrackingConfig(
            objective=args.objective,
            num_intervals=args.intervals,
            control_interval_sec=args.control_interval,
            target_period_sec=args.target_period,
            target_x_amplitude=args.target_x_amplitude,
            target_y_amplitude=args.target_y_amplitude,
            target_y_phase=args.target_y_phase,
            ssfm_steps=args.ssfm_steps,
            theta_max=args.theta_max,
            theta_slew_rate=args.theta_slew_rate,
            theta_slew_max_override=args.theta_slew_max,
            max_opt_iterations=args.max_opt_iters,
            optimizer_step=args.optimizer_step,
            line_search_steps=args.line_search_steps,
            optimizer_min_step=args.optimizer_min_step,
            gradient_norm_tol=args.gradient_tol,
            objective_rel_tol=args.objective_rel_tol,
            objective_abs_tol=args.objective_abs_tol,
            control_regularization=args.control_regularization,
        ),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.intervals < 1:
        parser.error("--intervals must be >= 1")
    if args.repeats < 1:
        parser.error("--repeat must be >= 1")
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    if args.max_opt_iters < 1:
        parser.error("--max-opt-iters must be >= 1")
    if args.line_search_steps < 1:
        parser.error("--line-search-steps must be >= 1")
    if args.control_interval <= 0:
        parser.error("--control-interval must be > 0")
    if args.target_period <= 0:
        parser.error("--target-period must be > 0")
    if args.theta_slew_rate <= 0:
        parser.error("--theta-slew-rate must be > 0")
    if args.theta_slew_max is not None and args.theta_slew_max <= 0:
        parser.error("--theta-slew-max must be > 0 when provided")
    if args.optimizer_step <= 0:
        parser.error("--optimizer-step must be > 0")

    cfg = make_config(args)

    print("Experiment configuration")
    print(f"  name          : {cfg.name}")
    print(f"  objective     : {args.objective}")
    print(f"  solvers       : {', '.join(args.solvers)}")
    print(f"  intervals     : {args.intervals}")
    print(f"  repeats       : {args.repeats}")
    print(f"  common device : {args.device}")
    print(f"  common dtype  : {args.dtype}")
    print(f"  control dt    : {args.control_interval:.6f} s")
    print(f"  target period : {args.target_period:.6f} s")
    print(f"  sim duration  : {args.intervals*args.control_interval:.6f} s")
    print(f"  max opt iters : {args.max_opt_iters}")
    print(f"  optimizer step: {args.optimizer_step*1e6:.2f} urad")
    print(f"  FSM slew rate : {args.theta_slew_rate:.6g} rad/s")
    print(f"  FSM slew/step : {cfg.tracking.theta_slew_max*1e6:.2f} urad")
    print(f"  GIF           : {args.gif}")
    if args.gif:
        print(f"  GIF FPS       : {args.fps}")

    result = run_repeated_comparison(
        cfg=cfg,
        solver_names=args.solvers,
        output_root=Path(args.output_root),
        repeats=args.repeats,
        make_gif=args.gif,
        gif_fps=args.fps,
        vary_frozen_seed=args.vary_frozen_seed,
    )

    print("\nFinished.")
    for idx, run in enumerate(result["runs"]):
        print(f"Run {idx:03d} JSON : {run['json_path']}")
        for path in run["plot_paths"]:
            print(f"Run {idx:03d} plot : {path}")
        if run["gif_path"] is not None:
            print(f"Run {idx:03d} GIF  : {run['gif_path']}")

    if result["aggregate_json_path"] is not None:
        print(f"Aggregate JSON : {result['aggregate_json_path']}")
        print(f"Aggregate plot : {result['aggregate_plot_path']}")


if __name__ == "__main__":
    main()
