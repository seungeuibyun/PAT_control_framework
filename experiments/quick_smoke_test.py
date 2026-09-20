#!/usr/bin/env python3
from __future__ import annotations

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
from simulation.runner import run_comparison


cfg = ExperimentConfig(
    name="smoke_test",
    optical=OpticalConfig(
        grid_size=32,
        propagation_distance=10.0,
    ),
    turbulence=TurbulenceConfig(
        delta_n_rms=5e-9,
        seed=7,
    ),
    frozen_pinn=FrozenPINNConfig(
        hidden_width=72,
        collocation_side=9,
        boundary_side=8,
        num_opt_iterations=2,
        ode_rtol=1e-3,
        ode_atol=1e-6,
    ),
    baselines=BaselineConfig(
        oracle_num_iterations=1,
        oracle_line_search_steps=2,
    ),
    tracking=TrackingConfig(
        objective="power",
        num_intervals=2,
        ssfm_steps=10,
    ),
)


if __name__ == "__main__":
    run_comparison(
        cfg,
        solver_names=[
            "frozen_pinn",
            "pid",
            "linear_mpc",
            "ssfm_oracle",
            "no_control",
        ],
        output_root=PROJECT_ROOT / "results",
    )
