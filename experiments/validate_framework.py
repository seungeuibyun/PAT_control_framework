#!/usr/bin/env python3
"""Deterministic sanity checks for the PAT simulation framework."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ExperimentConfig,
    OpticalConfig,
    TurbulenceConfig,
    FrozenPINNConfig,
    TrackingConfig,
)
from system_model.optical_system import OpticalPATSystem
from solver.frozen_pinn import FrozenPINNSolver
from solver.frozen_pinn_torch import TorchFrozenPINNSolver


def assert_close(name, a, b, atol, rtol=0.0):
    a = np.asarray(a)
    b = np.asarray(b)
    err = np.max(np.abs(a - b))
    scale = max(float(np.max(np.abs(b))), 1e-30)
    if err > atol + rtol * scale:
        raise AssertionError(
            f"{name}: max error {err:.3e} exceeds tolerance "
            f"{atol + rtol * scale:.3e}"
        )
    print(f"PASS {name}: max error={err:.3e}")


def make_cfg(objective="power", intervals=20):
    return ExperimentConfig(
        name="validation",
        optical=OpticalConfig(
            grid_size=48,
            propagation_distance=20.0,
        ),
        turbulence=TurbulenceConfig(
            delta_n_rms=8e-9,
            seed=7,
        ),
        frozen_pinn=FrozenPINNConfig(
            sampler="elm",
            backend="torch",
            device="cpu",
            dtype="float64",
            rk_steps=48,
            hidden_width=600,
            collocation_side=20,
            boundary_side=14,
            svd_cutoff=3e-6,
            pinv_rcond=3e-6,
        ),
        tracking=TrackingConfig(
            objective=objective,
            num_intervals=intervals,
            control_interval_sec=0.01,
            target_period_sec=2.0,
            target_x_amplitude=2.4e-3,
            target_y_amplitude=1.8e-3,
            target_y_phase=0.0,
            ssfm_steps=20,
        ),
    )


def test_physical_time_invariance():
    cfg_a = make_cfg(intervals=8)
    cfg_b = make_cfg(intervals=200)
    a = OpticalPATSystem(cfg_a)
    b = OpticalPATSystem(cfg_b)

    for k in range(8):
        assert_close(
            f"target path independent of num_intervals at k={k}",
            a.target_position(k),
            b.target_position(k),
            atol=1e-15,
        )

    n_period = int(round(cfg_a.tracking.target_period_sec / cfg_a.tracking.control_interval_sec))
    assert_close(
        "target path closes after one physical period",
        a.target_position(n_period),
        a.target_position(0),
        atol=1e-12,
    )


def test_torch_vs_scipy():
    theta = np.array([35e-6, 55e-6])
    interval = 7

    for objective in ("power", "coupling", "centroid"):
        cfg = make_cfg(objective=objective)
        system = OpticalPATSystem(cfg)
        target = system.target_position(interval)

        scipy_solver = FrozenPINNSolver(system, cfg)
        U, dUx, dUy = scipy_solver.propagate(theta, interval)
        q_scipy, g_scipy = scipy_solver._metric_gradient(U, dUx, dUy, target)

        torch_solver = TorchFrozenPINNSolver(
            system,
            cfg,
            device="cpu",
            dtype="float64",
            rk_steps=48,
        )
        q_torch, g_torch = torch_solver.objective_and_gradient(theta, interval, target)

        assert_close(
            f"{objective}: torch vs scipy objective",
            q_torch,
            q_scipy,
            atol=2e-5,
            rtol=2e-4,
        )
        assert_close(
            f"{objective}: torch vs scipy gradient",
            g_torch,
            g_scipy,
            atol=2e-4,
            rtol=3e-3,
        )


def test_reduced_centroid_matches_reconstruction():
    cfg = make_cfg(objective="centroid")
    system = OpticalPATSystem(cfg)
    solver = TorchFrozenPINNSolver(
        system,
        cfg,
        device="cpu",
        dtype="float64",
        rk_steps=48,
    )

    interval = 9
    target = system.target_position(interval)
    theta = np.array([28e-6, 45e-6])

    reduced = solver.predicted_centroid(theta, interval, target)
    U = solver.reconstruct_field(theta, interval)
    reconstructed = system.centroid(U, target)

    assert_close(
        "reduced centroid vs reconstructed Frozen-PINN field centroid",
        reduced,
        reconstructed,
        atol=1e-10,
    )


def main():
    print("PAT framework validation")
    test_physical_time_invariance()
    test_torch_vs_scipy()
    test_reduced_centroid_matches_reconstruction()
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
