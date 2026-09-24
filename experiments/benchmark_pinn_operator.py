#!/usr/bin/env python3
"""Paired, identical-discretization benchmark of reference and projected RK45."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config.settings import make_preset
from system_model.optical_system import OpticalPATSystem
from solver.frozen_pinn_torch import TorchFrozenPINNSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['demo', 'paper'], default='demo')
    parser.add_argument('--intervals', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='mps')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.intervals < 1:
        parser.error('intervals must be positive')
    if args.output.exists():
        parser.error(f'Output exists: {args.output}')
    cfg = make_preset(args.preset)
    cfg.frozen_pinn.device = args.device
    cfg.tracking.num_intervals = args.intervals
    system = OpticalPATSystem(cfg)
    solvers = {}
    for mode in ('matrix_free', 'projected'):
        selected = deepcopy(cfg)
        selected.frozen_pinn.operator_backend = mode
        solvers[mode] = TorchFrozenPINNSolver(system, selected)
    rows = []
    commands = np.array([[0., 0.], [35e-6, -25e-6], [-40e-6, 30e-6]])
    for interval in range(args.intervals):
        row = dict(interval=interval)
        order = list(solvers) if interval % 2 == 0 else list(reversed(solvers))
        for mode in order:
            solver = solvers[mode]
            tick = time.perf_counter()
            solver.prepare_interval(interval)
            row[mode+'_prepare_sec'] = time.perf_counter()-tick
            row[mode+'_diagnostics'] = solver.interval_diagnostics
        reference, fast = solvers['matrix_free'], solvers['projected']
        row['coefficient_relative_error'] = float(np.linalg.norm(fast._coefficients-reference._coefficients)/np.linalg.norm(reference._coefficients))
        row['reduced_field_relative_error'] = float(np.linalg.norm(fast._reduced-reference._reduced)/np.linalg.norm(reference._reduced))
        power_errors, gradient_errors = [], []
        for theta in commands:
            value, grad = reference.objective_and_gradient(theta, interval)
            value_fast, grad_fast = fast.objective_and_gradient(theta, interval)
            power_errors.append(abs(value_fast-value)/max(value, 1e-30))
            gradient_errors.append(np.linalg.norm(grad_fast-grad)/max(np.linalg.norm(grad), 1e-30))
        row['max_power_relative_error'] = float(max(power_errors))
        row['max_gradient_relative_error'] = float(max(gradient_errors))
        # This checks the optimization preserves the old solver, not that the
        # frozen spatial basis has converged to the independent SSFM solution.
        if max(row['coefficient_relative_error'], row['max_power_relative_error'], row['max_gradient_relative_error']) > 1e-3:
            raise AssertionError(f'Projected-operator accuracy regression: {row}')
        rows.append(row)
        print(f"[{interval+1}/{args.intervals}] reference={row['matrix_free_prepare_sec']:.4f}s projected={row['projected_prepare_sec']:.4f}s coefficient_error={row['coefficient_relative_error']:.3g}", flush=True)
    reference_mean = float(np.mean([r['matrix_free_prepare_sec'] for r in rows]))
    projected_mean = float(np.mean([r['projected_prepare_sec'] for r in rows]))
    summary = dict(reference_mean_prepare_sec=reference_mean, projected_mean_prepare_sec=projected_mean,
                   preparation_speedup=reference_mean/projected_mean,
                   max_coefficient_relative_error=max(r['coefficient_relative_error'] for r in rows),
                   max_power_relative_error=max(r['max_power_relative_error'] for r in rows),
                   max_gradient_relative_error=max(r['max_gradient_relative_error'] for r in rows))
    payload = dict(config=cfg.to_dict(), device=str(fast.engine.device),
        scope='Atmospheric prediction + receiver field upload; excludes offline basis/operator construction and control queries',
        comparison='Same features, rank, collocation, channels and RK45 tolerances; CPU matrix-free versus GPU-projected operator',
        summary=summary, records=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
