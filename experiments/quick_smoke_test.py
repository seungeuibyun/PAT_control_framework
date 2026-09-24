#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from config.settings import make_preset
from simulation.runner import run_comparison


def main():
    cfg=make_preset("demo")
    cfg.name="pdf_smoke_test"
    cfg.tracking.num_intervals=2
    cfg.tracking.max_opt_iterations=5
    run_comparison(cfg,["frozen_pinn","pid","linear_mpc","ssfm_oracle","no_control"],ROOT/"results",make_gif=True)


if __name__ == "__main__":
    main()
