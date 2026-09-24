#!/usr/bin/env python3
"""Benchmark cached receiver queries and full interval preparation separately.

FSM commands never trigger another atmospheric solve. Grid scaling changes
only the SSFM atmospheric resolution, not the receiver detector quadrature.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config.settings import make_preset
from system_model.optical_system import OpticalPATSystem
from simulation.runner import build_solvers
from simulation.results import _jsonable


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset",choices=["demo","paper"],default="demo")
    p.add_argument("--grid-sizes",type=int,nargs="+",default=[128,256,512])
    p.add_argument("--queries",type=int,nargs="+",default=[1,5,20])
    p.add_argument("--repeat",type=int,default=3)
    p.add_argument("--device",choices=["cpu","cuda","mps","auto"],default="cpu")
    p.add_argument("--dtype",choices=["float64","float32"],default="float64")
    p.add_argument("--output-dir",type=Path,default=ROOT/"results"/"pdf_runtime_scaling")
    return p


def main():
    p=build_parser();a=p.parse_args()
    if min(a.grid_sizes+a.queries+[a.repeat])<1:
        p.error("grid sizes, query counts and repeats must be positive")
    if (a.output_dir/"results.json").exists():
        raise FileExistsError("Choose a new --output-dir; existing results are preserved")
    cfg=make_preset(a.preset)
    cfg.frozen_pinn.backend="torch"
    cfg.frozen_pinn.device=cfg.baselines.ssfm_device=a.device
    cfg.frozen_pinn.dtype=cfg.baselines.ssfm_dtype=a.dtype
    records=[]
    for grid in a.grid_sizes:
        c=deepcopy(cfg);c.optical.grid_size=grid
        s=OpticalPATSystem(c)
        solvers=build_solvers(s,c,["frozen_pinn","ssfm_oracle","ssfm_oracle_cpu"])
        reduced=s.reduce_field(s.atmospheric_field(0))
        for solver in solvers:
            start=time.perf_counter();solver.prepare_interval(0);prep=time.perf_counter()-start
            solver.objective_and_gradient(np.zeros(2),0)  # warm receiver kernels
            for queries in a.queries:
                samples=[]
                for repeat in range(a.repeat):
                    theta=np.zeros(2)
                    start=time.perf_counter()
                    for q in range(queries):
                        _,gradient=solver.objective_and_gradient(theta,0)
                        # Query workload is explicit; final applied command is
                        # constrained against the same interval-start command.
                        from solver.optimization import project_pat_command, quantize_pat_command
                        norm=np.linalg.norm(gradient)
                        if norm>0:
                            theta=project_pat_command(theta+c.tracking.optimizer_step*gradient/norm,np.zeros(2),c.tracking)
                    samples.append(time.perf_counter()-start)
                theta=quantize_pat_command(theta,np.zeros(2),c.tracking)
                truth=s.power(s.detector_field(reduced,theta))
                record=dict(solver=solver.name,grid_size=grid,queries=queries,
                    preparation_sec=prep,mean_query_loop_sec=float(np.mean(samples)),
                    mean_interval_sec=prep+float(np.mean(samples)),actual_power_w=truth,
                    theta=theta,offline_basis_sec=getattr(solver,"basis_setup_time_sec",0))
                record["offline_operator_sec"] = getattr(solver, "operator_setup_time_sec", 0)
                record["atmosphere_backend"] = getattr(solver, "interval_diagnostics", {}).get("atmosphere_backend", "numpy_cpu_ssfm")
                engine = getattr(solver, "engine", None)
                if engine is not None:
                    record.update(device=str(engine.device), requested_dtype=engine.requested_dtype,
                        effective_dtype=engine.effective_dtype, receiver_complex_backend=engine.complex_backend)
                records.append(record);print(record,flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    payload=dict(config=cfg.to_dict(),timing_scope="one channel prediction plus cached receiver queries; offline basis excluded",records=records)
    (a.output_dir/"results.json").write_text(json.dumps(_jsonable(payload),indent=2,allow_nan=False))
    for key,filename,ylabel in [("mean_query_loop_sec","runtime_vs_queries.png","Receiver query runtime [s]"),
                               ("mean_interval_sec","interval_total_vs_queries.png","Prediction + receiver queries [s]")]:
        fig,ax=plt.subplots(figsize=(8,5))
        for name in dict.fromkeys(r["solver"] for r in records):
            for grid in a.grid_sizes:
                points=[r for r in records if r["solver"]==name and r["grid_size"]==grid]
                ax.plot([r["queries"] for r in points],[r[key] for r in points],marker="o",label=f"{name}, N={grid}")
        ax.set(xlabel="Receiver power/gradient queries",ylabel=ylabel)
        ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(a.output_dir/filename,dpi=180);plt.close(fig)


if __name__=="__main__":
    main()
