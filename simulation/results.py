from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_results_json(output_dir, cfg, solver_results, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   config=cfg.to_dict(), metadata=metadata or {}, solvers=solver_results)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path
