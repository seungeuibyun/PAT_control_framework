from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_results_json(output_dir: Path, cfg, solver_results):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "solvers": solver_results,
    }
    path = output_dir / "results.json"
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
