"""
Result files: ``<output_dir>/<model>[_<tag>]_<YYYYmmdd_HHMMSS>.json``.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def results_path(output_dir: str, model_name: str, tag: str = '') -> Path:
    """Timestamped JSON path for one run, creating ``output_dir`` if needed."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = model_name.replace('/', '_') + (f'_{tag}' if tag else '')
    return out_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def _to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'{type(obj).__name__} is not JSON serializable')


def write_json(path: Path, payload) -> None:
    """Write ``payload`` (numpy scalars and arrays allowed) and report the path."""
    path.write_text(json.dumps(payload, indent=2, default=_to_builtin))
    print(f'wrote {path}')
