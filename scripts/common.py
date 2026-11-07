"""
common.py -- shared helpers: configuration, paths, JSON I/O, logging, seeding.

Every script reads the analysis settings from configs/experiment.yaml (or the
file named by $EXPERIMENT_YAML) through load_cfg(), and the folders from the
environment variables exported by config.sh (with sensible fallbacks so the
scripts also run outside HPRC, e.g. in the synthetic smoke test).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PKG_DIR = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- config
def load_cfg(path=None):
    """Load experiment.yaml; relative paths inside it are resolved against the package folder."""
    if yaml is None:
        sys.exit("pyyaml is missing: pip install pyyaml")
    path = Path(path or os.environ.get("EXPERIMENT_YAML") or PKG_DIR / "configs" / "experiment.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def pkg_path(rel):
    """Resolve a path from the config relative to the package folder."""
    p = Path(rel)
    return p if p.is_absolute() else PKG_DIR / p


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def env_dir(name, default):
    """Folder from an environment variable exported by config.sh, else a default."""
    return Path(os.environ.get(name) or default)


# ----------------------------------------------------------------------------- JSON
class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _clean(o):
    """Replace NaN/inf floats by None so the JSON stays valid."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean(json.loads(json.dumps(obj, cls=_NpEncoder))), f, indent=2)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- misc
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def m1_default(cfg):
    """Fallback M1 parameters: the middle of the eps and m_pts grids."""
    g, m = cfg["m1"]["eps_grid"], cfg["m1"]["min_pts_grid"]
    return (g[len(g) // 2], m[len(m) // 2])


def class_index(cfg):
    """{class name: index} in the order of brpcd.classes."""
    return {c: i for i, c in enumerate(cfg["brpcd"]["classes"])}


FT_TO_M = 0.3048
