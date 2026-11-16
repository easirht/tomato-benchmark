"""YAML config system: layered configs, command-line overrides, config hash, run IDs.

Layers (later overrides earlier):
    configs/base.yaml  ->  configs/platform/<kaggle|colab|local>.yaml  ->  configs/exp/<name>.yaml
    ->  overrides such as  ["run.seed=43", "model.name=resnet50"]
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from .env import default_roots, detect_platform


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    d = cfg
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def apply_overrides(cfg: dict, overrides) -> dict:
    """overrides: list of 'a.b=value' strings. The value is parsed as YAML (so 5, 0.1, true,
    [1,2] and plain strings all work)."""
    cfg = copy.deepcopy(cfg)
    for o in overrides or []:
        if "=" not in o:
            raise ValueError(f"Override must look like key=value, got: {o!r}")
        key, val = o.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str | Path = "configs", exp: str | None = None,
                overrides=(), platform: str | None = None, repo_root: str | Path = ".") -> dict:
    """Load base + platform + experiment config, apply overrides, resolve paths."""
    config_dir = Path(config_dir)
    platform = platform or detect_platform()
    cfg = _read_yaml(config_dir / "base.yaml")
    plat_file = config_dir / "platform" / f"{platform}.yaml"
    if plat_file.exists():
        cfg = deep_merge(cfg, _read_yaml(plat_file))
    if exp:
        exp_file = Path(exp) if str(exp).endswith(".yaml") else config_dir / "exp" / f"{exp}.yaml"
        cfg = deep_merge(cfg, _read_yaml(exp_file))
    cfg = apply_overrides(cfg, overrides)
    return resolve_paths(cfg, repo_root)


def resolve_paths(cfg: dict, repo_root: str | Path = ".") -> dict:
    """Fill null paths from platform defaults. Paths are excluded from the config hash, so the
    same experiment gets the same hash on Kaggle, Colab and local."""
    cfg = copy.deepcopy(cfg)
    roots = default_roots(repo_root)
    p = cfg.setdefault("paths", {})
    p.setdefault("persist_dirs", [])
    if not p.get("data_root"):
        p["data_root"] = roots["data_root"]
    if not p.get("work_root"):
        p["work_root"] = roots["work_root"]
    work = Path(p["work_root"])
    p.setdefault("runs_dir", str(work / "runs"))
    p.setdefault("results_dir", str(work / "results"))
    p.setdefault("splits_dir", str(Path(repo_root) / "splits"))
    cfg["runtime"] = {"platform": roots["platform"]}
    return cfg


def config_hash(cfg: dict, length: int = 10) -> str:
    """Hash of everything that defines the *experiment* (not paths, not runtime info, not the
    seed). Two runs with different seeds but identical settings share a hash."""
    c = copy.deepcopy(cfg)
    c.pop("paths", None)
    c.pop("runtime", None)
    if isinstance(c.get("run"), dict):
        c["run"].pop("seed", None)
    blob = json.dumps(c, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def make_run_id(cfg: dict) -> str:
    """Deterministic run ID, e.g. main_benchmark__resnet50__s42__a1b2c3d4e5."""
    exp = cfg.get("exp_name", "exp")
    model = get_by_path(cfg, "model.name", "model")
    seed = get_by_path(cfg, "run.seed", 0)
    return f"{exp}__{model}__s{seed}__{config_hash(cfg)}"


def run_dir_for(cfg: dict) -> Path:
    return Path(cfg["paths"]["runs_dir"]) / make_run_id(cfg)


def iter_runs(cfg: dict, seeds_key: str = "seeds_main", models_key: str = "models_main") -> Iterator[dict]:
    """Yield one config per (model, seed) combination of the benchmark grid.

    Order is model-major so that if a session dies you lose the *last* model's seeds, not a
    random hole in the grid. Finished runs are skipped by the caller via CheckpointManager.is_done().
    """
    seeds = get_by_path(cfg, f"protocol.{seeds_key}")
    models = get_by_path(cfg, f"protocol.{models_key}")
    for m in models:
        for s in seeds:
            c = copy.deepcopy(cfg)
            set_by_path(c, "model.name", m["name"])
            set_by_path(c, "model.family", m.get("family", "unknown"))
            set_by_path(c, "run.seed", s)
            yield c
