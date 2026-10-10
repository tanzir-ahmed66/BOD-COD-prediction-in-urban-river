"""
Provenance for phase manifests: which code, which data, which environment.

Every manifest written by a notebook carries `provenance()`, so a result file can
be traced to the exact src/ version and input files that produced it. Two runs
with equal `code_sha256` and `data_sha256` used identical code and inputs.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional

from .config import PROJECT_ROOT, DATA_DIR

_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "xgboost", "lightgbm", "matplotlib")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _combined(hashes: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for name in sorted(hashes):
        h.update(f"{name}:{hashes[name]}\n".encode())
    return h.hexdigest()


def provenance(project_root: Optional[Path] = None) -> Dict[str, object]:
    """
    code_sha256 : one hash over every src/*.py (name + content)
    data_sha256 : one hash over every data/*.csv, data/*.json and data/external/*.csv
                  (blank *_TEMPLATE.csv files excluded)
    files       : the per-file hashes behind both
    environment : Python and package versions (read from metadata; nothing is imported)
    """
    root = Path(project_root) if project_root else Path(PROJECT_ROOT)
    src = {p.name: _sha256(p) for p in sorted((root / "src").glob("*.py"))}
    data_dir = Path(DATA_DIR)
    data_files = [*data_dir.glob("*.csv"), *data_dir.glob("*.json"),
                  # *_TEMPLATE.csv files are blank templates, not inputs
                  *(p for p in (data_dir / "external").glob("*.csv") if "_TEMPLATE" not in p.name)]
    data = {str(p.relative_to(data_dir)): _sha256(p) for p in sorted(data_files)}
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in _PACKAGES:
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[pkg] = None
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_sha256": _combined(src),
        "data_sha256": _combined(data),
        "files": {"src": src, "data": data},
        "environment": env,
    }
