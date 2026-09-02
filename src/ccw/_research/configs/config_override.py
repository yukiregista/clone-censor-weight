from __future__ import annotations

from pathlib import Path
from importlib.resources import files
_OVERRIDE_DIR: Path | None = None


def set_config_override_dir(path: str | None) -> None:
    """Set a base directory for full YAML overrides (scenario*.yaml)."""
    global _OVERRIDE_DIR
    _OVERRIDE_DIR = Path(path).resolve() if path else None


def read_yaml_text(filename: str, package: str) -> str:
    """Read YAML text from override dir if present; otherwise from package data."""
    if _OVERRIDE_DIR is not None:
        override_path = _OVERRIDE_DIR / filename
        if override_path.is_file():
            return override_path.read_text()
    return (files(package) / filename).read_text()
