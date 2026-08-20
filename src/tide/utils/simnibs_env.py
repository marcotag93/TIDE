"""
SimNIBS Installation Descriptor
===============================
Single, standard-library-only source of truth for locating a SimNIBS
installation: the bundled Python interpreter and its pip, the
``get_fields_at_coordinates`` CLI launcher, and the bundled coil-models
directory.

No NumPy or SimNIBS import happens here, so the module stays importable under
any interpreter (source checkout, foreign venv, or the SimNIBS environment
itself) and can run before the environment relaunch. The resolver functions are
pure: they return ``None`` (or an empty list) when a component is absent and
never call ``sys.exit`` or print. Callers own their error messages and exit
behaviour, preserving the existing CLI, configuration, and sampling contracts.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional


def find_simnibs_launcher() -> Optional[str]:
    """Return the ``simnibs`` launcher path on PATH, or ``None``."""
    return shutil.which("simnibs")


def simnibs_root() -> Optional[Path]:
    """Return the SimNIBS installation root inferred from the launcher."""
    launcher = find_simnibs_launcher()
    if not launcher:
        return None
    return Path(launcher).resolve().parent.parent


def python_candidates(root: Path) -> List[Path]:
    """Return the ordered SimNIBS python interpreter candidates for ``root``."""
    if sys.platform == "win32":
        return [
            root / "simnibs_env" / "Scripts" / "python.exe",
            root / "simnibs_env" / "python.exe",
            root / "Scripts" / "python.exe",
            root / "python.exe",
        ]
    return [
        root / "simnibs_env" / "bin" / "python3",
        root / "simnibs_env" / "bin" / "python",
        root / "bin" / "python3",
        root / "bin" / "python",
    ]


def select_python(candidates: List[Path]) -> Optional[Path]:
    """Return the first existing, executable python candidate, or ``None``."""
    for candidate in candidates:
        if candidate.exists() and (sys.platform == "win32" or os.access(candidate, os.X_OK)):
            return candidate
    return None


def pip_candidates(python: Path) -> List[Path]:
    """Return the ordered pip candidates next to ``python``."""
    bin_dir = python.parent
    if sys.platform == "win32":
        return [bin_dir / "pip.exe", bin_dir / "pip3.exe"]
    return [bin_dir / "pip", bin_dir / "pip3"]


def select_pip(candidates: List[Path]) -> Optional[Path]:
    """Return the first existing pip candidate, or ``None``."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_get_fields_at_coordinates() -> Optional[str]:
    """Return the ``get_fields_at_coordinates`` launcher path, or ``None``.

    Handles the Windows ``.cmd``/``.exe`` suffixes.
    """
    for name in (
        "get_fields_at_coordinates",
        "get_fields_at_coordinates.cmd",
        "get_fields_at_coordinates.exe",
    ):
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def find_coil_models_dir() -> Optional[Path]:
    """Return the bundled Drakaki coil-models directory, or ``None``.

    Falls back to the parent ``coil_models`` directory when the specific
    ``Drakaki_BrainStim_2022`` subdirectory is absent.
    """
    root = simnibs_root()
    if root is None:
        return None

    lib_dir = root / "simnibs_env" / "lib"
    if not lib_dir.exists():
        return None

    site_packages = list(lib_dir.glob("*/site-packages"))
    if not site_packages:
        return None

    coil_path = (
        site_packages[0] / "simnibs" / "resources" / "coil_models" / "Drakaki_BrainStim_2022"
    )
    if coil_path.exists():
        return coil_path
    if coil_path.parent.exists():
        return coil_path.parent
    return None
