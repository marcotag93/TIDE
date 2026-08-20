#!/usr/bin/env python3
"""
TIDE Pipeline - Development Entry Point
=======================================

Thin wrapper around :func:`tide.cli.main` for running the pipeline directly
from a source checkout (without installing the package).

Once installed via ``python -m pip install -e .``, prefer the registered console
script::

    tide --config config.yml --workflow estimation

Check install mode:
~/SimNIBS-4.5/simnibs_env/bin/python -m pip show tide-pipeline | grep -E "Location|Editable"

"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """Make ``src/`` importable when running from a source checkout."""
    src_path = Path(__file__).resolve().parent / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    _bootstrap_src_path()
    from tide.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
