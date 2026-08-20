#!/usr/bin/env python3
"""Validate release metadata that must stay synchronized before publishing."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src" / "tide" / "__init__.py"
CITATION_FILE = ROOT / "CITATION.cff"
README_FILE = ROOT / "README.md"


def _extract(pattern: str, text: str, *, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read {label}.")
    return match.group(1).strip()


def package_version() -> str:
    return _extract(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        VERSION_FILE.read_text(encoding="utf-8"),
        label="src/tide/__init__.py version",
    )


def citation_version() -> str:
    value = _extract(
        r'^version:\s*["\']?([^"\'\n]+)["\']?\s*$',
        CITATION_FILE.read_text(encoding="utf-8"),
        label="CITATION.cff version",
    )
    return value.strip()


def readme_logo_version() -> str:
    return _extract(
        r"raw\.githubusercontent\.com/marcotag93/TIDE/v([^/]+)/src/tide/assets/logo\.png",
        README_FILE.read_text(encoding="utf-8"),
        label="README.md immutable logo version",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        help="Optional GitHub release tag; must equal v<package-version>.",
    )
    args = parser.parse_args()

    package = package_version()
    citation = citation_version()
    logo = readme_logo_version()

    errors = []
    if citation != package:
        errors.append(
            f"CITATION.cff version ({citation}) does not match package version ({package})."
        )

    if logo != package:
        errors.append(
            f"README.md logo version ({logo}) does not match package version ({package})."
        )

    if args.tag is not None:
        expected_tag = f"v{package}"
        if args.tag != expected_tag:
            errors.append(
                f"GitHub release tag ({args.tag}) does not match expected tag ({expected_tag})."
            )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Release metadata OK: version={package}")
    if args.tag is not None:
        print(f"Release tag OK: {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
