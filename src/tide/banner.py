"""
TIDE Pipeline Logo Banner
=========================

Renders the TIDE ANSI logo and metadata blocks (author, version) used by
the CLI ``--help`` and ``--version`` outputs.

The banner renders only on TTY-capable streams that meet the minimum width.
On unsupported terminals (pipes, narrow viewports, missing asset) the helper
silently returns so the pipeline behaves identically to before.
"""

from __future__ import annotations

import os
import shutil
import sys
from importlib.resources import files
from pathlib import Path
from typing import IO, List, Optional

from . import __version__ as VERSION

_LOGO_RESOURCE = files("tide").joinpath("assets/logo.ansi")
LOGO_PATH: Path = Path(str(_LOGO_RESOURCE))
LOGO_NATIVE_WIDTH: int = 50


def _read_logo_text() -> Optional[str]:
    try:
        return _LOGO_RESOURCE.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


LOGO_MIN_WIDTH: int = LOGO_NATIVE_WIDTH

TITLE: str = "TIDE"
SUBTITLE: str = "Tractography-Informed Dose Estimation"
TAGLINE: str = "TMS Target Intensity Estimation via the Activating Function"

AUTHOR_NAME: str = "Marco Tagliaferri"
AUTHOR_ROLE: str = "PhD Candidate in Cognitive Neuroscience"
AUTHOR_AFFILIATION: str = "CIMeC, University of Trento (Italy)"
AUTHOR_EMAILS: tuple = (
    "marco.tagliaferri@unitn.it",
    "marco.tagliaferri93@gmail.com",
)
LICENSE_NAME: str = "GNU GPL v3.0"

RESEARCH_USE_HEADING: str = "Research use only"
RESEARCH_USE_LINES: tuple = (
    "TIDE is intended for research use only. It is not a medical device and",
    "must not be used to diagnose or treat patients, or to guide clinical",
    "decisions. It has not been evaluated by any regulatory authority and holds",
    "no regulatory clearance: neither CE marking under the European Medical",
    "Device Regulation nor United States Food and Drug Administration approval.",
    "Any clinical use would require independent validation and the appropriate",
    "regulatory authorisation.",
)

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[38;5;44m"
_CYAN_BRIGHT = "\033[38;5;51m"
_GRAY = "\033[38;5;245m"
_WHITE = "\033[38;5;255m"


def _stream_supports_ansi(stream: IO[str]) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TIDE_NO_LOGO"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _center(text: str, width: int) -> str:
    pad = max(0, (width - len(text)) // 2)
    return " " * pad + text


def _terminal_width(default: int = 80) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def _render_logo_block(target: IO[str], columns: int) -> bool:
    content = _read_logo_text()
    if content is None:
        return False

    pad = max(0, (columns - LOGO_NATIVE_WIDTH) // 2)
    indent = " " * pad

    target.write("\n")
    for line in content.splitlines():
        target.write(indent + line + "\n")
    target.write("\n")
    return True


def _render_title_block(target: IO[str], columns: int) -> None:
    target.write(f"{_BOLD}{_CYAN_BRIGHT}{_center(TITLE, columns)}{_RESET}\n")
    target.write(f"{_BOLD}{_CYAN}{_center(SUBTITLE, columns)}{_RESET}\n")
    target.write(f"{_DIM}{_GRAY}{_center(TAGLINE, columns)}{_RESET}\n")


def render_logo(stream: Optional[IO[str]] = None) -> bool:
    """Render the ANSI logo + title block. Returns True if rendered."""
    target = stream if stream is not None else sys.stdout

    if not _stream_supports_ansi(target):
        return False

    columns = _terminal_width(LOGO_MIN_WIDTH)
    if columns < LOGO_MIN_WIDTH:
        return False

    if not _render_logo_block(target, columns):
        return False

    _render_title_block(target, columns)
    target.write("\n")
    target.flush()
    return True


def render_version(stream: Optional[IO[str]] = None) -> bool:
    """Render the full version screen: logo, title, version, author. Returns True if rendered."""
    target = stream if stream is not None else sys.stdout
    tty = _stream_supports_ansi(target)
    columns = _terminal_width(LOGO_MIN_WIDTH)

    if tty and columns >= LOGO_MIN_WIDTH:
        _render_logo_block(target, columns)
        _render_title_block(target, columns)
        target.write("\n")
        target.write(f"{_BOLD}{_WHITE}{_center(f'Version {VERSION}', columns)}{_RESET}\n\n")
    else:
        target.write(f"{TITLE} - {SUBTITLE}\n")
        target.write(f"{TAGLINE}\n")
        target.write(f"Version {VERSION}\n\n")

    for line in _author_lines(tty=tty):
        target.write(line + "\n")
    target.write("\n")
    target.write(format_research_use_block(tty=tty) + "\n\n")
    target.flush()
    return True


def _author_lines(tty: bool) -> List[str]:
    bold_cyan = f"{_BOLD}{_CYAN}" if tty else ""
    bold = _BOLD if tty else ""
    dim = _DIM if tty else ""
    reset = _RESET if tty else ""

    lines: List[str] = []
    lines.append(f"{bold_cyan}Author{reset}")
    lines.append(f"  {bold}{AUTHOR_NAME}{reset}")
    lines.append(f"  {AUTHOR_ROLE}")
    lines.append(f"  {AUTHOR_AFFILIATION}")
    lines.append("")
    lines.append(f"{bold_cyan}Contact{reset}")
    for email in AUTHOR_EMAILS:
        lines.append(f"  {email}")
    lines.append("")
    lines.append(f"{dim}Licensed under {LICENSE_NAME}.{reset}")
    return lines


def format_research_use_block(tty: bool) -> str:
    """Research-Use-Only statement shown by ``--version`` and ``--help``."""
    bold_cyan = f"{_BOLD}{_CYAN}" if tty else ""
    dim = _DIM if tty else ""
    reset = _RESET if tty else ""

    lines = [f"{bold_cyan}{RESEARCH_USE_HEADING}:{reset}"]
    lines.extend(f"  {dim}{line}{reset}" for line in RESEARCH_USE_LINES)
    return "\n".join(lines)


def format_author_block(tty: bool) -> str:
    """Author block formatted for inclusion in the argparse epilog."""
    bold_cyan = f"{_BOLD}{_CYAN}" if tty else ""
    bold = _BOLD if tty else ""
    dim = _DIM if tty else ""
    reset = _RESET if tty else ""

    emails = ", ".join(AUTHOR_EMAILS)
    lines = [
        f"{bold_cyan}Author:{reset}",
        f"  {bold}{AUTHOR_NAME}{reset} - {AUTHOR_ROLE}",
        f"  {AUTHOR_AFFILIATION}",
        f"  {emails}",
        "",
        f"{dim}Version {VERSION} - Licensed under {LICENSE_NAME}.{reset}",
    ]
    return "\n".join(lines)
