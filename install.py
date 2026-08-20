#!/usr/bin/env python3
"""
TIDE Pipeline - Cross-Platform Installation Script
===================================================
Convenience wrapper for TIDE installation. For normal pipeline use from a source
checkout, ``--simnibs-env`` is the recommended mode: it detects the SimNIBS-bundled
Python, verifies its dependency versions match TIDE's pins, and installs into that
environment. Without ``--simnibs-env`` the script intentionally installs into the
**current** Python interpreter, which is useful for development and packaging checks.

Equivalent manual commands:
    python -m pip install .          # install into this exact Python
    python -m pip install ".[viz]"   # optional 3D rendering (pyvista, vtk)
    python -m pip install ".[dev]"   # development dependencies

Works on: Windows, Linux, macOS

Usage:
    python install.py --simnibs-env --editable  # Recommended source installation
    python install.py --simnibs-env             # Install into detected SimNIBS env
    python install.py --simnibs-env --viz       # SimNIBS env + viz extras
    python install.py --dev                     # Current interpreter + dev dependencies
    python install.py                           # Current interpreter (advanced)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes (disabled on Windows CMD without ANSI support)
class Colors:
    """Cross-platform terminal colors."""

    def __init__(self):
        # Enable ANSI on Windows 10+
        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                self._enabled = True
            except Exception:
                self._enabled = False
        else:
            self._enabled = True

    @property
    def RED(self):
        return "\033[0;31m" if self._enabled else ""

    @property
    def GREEN(self):
        return "\033[0;32m" if self._enabled else ""

    @property
    def YELLOW(self):
        return "\033[1;33m" if self._enabled else ""

    @property
    def BLUE(self):
        return "\033[0;34m" if self._enabled else ""

    @property
    def NC(self):
        return "\033[0m" if self._enabled else ""


C = Colors()


def print_banner():
    """Print installation banner."""
    print(f"{C.BLUE}╔══════════════════════════════════════════════════════════════╗{C.NC}")
    print(f"{C.BLUE}║        TIDE Pipeline - Installation Script                  ║{C.NC}")
    print(f"{C.BLUE}╚══════════════════════════════════════════════════════════════╝{C.NC}")
    print()


def _python_scripts_dir(target_python: Path) -> Path:
    """Return the directory containing the installed ``tide`` console script.

    ``sysconfig.get_path('scripts')`` covers normal/venv installs, while a
    user install may instead write to ``site.USER_BASE/bin`` (or ``Scripts``
    on Windows). Query both from the target interpreter and prefer the one
    where the generated entry point actually exists.
    """
    code = (
        "import json, site, sys, sysconfig; "
        "suffix = 'Scripts' if sys.platform == 'win32' else 'bin'; "
        "print(json.dumps([sysconfig.get_path('scripts'), "
        "str(__import__('pathlib').Path(site.getuserbase()) / suffix)]))"
    )
    result = subprocess.run(
        [str(target_python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            candidates = [Path(item) for item in json.loads(result.stdout)]
        except (json.JSONDecodeError, TypeError):
            candidates = []
        script_name = "tide.exe" if sys.platform == "win32" else "tide"
        for directory in candidates:
            if (directory / script_name).exists():
                return directory
        if candidates:
            return candidates[0]
    return target_python.parent


def verify_tide_install(target_python: Path) -> tuple[bool, str]:
    """Verify distribution metadata and ``import tide`` under the target Python."""
    code = (
        "import importlib.metadata as m, tide; "
        "print('version=' + m.version('tide-pipeline')); "
        "print('module=' + str(tide.__file__))"
    )
    env = os.environ.copy()
    for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(var, None)
    result = subprocess.run(
        [str(target_python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    detail = (result.stdout + result.stderr).strip()
    return result.returncode == 0, detail


def _resolved_tide_on_path() -> Optional[Path]:
    """Return the ``tide`` command currently selected by PATH, if any."""
    resolved = shutil.which("tide")
    return Path(resolved).resolve() if resolved else None


def print_success(target_python: Path):
    """Print success message and warn about stale PATH launchers."""
    bin_dir = _python_scripts_dir(target_python)
    tide_cmd = (bin_dir / ("tide.exe" if sys.platform == "win32" else "tide")).resolve()
    path_tide = _resolved_tide_on_path()

    print()
    print(f"{C.GREEN}╔══════════════════════════════════════════════════════════════╗{C.NC}")
    print(f"{C.GREEN}║              Installation Complete!                          ║{C.NC}")
    print(f"{C.GREEN}╚══════════════════════════════════════════════════════════════╝{C.NC}")
    print()
    print(f"Target Python: {C.GREEN}{target_python}{C.NC}")
    print("Console script registered:")
    print(f"  {C.GREEN}{tide_cmd}{C.NC}")

    if path_tide is not None and path_tide != tide_cmd:
        print()
        print(
            f"{C.YELLOW}WARNING: your shell currently resolves 'tide' to a different script:{C.NC}"
        )
        print(f"  PATH tide:      {path_tide}")
        print(f"  Installed tide: {tide_cmd}")
        print("This usually means an older launcher from another Python environment is")
        print("earlier on PATH. Run the installed path above directly, or adjust PATH.")
        if sys.platform != "win32":
            print("For Bash, run 'hash -r' after changing/removing a stale launcher.")

    print()
    print("Interpreter-safe checks:")
    print(f'  {target_python} -c "import tide; print(tide.__version__, tide.__file__)"')
    print(f"  {target_python} -m tide --help")
    print()
    print("You can now run the TIDE Pipeline:")
    print()
    print(f"  {C.BLUE}Estimation:{C.NC}     tide --config config.yml --workflow estimation")
    print(f"  {C.BLUE}Grid search:{C.NC}    tide --config config.yml --workflow grid")
    print(f"  {C.BLUE}Create config:{C.NC}  tide --init-config config.yml")
    print(f"  {C.BLUE}Show help:{C.NC}      tide --help")
    print(f"  {C.BLUE}Show version:{C.NC}   tide --version")
    print()


def find_simnibs_binary() -> Path:
    """
    Find the SimNIBS binary/executable in PATH.

    Returns:
        Path to simnibs executable

    Raises:
        FileNotFoundError: If simnibs not found
    """
    # On Windows, look for simnibs.exe or simnibs.cmd
    if sys.platform == "win32":
        candidates = ["simnibs.exe", "simnibs.cmd", "simnibs.bat", "simnibs"]
    else:
        candidates = ["simnibs"]

    for name in candidates:
        simnibs_bin = shutil.which(name)
        if simnibs_bin:
            return Path(simnibs_bin).resolve()

    # Try common installation paths
    common_paths = []

    if sys.platform == "win32":
        # Windows common paths
        user_home = Path.home()
        common_paths = [
            user_home / "SimNIBS-4.5" / "bin" / "simnibs.exe",
            user_home / "SimNIBS-4.5" / "bin" / "simnibs.cmd",
            Path("C:/SimNIBS-4.5/bin/simnibs.exe"),
            Path("C:/SimNIBS-4.5/bin/simnibs.cmd"),
            user_home / "AppData" / "Local" / "SimNIBS" / "bin" / "simnibs.exe",
        ]
    else:
        # Linux/macOS common paths
        user_home = Path.home()
        common_paths = [
            user_home / "SimNIBS-4.5" / "bin" / "simnibs",
            user_home / "simnibs_env" / "bin" / "simnibs",
            Path("/usr/local/SimNIBS-4.5/bin/simnibs"),
            Path("/opt/SimNIBS-4.5/bin/simnibs"),
        ]

    for path in common_paths:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "Could not find 'simnibs' command.\n"
        "Please ensure SimNIBS is installed and added to your PATH.\n"
        "Installation guide: https://simnibs.github.io/simnibs/"
    )


def find_simnibs_python(simnibs_root: Path) -> Path:
    """
    Find Python executable in SimNIBS environment.

    Args:
        simnibs_root: Root directory of SimNIBS installation

    Returns:
        Path to Python executable

    Raises:
        FileNotFoundError: If Python not found
    """
    if sys.platform == "win32":
        # Windows paths
        candidates = [
            simnibs_root / "simnibs_env" / "Scripts" / "python.exe",
            simnibs_root / "simnibs_env" / "python.exe",
            simnibs_root / "python.exe",
        ]
    else:
        # Linux/macOS paths
        candidates = [
            simnibs_root / "simnibs_env" / "bin" / "python3",
            simnibs_root / "simnibs_env" / "bin" / "python",
            simnibs_root / "bin" / "python3",
            simnibs_root / "bin" / "python",
        ]

    for path in candidates:
        if path.exists() and os.access(path, os.X_OK if sys.platform != "win32" else os.F_OK):
            return path.resolve()

    raise FileNotFoundError(
        f"Could not find Python in SimNIBS installation at: {simnibs_root}\n"
        f"Searched:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


def find_simnibs_pip(simnibs_python: Path) -> list:
    """
    Get pip command for SimNIBS environment.

    Args:
        simnibs_python: Path to SimNIBS Python

    Returns:
        List of command parts for pip
    """
    # Use python -m pip for maximum compatibility
    return [str(simnibs_python), "-m", "pip"]


def run_pip(pip_cmd: list, args: list, quiet: bool = False) -> bool:
    """
    Run pip with given arguments.

    Args:
        pip_cmd: Base pip command (list)
        args: Additional pip arguments
        quiet: Suppress output

    Returns:
        True if successful
    """
    cmd = pip_cmd + args
    if quiet:
        cmd.append("--quiet")

    try:
        subprocess.run(cmd, check=True, capture_output=quiet, text=True)
        return True
    except subprocess.CalledProcessError as e:
        if quiet:
            print(f"{C.RED}Error:{C.NC} {e.stderr if e.stderr else e}")
        return False


def verify_simnibs_import(simnibs_python: Path) -> tuple:
    """
    Verify SimNIBS can be imported and get version.

    Returns:
        Tuple of (success, version_string)
    """
    try:
        result = subprocess.run(
            [str(simnibs_python), "-c", "import simnibs; print(simnibs.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, None
    except Exception:
        return False, None


def resolve_simnibs_target(script_dir: Path, force: bool) -> tuple:
    """Locate the SimNIBS python, verify its deps match the pins, return targets.

    Returns ``(simnibs_python, pip_cmd)``. Exits on a version mismatch unless
    ``force`` is set. Reuses ``tide.cli._verify_simnibs_deps`` from the source
    tree so the pin list has a single source of truth.
    """
    try:
        simnibs_bin = find_simnibs_binary()
        simnibs_root = simnibs_bin.parent.parent
        simnibs_python = find_simnibs_python(simnibs_root)
    except FileNotFoundError as exc:
        print(f"{C.RED}Error:{C.NC} {exc}")
        sys.exit(1)

    print(f"  SimNIBS python: {C.GREEN}{simnibs_python}{C.NC}")
    success, simnibs_version = verify_simnibs_import(simnibs_python)
    if success:
        print(f"  SimNIBS version: {C.GREEN}{simnibs_version}{C.NC}")

    sys.path.insert(0, str(script_dir / "src"))
    try:
        from tide.cli import _verify_simnibs_deps
    except Exception as exc:  # pragma: no cover - defensive
        print(f"{C.YELLOW}Warning: could not run dependency verification: {exc}{C.NC}")
        return simnibs_python, find_simnibs_pip(simnibs_python)

    print("  Verifying dependency versions against tide pins...")
    ok, lines = _verify_simnibs_deps(simnibs_python)
    for line in lines:
        print(f"  {line}")
    if not ok and not force:
        print(
            f"{C.RED}Error:{C.NC} SimNIBS dependency versions differ from tide's "
            "pins; installing could change SimNIBS packages and shift numerics."
        )
        print("Re-run with --force to override once you have verified compatibility.")
        sys.exit(1)
    if not ok and force:
        print(f"{C.YELLOW}--force set: proceeding despite the mismatch above.{C.NC}")

    return simnibs_python, find_simnibs_pip(simnibs_python)


def main():
    """Main installation routine."""
    parser = argparse.ArgumentParser(
        description="Install the TIDE Pipeline (python -m pip wrapper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py --simnibs-env --editable  Recommended source installation
  python install.py --simnibs-env             Install into the detected SimNIBS env
  python install.py --simnibs-env --viz       SimNIBS env + optional 3D rendering
  python install.py --dev                     Current interpreter + development dependencies
  python install.py                           Current interpreter (advanced)
        """,
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="Core dependencies only (default; kept for compatibility)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Add optional 3D rendering dependencies (pyvista, vtk)",
    )
    parser.add_argument("--dev", action="store_true", help="Add development dependencies")
    parser.add_argument(
        "--simnibs-env",
        action="store_true",
        dest="simnibs_env",
        help="Install into the detected SimNIBS python environment "
        "(recommended for pipeline use; default: the current interpreter)",
    )
    parser.add_argument(
        "-e", "--editable", action="store_true", help="Editable install (python -m pip install -e)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --simnibs-env, install even if dependency versions mismatch",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()
    quiet = not args.verbose

    print_banner()

    # Directory holding pyproject.toml; also the install target ".".
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)

    extras = []
    if args.viz:
        extras.append("viz")
    if args.dev:
        extras.append("dev")
    target_spec = "." + (f"[{','.join(extras)}]" if extras else "")

    if args.simnibs_env:
        print(f"{C.YELLOW}[1/2]{C.NC} Resolving SimNIBS environment...")
        target_python, pip_cmd = resolve_simnibs_target(script_dir, args.force)
    else:
        target_python = Path(sys.executable)
        pip_cmd = [sys.executable, "-m", "pip"]
        print(f"{C.YELLOW}[1/2]{C.NC} Target: {C.GREEN}{target_python}{C.NC} (current interpreter)")
        print(
            f"{C.YELLOW}Note:{C.NC} this mode does not redirect TIDE into SimNIBS. "
            "For normal source-based pipeline use, prefer "
            "'python install.py --simnibs-env --editable'."
        )

    print(f"{C.YELLOW}[2/2]{C.NC} Installing TIDE Pipeline...")
    print(f"  Spec: {C.BLUE}{target_spec}{C.NC}{' (editable)' if args.editable else ''}")
    print()

    install_args = ["install"]
    if args.editable:
        install_args.append("-e")
    install_args.append(target_spec)
    if not run_pip(pip_cmd, install_args, quiet=quiet):
        print(f"{C.RED}Failed to install TIDE Pipeline{C.NC}")
        sys.exit(1)

    ok, detail = verify_tide_install(target_python)
    if not ok:
        print(f"{C.RED}Installation verification failed.{C.NC}")
        print(f"Target Python: {target_python}")
        if detail:
            print(detail)
        print(
            "The package was not importable by the interpreter used for the "
            "installation. Re-run using the explicit interpreter form "
            "'<python> -m pip install ...' and remove any stale tide script "
            "from another environment."
        )
        sys.exit(1)

    if args.verbose and detail:
        print(detail)
    print_success(target_python)


if __name__ == "__main__":
    main()
