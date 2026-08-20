"""
TIDE Pipeline - CLI Entry Point
================================

This module provides the ``tide`` console script registered in
``pyproject.toml``. It runs the headless workflows and ensures the SimNIBS
python environment is active before importing simnibs-dependent code.

Examples:
    tide --config config.yml --workflow estimation         # Estimation
    tide --config config.yml --workflow grid               # Grid search
    tide --init-config config.yml                             # Write annotated config template
    tide --version                                          # Version + author + research-use notice
    tide --help                                             # Show help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from tide.utils import simnibs_env

if TYPE_CHECKING:
    from tide.utils.config import SimNIBSConfig

_RELAUNCH_MARKER = "_TIDE_SIMNIBS_RELAUNCHED"

_SIMNIBS_INSTALL_URL = "https://simnibs.github.io/simnibs/"

# Numerics-critical dependencies whose versions must match between the SimNIBS
# environment and tide's pins before installing into it. A mismatch would either
# break SimNIBS or shift the pipeline's frozen numerics.
_CORE_PACKAGES = ("numpy", "scipy", "nibabel", "dipy", "pandas")

# Valid workflow selections. Single source of truth for the --workflow flag and
# the top-level `workflow` config entry.
WORKFLOW_CHOICES = ("estimation", "grid", "simulation", "optimization")


def _config_template_text() -> str:
    """Return the bundled configuration template as text.

    Installed wheels include ``config_template.yml`` as ``tide/data`` package
    data. Editable/source checkouts fall back to the repository-root template,
    which remains the single canonical source file.
    """
    try:
        resource = resources.files("tide").joinpath("data").joinpath("config_template.yml")
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    source_template = _repo_root() / "config_template.yml"
    if source_template.is_file():
        return source_template.read_text(encoding="utf-8")

    raise FileNotFoundError(
        "The bundled TIDE configuration template could not be located. "
        "Reinstall tide-pipeline or use a complete source checkout."
    )


def write_config_template(destination: Path) -> Path:
    """Write a fresh configuration template without overwriting existing files."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing configuration: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_config_template_text(), encoding="utf-8")
    return destination


def _locate_simnibs_python() -> Path:
    """Return the absolute path to the SimNIBS-bundled Python interpreter.

    Exits the process with an actionable message if SimNIBS or its python
    cannot be found.
    """
    simnibs_root = simnibs_env.simnibs_root()
    if simnibs_root is None:
        print("Error: 'simnibs' command not found in PATH.", file=sys.stderr)
        print("Please ensure SimNIBS is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)

    candidates = simnibs_env.python_candidates(simnibs_root)
    simnibs_python = simnibs_env.select_python(candidates)
    if simnibs_python is not None:
        return simnibs_python

    print(f"Error: Could not find python in SimNIBS: {simnibs_root}", file=sys.stderr)
    print("Searched paths:", file=sys.stderr)
    for candidate in candidates:
        print(f"  - {candidate}", file=sys.stderr)
    sys.exit(1)


def _locate_simnibs_pip(simnibs_python: Path) -> Path:
    """Return the pip executable next to the SimNIBS python interpreter."""
    candidates = simnibs_env.pip_candidates(simnibs_python)
    pip = simnibs_env.select_pip(candidates)
    if pip is not None:
        return pip
    print(
        f"Error: Could not find pip alongside SimNIBS python: {simnibs_python}",
        file=sys.stderr,
    )
    sys.exit(1)


def _repo_root() -> Path:
    """Return the repository root (parent of the ``src/`` directory)."""
    return Path(__file__).resolve().parents[2]


def _source_checkout_src_dir():
    """Return the ``src/`` dir if tide runs from a source checkout, else None.

    A source checkout has a ``src/tide`` layout with a sibling ``pyproject.toml``
    and is not located under a ``site-packages``/``dist-packages`` directory.
    """
    import tide as _tide_pkg

    tide_parent = Path(_tide_pkg.__file__).resolve().parent.parent
    lowered = {part.lower() for part in tide_parent.parts}
    if "site-packages" in lowered or "dist-packages" in lowered:
        return None
    if (tide_parent.parent / "pyproject.toml").exists():
        return tide_parent
    return None


def _tide_importable_under(python: Path, env: dict = None) -> bool:
    """Return True if ``import tide`` succeeds under the given interpreter.

    ``env`` should be the environment the relaunch will use so the probe sees the
    same import path (no injected PYTHONPATH), reflecting the target env's own
    packages rather than the caller's.
    """
    try:
        result = subprocess.run(
            [str(python), "-c", "import tide"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        return result.returncode == 0
    except Exception:
        return False


def _pins_from_pyproject(pyproject_path: Path) -> dict:
    """Parse ``==`` pins for the core packages from ``[project.dependencies]``."""
    pins: dict = {}
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return pins
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    scope = block.group(1) if block else text
    for name, version in re.findall(r'"([A-Za-z0-9_.-]+)==([^"\s;]+)"', scope):
        if name.lower() in _CORE_PACKAGES:
            pins[name.lower()] = version
    return pins


def _expected_core_pins() -> dict:
    """Return ``{package: pinned_version}`` for the numerics-critical core.

    Prefers tide's installed distribution metadata (the authoritative pins);
    falls back to the source-checkout ``pyproject.toml`` when tide is not
    installed as a distribution.
    """
    from importlib import metadata

    pins: dict = {}
    try:
        requirements = metadata.requires("tide-pipeline") or []
    except metadata.PackageNotFoundError:
        requirements = []

    for requirement in requirements:
        if ";" in requirement:  # skip extras (viz/dev markers)
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$", requirement.strip())
        if match and match.group(1).lower() in _CORE_PACKAGES:
            pins[match.group(1).lower()] = match.group(2)

    if len(pins) < len(_CORE_PACKAGES):
        src_dir = _source_checkout_src_dir()
        if src_dir is not None:
            for name, version in _pins_from_pyproject(src_dir.parent / "pyproject.toml").items():
                pins.setdefault(name, version)
    return pins


def _simnibs_core_versions(simnibs_python: Path) -> dict:
    """Return ``{package: version|None}`` for the core packages in the SimNIBS env."""
    code = (
        "import importlib.metadata as m, json;"
        f"names={list(_CORE_PACKAGES)!r};"
        "out={}\n"
        "for n in names:\n"
        "    try:\n"
        "        out[n]=m.version(n)\n"
        "    except Exception:\n"
        "        out[n]=None\n"
        "print(json.dumps(out))"
    )
    try:
        result = subprocess.run(
            [str(simnibs_python), "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
    except (subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return {name: None for name in _CORE_PACKAGES}


def _verify_simnibs_deps(simnibs_python: Path) -> tuple:
    """Compare SimNIBS-env core versions against tide's pins.

    Returns ``(ok, lines)`` where each line reports a package as match or
    mismatch.
    """
    expected = _expected_core_pins()
    found = _simnibs_core_versions(simnibs_python)

    ok = True
    lines: list = []
    for name in _CORE_PACKAGES:
        want = expected.get(name)
        have = found.get(name)
        if want is None:
            lines.append(f"  {name}: pin unknown (skipped)")
            continue
        if have != want:
            ok = False
        status = "OK" if have == want else "MISMATCH"
        lines.append(f"  {name}: expected {want}, found {have or 'absent'} [{status}]")
    return ok, lines


def _exit_simnibs_import_failed(exc: ImportError) -> None:
    """Exit after a failed simnibs import in the relaunched interpreter."""
    print(
        "Error: Failed to import simnibs even after relaunching with SimNIBS python.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    print(f"\nEnsure SimNIBS 4.5+ is installed: {_SIMNIBS_INSTALL_URL}", file=sys.stderr)
    print("Then install tide into it:  tide --bootstrap", file=sys.stderr)
    sys.exit(1)


def _exit_tide_not_in_simnibs_env(simnibs_python: Path) -> None:
    """Exit when tide is installed outside the SimNIBS env and cannot relaunch."""
    print(
        "Error: tide is installed outside the SimNIBS environment and is not "
        "importable by the SimNIBS python.",
        file=sys.stderr,
    )
    print(f"SimNIBS python: {simnibs_python}", file=sys.stderr)
    print("\nInstall tide into the SimNIBS environment, then re-run:", file=sys.stderr)
    print("  tide --bootstrap", file=sys.stderr)
    sys.exit(1)


def _build_relaunch_env(simnibs_python: Path) -> dict:
    """Build a clean environment dict for re-execing under SimNIBS python.

    Injects a PYTHONPATH that points at the tide source directory so the
    relaunched interpreter can import ``tide`` even if the package is not
    installed in the SimNIBS environment.
    """
    env = os.environ.copy()
    env[_RELAUNCH_MARKER] = "1"

    for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(var, None)

    # Only inject PYTHONPATH for a source checkout (src/ layout with a sibling
    # pyproject.toml). For an installed package that directory is the venv's
    # site-packages; injecting it into the SimNIBS python would shadow SimNIBS's
    # own numpy/scipy and corrupt the numerics.
    src_dir = _source_checkout_src_dir()
    if src_dir is not None:
        env["PYTHONPATH"] = str(src_dir)

    if sys.platform == "win32":
        simnibs_env_root = simnibs_python.parent.parent
        simnibs_env_dir = simnibs_python.parent
        simnibs_scripts = simnibs_env_root / "Scripts"
        simnibs_lib = simnibs_env_root / "Library" / "bin"
        simnibs_dll = simnibs_env_root / "DLLs"

        current_path = env.get("PATH", "")
        filtered_paths = []
        for entry in current_path.split(";"):
            entry_lower = entry.lower()
            if any(m in entry_lower for m in ("python", "anaconda", "miniconda", "conda")):
                if "simnibs" in entry_lower:
                    filtered_paths.append(entry)
            else:
                filtered_paths.append(entry)

        new_paths = [str(simnibs_env_dir), str(simnibs_scripts), str(simnibs_lib), str(simnibs_dll)]
        env["PATH"] = ";".join(new_paths) + ";" + ";".join(filtered_paths)
        env["CONDA_PREFIX"] = str(simnibs_env_root)

    return env


def ensure_simnibs_environment() -> None:
    """Ensure the running interpreter has access to the simnibs package.

    If simnibs cannot be imported, locate the SimNIBS python and re-exec the
    current command under it. Uses an env-var marker to prevent infinite loops.
    """
    already_relaunched = os.environ.get(_RELAUNCH_MARKER) == "1"

    try:
        import simnibs  # noqa: F401

        return
    except ImportError as exc:
        if already_relaunched:
            _exit_simnibs_import_failed(exc)

    print("--- Locating SimNIBS environment... ---", file=sys.stderr)
    simnibs_python = _locate_simnibs_python()

    env = _build_relaunch_env(simnibs_python)

    # No PYTHONPATH means tide is an installed package (not a source checkout)
    if "PYTHONPATH" not in env and not _tide_importable_under(simnibs_python, env):
        _exit_tide_not_in_simnibs_env(simnibs_python)

    print(f"--- Relaunching via: {simnibs_python} ---", file=sys.stderr)
    cmd = [str(simnibs_python), sys.argv[0]] + sys.argv[1:]

    try:
        if sys.platform == "win32":
            result = subprocess.run(cmd, env=env)
            sys.exit(result.returncode)
        else:
            os.execve(str(simnibs_python), cmd, env)
    except Exception as exc:
        print(f"Failed to relaunch: {exc}", file=sys.stderr)
        sys.exit(1)


def run_bootstrap(editable: bool = True, force: bool = False) -> None:
    """Install the tide package into the detected SimNIBS python environment.

    This is an opt-in convenience for users who installed ``tide`` under a
    different interpreter (system pip, pyenv, conda, ...) and want it available
    under the SimNIBS-bundled python. Before installing, the SimNIBS env's
    numerics-critical dependency versions are verified against tide's pins; a
    mismatch aborts (use ``force`` to override) so SimNIBS packages are never
    silently changed. The exact pip command is printed before execution.
    """
    print("--- Locating SimNIBS environment... ---", file=sys.stderr)
    simnibs_python = _locate_simnibs_python()

    print("--- Verifying SimNIBS dependency versions against tide pins ---", file=sys.stderr)
    ok, lines = _verify_simnibs_deps(simnibs_python)
    for line in lines:
        print(line, file=sys.stderr)
    if not ok:
        if not force:
            print(
                "\nError: SimNIBS dependency versions differ from tide's pins; "
                "installing could change SimNIBS packages and shift the pipeline "
                "numerics.",
                file=sys.stderr,
            )
            print("Re-run 'tide --bootstrap --force' to override.", file=sys.stderr)
            sys.exit(1)
        print("\n--force set: proceeding despite the mismatch above.", file=sys.stderr)

    src_dir = _source_checkout_src_dir()
    # Always invoke pip as a module of the exact SimNIBS interpreter. Calling a
    # standalone ``pip`` script can target a different Python when PATHs, user
    # installs, or stale environment launchers overlap.
    install_args: list = [str(simnibs_python), "-m", "pip", "install"]
    if src_dir is not None:
        if editable:
            install_args.append("-e")
        install_args.append(str(src_dir.parent))
    else:
        from tide import __version__

        install_args.append(f"tide-pipeline=={__version__}")

    print(f"--- Bootstrapping tide via: {' '.join(install_args)} ---", file=sys.stderr)
    try:
        result = subprocess.run(install_args, check=False)
    except Exception as exc:
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"Bootstrap pip exited with status {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    probe_env = os.environ.copy()
    for var in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        probe_env.pop(var, None)
    if not _tide_importable_under(simnibs_python, probe_env):
        print(
            "Bootstrap installation completed, but 'import tide' still fails "
            f"under {simnibs_python}.",
            file=sys.stderr,
        )
        print(
            "Inspect the installation with: " f"{simnibs_python} -m pip show tide-pipeline",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\ntide installed into {simnibs_python.parent}. "
        "Re-run without --bootstrap to use the pipeline.",
        file=sys.stderr,
    )


def run_cache_command(argv: list) -> None:
    """Handle ``--cache-info`` / ``--cache-clear`` and exit.

    Runs before the SimNIBS relaunch, so it stays standard-library only. When a
    ``--config`` is supplied its ``subject.cache_dir`` is exported to
    ``TIDE_CACHE_DIR`` so the command targets the configured store.
    """
    from tide.utils.artifacts import (
        CACHE_DISABLE_TOKENS,
        cache_total_size,
        clear_cache,
        fixed_pose_cache_root,
        iter_cache_entries,
    )

    config_path = None
    config_requested = False
    for index, arg in enumerate(argv):
        if arg == "--config":
            config_requested = True
            config_path = argv[index + 1] if index + 1 < len(argv) else ""
        elif arg.startswith("--config="):
            config_requested = True
            config_path = arg.split("=", 1)[1]

    if config_requested and not config_path:
        print("Error: Could not read cache configuration: path is missing", file=sys.stderr)
        sys.exit(1)

    if config_path:
        import yaml

        try:
            loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
            raw = {} if loaded is None else loaded
            if not isinstance(raw, dict):
                raise ValueError("configuration root must be a YAML mapping")
            subject = raw.get("subject", {})
            if subject is None:
                subject = {}
            if not isinstance(subject, dict):
                raise ValueError("subject must be a YAML mapping")
            cache_dir = subject.get("cache_dir")
            disabled_scalar = cache_dir is False or (
                isinstance(cache_dir, (int, float))
                and not isinstance(cache_dir, bool)
                and cache_dir == 0
            )
            if cache_dir is not None and not isinstance(cache_dir, str) and not disabled_scalar:
                raise ValueError("subject.cache_dir must be a path or disable token")
            if cache_dir and str(cache_dir).strip().lower() not in CACHE_DISABLE_TOKENS:
                os.environ["TIDE_CACHE_DIR"] = str(Path(cache_dir).expanduser())
        except (OSError, UnicodeError, TypeError, ValueError, yaml.YAMLError) as exc:
            print(
                f"Error: Could not read cache configuration '{config_path}': {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    root = fixed_pose_cache_root()

    if "--cache-clear" in argv:
        removed, freed = clear_cache(root)
        print(f"Fixed-pose cache: removed {removed} entries, freed {freed / 1024**3:.2f} GB")
        print(f"Cache root: {root}")
        return

    entries = iter_cache_entries(root)
    total = cache_total_size(root)
    print(f"Cache root:   {root}")
    print(f"Entries:      {len(entries)}")
    print(f"Total size:   {total / 1024**3:.2f} GB")


class _TideHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Help formatter that colors section headings on TTY streams."""

    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _CYAN = "\033[38;5;44m"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tty = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def start_section(self, heading):
        if self._tty and heading:
            heading = f"{self._BOLD}{self._CYAN}{heading}{self._RESET}"
        super().start_section(heading)


def _styled_description(tty: bool) -> str:
    title = "T.I.D.E. (Tractography-Informed Dose Estimation) Pipeline"
    subtitle = "TMS target intensity estimation via the Activating Function."
    if tty:
        return f"\033[1m\033[38;5;44m{title}\033[0m\n\033[2m{subtitle}\033[0m"
    return f"{title}\n{subtitle}"


def _styled_epilog(tty: bool) -> str:
    bold_cyan = "\033[1m\033[38;5;44m" if tty else ""
    dim = "\033[2m" if tty else ""
    reset = "\033[0m" if tty else ""
    sections = [
        (
            "Workflows",
            [
                ("estimation", "Full TIDE estimation with CST calibration and target optimization"),
                ("grid", "Grid search for optimal target position"),
                ("simulation", "Standard simulation only"),
                ("optimization", "Standard optimization only"),
            ],
        ),
        (
            "Verbosity levels",
            [
                ("quiet", "Minimal output (highlights, warnings, errors only)"),
                ("standard", "Normal output (default)"),
                ("verbose", "Full debug output"),
            ],
        ),
        (
            "Softaxic export",
            [
                (
                    "--stmpx PATH",
                    "Estimation-only template; writes <input_stem>_updated.stmpx",
                ),
                (
                    "rotation",
                    "Softaxic columns are SimNIBS columns 1, 0, and negated 2",
                ),
                ("translation", "Copied directly from matsimnibs column 3"),
            ],
        ),
    ]
    lines: list = []
    key_width = max(len(key) for _, items in sections for key, _ in items)
    for title, items in sections:
        lines.append(f"{bold_cyan}{title}:{reset}")
        for key, value in items:
            lines.append(f"  {key.ljust(key_width)}  {value}")
        lines.append("")

    examples = [
        ("tide --config config.yml --workflow estimation", "Full estimation"),
        (
            "tide --config config.yml --workflow estimation --stmpx session.stmpx",
            "Full estimation with automatic Softaxic export",
        ),
        ("tide --config config.yml --workflow grid", "Grid search"),
        ("tide --config config.yml --workflow simulation", "Standard simulation"),
        ("tide --config config.yml --verbosity quiet", "Quiet run"),
        ("tide --init-config config.yml", "Write an annotated configuration template"),
        ("tide --version", "Show version, author, and research-use information"),
    ]
    lines.append(f"{bold_cyan}Examples:{reset}")
    for cmd, label in examples:
        lines.append(f"  {dim}# {label}{reset}")
        lines.append(f"  {cmd}")
        lines.append("")

    from tide.banner import format_author_block, format_research_use_block

    lines.append(format_author_block(tty))
    lines.append("")
    lines.append(format_research_use_block(tty))

    return "\n".join(lines).rstrip() + "\n"


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    tty = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    parser = argparse.ArgumentParser(
        prog="tide",
        description=_styled_description(tty),
        formatter_class=_TideHelpFormatter,
        epilog=_styled_epilog(tty),
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.yml (presence triggers headless mode)",
    )
    parser.add_argument(
        "--stmpx",
        type=Path,
        help=(
            "Softaxic STMPX template to update after a successful estimation. "
            "Writes <input_stem>_updated.stmpx beside the input."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=list(WORKFLOW_CHOICES),
        help="Workflow selection (overrides the config `workflow` entry)",
    )
    # Deprecated no-op: retained so existing scripts passing --no-gui keep working.
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verbosity",
        choices=["quiet", "standard", "verbose"],
        default="standard",
        help="Output verbosity level (default: standard)",
    )
    parser.add_argument(
        "--no-console-ui",
        action="store_true",
        help="Disable rich console UI for grid search (use simple logging)",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version, author, and research-use information, and exit",
    )
    parser.add_argument(
        "--init-config",
        nargs="?",
        const=Path("config.yml"),
        type=Path,
        metavar="PATH",
        help=(
            "Write the bundled annotated configuration template and exit "
            "(default path: ./config.yml). Existing files are never overwritten."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "Install tide into the detected SimNIBS python environment and exit. "
            "Verifies the SimNIBS env's core dependency versions match tide's "
            "pins first. Useful when 'tide' was installed under a different "
            "interpreter."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --bootstrap, install even if dependency versions mismatch.",
    )
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Print fixed-pose cache root, entry count, and total size, then exit.",
    )
    parser.add_argument(
        "--cache-clear",
        action="store_true",
        help="Remove all fixed-pose cache entries and exit.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the fixed-pose cache for this run (no lookup or publication).",
    )

    return parser


def _prepare_anatomy(config: SimNIBSConfig) -> None:
    """Ensure the T1w file exists at the derivatives root."""
    source = config.subject.t1w_path
    is_compressed_nifti = source.name.lower().endswith(".nii.gz")
    suffix = ".nii.gz" if is_compressed_nifti else source.suffix
    dest = config.subject.derivatives_path / f"t1w{suffix}"
    if not dest.exists():
        shutil.copy(source, dest)
    if is_compressed_nifti:
        legacy_dest = config.subject.derivatives_path / "t1w.gz"
        if not legacy_dest.exists():
            try:
                os.link(dest, legacy_dest)
            except OSError:
                shutil.copy(dest, legacy_dest)


def run_headless(args: argparse.Namespace) -> None:
    """Run the pipeline in headless (CLI) mode."""
    start_time = time.time()

    from tide.banner import render_logo

    render_logo()

    if not args.config:
        print("Error: --config is required for headless mode.", file=sys.stderr)
        print(
            "Usage: tide --config <config.yml> [--workflow estimation|grid|simulation|optimization]",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    from tide.utils.config import SimNIBSConfig, orientation_is_matrix, validate_workflow_config
    from tide.utils.logging import setup_logging

    log = logging.getLogger(__name__)

    try:
        config = SimNIBSConfig.from_yaml(args.config)
    except Exception as exc:
        print(f"Configuration Error: {exc}")
        sys.exit(1)

    # Resolve the effective workflow: the --workflow flag overrides the
    # top-level `workflow` config entry; the resolved value then drives the
    # identical dispatch below regardless of its source.
    workflow = args.workflow if args.workflow is not None else config.workflow
    if workflow is not None and workflow not in WORKFLOW_CHOICES:
        print(
            f"Configuration Error: invalid workflow '{workflow}'. "
            f"Valid choices: {', '.join(WORKFLOW_CHOICES)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if workflow is None:
        workflow = "estimation" if config.target.bundle_path else "simulation"

    stmpx_path = getattr(args, "stmpx", None)
    if stmpx_path is not None and workflow != "estimation":
        print(
            "Configuration Error: --stmpx is supported only with the estimation workflow.",
            file=sys.stderr,
        )
        sys.exit(1)
    if stmpx_path is not None:
        from tide.interfaces.stmpx import validate_stmpx_input

        try:
            validate_stmpx_input(stmpx_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Configuration Error: {exc}", file=sys.stderr)
            sys.exit(1)

    # Grid explores scalp positions around the target and uses target.orientation
    # as the per-point pos_ydir seed; a 4x4 matrix cannot seed that search.
    if workflow == "grid" and orientation_is_matrix(config.grid.orientation):
        print(
            "Configuration Error: the grid workflow needs a coordinate seed for "
            "orientation, but experiment.target.orientation is a 4x4 matrix, "
            "which cannot seed the per-point optimization.\n"
            "Suggestion: set experiment.target.orientation to an [x, y, z] vector "
            'or an EEG label (e.g. "F3"), or run --workflow estimation to use the '
            "matrix as a fixed coil pose.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        validate_workflow_config(config, workflow)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration Error: {exc}", file=sys.stderr)
        sys.exit(1)

    config.subject.derivatives_path.mkdir(parents=True, exist_ok=True)
    setup_logging(config.subject.derivatives_path, config.subject.id, args.verbosity)

    # Disable the fixed-pose cache when requested (--no-cache or `cache_dir: no`).
    # Set before any worker pool spawns so children inherit it (all 4 workflows).
    cache_disabled = getattr(args, "no_cache", False) or config.subject.cache_disabled
    if cache_disabled:
        os.environ["TIDE_FIXED_POSE_CACHE"] = "0"
        log.debug("Fixed-pose cache disabled (--no-cache or cache_dir: no).")

    # Route the optional subject.cache_dir into the fixed-pose cache resolver.
    # Set before any worker pool spawns so children inherit it (all 4 workflows).
    if config.subject.cache_dir is not None:
        os.environ["TIDE_CACHE_DIR"] = str(config.subject.cache_dir)
        log.debug(f"Fixed-pose cache dir set from config: {config.subject.cache_dir}")

    # Opt-in LRU size cap: enforce once in the parent before any pool spawns.
    # No-op when unlimited (default) or the cache is disabled.
    from tide.utils.artifacts import (
        enforce_cache_limit,
        fixed_pose_cache_root,
        resolve_cache_max_bytes,
    )

    max_bytes = (
        None if cache_disabled else resolve_cache_max_bytes(config.subject.cache_max_size_gb)
    )
    if max_bytes is not None:
        evicted, freed = enforce_cache_limit(fixed_pose_cache_root(), max_bytes)
        if evicted:
            log.info(
                f"Fixed-pose cache: evicted {evicted} LRU entr"
                f"{'y' if evicted == 1 else 'ies'}, freed {freed / 1024**3:.2f} GB"
            )

    log.highlight(f"=== TIDE Pipeline - {config.subject.id} ===")

    _prepare_anatomy(config)

    use_console_ui = not getattr(args, "no_console_ui", False)

    try:
        if workflow == "grid":
            from tide.workflows.grid_search import run_grid_search_workflow

            run_grid_search_workflow(config, console_ui=use_console_ui)
        elif workflow == "estimation":
            from tide.workflows.estimation import run_estimation_workflow

            run_estimation_workflow(config, console_ui=use_console_ui)
            if stmpx_path is not None:
                from tide.interfaces.stmpx import export_target_to_stmpx

                summary_path = (
                    config.subject.derivatives_path
                    / f"TIDE_{config.target.label}"
                    / f"TIDE_Results_{config.target.label}.txt"
                )
                stmpx_output = export_target_to_stmpx(
                    stmpx_path,
                    summary_path,
                    dataset_name=config.options.stmpx_dataset_name,
                )
                log.highlight(f"STMPX export: {stmpx_output.resolve()}")
        elif workflow == "simulation":
            from tide.workflows.standard import run_standard_simulation

            run_standard_simulation(config)
        elif workflow == "optimization":
            from tide.workflows.standard import run_standard_optimization

            run_standard_optimization(config)
    except Exception as exc:
        print(f"Pipeline Error: {exc}", file=sys.stderr)
        sys.exit(1)

    duration = time.time() - start_time
    minutes = int(duration // 60)
    seconds = duration % 60

    log.highlight("=== PIPELINE COMPLETE ===")
    log.highlight(f"Total time: {minutes}m {seconds:.1f}s")


def main() -> None:
    """Console-script entry point registered in pyproject.toml."""
    argv = sys.argv[1:]

    if any(arg in ("-v", "--version") for arg in argv):
        from tide.banner import render_version

        render_version()
        sys.exit(0)

    if any(arg in ("-h", "--help") for arg in argv):
        from tide.banner import render_logo

        render_logo()
        parser = create_argument_parser()
        parser.print_help()
        sys.exit(0)

    if "--init-config" in argv:
        parser = create_argument_parser()
        args = parser.parse_args(argv)
        try:
            output_path = write_config_template(args.init_config)
        except (FileExistsError, FileNotFoundError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Wrote TIDE configuration template: {output_path}")
        sys.exit(0)

    if "--bootstrap" in argv:
        run_bootstrap(force="--force" in argv)
        sys.exit(0)

    if "--cache-info" in argv or "--cache-clear" in argv:
        run_cache_command(argv)
        sys.exit(0)

    ensure_simnibs_environment()

    parser = create_argument_parser()
    args = parser.parse_args()

    if args.config is None and args.workflow is None:
        from tide.banner import render_logo

        render_logo()
        create_argument_parser().print_help()
        print(
            "\nError: --config is required (with optional --workflow).",
            file=sys.stderr,
        )
        sys.exit(1)

    run_headless(args)


if __name__ == "__main__":
    main()
