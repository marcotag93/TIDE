"""
Logging Configuration for TIDE Pipeline
========================================
Provides colored console output and file logging with configurable verbosity.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Define standard levels
LOG_FORMAT_VERBOSE = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_FORMAT_STANDARD = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FORMAT_MINIMAL = "%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Custom HIGHLIGHT Level Definition ---
HIGHLIGHT_LEVEL = 25
logging.addLevelName(HIGHLIGHT_LEVEL, "HIGHLIGHT")


def highlight(self, message, *args, **kwargs):
    if self.isEnabledFor(HIGHLIGHT_LEVEL):
        self._log(HIGHLIGHT_LEVEL, message, args, **kwargs)


# Monkey-patch the Logger class to add the .highlight() method
logging.Logger.highlight = highlight


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add ANSI colors to console output."""

    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color

    def __init__(self, fmt: str, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt=datefmt)

        # Create a specific format for HIGHLIGHT that removes the levelname
        highlight_fmt = fmt.replace("%(levelname)s - ", "").replace("%(levelname)s", "")

        self.formatters = {
            logging.DEBUG: logging.Formatter(f"{self.CYAN}{fmt}{self.NC}", datefmt=datefmt),
            logging.INFO: logging.Formatter(f"{self.NC}{fmt}{self.NC}", datefmt=datefmt),
            HIGHLIGHT_LEVEL: logging.Formatter(
                f"{self.GREEN}{self.BOLD}{highlight_fmt}{self.NC}", datefmt=datefmt
            ),
            logging.WARNING: logging.Formatter(f"{self.YELLOW}{fmt}{self.NC}", datefmt=datefmt),
            logging.ERROR: logging.Formatter(f"{self.RED}{fmt}{self.NC}", datefmt=datefmt),
            logging.CRITICAL: logging.Formatter(
                f"{self.RED}{self.BOLD}{fmt}{self.NC}", datefmt=datefmt
            ),
        }
        self.default_formatter = logging.Formatter(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)


class QuietFilter(logging.Filter):
    """Filter to suppress verbose messages from specific modules."""

    QUIET_MODULES = [
        "simnibs",
        "nibabel",
        "dipy",
        "scipy",
        "numpy",
        "matplotlib",
        "pyvista",
        "vtk",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        # Allow all HIGHLIGHT and above
        if record.levelno >= HIGHLIGHT_LEVEL:
            return True

        # Filter out DEBUG from external modules
        if record.levelno <= logging.DEBUG:
            for module in self.QUIET_MODULES:
                if record.name.startswith(module):
                    return False

        return True


def setup_logging(output_dir: Path, subject_id: str, verbosity: str = "standard"):
    """
    Configures the root logger.

    Args:
        output_dir: Directory for log file
        subject_id: Subject ID for log filename
        verbosity: 'quiet', 'standard', or 'verbose'
            - quiet: Only HIGHLIGHT, WARNING, ERROR (minimal output)
            - standard: INFO and above (default)
            - verbose: DEBUG and above (full output)
    """
    log = logging.getLogger()

    # Set base level based on verbosity
    level_map = {"quiet": HIGHLIGHT_LEVEL, "standard": logging.INFO, "verbose": logging.DEBUG}
    log.setLevel(level_map.get(verbosity, logging.INFO))

    # Clear existing handlers to prevent duplicates during re-runs
    if log.hasHandlers():
        log.handlers.clear()

    # Select format based on verbosity
    if verbosity == "quiet":
        fmt = LOG_FORMAT_MINIMAL
    elif verbosity == "verbose":
        fmt = LOG_FORMAT_VERBOSE
    else:
        fmt = LOG_FORMAT_STANDARD

    # 1. Console Handler (Colored)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter(fmt, DATE_FORMAT))
    console.addFilter(QuietFilter())
    log.addHandler(console)

    # 2. File Handler (Always verbose, no color)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{subject_id}_simNIBS_tide.log"

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)  # Always capture everything to file
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT_VERBOSE, DATE_FORMAT))
    log.addHandler(file_handler)

    # Suppress verbose output from third-party libraries
    for lib in ["simnibs", "nibabel", "dipy", "pyvista", "vtk", "matplotlib"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    log.debug(f"Logging initialized. File: {log_file}, Verbosity: {verbosity}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
