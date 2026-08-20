"""
Worker Reporter Module
======================

Wraps the grid point processing function to add status reporting.
This module provides the bridge between the worker execution and the console UI.
"""

import logging
import os
import sys
from multiprocessing import Queue
from pathlib import Path
from typing import Optional

from .ipc import StatusReporter, WorkerPhase, create_status_reporter


class WorkerLoggingHandler(logging.Handler):
    """
    Custom logging handler that captures log messages for the console UI.

    This handler sends log messages to the status reporter for display
    in focus mode, and optionally writes to a per-worker log file.
    """

    def __init__(
        self,
        reporter: StatusReporter,
        file_path: Optional[Path] = None,
        level: int = logging.DEBUG,
    ):
        """
        Initialize the worker logging handler.

        Args:
            reporter: StatusReporter to send log messages through
            file_path: Optional path for log file output
            level: Minimum log level to capture
        """
        super().__init__(level)
        self._reporter = reporter
        self._file_handler: Optional[logging.FileHandler] = None

        if file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handler = logging.FileHandler(file_path, mode="w")
            self._file_handler.setLevel(logging.DEBUG)
            self._file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.

        Sends to status reporter for focus mode display and
        writes to file if configured.
        """
        try:
            msg = self.format(record)

            # Send to reporter for focus mode
            self._reporter.log(record.levelname, msg)

            # Write to file
            if self._file_handler:
                self._file_handler.emit(record)

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the handler and any file handlers."""
        if self._file_handler:
            self._file_handler.close()
        super().close()


def setup_worker_logging(
    reporter: StatusReporter,
    log_dir: Optional[Path] = None,
    point_label: str = "",
    worker_id: int = 0,
) -> logging.Handler:
    """
    Configure logging for a worker process.

    This redirects log output to the status reporter (for UI) and
    optionally to a per-worker log file.

    Args:
        reporter: StatusReporter for sending log messages
        log_dir: Directory for worker log files (optional)
        point_label: Current grid point label for log filename
        worker_id: Worker ID for log filename

    Returns:
        The configured logging handler (for cleanup)
    """
    # Determine log file path
    log_file = None
    if log_dir:
        log_file = log_dir / f"worker_{worker_id}_{point_label}.log"

    # Create and configure handler
    handler = WorkerLoggingHandler(reporter, log_file)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    )

    # Get root logger and configure
    root = logging.getLogger()

    # Remove console handlers (we're redirecting output)
    handlers_to_remove = [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr)
    ]
    for h in handlers_to_remove:
        root.removeHandler(h)

    # Add our handler
    root.addHandler(handler)

    return handler


class _ConsoleGridPointReporter:
    """
    Forward grid-point stage transitions to the console ``StatusReporter``.

    The physics lives once in ``grid_search.process_grid_point``; this adapter
    maps its stage hooks onto the live UI phases (audit C-003), so the console
    path does not re-implement the numeric pipeline.
    """

    def __init__(self, reporter: StatusReporter):
        self._reporter = reporter

    def optimization(self) -> None:
        self._reporter.phase(WorkerPhase.OPTIMIZATION, 0)

    def simulation(self) -> None:
        self._reporter.phase(WorkerPhase.FEM_SIMULATION, 0)

    def sampling(self) -> None:
        self._reporter.phase(WorkerPhase.EFIELD_SAMPLING, 0)

    def activating_function(self) -> None:
        self._reporter.phase(WorkerPhase.ACTIVATING_FUNCTION, 0)

    def bundle_analysis(self) -> None:
        self._reporter.phase(WorkerPhase.BUNDLE_ANALYSIS, 0)

    def saving_results(self) -> None:
        self._reporter.phase(WorkerPhase.SAVING_RESULTS, 0)

    def progress(self, pct: int) -> None:
        self._reporter.progress(pct)


def process_grid_point_with_reporting(
    task,  # GridPointTask - imported inside to avoid circular deps
    status_queue: Optional[Queue],
    worker_id: int,
    log_dir: Optional[Path] = None,
):
    """
    Process a grid point with status reporting to the console UI.

    This wraps the original process_grid_point function, adding:
    1. Status updates via queue to main process
    2. Log redirection to per-worker files
    3. Phase tracking for UI display

    Args:
        task: GridPointTask containing all processing parameters
        status_queue: Queue for sending status updates (or None)
        worker_id: This worker's ID (0-indexed)
        log_dir: Directory for worker log files (optional)

    Returns:
        GridPointResult with processing results
    """
    # Import here to avoid issues with spawn context and circular imports
    from tide.workflows.grid_search import (
        GridPointResult,
        _configure_worker_environment,
        process_grid_point,
    )

    # CRITICAL: Configure environment before any heavy imports
    _configure_worker_environment()

    # Use persistent worker ID if provided by the process environment/initializer
    actual_worker_id = worker_id
    if worker_id == -1:
        # Try to get from global (set by initializer)
        import tide.workflows.grid_search as gs

        actual_worker_id = getattr(gs, "_process_worker_id", 0)
        if actual_worker_id is None:
            actual_worker_id = 0

    # Create status reporter
    reporter = create_status_reporter(status_queue, actual_worker_id)
    reporter.started(task.point_label)

    saved_stdout_fd = None
    saved_stderr_fd = None
    try:
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"worker_{worker_id}_{task.point_label}_stdout.log"
            stderr_path = log_dir / f"worker_{worker_id}_{task.point_label}_stderr.log"
            stdout_fd = os.open(str(stdout_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            stderr_fd = os.open(str(stderr_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        else:
            stdout_fd = os.open(os.devnull, os.O_WRONLY)
            stderr_fd = os.open(os.devnull, os.O_WRONLY)

        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)

        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)

        os.close(stdout_fd)
        os.close(stderr_fd)

        sys.stdout = open(1, mode="w", buffering=1, closefd=False)
        sys.stderr = open(2, mode="w", buffering=1, closefd=False)
    except Exception:
        if saved_stdout_fd is not None:
            try:
                os.close(saved_stdout_fd)
            except Exception:
                pass
            saved_stdout_fd = None
        if saved_stderr_fd is not None:
            try:
                os.close(saved_stderr_fd)
            except Exception:
                pass
            saved_stderr_fd = None

    # Setup logging
    log_handler = None
    if log_dir:
        log_handler = setup_worker_logging(reporter, log_dir, task.point_label, worker_id)

    # process_grid_point owns the physics; this adapter forwards UI stages.
    log = logging.getLogger(__name__)
    grid_reporter = _ConsoleGridPointReporter(reporter)

    try:
        result = process_grid_point(task, reporter=grid_reporter)

        if result.success:
            reporter.completed(
                {
                    "weighted_mso": result.weighted_mso,
                    "unweighted_mso": result.unweighted_mso,
                    "success": True,
                }
            )
        else:
            reporter.failed(result.error_message or "processing failed")

        return result

    except Exception as e:
        log.error(f"Processing failed for {task.point_label}: {e}")
        reporter.failed(str(e))

        return GridPointResult(
            index=task.index,
            point_label=task.point_label,
            success=False,
            weighted_mso=999.9,
            unweighted_mso=999.9,
            cortex_coord=task.cortex_coord,
            opt_scalp_coords=None,
            opt_matrix=None,
            error_message=str(e),
        )

    finally:
        # Cleanup logging handler
        if log_handler:
            root = logging.getLogger()
            root.removeHandler(log_handler)
            log_handler.close()

        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
            except Exception:
                pass
            try:
                os.close(saved_stdout_fd)
            except Exception:
                pass

        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
            except Exception:
                pass
            try:
                os.close(saved_stderr_fd)
            except Exception:
                pass
