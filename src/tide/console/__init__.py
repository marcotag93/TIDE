"""
TIDE Console UI Package
=======================

Rich terminal console UI for the grid search workflow.

This package provides a professional terminal-based interface for monitoring
parallel grid search execution, with real-time progress visualization,
worker status tracking, and optional focus mode for detailed worker output.

Main Components:
    ConsoleUI: Main orchestrator class for the console UI
    StatusReporter: Helper class for workers to report status
    create_console_ui: Factory function to create appropriate UI instance

Example Usage:
    from tide.console import ConsoleUI, create_console_ui

    # Using factory (recommended - handles non-interactive gracefully)
    ui = create_console_ui(
        subject_id='sub-001',
        num_workers=4,
        total_points=25,
        enabled=True
    )

    with ui:
        # Workers send status to ui.status_queue
        # UI automatically renders progress
        pass

    ui.render_final_summary(results, elapsed_time)

    # Or with ConsoleUI directly
    ui = ConsoleUI(subject_id='sub-001', num_workers=4, total_points=25)
    with ui:
        # ... processing ...
        pass

Worker Usage:
    from tide.console import StatusReporter, WorkerPhase

    def worker_function(task, status_queue, worker_id):
        reporter = StatusReporter(status_queue, worker_id)
        reporter.started(task.point_label)

        reporter.phase(WorkerPhase.OPTIMIZATION)
        # ... do optimization ...
        reporter.progress(100)

        reporter.phase(WorkerPhase.FEM_SIMULATION)
        # ... do simulation ...
        reporter.progress(100)

        reporter.completed({'weighted_mso': 45.2, 'unweighted_mso': 47.1})
"""

from .console_ui import (
    WORKFLOW_STEPS,
    ConsoleUI,
    NullConsoleUI,
    UIState,
    WorkerState,
    create_console_ui,
)
from .estimation_reporter import process_pipeline_task_with_reporting
from .ipc import (
    PHASE_ORDER,
    MessageType,
    NullStatusReporter,
    StatusReporter,
    WorkerMessage,
    WorkerPhase,
    create_status_reporter,
    phase_to_percent,
)
from .renderer import Renderer, TableColumn
from .styles import (
    BoxChars,
    Colors,
    ProgressChars,
    StatusIcons,
    Symbols,
    phase_color,
    status_icon,
    supports_colors,
    supports_unicode,
)
from .terminal import Terminal, get_terminal
from .worker_reporter import (
    WorkerLoggingHandler,
    process_grid_point_with_reporting,
    setup_worker_logging,
)

__all__ = [
    # Main UI
    "ConsoleUI",
    "NullConsoleUI",
    "create_console_ui",
    "WorkerState",
    "UIState",
    "WORKFLOW_STEPS",
    # IPC
    "MessageType",
    "WorkerMessage",
    "WorkerPhase",
    "StatusReporter",
    "NullStatusReporter",
    "create_status_reporter",
    "PHASE_ORDER",
    "phase_to_percent",
    # Rendering
    "Renderer",
    "TableColumn",
    # Terminal
    "Terminal",
    "get_terminal",
    # Styles
    "Colors",
    "BoxChars",
    "ProgressChars",
    "StatusIcons",
    "Symbols",
    "supports_unicode",
    "supports_colors",
    "status_icon",
    "phase_color",
    # Worker
    "process_grid_point_with_reporting",
    "setup_worker_logging",
    "WorkerLoggingHandler",
    # Estimation
    "process_pipeline_task_with_reporting",
]
