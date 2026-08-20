"""
Console UI Module
=================

Main orchestrator for the rich terminal console UI.
Coordinates rendering, input handling, and worker status tracking.
"""

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from multiprocessing.context import BaseContext
from queue import Empty
from typing import Dict, List, Optional

from .ipc import MessageType, WorkerMessage
from .renderer import Renderer, TableColumn
from .styles import Colors
from .terminal import Terminal

# Workflow step names for display
WORKFLOW_STEPS = {
    1: "M1 Calibration",
    2: "CST Analysis",
    3: "Fixed Scalp Center",
    4: "Grid Orientation",
    5: "Grid Generation",
    6: "Grid Search Processing",
    7: "Results Writing",
    8: "Visualization",
}


@dataclass
class WorkerState:
    """
    Track state of a single worker process.

    Attributes:
        worker_id: Worker ID (0-indexed)
        point_label: Grid point being processed
        phase: Current processing phase
        progress: Progress within phase (0-100)
        start_time: When worker started current point
        status: Overall status (idle, running, complete, failed)
    """

    worker_id: int
    point_label: str = ""
    phase: str = "Idle"
    progress: int = 0
    start_time: float = 0.0
    status: str = "idle"
    last_log: str = ""
    log_buffer: List[str] = field(default_factory=list)


class WarningCaptureHandler(logging.Handler):
    """
    Custom handler to capture WARNING logs and attach them to the current step.
    """

    def __init__(self, ui_instance):
        super().__init__()
        self.ui = ui_instance

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            msg = self.format(record)
            # Only capture if in sequential mode and valid step
            if self.ui._state.mode == "sequential":
                step = self.ui._state.current_step
                if step not in self.ui._state.step_warnings:
                    self.ui._state.step_warnings[step] = []
                # Avoid duplicates
                if msg not in self.ui._state.step_warnings[step]:
                    self.ui._state.step_warnings[step].append(msg)


@dataclass
class UIState:
    """
    Complete UI state for rendering.

    Attributes:
        subject_id: Subject identifier
        workflow_name: Name of current workflow
        current_step: Current step number
        total_steps: Total number of steps
        num_workers: Number of worker processes
        total_points: Total grid points to process
        completed_points: Number of completed points
        workers: Dict of worker states by ID
        completed_results: List of completed results
        focus_mode: Whether focus mode is active
        focused_worker: Which worker is focused (0-indexed)
        start_time: When UI started
        mode: UI mode ('sequential' or 'parallel')
        sequential_step_name: Name of current sequential step
        sequential_step_status: Status of sequential step
        sequential_spinner_frame: Animation frame for spinner
        completed_steps: List of completed step numbers
    """

    subject_id: str = ""
    workflow_name: str = "Grid Search"
    current_step: int = 1
    total_steps: int = 8
    num_workers: int = 0
    total_points: int = 0
    completed_points: int = 0
    workers: Dict[int, WorkerState] = field(default_factory=dict)
    completed_results: List[dict] = field(default_factory=list)
    focus_mode: bool = False
    focused_worker: int = 0
    scroll_offset: int = 0
    start_time: float = field(default_factory=time.time)
    mode: str = "sequential"
    sequential_step_name: str = ""
    sequential_step_status: str = "pending"
    sequential_spinner_frame: int = 0
    sequential_step_detail: str = ""
    completed_steps: List[int] = field(default_factory=list)
    step_warnings: Dict[int, List[str]] = field(default_factory=dict)


class ConsoleUI:
    """
    Rich terminal console UI for grid search workflow.

    Provides real-time visualization of worker status and progress
    during parallel grid search execution.

    Features:
    - Real-time progress visualization
    - Worker status table with phases
    - Completed results list
    - Focus mode for individual worker output
    - Graceful fallback for non-interactive terminals

    Usage:
        ui = ConsoleUI(
            subject_id='sub-001',
            num_workers=4,
            total_points=25
        )

        with ui:
            # Workers send status to ui.status_queue
            # UI automatically renders progress
            pass

        # After completion:
        ui.render_final_summary(results, elapsed_time)
    """

    # Default table columns for worker display
    WORKER_COLUMNS = [
        TableColumn("Worker", 10, "center", "worker"),
        TableColumn("Grid Point", 12, "center", "grid_point"),
        TableColumn("Current Phase", 18, "left", "current_phase"),
        TableColumn("Last Activity", 30, "left", "last_log"),
        TableColumn("Time", 10, "center", "time"),
    ]

    def __init__(
        self,
        subject_id: str,
        num_workers: int = 0,
        total_points: int = 0,
        current_step: int = 1,
        total_steps: int = 7,
        workflow_name: str = "Grid Search",
        mode: str = "sequential",
        step_names: Optional[Dict[int, str]] = None,
        mp_context: Optional[BaseContext] = None,
    ):
        """
        Initialize console UI.

        Args:
            subject_id: Subject identifier for header display
            num_workers: Number of parallel workers (0 for sequential mode)
            total_points: Total grid points to process
            current_step: Current workflow step number
            total_steps: Total workflow steps
            workflow_name: Name of current workflow
            mode: UI mode ('sequential' or 'parallel')
            step_names: Optional dictionary of step names (step_num -> name)
            mp_context: Optional multiprocessing context
        """
        self._terminal = Terminal()

        # Use provided step names or default
        self._step_names = step_names if step_names else WORKFLOW_STEPS

        self._state = UIState(
            subject_id=subject_id,
            workflow_name=workflow_name,
            current_step=current_step,
            total_steps=total_steps,
            num_workers=num_workers,
            total_points=total_points,
            mode=mode,
            sequential_step_name=self._step_names.get(current_step, f"Step {current_step}"),
        )

        # Initialize worker states for parallel mode
        for i in range(num_workers):
            self._state.workers[i] = WorkerState(worker_id=i)

        # IPC queue for status messages from workers
        if mp_context is None:
            self._mp_manager = None
            self._status_queue = Queue()
        else:
            self._mp_manager = mp_context.Manager()
            self._status_queue = self._mp_manager.Queue()

        # Rendering
        cols, _ = self._terminal.get_size()
        self._renderer = Renderer(terminal_width=cols)

        # Control flags
        self._running = False
        self._render_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Maximum log lines to cache per worker
        self._max_log_lines = 100

        # Signal handling
        self._original_sigint = None

    @property
    def status_queue(self) -> Queue:
        """Get the status queue for workers to send updates."""
        return self._status_queue

    @property
    def is_interactive(self) -> bool:
        """Check if running in interactive terminal."""
        return self._terminal.is_interactive

    @property
    def num_workers(self) -> int:
        """Get number of workers."""
        return self._state.num_workers

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def start(self) -> None:
        """Start the UI rendering loop."""
        if not self.is_interactive:
            return  # Fallback to simple logging

        self._running = True
        self._state.start_time = time.time()
        self._shutdown_event.clear()

        # Setup terminal for UI rendering (alternate buffer, hide cursor)
        self._terminal.setup()

        # Setup resize handler (sets flag in Terminal)
        self._terminal.setup_resize_handler(True)

        # Setup signal handler for graceful shutdown
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        # Add warning capture handler
        self._warning_handler = WarningCaptureHandler(self)
        self._warning_handler.setLevel(logging.WARNING)
        # Use a simple formatter for warnings in the UI
        self._warning_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self._warning_handler)

        # Render initial frame
        self._render_frame()

        # Start render thread
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="ConsoleUI-Render"
        )
        self._render_thread.start()

    def stop(self) -> None:
        """Stop the UI and cleanup."""
        self._running = False
        self._shutdown_event.set()

        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=1.0)

        # Cleanup terminal (show cursor, exit alternate buffer)
        if self.is_interactive:
            self._terminal.cleanup()

        # Restore signal handler
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)

        # Remove warning handler
        if hasattr(self, "_warning_handler"):
            logging.getLogger().removeHandler(self._warning_handler)

        if self._mp_manager is not None:
            self._mp_manager.shutdown()
            self._mp_manager = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def _handle_sigint(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self._running = False
        self._shutdown_event.set()
        # Re-raise to allow proper cleanup
        if self._original_sigint:
            self._original_sigint(signum, frame)

    # =========================================================================
    # Step Management Methods
    # =========================================================================

    def update_step(self, step: int, status: str = "running") -> None:
        """
        Update the current sequential step.

        Call this method to update the UI when starting or completing
        a sequential workflow step (Steps 1-5).

        Args:
            step: Step number (1-7)
            status: Step status ('pending', 'running', or 'complete')
        """
        if status == "complete" and step not in self._state.completed_steps:
            self._state.completed_steps.append(step)

        # Force sequential mode when updating step
        self._state.mode = "sequential"

        self._state.current_step = step
        self._state.sequential_step_name = self._step_names.get(step, f"Step {step}")
        self._state.sequential_step_status = status

        if status == "running":
            self._state.start_time = time.time()
            # Clear detail when starting a new step
            self._state.sequential_step_detail = ""

    def update_step_detail(self, detail: str) -> None:
        """
        Update the sub-step detail text shown under the current step.

        Call this method to provide more context about what's happening
        during a sequential step (e.g., "Running M1 optimization...",
        "Loading tractogram...", etc.)

        Args:
            detail: Sub-step detail text to display
        """
        self._state.sequential_step_detail = detail

    def transition_to_parallel(
        self,
        num_workers: int,
        total_points: int,
        step_num: int = 6,
    ) -> None:
        """
        Transition from sequential mode to parallel mode for Step 6.

        Call this method before starting parallel grid search processing
        to switch the UI from sequential step display to worker table view.

        Args:
            num_workers: Number of parallel workers
            total_points: Total grid points to process
            step_num: Step number to associate with parallel processing (default: 6)
        """
        self._state.mode = "parallel"
        self._state.num_workers = num_workers
        self._state.total_points = total_points
        self._state.current_step = step_num
        self._state.completed_points = 0
        self._state.start_time = time.time()
        self._state.sequential_step_name = self._step_names.get(step_num, "Processing")

        # Initialize worker states
        self._state.workers.clear()
        for i in range(num_workers):
            self._state.workers[i] = WorkerState(worker_id=i)

        # Clear step detail when transitioning
        self._state.sequential_step_detail = ""

        # Note: Do NOT call clear_and_home() here - it causes visual
        # "flicker" that makes it appear as two separate UIs.
        # The render loop will naturally update to show the new mode.

    # =========================================================================
    # Render Loop
    # =========================================================================

    def _render_loop(self) -> None:
        """Main rendering loop (runs in separate thread)."""
        last_render = 0
        render_interval = 0.1  # 10 FPS
        spinner_interval = 0.08  # Spinner updates at ~12 FPS
        last_spinner = 0

        while self._running and not self._shutdown_event.is_set():
            # Process incoming messages (parallel mode only)
            if self._state.mode == "parallel":
                self._process_messages()

            # Check for resize event (thread-safe)
            resize_info = self._terminal.check_resize()
            if resize_info:
                self._on_resize(*resize_info)

            # Check for keypress
            key = self._terminal.get_keypress(timeout=0.05)
            if key:
                self._handle_keypress(key)

            # Update spinner for sequential mode
            now = time.time()
            if self._state.mode == "sequential" and now - last_spinner >= spinner_interval:
                self._state.sequential_spinner_frame += 1
                last_spinner = now

            # Render at fixed interval
            if now - last_render >= render_interval:
                self._render_frame()
                last_render = now

    def _process_messages(self) -> None:
        """Process all pending messages from workers."""
        while True:
            try:
                msg_dict = self._status_queue.get_nowait()
                msg = WorkerMessage.from_dict(msg_dict)
                self._handle_message(msg)
            except Empty:
                break
            except Exception:
                break

    def _handle_message(self, msg: WorkerMessage) -> None:
        """Handle a single worker message."""
        worker = self._state.workers.get(msg.worker_id)
        if not worker:
            return

        if msg.msg_type == MessageType.WORKER_STARTED:
            worker.point_label = msg.point_label
            worker.phase = "Starting..."
            worker.start_time = msg.timestamp
            worker.status = "running"
            worker.progress = 0

        elif msg.msg_type == MessageType.PHASE_CHANGED:
            worker.phase = msg.phase
            worker.progress = msg.progress

        elif msg.msg_type == MessageType.PROGRESS_UPDATE:
            worker.progress = msg.progress

        elif msg.msg_type == MessageType.WORKER_COMPLETED:
            self._state.completed_points += 1

            if msg.data:
                self._state.completed_results.append(
                    {
                        "label": msg.point_label,
                        "weighted_mso": msg.data.get("weighted_mso", 0),
                        "unweighted_mso": msg.data.get("unweighted_mso", 0),
                        "success": True,
                    }
                )

            # Reset worker for next task
            worker.point_label = ""
            worker.phase = "Idle"
            worker.status = "idle"
            worker.progress = 0
            worker.start_time = 0
            worker.last_log = "Completed"

        elif msg.msg_type == MessageType.WORKER_FAILED:
            self._state.completed_points += 1

            error_msg = "Failed"
            if msg.data:
                error_msg = msg.data.get("error", "Failed")

            self._state.completed_results.append(
                {
                    "label": msg.point_label,
                    "weighted_mso": 999.9,
                    "unweighted_mso": 999.9,
                    "success": False,
                    "error": error_msg,
                }
            )

            # Reset worker
            worker.point_label = ""
            worker.phase = "Idle"
            worker.status = "idle"
            worker.progress = 0
            worker.start_time = 0
            worker.last_log = f"Failed: {error_msg}"

        elif msg.msg_type == MessageType.LOG_MESSAGE:
            if msg.data:
                level = msg.data.get("level", "INFO")
                message = msg.data.get("message", "")
                log_entry = f"[{level}] {message}"

                # Update worker state
                worker.last_log = message
                worker.log_buffer.append(log_entry)

                # Limit buffer size per worker
                if len(worker.log_buffer) > self._max_log_lines:
                    worker.log_buffer.pop(0)

    def _handle_keypress(self, key: str) -> None:
        """Handle keyboard input."""
        if key.lower() == "f":
            # Toggle focus mode
            self._state.focus_mode = not self._state.focus_mode
            self._terminal.clear_and_home()

        elif key.lower() == "q":
            # Exit focus mode or quit entire pipeline
            if self._state.focus_mode:
                self._state.focus_mode = False
                self._state.scroll_offset = 0
                self._terminal.clear_and_home()
            else:
                # Signal main process to shut down
                os.kill(os.getpid(), signal.SIGINT)

        elif key in "1234567890":
            # Select worker to focus (1-indexed)
            num = int(key)
            worker_id = 9 if num == 0 else num - 1
            if worker_id < self._state.num_workers:
                if self._state.focus_mode:
                    self._state.focused_worker = worker_id
                    self._state.scroll_offset = 0
                    self._terminal.clear_and_home()
                else:
                    self._state.focus_mode = True
                    self._state.focused_worker = worker_id
                    self._state.scroll_offset = 0
                    self._terminal.clear_and_home()

        # Scrolling keys
        elif key == "\x1b[A":  # Up
            self._state.scroll_offset = max(0, self._state.scroll_offset - 1)
        elif key == "\x1b[B":  # Down
            self._state.scroll_offset += 1
        elif key == "\x1b[5~":  # Page Up
            _, rows = self._terminal.get_size()
            self._state.scroll_offset = max(0, self._state.scroll_offset - (rows // 2))
        elif key == "\x1b[6~":  # Page Down
            _, rows = self._terminal.get_size()
            self._state.scroll_offset += rows // 2
        elif key.lower() == "h":  # Home
            self._state.scroll_offset = 0

    # =========================================================================
    # Rendering Methods
    # =========================================================================

    def _render_frame(self) -> None:
        """Render a single frame."""
        if self._state.focus_mode:
            self._render_focus_mode()
        else:
            self._render_overview_mode()

    def _render_overview_mode(self) -> None:
        """Render the overview mode (sequential or parallel)."""
        lines = []

        # Header box (same for both modes)
        lines.append(
            self._renderer.render_header_box(
                f"TIDE {self._state.workflow_name} Pipeline", f"Subject: {self._state.subject_id}"
            )
        )
        lines.append("")

        if self._state.mode == "sequential":
            # Sequential mode: show step with spinner
            elapsed = time.time() - self._state.start_time
            current_warnings = self._state.step_warnings.get(self._state.current_step, [])

            lines.append(
                self._renderer.render_sequential_step(
                    self._state.current_step,
                    self._state.total_steps,
                    self._state.sequential_step_name,
                    self._state.sequential_step_status,
                    elapsed,
                    self._state.sequential_spinner_frame,
                    detail=self._state.sequential_step_detail,
                    warnings=current_warnings,
                )
            )
            lines.append("")

            # Show completed steps summary
            if self._state.completed_steps:
                lines.append(
                    self._renderer.render_completed_steps(
                        self._state.completed_steps, self._step_names, self._state.step_warnings
                    )
                )
                lines.append("")

            # Status line for sequential mode
            lines.append(self._renderer.render_status_line(elapsed, "q to quit"))
        else:
            # Parallel mode: existing worker table view
            lines.append(
                self._renderer.render_step_indicator(
                    self._state.current_step,
                    self._state.total_steps,
                    self._state.sequential_step_name,
                    self._state.num_workers,
                )
            )
            lines.append("")

            # Show completed steps summary to keep context
            if self._state.completed_steps:
                lines.append(
                    self._renderer.render_completed_steps(
                        self._state.completed_steps, self._step_names, self._state.step_warnings
                    )
                )
                lines.append("")

            # Progress bar
            lines.append(
                self._renderer.render_progress_bar(
                    self._state.completed_points,
                    self._state.total_points,
                    width=50,
                    label="Progress",
                )
            )
            lines.append("")

            # Worker table
            worker_data = self._build_worker_table_data()
            lines.append(self._renderer.render_worker_table(worker_data, self.WORKER_COLUMNS))
            lines.append("")

            # Completed results
            if self._state.completed_results:
                lines.append(
                    self._renderer.render_completed_list(
                        self._state.completed_results, max_items=None
                    )
                )
                lines.append("")

            # Status line for parallel mode
            elapsed = time.time() - self._state.start_time
            lines.append(
                self._renderer.render_status_line(
                    elapsed, "Arrows to scroll | 1-9 to focus | q to quit"
                )
            )

        # Render to terminal
        self._render_lines(lines)

    def _render_focus_mode(self) -> None:
        """Render focus mode showing single worker output."""
        worker = self._state.workers.get(self._state.focused_worker)
        lines = []

        # Focus header
        lines.append(
            self._renderer.render_focus_header(
                self._state.focused_worker,
                worker.point_label if worker else "",
                self._state.num_workers,
            )
        )
        lines.append("")

        # Log output
        _, term_rows = self._terminal.get_size()
        max_log_lines = term_rows - 8  # Reserve space for header/footer

        logs = worker.log_buffer if worker else []
        if logs:
            displayed = logs[-max_log_lines:]
            for line in displayed:
                # Truncate long lines
                cols, _ = self._terminal.get_size()
                if len(line) > cols - 2:
                    line = line[: cols - 5] + "..."
                lines.append(line)
        else:
            lines.append(f"{Colors.GRAY}(Waiting for log output...){Colors.RESET}")

        # Render to terminal
        self._render_lines(lines)

    def _build_worker_table_data(self) -> List[dict]:
        """Build worker data for table rendering."""
        worker_data = []

        for worker_id in sorted(self._state.workers.keys()):
            worker = self._state.workers[worker_id]

            # Calculate elapsed time
            elapsed_str = ""
            if worker.start_time > 0 and worker.status == "running":
                elapsed_sec = time.time() - worker.start_time
                minutes = int(elapsed_sec // 60)
                seconds = int(elapsed_sec % 60)
                elapsed_str = f"{minutes:02d}:{seconds:02d}"

            worker_data.append(
                {
                    "worker": f"Worker {worker_id + 1}",
                    "grid_point": worker.point_label or "-",
                    "current_phase": worker.phase,
                    "last_log": worker.last_log or "-",
                    "time": elapsed_str,
                    "status": worker.status,
                }
            )

        return worker_data

    def _render_lines(self, lines: List[str]) -> None:
        """Render lines to terminal with proper cursor positioning, scrolling, and clipping."""
        _, rows = self._terminal.get_size()

        # Ensure scroll offset is valid
        num_lines = len(lines)
        if num_lines <= rows:
            self._state.scroll_offset = 0
        else:
            self._state.scroll_offset = min(self._state.scroll_offset, num_lines - rows)

        # Slice lines based on scroll offset
        start = self._state.scroll_offset
        end = start + rows
        displayed_lines = lines[start:end]

        # Move to top-left and clear to end of screen
        output = self._terminal.move_cursor(1, 1)
        output += self._terminal.clear_to_end_of_screen()

        # Add content - be careful not to trigger a scroll on the last line
        output += "\n".join(displayed_lines).rstrip("\n")

        # Write to terminal
        self._terminal.write(output)

    def _on_resize(self, cols: int, rows: int) -> None:
        """Handle terminal resize."""
        self._renderer.set_width(cols)
        if self._running:
            self._terminal.clear_and_home()
            self._render_frame()

    # =========================================================================
    # Summary Rendering
    # =========================================================================

    def render_final_summary(
        self,
        results: List[dict],
        elapsed_time: float,
        output_files: Optional[List[tuple]] = None,
        show_table: bool = True,
        stats_summary: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """
        Render final summary after completion.

        Args:
            results: List of result dicts with 'label', 'weighted_mso', etc.
            elapsed_time: Total elapsed time in seconds
            output_files: Optional list of (label, path) tuples for output files
            show_table: Whether to show the detailed results table
            stats_summary: Optional dictionary containing 'weighted' and 'unweighted' stats
        """
        # Ensure UI is stopped
        self.stop()

        # Calculate summary stats
        successes = sum(1 for r in results if r.get("success", True))
        failures = len(results) - successes

        # Format elapsed time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes}m {seconds:.1f}s"

        # Build summary
        print()  # Blank line after UI

        summary_rows = [
            ("Total time", time_str),
            ("Grid points", f"{successes}/{len(results)} successful"),
        ]

        if failures > 0:
            summary_rows.append(("Failed", str(failures)))

        print(self._renderer.render_summary_box("PIPELINE COMPLETE", summary_rows))

        # Render statistical summary if provided
        if stats_summary:
            print()
            stats_cols = [
                TableColumn("Metric", 25, "left", "metric"),
                TableColumn("Unweighted I", 20, "right", "unweighted"),
                TableColumn("Weighted I", 20, "right", "weighted"),
            ]

            w_stats = stats_summary.get("weighted", {})
            u_stats = stats_summary.get("unweighted", {})

            # Prepare rows
            stats_data = [
                {
                    "metric": "Mean",
                    "unweighted": f"{u_stats.get('mean', 0):.2f}",
                    "weighted": f"{w_stats.get('mean', 0):.2f}",
                    "status": "idle",  # for default white color
                },
                {
                    "metric": "Median",
                    "unweighted": f"{u_stats.get('median', 0):.2f}",
                    "weighted": f"{w_stats.get('median', 0):.2f}",
                    "status": "idle",
                },
                {
                    "metric": "Std Dev",
                    "unweighted": f"{u_stats.get('std', 0):.2f}",
                    "weighted": f"{w_stats.get('std', 0):.2f}",
                    "status": "idle",
                },
                {
                    "metric": "Mean (w/o outliers)*",
                    "unweighted": f"{u_stats.get('mean_no_outliers', 0):.2f}",
                    "weighted": f"{w_stats.get('mean_no_outliers', 0):.2f}",
                    "status": "idle",
                },
                {
                    "metric": "Outliers (>2 SD)",
                    "unweighted": f"{u_stats.get('outlier_count', 0)}",
                    "weighted": f"{w_stats.get('outlier_count', 0)}",
                    "status": "idle",
                },
            ]

            # Optional multiplier rows. Multiplier (M_CST/M_target) lets the
            # reader rescale intensity to a different RMT without rerunning, so it
            # belongs in the same statistical block as the intensity summary.
            mult_w = stats_summary.get("multiplier_weighted")
            mult_u = stats_summary.get("multiplier_unweighted")
            if mult_w is not None or mult_u is not None:
                mult_w = mult_w or {}
                mult_u = mult_u or {}
                stats_data.extend(
                    [
                        {
                            "metric": "Mean Multiplier",
                            "unweighted": f"{mult_u.get('mean', 0):.4f}",
                            "weighted": f"{mult_w.get('mean', 0):.4f}",
                            "status": "idle",
                        },
                        {
                            "metric": "Median Multiplier",
                            "unweighted": f"{mult_u.get('median', 0):.4f}",
                            "weighted": f"{mult_w.get('median', 0):.4f}",
                            "status": "idle",
                        },
                    ]
                )

            print(self._renderer.render_section_header("Statistical Summary"))
            print(self._renderer.render_worker_table(stats_data, stats_cols))
            print(
                f"  {Colors.GRAY}* Mean excluding values outside [mean \u00b1 2 * std]{Colors.RESET}"
            )
            if mult_w is not None or mult_u is not None:
                print(
                    f"  {Colors.GRAY}Multiplier = M_CST / M_target "
                    f"(I_raw = RMT \u00d7 multiplier){Colors.RESET}"
                )

        # Output files
        if output_files:
            print()
            print(self._renderer.render_output_files_list(output_files))

        # Render results list if requested
        if show_table and results:
            print()
            # Sort results by label if possible
            sorted_results = sorted(results, key=lambda x: x.get("label", ""))
            print(self._renderer.render_completed_list(sorted_results, max_items=None))

        print()


class NullConsoleUI:
    """
    No-op console UI for non-interactive mode.

    Provides the same interface as ConsoleUI but does nothing,
    allowing the same code to work with or without the UI.
    """

    def __init__(
        self,
        *args,
        mp_context: Optional[BaseContext] = None,
        **kwargs,
    ):
        """Initialize null UI (ignores all arguments)."""
        if mp_context is None:
            self._mp_manager = None
            self._status_queue = Queue()
        else:
            self._mp_manager = mp_context.Manager()
            self._status_queue = self._mp_manager.Queue()

    @property
    def status_queue(self) -> Queue:
        """Get status queue (messages are discarded)."""
        return self._status_queue

    @property
    def is_interactive(self) -> bool:
        """Always returns False."""
        return False

    @property
    def num_workers(self) -> int:
        """Returns 0."""
        return 0

    def start(self) -> None:
        """No-op."""
        pass

    def stop(self) -> None:
        """No-op."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def render_final_summary(self, *args, **kwargs) -> None:
        """No-op."""
        pass

    def update_step(self, step: int, status: str = "running") -> None:
        """No-op."""
        pass

    def update_step_detail(self, detail: str) -> None:
        """No-op."""
        pass

    def transition_to_parallel(self, num_workers: int, total_points: int) -> None:
        """No-op."""
        pass


def create_console_ui(
    subject_id: str,
    num_workers: int = 0,
    total_points: int = 0,
    enabled: bool = True,
    mode: str = "sequential",
    mp_context: Optional[BaseContext] = None,
    step_names: Optional[Dict[int, str]] = None,
    **kwargs,
) -> ConsoleUI:
    """
    Factory function to create appropriate console UI.

    Args:
        subject_id: Subject identifier
        num_workers: Number of workers (0 for sequential mode)
        total_points: Total grid points
        enabled: Whether UI should be enabled
        mode: UI mode ('sequential' or 'parallel')
        mp_context: Optional multiprocessing context
        step_names: Optional custom step names
        **kwargs: Additional arguments for ConsoleUI

    Returns:
        ConsoleUI or NullConsoleUI instance
    """
    if not enabled or not sys.stdout.isatty():
        return NullConsoleUI(mp_context=mp_context)

    return ConsoleUI(
        subject_id=subject_id,
        num_workers=num_workers,
        total_points=total_points,
        mode=mode,
        mp_context=mp_context,
        step_names=step_names,
        **kwargs,
    )
