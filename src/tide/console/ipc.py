"""
Inter-Process Communication Module
==================================

Message protocol and queue management for worker status reporting.
Used to communicate between worker processes and the main UI process.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing import Queue
from typing import Any, Dict, List, Optional


class MessageType(Enum):
    """Types of IPC messages from workers to main process."""

    WORKER_STARTED = auto()  # Worker began processing a grid point
    PHASE_CHANGED = auto()  # Worker entered a new processing phase
    PROGRESS_UPDATE = auto()  # Incremental progress within current phase
    WORKER_COMPLETED = auto()  # Worker finished successfully
    WORKER_FAILED = auto()  # Worker finished with error
    LOG_MESSAGE = auto()  # Log message from worker (for focus mode)
    HEARTBEAT = auto()  # Periodic heartbeat for liveness detection


class WorkerPhase(Enum):
    """
    Processing phases for grid point workflow.

    These correspond to the major steps in process_grid_point().
    """

    STARTING = "Starting"
    OPTIMIZATION = "Optimization"
    FEM_SIMULATION = "FEM Simulation"
    EFIELD_SAMPLING = "E-field Sampling"
    ACTIVATING_FUNCTION = "Activating Function"
    BUNDLE_ANALYSIS = "Bundle Analysis"
    SAVING_RESULTS = "Saving Results"
    COMPLETE = "Complete"
    FAILED = "Failed"
    IDLE = "Idle"


# Phase order for progress calculation
PHASE_ORDER: List[WorkerPhase] = [
    WorkerPhase.STARTING,
    WorkerPhase.OPTIMIZATION,
    WorkerPhase.FEM_SIMULATION,
    WorkerPhase.EFIELD_SAMPLING,
    WorkerPhase.ACTIVATING_FUNCTION,
    WorkerPhase.BUNDLE_ANALYSIS,
    WorkerPhase.SAVING_RESULTS,
    WorkerPhase.COMPLETE,
]


def phase_to_percent(phase: WorkerPhase) -> int:
    """
    Convert phase to approximate progress percentage.

    Args:
        phase: Current worker phase

    Returns:
        Progress percentage (0-100)
    """
    try:
        idx = PHASE_ORDER.index(phase)
        return int((idx / (len(PHASE_ORDER) - 1)) * 100)
    except ValueError:
        return 0


@dataclass
class WorkerMessage:
    """
    Message from worker to main process.

    This is the primary communication unit between workers and the
    console UI. Messages are serialized to dict for queue transport.

    Attributes:
        msg_type: Type of message (from MessageType enum)
        worker_id: Which worker sent this (0-indexed)
        timestamp: When message was created (Unix timestamp)
        point_label: Grid point being processed (e.g., 'grid_P00')
        phase: Current processing phase name
        progress: Progress within phase (0-100)
        data: Additional payload (results, errors, log messages)
    """

    msg_type: MessageType
    worker_id: int
    timestamp: float = field(default_factory=time.time)
    point_label: str = ""
    phase: str = ""
    progress: int = 0
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize message for queue transport.

        Returns:
            Dictionary representation of message
        """
        return {
            "msg_type": self.msg_type.value,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "point_label": self.point_label,
            "phase": self.phase,
            "progress": self.progress,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkerMessage":
        """
        Deserialize message from queue.

        Args:
            d: Dictionary from queue

        Returns:
            WorkerMessage instance
        """
        return cls(
            msg_type=MessageType(d["msg_type"]),
            worker_id=d["worker_id"],
            timestamp=d["timestamp"],
            point_label=d["point_label"],
            phase=d["phase"],
            progress=d["progress"],
            data=d.get("data"),
        )


class StatusReporter:
    """
    Helper class for workers to report status to the console UI.

    This provides a simple interface for workers to send status updates
    without needing to construct WorkerMessage objects directly.

    Usage in worker:
        reporter = StatusReporter(queue, worker_id=0)
        reporter.started('grid_P00')
        reporter.phase(WorkerPhase.OPTIMIZATION)
        reporter.progress(50)
        reporter.completed({'weighted_mso': 45.2, 'unweighted_mso': 47.1})

    Attributes:
        queue: Multiprocessing queue to send messages
        worker_id: This worker's ID (0-indexed)
    """

    def __init__(self, queue: Queue, worker_id: int):
        """
        Initialize status reporter.

        Args:
            queue: Multiprocessing queue for sending messages
            worker_id: This worker's ID (0-indexed)
        """
        self._queue = queue
        self._worker_id = worker_id
        self._current_point = ""
        self._current_phase = WorkerPhase.IDLE

    @property
    def worker_id(self) -> int:
        """Get this worker's ID."""
        return self._worker_id

    @property
    def current_point(self) -> str:
        """Get current grid point being processed."""
        return self._current_point

    @property
    def current_phase(self) -> WorkerPhase:
        """Get current processing phase."""
        return self._current_phase

    def started(self, point_label: str) -> None:
        """
        Report that worker started processing a grid point.

        Args:
            point_label: Grid point label (e.g., 'grid_P00')
        """
        self._current_point = point_label
        self._current_phase = WorkerPhase.STARTING
        self._send(
            MessageType.WORKER_STARTED,
            point_label=point_label,
            phase=WorkerPhase.STARTING.value,
        )

    def phase(self, phase: WorkerPhase, progress: int = 0) -> None:
        """
        Report entering a new processing phase.

        Args:
            phase: The new phase
            progress: Progress within phase (0-100), default 0
        """
        self._current_phase = phase
        self._send(
            MessageType.PHASE_CHANGED,
            phase=phase.value,
            progress=progress,
        )

    def progress(self, percent: int) -> None:
        """
        Report progress within current phase.

        Args:
            percent: Progress percentage (0-100)
        """
        self._send(
            MessageType.PROGRESS_UPDATE,
            progress=min(100, max(0, percent)),
        )

    def completed(self, result: Dict[str, Any]) -> None:
        """
        Report successful completion of grid point processing.

        Args:
            result: Result data dictionary containing at least:
                   - weighted_mso: float
                   - unweighted_mso: float
                   - success: bool (should be True)
        """
        self._current_phase = WorkerPhase.COMPLETE
        self._send(
            MessageType.WORKER_COMPLETED,
            phase=WorkerPhase.COMPLETE.value,
            progress=100,
            data=result,
        )

    def failed(self, error: str) -> None:
        """
        Report failure during processing.

        Args:
            error: Error message or description
        """
        self._current_phase = WorkerPhase.FAILED
        self._send(
            MessageType.WORKER_FAILED,
            phase=WorkerPhase.FAILED.value,
            data={"error": error},
        )

    def log(self, level: str, message: str) -> None:
        """
        Send a log message (captured for focus mode display).

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message text
        """
        self._send(
            MessageType.LOG_MESSAGE,
            data={"level": level, "message": message},
        )

    def heartbeat(self) -> None:
        """
        Send heartbeat for liveness detection.

        Call periodically during long operations to indicate
        the worker is still alive and working.
        """
        self._send(MessageType.HEARTBEAT)

    def reset(self) -> None:
        """
        Reset reporter state for processing a new grid point.

        Called automatically after completed() or failed(), but
        can be called manually if needed.
        """
        self._current_point = ""
        self._current_phase = WorkerPhase.IDLE

    def _send(self, msg_type: MessageType, **kwargs) -> None:
        """
        Send a message to the queue.

        Args:
            msg_type: Message type
            **kwargs: Additional message fields
        """
        msg = WorkerMessage(
            msg_type=msg_type,
            worker_id=self._worker_id,
            point_label=kwargs.get("point_label", self._current_point),
            phase=kwargs.get("phase", self._current_phase.value),
            progress=kwargs.get("progress", 0),
            data=kwargs.get("data"),
        )

        try:
            # Use put_nowait to avoid blocking if queue is full
            self._queue.put_nowait(msg.to_dict())
        except Exception:
            # Non-critical: don't crash worker if queue is full or broken
            pass


class NullStatusReporter(StatusReporter):
    """
    No-op status reporter for when console UI is disabled.

    All methods are no-ops, allowing the same worker code to run
    with or without the console UI.
    """

    def __init__(self):
        """Initialize null reporter (no queue needed)."""
        self._worker_id = 0
        self._current_point = ""
        self._current_phase = WorkerPhase.IDLE

    def _send(self, msg_type: MessageType, **kwargs) -> None:
        """No-op send."""
        pass


def create_status_reporter(queue: Optional[Queue], worker_id: int) -> StatusReporter:
    """
    Factory function to create appropriate status reporter.

    Args:
        queue: Status queue (or None for null reporter)
        worker_id: Worker ID

    Returns:
        StatusReporter or NullStatusReporter instance
    """
    if queue is None:
        return NullStatusReporter()
    return StatusReporter(queue, worker_id)
