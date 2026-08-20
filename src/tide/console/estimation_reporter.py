"""
Estimation Workflow Reporter
===========================

Provides status reporting for the parallel estimation workflow.
Wraps the pipeline task processing function to add status updates and logging.
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

# Import the base pipeline task types
from tide.workflows.estimation import PipelineResult, PipelineTask, _run_pipeline_task

from .ipc import WorkerPhase, create_status_reporter
from .worker_reporter import setup_worker_logging


def process_pipeline_task_with_reporting(
    task: PipelineTask,
    status_queue: Optional[mp.Queue],
    worker_id: int,
    log_dir: Optional[Path] = None,
) -> PipelineResult:
    """
    Process a pipeline task with status reporting to the console UI.

    Wraps _run_pipeline_task with:
    1. Status updates via queue
    2. Log redirection
    3. Phase tracking

    Args:
        task: PipelineTask (m1 or target)
        status_queue: Queue for sending status updates
        worker_id: Worker ID
        log_dir: Directory for worker logs

    Returns:
        PipelineResult
    """
    # Create status reporter
    reporter = create_status_reporter(status_queue, worker_id)
    reporter.started(task.label)

    # Redirect stdout/stderr to capture underlying tool output
    saved_stdout_fd = None
    saved_stderr_fd = None

    try:
        # 1. Setup IO Redirection
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"worker_{worker_id}_{task.label}_stdout.log"
            stderr_path = log_dir / f"worker_{worker_id}_{task.label}_stderr.log"
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

        # 2. Setup Logging Handler
        log_handler = None
        if log_dir:
            log_handler = setup_worker_logging(reporter, log_dir, task.label, worker_id)

        # 3. Define phase callback to update reporter from log interception
        # Note: Since _run_pipeline_task is monolithic, we'll infer phase from logs
        # or just update generic progress.
        # Ideally, we would inject a reporter into _run_pipeline_task, but to minimize
        # changes to the core logic, we'll rely on the fact that _run_pipeline_task
        # logs informative messages.

        # We can also manually update phases before/after major blocks if we
        # were to decompose _run_pipeline_task, but here we wrap the whole thing.
        # To make it livelier, we'll set initial phase.

        if task.needs_optimization:
            reporter.phase(WorkerPhase.OPTIMIZATION, 0)
        else:
            reporter.phase(WorkerPhase.FEM_SIMULATION, 0)

        # 4. Run the Task
        result = _run_pipeline_task(task)

        # 5. Report completion
        if result.success:
            # Calculate what we can for the UI summary
            # For intermediate tasks, we might not have final intensity yet
            # But we can report success
            reporter.completed({})
        else:
            reporter.failed(result.error_message or "Unknown error")

        return result

    except Exception as e:
        reporter.failed(str(e))
        # Return generic failure result matching the expected type
        return PipelineResult(
            task_type=task.task_type,
            label=task.label,
            success=False,
            opt_matrix=None,
            opt_scalp_coords=None,
            mesh_path=None,
            trk_path=None,
            streamlines=None,
            af_values=None,
            len_values=None,
            e_vecs_list=None,
            roi_masks=None,
            roi_segments=None,
            error_message=str(e),
        )

    finally:
        # Cleanup logging
        if log_handler:
            root = logging.getLogger()
            root.removeHandler(log_handler)
            log_handler.close()

        # Restore IO
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
            except Exception:
                pass

        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
            except Exception:
                pass
