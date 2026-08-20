"""
Optimized Grid Search Workflow Module
=====================================

This module implements a parallelized version of the TIDE grid search workflow
using Python's multiprocessing to process multiple grid points concurrently.

Performance Characteristics
---------------------------
- **Speedup formula**: T_parallel ≈ T_sequential / min(N_workers, N_grid_points)
- **Memory usage**: ~12GB × N_workers + 4GB parent/OS reserve
- **Default workers**: min(cpu_count - 1, 2, memory-safe worker limit)

OpenMP Thread Control (CRITICAL)
--------------------------------
SimNIBS uses the PARDISO solver with OpenMP parallelization internally.
Running multiple SimNIBS instances without controlling thread count causes
CPU oversubscription (N_workers × M_threads >> CPU_cores), leading to severe
cache thrashing and performance degradation. Each worker is configured to use
single-threaded mode via environment variables set BEFORE any SimNIBS imports.

Mesh Caching Investigation (Proposal F Findings)
------------------------------------------------
SimNIBS mesh objects (`sesame.FEM`) contain internal solver state and are not
picklable, preventing direct sharing between processes. Alternative approaches:
- Memory-mapped files: Not applicable to SimNIBS mesh format
- Shared memory: Would require deep modifications to SimNIBS internals

Current approach: Rely on OS-level disk cache. After the first simulation loads
the ~500MB mesh file, subsequent workers benefit from the file being cached in
RAM by the OS. This provides near-instantaneous I/O for parallel workers without
requiring code changes to SimNIBS.

Verification of Parallelization
-------------------------------
To verify parallelization is working:
1. Run with grid search workflow and check system monitor for multiple Python
   processes (one per worker)
2. Compare execution time with `no_parallel=True` in config
3. Check log output for "Processing X grid points with Y workers"

Usage
-----
    from tide.workflows.grid_search_optimized import run_grid_search_workflow
    run_grid_search_workflow(config)  # Uses config.options.max_workers

    # Or override parallelization settings:
    run_grid_search_workflow(config, max_workers=4)  # Force 4 workers
    run_grid_search_workflow(config, no_parallel=True)  # Force sequential
"""

import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Dict, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

import numpy as np

from tide.core import io, physics, tractography
from tide.core.geometry import (
    calculate_alignment_and_depth,
    calculate_alignment_corrected,
    compute_default_coil_orientation,
    evaluate_coil_pose_qc,
    project_target_to_scalp,
    validate_coil_pose_for_dose,
)
from tide.interfaces.sampling import sample_field_at_coordinates
from tide.interfaces.simnibs_interface import SimNIBSInterface
from tide.interfaces.unified_estimation import (
    AnalysisConfig,
    analyze_bundle,
    load_surface_tree,
    validate_calibration_metrics,
)
from tide.interfaces.visualization import save_af_visualization
from tide.utils.config import (
    SimNIBSConfig,
    orientation_is_matrix,
    save_config_to_output,
    validate_workflow_config,
)
from tide.workflows._grid_reporting import (
    GridReportingContext,
    initialize_grid_results_csv,
    write_grid_results,
    write_grid_summary,
)
from tide.workflows._shared import WorkflowError
from tide.workflows._shared import configure_worker_environment as _configure_worker_environment
from tide.workflows._shared import single_thread_child_environment as _single_thread_child_env
from tide.workflows._shared import split_vectors_by_streamline

log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_MAX_WORKERS = 2
PARDISO_MEMORY_PER_WORKER_GB = 12.0
GRID_MEMORY_RESERVE_GB = 4.0
GRID_FORCE_WORKERS_ENV = "TIDE_GRID_FORCE_WORKERS"
MIN_WORKERS = 1

# Global variable to store assigned worker ID within each process
_process_worker_id = None
_worker_lock_file = None  # Keep reference to prevent gc/closing


def _lock_worker_file(lock_file: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if msvcrt is None:
        raise RuntimeError("No supported file-locking implementation is available.")

    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write("\0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        raise BlockingIOError(str(exc)) from exc


def _init_worker(num_workers: int, lock_dir: str) -> None:
    """
    Initialize worker process by attempting to acquire a persistent ID
    via file locking. This is robust to process restarts (crashes).
    """
    global _process_worker_id, _worker_lock_file

    _process_worker_id = 0
    try:
        # Try to acquire a lock for each worker ID
        for i in range(num_workers):
            lock_path = os.path.join(lock_dir, f"worker_{i}.lock")
            # Open file in append mode (creates if not exists)
            f = open(lock_path, "a")  # ignore: consider-using-with
            try:
                # Try non-blocking lock
                _lock_worker_file(f)

                # If we got here, we acquired the lock
                _process_worker_id = i
                _worker_lock_file = f  # Keep it open
                return
            except BlockingIOError:
                # Lock held by another process, try next
                f.close()
                continue

    except Exception as e:
        # Fallback (should normally not happen unless FS issues)
        print(f"Worker init failed: {e}", file=sys.stderr)
        _process_worker_id = 0


# =============================================================================
# DATACLASSES FOR SERIALIZATION
# =============================================================================


@dataclass
class GridPointTask:
    """
    Encapsulates all serializable data needed to process one grid point.

    This dataclass is designed to be picklable for inter-process communication
    with ProcessPoolExecutor. All Path objects are converted to strings.

    Attributes:
        index: Grid point index for result ordering.
        cortex_coord: Target cortex coordinates [x, y, z].
        point_label: Human-readable label (e.g., "grid_P00").
        point_dir: Output directory path for this grid point.
        msh_file: Path to head mesh file.
        m2m_path: Path to m2m directory.
        coil_path: Path to coil model file.
        fixed_scalp_coords: Fixed scalp center coordinates.
        grid_orientation_ref: Orientation reference (pos_ydir).
        target_bundle_path: Path to target tractogram.
        t1w_path: Path to T1w image.
        surface_path: Optional path to cortical surface.
        weights_target_path: Optional path to target weights.
        adm_optimization: Whether to use ADM optimization.
        opt_spatial_resolution: Spatial resolution for optimization.
        opt_angle_resolution: Angle resolution for optimization.
        opt_search_angle: Search angle for optimization.
        opt_search_radius: Search radius for optimization.
        field_mode: Field mode for physics calculation.
        roi_size_mm: ROI radius in mm.
        activation_length_mm: Activation length in mm.
        gwi_threshold_mm: Max distance from the GWI surface in mm.
        measured_rmt_mso: Measured RMT in %MSO.
        af_cst_calibration: CST calibration efficiency value.
        cst_metric_unweighted: Unweighted CST metric.
    """

    index: int
    cortex_coord: List[float]
    point_label: str
    point_dir: str
    msh_file: str
    m2m_path: str
    coil_path: str
    fixed_scalp_coords: List[float]
    grid_orientation_ref: List[float]
    target_bundle_path: str
    t1w_path: str
    surface_path: Optional[str]
    weights_target_path: Optional[str]
    adm_optimization: bool
    opt_spatial_resolution: float
    opt_angle_resolution: float
    opt_search_angle: float
    opt_search_radius: float
    field_mode: str
    roi_size_mm: float
    activation_length_mm: float
    gwi_threshold_mm: float
    max_angular_deviation_deg: float
    measured_rmt_mso: float
    af_cst_calibration: float
    cst_metric_unweighted: float
    mso_floor_ratio: float
    mso_ceiling_ratio: float
    generate_visualizations: bool


@dataclass
class GridPointResult:
    """
    Results from processing one grid point.

    Attributes:
        index: Grid point index for result ordering.
        point_label: Human-readable label.
        success: Whether processing completed successfully.
        weighted_mso: Estimated MSO (weighted), or 999.9 on failure.
        unweighted_mso: Estimated MSO (unweighted), or 999.9 on failure.
        cortex_coord: Target cortex coordinates.
        opt_scalp_coords: Optimized scalp coordinates.
        opt_matrix: Optimized 4x4 coil matrix.
        error_message: Error message if success is False.
    """

    index: int
    point_label: str
    success: bool
    weighted_mso: float
    unweighted_mso: float
    cortex_coord: List[float]
    opt_scalp_coords: Optional[List[float]]
    opt_matrix: Optional[List[List[float]]]
    weighted_mso_raw: float = 0.0
    unweighted_mso_raw: float = 0.0
    weighted_mso_flag: str = "WITHIN_RANGE"
    unweighted_mso_flag: str = "WITHIN_RANGE"
    sei_weighted: float = 0.0
    sei_unweighted: float = 0.0
    multiplier_weighted: float = 0.0
    multiplier_unweighted: float = 0.0
    target_metric_weighted: float = 0.0
    target_metric_unweighted: float = 0.0
    target_aggregates_weighted: Dict[str, float] = field(default_factory=dict)
    target_aggregates_unweighted: Dict[str, float] = field(default_factory=dict)
    target_weight_source: Optional[str] = None
    tgt_align: float = 0.0
    tgt_align_corrected: float = 0.0
    tgt_depth: float = 0.0
    pose_qc: Optional[Dict[str, object]] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class GridWorkerPlan:
    workers: int
    requested_workers: Optional[int]
    num_grid_points: int
    cpu_count: int
    available_memory_gb: Optional[float]
    memory_worker_limit: int
    memory_per_worker_gb: float
    memory_reserve_gb: float
    forced: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "workers": self.workers,
            "requested_workers": self.requested_workers,
            "num_grid_points": self.num_grid_points,
            "cpu_count": self.cpu_count,
            "available_memory_gb": self.available_memory_gb,
            "memory_worker_limit": self.memory_worker_limit,
            "memory_per_worker_gb": self.memory_per_worker_gb,
            "memory_reserve_gb": self.memory_reserve_gb,
            "forced": self.forced,
            "solver": "PARDISO",
        }


def _collect_parallel_grid_results(
    future_to_task: Dict[Future, GridPointTask],
    num_points: int,
    use_parallel_ui: bool,
    progress_callback: Optional[Callable[[int, int, str], None]],
) -> tuple[List[GridPointResult], int]:
    results: List[GridPointResult] = []
    failed_count = 0

    for completed, future in enumerate(as_completed(future_to_task), start=1):
        task = future_to_task[future]

        try:
            result = future.result()
            results.append(result)

            if result.success:
                if not use_parallel_ui:
                    log.info(
                        f"[{completed}/{num_points}] {result.point_label}: "
                        f"Weighted MSO={result.weighted_mso:.2f}%"
                    )
            else:
                failed_count += 1
                if not use_parallel_ui:
                    log.error(
                        f"[{completed}/{num_points}] {result.point_label}: "
                        f"FAILED - {result.error_message}"
                    )
        except Exception as e:
            failed_count += 1
            if not use_parallel_ui:
                log.error(f"[{completed}/{num_points}] {task.point_label}: Worker exception - {e}")
            results.append(
                GridPointResult(
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
            )

        if progress_callback:
            progress_callback(completed, num_points, task.point_label)

    return results, failed_count


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================


class _GridPointReporter:
    """
    No-op progress sink for :func:`process_grid_point`.

    The console worker passes an adapter whose hooks forward to the live UI
    (audit C-003); the default instance makes every hook a no-op so the
    non-console path is unchanged. ``process_grid_point`` owns the grid-point
    physics; reporters only observe stage transitions.
    """

    def optimization(self) -> None: ...

    def simulation(self) -> None: ...

    def sampling(self) -> None: ...

    def activating_function(self) -> None: ...

    def bundle_analysis(self) -> None: ...

    def saving_results(self) -> None: ...

    def progress(self, pct: int) -> None: ...


_NULL_REPORTER = _GridPointReporter()


def _save_grid_point_visualizations(
    task: GridPointTask,
    streamlines: List[np.ndarray],
    af_values: List[np.ndarray],
) -> None:
    if not task.generate_visualizations:
        return

    point_dir = Path(task.point_dir)
    io.save_points_as_nifti(
        np.concatenate(streamlines),
        Path(task.t1w_path),
        point_dir / f"{task.point_label}_af.nii.gz",
        values=np.abs(np.concatenate(af_values)),
    )
    save_af_visualization(
        streamlines=streamlines,
        values=af_values,
        roi_center=task.cortex_coord,
        roi_radius=task.roi_size_mm,
        output_dir=point_dir,
        prefix=task.point_label,
    )


def process_grid_point(
    task: GridPointTask, reporter: Optional[_GridPointReporter] = None
) -> GridPointResult:
    """
    Process a single grid point: optimization, simulation, and analysis.

    This is the main worker function executed in parallel processes.
    It performs the complete pipeline for one grid point:
    1. Coil position optimization
    2. FEM E-field simulation
    3. E-field sampling on tractogram
    4. Activating function calculation
    5. Bundle analysis and intensity estimation

    Args:
        task: GridPointTask containing all necessary parameters.

    Returns:
        GridPointResult with success status and computed metrics.

    Note:
        This function configures single-threaded mode at startup to prevent
        OpenMP thread oversubscription across parallel workers.
    """
    # CRITICAL: Configure environment before any heavy imports
    _configure_worker_environment()
    if reporter is None:
        reporter = _NULL_REPORTER

    # Import heavy modules inside worker to ensure environment is set first
    from pathlib import Path

    import numpy as np

    from tide.core import io, physics, tractography
    from tide.core.geometry import (
        calculate_alignment_and_depth,
        calculate_alignment_corrected,
        evaluate_coil_pose_qc,
        validate_coil_pose_for_dose,
    )
    from tide.interfaces.sampling import sample_field_at_coordinates
    from tide.interfaces.simnibs_interface import SimNIBSInterface
    from tide.interfaces.unified_estimation import (
        AnalysisConfig,
        analyze_bundle,
        apply_intensity_bounds,
        load_surface_tree,
        validate_calibration_metrics,
    )

    # Import logging utilities to register the highlight method in spawned process
    # (spawn creates fresh interpreter without parent's monkey-patched Logger class)
    from tide.utils.logging import highlight

    logging.Logger.highlight = highlight

    log = logging.getLogger(__name__)

    try:
        validate_calibration_metrics(task.af_cst_calibration, task.cst_metric_unweighted)

        point_dir = Path(task.point_dir)
        point_dir.mkdir(exist_ok=True, parents=True)

        log.highlight(f"=== Processing {task.point_label} {task.cortex_coord} ===")

        # =====================================================================
        # 1. Optimization
        # =====================================================================
        reporter.optimization()
        log.info(f"Running standard optimization for {task.point_label}...")

        opt_matrix, opt_scalp_coords = SimNIBSInterface.run_optimization(
            mesh_path=Path(task.msh_file),
            output_dir=point_dir,
            coil_path=Path(task.coil_path),
            target_coords=task.cortex_coord,
            scalp_centre=task.fixed_scalp_coords,
            orientation_ref=task.grid_orientation_ref,
            didt=1e6,
            use_adm=task.adm_optimization,
            spatial_resolution=task.opt_spatial_resolution,
            angle_resolution=task.opt_angle_resolution,
            search_angle=task.opt_search_angle,
            search_radius_mm=task.opt_search_radius,
        )
        pose_qc = evaluate_coil_pose_qc(Path(task.msh_file), opt_matrix, opt_scalp_coords)
        if pose_qc.status == "WARN":
            log.warning(f"{task.point_label}: Coil pose QC warning: {pose_qc.reasons}")

        io.save_optimization_result_txt(
            point_dir / f"{task.point_label}_opt_result.txt",
            opt_matrix,
            opt_scalp_coords,
            pose_qc=pose_qc.as_dict(),
        )
        validate_coil_pose_for_dose(pose_qc, explicit_matrix=False)
        reporter.progress(100)

        # =====================================================================
        # 2. Simulation
        # =====================================================================
        reporter.simulation()
        log.info(f"Running FEM simulation for {task.point_label}...")

        mesh_pt = SimNIBSInterface.run_simulation(
            mesh_path=Path(task.m2m_path),
            output_dir=point_dir,
            coil_path=Path(task.coil_path),
            didt=1e6,
            orientation=opt_matrix.tolist(),
            coords=None,
        )
        reporter.progress(100)

        # =====================================================================
        # 3. Sampling E-field on tractogram
        # =====================================================================
        reporter.sampling()
        log.info(f"Sampling E-field on tractogram for {task.point_label}...")

        sft_tgt = tractography.load_tract(Path(task.target_bundle_path), Path(task.t1w_path))
        points_tgt = np.concatenate(sft_tgt.streamlines)
        e_vectors_tgt = sample_field_at_coordinates(
            mesh_pt,
            points_tgt,
            "E",
            output_dir=point_dir,
            file_prefix=task.point_label,
        )

        e_vecs_list = split_vectors_by_streamline(e_vectors_tgt, sft_tgt.streamlines)
        reporter.progress(100)

        # =====================================================================
        # 3b. Filter streamlines by angular deviation
        # =====================================================================
        # Track original streamline ids through the drop chain so SIFT2 weights
        # stay attached to their streamlines in analyze_bundle (audit C-002).
        orig_idx = np.arange(len(sft_tgt.streamlines))
        if task.max_angular_deviation_deg > 0:
            (
                filtered_sl,
                e_vecs_list,
                n_removed,
                orig_idx,
            ) = tractography.filter_by_angular_deviation(
                list(sft_tgt.streamlines),
                e_field_vectors=e_vecs_list,
                max_angle_deg=task.max_angular_deviation_deg,
                roi_center=task.cortex_coord,
                roi_radius=task.roi_size_mm,
                indices=orig_idx,
            )
        else:
            filtered_sl = list(sft_tgt.streamlines)

        # =====================================================================
        # 4. Physics calculation and TRK saving
        # =====================================================================
        reporter.activating_function()
        log.info(f"Calculating activating function for {task.point_label}...")

        new_sl_tgt, af_tgt, len_tgt, orig_idx = physics.calculate_scalar_map(
            filtered_sl,
            e_vecs_list,
            mode=task.field_mode,
            indices=orig_idx,
        )
        reporter.progress(50)

        tgt_roi_masks_post, _ = tractography.get_roi_masks(
            new_sl_tgt, task.roi_size_mm, task.cortex_coord
        )
        tgt_align, tgt_depth = calculate_alignment_and_depth(
            new_sl_tgt,
            e_vecs_list,
            tgt_roi_masks_post,
            Path(task.msh_file),
            task.cortex_coord,
        )
        tgt_align_corrected = calculate_alignment_corrected(
            new_sl_tgt,
            e_vecs_list,
            tgt_roi_masks_post,
        )

        tgt_trk_path = point_dir / f"{task.point_label}_af.trk"
        io.save_tract_with_data(sft_tgt, new_sl_tgt, tgt_trk_path, "AF", af_tgt, len_tgt)

        _save_grid_point_visualizations(task, new_sl_tgt, af_tgt)
        reporter.progress(100)

        # =====================================================================
        # 5. Unified Analysis
        # =====================================================================
        reporter.bundle_analysis()
        log.info(f"Running unified bundle analysis for {task.point_label}...")

        # Load surface if available
        surf_tree = None
        if task.surface_path:
            surf_tree = load_surface_tree(task.surface_path)

        point_config = AnalysisConfig(
            cst_trk="",
            target_trk=str(tgt_trk_path),
            rmt=task.measured_rmt_mso,
            cst_coords=np.zeros(3),
            target_coords=np.array(task.cortex_coord),
            surf_path=task.surface_path,
            gwi_threshold=task.gwi_threshold_mm,
            roi_radius=task.roi_size_mm,
            activation_len=task.activation_length_mm,
            target_weights=task.weights_target_path,
        )

        tgt_res = analyze_bundle(
            name=task.point_label,
            trk_path=str(tgt_trk_path),
            roi_center=np.array(task.cortex_coord),
            config=point_config,
            surface_tree=surf_tree,
            weight_path=point_config.target_weights,
            orig_indices=orig_idx,
        )
        reporter.progress(100)

        # =====================================================================
        # 6. Metrics and Estimation
        # =====================================================================
        reporter.saving_results()
        af_target_w = tgt_res.metric_weighted
        af_target_u = tgt_res.metric_unweighted

        # SEI: Stimulation Efficiency Index = AF_target / AF_CST
        # SEI > 1: target more efficient than CST (less stimulation needed)
        # SEI = 1: same efficiency as CST; SEI < 1: more stimulation needed
        sei_w = af_target_w / task.af_cst_calibration if task.af_cst_calibration > 0 else 0.0
        sei_u = af_target_u / task.cst_metric_unweighted if task.cst_metric_unweighted > 0 else 0.0

        # Multiplier k = M_CST / M_target (intensity-invariant; I_raw = RMT * k)
        multiplier_w = task.af_cst_calibration / af_target_w if af_target_w > 0 else 0.0
        multiplier_u = task.cst_metric_unweighted / af_target_u if af_target_u > 0 else 0.0

        est_mso = (
            task.measured_rmt_mso * (task.af_cst_calibration / af_target_w)
            if af_target_w > 0
            else float("nan")
        )
        est_mso_u = (
            task.measured_rmt_mso * (task.cst_metric_unweighted / af_target_u)
            if af_target_u > 0
            else float("nan")
        )

        # Apply physiological intensity bounds (non-finite -> ESTIMATION_FAILED).
        bounded_w = apply_intensity_bounds(
            est_mso,
            task.measured_rmt_mso,
            floor_ratio=task.mso_floor_ratio,
            ceiling_ratio=task.mso_ceiling_ratio,
        )
        raw_mso_w = bounded_w["model_raw"]
        est_mso = bounded_w["best_estimate"]
        flag_w = bounded_w["flag"]
        if flag_w != "WITHIN_RANGE":
            log.info(
                f"{task.point_label}: Weighted I {flag_w} (raw={raw_mso_w:.1f}%, clamped={est_mso:.1f}%)"
            )

        bounded_u = apply_intensity_bounds(
            est_mso_u,
            task.measured_rmt_mso,
            floor_ratio=task.mso_floor_ratio,
            ceiling_ratio=task.mso_ceiling_ratio,
        )
        raw_mso_u = bounded_u["model_raw"]
        est_mso_u = bounded_u["best_estimate"]
        flag_u = bounded_u["flag"]
        if flag_u != "WITHIN_RANGE":
            log.info(
                f"{task.point_label}: Unweighted I {flag_u} (raw={raw_mso_u:.1f}%, clamped={est_mso_u:.1f}%)"
            )

        log.highlight(
            f"==> ESTIMATED TARGET {task.point_label} I: {est_mso:.2f}% (Weighted) | {est_mso_u:.2f}% (Unweighted)"
        )
        reporter.progress(100)

        return GridPointResult(
            index=task.index,
            point_label=task.point_label,
            success=True,
            weighted_mso=est_mso,
            unweighted_mso=est_mso_u,
            weighted_mso_raw=raw_mso_w,
            unweighted_mso_raw=raw_mso_u,
            weighted_mso_flag=flag_w,
            unweighted_mso_flag=flag_u,
            sei_weighted=sei_w,
            sei_unweighted=sei_u,
            multiplier_weighted=multiplier_w,
            multiplier_unweighted=multiplier_u,
            target_metric_weighted=af_target_w,
            target_metric_unweighted=af_target_u,
            target_aggregates_weighted=tgt_res.aggregates_weighted,
            target_aggregates_unweighted=tgt_res.aggregates_unweighted,
            target_weight_source=tgt_res.weight_source,
            tgt_align=tgt_align,
            tgt_align_corrected=tgt_align_corrected,
            tgt_depth=tgt_depth,
            pose_qc=pose_qc.as_dict(),
            cortex_coord=task.cortex_coord,
            opt_scalp_coords=opt_scalp_coords.tolist(),
            opt_matrix=opt_matrix.tolist(),
        )

    except Exception as e:
        log.error(f"Processing failed for {task.point_label}: {e}")
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _get_available_memory_gb() -> Optional[float]:
    try:
        import psutil

        return float(psutil.virtual_memory().available / (1024**3))
    except ImportError:
        pass

    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return float(available_pages * page_size / (1024**3))


def _resolve_grid_worker_plan(
    requested_workers: Optional[int] = None,
    num_grid_points: int = 1,
) -> GridWorkerPlan:
    cpu_count = os.cpu_count() or 4
    available_memory_gb = _get_available_memory_gb()
    if available_memory_gb is None:
        memory_worker_limit = 1
    else:
        usable_memory_gb = max(0.0, available_memory_gb - GRID_MEMORY_RESERVE_GB)
        memory_worker_limit = max(
            MIN_WORKERS,
            int(usable_memory_gb / PARDISO_MEMORY_PER_WORKER_GB),
        )

    forced = os.environ.get(GRID_FORCE_WORKERS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if requested_workers is None:
        cpu_based = max(1, cpu_count - 1)
        desired_workers = min(cpu_based, DEFAULT_MAX_WORKERS)
    else:
        desired_workers = max(MIN_WORKERS, requested_workers)

    desired_workers = min(desired_workers, num_grid_points)
    workers = desired_workers if forced else min(desired_workers, memory_worker_limit)

    return GridWorkerPlan(
        workers=max(MIN_WORKERS, workers),
        requested_workers=requested_workers,
        num_grid_points=num_grid_points,
        cpu_count=cpu_count,
        available_memory_gb=available_memory_gb,
        memory_worker_limit=memory_worker_limit,
        memory_per_worker_gb=PARDISO_MEMORY_PER_WORKER_GB,
        memory_reserve_gb=GRID_MEMORY_RESERVE_GB,
        forced=forced,
    )


def _calculate_max_workers(
    requested_workers: Optional[int] = None,
    num_grid_points: int = 1,
) -> int:
    return _resolve_grid_worker_plan(requested_workers, num_grid_points).workers


# =============================================================================
# MAIN WORKFLOW FUNCTION
# =============================================================================


def run_grid_search_workflow(
    config: SimNIBSConfig,
    max_workers: Optional[int] = None,
    no_parallel: Optional[bool] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    console_ui: bool = True,
) -> None:
    """
    Execute the Conformal TIDE Grid Search Workflow with optional parallelization.

    This function is a drop-in replacement for the original `grid_search.py`
    implementation. It maintains identical output format (CSV, TXT summary,
    per-point configs) while adding multi-process parallelization.

    The workflow:
    1. M1 Calibration (sequential) - establishes baseline
    2. CST Analysis (sequential) - computes calibration efficiency
    3. Grid Generation (sequential) - creates target points
    4. Grid Point Processing (PARALLEL) - optimization + simulation per point
    5. Summary Generation (sequential) - aggregates parallel results

    Args:
        config: SimNIBSConfig object with all pipeline parameters.
        max_workers: Override for number of parallel workers.
            - None: Use config.options.max_workers or auto-detect
            - int: Force specific number of workers
        no_parallel: Override for sequential processing.
            - None: Use config.options.no_parallel
            - True: Force sequential processing
            - False: Force parallel processing (unless max_workers=1)
        progress_callback: Optional callback for progress updates.
            - Called as progress_callback(completed, total, point_label)
            - completed: Number of grid points processed so far
            - total: Total number of grid points
            - point_label: Label of the just-completed grid point
        console_ui: Enable rich console UI for parallel processing.
            - True: Use console UI if running in interactive terminal
            - False: Use simple text logging

    Note:
        Parallelization settings priority (highest to lowest):
        1. Function arguments (max_workers, no_parallel)
        2. Config file (config.options.max_workers, config.options.no_parallel)
        3. Auto-detection based on system resources
    """
    validate_workflow_config(config, "grid")

    log.info("=" * 60)
    log.info("=== Starting TIDE Grid Search Workflow (OPTIMIZED) ===")
    log.info("=" * 60)

    start_time = time.time()

    # Create multiprocessing context BEFORE UI creation
    # CRITICAL: The UI's status_queue must use the same context as workers
    # to avoid message loss between processes with different contexts
    ctx = mp.get_context("spawn")

    # Create console UI early (in sequential mode) for Steps 1-5
    ui = None
    if console_ui and sys.stdout.isatty():
        try:
            from tide.console import create_console_ui

            ui = create_console_ui(
                subject_id=config.subject.id,
                num_workers=0,
                total_points=0,
                current_step=1,
                total_steps=8,
                workflow_name="Grid Search",
                enabled=True,
                mode="sequential",
                mp_context=ctx,
            )
            ui.start()
        except ImportError:
            log.warning("Console UI not available, falling back to text logging")
            ui = None

    # Resolve parallelization settings
    use_no_parallel = no_parallel if no_parallel is not None else config.options.no_parallel
    use_max_workers = max_workers if max_workers is not None else config.options.max_workers

    # Use mesh path from config (m2m folder input)
    msh_file = config.subject.mesh_path

    # ==========================================================================
    # MEDOID LOGIC - Calculate cortex coords if requested
    # ==========================================================================
    if config.target.medoid_endpoint:
        if not config.target.bundle_path or not config.target.bundle_path.exists():
            if ui:
                ui.stop()
            raise WorkflowError("Medoid endpoint requested but bundle path is missing or invalid.")

        log.warning(f"Medoid Endpoint Calculation ENABLED for Target: {config.target.label}")
        try:
            new_coords = tractography.get_bundle_cortical_medoid(
                config.target.bundle_path,
                config.subject.t1w_path,
                reference_coord=config.target.coords,
            )
            log.info(f"--> Replacing input coords with calculated Medoid: {new_coords}")
            config.target.coords = new_coords.tolist()
            config.grid.coords = new_coords.tolist()
            log.info("Updated Grid Center to new Medoid coordinates.")
        except Exception as e:
            if ui:
                ui.stop()
            raise WorkflowError(f"Failed to calculate medoid endpoint: {e}") from e

    # ==========================================================================
    # OUTPUT DIRECTORY SETUP
    # ==========================================================================
    grid_folder_name = "TIDE_grid_search"
    if config.target.label:
        grid_folder_name += f"_{config.target.label}"
    out_dir = config.subject.derivatives_path / grid_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_calibration_matrix = None
    calibration_pose_qc = None

    sims_dir = out_dir / "simulations"
    sims_dir.mkdir(exist_ok=True)
    qc_dir = out_dir / "QC"
    qc_dir.mkdir(exist_ok=True)

    # ==========================================================================
    # PRE-LOAD SURFACE TREE (for GWI filtering)
    # ==========================================================================
    surf_tree = None
    if config.subject.surface_path:
        surf_tree = load_surface_tree(str(config.subject.surface_path))
        log.info("Surface loaded for GWI analysis in Grid Search.")

    # Determine Spatial Mode string for report
    spatial_mode = "Baseline (Sphere Only)"
    if surf_tree is not None:
        spatial_mode = "Surface-Constrained (GWI)"
    if config.subject.weights_cst_path or config.subject.weights_target_path:
        spatial_mode += " + Weighted"

    # Log optimization parameters
    log.info("=== Optimization Parameters from Config ===")
    log.info(f"  opt_search_radius: {config.options.opt_search_radius} mm")
    log.info(f"  opt_spatial_resolution: {config.options.opt_spatial_resolution} mm")
    log.info(f"  opt_angle_resolution: {config.options.opt_angle_resolution} deg")
    log.info(f"  opt_search_angle: {config.options.opt_search_angle} deg")
    log.info(f"  adm_optimization: {config.options.adm_optimization}")

    # ==========================================================================
    # STEP 1: M1 CALIBRATION (Sequential)
    # ==========================================================================
    if ui:
        ui.update_step(1, "running")
    log.info("--- Step 1: M1 Calibration ---")

    m1_out = out_dir / "calibration_m1"
    m1_out.mkdir(exist_ok=True, parents=True)

    cal_coords = config.calibration.coords
    cal_orientation = config.calibration.orientation
    generated_calibration_scalp_coords: Optional[List[float]] = None
    is_matrix = orientation_is_matrix(cal_orientation)

    # Optimize the M1 coil pose unless a full 4x4 matrix was supplied.
    # run_optimization auto-projects the scalp position and auto-orients the
    # handle when scalp_coords / orientation are absent.
    if not is_matrix:
        if ui:
            ui.update_step_detail("Running coil position optimization...")
        log.info("M1 Calibration: Running coil position optimization...")
        log.info(f"[M1_OPT] orientation_ref (pos_ydir): {config.calibration.orientation}")
        opt_matrix, opt_scalp = SimNIBSInterface.run_optimization(
            mesh_path=msh_file,
            output_dir=m1_out,
            coil_path=config.coil.coil_path,
            target_coords=config.calibration.coords,
            scalp_centre=config.calibration.scalp_coords,
            orientation_ref=config.calibration.orientation,
            didt=1e6,
            use_adm=config.options.adm_optimization,
            spatial_resolution=config.options.opt_spatial_resolution,
            angle_resolution=config.options.opt_angle_resolution,
            search_angle=config.options.opt_search_angle,
            search_radius_mm=config.options.opt_search_radius,
        )
        qc = evaluate_coil_pose_qc(msh_file, opt_matrix, opt_scalp)
        calibration_pose_qc = qc.as_dict()
        if qc.status == "WARN":
            log.warning(f"M1 Calibration: Coil pose QC warning: {qc.reasons}")
        validate_coil_pose_for_dose(qc, explicit_matrix=False)
        cal_orientation = opt_matrix.tolist()
        generated_calibration_matrix = opt_matrix.tolist()
        generated_calibration_scalp_coords = opt_scalp.tolist()
        cal_coords = None
    elif is_matrix:
        qc = evaluate_coil_pose_qc(msh_file, np.asarray(cal_orientation))
        calibration_pose_qc = qc.as_dict()
        if qc.status == "WARN":
            log.warning(f"M1 Calibration: Coil pose QC warning: {qc.reasons}")
        validate_coil_pose_for_dose(qc, explicit_matrix=True)

    # M1 Simulation
    if ui:
        ui.update_step_detail("Running FEM E-field simulation...")
    mesh_m1 = SimNIBSInterface.run_simulation(
        mesh_path=config.subject.m2m_path,
        output_dir=m1_out,
        coil_path=config.coil.coil_path,
        didt=1e6,
        coords=cal_coords,
        orientation=cal_orientation,
        distance_mm=config.coil.coil_distance_mm,
    )

    # Update config with actual parallelization settings used
    # This ensures the output YAML reflects the runtime parameters (whether from CLI args or original config)
    config.options.max_workers = use_max_workers
    config.options.no_parallel = use_no_parallel

    # Save configuration (after M1 optimization). Includes generated
    # calibration matrix and resolved scalp coords so the top-level grid
    # config is re-runnable without re-triggering M1 optimization or medoid
    # computation. Per-grid-point configs are written separately by
    # ``save_grid_point_config`` with the point-specific target settings.
    save_config_to_output(
        config,
        out_dir,
        "grid_search",
        generated_calibration_matrix=generated_calibration_matrix,
        generated_calibration_scalp_coords=generated_calibration_scalp_coords,
        medoid_resolved=bool(config.target.medoid_endpoint),
    )

    if ui:
        ui.update_step(1, "complete")

    # ==========================================================================
    # STEP 2: CST ANALYSIS (Sequential)
    # ==========================================================================
    if ui:
        ui.update_step(2, "running")
    log.info("--- Step 2: CST Analysis ---")

    if ui:
        ui.update_step_detail("Loading CST tractogram...")
    sft_cst = tractography.load_tract(config.calibration.bundle_path, config.subject.t1w_path)
    points_cst = np.concatenate(sft_cst.streamlines)
    if ui:
        ui.update_step_detail("Sampling E-field on CST streamlines...")
    e_vectors_cst = sample_field_at_coordinates(
        mesh_m1, points_cst, "E", output_dir=m1_out, file_prefix="M1_CST"
    )

    e_vecs_list_cst = split_vectors_by_streamline(e_vectors_cst, sft_cst.streamlines)

    # Filter CST streamlines by angular deviation. Track original ids through
    # the drop chain so SIFT2 weights stay aligned in analyze_bundle (C-002).
    cst_streamlines = list(sft_cst.streamlines)
    cst_orig_idx = np.arange(len(sft_cst.streamlines))
    if config.options.max_angular_deviation_deg > 0:
        (
            cst_streamlines,
            e_vecs_list_cst,
            _,
            cst_orig_idx,
        ) = tractography.filter_by_angular_deviation(
            cst_streamlines,
            e_field_vectors=e_vecs_list_cst,
            max_angle_deg=config.options.max_angular_deviation_deg,
            roi_center=config.calibration.coords,
            roi_radius=config.options.roi_size_mm,
            indices=cst_orig_idx,
        )

    # Calculate Physics
    if ui:
        ui.update_step_detail("Calculating activating function...")
    new_sl_cst, af_cst, len_cst, cst_orig_idx = physics.calculate_scalar_map(
        cst_streamlines,
        e_vecs_list_cst,
        mode="af",
        indices=cst_orig_idx,
    )

    # Save CST TRK
    cst_trk_path = m1_out / "M1_CST_af.trk"
    io.save_tract_with_data(sft_cst, new_sl_cst, cst_trk_path, "AF", af_cst, len_cst)

    # Save CST NIfTI (AF is signed; NIfTI stores magnitude for viewer compat).
    if config.options.generate_visualizations:
        io.save_points_as_nifti(
            np.concatenate(new_sl_cst),
            config.subject.t1w_path,
            m1_out / "M1_CST_af.nii.gz",
            values=np.abs(np.concatenate(af_cst)),
        )

    # Configure CST Analysis
    cst_config = AnalysisConfig(
        cst_trk=str(cst_trk_path),
        target_trk="",
        rmt=config.calibration.measured_rmt_mso,
        cst_coords=np.array(config.calibration.coords),
        target_coords=np.zeros(3),
        surf_path=str(config.subject.surface_path) if config.subject.surface_path else None,
        gwi_threshold=config.options.gwi_threshold_mm,
        roi_radius=config.options.roi_size_mm,
        activation_len=config.options.activation_length_mm,
        cst_weights=(
            str(config.subject.weights_cst_path) if config.subject.weights_cst_path else None
        ),
    )

    # Run Unified Analysis for CST
    if ui:
        ui.update_step_detail("Computing CST calibration efficiency...")
    log.info("Calculating CST Efficiency (Unified Method)...")
    cst_res = analyze_bundle(
        name="CST",
        trk_path=str(cst_trk_path),
        roi_center=np.array(config.calibration.coords),
        config=cst_config,
        surface_tree=surf_tree,
        weight_path=cst_config.cst_weights,
        orig_indices=cst_orig_idx,
    )

    af_cst_calibration = cst_res.metric_weighted

    validate_calibration_metrics(af_cst_calibration, cst_res.metric_unweighted)

    log.info(f"CST Calibration Efficiency (Weighted): {af_cst_calibration:.4f} V/m²")

    # Pre-Calculation for Summary Report
    intensity_rmt = config.coil.device_didt_max * (config.calibration.measured_rmt_mso / 100.0)
    biological_threshold = intensity_rmt * (af_cst_calibration / 1e6)

    # CST geometric metrics (constant across grid points)
    cst_roi_masks_post, _ = tractography.get_roi_masks(
        new_sl_cst, config.options.roi_size_mm, config.calibration.coords
    )
    cst_align, cst_depth = calculate_alignment_and_depth(
        new_sl_cst,
        e_vecs_list_cst,
        cst_roi_masks_post,
        mesh_m1,
        config.calibration.coords,
    )
    cst_align_corrected = calculate_alignment_corrected(
        new_sl_cst,
        e_vecs_list_cst,
        cst_roi_masks_post,
    )

    # Sample target tract E-field on M1 mesh once for per-point validation
    sft_tgt_full = tractography.load_tract(config.target.bundle_path, config.subject.t1w_path)
    points_tgt_full = np.concatenate(sft_tgt_full.streamlines)
    e_vectors_tgt_in_m1 = sample_field_at_coordinates(
        mesh_m1, points_tgt_full, "E", output_dir=m1_out, file_prefix="Target_in_M1"
    )
    e_vecs_list_tgt_in_m1_full = split_vectors_by_streamline(
        e_vectors_tgt_in_m1,
        sft_tgt_full.streamlines,
    )
    tgt_streamlines_full = list(sft_tgt_full.streamlines)

    # M1 coil matrix string for per-point summary
    if (
        isinstance(cal_orientation, list)
        and len(cal_orientation) == 4
        and isinstance(cal_orientation[0], list)
    ):
        m1_matrix_str = str(cal_orientation).replace("\n", "")
    else:
        m1_matrix_str = (
            str(cal_orientation).replace("\n", "") if cal_orientation is not None else "N/A"
        )

    if ui:
        ui.update_step(2, "complete")

    # ==========================================================================
    # STEP 3: COORDINATE CHAIN (Sequential)
    # ==========================================================================
    if ui:
        ui.update_step(3, "running")
    log.info("--- Step 3: Establish Fixed Scalp Center ---")

    if config.grid.scalp_coords:
        if ui:
            ui.update_step_detail("Using provided scalp coordinates...")
        log.info(f"Using provided grid scalp coordinates: {config.grid.scalp_coords}")
        fixed_scalp_coords = np.array(config.grid.scalp_coords)
    else:
        if ui:
            ui.update_step_detail("Projecting cortex target to scalp surface...")
        log.info("No grid scalp_coords provided. Projecting grid center (cortex) to scalp...")
        fixed_scalp_coords = project_target_to_scalp(
            mesh_path=msh_file, target_coords=np.array(config.grid.coords)
        )
        log.info(f"Calculated Fixed Scalp Center: {fixed_scalp_coords}")

    if ui:
        ui.update_step(3, "complete")

    # ==========================================================================
    # STEP 4: GRID ORIENTATION (Sequential)
    # ==========================================================================
    if ui:
        ui.update_step(4, "running")
    log.info("--- Step 4: Establish Grid Orientation Reference ---")

    if config.grid.orientation:
        if ui:
            ui.update_step_detail("Using provided grid orientation...")
        grid_orientation_ref = config.grid.orientation
        log.info(f"Using provided grid orientation (pos_ydir): {grid_orientation_ref}")
    else:
        if ui:
            ui.update_step_detail("Computing automatic coil orientation...")
        log.info(
            "No grid orientation provided. Computing automatic default (45 deg posterior-medial)..."
        )
        grid_orientation_ref = compute_default_coil_orientation(
            mesh_path=msh_file, scalp_coords=fixed_scalp_coords
        )
        log.info(f"Calculated Automatic Orientation (pos_ydir): {grid_orientation_ref}")

    if hasattr(grid_orientation_ref, "tolist"):
        grid_orientation_ref = grid_orientation_ref.tolist()

    log.info(f"[GRID_SEARCH] Final grid_orientation_ref (pos_ydir): {grid_orientation_ref}")

    if ui:
        ui.update_step(4, "complete")

    # ==========================================================================
    # STEP 5: GRID GENERATION (Sequential)
    # ==========================================================================
    if ui:
        ui.update_step(5, "running")
    log.info("--- Step 5: Generating Grid Targets ---")

    if ui:
        ui.update_step_detail("Extracting cortical grid points from tractogram...")
    grid_points = tractography.extract_grid_endpoints(
        trk_path=config.target.bundle_path,
        anat_path=config.subject.t1w_path,
        step_mm=config.grid.step_size_mm,
        cortex_thickness_mm=config.grid.cortex_depth_mm,
        target_center=config.grid.coords,
        search_radius=config.grid.search_radius_mm,
    )

    # Save Global Grid Mask
    if grid_points and config.options.generate_visualizations:
        if ui:
            ui.update_step_detail(f"Saving {len(grid_points)} grid points...")
        log.info(f"Saving {len(grid_points)} grid points to NIfTI mask...")
        io.save_points_as_nifti(
            np.array(grid_points),
            config.subject.t1w_path,
            out_dir / "grid_points_mask.nii.gz",
        )
    elif not grid_points:
        if ui:
            ui.stop()
        raise WorkflowError("No grid points generated.")

    # Init CSV
    results_csv = out_dir / "TIDE_grid_results.csv"
    initialize_grid_results_csv(results_csv)

    if ui:
        ui.update_step(5, "complete")

    # ==========================================================================
    # STEP 6: GRID SEARCH (PARALLEL or Sequential)
    # ==========================================================================
    log.info("--- Step 6: Running Grid Search (Optimization + Simulation) ---")

    num_points = len(grid_points)

    # Determine actual worker count
    if use_no_parallel:
        worker_plan = _resolve_grid_worker_plan(1, num_points)
        actual_workers = worker_plan.workers
        log.info("Parallel processing DISABLED. Running sequentially.")
    else:
        worker_plan = _resolve_grid_worker_plan(use_max_workers, num_points)
        actual_workers = worker_plan.workers
        if use_max_workers is None:
            desired_workers = min(
                max(1, worker_plan.cpu_count - 1),
                DEFAULT_MAX_WORKERS,
                num_points,
            )
        else:
            desired_workers = min(use_max_workers, num_points)
        if not worker_plan.forced and actual_workers < desired_workers:
            log.warning(
                "Grid workers capped at %d by the PARDISO memory model "
                "(%s GB available, %.1f GB reserve, %.1f GB/worker). Set %s=1 "
                "to force the requested count.",
                actual_workers,
                (
                    f"{worker_plan.available_memory_gb:.1f}"
                    if worker_plan.available_memory_gb is not None
                    else "unknown"
                ),
                worker_plan.memory_reserve_gb,
                worker_plan.memory_per_worker_gb,
                GRID_FORCE_WORKERS_ENV,
            )
        log.info(f"Processing {num_points} grid points with {actual_workers} workers")

    # Transition UI to parallel mode if using multiple workers
    if ui and actual_workers > 1:
        ui.transition_to_parallel(actual_workers, num_points)
    elif ui:
        ui.update_step(6, "running")

    # Create tasks
    tasks = []
    for i, cortex_coord in enumerate(grid_points):
        point_label = f"grid_P0{i}" if i < 10 else f"grid_P{i}"
        point_dir = sims_dir / point_label

        task = GridPointTask(
            index=i,
            cortex_coord=list(cortex_coord),
            point_label=point_label,
            point_dir=str(point_dir),
            msh_file=str(msh_file),
            m2m_path=str(config.subject.m2m_path),
            coil_path=str(config.coil.coil_path),
            fixed_scalp_coords=(
                fixed_scalp_coords.tolist()
                if hasattr(fixed_scalp_coords, "tolist")
                else list(fixed_scalp_coords)
            ),
            grid_orientation_ref=grid_orientation_ref,
            target_bundle_path=str(config.target.bundle_path),
            t1w_path=str(config.subject.t1w_path),
            surface_path=(
                str(config.subject.surface_path) if config.subject.surface_path else None
            ),
            weights_target_path=(
                str(config.subject.weights_target_path)
                if config.subject.weights_target_path
                else None
            ),
            adm_optimization=config.options.adm_optimization,
            opt_spatial_resolution=config.options.opt_spatial_resolution,
            opt_angle_resolution=config.options.opt_angle_resolution,
            opt_search_angle=config.options.opt_search_angle,
            opt_search_radius=config.options.opt_search_radius,
            field_mode=config.options.field_mode,
            roi_size_mm=config.options.roi_size_mm,
            activation_length_mm=config.options.activation_length_mm,
            gwi_threshold_mm=config.options.gwi_threshold_mm,
            max_angular_deviation_deg=config.options.max_angular_deviation_deg,
            measured_rmt_mso=config.calibration.measured_rmt_mso,
            af_cst_calibration=af_cst_calibration,
            cst_metric_unweighted=cst_res.metric_unweighted,
            mso_floor_ratio=config.options.mso_floor_ratio,
            mso_ceiling_ratio=config.options.mso_ceiling_ratio,
            generate_visualizations=config.options.generate_visualizations,
        )
        tasks.append(task)

    # Execute tasks
    results: List[GridPointResult] = []
    failed_count = 0

    # ctx was created at the start of the workflow (before UI creation)
    # to ensure the UI's status_queue uses the same context as workers

    # Determine if we should use parallel UI features
    use_parallel_ui = ui is not None and actual_workers > 1

    # Create worker logs directory for parallel mode
    worker_logs_dir = out_dir / "worker_logs" if use_parallel_ui else None
    if worker_logs_dir:
        worker_logs_dir.mkdir(exist_ok=True)

    if actual_workers == 1:
        # Sequential processing (no console UI)
        for i, task in enumerate(tasks):
            log.info(f"=== Processing {task.point_label} ({i + 1}/{num_points}) ===")
            result = process_grid_point(task)
            results.append(result)
            if not result.success:
                failed_count += 1
                log.error(f"Failed: {result.point_label} - {result.error_message}")

            # Emit progress callback
            if progress_callback:
                progress_callback(i + 1, num_points, task.point_label)
    else:
        # Parallel processing with optional console UI
        # CRITICAL: Use "spawn" context to avoid MPI/PETSc fork conflicts.
        # The default "fork" method inherits parent MPI state, causing segfaults.

        # Import worker wrapper if parallel UI is enabled
        if use_parallel_ui:
            from tide.console import process_grid_point_with_reporting

            worker_func = process_grid_point_with_reporting
        else:
            worker_func = process_grid_point

        # Prepare lock directory for robust worker ID assignment
        lock_dir_obj = tempfile.TemporaryDirectory(prefix="tide_worker_locks_")
        lock_dir = lock_dir_obj.name

        # Single-thread the numerical libraries for the spawned workers before
        # the pool starts, so children inherit it before importing NumPy/SimNIBS
        # (audit C-004); restored on exit for parent-side subprocesses.
        worker_pool = ProcessPoolExecutor(
            max_workers=actual_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(actual_workers, lock_dir),
        )
        with _single_thread_child_env(), worker_pool as executor:
            # Submit all tasks
            if use_parallel_ui:
                # Use reporting wrapper with status queue
                future_to_task = {}
                for idx, task in enumerate(tasks):
                    future = executor.submit(
                        worker_func,
                        task,
                        ui.status_queue,
                        -1,  # Use persistent ID from initializer
                        worker_logs_dir,
                    )
                    future_to_task[future] = task
            else:
                # Use original worker function
                future_to_task = {executor.submit(process_grid_point, task): task for task in tasks}

            results, failed_count = _collect_parallel_grid_results(
                future_to_task,
                num_points,
                use_parallel_ui,
                progress_callback,
            )

    # Sort results by index to maintain order
    results.sort(key=lambda r: r.index)

    estimation_valid_count = sum(
        result.success
        and result.weighted_mso_flag != "ESTIMATION_FAILED"
        and result.unweighted_mso_flag != "ESTIMATION_FAILED"
        and np.isfinite(result.weighted_mso_raw)
        and np.isfinite(result.unweighted_mso_raw)
        for result in results
    )
    log.info(
        "Grid search complete: %d/%d processed successfully; %d/%d produced "
        "valid intensity estimates",
        num_points - failed_count,
        num_points,
        estimation_valid_count,
        num_points,
    )

    if ui:
        ui.update_step(6, "complete")

    # ==========================================================================
    # STEP 7: WRITE RESULTS AND SAVE CONFIGS
    # ==========================================================================
    if ui:
        ui.update_step(7, "running")
    log.info("--- Step 7: Writing Results ---")

    if ui:
        ui.update_step_detail("Writing results to CSV and configs...")
    reporting_context = GridReportingContext(
        config=config,
        out_dir=out_dir,
        sims_dir=sims_dir,
        results_csv=results_csv,
        fixed_scalp_coords=fixed_scalp_coords,
        grid_orientation_ref=grid_orientation_ref,
        calibration_orientation=cal_orientation,
        target_streamlines_full=tgt_streamlines_full,
        target_vectors_in_m1=e_vecs_list_tgt_in_m1_full,
        cst_result=cst_res,
        af_cst_calibration=af_cst_calibration,
        cst_align=cst_align,
        cst_align_corrected=cst_align_corrected,
        cst_depth=cst_depth,
        intensity_rmt=intensity_rmt,
        biological_threshold=biological_threshold,
        m1_matrix_str=m1_matrix_str,
        spatial_mode=spatial_mode,
        num_workers=actual_workers,
        calibration_pose_qc=calibration_pose_qc,
        start_time=start_time,
        worker_memory_model=worker_plan.as_dict(),
    )
    final_grid_results = write_grid_results(results, reporting_context)

    if ui:
        ui.update_step_detail("Generating summary report...")
    log.info("Generating final summary report...")

    summary_result = write_grid_summary(results, final_grid_results, reporting_context)
    summary_path = summary_result.summary_path
    elapsed_time = summary_result.elapsed_time
    stats_weighted = summary_result.weighted_statistics
    stats_unweighted = summary_result.unweighted_statistics
    stats_raw_weighted = summary_result.weighted_raw_statistics
    stats_raw_unweighted = summary_result.unweighted_raw_statistics
    stats_mult_weighted = summary_result.weighted_multiplier_statistics
    stats_mult_unweighted = summary_result.unweighted_multiplier_statistics
    status_counts = summary_result.status_counts

    if ui:
        ui.update_step(7, "complete")

    # ==========================================================================
    # STEP 8: VISUALIZATION
    # ==========================================================================
    log.info("--- Step 8: Generating Visualizations ---")
    if ui:
        ui.update_step(8, "running")
        ui.update_step_detail("Running grid visualization script...")

    viz_dir = out_dir / "visualization"
    if config.options.generate_visualizations:
        viz_dir.mkdir(parents=True, exist_ok=True)
        try:
            from tide.interfaces.grid_visualization import run_grid_visualization

            run_grid_visualization(
                csv_path=results_csv,
                t1w_path=config.subject.t1w_path,
                trk_path=config.target.bundle_path,
                output_dir=viz_dir,
                generate_interactive=config.options.generate_3d_visualization,
            )
            log.info(f"Visualization outputs saved to: {viz_dir}")
        except Exception as e:
            log.warning(f"Visualization step failed: {e}")
    else:
        log.info("Visualization artifacts disabled by configuration.")

    if ui:
        ui.update_step(8, "complete")

    # Render console UI final summary if enabled
    if ui:
        # Prepare results for summary display
        ui_results = [
            {
                "label": r.point_label,
                "weighted_mso": r.weighted_mso,
                "unweighted_mso": r.unweighted_mso,
                "weighted_mso_raw": r.weighted_mso_raw,
                "unweighted_mso_raw": r.unweighted_mso_raw,
                "weighted_flag": r.weighted_mso_flag,
                "unweighted_flag": r.unweighted_mso_flag,
                "success": r.success,
                "cortex_coord": r.cortex_coord,
            }
            for r in results
        ]

        # Prepare output files list
        output_files = [
            ("Results CSV", results_csv),
            ("Summary", summary_path),
            ("HTML Report", summary_path.with_suffix(".html")),
            ("Simulations", sims_dir),
        ]
        if config.options.generate_visualizations:
            output_files.append(("Visualizations", viz_dir))

        # This will stop the UI and render final summary
        ui.render_final_summary(
            ui_results,
            elapsed_time,
            output_files,
            stats_summary={
                "weighted": stats_weighted,
                "unweighted": stats_unweighted,
                "raw_weighted": stats_raw_weighted,
                "raw_unweighted": stats_raw_unweighted,
                "multiplier_weighted": stats_mult_weighted,
                "multiplier_unweighted": stats_mult_unweighted,
                "status_counts": status_counts,
            },
        )
    else:
        # Log Stats for no-UI mode
        log.info("=== Statistical Summary ===")
        log.info(f"{'Metric':<25} | {'Unweighted':<10} | {'Weighted':<10}")
        log.info("-" * 51)

        metrics = ["mean", "median", "std", "mean_no_outliers", "outlier_count"]
        labels = [
            "Mean",
            "Median",
            "Std Dev",
            "Mean (w/o outliers)",
            "Outliers (>2 SD)",
        ]

        for metric, label in zip(metrics, labels):
            u_val = stats_unweighted.get(metric, 0)
            w_val = stats_weighted.get(metric, 0)

            if metric == "outlier_count":
                log.info(f"{label:<25} | {int(u_val):<10} | {int(w_val):<10}")
            else:
                log.info(f"{label:<25} | {u_val:<10.2f} | {w_val:<10.2f}")

        # Multiplier (M_CST/M_target) — same panel as intensity so the user can
        # rescale dose without rerunning.
        log.info(
            f"{'Mean Multiplier':<25} | "
            f"{stats_mult_unweighted.get('mean', 0):<10.4f} | "
            f"{stats_mult_weighted.get('mean', 0):<10.4f}"
        )
        log.info(
            f"{'Median Multiplier':<25} | "
            f"{stats_mult_unweighted.get('median', 0):<10.4f} | "
            f"{stats_mult_weighted.get('median', 0):<10.4f}"
        )
        log.info(
            f"{'Mean Raw I':<25} | "
            f"{stats_raw_unweighted.get('mean', 0):<10.2f} | "
            f"{stats_raw_weighted.get('mean', 0):<10.2f}"
        )
        log.info(
            f"{'Estimation Failures':<25} | "
            f"{status_counts['unweighted']['estimation_failed']:<10} | "
            f"{status_counts['weighted']['estimation_failed']:<10}"
        )
        log.info("=" * 51)
        log.info("--- Output files ---")
        log.info(f"  -> Results CSV: {Path(results_csv).resolve()}")
        log.info(f"  -> Summary: {Path(summary_path).resolve()}")
        log.info(f"  -> HTML Report: {Path(summary_path).with_suffix('.html').resolve()}")
        log.info(f"  -> Simulations: {Path(sims_dir).resolve()}")
        if config.options.generate_visualizations:
            log.info(f"  -> Visualizations: {Path(viz_dir).resolve()}")
        else:
            log.info("  -> Visualizations: Disabled by configuration")
