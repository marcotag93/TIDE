"""
TIDE Estimation Workflow
========================
Main workflow for estimating target intensity using the Unified Estimation Method.
Includes 3D visualization generation.

Parallelization Strategy (2-Worker System)
------------------------------------------
The workflow uses a 2-worker parallel processing system to accelerate the
computationally intensive optimization and simulation phases:

    ┌─────────────────────────────────────────────────────────────┐
    │ Worker 1 (M1/CST Pipeline)    │ Worker 2 (Target Pipeline)  │
    ├─────────────────────────────────────────────────────────────┤
    │ M1 Optimization               │ Target Optimization         │
    │         ↓                     │         ↓                   │
    │ M1 Simulation                 │ Target Simulation           │
    │         ↓                     │         ↓                   │
    │ CST E-field Sampling          │ Target E-field Sampling     │
    │         ↓                     │         ↓                   │
    │ CST AF Calculation            │ Target AF Calculation       │
    └─────────────────────────────────────────────────────────────┘
                                  ↓
                        [Synchronization Point]
                                  ↓
                  Validation (needs both M1 mesh + Target data)
                                  ↓
                          Unified Estimation
                                  ↓
                           Visualizations

This provides up to 2x speedup for the optimization and simulation phases,
which are the most computationally intensive parts of the workflow.

Mesh Caching Strategy
--------------------
SimNIBS mesh objects contain internal solver state and are not picklable,
preventing direct sharing between processes. To optimize mesh loading:

1. The main process pre-loads the mesh file before spawning workers
2. This "warms up" the OS disk cache with the ~500MB mesh data
3. Workers benefit from cached disk I/O when loading the same mesh file
4. Each worker still creates its own mesh object, but I/O is near-instant

This approach provides implicit parallelization benefits without requiring
modifications to SimNIBS internals or complex shared memory setups.
"""

import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tide.core import geometry, io, physics, tractography
from tide.core.geometry import (
    calculate_alignment_and_depth,
    calculate_alignment_corrected,
    evaluate_coil_pose_qc,
    validate_coil_pose_for_dose,
)
from tide.interfaces.sampling import sample_field_at_coordinates
from tide.interfaces.simnibs_interface import SimNIBSInterface
from tide.interfaces.unified_estimation import format_weight_sources, run_unified_estimation
from tide.interfaces.visualization_3d import (
    PYVISTA_AVAILABLE,
    VisualizationConfig,
    generate_bundle_visualization,
)
from tide.utils.config import (
    SimNIBSConfig,
    orientation_is_matrix,
    save_config_to_output,
    validate_workflow_config,
)
from tide.workflows._shared import WorkflowError, calculate_target_in_field_metric
from tide.workflows._shared import configure_worker_environment as _configure_worker_environment
from tide.workflows._shared import single_thread_child_environment as _single_thread_child_env
from tide.workflows._shared import split_vectors_by_streamline as _split_vectors_by_streamline

log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

NUM_ESTIMATION_WORKERS = 2

ESTIMATION_STEPS = {
    1: "Pre-processing",
    2: "Parallel Estimation",
    3: "Validation (Target in M1)",
    4: "Unified Estimation",
    5: "Visualizations",
    6: "Summary Report",
}


# =============================================================================
# DATACLASSES FOR PARALLEL TASKS
# =============================================================================


@dataclass
class PipelineTask:
    """
    Encapsulates all data needed to run a complete site pipeline
    (optimization + simulation + E-field sampling + AF calculation).

    This dataclass is designed to be picklable for inter-process communication.
    All Path objects are converted to strings for serialization.
    """

    task_type: str  # 'm1' or 'target'
    label: str  # Site label (e.g., 'M1' or target name)

    # Mesh paths
    mesh_path: str  # .msh file for optimization
    m2m_path: str  # m2m directory for simulation

    # Output
    output_dir: str

    # Coil configuration
    coil_path: str
    coil_distance_mm: float

    # Optimization parameters
    target_coords: List[float]  # Cortical target
    scalp_coords: Optional[List[float]]  # Initial scalp position
    orientation_ref: Optional[Any]  # Orientation reference (list or EEG label)
    needs_optimization: bool  # Whether to run optimization
    opt_didt: float
    opt_search_radius: float
    opt_spatial_resolution: float
    opt_angle_resolution: float
    opt_search_angle: float
    use_adm: bool

    # Simulation parameters
    sim_didt: float
    sim_coords: Optional[List[float]]  # Only used if not optimizing
    sim_orientation: Optional[Any]  # Only used if not optimizing (can be matrix)

    # Tractography and analysis
    bundle_path: str
    t1w_path: str
    roi_coords: List[float]
    roi_size_mm: float
    field_mode: str
    max_angular_deviation_deg: float = 0.0


@dataclass
class PipelineResult:
    """
    Results from a complete site pipeline execution.
    """

    task_type: str
    label: str
    success: bool

    # Optimization results
    opt_matrix: Optional[List[List[float]]]
    opt_scalp_coords: Optional[List[float]]

    # Simulation results
    mesh_path: Optional[str]
    trk_path: Optional[str]

    # Computed data (for downstream analysis)
    streamlines: Optional[List[np.ndarray]]
    af_values: Optional[List[np.ndarray]]
    len_values: Optional[List[np.ndarray]]
    e_vecs_list: Optional[List[np.ndarray]]
    roi_masks: Optional[List[np.ndarray]]
    roi_segments: Optional[List[np.ndarray]]

    # Original bundle ids of the surviving TRK streamlines, in TRK order
    # (audit C-002; keeps SIFT2 weights aligned in run_unified_estimation).
    orig_indices: Optional[np.ndarray] = None
    pose_qc: Optional[Dict[str, Any]] = None

    error_message: Optional[str] = None


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================


def _run_pipeline_task(task: PipelineTask) -> PipelineResult:
    """
    Worker function to run the complete site pipeline.

    This function is executed in a separate process and performs:
    1. Coil position optimization (if needed)
    2. FEM E-field simulation
    3. Tractogram loading
    4. E-field sampling on streamlines
    5. Activating function calculation
    6. TRK file saving

    Args:
        task: PipelineTask with all necessary parameters.

    Returns:
        PipelineResult with computed data or error information.
    """
    # Configure environment before heavy imports
    _configure_worker_environment()

    # Import modules inside worker to ensure environment is set
    from pathlib import Path

    import numpy as np

    from tide.core import io, tractography
    from tide.interfaces.sampling import sample_field_at_coordinates

    # Register highlight method for spawned process
    from tide.utils.logging import highlight

    logging.Logger.highlight = highlight

    log = logging.getLogger(__name__)

    output_dir = Path(task.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    opt_matrix = None
    opt_scalp_coords = None
    pose_qc = None
    sim_coords = task.sim_coords
    sim_orientation = task.sim_orientation

    try:
        # =====================================================================
        # Step 1: Optimization (if needed)
        # =====================================================================
        if task.needs_optimization:
            log.info(f"[{task.label}] Running coil position optimization...")

            opt_matrix_np, opt_scalp_np = SimNIBSInterface.run_optimization(
                mesh_path=Path(task.mesh_path),
                output_dir=output_dir,
                coil_path=Path(task.coil_path),
                target_coords=task.target_coords,
                scalp_centre=task.scalp_coords,
                orientation_ref=task.orientation_ref,
                didt=task.opt_didt,
                use_adm=task.use_adm,
                spatial_resolution=task.opt_spatial_resolution,
                angle_resolution=task.opt_angle_resolution,
                search_angle=task.opt_search_angle,
                search_radius_mm=task.opt_search_radius,
            )

            opt_matrix = opt_matrix_np.tolist()
            opt_scalp_coords = opt_scalp_np.tolist()
            sim_orientation = opt_matrix
            sim_coords = None
            qc = evaluate_coil_pose_qc(Path(task.mesh_path), opt_matrix_np, opt_scalp_np)
            pose_qc = qc.as_dict()
            if qc.status == "WARN":
                log.warning(f"[{task.label}] Coil pose QC warning: {qc.reasons}")

            # Save optimization result
            opt_filename = f"{task.label}_opt_result.txt"
            io.save_optimization_result_txt(
                output_dir / opt_filename,
                opt_matrix_np,
                opt_scalp_np,
                pose_qc=pose_qc,
            )
            validate_coil_pose_for_dose(qc, explicit_matrix=False)

            log.info(f"[{task.label}] Optimization complete.")
        elif (
            isinstance(sim_orientation, list)
            and len(sim_orientation) == 4
            and isinstance(sim_orientation[0], list)
        ):
            qc = evaluate_coil_pose_qc(Path(task.mesh_path), np.asarray(sim_orientation))
            pose_qc = qc.as_dict()
            if qc.status == "WARN":
                log.warning(f"[{task.label}] Coil pose QC warning: {qc.reasons}")
            validate_coil_pose_for_dose(qc, explicit_matrix=True)

        # =====================================================================
        # Step 2: Simulation
        # =====================================================================
        log.info(f"[{task.label}] Running FEM E-field simulation...")

        mesh_result = SimNIBSInterface.run_simulation(
            mesh_path=Path(task.m2m_path),
            output_dir=output_dir,
            coil_path=Path(task.coil_path),
            didt=task.sim_didt,
            coords=sim_coords,
            orientation=sim_orientation,
            distance_mm=task.coil_distance_mm,
        )

        log.info(f"[{task.label}] Simulation complete, loading tractogram...")

        # =====================================================================
        # Step 3: Load tractogram and sample E-field
        # =====================================================================
        sft = tractography.load_tract(Path(task.bundle_path), Path(task.t1w_path))
        points = np.concatenate(sft.streamlines)

        log.info(f"[{task.label}] Sampling E-field on streamlines...")

        # Sample E-field
        prefix = "M1_CST" if task.task_type == "m1" else task.label
        e_vectors = sample_field_at_coordinates(
            mesh_result, points, "E", output_dir=output_dir, file_prefix=prefix
        )

        # Split vectors by streamline
        e_vecs_list = []
        idx = 0
        for sl in sft.streamlines:
            e_vecs_list.append(e_vectors[idx : idx + len(sl)])
            idx += len(sl)

        # =====================================================================
        # Step 3b: Filter streamlines by angular deviation
        # =====================================================================
        # Track original streamline ids through the drop chain so SIFT2 weights
        # stay attached to their streamlines downstream (audit C-002).
        orig_idx = np.arange(len(sft.streamlines))
        if task.max_angular_deviation_deg > 0:
            (
                filtered_sl,
                e_vecs_list,
                n_removed,
                orig_idx,
            ) = tractography.filter_by_angular_deviation(
                list(sft.streamlines),
                e_field_vectors=e_vecs_list,
                max_angle_deg=task.max_angular_deviation_deg,
                roi_center=task.roi_coords,
                roi_radius=task.roi_size_mm,
                indices=orig_idx,
            )
        else:
            filtered_sl = list(sft.streamlines)

        # =====================================================================
        # Step 4: Calculate activating function
        # =====================================================================
        log.info(f"[{task.label}] Calculating activating function...")

        new_sl, af_values, len_values, orig_idx = physics.calculate_scalar_map(
            filtered_sl,
            e_vecs_list,
            mode=task.field_mode,
            indices=orig_idx,
        )

        # Save TRK
        trk_filename = "CST_M1_af.trk" if task.task_type == "m1" else f"{task.label}_af.trk"
        trk_path = output_dir / trk_filename
        io.save_tract_with_data(sft, new_sl, trk_path, "AF", af_values, len_values)

        # Get ROI masks (on filtered/midpoint streamlines to match af_values)
        roi_masks, roi_segments = tractography.get_roi_masks(
            new_sl, task.roi_size_mm, task.roi_coords
        )

        log.info(f"[{task.label}] Pipeline complete.")

        return PipelineResult(
            task_type=task.task_type,
            label=task.label,
            success=True,
            opt_matrix=opt_matrix,
            opt_scalp_coords=opt_scalp_coords,
            mesh_path=str(mesh_result),
            trk_path=str(trk_path),
            streamlines=new_sl,
            af_values=af_values,
            len_values=len_values,
            e_vecs_list=e_vecs_list,
            roi_masks=roi_masks,
            roi_segments=roi_segments,
            orig_indices=orig_idx,
            pose_qc=pose_qc,
        )

    except Exception as e:
        log.error(f"[{task.label}] Pipeline failed: {e}")
        import traceback

        log.error(traceback.format_exc())
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
            pose_qc=pose_qc,
            error_message=str(e),
        )


def _warmup_mesh_cache(mesh_path: Path) -> None:
    """
    Pre-load the mesh file to warm up the OS disk cache.

    This function reads the mesh file into memory, causing the OS to cache
    the file data. When worker processes subsequently load the same file,
    they benefit from the cached I/O, significantly reducing load times.

    Args:
        mesh_path: Path to the mesh file (.msh) or m2m directory.
    """
    log.info("Warming up mesh cache for parallel workers...")

    # Find the actual .msh file
    if mesh_path.is_dir():
        # m2m directory - find the mesh file
        msh_files = list(mesh_path.glob("*.msh"))
        if not msh_files:
            # Check parent directory for subject mesh
            parent_msh = list(mesh_path.parent.glob("*.msh"))
            msh_files = parent_msh
    else:
        msh_files = [mesh_path]

    for msh_file in msh_files:
        if msh_file.exists():
            try:
                # Read the file to populate OS cache
                file_size_mb = msh_file.stat().st_size / (1024 * 1024)
                log.debug(f"Pre-loading mesh: {msh_file.name} ({file_size_mb:.1f} MB)")

                # Read in chunks to avoid memory issues with very large files
                chunk_size = 64 * 1024 * 1024  # 64 MB chunks
                with open(msh_file, "rb") as f:
                    while f.read(chunk_size):
                        pass

                log.debug(f"Mesh cache warmed: {msh_file.name}")
            except Exception as e:
                log.warning(f"Could not warm cache for {msh_file}: {e}")


@dataclass(frozen=True)
class EstimationPreparation:
    out_dir: Path
    m1_out: Path
    tgt_out: Path
    viz_out: Path
    m1_task: PipelineTask
    tgt_task: PipelineTask
    calibration_orientation: Any
    target_orientation: Any
    target_orientation_is_matrix: bool


@dataclass(frozen=True)
class EstimationAnalysis:
    calibration_orientation: Any
    mesh_m1: Path
    mesh_target: Path
    cst_streamlines: List[np.ndarray]
    cst_af_values: List[np.ndarray]
    target_streamlines: List[np.ndarray]
    target_af_values: List[np.ndarray]
    optimized_matrix: Optional[np.ndarray]
    optimized_scalp_coords: Optional[np.ndarray]
    results: Dict[str, Any]
    af_cst_calibration: float
    af_target_optimized: float
    intensity_rmt: float
    biological_threshold: float
    cst_align: float
    target_align: float
    cst_depth: float
    target_depth: float
    optimization_gain: float
    ratio_at_m1: float
    intensity_from_m1_position: float
    cst_align_corrected: float
    target_align_corrected: float


@dataclass(frozen=True)
class EstimationSummaryContext:
    config: SimNIBSConfig
    ui: Any
    start_time: float
    out_dir: Path
    viz_out: Path
    calibration_orientation: Any
    target_orientation: Any
    target_orientation_is_matrix: bool
    optimized_matrix: Optional[np.ndarray]
    optimized_scalp_coords: Optional[np.ndarray]
    results: Dict[str, Any]
    af_cst_calibration: float
    af_target_optimized: float
    intensity_rmt: float
    biological_threshold: float
    cst_align: float
    target_align: float
    cst_depth: float
    target_depth: float
    optimization_gain: float
    ratio_at_m1: float
    intensity_from_m1_position: float
    cst_align_corrected: float
    target_align_corrected: float
    m1_result: PipelineResult
    target_result: PipelineResult


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


# =============================================================================
# MAIN WORKFLOW FUNCTION
# =============================================================================


def _prepare_estimation(
    config: SimNIBSConfig,
    ui: Any,
) -> EstimationPreparation:
    # =========================================================================
    # Pre-processing: Medoid Logic
    # =========================================================================
    if ui:
        ui.update_step(1, "running")

    if config.target.medoid_endpoint:
        if not config.target.bundle_path or not config.target.bundle_path.exists():
            raise WorkflowError("Medoid endpoint requested but bundle path is missing.")

        if ui:
            ui.update_step_detail(f"Computing cortical medoid for {config.target.label}...")
        log.highlight(f"Computing cortical medoid for: {config.target.label}")
        try:
            new_coords = tractography.get_bundle_cortical_medoid(
                config.target.bundle_path,
                config.subject.t1w_path,
                reference_coord=config.target.coords,
            )
            log.highlight(f"Medoid coordinates: {new_coords.tolist()}")
            config.target.coords = new_coords.tolist()
        except Exception as e:
            raise WorkflowError(f"Failed to calculate medoid: {e}") from e

    # =========================================================================
    # Setup output directories
    # =========================================================================
    out_dir = config.subject.derivatives_path / f"TIDE_{config.target.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    m1_out = out_dir / "sim_m1"
    m1_out.mkdir(parents=True, exist_ok=True)
    tgt_out = out_dir / "sim_target"
    tgt_out.mkdir(parents=True, exist_ok=True)
    viz_out = out_dir / "visualizations"
    viz_out.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Warm up mesh cache before spawning workers
    # =========================================================================
    _warmup_mesh_cache(config.subject.mesh_path)

    # =========================================================================
    # Determine optimization requirements
    # =========================================================================
    cal_orientation = config.calibration.orientation
    is_cal_matrix = orientation_is_matrix(cal_orientation)
    m1_needs_opt = not is_cal_matrix

    tgt_orientation = config.target.orientation
    is_tgt_matrix = orientation_is_matrix(tgt_orientation)
    tgt_needs_opt = not is_tgt_matrix

    if ui:
        ui.update_step(1, "complete")

    # =========================================================================
    # Create parallel tasks
    # =========================================================================
    log.highlight("--- Step 1: Running M1 and Target Pipelines in Parallel ---")

    m1_task = PipelineTask(
        task_type="m1",
        label="M1",
        mesh_path=str(config.subject.mesh_path),
        m2m_path=str(config.subject.m2m_path),
        output_dir=str(m1_out),
        coil_path=str(config.coil.coil_path),
        coil_distance_mm=config.coil.coil_distance_mm,
        target_coords=config.calibration.coords,
        scalp_coords=config.calibration.scalp_coords,
        orientation_ref=config.calibration.orientation if not is_cal_matrix else None,
        needs_optimization=m1_needs_opt,
        opt_didt=1e6,
        opt_search_radius=config.options.opt_search_radius,
        opt_spatial_resolution=config.options.opt_spatial_resolution,
        opt_angle_resolution=config.options.opt_angle_resolution,
        opt_search_angle=config.options.opt_search_angle,
        use_adm=config.options.adm_optimization,
        sim_didt=1e6,
        sim_coords=config.calibration.scalp_coords if not m1_needs_opt else None,
        sim_orientation=cal_orientation if is_cal_matrix else None,
        bundle_path=str(config.calibration.bundle_path),
        t1w_path=str(config.subject.t1w_path),
        roi_coords=config.calibration.coords,
        roi_size_mm=config.options.roi_size_mm,
        field_mode="af",
        max_angular_deviation_deg=config.options.max_angular_deviation_deg,
    )

    tgt_task = PipelineTask(
        task_type="target",
        label=config.target.label,
        mesh_path=str(config.subject.mesh_path),
        m2m_path=str(config.subject.m2m_path),
        output_dir=str(tgt_out),
        coil_path=str(config.coil.coil_path),
        coil_distance_mm=config.coil.coil_distance_mm,
        target_coords=config.target.coords,
        scalp_coords=config.target.scalp_coords,
        orientation_ref=config.target.orientation if not is_tgt_matrix else None,
        needs_optimization=tgt_needs_opt,
        opt_didt=config.coil.device_didt_max,
        opt_search_radius=config.options.opt_search_radius,
        opt_spatial_resolution=config.options.opt_spatial_resolution,
        opt_angle_resolution=config.options.opt_angle_resolution,
        opt_search_angle=config.options.opt_search_angle,
        use_adm=config.options.adm_optimization,
        sim_didt=1e6,
        sim_coords=config.target.scalp_coords if not tgt_needs_opt else None,
        sim_orientation=tgt_orientation if is_tgt_matrix else None,
        bundle_path=str(config.target.bundle_path),
        t1w_path=str(config.subject.t1w_path),
        roi_coords=config.target.coords,
        roi_size_mm=config.options.roi_size_mm,
        field_mode=config.options.field_mode,
        max_angular_deviation_deg=config.options.max_angular_deviation_deg,
    )
    return EstimationPreparation(
        out_dir=out_dir,
        m1_out=m1_out,
        tgt_out=tgt_out,
        viz_out=viz_out,
        m1_task=m1_task,
        tgt_task=tgt_task,
        calibration_orientation=cal_orientation,
        target_orientation=tgt_orientation,
        target_orientation_is_matrix=is_tgt_matrix,
    )


def _execute_estimation_tasks(
    ctx: Any,
    ui: Any,
    preparation: EstimationPreparation,
) -> tuple[PipelineResult, PipelineResult]:
    out_dir = preparation.out_dir
    m1_task = preparation.m1_task
    tgt_task = preparation.tgt_task

    # =========================================================================
    # Execute parallel pipelines
    # =========================================================================
    # ctx was created at start of workflow

    if ui:
        ui.transition_to_parallel(NUM_ESTIMATION_WORKERS, 2, step_num=2)
    else:
        log.highlight("--- Step 1: Running M1 and Target Pipelines in Parallel ---")

    # Import worker wrapper if UI is enabled
    if ui:
        from tide.console import process_pipeline_task_with_reporting

        worker_func = process_pipeline_task_with_reporting

        # Create logs directory
        worker_logs_dir = out_dir / "worker_logs"
        worker_logs_dir.mkdir(exist_ok=True)
    else:
        worker_func = _run_pipeline_task
        worker_logs_dir = None

    # Prepare worker IDs
    worker_id_queue = ctx.Queue()
    for i in range(NUM_ESTIMATION_WORKERS):
        worker_id_queue.put(i)

    m1_result = None
    tgt_result = None

    # Single-thread the numerical libraries for the spawned workers before the
    # pool starts, so children inherit it before importing NumPy/SimNIBS (audit
    # C-004); restored on exit for parent-side validation subprocesses.
    worker_pool = ProcessPoolExecutor(max_workers=NUM_ESTIMATION_WORKERS, mp_context=ctx)
    with _single_thread_child_env(), worker_pool as executor:
        # Submit tasks
        if ui:
            m1_future = executor.submit(worker_func, m1_task, ui.status_queue, 0, worker_logs_dir)
            tgt_future = executor.submit(worker_func, tgt_task, ui.status_queue, 1, worker_logs_dir)
        else:
            m1_future = executor.submit(_run_pipeline_task, m1_task)
            tgt_future = executor.submit(_run_pipeline_task, tgt_task)

        # Wait for both to complete
        wait([m1_future, tgt_future])

        # Get results
        m1_result = m1_future.result()
        tgt_result = tgt_future.result()

    if ui:
        ui.update_step(2, "complete")

    # Check for failures
    if not m1_result.success:
        raise WorkflowError(f"M1 pipeline failed: {m1_result.error_message}")

    if not tgt_result.success:
        raise WorkflowError(f"Target pipeline failed: {tgt_result.error_message}")

    log.info("Both pipelines completed successfully.")
    return m1_result, tgt_result


def _analyze_estimation(
    config: SimNIBSConfig,
    ui: Any,
    preparation: EstimationPreparation,
    m1_result: PipelineResult,
    tgt_result: PipelineResult,
) -> EstimationAnalysis:
    generated_calibration_matrix = m1_result.opt_matrix
    generated_target_matrix = tgt_result.opt_matrix
    cal_orientation = (
        m1_result.opt_matrix if m1_result.opt_matrix else config.calibration.orientation
    )

    mesh_m1 = Path(m1_result.mesh_path)
    mesh_tgt = Path(tgt_result.mesh_path)
    cst_trk_path = Path(m1_result.trk_path)
    tgt_trk_path = Path(tgt_result.trk_path)

    new_sl_cst = m1_result.streamlines
    af_cst = m1_result.af_values
    e_vecs_list_cst = m1_result.e_vecs_list
    cst_roi_masks = m1_result.roi_masks

    new_sl_tgt = tgt_result.streamlines
    af_tgt = tgt_result.af_values
    e_vecs_list_tgt = tgt_result.e_vecs_list
    tgt_roi_masks = tgt_result.roi_masks
    opt_matrix = np.array(tgt_result.opt_matrix) if tgt_result.opt_matrix else None
    opt_scalp_coords = (
        np.array(tgt_result.opt_scalp_coords) if tgt_result.opt_scalp_coords else None
    )

    # Save NIfTI visualizations if requested (AF is signed upstream; NIfTI
    # stores |AF| for compatibility with standard overlays/viewers).
    if config.options.generate_visualizations:
        io.save_points_as_nifti(
            np.concatenate(new_sl_cst),
            config.subject.t1w_path,
            preparation.m1_out / "CST_M1_af.nii.gz",
            values=np.abs(np.concatenate(af_cst)),
        )
        io.save_points_as_nifti(
            np.concatenate(new_sl_tgt),
            config.subject.t1w_path,
            preparation.tgt_out / f"{config.target.label}_af.nii.gz",
            values=np.abs(np.concatenate(af_tgt)),
        )

    # Calculate alignment and depth metrics
    cst_align, cst_depth = calculate_alignment_and_depth(
        new_sl_cst, e_vecs_list_cst, cst_roi_masks, mesh_m1, config.calibration.coords
    )
    tgt_align, tgt_depth = calculate_alignment_and_depth(
        new_sl_tgt, e_vecs_list_tgt, tgt_roi_masks, mesh_tgt, config.target.coords
    )
    cst_align_corrected = calculate_alignment_corrected(new_sl_cst, e_vecs_list_cst, cst_roi_masks)
    tgt_align_corrected = calculate_alignment_corrected(new_sl_tgt, e_vecs_list_tgt, tgt_roi_masks)

    # Save configuration (after optimization, with generated matrices and
    # post-optimization scalp coordinates) so the output YAML is fully
    # re-runnable via `--workflow estimation` without triggering a new
    # optimization or medoid computation.
    save_config_to_output(
        config,
        preparation.out_dir,
        "estimation",
        generated_calibration_matrix=generated_calibration_matrix,
        generated_target_matrix=generated_target_matrix,
        generated_calibration_scalp_coords=(
            list(m1_result.opt_scalp_coords) if m1_result.opt_scalp_coords is not None else None
        ),
        generated_target_scalp_coords=(
            list(tgt_result.opt_scalp_coords) if tgt_result.opt_scalp_coords is not None else None
        ),
        medoid_resolved=bool(config.target.medoid_endpoint),
    )

    # =========================================================================
    # Step 2: Validation - Target in M1 Field
    # =========================================================================
    if ui:
        ui.update_step(3, "running")
        ui.update_step_detail("Validating target in M1 field...")

    log.highlight("--- Step 2: Validation (Target in M1 Field) ---")

    # Load target tractogram for validation (need fresh sft object)
    sft_tgt = tractography.load_tract(config.target.bundle_path, config.subject.t1w_path)
    points_tgt = np.concatenate(sft_tgt.streamlines)

    e_vectors_tgt_m1 = sample_field_at_coordinates(
        mesh_m1, points_tgt, "E", output_dir=preparation.m1_out, file_prefix="Target_in_M1"
    )
    e_vecs_list_tgt_m1 = _split_vectors_by_streamline(e_vectors_tgt_m1, sft_tgt.streamlines)

    af_target_m1_calibration = calculate_target_in_field_metric(
        sft_tgt.streamlines,
        e_vecs_list_tgt_m1,
        roi_center=config.target.coords,
        roi_size_mm=config.options.roi_size_mm,
        activation_length_mm=config.options.activation_length_mm,
        max_angular_deviation_deg=config.options.max_angular_deviation_deg,
    )

    # =========================================================================
    # Step 3: Run Unified Estimation
    # =========================================================================
    if ui:
        ui.update_step(3, "complete")
        ui.update_step(4, "running")
        ui.update_step_detail("Calculating Unified Estimation Result...")
    log.highlight("--- Step 3: Unified Estimation ---")
    log.info(f"Calculating metrics for {config.target.label}...")

    results = run_unified_estimation(
        cst_trk=cst_trk_path,
        target_trk=tgt_trk_path,
        cst_coords=config.calibration.coords,
        target_coords=config.target.coords,
        rmt=config.calibration.measured_rmt_mso,
        weights_cst=config.subject.weights_cst_path,
        weights_target=config.subject.weights_target_path,
        surface_path=config.subject.surface_path,
        gwi_threshold=config.options.gwi_threshold_mm,
        roi_radius=config.options.roi_size_mm,
        activation_len=config.options.activation_length_mm,
        mso_floor_ratio=config.options.mso_floor_ratio,
        mso_ceiling_ratio=config.options.mso_ceiling_ratio,
        cst_orig_indices=m1_result.orig_indices,
        target_orig_indices=tgt_result.orig_indices,
    )

    af_cst_calibration = results["cst_metric"]
    af_target_optimized = results["tgt_metric"]
    optimization_gain = (
        af_target_optimized / af_target_m1_calibration if af_target_m1_calibration > 0 else 0.0
    )
    ratio_at_m1 = af_target_m1_calibration / af_cst_calibration if af_cst_calibration > 0 else 0.0
    intensity_from_m1_position = (
        config.calibration.measured_rmt_mso * (af_cst_calibration / af_target_m1_calibration)
        if af_target_m1_calibration > 0
        else 0.0
    )

    intensity_rmt = config.coil.device_didt_max * (config.calibration.measured_rmt_mso / 100.0)
    biological_threshold = intensity_rmt * (af_cst_calibration / 1e6)

    return EstimationAnalysis(
        calibration_orientation=cal_orientation,
        mesh_m1=mesh_m1,
        mesh_target=mesh_tgt,
        cst_streamlines=new_sl_cst,
        cst_af_values=af_cst,
        target_streamlines=new_sl_tgt,
        target_af_values=af_tgt,
        optimized_matrix=opt_matrix,
        optimized_scalp_coords=opt_scalp_coords,
        results=results,
        af_cst_calibration=af_cst_calibration,
        af_target_optimized=af_target_optimized,
        intensity_rmt=intensity_rmt,
        biological_threshold=biological_threshold,
        cst_align=cst_align,
        target_align=tgt_align,
        cst_depth=cst_depth,
        target_depth=tgt_depth,
        optimization_gain=optimization_gain,
        ratio_at_m1=ratio_at_m1,
        intensity_from_m1_position=intensity_from_m1_position,
        cst_align_corrected=cst_align_corrected,
        target_align_corrected=tgt_align_corrected,
    )


def _write_estimation_summary(context: EstimationSummaryContext) -> None:
    config = context.config
    ui = context.ui
    start_time = context.start_time
    out_dir = context.out_dir
    viz_out = context.viz_out
    cal_orientation = context.calibration_orientation
    tgt_orientation = context.target_orientation
    is_tgt_matrix = context.target_orientation_is_matrix
    opt_matrix = context.optimized_matrix
    opt_scalp_coords = context.optimized_scalp_coords
    results = context.results
    af_cst_calibration = context.af_cst_calibration
    af_target_optimized = context.af_target_optimized
    intensity_rmt = context.intensity_rmt
    biological_threshold = context.biological_threshold
    cst_align = context.cst_align
    tgt_align = context.target_align
    cst_depth = context.cst_depth
    tgt_depth = context.target_depth
    optimization_gain = context.optimization_gain
    ratio_at_m1 = context.ratio_at_m1
    intensity_from_m1_position = context.intensity_from_m1_position
    cst_align_corrected = context.cst_align_corrected
    tgt_align_corrected = context.target_align_corrected
    m1_result = context.m1_result
    tgt_result = context.target_result
    clamped_est_intensity = results["intensity_est_clamped"]
    intensity_flag_w = results["intensity_est_flag"]
    sei_weighted = results["sei_weighted"]
    sei_unweighted = results["sei_unweighted"]
    multiplier_weighted = results["multiplier_weighted"]
    multiplier_unweighted = results["multiplier_unweighted"]

    # =========================================================================
    # Step 5: Save Summary
    # =========================================================================
    if ui:
        ui.update_step(5, "complete")
        ui.update_step(6, "running")
        ui.update_step_detail("Writing final result summary...")

    log.highlight("--- Step 5: Saving Results ---")

    target_act_len = config.options.activation_length_mm

    # Format optimized matrix for display
    m1_matrix_str = str(cal_orientation).replace("\n", "") if cal_orientation is not None else "N/A"
    tgt_matrix_str = (
        str(opt_matrix.tolist()).replace("\n", "")
        if opt_matrix is not None
        else str(tgt_orientation).replace("\n", "") if is_tgt_matrix else "N/A"
    )
    tgt_scalp_str = (
        f"[{opt_scalp_coords[0]:.2f}, {opt_scalp_coords[1]:.2f}, {opt_scalp_coords[2]:.2f}]"
        if opt_scalp_coords is not None
        else "N/A"
    )

    summary_lines = io.build_estimation_summary_lines(
        subject_id=config.subject.id,
        timestamp_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        out_dir=out_dir,
        num_workers=NUM_ESTIMATION_WORKERS,
        t1w_path=config.subject.t1w_path,
        cst_bundle_path=config.calibration.bundle_path,
        target_bundle_path=config.target.bundle_path,
        spatial_mode=results["mode"],
        weight_source=format_weight_sources(
            results["weight_source_cst"],
            results["weight_source_target"],
        ),
        roi_size_mm=config.options.roi_size_mm,
        activation_length_mm=target_act_len,
        calibration_label=config.calibration.label,
        measured_rmt_mso=config.calibration.measured_rmt_mso,
        m1_matrix_str=m1_matrix_str,
        af_cst_w=af_cst_calibration,
        af_cst_u=results["cst_unweighted"],
        intensity_rmt=intensity_rmt,
        biological_threshold=biological_threshold,
        target_label=config.target.label,
        target_coords=config.target.coords,
        opt_scalp_str=tgt_scalp_str,
        tgt_matrix_str=tgt_matrix_str,
        af_tgt_w=af_target_optimized,
        af_tgt_u=results["tgt_unweighted"],
        cst_align=cst_align,
        tgt_align=tgt_align,
        cst_depth=cst_depth,
        tgt_depth=tgt_depth,
        optimization_gain=optimization_gain,
        ratio_at_m1=ratio_at_m1,
        intensity_from_m1_position=intensity_from_m1_position,
        intensity_raw_w=results["intensity_est_raw"],
        intensity_raw_u=results["intensity_est_u_raw"],
        intensity_clamped_w=results["intensity_est_clamped"],
        intensity_clamped_u=results["intensity_est_u_clamped"],
        intensity_flag_w=results["intensity_est_flag"],
        intensity_flag_u=results["intensity_est_u_flag"],
        mso_floor_ratio=results["mso_floor_ratio"],
        sei_w=sei_weighted,
        sei_u=sei_unweighted,
        multiplier_w=multiplier_weighted,
        multiplier_u=multiplier_unweighted,
        cst_align_corrected=cst_align_corrected,
        tgt_align_corrected=tgt_align_corrected,
        calibration_pose_qc=m1_result.pose_qc,
        target_pose_qc=tgt_result.pose_qc,
        aggregator_sensitivity=results.get("aggregator_sensitivity"),
    )

    summary_path = out_dir / f"TIDE_Results_{config.target.label}.txt"
    try:
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines))
        io.save_report_json(
            summary_path,
            "estimation_summary",
            data={
                "workflow": "estimation",
                "subject_id": config.subject.id,
                "target_label": config.target.label,
                "output_dir": out_dir,
                "weight_source_cst": results["weight_source_cst"],
                "weight_source_target": results["weight_source_target"],
                "aggregator_sensitivity": results.get("aggregator_sensitivity"),
                "cst_aggregates_weighted": results.get("cst_aggregates_weighted"),
                "cst_aggregates_unweighted": results.get("cst_aggregates_unweighted"),
                "target_aggregates_weighted": results.get("target_aggregates_weighted"),
                "target_aggregates_unweighted": results.get("target_aggregates_unweighted"),
            },
            text_lines=summary_lines,
        )
        log.debug(f"Saved summary: {summary_path}")
    except Exception as e:
        raise WorkflowError(f"Failed to save summary: {e}") from e

    # Print final result
    log.highlight("")
    if intensity_flag_w != "WITHIN_RANGE" or results["intensity_est_u_flag"] != "WITHIN_RANGE":
        log.highlight(
            f"  RESULT: Estimated Target I (Raw)     = {results['intensity_est_raw']:.1f}% (Weighted) | {results['intensity_est_u_raw']:.1f}% (Unweighted)"
        )
        log.highlight(
            f"  RESULT: Estimated Target I (Clamped)  = {clamped_est_intensity:.1f}% (Weighted) | {results['intensity_est_u_clamped']:.1f}% (Unweighted)"
        )
        log.highlight(
            f"  MSO Floor: {config.options.mso_floor_ratio * 100:.0f}% of RMT ({config.calibration.measured_rmt_mso}%)"
        )
    else:
        log.highlight(
            f"  RESULT: Estimated Target I = {clamped_est_intensity:.1f}% (Weighted) | {results['intensity_est_u_clamped']:.1f}% (Unweighted)"
        )
    log.highlight(f"  (Input RMT: {config.calibration.measured_rmt_mso}%)")
    log.highlight(
        f"  SEI: {sei_weighted:.4f} (Weighted) | {sei_unweighted:.4f} (Unweighted)  [AF_target/AF_CST; 1.0 = same as M1]"
    )
    log.highlight(
        f"  Multiplier (M_CST/M_target): {multiplier_weighted:.4f} (Weighted) | "
        f"{multiplier_unweighted:.4f} (Unweighted)  [I_raw = RMT x multiplier]"
    )
    log.highlight("")

    log.highlight("Workflow completed successfully.")
    log.highlight("Output files:")
    log.highlight(f"  -> Results File: {Path(summary_path).resolve()}")
    log.highlight(f"  -> HTML Report: {Path(summary_path).with_suffix('.html').resolve()}")
    log.highlight(f"  -> Simulations: {Path(out_dir).resolve()}")
    log.highlight(f"  -> Visualizations: {Path(viz_out).resolve()}")

    if ui:
        ui.update_step(6, "complete")

        # Prepare results for summary display
        ui_results = [
            {
                "label": config.target.label,
                "weighted_mso": clamped_est_intensity,
                "unweighted_mso": results["intensity_est_u_clamped"],
                "weighted_mso_raw": results["intensity_est_raw"],
                "unweighted_mso_raw": results["intensity_est_u_raw"],
                "weighted_flag": intensity_flag_w,
                "unweighted_flag": results["intensity_est_u_flag"],
                "sei_weighted": sei_weighted,
                "sei_unweighted": sei_unweighted,
                "multiplier_weighted": multiplier_weighted,
                "multiplier_unweighted": multiplier_unweighted,
                "success": True,
            }
        ]

        output_files = [
            ("Results File", summary_path),
            ("HTML Report", summary_path.with_suffix(".html")),
            ("Simulations", out_dir),
            ("Visualizations", viz_out),
        ]

        elapsed_time = time.time() - start_time
        ui.render_final_summary(ui_results, elapsed_time, output_files)


def _generate_estimation_visualizations(
    config: SimNIBSConfig,
    ui: Any,
    mesh_m1: Path,
    mesh_tgt: Path,
    new_sl_cst: List[np.ndarray],
    af_cst: List[np.ndarray],
    new_sl_tgt: List[np.ndarray],
    af_tgt: List[np.ndarray],
    viz_out: Path,
) -> None:
    # =========================================================================
    # Step 4: Generate 3D Visualizations
    # =========================================================================
    if ui:
        ui.update_step(4, "complete")

    if config.options.generate_3d_visualization and PYVISTA_AVAILABLE:
        if ui:
            ui.update_step(5, "running")
            ui.update_step_detail("Generating 3D visualizations...")

        log.highlight("--- Step 4: Generating 3D Visualizations ---")

        # Get scalp point for depth analysis
        try:
            scalp_point = geometry.project_target_to_scalp(mesh_tgt, np.array(config.target.coords))
        except Exception:
            scalp_point = None

        # Configure visualization (AF scalars are signed; colour scale uses
        # magnitude so the bar spans peak activation strength).
        viz_config = VisualizationConfig(
            efield_vmax=80.0,
            af_vmax=(float(np.percentile(np.abs(np.concatenate(af_tgt)), 99)) if af_tgt else 100.0),
            dpi=config.options.visualization_dpi,
        )

        # Generate CST visualization
        log.debug("Generating CST visualization...")
        try:
            cst_viz_outputs = generate_bundle_visualization(
                mesh_path=mesh_m1,
                streamlines=new_sl_cst,
                af_values=af_cst,
                roi_center=np.array(config.calibration.coords),
                roi_radius=config.options.roi_size_mm,
                output_dir=viz_out,
                prefix="CST_M1",
                config=viz_config,
                scalp_point=scalp_point,
            )
            log.debug(f"CST visualizations: {len(cst_viz_outputs)} files")
        except Exception as e:
            log.warning(f"CST visualization failed: {e}")

        # Generate Target visualization
        log.debug("Generating target visualization...")
        try:
            tgt_viz_outputs = generate_bundle_visualization(
                mesh_path=mesh_tgt,
                streamlines=new_sl_tgt,
                af_values=af_tgt,
                roi_center=np.array(config.target.coords),
                roi_radius=config.options.roi_size_mm,
                output_dir=viz_out,
                prefix=f"{config.target.label}_optimized",
                config=viz_config,
                scalp_point=scalp_point,
            )
            log.debug(f"Target visualizations: {len(tgt_viz_outputs)} files")
        except Exception as e:
            log.warning(f"Target visualization failed: {e}")
    else:
        if not PYVISTA_AVAILABLE:
            log.debug("PyVista not available - skipping 3D visualization")


def run_estimation_workflow(
    config: SimNIBSConfig,
    console_ui: bool = True,
):
    """
    Executes the TIDE Estimation Workflow using the Unified Estimation Module.

    This workflow uses a 2-worker parallel processing system to accelerate
    the optimization and simulation phases. Both the M1/CST and Target
    pipelines run concurrently, providing up to 2x speedup.

    Workflow Steps:
        1. Pre-processing: Medoid calculation (if requested)
        2. Parallel Phase: M1 and Target pipelines run concurrently
           - Each pipeline: Optimization → Simulation → E-field Sampling → AF
        3. Validation: Target in M1 field analysis (requires both results)
        4. Unified Estimation: Final intensity calculation
        5. Visualization: 3D bundle visualizations (if enabled)
        6. Summary: Results report generation

    Args:
    Args:
        config: SimNIBSConfig object with all pipeline parameters.
        console_ui: Enable rich console UI (default: True).
    """
    validate_workflow_config(config, "estimation")

    log.highlight("=== Starting TIDE Estimation Workflow (Parallel) ===")
    log.info(f"Using {NUM_ESTIMATION_WORKERS} parallel workers")

    start_time = time.time()

    # Create multiprocessing context BEFORE UI creation
    ctx = mp.get_context("spawn")

    # Create console UI
    ui = None
    if console_ui and sys.stdout.isatty():
        try:
            from tide.console import create_console_ui

            ui = create_console_ui(
                subject_id=config.subject.id,
                num_workers=NUM_ESTIMATION_WORKERS,
                total_points=2,  # M1 and Target
                current_step=1,
                total_steps=len(ESTIMATION_STEPS),
                workflow_name="Estimation",
                enabled=True,
                mode="sequential",
                mp_context=ctx,
                step_names=ESTIMATION_STEPS,
            )
            ui.start()
        except ImportError:
            log.warning("Console UI not available, falling back to text logging")
            ui = None

    # Log configuration parameters
    log.info("=== Configuration Parameters ===")
    log.info(f"Subject ID: {config.subject.id}")
    log.info(f"Calibration site: {config.calibration.label}")
    log.info(f"Target site: {config.target.label}")
    log.info(f"Coil model: {config.coil.coil_model}")
    log.info(f"Coil distance: {config.coil.coil_distance_mm} mm")
    log.info(f"dI/dt max: {config.coil.device_didt_max / 1e6:.2f} A/µs")
    log.info(f"Measured RMT (MSO): {config.calibration.measured_rmt_mso}%")
    log.info(f"ROI size: {config.options.roi_size_mm} mm")
    log.info(f"Activation length: {config.options.activation_length_mm} mm")
    log.info(f"Field mode: {config.options.field_mode}")
    log.info(f"ADM optimization: {config.options.adm_optimization}")
    log.info(f"Optimization search radius: {config.options.opt_search_radius} mm")
    log.info(f"Optimization spatial resolution: {config.options.opt_spatial_resolution} mm")
    log.info(f"Optimization angle resolution: {config.options.opt_angle_resolution}°")
    log.info(f"Optimization search angle: {config.options.opt_search_angle}°")
    log.info(f"MSO floor ratio: {config.options.mso_floor_ratio}")
    log.info(f"MSO ceiling ratio: {config.options.mso_ceiling_ratio}")
    log.info("=" * 50)

    try:
        preparation = _prepare_estimation(config, ui)
        m1_result, tgt_result = _execute_estimation_tasks(ctx, ui, preparation)
        analysis = _analyze_estimation(config, ui, preparation, m1_result, tgt_result)

        _generate_estimation_visualizations(
            config,
            ui,
            analysis.mesh_m1,
            analysis.mesh_target,
            analysis.cst_streamlines,
            analysis.cst_af_values,
            analysis.target_streamlines,
            analysis.target_af_values,
            preparation.viz_out,
        )
        _write_estimation_summary(
            EstimationSummaryContext(
                config=config,
                ui=ui,
                start_time=start_time,
                out_dir=preparation.out_dir,
                viz_out=preparation.viz_out,
                calibration_orientation=analysis.calibration_orientation,
                target_orientation=preparation.target_orientation,
                target_orientation_is_matrix=preparation.target_orientation_is_matrix,
                optimized_matrix=analysis.optimized_matrix,
                optimized_scalp_coords=analysis.optimized_scalp_coords,
                results=analysis.results,
                af_cst_calibration=analysis.af_cst_calibration,
                af_target_optimized=analysis.af_target_optimized,
                intensity_rmt=analysis.intensity_rmt,
                biological_threshold=analysis.biological_threshold,
                cst_align=analysis.cst_align,
                target_align=analysis.target_align,
                cst_depth=analysis.cst_depth,
                target_depth=analysis.target_depth,
                optimization_gain=analysis.optimization_gain,
                ratio_at_m1=analysis.ratio_at_m1,
                intensity_from_m1_position=analysis.intensity_from_m1_position,
                cst_align_corrected=analysis.cst_align_corrected,
                target_align_corrected=analysis.target_align_corrected,
                m1_result=m1_result,
                target_result=tgt_result,
            )
        )
    except Exception:
        if ui:
            ui.stop()
        raise
