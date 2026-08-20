from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from tide.core import io
from tide.core.physics import AGGREGATOR_KEYS
from tide.interfaces.unified_estimation import build_aggregator_sensitivity, format_weight_sources
from tide.utils.config import SimNIBSConfig, save_grid_point_config
from tide.workflows._shared import calculate_target_in_field_metric

if TYPE_CHECKING:
    from tide.workflows.grid_search import GridPointResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridReportingContext:
    config: SimNIBSConfig
    out_dir: Path
    sims_dir: Path
    results_csv: Path
    fixed_scalp_coords: Sequence[float]
    grid_orientation_ref: Any
    calibration_orientation: Any
    target_streamlines_full: List[np.ndarray]
    target_vectors_in_m1: List[np.ndarray]
    cst_result: Any
    af_cst_calibration: float
    cst_align: float
    cst_align_corrected: float
    cst_depth: float
    intensity_rmt: float
    biological_threshold: float
    m1_matrix_str: str
    spatial_mode: str
    num_workers: int
    calibration_pose_qc: Optional[Dict[str, object]]
    start_time: float
    worker_memory_model: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class GridSummaryResult:
    summary_path: Path
    elapsed_time: float
    weighted_statistics: Dict[str, float]
    unweighted_statistics: Dict[str, float]
    weighted_raw_statistics: Dict[str, float]
    unweighted_raw_statistics: Dict[str, float]
    weighted_multiplier_statistics: Dict[str, float]
    unweighted_multiplier_statistics: Dict[str, float]
    status_counts: Dict[str, Any]


def _aggregator_csv_columns() -> List[str]:
    """Additive per-aggregator raw-intensity column names, appended to the grid CSV."""
    columns: List[str] = []
    for key in AGGREGATOR_KEYS:
        columns.append(f"intensity_raw_{key}_unweighted")
        columns.append(f"intensity_raw_{key}_weighted")
    return columns


def initialize_grid_results_csv(results_csv: Path) -> None:
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "grid_point_labels",
                "grid_point_coords",
                "fixed_scalp_start_coords",
                "optimized_scalp_point_coords",
                "matrix4x4",
                "measured_m1_mso",
                "unweighted_mso_raw",
                "weighted_mso_raw",
                "unweighted_mso_clamped",
                "weighted_mso_clamped",
                "unweighted_mso_flag",
                "weighted_mso_flag",
                "sei_weighted",
                "sei_unweighted",
                "sei_rank_pct",
                "multiplier_weighted",
                "multiplier_unweighted",
            ]
            + _aggregator_csv_columns()
        )


def _write_point_summary_txt(
    *,
    config: SimNIBSConfig,
    point_dir: Path,
    result: "GridPointResult",
    tgt_streamlines_full: List[np.ndarray],
    e_vecs_list_tgt_in_m1_full: List[np.ndarray],
    cst_res: Any,
    af_cst_calibration: float,
    cst_align: float,
    cst_align_corrected: float,
    cst_depth: float,
    intensity_rmt: float,
    biological_threshold: float,
    m1_matrix_str: str,
    spatial_mode: str,
    num_workers: int,
    out_dir: Path,
    calibration_pose_qc: Optional[Dict[str, object]] = None,
    aggregator_sensitivity: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Write per-point TIDE_Results_<target>.txt mirroring the estimation workflow."""
    cortex_coord = result.cortex_coord
    target_weight_source = getattr(result, "target_weight_source", None) or (
        f"External ({config.subject.weights_target_path.name})"
        if config.subject.weights_target_path is not None
        else "Uniform"
    )

    af_target_m1_calibration = calculate_target_in_field_metric(
        tgt_streamlines_full,
        e_vecs_list_tgt_in_m1_full,
        roi_center=cortex_coord,
        roi_size_mm=config.options.roi_size_mm,
        activation_length_mm=config.options.activation_length_mm,
        max_angular_deviation_deg=config.options.max_angular_deviation_deg,
    )

    af_target_optimized = result.target_metric_weighted
    optimization_gain = (
        af_target_optimized / af_target_m1_calibration if af_target_m1_calibration > 0 else 0.0
    )
    ratio_at_m1 = af_target_m1_calibration / af_cst_calibration if af_cst_calibration > 0 else 0.0
    intensity_from_m1_position = (
        config.calibration.measured_rmt_mso * (af_cst_calibration / af_target_m1_calibration)
        if af_target_m1_calibration > 0
        else 0.0
    )

    tgt_matrix_str = (
        str(result.opt_matrix).replace("\n", "") if result.opt_matrix is not None else "N/A"
    )
    if result.opt_scalp_coords is not None:
        tgt_scalp_str = (
            f"[{result.opt_scalp_coords[0]:.2f}, "
            f"{result.opt_scalp_coords[1]:.2f}, "
            f"{result.opt_scalp_coords[2]:.2f}]"
        )
    else:
        tgt_scalp_str = "N/A"

    summary_lines = io.build_estimation_summary_lines(
        subject_id=config.subject.id,
        timestamp_str=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        out_dir=out_dir,
        num_workers=num_workers,
        t1w_path=config.subject.t1w_path,
        cst_bundle_path=config.calibration.bundle_path,
        target_bundle_path=config.target.bundle_path,
        spatial_mode=spatial_mode,
        weight_source=format_weight_sources(
            cst_res.weight_source,
            target_weight_source,
        ),
        roi_size_mm=config.options.roi_size_mm,
        activation_length_mm=config.options.activation_length_mm,
        calibration_label=config.calibration.label,
        measured_rmt_mso=config.calibration.measured_rmt_mso,
        m1_matrix_str=m1_matrix_str,
        af_cst_w=af_cst_calibration,
        af_cst_u=cst_res.metric_unweighted,
        intensity_rmt=intensity_rmt,
        biological_threshold=biological_threshold,
        target_label=config.target.label,
        target_coords=cortex_coord,
        opt_scalp_str=tgt_scalp_str,
        tgt_matrix_str=tgt_matrix_str,
        af_tgt_w=af_target_optimized,
        af_tgt_u=result.target_metric_unweighted,
        cst_align=cst_align,
        tgt_align=result.tgt_align,
        cst_depth=cst_depth,
        tgt_depth=result.tgt_depth,
        optimization_gain=optimization_gain,
        ratio_at_m1=ratio_at_m1,
        intensity_from_m1_position=intensity_from_m1_position,
        intensity_raw_w=result.weighted_mso_raw,
        intensity_raw_u=result.unweighted_mso_raw,
        intensity_clamped_w=result.weighted_mso,
        intensity_clamped_u=result.unweighted_mso,
        intensity_flag_w=result.weighted_mso_flag,
        intensity_flag_u=result.unweighted_mso_flag,
        mso_floor_ratio=config.options.mso_floor_ratio,
        sei_w=result.sei_weighted,
        sei_u=result.sei_unweighted,
        multiplier_w=result.multiplier_weighted,
        multiplier_u=result.multiplier_unweighted,
        cst_align_corrected=cst_align_corrected,
        tgt_align_corrected=result.tgt_align_corrected,
        calibration_pose_qc=calibration_pose_qc,
        target_pose_qc=result.pose_qc,
        aggregator_sensitivity=aggregator_sensitivity,
    )

    summary_path = point_dir / f"TIDE_Results_{config.target.label}.txt"
    try:
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines))
        io.save_report_json(
            summary_path,
            "grid_point_estimation_summary",
            data={
                "workflow": "grid",
                "point_label": result.point_label,
                "subject_id": config.subject.id,
                "target_label": config.target.label,
                "output_dir": point_dir,
                "weight_source_cst": cst_res.weight_source,
                "weight_source_target": target_weight_source,
                "aggregator_sensitivity": aggregator_sensitivity,
                "cst_aggregates_weighted": cst_res.aggregates_weighted,
                "cst_aggregates_unweighted": cst_res.aggregates_unweighted,
                "target_aggregates_weighted": result.target_aggregates_weighted,
                "target_aggregates_unweighted": result.target_aggregates_unweighted,
            },
            text_lines=summary_lines,
        )
        log.debug(f"Saved per-point summary: {summary_path}")
    except Exception as e:
        log.error(f"Failed to save per-point summary {summary_path}: {e}")
        raise


def _calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical summary for a list of values.

    Computes:
    - Mean
    - Median
    - Standard Deviation
    - Mean without outliers (exceeding 2 std dev)

    Args:
        values: List of numerical values.

    Returns:
        Dictionary with statistical metrics.
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "mean_no_outliers": 0.0,
            "outlier_count": 0,
        }

    arr = np.array(values)
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    std_val = np.std(arr)

    # Outlier detection (outside +/- 2 std dev)
    lower_bound = mean_val - 2 * std_val
    upper_bound = mean_val + 2 * std_val

    # Filter valid points (inclusive)
    valid_mask = (arr >= lower_bound) & (arr <= upper_bound)
    valid_points = arr[valid_mask]

    mean_no_outliers = np.mean(valid_points) if len(valid_points) > 0 else mean_val
    outlier_count = len(arr) - len(valid_points)

    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "std": float(std_val),
        "mean_no_outliers": float(mean_no_outliers),
        "outlier_count": outlier_count,
    }


def _valid_result_values(
    results: Sequence[GridPointResult],
    value_field: str,
    flag_field: str,
    exclude_zero: bool = False,
) -> List[float]:
    values = []
    for result in results:
        value = getattr(result, value_field, None)
        if (
            result.success
            and getattr(result, flag_field, "ESTIMATION_FAILED") != "ESTIMATION_FAILED"
            and value != 999.9
            and value is not None
            and np.isfinite(value)
            and (not exclude_zero or value != 0.0)
        ):
            values.append(float(value))
    return values


def _grid_status_counts(results: Sequence[GridPointResult]) -> Dict[str, Any]:
    counts: Dict[str, Any] = {
        "total_points": len(results),
        "processing_failed": sum(not result.success for result in results),
    }
    for prefix in ("weighted", "unweighted"):
        flag_field = f"{prefix}_mso_flag"
        raw_field = f"{prefix}_mso_raw"
        clamped_field = f"{prefix}_mso"
        status = {
            "included": 0,
            "within_range": 0,
            "clamped_low": 0,
            "clamped_high": 0,
            "estimation_failed": 0,
        }
        for result in results:
            if not result.success:
                continue
            flag = getattr(result, flag_field, "ESTIMATION_FAILED")
            raw_value = getattr(result, raw_field, None)
            clamped_value = getattr(result, clamped_field, None)
            finite = (
                raw_value is not None
                and clamped_value is not None
                and np.isfinite(raw_value)
                and np.isfinite(clamped_value)
            )
            if flag == "ESTIMATION_FAILED" or not finite:
                status["estimation_failed"] += 1
                continue
            status["included"] += 1
            status[flag.lower()] = status.get(flag.lower(), 0) + 1
        counts[prefix] = status
    return counts


def write_grid_results(
    results: Sequence[GridPointResult],
    context: GridReportingContext,
) -> List[Dict[str, Any]]:
    config = context.config
    out_dir = context.out_dir
    sims_dir = context.sims_dir
    results_csv = context.results_csv
    fixed_scalp_coords = context.fixed_scalp_coords
    grid_orientation_ref = context.grid_orientation_ref
    cal_orientation = context.calibration_orientation
    tgt_streamlines_full = context.target_streamlines_full
    e_vecs_list_tgt_in_m1_full = context.target_vectors_in_m1
    cst_res = context.cst_result
    af_cst_calibration = context.af_cst_calibration
    cst_align = context.cst_align
    cst_align_corrected = context.cst_align_corrected
    cst_depth = context.cst_depth
    intensity_rmt = context.intensity_rmt
    biological_threshold = context.biological_threshold
    m1_matrix_str = context.m1_matrix_str
    spatial_mode = context.spatial_mode
    actual_workers = context.num_workers
    calibration_pose_qc = context.calibration_pose_qc
    final_grid_results = []

    # Pre-compute SEI percentile ranks across successful grid points
    _sei_vals = np.array([r.sei_weighted for r in results if r.success and r.sei_weighted > 0])

    def _sei_rank_pct(sei_val: float) -> float:
        """Percentile rank of sei_val within the distribution of successful SEI values (0–100)."""
        if len(_sei_vals) == 0 or sei_val <= 0:
            return 0.0
        return float(np.sum(_sei_vals <= sei_val) / len(_sei_vals) * 100.0)

    for result in results:
        if result.success:
            # Write to CSV
            matrix_str = str(result.opt_matrix).replace("\n", "")
            cortex_str = (
                f"[{result.cortex_coord[0]:.2f}, "
                f"{result.cortex_coord[1]:.2f}, "
                f"{result.cortex_coord[2]:.2f}]"
            )
            scalp_opt_str = (
                f"[{result.opt_scalp_coords[0]:.2f}, "
                f"{result.opt_scalp_coords[1]:.2f}, "
                f"{result.opt_scalp_coords[2]:.2f}]"
            )

            sei_rank = _sei_rank_pct(result.sei_weighted)
            aggregator_sensitivity = build_aggregator_sensitivity(
                cst_weighted=cst_res.aggregates_weighted,
                cst_unweighted=cst_res.aggregates_unweighted,
                target_weighted=result.target_aggregates_weighted,
                target_unweighted=result.target_aggregates_unweighted,
                rmt=config.calibration.measured_rmt_mso,
            )
            aggregator_cells: List[str] = []
            for key in AGGREGATOR_KEYS:
                row = aggregator_sensitivity[key]
                aggregator_cells.append(f"{row['intensity_raw_unweighted']:.2f}")
                aggregator_cells.append(f"{row['intensity_raw_weighted']:.2f}")

            with open(results_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        result.point_label,
                        cortex_str,
                        str(fixed_scalp_coords),
                        scalp_opt_str,
                        matrix_str,
                        config.calibration.measured_rmt_mso,
                        f"{result.unweighted_mso_raw:.2f}",
                        f"{result.weighted_mso_raw:.2f}",
                        f"{result.unweighted_mso:.2f}",
                        f"{result.weighted_mso:.2f}",
                        result.unweighted_mso_flag,
                        result.weighted_mso_flag,
                        f"{result.sei_weighted:.4f}",
                        f"{result.sei_unweighted:.4f}",
                        f"{sei_rank:.1f}",
                        f"{result.multiplier_weighted:.4f}",
                        f"{result.multiplier_unweighted:.4f}",
                    ]
                    + aggregator_cells
                )

            # Save grid point configuration
            point_dir = sims_dir / result.point_label
            save_grid_point_config(
                config=config,
                output_dir=point_dir,
                point_label=result.point_label,
                cortex_coords=result.cortex_coord,
                scalp_coords=result.opt_scalp_coords,
                orientation_matrix=result.opt_matrix,
                fixed_scalp_coords=(
                    fixed_scalp_coords.tolist()
                    if hasattr(fixed_scalp_coords, "tolist")
                    else list(fixed_scalp_coords)
                ),
                grid_orientation_ref=grid_orientation_ref,
                calibration_orientation=cal_orientation,
            )

            _write_point_summary_txt(
                config=config,
                point_dir=point_dir,
                result=result,
                tgt_streamlines_full=tgt_streamlines_full,
                e_vecs_list_tgt_in_m1_full=e_vecs_list_tgt_in_m1_full,
                cst_res=cst_res,
                af_cst_calibration=af_cst_calibration,
                cst_align=cst_align,
                cst_align_corrected=cst_align_corrected,
                cst_depth=cst_depth,
                intensity_rmt=intensity_rmt,
                biological_threshold=biological_threshold,
                m1_matrix_str=m1_matrix_str,
                spatial_mode=spatial_mode,
                num_workers=actual_workers,
                out_dir=out_dir,
                calibration_pose_qc=calibration_pose_qc,
                aggregator_sensitivity=aggregator_sensitivity,
            )

            final_grid_results.append(
                {
                    "label": result.point_label,
                    "weighted_mso": result.weighted_mso,
                    "unweighted_mso": result.unweighted_mso,
                    "weighted_mso_raw": result.weighted_mso_raw,
                    "unweighted_mso_raw": result.unweighted_mso_raw,
                    "weighted_mso_flag": result.weighted_mso_flag,
                    "unweighted_mso_flag": result.unweighted_mso_flag,
                    "sei_weighted": result.sei_weighted,
                    "sei_unweighted": result.sei_unweighted,
                    "sei_rank_pct": sei_rank,
                    "multiplier_weighted": result.multiplier_weighted,
                    "multiplier_unweighted": result.multiplier_unweighted,
                    "target_pose_qc": result.pose_qc,
                    "target_align": result.tgt_align,
                    "target_align_corrected": result.tgt_align_corrected,
                    "target_depth": result.tgt_depth,
                }
            )
        else:
            # Record failed point with N/A values
            final_grid_results.append(
                {
                    "label": result.point_label,
                    "weighted_mso": 999.9,
                    "unweighted_mso": 999.9,
                    "weighted_mso_raw": 999.9,
                    "unweighted_mso_raw": 999.9,
                    "weighted_mso_flag": "N/A",
                    "unweighted_mso_flag": "N/A",
                    "sei_weighted": None,
                    "sei_unweighted": None,
                    "sei_rank_pct": None,
                    "multiplier_weighted": None,
                    "multiplier_unweighted": None,
                    "target_pose_qc": None,
                    "target_align": None,
                    "target_align_corrected": None,
                    "target_depth": None,
                }
            )

    return final_grid_results


def write_grid_summary(
    results: Sequence[GridPointResult],
    final_grid_results: Sequence[Dict[str, Any]],
    context: GridReportingContext,
) -> GridSummaryResult:
    config = context.config
    out_dir = context.out_dir
    results_csv = context.results_csv
    start_time = context.start_time
    actual_workers = context.num_workers
    spatial_mode = context.spatial_mode
    cal_orientation = context.calibration_orientation
    calibration_pose_qc = context.calibration_pose_qc
    af_cst_calibration = context.af_cst_calibration
    intensity_rmt = context.intensity_rmt
    biological_threshold = context.biological_threshold
    cst_align = context.cst_align
    cst_align_corrected = context.cst_align_corrected
    cst_depth = context.cst_depth
    grid_orientation_ref = context.grid_orientation_ref
    elapsed_time = time.time() - start_time
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = elapsed_time % 60
    worker_memory_model = context.worker_memory_model or {}
    worker_memory_lines = []
    if worker_memory_model:
        available_memory = worker_memory_model.get("available_memory_gb")
        available_memory_text = (
            f"{available_memory:.1f} GB" if available_memory is not None else "Unknown"
        )
        memory_reserve_gb = float(worker_memory_model.get("memory_reserve_gb", 0.0))
        memory_per_worker_gb = float(worker_memory_model.get("memory_per_worker_gb", 0.0))
        worker_memory_lines = [
            "",
            "--- Parallel Resource Plan ---",
            f"  Solver: {worker_memory_model.get('solver', 'PARDISO')}",
            f"  Available Memory: {available_memory_text}",
            f"  Parent/OS Reserve: {memory_reserve_gb:.1f} GB",
            f"  Estimated Memory per Worker: {memory_per_worker_gb:.1f} GB",
            f"  Memory Worker Limit: {worker_memory_model.get('memory_worker_limit', 1)}",
            f"  Forced Worker Override: {worker_memory_model.get('forced', False)}",
        ]

    # Calculate Statistics
    valid_weighted = _valid_result_values(
        results,
        "weighted_mso",
        "weighted_mso_flag",
    )
    valid_unweighted = _valid_result_values(
        results,
        "unweighted_mso",
        "unweighted_mso_flag",
    )
    valid_raw_weighted = _valid_result_values(
        results,
        "weighted_mso_raw",
        "weighted_mso_flag",
    )
    valid_raw_unweighted = _valid_result_values(
        results,
        "unweighted_mso_raw",
        "unweighted_mso_flag",
    )

    stats_weighted = _calculate_statistics(valid_weighted)
    stats_unweighted = _calculate_statistics(valid_unweighted)
    stats_raw_weighted = _calculate_statistics(valid_raw_weighted)
    stats_raw_unweighted = _calculate_statistics(valid_raw_unweighted)
    status_counts = _grid_status_counts(results)

    # Multiplier statistics (M_CST/M_target). I_raw = RMT * multiplier, so
    # surfacing mean/median multipliers next to intensity lets the user rescale the
    # predicted dose to any RMT without rerunning the simulation.
    valid_mult_weighted = _valid_result_values(
        results,
        "multiplier_weighted",
        "weighted_mso_flag",
        exclude_zero=True,
    )
    valid_mult_unweighted = _valid_result_values(
        results,
        "multiplier_unweighted",
        "unweighted_mso_flag",
        exclude_zero=True,
    )
    stats_mult_weighted = _calculate_statistics(valid_mult_weighted)
    stats_mult_unweighted = _calculate_statistics(valid_mult_unweighted)

    summary_lines = [
        "===========================================",
        "--- TIDE Grid Search Pipeline Summary ---",
        "===========================================",
        "",
        f"  Subject: {config.subject.id}",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Output Folder: {out_dir}",
        f"  Elapsed Time: {elapsed_min}m {elapsed_sec:.1f}s",
        f"  Workers Used: {actual_workers}",
        *worker_memory_lines,
        "",
        "--- Configuration ---",
        f"  Input T1w: {config.subject.t1w_path}",
        f"  Target Tractogram: {config.target.bundle_path}",
        f"  Spatial Mode: {spatial_mode}",
        f"  ROI Size: {config.options.roi_size_mm} mm",
        f"  Visualization Artifacts: {'Enabled' if config.options.generate_visualizations else 'Disabled'}",
        f"  Interactive 3D: {'Enabled' if config.options.generate_3d_visualization else 'Disabled'}",
        "",
        "--- Optimization Settings ---",
        f"  Search Radius: {config.options.opt_search_radius} mm",
        f"  Spatial Resolution: {config.options.opt_spatial_resolution} mm",
        f"  Angle Resolution: {config.options.opt_angle_resolution} deg",
        f"  Search Angle: {config.options.opt_search_angle} deg (±{config.options.opt_search_angle / 2} deg)",
        f"  ADM Optimization: {config.options.adm_optimization}",
        "",
        "--- Intensity Floor ---",
        f"  I Floor Ratio: {config.options.mso_floor_ratio}",
        f"  I Floor Value: {config.calibration.measured_rmt_mso * config.options.mso_floor_ratio:.1f} % max output",
        "",
        f"--- M1 Calibration ({config.calibration.label}) ---",
        f"  Measured RMT: {config.calibration.measured_rmt_mso} % max output",
        f"  Optimized Matrix: {cal_orientation}",
        f"  M1 Coil Pose QC: {io.format_pose_qc(calibration_pose_qc)}",
        f"  CST Efficiency (Weighted): {af_cst_calibration:.4f} V/m^2",
        f"  RMT Intensity (dI/dt): {intensity_rmt / 1e6:.2f} A/us",
        f"  Biological Threshold: {biological_threshold:.2f} V/m^2",
        f"  CST Alignment: {cst_align:.4f}",
        f"  CST Alignment Corrected: {cst_align_corrected:.4f}",
        f"  CST Depth: {cst_depth:.1f} mm",
        "",
        "--- Grid Settings ---",
        f"  Grid Center (Cortex): {config.grid.coords}",
        f"  Grid Orientation Ref (pos_ydir): {grid_orientation_ref}",
        f"  Search Radius: {config.grid.search_radius_mm} mm",
        f"  Step Size: {config.grid.step_size_mm} mm",
        "",
        "===========================================",
        "--- Statistical Summary ---",
        "===========================================",
        "Metric                 | Unweighted I         | Weighted I",
        "-----------------------|----------------------|---------------------",
        f"Mean                   | {stats_unweighted['mean']:<20.2f} | {stats_weighted['mean']:<20.2f}",
        f"Median                 | {stats_unweighted['median']:<20.2f} | {stats_weighted['median']:<20.2f}",
        f"Std Dev                | {stats_unweighted['std']:<20.2f} | {stats_weighted['std']:<20.2f}",
        f"Mean (w/o outliers) *  | {stats_unweighted['mean_no_outliers']:<20.2f} | {stats_weighted['mean_no_outliers']:<20.2f}",
        f"Outliers (>2 SD)       | {stats_unweighted['outlier_count']:<20} | {stats_weighted['outlier_count']:<20}",
        "",
        "* Mean calculated excluding values outside [mean +/- 2 * std]",
        "",
        "===========================================",
        "--- Raw Statistical Summary ---",
        "===========================================",
        "Metric                 | Unweighted Raw I     | Weighted Raw I",
        "-----------------------|----------------------|---------------------",
        f"Mean                   | {stats_raw_unweighted['mean']:<20.2f} | {stats_raw_weighted['mean']:<20.2f}",
        f"Median                 | {stats_raw_unweighted['median']:<20.2f} | {stats_raw_weighted['median']:<20.2f}",
        f"Std Dev                | {stats_raw_unweighted['std']:<20.2f} | {stats_raw_weighted['std']:<20.2f}",
        f"Mean (w/o outliers) *  | {stats_raw_unweighted['mean_no_outliers']:<20.2f} | {stats_raw_weighted['mean_no_outliers']:<20.2f}",
        f"Outliers (>2 SD)       | {stats_raw_unweighted['outlier_count']:<20} | {stats_raw_weighted['outlier_count']:<20}",
        "",
        "===========================================",
        "--- Grid Result Status Counts ---",
        "===========================================",
        f"Total Points: {status_counts['total_points']}",
        f"Processing Failures: {status_counts['processing_failed']}",
        "Status                  | Unweighted           | Weighted",
        "------------------------|----------------------|---------------------",
        f"Included                | {status_counts['unweighted']['included']:<20} | {status_counts['weighted']['included']:<20}",
        f"Within Range            | {status_counts['unweighted']['within_range']:<20} | {status_counts['weighted']['within_range']:<20}",
        f"Clamped Low             | {status_counts['unweighted']['clamped_low']:<20} | {status_counts['weighted']['clamped_low']:<20}",
        f"Clamped High            | {status_counts['unweighted']['clamped_high']:<20} | {status_counts['weighted']['clamped_high']:<20}",
        f"Estimation Failed       | {status_counts['unweighted']['estimation_failed']:<20} | {status_counts['weighted']['estimation_failed']:<20}",
        "",
        "===========================================",
        "--- Grid Point Results ---",
        "===========================================",
        f"{'Label':<15} | {'Unweighted Raw':<15} | {'Weighted Raw':<14} | {'Unweighted Clamp':<16} | {'Weighted Clamp':<14} | {'U-Flag':<14} | {'W-Flag':<14} | {'SEI (W)':<10} | {'SEI Rank':<10} | {'Mult (W)':<10}",
        "-" * 153,
    ]

    def _format_intensity(value: Any) -> str:
        if value is None or value == 999.9 or not np.isfinite(value):
            return "N/A"
        return f"{value:.1f}"

    for res in final_grid_results:
        w_val = _format_intensity(res["weighted_mso"])
        u_val = _format_intensity(res["unweighted_mso"])
        w_raw = _format_intensity(res["weighted_mso_raw"])
        u_raw = _format_intensity(res["unweighted_mso_raw"])
        w_flag = res.get("weighted_mso_flag", "N/A")
        u_flag = res.get("unweighted_mso_flag", "N/A")
        sei_w_str = f"{res['sei_weighted']:.4f}" if res.get("sei_weighted") is not None else "N/A"
        sei_rank_str = (
            f"{res['sei_rank_pct']:.1f}%" if res.get("sei_rank_pct") is not None else "N/A"
        )
        mult_w_str = (
            f"{res['multiplier_weighted']:.4f}"
            if res.get("multiplier_weighted") is not None
            else "N/A"
        )
        summary_lines.append(
            f"{res['label']:<15} | {u_raw:<15} | {w_raw:<14} | {u_val:<16} | {w_val:<14}"
            f" | {u_flag:<14} | {w_flag:<14} | {sei_w_str:<10} | {sei_rank_str:<10} | {mult_w_str:<10}"
        )

    summary_lines.append("=" * 153)
    summary_lines.extend(
        [
            "",
            "===========================================",
            "--- Grid Point QC ---",
            "===========================================",
            f"{'Label':<15} | {'Target Coil Pose QC':<22} | {'Target Alignment':<16} | {'Target Alignment Corrected':<26} | {'Target Depth':<12}",
            "-" * 102,
        ]
    )

    for res in final_grid_results:
        pose_qc = res.get("target_pose_qc") or {}
        pose_status = str(pose_qc.get("status", "N/A")) if pose_qc else "N/A"
        reasons = pose_qc.get("reasons") or [] if pose_qc else []
        if reasons:
            pose_status = f"{pose_status} ({', '.join(str(reason) for reason in reasons)})"
        target_align = (
            f"{res['target_align']:.4f}" if res.get("target_align") is not None else "N/A"
        )
        target_align_corrected = (
            f"{res['target_align_corrected']:.4f}"
            if res.get("target_align_corrected") is not None
            else "N/A"
        )
        target_depth = (
            f"{res['target_depth']:.1f} mm" if res.get("target_depth") is not None else "N/A"
        )
        summary_lines.append(
            f"{res['label']:<15} | {pose_status:<22} | {target_align:<16} | "
            f"{target_align_corrected:<26} | {target_depth:<12}"
        )

    summary_lines.append("=" * 102)

    try:
        summary_path = out_dir / f"TIDE_Grid_Summary_{config.target.label}.txt"
        with open(summary_path, "w") as f_txt:
            f_txt.write("\n".join(summary_lines))
        io.save_report_json(
            summary_path,
            "grid_summary",
            data={
                "workflow": "grid",
                "subject_id": config.subject.id,
                "target_label": config.target.label,
                "output_dir": out_dir,
                "results_csv": results_csv,
                "num_grid_points": len(final_grid_results),
                "statistics": {
                    "weighted_clamped": stats_weighted,
                    "unweighted_clamped": stats_unweighted,
                    "weighted_raw": stats_raw_weighted,
                    "unweighted_raw": stats_raw_unweighted,
                    "weighted_multiplier": stats_mult_weighted,
                    "unweighted_multiplier": stats_mult_unweighted,
                },
                "status_counts": status_counts,
                "worker_memory_model": context.worker_memory_model,
                "visualization_artifacts": {
                    "generate_visualizations": config.options.generate_visualizations,
                    "generate_3d_visualization": config.options.generate_3d_visualization,
                },
                "weight_source_cst": context.cst_result.weight_source,
                "weight_source_target": (
                    f"External ({config.subject.weights_target_path.name})"
                    if config.subject.weights_target_path is not None
                    else "Uniform"
                ),
            },
            text_lines=summary_lines,
        )
        log.info(f"Summary report saved to: {summary_path}")
    except Exception as e:
        log.error(f"Failed to save summary txt: {e}")
        raise

    log.info(f"Grid Search Complete. Results: {results_csv}")
    log.info(f"Total time: {elapsed_min}m {elapsed_sec:.1f}s")

    return GridSummaryResult(
        summary_path=summary_path,
        elapsed_time=elapsed_time,
        weighted_statistics=stats_weighted,
        unweighted_statistics=stats_unweighted,
        weighted_raw_statistics=stats_raw_weighted,
        unweighted_raw_statistics=stats_raw_unweighted,
        weighted_multiplier_statistics=stats_mult_weighted,
        unweighted_multiplier_statistics=stats_mult_unweighted,
        status_counts=status_counts,
    )
