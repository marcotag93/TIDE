import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram

from tide.core import _reporting
from tide.core.physics import AGGREGATOR_KEYS, AGGREGATOR_LABELS, PRIMARY_AGGREGATOR

log = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def save_report_json(
    txt_path: Path,
    report_type: str,
    data: Optional[Dict[str, Any]] = None,
    text_lines: Optional[List[str]] = None,
) -> Path:
    txt_path = Path(txt_path)
    json_path = txt_path.with_suffix(".json")

    try:
        if text_lines is None:
            text_content = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
            lines = text_content.splitlines()
        else:
            lines = list(text_lines)
            text_content = "\n".join(lines)

        payload = {
            "schema_version": "1.0",
            "report_type": report_type,
            "source_txt": str(txt_path),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "text": {
                "content": text_content,
                "lines": lines,
            },
            "sections": _reporting._parse_report_sections(lines),
        }
        if data is not None:
            payload["data"] = _json_safe(data)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

        save_report_html(
            txt_path,
            report_type,
            data=data,
            text_lines=lines,
            json_path=json_path,
            generated_at=payload["generated_at"],
        )
        log.info(f"Saved JSON report to: {json_path}")
        return json_path
    except Exception as e:
        log.error(f"Failed to save JSON report {json_path}: {e}")
        raise


def save_report_html(
    txt_path: Path,
    report_type: str,
    data: Optional[Dict[str, Any]] = None,
    text_lines: Optional[List[str]] = None,
    json_path: Optional[Path] = None,
    generated_at: Optional[str] = None,
) -> Path:
    txt_path = Path(txt_path)
    html_path = txt_path.with_suffix(".html")

    try:
        if text_lines is None:
            text_content = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
            lines = text_content.splitlines()
        else:
            lines = list(text_lines)

        sections = _reporting._parse_report_sections(lines)
        report_fields = _reporting._collect_report_fields(sections)
        generated = generated_at or datetime.now().isoformat(timespec="seconds")
        safe_data = _json_safe(data or {})
        images = _reporting._discover_report_images(txt_path.parent, limit=32)

        html = _reporting._build_report_html(
            txt_path=txt_path,
            html_path=html_path,
            json_path=json_path,
            report_type=report_type,
            generated_at=generated,
            fields=report_fields,
            data=safe_data,
            sections=sections,
            images=images,
        )
        html_path.write_text(html, encoding="utf-8")
        log.info(f"Saved HTML report to: {html_path}")
        return html_path
    except Exception as e:
        log.error(f"Failed to save HTML report {html_path}: {e}")
        raise


def save_tract_with_data(
    reference_sft: StatefulTractogram,
    new_streamlines: List[np.ndarray],
    output_path: Path,
    scalar_name: str,
    scalar_values: List[np.ndarray],
    segment_lengths: List[np.ndarray],
):
    """
    Saves a tractogram (.trk) using the NEW streamlines (midpoints) and scalar data.
    """
    formatted_scalars = [s.reshape(-1, 1) for s in scalar_values]
    formatted_lengths = [lengths.reshape(-1, 1) for lengths in segment_lengths]

    data_per_point = {
        scalar_name: formatted_scalars,
        "segment_length": formatted_lengths,
    }

    sft_out = StatefulTractogram.from_sft(
        new_streamlines, reference_sft, data_per_point=data_per_point
    )

    save_tractogram(sft_out, str(output_path), bbox_valid_check=False)
    log.info(f"Successfully saved .trk file: {output_path}")


def save_gmsh_pos(points: np.ndarray, scalars: np.ndarray, output_path: Path, view_name: str):
    try:
        with open(output_path, "w") as f:
            f.write(f'View "{view_name}" {{\n')
            for (x, y, z), val in zip(points, scalars):
                f.write(f"  SP({x},{y},{z}){{{val}}};\n")
            f.write("};\n")
        log.info(f"Successfully saved .pos file: {output_path}")
    except IOError as e:
        log.error(f"Failed to save .pos file: {e}")


def save_points_as_nifti(
    points: np.ndarray,
    ref_img_path: Path,
    output_path: Path,
    values: Optional[np.ndarray] = None,
):
    """
    Saves a set of RASMM points as a NIfTI mask using a reference image for affine/header.
    If 'values' is provided, the voxels are filled with these values (float32).
    Otherwise, voxels are set to 1 (uint8 binary mask).
    """
    if nib is None:
        raise RuntimeError("nibabel not installed. Cannot save NIfTI points.")

    if not ref_img_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_img_path}")

    try:
        ref_img = nib.load(str(ref_img_path))
        affine = ref_img.affine
        inv_affine = np.linalg.inv(affine)

        # Ensure points are Nx3
        points_arr = np.atleast_2d(points)
        if points_arr.shape[1] != 3:
            raise ValueError(f"Points array has invalid shape {points_arr.shape}, expected (N, 3)")

        # Convert RASMM points to Voxel Coordinates
        M = inv_affine[:3, :3]
        abc = inv_affine[:3, 3]
        voxel_coords = points_arr @ M.T + abc

        voxel_indices = np.rint(voxel_coords).astype(int)

        # Initialize empty volume
        if values is not None:
            if len(values) != len(points_arr):
                raise ValueError(
                    f"Shape mismatch: {len(values)} values for {len(points_arr)} points."
                )
            data = np.zeros(ref_img.shape, dtype=np.float32)
        else:
            data = np.zeros(ref_img.shape, dtype=np.uint8)

        # Filter out-of-bounds indices
        valid_mask = (
            (voxel_indices[:, 0] >= 0)
            & (voxel_indices[:, 0] < data.shape[0])
            & (voxel_indices[:, 1] >= 0)
            & (voxel_indices[:, 1] < data.shape[1])
            & (voxel_indices[:, 2] >= 0)
            & (voxel_indices[:, 2] < data.shape[2])
        )

        valid_indices = voxel_indices[valid_mask]

        # Set voxels
        if values is not None:
            valid_values = values[valid_mask]
            # Note: If multiple points map to the same voxel, the last one writes.
            data[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = valid_values
        else:
            data[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1

        # Save
        new_img = nib.Nifti1Image(data, affine, ref_img.header)
        nib.save(new_img, str(output_path))
        log.info(f"Saved NIfTI map: {output_path}")

    except Exception as e:
        log.error(f"Failed to save grid points NIfTI: {e}")
        raise


def plot_activation_depth(
    activated_lengths: List[float],
    output_path: Path,
    scalar_name: str,
    threshold_val: float,
    threshold_pct: float,
):
    try:
        plt.figure(figsize=(10, 6))
        max_len = np.max(activated_lengths) if activated_lengths else 0
        bins = min(50, int(max_len) + 1) if max_len > 0 else 1

        plt.hist(activated_lengths, bins=bins, edgecolor="black", color="#007ACC")
        plt.title(f"Activated Length ({scalar_name})", fontsize=14)
        plt.xlabel("Length > Threshold (mm)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        stats_text = (
            f"Total Streamlines: {len(activated_lengths)}\n"
            f"Threshold: {threshold_pct}% ({threshold_val:.4f})"
        )
        plt.text(
            0.95,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        log.info(f"Saved histogram: {output_path}")
    except Exception as e:
        log.warning(f"Failed to plot histogram: {e}")


def write_summary(output_path: Path, info: Dict[str, Any]):
    """Generic summary writer (key-value)."""
    try:
        with open(output_path, "w") as f:
            f.write("--- Analysis Summary ---\n")
            for key, val in info.items():
                f.write(f"{key}: {val}\n")
        save_report_json(output_path, "analysis_summary", data=info)
        log.info(f"Successfully saved summary to: {output_path}")
    except IOError as e:
        log.error(f"Failed to save summary: {e}")
        raise


def _format_pose_qc(pose_qc: Optional[Dict[str, Any]]) -> str:
    if not pose_qc:
        return "N/A"

    status = str(pose_qc.get("status", "N/A"))
    reasons = pose_qc.get("reasons") or []
    parts = [status]
    if reasons:
        parts.append(f"({', '.join(str(reason) for reason in reasons)})")

    details = []
    for key, label in (
        ("scalp_outward_dot", "normal_dot"),
        ("scalp_normal_z", "scalp_normal_z"),
        ("coil_normal_z", "coil_normal_z"),
        ("scalp_z_percentile", "scalp_z_pct"),
        ("nearest_scalp_distance_mm", "nearest_scalp_mm"),
    ):
        val = pose_qc.get(key)
        if val is not None:
            details.append(f"{label}={float(val):.3f}")
    if details:
        parts.append("[" + ", ".join(details) + "]")

    return " ".join(parts)


def format_pose_qc(pose_qc: Optional[Dict[str, Any]]) -> str:
    return _format_pose_qc(pose_qc)


def _format_optional_float(value: Optional[float], precision: int) -> str:
    return "N/A" if value is None else f"{value:.{precision}f}"


def _format_optional_depth(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1f} mm"


def _format_alignment_qc(
    alignment_qc: Optional[Dict[str, Any]],
    label: str = "",
) -> List[str]:
    prefix = f"{label} " if label else ""
    if not alignment_qc:
        return [
            f"{prefix}Alignment: N/A",
            f"{prefix}Alignment Corrected: N/A",
            f"{prefix}Depth: N/A",
        ]

    return [
        f"{prefix}Alignment: {_format_optional_float(alignment_qc.get('alignment'), 4)}",
        (
            f"{prefix}Alignment Corrected: "
            f"{_format_optional_float(alignment_qc.get('alignment_corrected'), 4)}"
        ),
        f"{prefix}Depth: {_format_optional_depth(alignment_qc.get('depth_mm'))}",
    ]


def save_mapping_summary(output_path: Path, info: Dict[str, Any]):
    """
    Saves the E-field mapping summary in the specific requested format.
    """
    try:
        with open(output_path, "w") as f:
            f.write("--- E-Field to Bundle Mapping Summary ---\n")
            f.write(f"Timestamp: {info['Timestamp']}\n")
            f.write(f"Prefix: {info['Prefix']}\n\n")

            f.write("--- INPUTS ---\n")
            f.write(f"Mesh: {info['Mesh']}\n")
            f.write(f"Bundle: {info['Bundle']}\n")
            f.write(f"Anatomy: {info['Anatomy']}\n\n")

            f.write("--- PARAMETERS ---\n")
            f.write(f"Mode (Scalar): {info['Mode']}\n")
            threshold = info["Threshold_Percent"]
            threshold_suffix = "" if str(threshold).upper() == "N/A" else "%"
            f.write(f"Activation Threshold: {threshold}{threshold_suffix}\n\n")

            f.write("--- RESULTS ---\n")
            f.write(f"Total Streamlines Processed: {info['Total_Streamlines']}\n")
            f.write(f"Max AF Value: {info['Max_Value']}\n")
            f.write(f"Min AF Value: {info['Min_Value']}\n\n")

            metrics = info.get("Metrics")
            if metrics:
                f.write("--- ROBUST METRICS ---\n")
                for key, val in metrics.items():
                    f.write(f"{key}: {val}\n")
                f.write("\n")

            qc = info.get("QC")
            if qc:
                f.write("--- QC ---\n")
                f.write(f"Target Coil Pose QC: {_format_pose_qc(qc.get('pose_qc'))}\n")
                for line in _format_alignment_qc(qc.get("alignment_qc"), label="Target"):
                    f.write(f"{line}\n")
                f.write("\n")

            f.write("--- OUTPUT FILES (in outdir) ---\n")
            for desc, fname in info["Output_Files"].items():
                f.write(f"{desc}: {fname}\n")

        save_report_json(output_path, "mapping_summary", data=info)
        log.info(f"Successfully saved summary to: {output_path}")
    except IOError as e:
        log.error(f"Failed to save summary: {e}")
        raise


def save_optimization_result_txt(
    output_path: Path,
    matrix: np.ndarray,
    scalp_coords: np.ndarray,
    setup_info: List[str] = None,
    pose_qc: Optional[Dict[str, Any]] = None,
    alignment_qc: Optional[Dict[str, Any]] = None,
):
    try:
        row_strings = []
        for r in range(3):
            row_strings.append(
                f"[{matrix[r, 0]:.8f}, {matrix[r, 1]:.8f}, {matrix[r, 2]:.8f}, {matrix[r, 3]:.8f}]"
            )
        row_strings.append("[0, 0, 0, 1]")
        single_line_matrix = f"[{', '.join(row_strings)}]"
        rot = matrix[0:3, 0:3]

        content = []
        if setup_info:
            content.extend(setup_info)
            content.append("")
            content.append("=" * 60)
            content.append("--- OPTIMIZATION RESULTS ---")
            content.append("=" * 60)
            content.append("")

        content.append("Optimized Scalp Position (x, y, z):")
        content.append(f"[{scalp_coords[0]:.8f}, {scalp_coords[1]:.8f}, {scalp_coords[2]:.8f}]")
        content.append("")
        content.append("Optimal 3x3 Orientation Matrix (m):")
        content.append(f"[[{rot[0, 0]:.8f}, {rot[0, 1]:.8f}, {rot[0, 2]:.8f}],")
        content.append(f" [{rot[1, 0]:.8f}, {rot[1, 1]:.8f}, {rot[1, 2]:.8f}],")
        content.append(f" [{rot[2, 0]:.8f}, {rot[2, 1]:.8f}, {rot[2, 2]:.8f}]]")
        content.append("")
        content.append("Full 4x4 Transformation Matrix:")
        content.append(np.array2string(matrix, precision=8, suppress_small=True))
        content.append("")
        content.append("--- COIL POSE QC ---")
        content.append(f"Pose QC: {_format_pose_qc(pose_qc)}")
        content.append("")
        content.append("--- ALIGNMENT QC ---")
        content.extend(_format_alignment_qc(alignment_qc))
        content.append("")
        content.append("--- FOR CONFIG FILE (copy/paste) ---")
        content.append(f"orientation: {single_line_matrix}")

        with open(output_path, "w") as f:
            f.write("\n".join(content))

        save_report_json(
            output_path,
            "optimization_result",
            data={
                "optimized_scalp_position": scalp_coords,
                "orientation_matrix_3x3": rot,
                "transformation_matrix_4x4": matrix,
                "config_orientation": single_line_matrix,
                "pose_qc": pose_qc,
                "alignment_qc": alignment_qc,
                "setup_info": setup_info or [],
            },
            text_lines=content,
        )
        log.info(f"Saved optimization results to: {output_path}")

    except Exception as e:
        log.error(f"Failed to write optimization result file: {e}")
        raise


def build_estimation_summary_lines(
    *,
    subject_id: str,
    timestamp_str: str,
    out_dir: Path,
    num_workers: int,
    t1w_path: Path,
    cst_bundle_path: Path,
    target_bundle_path: Path,
    spatial_mode: str,
    weight_source: str,
    roi_size_mm: float,
    activation_length_mm: float,
    calibration_label: str,
    measured_rmt_mso: float,
    m1_matrix_str: str,
    af_cst_w: float,
    af_cst_u: float,
    intensity_rmt: float,
    biological_threshold: float,
    target_label: str,
    target_coords: List[float],
    opt_scalp_str: str,
    tgt_matrix_str: str,
    af_tgt_w: float,
    af_tgt_u: float,
    cst_align: float,
    tgt_align: float,
    cst_depth: float,
    tgt_depth: float,
    optimization_gain: float,
    ratio_at_m1: float,
    intensity_from_m1_position: float,
    intensity_raw_w: float,
    intensity_raw_u: float,
    intensity_clamped_w: float,
    intensity_clamped_u: float,
    intensity_flag_w: str,
    intensity_flag_u: str,
    mso_floor_ratio: float,
    sei_w: float,
    sei_u: float,
    multiplier_w: float,
    multiplier_u: float,
    cst_align_corrected: Optional[float] = None,
    tgt_align_corrected: Optional[float] = None,
    calibration_pose_qc: Optional[Dict[str, Any]] = None,
    target_pose_qc: Optional[Dict[str, Any]] = None,
    aggregator_sensitivity: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[str]:
    """Build the TIDE_Results_<target>.txt summary lines."""
    lines = [
        "===========================================",
        "--- TIDE Estimation Pipeline Summary ---",
        "===========================================",
        "",
        f"  Subject: {subject_id}",
        f"  Date: {timestamp_str}",
        f"  Output Folder: {out_dir}",
        f"  Parallel Workers: {num_workers}",
        "",
        "--- Configuration ---",
        f"  Input T1w: {t1w_path}",
        f"  CST Tractogram: {cst_bundle_path}",
        f"  Target Tractogram: {target_bundle_path}",
        f"  Spatial Mode: {spatial_mode}",
        f"  Weight Source: {weight_source}",
        f"  ROI Size: {roi_size_mm} mm",
        f"  Activation Length: {activation_length_mm} mm",
        "",
        f"--- M1 Calibration ({calibration_label}) ---",
        f"  Measured RMT: {measured_rmt_mso} %MSO",
        f"  Optimized Matrix: {m1_matrix_str}",
        f"  M1 Coil Pose QC: {_format_pose_qc(calibration_pose_qc)}",
        f"  CST Efficiency (Weighted): {af_cst_w:.4f} V/m^2",
        f"  CST Efficiency (Unweighted): {af_cst_u:.4f} V/m^2",
        f"  RMT Intensity (dI/dt): {intensity_rmt / 1e6:.2f} A/us",
        f"  Biological Threshold: {biological_threshold:.2f} V/m^2",
        "",
        f"--- Target Estimation ({target_label}) ---",
        f"  Target Coords (Cortex): {target_coords}",
        f"  Optimized Scalp Position: {opt_scalp_str}",
        f"  Optimized Matrix: {tgt_matrix_str}",
        f"  Target Coil Pose QC: {_format_pose_qc(target_pose_qc)}",
        f"  Target Efficiency (Weighted): {af_tgt_w:.4f} V/m^2",
        f"  Target Efficiency (Unweighted): {af_tgt_u:.4f} V/m^2",
        "",
        "--- Geometric Analysis ---",
        f"  CST Alignment: {cst_align:.4f}",
        f"  Target Alignment: {tgt_align:.4f}",
        f"  CST Alignment Corrected: {_format_optional_float(cst_align_corrected, 4)}",
        f"  Target Alignment Corrected: {_format_optional_float(tgt_align_corrected, 4)}",
        f"  CST Depth: {cst_depth:.1f} mm",
        f"  Target Depth: {tgt_depth:.1f} mm",
        "",
        "--- Validation Metrics ---",
        f"  Optimization Gain: {optimization_gain:.2f}x",
        f"  Geometry Factor (at M1): {ratio_at_m1:.3f}",
        f"  I from M1 Position: {intensity_from_m1_position:.1f}%",
        "",
        "===========================================",
        "--- RESULTS ---",
        "===========================================",
        f"{'Metric':<30} | {'Unweighted':<15} | {'Weighted':<15}",
        "-" * 65,
        f"{'Target Efficiency (V/m^2)':<30} | {af_tgt_u:<15.4f} | {af_tgt_w:<15.4f}",
        f"{'Estimated I - Raw (%)':<30} | {intensity_raw_u:<15.1f} | {intensity_raw_w:<15.1f}",
        f"{'Estimated I - Clamped (%)':<30} | {intensity_clamped_u:<15.1f} | {intensity_clamped_w:<15.1f}",
        f"{'I Flag':<30} | {intensity_flag_u:<15} | {intensity_flag_w:<15}",
        f"{'MSO Floor Ratio':<30} | {mso_floor_ratio:<15} | {mso_floor_ratio:<15}",
        f"{'SEI (AF_target/AF_CST)':<30} | {sei_u:<15.4f} | {sei_w:<15.4f}",
        f"{'Multiplier (M_CST/M_target)':<30} | {multiplier_u:<15.4f} | {multiplier_w:<15.4f}",
        "=" * 65,
    ]
    lines.extend(build_aggregator_sensitivity_lines(aggregator_sensitivity))
    return lines


def build_aggregator_sensitivity_lines(
    aggregator_sensitivity: Optional[Dict[str, Dict[str, float]]],
) -> List[str]:
    """
    Build the additive aggregator-sensitivity block appended to the summary.

    Returns an empty list when no sensitivity data is supplied, so the report is
    byte-identical to its previous form for callers that do not pass it.
    """
    if not aggregator_sensitivity:
        return []

    lines = [
        "",
        "--- Aggregator Sensitivity ---",
        f"  Reported dose uses '{AGGREGATOR_LABELS[PRIMARY_AGGREGATOR]}'; "
        "rows below are diagnostic.",
        "",
        f"{'Aggregator':<20} | {'AF_CST (U)':<12} | {'AF_tgt (U)':<12} | {'SEI (U)':<9} | "
        f"{'I Raw (U)':<9} | {'AF_CST (W)':<12} | {'AF_tgt (W)':<12} | {'SEI (W)':<9} | "
        f"{'I Raw (W)':<9}",
        "-" * 122,
    ]
    for key in AGGREGATOR_KEYS:
        row = aggregator_sensitivity.get(key)
        if row is None:
            continue
        lines.append(
            f"{AGGREGATOR_LABELS[key]:<20} | "
            f"{row['af_cst_unweighted']:<12.4f} | {row['af_target_unweighted']:<12.4f} | "
            f"{row['sei_unweighted']:<9.4f} | {row['intensity_raw_unweighted']:<9.1f} | "
            f"{row['af_cst_weighted']:<12.4f} | {row['af_target_weighted']:<12.4f} | "
            f"{row['sei_weighted']:<9.4f} | {row['intensity_raw_weighted']:<9.1f}"
        )
    lines.append("=" * 122)
    return lines
