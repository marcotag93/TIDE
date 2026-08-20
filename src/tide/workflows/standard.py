"""
Standard Simulation and Optimization Workflows
==============================================
Standalone simulation and optimization without full TIDE estimation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tide.core import io, physics, tractography
from tide.core.geometry import (
    calculate_alignment_and_depth,
    calculate_alignment_corrected,
    evaluate_coil_pose_qc,
    project_target_to_scalp,
)
from tide.interfaces.sampling import sample_field_at_coordinates
from tide.interfaces.simnibs_interface import SimNIBSInterface
from tide.interfaces.unified_estimation import AnalysisConfig, analyze_bundle, load_surface_tree
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
from tide.workflows._shared import WorkflowError, split_vectors_by_streamline

log = logging.getLogger(__name__)


def run_standard_simulation(config: SimNIBSConfig) -> None:
    """
    Runs a standard TMS simulation.

    If orientation is provided as a 3D vector (not a 4x4 matrix), the workflow
    automatically runs optimization first to find the optimal coil position and
    orientation, then uses the resulting 4x4 matrix for the simulation.

    Includes AF calculation and visualization if a bundle is provided.
    """
    validate_workflow_config(config, "simulation")

    log.highlight("=== Starting Standard Simulation ===")

    # Log configuration parameters
    log.info("=== Configuration Parameters ===")
    log.info(f"Subject ID: {config.subject.id}")
    log.info(f"Target site: {config.target.label}")
    log.info(f"Coil model: {config.coil.coil_model}")
    log.info(f"Coil distance: {config.coil.coil_distance_mm} mm")
    log.info(f"dI/dt max: {config.coil.device_didt_max / 1e6:.2f} A/µs")
    log.info(f"ROI size: {config.options.roi_size_mm} mm")
    log.info(f"Activation length: {config.options.activation_length_mm} mm")
    log.info(f"Field mode: {config.options.field_mode}")
    log.info("=" * 50)

    # --- Medoid Logic ---
    if config.target.medoid_endpoint:
        if not config.target.bundle_path or not config.target.bundle_path.exists():
            raise WorkflowError("Medoid endpoint requested but bundle path is missing.")

        log.highlight(f"Computing cortical medoid for: {config.target.label}")
        try:
            new_coords = tractography.get_bundle_cortical_medoid(
                config.target.bundle_path,
                config.subject.t1w_path,
                reference_coord=config.target.coords,
            )
            log.highlight(f"Medoid: {new_coords.tolist()}")
            config.target.coords = new_coords.tolist()
        except Exception as e:
            raise WorkflowError(f"Failed to calculate medoid: {e}") from e

    out_dir = config.subject.derivatives_path / f"simulation_{config.target.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Note: Config save is moved AFTER optimization to include generated matrix
    # Variables to store generated matrix and scalp coords for config save
    generated_target_matrix = None
    generated_target_scalp_coords = None
    pose_qc = None

    sim_coords = config.target.scalp_coords
    sim_orientation = config.target.orientation

    # Detect if orientation is a 4x4 matrix or a 3D vector
    is_matrix = orientation_is_matrix(config.target.orientation)

    # --- Auto-Optimization Logic ---
    # If orientation is a 3D vector (not a 4x4 matrix), run optimization first
    # to find the optimal coil position and orientation
    if not is_matrix:
        if config.target.coords:
            log.highlight("--- Running Auto-Optimization (3D vector orientation detected) ---")
            log.info(
                "Orientation is a 3D vector reference point. "
                "Running optimization to find optimal coil position..."
            )

            try:
                opt_matrix, opt_scalp_coords = SimNIBSInterface.run_optimization(
                    mesh_path=config.subject.mesh_path,
                    output_dir=out_dir,
                    coil_path=config.coil.coil_path,
                    target_coords=config.target.coords,
                    scalp_centre=config.target.scalp_coords,
                    orientation_ref=config.target.orientation,
                    didt=1e6,  # Use normalized dI/dt for optimization
                    use_adm=config.options.adm_optimization,
                    spatial_resolution=config.options.opt_spatial_resolution,
                    angle_resolution=config.options.opt_angle_resolution,
                    search_angle=config.options.opt_search_angle,
                    search_radius_mm=config.options.opt_search_radius,
                )

                # Use the optimized 4x4 matrix for simulation
                sim_orientation = opt_matrix.tolist()
                generated_target_matrix = opt_matrix.tolist()  # Save for config
                generated_target_scalp_coords = opt_scalp_coords.tolist()
                sim_coords = None  # Matrix includes position, no separate coords needed
                is_matrix = True  # Update flag since we now have a matrix
                qc = evaluate_coil_pose_qc(
                    config.subject.mesh_path,
                    opt_matrix,
                    opt_scalp_coords,
                )
                pose_qc = qc.as_dict()
                if qc.status == "WARN":
                    log.warning(f"Target coil pose QC warning: {qc.reasons}")

                log.highlight(
                    f"Optimization complete. Optimal scalp position: {opt_scalp_coords.tolist()}"
                )

                # Save optimization result
                io.save_optimization_result_txt(
                    out_dir / f"{config.target.label}_opt_result.txt",
                    opt_matrix,
                    opt_scalp_coords,
                    pose_qc=pose_qc,
                )

            except Exception as e:
                raise WorkflowError(f"Auto-optimization failed: {e}") from e
        else:
            raise WorkflowError("No cortical coordinates provided for optimization.")
    else:
        # --- Projection Logic (only when using matrix directly) ---
        # If we have a matrix, we don't need scalp projection
        log.info("Using provided 4x4 transformation matrix directly.")
        qc = evaluate_coil_pose_qc(config.subject.mesh_path, np.asarray(sim_orientation))
        pose_qc = qc.as_dict()
        if qc.status == "WARN":
            log.warning(f"Target coil pose QC warning: {qc.reasons}")

    # Save configuration (after optimization, with generated matrix and
    # resolved scalp coords) so the output YAML is fully re-runnable via
    # `--workflow simulation` without re-triggering optimization or medoid logic.
    save_config_to_output(
        config,
        out_dir,
        "simulation",
        generated_target_matrix=generated_target_matrix,
        generated_target_scalp_coords=generated_target_scalp_coords,
        medoid_resolved=bool(config.target.medoid_endpoint),
    )

    # --- Intensity Logic ---
    sim_didt = 1e6
    if config.target.didt is not None:
        sim_didt = config.target.didt
    elif config.target.mso is not None and config.coil.device_didt_max:
        sim_didt = config.coil.device_didt_max * (config.target.mso / 100.0)

    log.info(f"Starting simulation for {config.target.label}...")
    log.info(f"Simulation: dI/dt={sim_didt / 1e6:.2f} A/µs")

    # --- Run Simulation ---
    try:
        mesh_path = SimNIBSInterface.run_simulation(
            mesh_path=config.subject.m2m_path,
            output_dir=out_dir,
            coil_path=config.coil.coil_path,
            didt=sim_didt,
            coords=sim_coords,
            orientation=sim_orientation,
            distance_mm=config.coil.coil_distance_mm,
            fields="veEjJ",
        )
    except Exception as e:
        raise WorkflowError(f"Simulation failed: {e}") from e

    log.highlight(f"Simulation complete: {mesh_path.name}")

    # --- E-field Mapping (if bundle provided) ---
    if config.target.bundle_path and config.target.bundle_path.exists():
        log.info(f"Computing analysis for {config.target.label}...")
        _process_bundle_mapping(config, mesh_path, out_dir, pose_qc=pose_qc)

    log.highlight("Standard simulation finished.")
    log.highlight("Output files:")
    log.highlight(f"  -> Simulation: {out_dir.resolve()}")
    log.highlight(f"  -> Mesh: {mesh_path.resolve()}")


def run_standard_optimization(config: SimNIBSConfig) -> None:
    """
    Runs a standard TMS optimization.
    """
    log.highlight("=== Starting Standard Optimization ===")

    # Log configuration parameters
    log.info("=== Configuration Parameters ===")
    log.info(f"Subject ID: {config.subject.id}")
    log.info(f"Target site: {config.target.label}")
    log.info(f"Coil model: {config.coil.coil_model}")
    log.info(f"Coil distance: {config.coil.coil_distance_mm} mm")
    log.info(f"dI/dt max: {config.coil.device_didt_max / 1e6:.2f} A/µs")
    log.info(f"ROI size: {config.options.roi_size_mm} mm")
    log.info(f"Activation length: {config.options.activation_length_mm} mm")
    log.info(f"Field mode: {config.options.field_mode}")
    log.info(f"ADM optimization: {config.options.adm_optimization}")
    log.info(f"Optimization search radius: {config.options.opt_search_radius} mm")
    log.info(f"Optimization spatial resolution: {config.options.opt_spatial_resolution} mm")
    log.info(f"Optimization angle resolution: {config.options.opt_angle_resolution}°")
    log.info(f"Optimization search angle: {config.options.opt_search_angle}°")
    log.info("=" * 50)

    # --- Medoid Logic ---
    if config.target.medoid_endpoint:
        if not config.target.bundle_path or not config.target.bundle_path.exists():
            raise WorkflowError("Medoid endpoint requested but bundle path is missing.")

        log.highlight(f"Computing cortical medoid for: {config.target.label}")
        try:
            new_coords = tractography.get_bundle_cortical_medoid(
                config.target.bundle_path,
                config.subject.t1w_path,
                reference_coord=config.target.coords,
            )
            log.highlight(f"Medoid: {new_coords.tolist()}")
            config.target.coords = new_coords.tolist()
            config.target.scalp_coords = None  # Force re-projection
        except Exception as e:
            raise WorkflowError(f"Failed to calculate medoid: {e}") from e

    out_dir = config.subject.derivatives_path / f"optimization_{config.target.label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.target.coords:
        raise WorkflowError("Target coordinates required for optimization.")

    sim_didt = 1e6
    if config.target.didt is not None:
        sim_didt = config.target.didt
    elif config.target.mso is not None and config.coil.device_didt_max:
        sim_didt = config.coil.device_didt_max * (config.target.mso / 100.0)

    log.info(f"Starting optimization for {config.target.label}...")
    log.info(f"Optimization: target={config.target.coords}, dI/dt={sim_didt / 1e6:.2f} A/µs")

    try:
        orientation_to_use = (
            config.target.orientation
            if isinstance(config.target.orientation, (list, str))
            else None
        )

        # Detailed logging for debugging orientation issues
        log.info("=" * 60)
        log.info("[STANDARD_OPT] === OPTIMIZATION PARAMETERS ===")
        log.info(f"[STANDARD_OPT] target_coords (cortex): {config.target.coords}")
        log.info(f"[STANDARD_OPT] scalp_coords: {config.target.scalp_coords}")
        log.info(f"[STANDARD_OPT] orientation from config: {config.target.orientation}")
        log.info(f"[STANDARD_OPT] orientation type: {type(config.target.orientation).__name__}")
        log.info(f"[STANDARD_OPT] orientation_ref being sent: {orientation_to_use}")
        log.info(
            f"[STANDARD_OPT] search_angle: {config.options.opt_search_angle}° (±{config.options.opt_search_angle / 2}°)"
        )
        log.info(f"[STANDARD_OPT] angle_resolution: {config.options.opt_angle_resolution}°")
        log.info(f"[STANDARD_OPT] spatial_resolution: {config.options.opt_spatial_resolution} mm")
        log.info(f"[STANDARD_OPT] search_radius: {config.options.opt_search_radius} mm")
        log.info("=" * 60)

        matrix, scalp_coords = SimNIBSInterface.run_optimization(
            mesh_path=config.subject.mesh_path,
            output_dir=out_dir,
            coil_path=config.coil.coil_path,
            target_coords=config.target.coords,
            scalp_centre=config.target.scalp_coords,
            orientation_ref=orientation_to_use,
            didt=sim_didt,
            use_adm=config.options.adm_optimization,
            spatial_resolution=config.options.opt_spatial_resolution,
            angle_resolution=config.options.opt_angle_resolution,
            search_angle=config.options.opt_search_angle,
            search_radius_mm=config.options.opt_search_radius,
        )
        qc = evaluate_coil_pose_qc(config.subject.mesh_path, matrix, scalp_coords)
        pose_qc = qc.as_dict()
        if qc.status == "WARN":
            log.warning(f"Target coil pose QC warning: {qc.reasons}")

        log.highlight(f"Optimal scalp position: {scalp_coords.tolist()}")

        result_file = out_dir / f"{config.target.label}_opt_result.txt"
        io.save_optimization_result_txt(result_file, matrix, scalp_coords, pose_qc=pose_qc)

    except Exception as e:
        raise WorkflowError(f"Optimization failed: {e}") from e

    # Save configuration (after optimization) so the output YAML carries the
    # generated 4x4 matrix and resolved scalp coords, enabling a direct
    # `--workflow simulation` re-run without re-triggering optimization.
    save_config_to_output(
        config,
        out_dir,
        "optimization",
        generated_target_matrix=matrix.tolist(),
        generated_target_scalp_coords=scalp_coords.tolist(),
        medoid_resolved=bool(config.target.medoid_endpoint),
    )

    log.highlight("Standard optimization finished.")
    log.highlight("Output files:")
    log.highlight(f"  -> Optimization: {out_dir.resolve()}")
    log.highlight(f"  -> Result file: {result_file.resolve()}")


def _process_bundle_mapping(
    config: SimNIBSConfig,
    mesh_path: Path,
    out_dir: Path,
    pose_qc: Optional[Dict[str, object]] = None,
) -> None:
    """Process E-field mapping to bundle with AF calculation and visualization."""
    prefix = f"{config.target.label}_{config.options.field_mode}"
    viz_out = out_dir / "visualizations"

    log.debug(f"Mapping E-field to bundle: {config.target.bundle_path.name}")

    try:
        sft = tractography.load_tract(config.target.bundle_path, config.subject.t1w_path)
        points = np.concatenate(sft.streamlines)

        e_vecs = sample_field_at_coordinates(
            mesh_path, points, "E", output_dir=out_dir, file_prefix=prefix
        )

        e_vecs_list = split_vectors_by_streamline(e_vecs, sft.streamlines)

        # Filter streamlines by angular deviation. Track original ids through the
        # drop chain so SIFT2 weights stay aligned in analyze_bundle (C-002).
        streamlines_to_process = list(sft.streamlines)
        orig_idx = np.arange(len(sft.streamlines))
        if config.options.max_angular_deviation_deg > 0:
            (
                streamlines_to_process,
                e_vecs_list,
                _,
                orig_idx,
            ) = tractography.filter_by_angular_deviation(
                streamlines_to_process,
                e_field_vectors=e_vecs_list,
                max_angle_deg=config.options.max_angular_deviation_deg,
                roi_center=config.target.coords,
                roi_radius=config.options.roi_size_mm,
                indices=orig_idx,
            )

        # Calculate AF
        new_sl, scalars, lengths, orig_idx = physics.calculate_scalar_map(
            streamlines_to_process,
            e_vecs_list,
            mode=config.options.field_mode,
            indices=orig_idx,
        )

        alignment_qc = None
        if config.target.coords:
            roi_masks, _ = tractography.get_roi_masks(
                new_sl,
                config.options.roi_size_mm,
                config.target.coords,
            )
            alignment, depth = calculate_alignment_and_depth(
                new_sl,
                e_vecs_list,
                roi_masks,
                mesh_path,
                config.target.coords,
            )
            alignment_corrected = calculate_alignment_corrected(
                new_sl,
                e_vecs_list,
                roi_masks,
            )
            alignment_qc = {
                "alignment": alignment,
                "alignment_corrected": alignment_corrected,
                "depth_mm": depth,
            }

        # Save outputs
        out_trk = out_dir / f"{prefix}.trk"
        io.save_tract_with_data(sft, new_sl, out_trk, "AF", scalars, lengths)

        out_nii = out_dir / f"{prefix}.nii.gz"
        if config.options.generate_visualizations:
            io.save_points_as_nifti(
                np.concatenate(new_sl),
                config.subject.t1w_path,
                out_nii,
                values=np.abs(np.concatenate(scalars)),
            )

        # Run unified analysis
        ana_config = AnalysisConfig(
            cst_trk="",
            target_trk=str(out_trk),
            rmt=0.0,
            cst_coords=np.zeros(3),
            target_coords=np.array(config.target.coords) if config.target.coords else np.zeros(3),
            surf_path=str(config.subject.surface_path) if config.subject.surface_path else None,
            gwi_threshold=config.options.gwi_threshold_mm,
            roi_radius=config.options.roi_size_mm,
            activation_len=config.options.activation_length_mm,
            target_weights=(
                str(config.subject.weights_target_path)
                if config.subject.weights_target_path
                else None
            ),
        )

        surf_tree = None
        if ana_config.surf_path:
            surf_tree = load_surface_tree(ana_config.surf_path)

        try:
            result = analyze_bundle(
                name=config.target.label,
                trk_path=str(out_trk),
                roi_center=ana_config.target_coords,
                config=ana_config,
                surface_tree=surf_tree,
                weight_path=ana_config.target_weights,
                orig_indices=orig_idx,
            )

            unit_str = "V/m²" if config.options.field_mode == "af" else "V/m"
            log.highlight(f"Robust Metric (Weighted): {result.metric_weighted:.2f} {unit_str}")
            log.highlight(f"Robust Metric (Unweighted): {result.metric_unweighted:.2f} {unit_str}")

        except Exception as e:
            raise WorkflowError(f"Unified analysis failed: {e}") from e

        scalar_values = np.concatenate(scalars) if scalars else np.array([])
        finite_scalar_values = scalar_values[np.isfinite(scalar_values)]
        metrics = {}
        if result is not None:
            metrics = {
                "Robust Metric (Weighted)": f"{result.metric_weighted:.4f}",
                "Robust Metric (Unweighted)": f"{result.metric_unweighted:.4f}",
            }
            for key in physics.AGGREGATOR_KEYS:
                label = physics.AGGREGATOR_LABELS[key]
                metrics[f"{label} (Weighted)"] = f"{result.aggregates_weighted.get(key, 0.0):.4f}"
                metrics[f"{label} (Unweighted)"] = (
                    f"{result.aggregates_unweighted.get(key, 0.0):.4f}"
                )
        io.save_mapping_summary(
            out_dir / f"{prefix}_summary.txt",
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Prefix": prefix,
                "Mesh": mesh_path,
                "Bundle": config.target.bundle_path,
                "Anatomy": config.subject.t1w_path,
                "Mode": config.options.field_mode,
                "Threshold_Percent": "N/A",
                "Total_Streamlines": len(new_sl),
                "Max_Value": (
                    f"{float(np.max(finite_scalar_values)):.6g}"
                    if finite_scalar_values.size
                    else "N/A"
                ),
                "Min_Value": (
                    f"{float(np.min(finite_scalar_values)):.6g}"
                    if finite_scalar_values.size
                    else "N/A"
                ),
                "Metrics": metrics,
                "QC": {
                    "pose_qc": pose_qc,
                    "alignment_qc": alignment_qc,
                },
                "Output_Files": {
                    "Tractogram": out_trk.name,
                    "NIfTI map": (
                        out_nii.name
                        if config.options.generate_visualizations
                        else "Disabled by configuration"
                    ),
                },
            },
        )

        # Generate 3D visualization
        if config.options.generate_3d_visualization and PYVISTA_AVAILABLE:
            log.debug("Generating 3D visualization...")
            try:
                viz_out.mkdir(parents=True, exist_ok=True)
                scalp_point = None
                if config.target.coords:
                    try:
                        scalp_point = project_target_to_scalp(
                            mesh_path, np.array(config.target.coords)
                        )
                    except Exception:
                        pass

                viz_config = VisualizationConfig(
                    efield_vmax=80.0,
                    af_vmax=(
                        float(np.percentile(np.abs(np.concatenate(scalars)), 99))
                        if scalars
                        else 100.0
                    ),
                )

                generate_bundle_visualization(
                    mesh_path=mesh_path,
                    streamlines=new_sl,
                    af_values=scalars,
                    roi_center=(
                        np.array(config.target.coords) if config.target.coords else np.zeros(3)
                    ),
                    roi_radius=config.options.roi_size_mm,
                    output_dir=viz_out,
                    prefix=prefix,
                    config=viz_config,
                    scalp_point=scalp_point,
                )
            except Exception as e:
                log.warning(f"Visualization failed: {e}")

    except Exception as e:
        if isinstance(e, WorkflowError):
            raise
        raise WorkflowError(f"E-field mapping failed: {e}") from e
