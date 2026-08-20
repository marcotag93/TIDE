"""
SimNIBS Interface Module
========================
Wrapper around SimNIBS functions for simulation and optimization.
"""

import logging
import os
import platform
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# Core imports from SimNIBS
try:
    import simnibs
    from simnibs import opt_struct, run_simnibs, sim_struct
except ImportError:
    raise ImportError(
        "SimNIBS is not installed. This pipeline requires the SimNIBS python environment."
    )

from tide.core.geometry import (
    compute_default_coil_orientation,
    coords_inside_brain,
    project_target_to_scalp,
)
from tide.utils.artifacts import (
    capture_artifacts,
    fixed_pose_cache_enabled,
    fixed_pose_cache_key,
    fresh_artifacts,
    record_artifact,
    restore_fixed_pose_artifacts,
    store_fixed_pose_artifacts,
)

log = logging.getLogger(__name__)


class SimNIBSInterface:
    """
    Wrapper around SimNIBS functions to handle Simulation and Optimization.
    Isolates the pipeline from direct calls to simnibs libraries.
    """

    @staticmethod
    def run_simulation(
        mesh_path: Path,
        output_dir: Path,
        coil_path: Path,
        didt: float,
        coords: Optional[List[float]] = None,
        orientation: Optional[Union[str, List[float], List[List[float]]]] = None,
        distance_mm: float = 4.0,
        fields: str = "E",
    ) -> Path:
        """
        Runs a standard isotropic FEM simulation.

        Args:
            mesh_path: Path to m2m directory or .msh file
            output_dir: Output directory for results
            coil_path: Path to coil model file
            didt: Stimulation intensity (A/s)
            coords: Scalp coordinates [x, y, z]
            orientation: Coil orientation (vector, 4x4 matrix, or EEG label)
            distance_mm: Coil-scalp distance
            fields: Which fields to compute (default 'E')

        Returns:
            Path to the generated mesh file (.msh)
        """
        log.debug(f"Setting up simulation: mesh={mesh_path}, output={output_dir}")

        s = sim_struct.SESSION()
        s.subpath = str(mesh_path)
        s.pathfem = str(output_dir)
        s.open_in_gmsh = False
        s.fields = fields

        tms = s.add_tmslist()
        tms.fnamecoil = str(coil_path)
        tms.anisotropy_type = "scalar"

        # Add position
        pos = tms.add_position()
        pos.didt = didt
        pos.distance = distance_mm

        is_matrix = (
            isinstance(orientation, list)
            and len(orientation) == 4
            and isinstance(orientation[0], list)
        )

        if is_matrix:
            log.debug("Using 4x4 transformation matrix")
            pos.matsimnibs = orientation
            pos.centre = None
        else:
            if not coords:
                raise ValueError("Coordinates required when orientation is not a 4x4 matrix.")
            pos.centre = coords
            pos.pos_ydir = orientation if orientation else "F8"
            # Log the pos_ydir being used for simulation
            log.info(f"[SIMULATION] pos_ydir (y_dir) = {pos.pos_ydir}")

            # Sanity check: pos.centre is expected to be a scalp coordinate.
            try:
                if coords_inside_brain(mesh_path, np.asarray(coords, dtype=float)):
                    log.warning(
                        f"[SIMULATION] Coil centre {list(coords)} lies inside the "
                        "grey-matter extent; expected a scalp coordinate. "
                        "Check the configuration."
                    )
            except Exception:
                pass

        # Archive existing simulation results to prevent OSError
        _archive_existing_results(output_dir)

        before = capture_artifacts(_simulation_mesh_candidates(output_dir))
        cache_key = None
        if is_matrix and fixed_pose_cache_enabled():
            try:
                cache_key = fixed_pose_cache_key(
                    mesh_path=mesh_path,
                    coil_path=coil_path,
                    orientation=orientation,
                    didt=didt,
                    distance_mm=distance_mm,
                    fields=fields,
                    runtime_signature={
                        "simnibs": str(getattr(simnibs, "__version__", "unknown")),
                        "numpy": np.__version__,
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
                        "MKL_DOMAIN_NUM_THREADS": os.environ.get("MKL_DOMAIN_NUM_THREADS", ""),
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
                    },
                )
                restored = restore_fixed_pose_artifacts(cache_key, output_dir)
            except (OSError, ValueError) as error:
                log.warning(f"Fixed-pose cache lookup failed; running SimNIBS: {error}")
                restored = []

            if restored:
                fresh = fresh_artifacts(before, _simulation_mesh_candidates(output_dir))
                result = _select_simulation_mesh(fresh, output_dir)
                tms.postprocess = fields
                _write_cached_simulation_metadata(s, output_dir, cache_key)
                record_artifact(
                    output_dir,
                    "simulation_mesh",
                    result,
                    fresh,
                    details={"cache": {"status": "hit", "key": cache_key}},
                )
                log.info(f"Fixed-pose cache hit: {cache_key}")
                return result

        log.debug("Starting SimNIBS simulation...")
        run_simnibs(s)

        fresh = fresh_artifacts(before, _simulation_mesh_candidates(output_dir))
        result = _select_simulation_mesh(fresh, output_dir)
        details = None
        if cache_key is not None:
            stored = False
            try:
                stored = store_fixed_pose_artifacts(
                    cache_key,
                    _simulation_cache_artifacts(result),
                )
            except (OSError, ValueError) as error:
                log.warning(f"Fixed-pose cache storage failed: {error}")
            details = {
                "cache": {
                    "status": "miss",
                    "key": cache_key,
                    "stored": stored,
                }
            }
        record_artifact(
            output_dir,
            "simulation_mesh",
            result,
            fresh,
            details=details,
        )

        log.debug(f"Simulation complete: {result}")
        return result

    @staticmethod
    def run_optimization(
        mesh_path: Path,
        output_dir: Path,
        coil_path: Path,
        target_coords: List[float],
        scalp_centre: Optional[List[float]] = None,
        orientation_ref: Optional[Union[str, List[float], List[List[float]]]] = None,
        didt: float = 1e6,
        distance_mm: float = 4.0,
        search_radius_mm: float = 10.0,
        spatial_resolution: float = 5.0,
        angle_resolution: float = 30.0,
        search_angle: float = 30.0,
        use_adm: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs isotropic TMS Optimization.

        Args:
            mesh_path: Path to .msh head mesh file
            output_dir: Output directory
            coil_path: Path to coil model
            target_coords: Cortical target coordinates [x, y, z]
            scalp_centre: Initial scalp position (auto-projected if None)
            orientation_ref: Handle direction reference point
            didt: Stimulation intensity
            distance_mm: Coil-scalp distance
            search_radius_mm: Search radius on scalp
            spatial_resolution: Spatial search step (mm)
            angle_resolution: Angular search step (degrees)
            search_angle: Total angular search range
            use_adm: Use ADM method (faster)

        Returns:
            Tuple of (best_4x4_matrix, best_scalp_coords)
        """
        # Log the incoming orientation_ref for debugging
        log.info(
            f"[OPTIMIZATION] Received orientation_ref = {orientation_ref} (type: {type(orientation_ref).__name__})"
        )

        # Auto-projection if needed
        if not scalp_centre:
            log.debug("Projecting cortical target to scalp...")
            scalp_centre = project_target_to_scalp(mesh_path, np.array(target_coords))
            scalp_centre = scalp_centre.tolist()

        # Auto-orientation if needed
        if not orientation_ref:
            log.debug("Computing default coil orientation (no orientation_ref provided)...")
            orientation_ref = compute_default_coil_orientation(mesh_path, np.array(scalp_centre))
            log.info(f"[OPTIMIZATION] Auto-computed pos_ydir (y_dir) = {orientation_ref}")

        log.debug(f"Optimization setup: target={target_coords}, scalp={scalp_centre}")

        # Setup optimization
        opt = opt_struct.TMSoptimize()
        opt.open_in_gmsh = False
        opt.fnamehead = str(mesh_path)
        opt.pathfem = str(output_dir)
        opt.fnamecoil = str(coil_path)
        opt.target = target_coords
        opt.centre = scalp_centre
        opt.distance = distance_mm
        opt.search_radius = search_radius_mm
        opt.spatial_resolution = spatial_resolution
        opt.angle_resolution = angle_resolution
        opt.search_angle = search_angle
        opt.solver_options = "pardiso"
        opt.didt = didt
        opt.method = "ADM" if use_adm else "direct"

        # NOTE: ADM method may handle orientation constraints differently
        if use_adm:
            log.info("[OPTIMIZATION] Using ADM method (faster but may have orientation quirks)")
        else:
            log.info("[OPTIMIZATION] Using DIRECT method (slower but more reliable orientation)")

        if orientation_ref:
            is_matrix = (
                isinstance(orientation_ref, list)
                and len(orientation_ref) == 4
                and isinstance(orientation_ref[0], list)
            )

            if is_matrix:
                raise ValueError(
                    "4x4 Matrix orientation is not supported for TMS Optimization. "
                    "Use cortex coordinates [x, y, z] or EEG label (e.g., 'F8')."
                )

            opt.pos_ydir = orientation_ref
            # pos_ydir_is_position=True tells SimNIBS to compute the handle direction
            # as (pos_ydir - coil_centre). This is only valid for coordinate lists;
            # EEG label strings (e.g. "F7") are looked up internally by SimNIBS and
            # must NOT have this flag set, otherwise numpy tries to subtract a string.
            if isinstance(orientation_ref, list):
                opt.pos_ydir_is_position = True
            # Log the final pos_ydir being sent to SimNIBS optimizer
            log.info(f"[OPTIMIZATION] Final pos_ydir (y_dir) sent to SimNIBS = {opt.pos_ydir}")
            log.info(
                f"[OPTIMIZATION] pos_ydir_is_position = {getattr(opt, 'pos_ydir_is_position', False)}"
            )
        else:
            log.warning(
                "[OPTIMIZATION] No orientation_ref provided - SimNIBS will use default orientation!"
            )

        # Log all optimization parameters being sent to SimNIBS
        log.info("[OPTIMIZATION] === SimNIBS TMSoptimize Parameters ===")
        log.info(f"[OPTIMIZATION] opt.target = {opt.target}")
        log.info(f"[OPTIMIZATION] opt.centre = {opt.centre}")
        log.info(f"[OPTIMIZATION] opt.pos_ydir = {getattr(opt, 'pos_ydir', 'NOT SET')}")
        log.info(f"[OPTIMIZATION] opt.search_radius = {opt.search_radius}")
        log.info(f"[OPTIMIZATION] opt.spatial_resolution = {opt.spatial_resolution}")
        log.info(f"[OPTIMIZATION] opt.angle_resolution = {opt.angle_resolution}")
        log.info(f"[OPTIMIZATION] opt.search_angle = {opt.search_angle}")
        log.info(f"[OPTIMIZATION] opt.method = {opt.method}")

        # Archive existing results
        _archive_existing_results(output_dir)

        log.debug("Starting TMS optimization...")
        # CRITICAL: opt.run() returns the optimal matsimnibs matrix directly!
        opt_matrix = opt.run()

        # Extract results from the returned matrix
        try:
            # opt.run() returns a 4x4 numpy array (or can be squeezed from higher dims)
            opt_matrix = np.atleast_2d(np.squeeze(opt_matrix))

            if opt_matrix.shape != (4, 4):
                raise ValueError(f"Unexpected matrix shape: {opt_matrix.shape}, expected (4, 4)")

            # Extract scalp coordinates from the transformation matrix (translation column)
            scalp_coords = opt_matrix[0:3, 3]

            # Extract and log the ACTUAL Y-direction from the result matrix
            # In SimNIBS matsimnibs format: column 0 = X-axis, column 1 = Y-axis, column 2 = Z-axis (normal)
            result_y_direction = opt_matrix[0:3, 1]
            result_z_direction = opt_matrix[0:3, 2]  # Coil normal (should point into head)

            log.info("=" * 60)
            log.info("[OPTIMIZATION] === RESULT ANALYSIS ===")
            log.info(f"[OPTIMIZATION] Result coil position: {scalp_coords.tolist()}")
            log.info(f"[OPTIMIZATION] Result Y-direction (handle): {result_y_direction.tolist()}")
            log.info(f"[OPTIMIZATION] Result Z-direction (normal): {result_z_direction.tolist()}")
            log.info("=" * 60)

            log.debug(f"Optimization complete: scalp_coords={scalp_coords}")
            return opt_matrix, scalp_coords

        except Exception as e:
            log.error(f"Failed to process optimization results: {e}")
            raise

    @staticmethod
    def _parse_optimization_log(log_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse SimNIBS optimization log to extract the best 4x4 matrix.

        DEPRECATED: This method is kept for backwards compatibility but is no longer
        the primary way to get optimization results. The opt.run() method returns
        the matrix directly, which is more reliable across SimNIBS versions.
        """
        matrix_lines = []
        found_header = False
        header_regex = re.compile(r"Best coil position")
        matrix_line_regex = re.compile(r"^\s*\[")

        with open(log_file, "r") as f:
            for line in f:
                if not found_header and header_regex.search(line):
                    found_header = True
                    continue

                if found_header:
                    clean_line = line.strip()
                    if matrix_line_regex.search(clean_line):
                        matrix_lines.append(clean_line)
                    if len(matrix_lines) == 4:
                        break

        if len(matrix_lines) != 4:
            raise ValueError(f"Could not parse 4x4 matrix from log: {log_file}")

        # Clean and convert to numpy
        matrix_string = " ".join(matrix_lines)
        matrix_string = re.sub(r"[\[\]]", "", matrix_string)
        matrix_string = re.sub(r"\s+", ",", matrix_string)
        matrix_string = matrix_string.strip(",")
        matrix_string = re.sub(r",,", ",", matrix_string)

        flat_arr = np.fromstring(matrix_string, sep=",")
        if flat_arr.size != 16:
            raise ValueError("Parsed matrix does not have 16 elements.")

        matrix_4x4 = flat_arr.reshape((4, 4))
        coords = matrix_4x4[0:3, 3]

        log.debug(f"Parsed optimization result: scalp_coords={coords}")
        return matrix_4x4, coords


def _archive_existing_results(output_dir: Path):
    """Archive existing SimNIBS results to prevent conflicts."""
    if not output_dir.exists():
        return

    existing_mats = sorted(output_dir.glob("simnibs_simulation*.mat"))

    if existing_mats:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_dir = output_dir / f"archive_{timestamp}"

        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            log.debug(f"Archiving {len(existing_mats)} existing result files")

            for mat_file in existing_mats:
                shutil.move(str(mat_file), str(archive_dir / mat_file.name))
        except Exception as e:
            log.warning(f"Failed to archive existing files: {e}")


def _simulation_mesh_candidates(output_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in output_dir.glob("*.msh")
        if path.name != "target.msh" and "optimize" not in path.name.lower()
    )


def _simulation_cache_artifacts(mesh_path: Path) -> List[Path]:
    artifacts = [mesh_path]
    options_path = Path(f"{mesh_path}.opt")
    if options_path.is_file():
        artifacts.append(options_path)

    summary_path = mesh_path.parent / "fields_summary.txt"
    if summary_path.is_file():
        artifacts.append(summary_path)

    for suffix in ("_scalar.msh", "_E.msh"):
        if mesh_path.name.endswith(suffix):
            prefix = mesh_path.name[: -len(suffix)]
            geometry_path = mesh_path.parent / f"{prefix}_coil_pos.geo"
            if geometry_path.is_file():
                artifacts.append(geometry_path)
            break
    return artifacts


def _write_cached_simulation_metadata(session: object, output_dir: Path, cache_key: str) -> None:
    time_str = getattr(session, "time_str", None)
    save_struct = getattr(sim_struct, "save_matlab_sim_struct", None)
    if not time_str or not callable(save_struct):
        return

    try:
        save_struct(session, str(output_dir / f"simnibs_simulation_{time_str}.mat"))
        (output_dir / f"simnibs_simulation_{time_str}.log").write_text(
            f"Fixed-pose cache hit: {cache_key}\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        log.warning(f"Could not write cached SimNIBS session metadata: {error}")


def _select_simulation_mesh(candidates: List[Path], output_dir: Path) -> Path:
    priority_groups = (
        [path for path in candidates if path.name.endswith("_scalar.msh")],
        [path for path in candidates if path.name.endswith("_E.msh")],
        candidates,
    )
    for group in priority_groups:
        if not group:
            continue
        if len(group) > 1:
            names = ", ".join(path.name for path in group)
            raise RuntimeError(f"Ambiguous simulation output: {names}")
        return group[0]

    raise FileNotFoundError(f"Simulation finished but no fresh .msh file was found in {output_dir}")
