"""
E-field Sampling Module
=======================
Interpolates E-field values at streamline coordinates using SimNIBS CLI.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree

from tide.utils import simnibs_env
from tide.utils.artifacts import capture_artifacts, fresh_artifacts, record_artifact

log = logging.getLogger(__name__)


def sample_field_at_coordinates(
    mesh_path: Path,
    coordinates: np.ndarray,
    field_name: str = "E",
    output_dir: Optional[Path] = None,
    file_prefix: str = "bundle",
) -> np.ndarray:
    """
    Interpolates E-field using SimNIBS CLI tool (get_fields_at_coordinates).

    Args:
        mesh_path: Path to SimNIBS mesh file
        coordinates: Nx3 array of coordinates to sample
        field_name: Field to sample ('E' for vector E-field)
        output_dir: Output directory for intermediate files
        file_prefix: Prefix for output files

    Returns:
        Nx3 array of E-field vectors at coordinates
    """
    # Find the CLI command (handle Windows .cmd/.exe extensions)
    cli_cmd = simnibs_env.find_get_fields_at_coordinates()
    if cli_cmd is None:
        raise RuntimeError(
            "Command 'get_fields_at_coordinates' not found in PATH. "
            "Ensure SimNIBS is properly installed and in your PATH."
        )

    work_dir = output_dir if output_dir else mesh_path.parent
    coords_csv = work_dir / f"{file_prefix}_coords.csv"

    if Path(cli_cmd).suffix.lower() in {".bat", ".cmd"}:
        batch_metacharacters = '&|<>()^%!"\r\n'
        batch_args = (cli_cmd, str(mesh_path), str(coords_csv))
        if any(char in arg for arg in batch_args for char in batch_metacharacters):
            raise ValueError("Windows batch command arguments cannot contain shell metacharacters")

    np.savetxt(coords_csv, coordinates, delimiter=",", comments="")

    def output_candidates() -> List[Path]:
        return [path for path in work_dir.glob(f"{file_prefix}_coords_*.csv") if path != coords_csv]

    before = capture_artifacts(output_candidates(), hash_contents=True)

    log.debug(f"Sampling E-field at {len(coordinates)} points using CLI: {cli_cmd}")

    cmd = [
        cli_cmd,
        "--mesh",
        str(mesh_path),
        "--csv",
        str(coords_csv),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(work_dir),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            log.debug(f"CLI stdout: {result.stdout[:200]}")
    except subprocess.CalledProcessError as e:
        log.error(f"CLI stderr: {e.stderr}")
        raise RuntimeError(f"CLI interpolation failed: {e}")

    expected_out = work_dir / f"{file_prefix}_coords_{field_name}.csv"
    candidates = fresh_artifacts(before, output_candidates(), hash_contents=True)
    if expected_out in candidates:
        selected_candidates = [expected_out]
    else:
        selected_candidates = candidates

    if not selected_candidates:
        raise FileNotFoundError(
            f"Sampling finished but no fresh output CSV was found in {work_dir}"
        )
    if len(selected_candidates) > 1:
        names = ", ".join(path.name for path in selected_candidates)
        raise RuntimeError(f"Ambiguous sampling output: {names}")

    expected_out = selected_candidates[0]
    record_artifact(
        work_dir,
        f"sampling:{file_prefix}:{field_name}",
        expected_out,
        candidates,
    )

    try:
        df = pd.read_csv(expected_out, header=None, comment="#")
        raw_data = df.values
    except Exception as e:
        log.error(f"Failed to read output CSV: {e}")
        raise

    # Parse output format
    if raw_data.shape[1] == 3:
        out_coords = coordinates
        out_vals = raw_data
    elif raw_data.shape[1] >= 6:
        out_coords = raw_data[:, :3]
        out_vals = raw_data[:, 3:6]
    else:
        if raw_data.shape[1] == 4:
            out_coords = raw_data[:, :3]
            out_vals = raw_data[:, 3].reshape(-1, 1)
        else:
            raise ValueError(f"Unexpected CSV column count: {raw_data.shape[1]}")

    log.debug(f"Loaded E-field vectors: shape={out_vals.shape}")

    # Clean up temp file
    try:
        coords_csv.unlink()
    except Exception:
        pass

    if out_coords is coordinates:
        if len(out_vals) != len(coordinates):
            raise RuntimeError(
                "E-field sampling output omitted coordinates and changed the row count; "
                "realignment is not possible."
            )
        return out_vals

    if len(out_vals) == len(coordinates) and np.array_equal(out_coords, coordinates):
        return out_vals

    log.debug("Verifying and re-aligning sampled coordinates via nearest neighbor")
    return _realign_sampled_field(coordinates, out_coords, out_vals)


def _realign_sampled_field(
    coordinates: np.ndarray,
    out_coords: np.ndarray,
    out_vals: np.ndarray,
    tol_mm: float = 0.1,
) -> np.ndarray:
    """
    Map CLI-returned field values back onto the requested coordinates.

    The ``tol_mm`` match tolerance sits well below the streamline step size and
    well above the float round-trip error of the coordinate CSV. Points with no
    returned value are filled with NaN (not 0.0) so the absence is detectable
    downstream instead of silently deflating the AF. Raises if more than 1% of
    points are unmatched, which signals a coordinate/ordering mismatch.
    """
    n_components = out_vals.shape[1]
    tree = KDTree(out_coords)
    _, indices = tree.query(coordinates, k=1, distance_upper_bound=tol_mm)
    matched = indices < len(out_coords)

    matched_indices = indices[matched]
    if len(np.unique(matched_indices)) != len(matched_indices):
        raise RuntimeError(
            "E-field sampling produced duplicate nearest-neighbour assignments; "
            "returned coordinates are not one-to-one."
        )

    full_vectors = np.full((len(coordinates), n_components), np.nan, dtype=float)
    full_vectors[matched] = out_vals[indices[matched]]

    n_missing = int((~matched).sum())
    if n_missing:
        missing_frac = n_missing / len(coordinates)
        log.warning(
            f"E-field sampling: {n_missing}/{len(coordinates)} points "
            f"({missing_frac:.1%}) unmatched within {tol_mm} mm; filled with NaN."
        )
        if missing_frac > 0.01:
            raise RuntimeError(
                f"E-field sampling dropped {missing_frac:.1%} of points "
                "(> 1% threshold); coordinate or ordering mismatch likely."
            )

    return full_vectors
