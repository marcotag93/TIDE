import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# SimNIBS imports for mesh reading
try:
    from simnibs.msh import read_msh
except ImportError:
    try:
        from simnibs import read_msh
    except ImportError:
        read_msh = None

log = logging.getLogger(__name__)

# GM node coordinates cached per resolved .msh path (read_msh is expensive).
_GM_NODE_CACHE: Dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class CoilPoseQC:
    """Coil-pose accessibility diagnostic."""

    status: str
    reasons: Tuple[str, ...] = ()
    scalp_outward_dot: Optional[float] = None
    scalp_normal_z: Optional[float] = None
    coil_normal_z: Optional[float] = None
    scalp_z_percentile: Optional[float] = None
    nearest_scalp_distance_mm: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "scalp_outward_dot": self.scalp_outward_dot,
            "scalp_normal_z": self.scalp_normal_z,
            "coil_normal_z": self.coil_normal_z,
            "scalp_z_percentile": self.scalp_z_percentile,
            "nearest_scalp_distance_mm": self.nearest_scalp_distance_mm,
        }


def format_coil_pose_qc(qc: Optional[CoilPoseQC]) -> str:
    """Return a compact single-line representation for reports/logs."""
    if qc is None:
        return "N/A"

    parts = [qc.status]
    if qc.reasons:
        parts.append(f"({', '.join(qc.reasons)})")

    details = []
    if qc.scalp_outward_dot is not None:
        details.append(f"normal_dot={qc.scalp_outward_dot:.3f}")
    if qc.scalp_normal_z is not None:
        details.append(f"scalp_normal_z={qc.scalp_normal_z:.3f}")
    if qc.coil_normal_z is not None:
        details.append(f"coil_normal_z={qc.coil_normal_z:.3f}")
    if qc.scalp_z_percentile is not None:
        details.append(f"scalp_z_pct={qc.scalp_z_percentile:.1f}")
    if qc.nearest_scalp_distance_mm is not None:
        details.append(f"nearest_scalp_mm={qc.nearest_scalp_distance_mm:.1f}")
    if details:
        parts.append("[" + ", ".join(details) + "]")

    return " ".join(parts)


def validate_coil_pose_for_dose(qc: CoilPoseQC, *, explicit_matrix: bool) -> None:
    if qc.status != "WARN" or explicit_matrix:
        return

    reasons = ", ".join(qc.reasons) if qc.reasons else "unspecified pose QC warning"
    raise ValueError(
        f"Automatically optimized coil pose is not dose-eligible ({reasons}). "
        "Supply a verified 4x4 matsimnibs orientation for a specialist override."
    )


def _mesh_path(mesh_path: Path) -> Optional[Path]:
    msh = Path(mesh_path)
    if msh.is_dir():
        meshes = sorted(msh.glob("*.msh"))
        if not meshes:
            return None
        if len(meshes) > 1:
            raise ValueError(f"Head-model directory contains multiple .msh files: {msh}")
        msh = meshes[0]
    return msh


def _gm_nodes(mesh_path: Path) -> Optional[np.ndarray]:
    """Return grey-matter node coordinates for a head mesh, cached per file."""
    if read_msh is None:
        return None

    msh = _mesh_path(mesh_path)
    if msh is None:
        return None

    key = str(msh)
    cached = _GM_NODE_CACHE.get(key)
    if cached is not None:
        return cached

    mesh = read_msh(key)
    elm_tags = mesh.elm.tag1
    gm_tag = 1002 if 1002 in elm_tags else 2
    gm_node_indices = np.unique(mesh.elm.node_number_list[elm_tags == gm_tag])
    coords = mesh.nodes[:][gm_node_indices]
    _GM_NODE_CACHE[key] = coords
    return coords


def _mesh_scalp_nodes_and_center(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if read_msh is None:
        raise ImportError("SimNIBS not installed. Cannot read mesh.")

    msh = _mesh_path(mesh_path)
    if msh is None:
        raise FileNotFoundError(f"No .msh file found in {mesh_path}")

    mesh = read_msh(str(msh))
    all_nodes = mesh.nodes[:]
    elm_tags = mesh.elm.tag1

    if 1002 in elm_tags or 1005 in elm_tags:
        gm_tag, scalp_tag = 1002, 1005
    else:
        gm_tag, scalp_tag = 2, 5

    scalp_mask = elm_tags == scalp_tag
    elm_types = getattr(mesh.elm, "elm_type", None)
    if elm_types is not None and len(elm_types) == len(elm_tags):
        tri_mask = scalp_mask & (elm_types == 2)
        if np.any(tri_mask):
            scalp_mask = tri_mask

    scalp_elems = np.asarray(mesh.elm.node_number_list[scalp_mask])
    if scalp_elems.size == 0:
        raise ValueError(f"No scalp elements found with tag {scalp_tag}")
    scalp_node_indices = np.unique(scalp_elems[:, :3])
    scalp_nodes = all_nodes[scalp_node_indices]

    gm_elems = np.asarray(mesh.elm.node_number_list[elm_tags == gm_tag])
    if gm_elems.size == 0:
        raise ValueError(f"No grey matter elements found with tag {gm_tag}")
    gm_node_indices = np.unique(gm_elems[:, :3])
    brain_center = np.mean(all_nodes[gm_node_indices], axis=0)

    return scalp_nodes, brain_center


def _local_scalp_normal(
    scalp_nodes: np.ndarray,
    brain_center: np.ndarray,
    scalp_coords: np.ndarray,
    n_neighbors: int,
) -> Tuple[np.ndarray, float, float]:
    distances = np.linalg.norm(scalp_nodes - scalp_coords, axis=1)
    nearest_count = min(max(n_neighbors, 3), len(scalp_nodes))
    nearest_indices = np.argsort(distances)[:nearest_count]
    neighbors = scalp_nodes[nearest_indices]

    centered = neighbors - neighbors.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    scalp_normal = vh[-1]

    if np.dot(scalp_normal, scalp_coords - brain_center) < 0:
        scalp_normal = -scalp_normal

    norm = np.linalg.norm(scalp_normal)
    if norm < 1e-9:
        raise ValueError("Could not estimate a stable local scalp normal")

    z_percentile = float(np.mean(scalp_nodes[:, 2] <= scalp_coords[2]) * 100.0)
    return scalp_normal / norm, float(distances[nearest_indices[0]]), z_percentile


def evaluate_coil_pose_qc(
    mesh_path: Path,
    matrix: np.ndarray,
    scalp_coords: Optional[np.ndarray] = None,
    *,
    n_neighbors: int = 30,
    inward_dot_threshold: float = -0.20,
    inferior_percentile: float = 10.0,
    inferior_normal_z: float = 0.25,
    upward_coil_z: float = 0.25,
    max_scalp_distance_mm: float = 15.0,
) -> CoilPoseQC:
    """
    Evaluate coil-pose accessibility without relocating or modifying the pose.

    The SimNIBS matsimnibs third column is treated as the inward coil normal.
    A valid scalp-side pose should therefore have a negative dot product with
    the local outward scalp normal. Inferior-surface checks catch the documented
    skull-base/upward-firing failure mode. Callers decide whether WARN is
    report-only or a dose-eligibility gate.
    """
    if read_msh is None:
        return CoilPoseQC(status="UNAVAILABLE", reasons=("simnibs_unavailable",))

    matrix_arr = np.asarray(matrix, dtype=float)
    if matrix_arr.shape != (4, 4) or not np.isfinite(matrix_arr).all():
        return CoilPoseQC(status="UNAVAILABLE", reasons=("invalid_matrix",))

    scalp_arr = (
        np.asarray(scalp_coords, dtype=float)
        if scalp_coords is not None
        else np.asarray(matrix_arr[:3, 3], dtype=float)
    )
    if scalp_arr.shape != (3,) or not np.isfinite(scalp_arr).all():
        return CoilPoseQC(status="UNAVAILABLE", reasons=("invalid_scalp_coords",))

    coil_normal = np.asarray(matrix_arr[:3, 2], dtype=float)
    coil_norm = np.linalg.norm(coil_normal)
    if coil_norm < 1e-9:
        return CoilPoseQC(status="UNAVAILABLE", reasons=("invalid_coil_normal",))
    coil_normal = coil_normal / coil_norm

    try:
        scalp_nodes, brain_center = _mesh_scalp_nodes_and_center(mesh_path)
        inferior_z_cut = float(np.percentile(scalp_nodes[:, 2], inferior_percentile))
        scalp_normal, nearest_distance, z_percentile = _local_scalp_normal(
            scalp_nodes,
            brain_center,
            scalp_arr,
            n_neighbors,
        )
    except Exception as exc:
        return CoilPoseQC(status="UNAVAILABLE", reasons=(f"mesh_unavailable:{exc}",))

    outward_dot = float(np.dot(coil_normal, scalp_normal))
    reasons = []
    if outward_dot >= inward_dot_threshold:
        reasons.append("coil_normal_not_inward")
    if nearest_distance > max_scalp_distance_mm:
        reasons.append("coil_center_far_from_scalp")
    is_inferior = scalp_arr[2] <= inferior_z_cut + 1e-6
    if is_inferior and scalp_normal[2] < -inferior_normal_z:
        reasons.append("inferior_scalp_surface")
    if is_inferior and coil_normal[2] > upward_coil_z:
        reasons.append("upward_firing_low_inferior_pose")

    return CoilPoseQC(
        status="WARN" if reasons else "PASS",
        reasons=tuple(reasons),
        scalp_outward_dot=outward_dot,
        scalp_normal_z=float(scalp_normal[2]),
        coil_normal_z=float(coil_normal[2]),
        scalp_z_percentile=z_percentile,
        nearest_scalp_distance_mm=nearest_distance,
    )


def coords_inside_brain(mesh_path: Path, coords: np.ndarray) -> Optional[bool]:
    """
    Heuristic test for whether a point lies within the grey-matter extent.

    Compares the point's distance from the brain centre against the GM extent
    along the point's own radial direction. True means the point looks like a
    cortical coordinate (inside GM) rather than a scalp coordinate. Returns
    None when the mesh cannot be read.
    """
    gm = _gm_nodes(mesh_path)
    if gm is None or len(gm) == 0:
        return None

    coords = np.asarray(coords, dtype=float)
    center = gm.mean(axis=0)
    offset = coords - center
    dist = float(np.linalg.norm(offset))
    if dist < 1e-6:
        return True

    direction = offset / dist
    gm_extent = float(((gm - center) @ direction).max())
    return dist < gm_extent


def project_target_to_scalp(mesh_path: Path, target_coords: np.ndarray) -> np.ndarray:
    """
    Projects a cortical target coordinate to the outermost scalp surface using
    Ray-Triangle Intersection (Ray Casting).

    This offers sub-millimeter precision compared to node-based cone search.

    Args:
        mesh_path: Path to the SimNIBS .msh file.
        target_coords: Numpy array [x, y, z] of the cortical target.

    Returns:
        Numpy array [x, y, z] of the exact intersection point on the scalp.

    Raises:
        ImportError: If SimNIBS is not installed.
        FileNotFoundError: If mesh file doesn't exist.
        ValueError: If target coordinates are invalid.
    """
    if read_msh is None:
        raise ImportError("SimNIBS not installed. Cannot read mesh.")

    # Input validation
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    target_coords = np.asarray(target_coords, dtype=float)
    if target_coords.shape != (3,):
        raise ValueError(f"target_coords must be shape (3,), got {target_coords.shape}")
    if not np.isfinite(target_coords).all():
        raise ValueError(f"target_coords contains non-finite values: {target_coords}")

    try:
        # --- Load Mesh ---
        log.debug(f"Loading mesh: {mesh_path}")
        mesh = read_msh(str(mesh_path))
        all_nodes = mesh.nodes[:]
        elm_tags = mesh.elm.tag1

        # Determine Tags (SimNIBS v3 vs v4 compatibility)
        if 1002 in elm_tags:
            gm_tag, scalp_tag = 1002, 1005
        else:
            gm_tag, scalp_tag = 2, 5

        # --- 1. Get Brain Center (Origin of Ray) ---
        gm_elm_mask = elm_tags == gm_tag
        gm_node_indices = np.unique(mesh.elm.node_number_list[gm_elm_mask])

        if len(gm_node_indices) == 0:
            raise ValueError(f"No grey matter elements found with tag {gm_tag}")

        brain_center = np.mean(all_nodes[gm_node_indices], axis=0)

        # --- 2. Define Ray ---
        ray_origin = brain_center
        target_vec = target_coords - brain_center
        target_dist = np.linalg.norm(target_vec)
        if target_dist < 1e-6:
            raise ValueError(
                f"Target ({target_coords}) is too close to brain center ({brain_center}). "
                "Cannot define projection ray."
            )
        ray_direction = target_vec / target_dist

        # --- 3. Extract Scalp Triangles ---
        scalp_elm_mask = elm_tags == scalp_tag
        scalp_tris_indices = mesh.elm.node_number_list[scalp_elm_mask]

        # Ensure we are looking at triangles (3 columns)
        if scalp_tris_indices.shape[1] != 3:
            tri_types = mesh.elm.elm_type[scalp_elm_mask] == 2
            scalp_tris_indices = scalp_tris_indices[tri_types]

        # Get triangle vertices (SimNIBS nodes array has dummy row at index 0)
        vert0 = all_nodes[scalp_tris_indices[:, 0]]
        vert1 = all_nodes[scalp_tris_indices[:, 1]]
        vert2 = all_nodes[scalp_tris_indices[:, 2]]

        # --- 4. Vectorized Möller–Trumbore Intersection ---
        edge1 = vert1 - vert0
        edge2 = vert2 - vert0

        h = np.cross(ray_direction, edge2)
        a = np.einsum("ij,ij->i", edge1, h)

        # Handle parallel triangles
        epsilon = 1e-7
        valid_a = np.abs(a) > epsilon

        f = np.zeros_like(a)
        f[valid_a] = 1.0 / a[valid_a]

        s = ray_origin - vert0
        u = f * np.einsum("ij,ij->i", s, h)

        q = np.cross(s, edge1)
        v = f * np.einsum("j,ij->i", ray_direction, q)
        t = f * np.einsum("ij,ij->i", edge2, q)

        # Intersection validity conditions
        valid_mask = valid_a & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > epsilon)

        if not np.any(valid_mask):
            log.warning("Ray casting found no intersection. Falling back to nearest scalp node.")
            scalp_node_indices = np.unique(scalp_tris_indices)
            scalp_nodes = all_nodes[scalp_node_indices]
            dists = np.linalg.norm(scalp_nodes - target_coords, axis=1)
            return scalp_nodes[np.argmin(dists)]

        # --- 5. Select Outermost Intersection (max t) ---
        valid_t = t[valid_mask]
        best_t = np.max(valid_t)

        intersection_point = ray_origin + ray_direction * best_t

        return intersection_point

    except Exception as e:
        log.error(f"Error during geometric projection: {e}")
        raise


def compute_default_coil_orientation(mesh_path: Path, scalp_coords: np.ndarray) -> List[float]:
    """
    Computes a default coil handle orientation (pos_ydir) for TMS optimization.

    The handle is oriented 45° from ANTERIOR toward the MEDIAL direction
    (contralateral hemisphere), constrained to the scalp tangent plane.

    - Left hemisphere (X < 0):  Handle points anterior-RIGHT (45° toward midline)
    - Right hemisphere (X > 0): Handle points anterior-LEFT (45° toward midline)

    Args:
        mesh_path: Path to SimNIBS .msh head mesh
        scalp_coords: [x, y, z] coil center position on scalp

    Returns:
        [x, y, z] reference point defining handle direction (for SimNIBS pos_ydir)
    """
    if read_msh is None:
        raise ImportError("SimNIBS not installed. Cannot read mesh.")

    scalp_coords = np.asarray(scalp_coords, dtype=float)

    # --- Load mesh and extract scalp surface ---
    mesh = read_msh(str(mesh_path))
    all_nodes = mesh.nodes[:]
    elm_tags = mesh.elm.tag1

    # SimNIBS v3/v4 tag compatibility
    scalp_tag = 1005 if 1002 in elm_tags else 5
    gm_tag = 1002 if 1002 in elm_tags else 2

    # Get scalp nodes
    scalp_elm_mask = elm_tags == scalp_tag
    scalp_node_indices = np.unique(mesh.elm.node_number_list[scalp_elm_mask])
    scalp_nodes = all_nodes[scalp_node_indices]

    # Get brain center for outward direction reference
    gm_elm_mask = elm_tags == gm_tag
    gm_node_indices = np.unique(mesh.elm.node_number_list[gm_elm_mask])
    brain_center = np.mean(all_nodes[gm_node_indices], axis=0)

    # --- Compute local scalp normal at coil position ---
    distances = np.linalg.norm(scalp_nodes - scalp_coords, axis=1)
    nearest_indices = np.argsort(distances)[:30]
    neighbors = scalp_nodes[nearest_indices]

    # PCA: smallest eigenvector = surface normal
    centered = neighbors - neighbors.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    scalp_normal = vh[2]

    # Ensure normal points OUTWARD (away from brain center)
    if np.dot(scalp_normal, scalp_coords - brain_center) < 0:
        scalp_normal = -scalp_normal

    # --- Compute handle direction in tangent plane ---
    # 1. Start with ANTERIOR direction (+Y in RAS/MNI coordinates)
    anterior = np.array([0.0, 1.0, 0.0])

    # 2. Project anterior onto scalp tangent plane
    anterior_tangent = anterior - np.dot(anterior, scalp_normal) * scalp_normal
    norm = np.linalg.norm(anterior_tangent)

    if norm < 1e-6:
        # Edge case: scalp normal is nearly vertical (top of head)
        # Use -X as fallback anterior reference
        anterior_tangent = np.array([-1.0, 0.0, 0.0])
        anterior_tangent = anterior_tangent - np.dot(anterior_tangent, scalp_normal) * scalp_normal
        norm = np.linalg.norm(anterior_tangent)

    anterior_tangent = anterior_tangent / norm

    # 3. Compute lateral direction (perpendicular to anterior, in tangent plane)
    # Cross product: normal × anterior gives lateral direction
    lateral = np.cross(scalp_normal, anterior_tangent)
    lateral = lateral / np.linalg.norm(lateral)

    # 4. Rotate 45° from anterior toward MEDIAL (contralateral) side
    angle_rad = np.radians(45.0)

    # Determine hemisphere and medial direction:
    # Left hemisphere (X < 0): medial is toward +X (right)
    # Right hemisphere (X > 0): medial is toward -X (left)
    if scalp_coords[0] < 0:
        # Left hemisphere: medial is +X direction
        medial_sign = 1.0 if lateral[0] > 0 else -1.0
    else:
        # Right hemisphere: medial is -X direction
        medial_sign = 1.0 if lateral[0] < 0 else -1.0

    # Combine anterior + medial rotation
    handle_direction = (
        np.cos(angle_rad) * anterior_tangent + np.sin(angle_rad) * medial_sign * lateral
    )

    handle_direction = handle_direction / np.linalg.norm(handle_direction)

    # --- Create reference point for SimNIBS ---
    # pos_ydir expects a point that the handle "looks at"
    orientation_point = scalp_coords + handle_direction * 100.0

    hemisphere = "Left" if scalp_coords[0] < 0 else "Right"
    log.info(
        f"Auto-orientation ({hemisphere} hemisphere): "
        f"45° anterior-medial, handle vector = {handle_direction.round(3)}"
    )

    return orientation_point.tolist()


def calculate_alignment_and_depth(
    streamlines: list,
    e_vectors: list,
    roi_masks: list,
    mesh_path: Path,
    roi_center: list,
):
    """
    Calculates geometric bias metrics (Alignment and Depth) for a bundle.

    Args:
        streamlines: List of streamline coordinates
        e_vectors: List of E-field vectors per streamline
        roi_masks: List of boolean masks for ROI
        mesh_path: Path to SimNIBS mesh
        roi_center: ROI center coordinates

    Returns:
        Tuple of (mean_alignment, depth_mm)
    """
    valid_alignments = []

    for sl, e, mask in zip(streamlines, e_vectors, roi_masks):
        if len(sl) < 2 or len(sl) != len(e) or len(mask) != len(sl):
            continue

        tangents = np.gradient(sl, axis=0)
        norm = np.linalg.norm(tangents, axis=1)[:, None]
        tangents = tangents / (norm + 1e-9)

        e_mag = np.linalg.norm(e, axis=1)[:, None]
        e_norm = e / (e_mag + 1e-9)

        alignment = np.abs(np.sum(e_norm * tangents, axis=1))
        roi_align = alignment[mask]

        if len(roi_align) > 0:
            roi_e = e_mag[mask].flatten()
            e_thresh = np.max(roi_e) * 0.1
            significant_mask = roi_e > e_thresh
            if np.any(significant_mask):
                valid_alignments.extend(roi_align[significant_mask])

    mean_alignment = float(np.mean(valid_alignments)) if valid_alignments else 0.0

    try:
        scalp_point = project_target_to_scalp(mesh_path, np.array(roi_center))
        depth = float(np.linalg.norm(scalp_point - np.array(roi_center)))
    except Exception:
        depth = 0.0

    return mean_alignment, depth


def calculate_alignment_corrected(
    streamlines: List[np.ndarray],
    e_vectors: List[np.ndarray],
    roi_masks: List[np.ndarray],
) -> float:
    """Calculate alignment on midpoint streamlines using midpoint E vectors."""
    valid_alignments: List[float] = []

    for sl, e, mask in zip(streamlines, e_vectors, roi_masks):
        sl_arr = np.asarray(sl, dtype=float)
        e_arr = np.asarray(e, dtype=float)
        mask_arr = np.asarray(mask, dtype=bool)

        if len(sl_arr) < 2 or len(mask_arr) != len(sl_arr):
            continue

        if len(e_arr) == len(sl_arr) + 1:
            e_eval = 0.5 * (e_arr[:-1] + e_arr[1:])
        elif len(e_arr) == len(sl_arr):
            e_eval = e_arr
        else:
            continue

        tangents = np.gradient(sl_arr, axis=0)
        tangent_norm = np.linalg.norm(tangents, axis=1)[:, None]
        tangents = tangents / (tangent_norm + 1e-9)

        e_mag = np.linalg.norm(e_eval, axis=1)[:, None]
        e_norm = e_eval / (e_mag + 1e-9)

        alignment = np.abs(np.sum(e_norm * tangents, axis=1))
        roi_align = alignment[mask_arr]

        if len(roi_align) > 0:
            roi_e = e_mag[mask_arr].flatten()
            e_thresh = np.max(roi_e) * 0.1
            significant_mask = roi_e > e_thresh
            if np.any(significant_mask):
                valid_alignments.extend(roi_align[significant_mask])

    return float(np.mean(valid_alignments)) if valid_alignments else 0.0
