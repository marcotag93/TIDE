import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

try:
    import nibabel as nib  # noqa: F401
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import load_tractogram, save_tractogram  # noqa: F401
    from dipy.tracking.metrics import frenet_serret
    from sklearn.cluster import KMeans
except ImportError:
    raise ImportError("Missing dependencies: nibabel, dipy, or sklearn.")

log = logging.getLogger(__name__)


def load_tract(trk_path: Path, anat_ref: Path) -> StatefulTractogram:
    """
    Loads a tractogram in RASMM space.

    Args:
        trk_path: Path to the tractography file (.trk)
        anat_ref: Path to the reference anatomy NIfTI file

    Returns:
        StatefulTractogram with streamlines in RASMM space
    """
    if not trk_path.exists():
        raise FileNotFoundError(f"Tractogram not found: {trk_path}")
    if not anat_ref.exists():
        raise FileNotFoundError(f"Anatomy reference not found: {anat_ref}")

    try:
        sft = load_tractogram(str(trk_path), str(anat_ref), to_space=Space.RASMM)
        return sft
    except Exception as e:
        raise RuntimeError(f"Failed to load tractogram {trk_path}: {e}")


def get_roi_masks(
    streamlines: List[np.ndarray],
    roi_size_mm: float,
    target_coords: Optional[Union[List[float], np.ndarray]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generates boolean masks for Points and Segments indicating which parts
    of the streamline are within the ROI.

    Args:
        streamlines: List of streamline coordinate arrays (Nx3)
        roi_size_mm: Radius of spherical ROI in mm
        target_coords: Optional ROI center coordinates [x, y, z]

    Returns:
        Tuple of (point_masks, segment_masks)

    Raises:
        TypeError: If streamlines is not a list
        ValueError: If roi_size_mm is invalid
    """
    # Input validation - accept list or any array-like sequence
    try:
        len(streamlines)
    except TypeError:
        raise TypeError(
            f"streamlines must be an iterable sequence, got {type(streamlines).__name__}"
        )
    if roi_size_mm <= 0:
        raise ValueError(f"roi_size_mm must be positive, got {roi_size_mm}")

    if len(streamlines) == 0:
        log.debug("Empty streamlines list provided to get_roi_masks")
        return [], []

    point_masks = []
    segment_masks = []

    target_arr = np.asarray(target_coords, dtype=float) if target_coords is not None else None

    if target_arr is not None and target_arr.shape != (3,):
        raise ValueError(f"target_coords must be shape (3,), got {target_arr.shape}")

    for s_points in streamlines:
        if len(s_points) < 1:
            point_masks.append(np.array([], dtype=bool))
            segment_masks.append(np.array([], dtype=bool))
            continue

        if target_arr is not None:
            # Distance to specific target coordinate
            dists_to_target = np.linalg.norm(s_points - target_arr, axis=1)
            p_mask = dists_to_target < roi_size_mm
            s_mask = (
                dists_to_target[:-1] < roi_size_mm
                if len(s_points) > 1
                else np.array([], dtype=bool)
            )
        else:
            # Distance to tips
            dist_to_start_tip = np.linalg.norm(s_points - s_points[0], axis=1)
            dist_to_end_tip = np.linalg.norm(s_points - s_points[-1], axis=1)

            p_mask = np.logical_or(dist_to_start_tip < roi_size_mm, dist_to_end_tip < roi_size_mm)

            if len(s_points) > 1:
                mask_s_start = dist_to_start_tip[:-1] < roi_size_mm
                mask_s_end = dist_to_end_tip[:-1] < roi_size_mm
                s_mask = np.logical_or(mask_s_start, mask_s_end)
            else:
                s_mask = np.array([], dtype=bool)

        point_masks.append(p_mask)
        segment_masks.append(s_mask)

    return point_masks, segment_masks


def get_data_in_roi(
    streamlines: List[np.ndarray],
    values: List[np.ndarray],
    roi_size_mm: float,
    target_coords: Optional[Union[List[float], np.ndarray]] = None,
    lengths: Optional[List[np.ndarray]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts specific scalar values (and optionally lengths) for points/segments within the ROI.

    Args:
        streamlines: List of streamline coordinates.
        values: List of scalar arrays (e.g. AF) per streamline.
        roi_size_mm: Radius of the ROI.
        target_coords: Optional coordinate center for the ROI.
        lengths: Optional List of length arrays (segment lengths) per streamline.

    Returns:
        If lengths is None: flattened np.array of values.
        If lengths is provided: Tuple (flattened_values, flattened_lengths)
    """
    point_masks, segment_masks = get_roi_masks(streamlines, roi_size_mm, target_coords)
    all_values_in_roi = []
    all_lengths_in_roi = []

    for i, (p_mask, s_mask) in enumerate(zip(point_masks, segment_masks)):
        current_vals = values[i].flatten()

        # Determine if values correspond to points or segments
        if len(current_vals) == len(p_mask):
            mask_to_use = p_mask
        elif len(current_vals) == len(s_mask):
            mask_to_use = s_mask
        else:
            continue

        all_values_in_roi.extend(current_vals[mask_to_use])

        if lengths is not None:
            current_lens = lengths[i].flatten()
            # Lengths always correspond to segments
            if len(current_lens) == len(s_mask):
                # If we are using a point mask for values, we might have a mismatch
                # (N points vs N-1 lengths). Usually AF is calculated on segments or midpoints (N-1).
                # We assume here values and lengths are aligned (both N-1).
                if len(current_vals) == len(current_lens):
                    all_lengths_in_roi.extend(current_lens[mask_to_use])

    if lengths is not None:
        return np.array(all_values_in_roi), np.array(all_lengths_in_roi)

    return np.array(all_values_in_roi)


def validate_curvature(trk_path: Path, anat_path: Path, roi_size_mm: float) -> Dict[str, float]:
    """Calculates curvature statistics for the tractogram endpoints."""
    sft = load_tract(trk_path, anat_path)
    streamlines = sft.streamlines

    all_curvatures = []
    for s in streamlines:
        if len(s) < 3:
            continue
        try:
            curv_values = frenet_serret(s)[3]  # k
            all_curvatures.append(curv_values)
        except Exception:
            continue

    # Note: Validate Curvature is usually a generic check, so we typically
    # check *both* ends (target_coords=None) to ensure the whole bundle is healthy.
    roi_curvatures = get_data_in_roi(streamlines, all_curvatures, roi_size_mm, target_coords=None)

    valid_curv = roi_curvatures[np.isfinite(roi_curvatures)]

    if len(valid_curv) == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}

    stats = {
        "mean": float(np.mean(valid_curv)),
        "median": float(np.median(valid_curv)),
        "std": float(np.std(valid_curv)),
        "p95": float(np.percentile(valid_curv, 95)),
        "max": float(np.max(valid_curv)),
    }
    return stats


def extract_grid_endpoints(
    trk_path: Path,
    anat_path: Path,
    step_mm: float,
    cortex_thickness_mm: float,
    target_center: Optional[List[float]] = None,
    search_radius: float = 20.0,
) -> List[List[float]]:
    """
    Extracts cortical endpoints for Grid Search.

    Loads tractogram in RASMM space and extracts streamline endpoints
    from the cortical region for grid-based targeting.

    Args:
        trk_path: Path to tractography file
        anat_path: Path to reference T1w anatomy
        step_mm: Grid spacing in mm
        cortex_thickness_mm: Depth of cortical layer to sample
        target_center: Optional center point for focused search
        search_radius: Radius around target_center to search (if provided)

    Returns:
        List of [x, y, z] coordinates in RASMM space
    """
    log.info(f"Extracting grid endpoints from {trk_path.name}")
    log.debug(f"Using reference anatomy: {anat_path.name}")

    sft = load_tract(trk_path, anat_path)
    streamlines = sft.streamlines
    if not streamlines:
        log.warning("No streamlines found in tractogram")
        return []

    start_pts = np.array([s[0] for s in streamlines])
    end_pts = np.array([s[-1] for s in streamlines])
    all_pts = np.vstack((start_pts, end_pts))

    log.debug(f"Extracted {len(start_pts)} start points and {len(end_pts)} end points")
    log.debug(
        f"Coordinate range: X=[{all_pts[:, 0].min():.1f}, {all_pts[:, 0].max():.1f}], "
        f"Y=[{all_pts[:, 1].min():.1f}, {all_pts[:, 1].max():.1f}], "
        f"Z=[{all_pts[:, 2].min():.1f}, {all_pts[:, 2].max():.1f}]"
    )

    if target_center:
        center = np.array(target_center)
        dists = np.linalg.norm(all_pts - center, axis=1)
        mask_prox = dists <= search_radius
        candidates = all_pts[mask_prox]
    else:
        if len(all_pts) < 2:
            candidates = all_pts
        else:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(all_pts)
            centers = kmeans.cluster_centers_
            target_label = 0 if centers[0][2] > centers[1][2] else 1
            candidates = all_pts[kmeans.labels_ == target_label]

    if len(candidates) == 0:
        log.warning("No candidate points found after filtering")
        return []

    # Filter to cortical depth
    z_coords = candidates[:, 2]
    top_peak = np.percentile(z_coords, 98)
    z_cutoff = top_peak - cortex_thickness_mm
    filtered = candidates[z_coords >= z_cutoff]

    log.debug(f"Filtered to {len(filtered)} points within cortex depth")
    log.debug(f"Z-range after filtering: [{filtered[:, 2].min():.1f}, {filtered[:, 2].max():.1f}]")

    # Round to grid and get unique points
    rounded = np.round(filtered / step_mm) * step_mm
    unique_targets = np.unique(rounded, axis=0)

    log.info(f"Generated {len(unique_targets)} unique grid points")
    if len(unique_targets) > 0:
        # Log first few points and centroid for verification
        centroid = unique_targets.mean(axis=0)
        log.debug(f"Grid centroid (RAS): [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
        log.debug("Sample grid points (first 3):")
        for i, pt in enumerate(unique_targets[:3]):
            log.debug(f"  Point {i}: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")

    return unique_targets.tolist()


def get_bundle_cortical_medoid(
    trk_path: Path,
    anat_path: Path,
    cortex_thickness_mm: float = 4.0,
    percentile_depth: float = 98.0,
    reference_coord: Optional[List[float]] = None,
) -> np.ndarray:
    """Calculates the 'Cortical Medoid' of a bundle."""
    log.info(f"--- Processing Medoid for {trk_path.name} ---")
    sft = load_tract(trk_path, anat_path)
    streamlines = sft.streamlines
    if not streamlines:
        raise ValueError("Tractogram is empty.")

    start_pts = np.array([s[0] for s in streamlines])
    end_pts = np.array([s[-1] for s in streamlines])
    all_pts = np.vstack((start_pts, end_pts))

    if len(all_pts) < 2:
        return all_pts[0]

    log.info("Clustering endpoints (K=2) to separate bundle ends...")
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(all_pts)
    centers = kmeans.cluster_centers_

    if reference_coord is not None:
        ref = np.array(reference_coord)
        dist0 = np.linalg.norm(centers[0] - ref)
        dist1 = np.linalg.norm(centers[1] - ref)
        target_label = 0 if dist0 < dist1 else 1
        log.info(f"Using Spatial Hint: {reference_coord}")
    else:
        target_label = 0 if centers[0][2] > centers[1][2] else 1
        log.info("No reference coord provided. Auto-selecting superior (highest Z) end.")

    cortical_candidates = all_pts[kmeans.labels_ == target_label]
    if len(cortical_candidates) == 0:
        cortical_candidates = all_pts

    z_coords = cortical_candidates[:, 2]
    top_peak = np.percentile(z_coords, percentile_depth)
    z_cutoff = top_peak - cortex_thickness_mm
    filtered_pts = cortical_candidates[z_coords >= z_cutoff]

    if len(filtered_pts) == 0:
        filtered_pts = cortical_candidates

    if len(filtered_pts) <= 2:
        best_medoid = filtered_pts[0]
    else:
        dists = cdist(filtered_pts, filtered_pts, metric="euclidean")
        sum_dists = dists.sum(axis=1)
        medoid_idx = np.argmin(sum_dists)
        best_medoid = filtered_pts[medoid_idx]

    return best_medoid


def filter_by_angular_deviation(
    streamlines: List[np.ndarray],
    e_field_vectors: Optional[List[np.ndarray]] = None,
    max_angle_deg: float = 80.0,
    roi_center: Optional[Union[List[float], np.ndarray]] = None,
    roi_radius: float = 40.0,
    indices: Optional[np.ndarray] = None,
) -> Union[
    Tuple[List[np.ndarray], int],
    Tuple[List[np.ndarray], List[np.ndarray], int],
    Tuple[List[np.ndarray], List[np.ndarray], int, np.ndarray],
]:
    """
    Filters out streamlines with angular deviations exceeding a threshold.

    Checks the maximum angle between consecutive tangent vectors within the
    ROI region. Streamlines where any consecutive pair of tangent vectors
    deviates by more than max_angle_deg are excluded (likely tractography
    artifacts, e.g. streamlines curling back at cortical endpoints).

    Args:
        streamlines: List of streamline coordinate arrays (Nx3).
        e_field_vectors: Optional parallel list of E-field vector arrays.
            If provided, filtered in sync with streamlines.
        max_angle_deg: Maximum allowed angular deviation in degrees.
            Streamlines with any consecutive tangent angle > this are removed.
        roi_center: If provided, only check angular deviation within this
            spherical ROI. If None, check entire streamline.
        roi_radius: Radius of the ROI sphere in mm.
        indices: Optional identifier array parallel to ``streamlines`` (e.g.
            original streamline ids). When provided, it is filtered in sync and
            returned as a trailing element so per-streamline weights stay
            aligned after removals (audit C-002). Requires ``e_field_vectors``.

    Returns:
        If e_field_vectors is None:
            (filtered_streamlines, n_removed)
        If e_field_vectors is provided:
            (filtered_streamlines, filtered_e_vectors, n_removed)
        If indices is also provided:
            (filtered_streamlines, filtered_e_vectors, n_removed, filtered_indices)
    """
    track_indices = indices is not None
    if track_indices:
        indices = np.asarray(indices)

    if max_angle_deg <= 0 or max_angle_deg >= 180:
        log.debug(f"Angular filter disabled (max_angle_deg={max_angle_deg})")
        if track_indices:
            return streamlines, e_field_vectors, 0, indices
        if e_field_vectors is not None:
            return streamlines, e_field_vectors, 0
        return streamlines, 0

    cos_threshold = np.cos(np.radians(max_angle_deg))
    roi_arr = np.asarray(roi_center, dtype=float) if roi_center is not None else None

    filtered_sl = []
    filtered_ev = [] if e_field_vectors is not None else None
    filtered_idx = [] if track_indices else None
    n_removed = 0

    for i, sl in enumerate(streamlines):
        if len(sl) < 4:
            # Streamlines with fewer than 4 points cannot contribute a valid
            # AF estimate (matches the physics cutoff in calculate_scalar_map)
            # and are typically tractography stubs. Drop them rather than
            # bypass the quality filter.
            n_removed += 1
            continue

        # Compute tangent vectors (unnormalized differences)
        diffs = np.diff(sl, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tangents = diffs / norms

        # Cosine of angle between consecutive tangents
        cos_angles = np.sum(tangents[:-1] * tangents[1:], axis=1)

        if roi_arr is not None:
            # Only check within ROI: use midpoints of consecutive tangent pairs
            # The midpoint between point[j] and point[j+2] (where cos_angles[j] is defined)
            midpoints = (sl[:-2] + sl[2:]) / 2.0
            dists = np.linalg.norm(midpoints - roi_arr, axis=1)
            in_roi = dists <= roi_radius

            if not np.any(in_roi):
                # No points in ROI — keep streamline (no evidence of artifact)
                filtered_sl.append(sl)
                if filtered_ev is not None:
                    filtered_ev.append(e_field_vectors[i])
                if filtered_idx is not None:
                    filtered_idx.append(indices[i])
                continue

            cos_in_roi = cos_angles[in_roi]
        else:
            cos_in_roi = cos_angles

        # Check if any angle exceeds threshold
        # cos(angle) < cos(threshold) means angle > threshold (cosine is decreasing)
        if np.any(cos_in_roi < cos_threshold):
            n_removed += 1
            continue

        filtered_sl.append(sl)
        if filtered_ev is not None:
            filtered_ev.append(e_field_vectors[i])
        if filtered_idx is not None:
            filtered_idx.append(indices[i])

    if n_removed > 0:
        log.info(
            f"Angular deviation filter: removed {n_removed}/{len(streamlines)} "
            f"streamlines (threshold: {max_angle_deg}°)"
        )

    if track_indices:
        return filtered_sl, filtered_ev, n_removed, np.array(filtered_idx, dtype=int)
    if filtered_ev is not None:
        return filtered_sl, filtered_ev, n_removed
    return filtered_sl, n_removed
