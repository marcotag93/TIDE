"""
Physics Module for TIDE Pipeline
=================================
Implements activating function calculation and threshold estimation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from dipy.tracking.metrics import frenet_serret
from scipy.ndimage import gaussian_filter1d

log = logging.getLogger(__name__)

# Constants
UNIT_DIDT = 1e6  # 1 A/µs = 1e6 A/s
AF_RESAMPLE_STEP_MM = 0.5
AF_BOUNDARY_MODE = "nearest"

# Cross-streamline aggregators of the per-streamline threshold distribution.
# "median_top5" is the committed TIDE statistic; the rest are diagnostic
# alternatives reported alongside it (see cross_streamline_aggregates).
AGGREGATOR_KEYS: Tuple[str, ...] = (
    "median_top5",
    "median_top1",
    "q95",
    "q90",
    "median",
    "mean",
)
AGGREGATOR_LABELS: Dict[str, str] = {
    "median_top5": "Median of Top 5%",
    "median_top1": "Median of Top 1%",
    "q95": "Q0.95",
    "q90": "Q0.90",
    "median": "Median",
    "mean": "Mean",
}
PRIMARY_AGGREGATOR = "median_top5"


def calculate_scalar_map(
    streamlines: List[np.ndarray],
    e_field_vectors: List[np.ndarray],
    mode: str = "af",
    smooth_sigma: Optional[float] = None,
    target_smooth_length_mm: float = 2.5,
    signed: bool = True,
    indices: Optional[np.ndarray] = None,
) -> Union[
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray],
]:
    """
    Calculates scalar values along streamlines.

    If mode='af', computes the Activating Function (gradient term):
       AF = d(E·T)/ds

    AF is evaluated after 0.5 mm uniform arc-length resampling and physical
    Gaussian smoothing. E_parallel retains point-based projection with an
    adaptive smoothing scale for the streamline geometry.

    Args:
        streamlines: List of streamline coordinates
        e_field_vectors: List of E-field vectors per streamline
        mode: 'af' for activating function, 'e_parallel' for parallel E-field
        smooth_sigma: Gaussian smoothing sigma in points. If provided, applied
            uniformly to every streamline (override for testing/debug). If None,
            the physical smoothing length is converted to the sampling support.
        target_smooth_length_mm: Target physical smoothing length in mm,
            used to derive sigma when ``smooth_sigma`` is None.
        signed: If True (default), return signed scalars preserving polarity
            (depolarising vs hyperpolarising for AF; field direction for E_parallel).
            If False, return absolute magnitudes. Consumers that require magnitude
            (e.g., thresholding) should apply ``np.abs`` explicitly.
        indices: Optional identifier array parallel to ``streamlines`` (e.g.
            original streamline ids). When provided, the identifiers of the
            surviving streamlines are returned as a 4th element so callers can
            keep per-streamline weights aligned after drops (audit C-002).

    Returns:
        ``(midpoint_streamlines, scalar_values, segment_lengths)``, or, when
        ``indices`` is provided, ``(..., surviving_indices)``.
    """
    if mode == "af":
        return _calculate_af_map(
            streamlines,
            e_field_vectors,
            smooth_sigma=smooth_sigma,
            target_smooth_length_mm=target_smooth_length_mm,
            signed=signed,
            indices=indices,
        )

    track_indices = indices is not None
    if track_indices:
        indices = np.asarray(indices)
    # Input validation - accept list or any array-like sequence (e.g., DIPY's ArraySequence)
    try:
        len(streamlines)
        len(e_field_vectors)
    except TypeError:
        raise TypeError("streamlines and e_field_vectors must be iterable sequences")
    if len(streamlines) != len(e_field_vectors):
        raise ValueError(
            f"Mismatch: {len(streamlines)} streamlines but {len(e_field_vectors)} E-field vectors"
        )
    if mode != "e_parallel":
        raise ValueError(f"Invalid mode '{mode}'. Must be 'af' or 'e_parallel'.")

    if len(streamlines) == 0:
        log.warning("Empty streamlines list provided")
        if track_indices:
            return [], [], [], np.array([], dtype=int)
        return [], [], []

    log.debug(f"Computing {mode.upper()} for {len(streamlines)} streamlines")

    fallback_sigma = 3.0  # Used when a streamline step size cannot be estimated.

    final_streamlines_list = []
    final_scalars_list = []
    final_distances_list = []
    final_indices_list = []

    skipped_short = 0
    skipped_frenet = 0
    skipped_mismatch = 0

    for i, s_points in enumerate(streamlines):
        if len(s_points) < 4:
            skipped_short += 1
            continue

        # Per-streamline adaptive sigma: derive from this streamline's own
        # mean step size so the physical smoothing length stays uniform
        # across streamlines with heterogeneous resolution.
        if smooth_sigma is None:
            steps = np.linalg.norm(np.diff(s_points, axis=0), axis=1)
            mean_step_mm = float(np.mean(steps)) if steps.size > 0 else 0.0
            sigma_i = target_smooth_length_mm / mean_step_mm if mean_step_mm > 0 else fallback_sigma
        else:
            sigma_i = smooth_sigma

        # Smooth geometry for stable tangent estimation.
        s_smooth = gaussian_filter1d(s_points, sigma=sigma_i, axis=0, mode="nearest")

        try:
            # Frenet frame from smoothed points; its tangent defines the projection.
            T, _, _, _, _ = frenet_serret(s_smooth)
        except Exception as e:
            skipped_frenet += 1
            log.debug(f"Frenet-Serret failed for streamline {i}: {e}")
            continue

        E = e_field_vectors[i]

        if len(E) != len(T):
            skipped_mismatch += 1
            log.debug(f"Length mismatch for streamline {i}: E={len(E)}, T={len(T)}")
            continue

        # E_parallel: project E onto tangent
        s_e_parallel = np.sum(E * T, axis=1)

        # Segment lengths
        distances_mm = np.linalg.norm(np.diff(s_smooth, axis=0), axis=1)

        s_e_parallel_mid = (s_e_parallel[:-1] + s_e_parallel[1:]) / 2
        val_to_store = s_e_parallel_mid if signed else np.abs(s_e_parallel_mid)

        # Store midpoints for visualization
        s_mid_points = (s_points[:-1] + s_points[1:]) / 2

        final_scalars_list.append(val_to_store)
        final_distances_list.append(distances_mm)
        final_streamlines_list.append(s_mid_points)
        if track_indices:
            final_indices_list.append(indices[i])

    # Log summary of skipped streamlines
    total_skipped = skipped_short + skipped_frenet + skipped_mismatch
    if total_skipped > 0:
        log.debug(
            f"Skipped {total_skipped} streamlines: "
            f"{skipped_short} too short, {skipped_frenet} Frenet errors, {skipped_mismatch} length mismatch"
        )
    log.debug(f"Processed {len(final_scalars_list)}/{len(streamlines)} streamlines successfully")

    if track_indices:
        surviving_indices = np.array(final_indices_list, dtype=int)
        return final_streamlines_list, final_scalars_list, final_distances_list, surviving_indices

    return final_streamlines_list, final_scalars_list, final_distances_list


def _calculate_af_map(
    streamlines: List[np.ndarray],
    e_field_vectors: List[np.ndarray],
    smooth_sigma: Optional[float],
    target_smooth_length_mm: float,
    signed: bool,
    indices: Optional[np.ndarray],
) -> Union[
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray],
]:
    track_indices = indices is not None
    if track_indices:
        indices = np.asarray(indices)

    try:
        len(streamlines)
        len(e_field_vectors)
    except TypeError:
        raise TypeError("streamlines and e_field_vectors must be iterable sequences")
    if len(streamlines) != len(e_field_vectors):
        raise ValueError(
            f"Mismatch: {len(streamlines)} streamlines but "
            f"{len(e_field_vectors)} E-field vectors"
        )

    if len(streamlines) == 0:
        log.warning("Empty streamlines list provided")
        if track_indices:
            return [], [], [], np.array([], dtype=int)
        return [], [], []

    log.debug(f"Computing AF for {len(streamlines)} streamlines on arc-length support")

    final_streamlines_list = []
    final_scalars_list = []
    final_distances_list = []
    final_indices_list = []

    skipped_short = 0
    skipped_degenerate = 0
    skipped_frenet = 0
    skipped_mismatch = 0

    for i, s_points in enumerate(streamlines):
        if len(s_points) < 4:
            skipped_short += 1
            continue

        points = np.asarray(s_points)
        e_vectors = np.asarray(e_field_vectors[i])
        if len(e_vectors) != len(points):
            skipped_mismatch += 1
            log.debug(
                "Length mismatch for streamline %d: E=%d, points=%d",
                i,
                len(e_vectors),
                len(points),
            )
            continue

        original_distances_mm = np.linalg.norm(np.diff(points, axis=0), axis=1)
        original_s = np.concatenate(([0.0], np.cumsum(original_distances_mm)))
        unique = np.concatenate(([True], np.diff(original_s) > 0.0))
        unique_s = original_s[unique]
        if len(unique_s) < 4:
            skipped_degenerate += 1
            continue

        total_length_mm = unique_s[-1]
        interval_count = max(3, int(np.ceil(total_length_mm / AF_RESAMPLE_STEP_MM)))
        uniform_s = np.linspace(0.0, total_length_mm, interval_count + 1)
        uniform_step_mm = uniform_s[1] - uniform_s[0]

        unique_points = points[unique]
        unique_e_vectors = e_vectors[unique]
        resampled_points = np.column_stack(
            [np.interp(uniform_s, unique_s, unique_points[:, axis]) for axis in range(3)]
        )
        resampled_e_vectors = np.column_stack(
            [np.interp(uniform_s, unique_s, unique_e_vectors[:, axis]) for axis in range(3)]
        )

        sigma_i = (
            target_smooth_length_mm / uniform_step_mm if smooth_sigma is None else smooth_sigma
        )
        smooth_points = gaussian_filter1d(
            resampled_points,
            sigma=sigma_i,
            axis=0,
            mode=AF_BOUNDARY_MODE,
        )
        smooth_e_vectors = gaussian_filter1d(
            resampled_e_vectors,
            sigma=sigma_i,
            axis=0,
            mode=AF_BOUNDARY_MODE,
        )

        try:
            tangents, _, _, _, _ = frenet_serret(smooth_points)
        except Exception as e:
            skipped_frenet += 1
            log.debug(f"Frenet-Serret failed for streamline {i}: {e}")
            continue

        e_parallel = np.sum(smooth_e_vectors * tangents, axis=1)
        smooth_distances_m = np.linalg.norm(np.diff(smooth_points, axis=0), axis=1) / 1000.0
        uniform_af = np.divide(
            np.diff(e_parallel),
            smooth_distances_m,
            out=np.zeros_like(smooth_distances_m),
            where=smooth_distances_m > np.finfo(float).eps,
        )

        uniform_mid_s = (uniform_s[:-1] + uniform_s[1:]) / 2.0
        original_mid_s = (original_s[:-1] + original_s[1:]) / 2.0
        mapped_af = np.interp(original_mid_s, uniform_mid_s, uniform_af)
        value_to_store = mapped_af if signed else np.abs(mapped_af)

        final_streamlines_list.append((points[:-1] + points[1:]) / 2.0)
        final_scalars_list.append(value_to_store)
        final_distances_list.append(original_distances_mm)
        if track_indices:
            final_indices_list.append(indices[i])

    total_skipped = skipped_short + skipped_degenerate + skipped_frenet + skipped_mismatch
    if total_skipped > 0:
        log.debug(
            f"Skipped {total_skipped} streamlines: "
            f"{skipped_short} too short, "
            f"{skipped_degenerate} fewer than four distinct arc-length positions, "
            f"{skipped_frenet} Frenet errors, "
            f"{skipped_mismatch} length mismatch"
        )
    log.debug(f"Processed {len(final_scalars_list)}/{len(streamlines)} streamlines successfully")

    if track_indices:
        surviving_indices = np.array(final_indices_list, dtype=int)
        return final_streamlines_list, final_scalars_list, final_distances_list, surviving_indices

    return final_streamlines_list, final_scalars_list, final_distances_list


def get_max_contiguous_threshold(
    values: np.ndarray, lengths: np.ndarray, target_length: float
) -> float:
    """
    Finds the highest threshold T such that there exists at least one
    contiguous segment of 'target_length' where all values >= T.

    Args:
        values: Array of scalar values (e.g., AF)
        lengths: Array of corresponding segment lengths (mm)
        target_length: Required contiguous length (mm)

    Returns:
        Maximum threshold value
    """
    if len(values) == 0 or np.sum(lengths) < target_length:
        return 0.0

    cum_len = np.cumsum(lengths)
    total_len = cum_len[-1]

    if total_len < target_length:
        return 0.0

    max_thresh_found = 0.0

    # Sliding window approach
    start_idx = 0
    for end_idx in range(len(lengths)):
        len_start = cum_len[start_idx - 1] if start_idx > 0 else 0.0
        current_window_len = cum_len[end_idx] - len_start

        while current_window_len >= target_length:
            window_min = np.min(values[start_idx : end_idx + 1])

            if window_min > max_thresh_found:
                max_thresh_found = window_min

            start_idx += 1
            if start_idx > end_idx:
                break
            len_start = cum_len[start_idx - 1] if start_idx > 0 else 0.0
            current_window_len = cum_len[end_idx] - len_start

    return max_thresh_found


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """
    Weighted percentile of ``values``.

    Args:
        values: Sample values.
        weights: Non-negative weights, one per value.
        percentile: Percentile in [0, 100].

    Returns:
        The value at the requested weighted percentile, or 0.0 if the input is
        empty or all weights are zero.
    """
    if len(values) == 0:
        return 0.0

    w_sum = np.sum(weights)
    if w_sum == 0:
        return 0.0

    order = np.argsort(values)
    sorted_vals = values[order]
    cum_weights = np.cumsum(weights[order]) / w_sum

    cutoff_idx = np.searchsorted(cum_weights, percentile / 100.0)
    if cutoff_idx >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[cutoff_idx]


def median_of_top_percentile(values: np.ndarray, pct: float = 95.0) -> float:
    """
    Median of the values at or above the ``pct`` percentile (top-tail aggregator).

    Single source of the unweighted "median of the top 5%" statistic used for the
    per-streamline threshold distribution in both bundle analysis and M1
    validation. Returns 0.0 for an empty input or an empty top tail.
    """
    if values.size == 0:
        return 0.0
    cutoff = np.percentile(values, pct)
    top = values[values >= cutoff]
    return float(np.median(top)) if top.size else 0.0


def weighted_median_of_top_percentile(
    values: np.ndarray, weights: np.ndarray, pct: float = 95.0
) -> float:
    """
    Weighted counterpart of :func:`median_of_top_percentile`.

    The cutoff and the median of the surviving tail are both taken with
    :func:`weighted_percentile`, matching the weighted "median of the top 5%"
    used for the bundle metric. Returns 0.0 for an empty input or top tail.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    cutoff = weighted_percentile(values, weights, pct)
    top_mask = values >= cutoff
    if not np.any(top_mask):
        return 0.0
    return float(weighted_percentile(values[top_mask], np.asarray(weights)[top_mask], 50.0))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted arithmetic mean; 0.0 for an empty input or zero total weight."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.0
    w_sum = float(np.sum(weights))
    if w_sum == 0.0:
        return 0.0
    return float(np.dot(values, weights) / w_sum)


def cross_streamline_aggregates(
    values: np.ndarray, weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Every cross-streamline aggregate of a per-streamline threshold distribution.

    ``median_top5`` is the aggregator TIDE commits to and the only one that
    feeds the calibration ratio and the reported dose. The remaining entries are
    diagnostic alternatives evaluated on the same distribution, so the
    sensitivity of AF_CST/AF_target to the aggregation rule is readable without
    re-running the field solve.

    With ``weights=None`` the unweighted (NumPy quantile) definitions are used;
    with weights, every entry uses its weighted counterpart. Keys are ordered as
    in :data:`AGGREGATOR_KEYS`.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {key: 0.0 for key in AGGREGATOR_KEYS}

    if weights is None:
        return {
            "median_top5": median_of_top_percentile(values, 95.0),
            "median_top1": median_of_top_percentile(values, 99.0),
            "q95": float(np.percentile(values, 95.0)),
            "q90": float(np.percentile(values, 90.0)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
        }

    weights = np.asarray(weights, dtype=float)
    return {
        "median_top5": weighted_median_of_top_percentile(values, weights, 95.0),
        "median_top1": weighted_median_of_top_percentile(values, weights, 99.0),
        "q95": float(weighted_percentile(values, weights, 95.0)),
        "q90": float(weighted_percentile(values, weights, 90.0)),
        "median": float(weighted_percentile(values, weights, 50.0)),
        "mean": weighted_mean(values, weights),
    }


def get_robust_threshold(
    values: np.ndarray, lengths: np.ndarray, min_activation_length_mm: float = 3.0
) -> float:
    """
    Calculates a robust threshold value V such that 'min_activation_length_mm'
    of the bundle has a value >= V.

    Args:
        values: Array of scalar values (e.g., AF)
        lengths: Array of corresponding segment lengths (mm)
        min_activation_length_mm: Target length of tissue to activate

    Returns:
        The scalar value threshold
    """
    if len(values) == 0 or len(lengths) == 0:
        return 0.0

    # Sort values descending
    sort_idx = np.argsort(values)[::-1]
    sorted_vals = values[sort_idx]
    sorted_lens = lengths[sort_idx]

    cum_len = np.cumsum(sorted_lens)

    if cum_len[-1] < min_activation_length_mm:
        return sorted_vals[-1]

    idx = np.searchsorted(cum_len, min_activation_length_mm)

    if idx < len(sorted_vals):
        return sorted_vals[idx]
    else:
        return sorted_vals[-1]


def estimate_rmt_threshold(
    cst_af_roi: np.ndarray,
    cst_len_roi: np.ndarray,
    target_af_roi: np.ndarray,
    target_len_roi: np.ndarray,
    measured_rmt_mso: float,
    didt_max: float,
    activation_length_mm: float = 3.0,
) -> Dict[str, float]:
    """
    Estimates the target intensity using Length-Based Estimation.

    Args:
        cst_af_roi: AF values in CST ROI (1 A/µs)
        cst_len_roi: Segment lengths in CST ROI (mm)
        target_af_roi: AF values in Target ROI (1 A/µs)
        target_len_roi: Segment lengths in Target ROI (mm)
        measured_rmt_mso: Patient RMT percentage
        didt_max: Device max dI/dt
        activation_length_mm: Required activation length (mm)

    Returns:
        Dictionary with estimation results
    """
    if len(cst_af_roi) == 0:
        raise ValueError("No AF values in CST ROI.")
    if len(target_af_roi) == 0:
        raise ValueError("No AF values in Target ROI.")

    # Magnitude aggregation: AF polarity is preserved upstream; thresholding
    # requires absolute activation strength.
    af_cst_robust = get_robust_threshold(np.abs(cst_af_roi), cst_len_roi, activation_length_mm)
    af_target_robust = get_robust_threshold(
        np.abs(target_af_roi), target_len_roi, activation_length_mm
    )

    # Calculate absolute intensity at RMT
    intensity_rmt = didt_max * (measured_rmt_mso / 100.0)

    # Calculate biological threshold
    biological_thresh = intensity_rmt * (af_cst_robust / UNIT_DIDT)

    # Calculate target efficiency
    eff_target = af_target_robust / UNIT_DIDT

    if eff_target <= 1e-12:
        raise ValueError("Target efficiency is zero. Cannot stimulate this target.")

    # Calculate required target intensity
    intensity_target = biological_thresh / eff_target

    # Convert to intensity (% MSO)
    estimated_intensity = (intensity_target / didt_max) * 100.0

    return {
        "estimated_intensity": estimated_intensity,
        "biological_thresh": biological_thresh,
        "intensity_rmt": intensity_rmt,
        "intensity_target": intensity_target,
        "af_cst_99": af_cst_robust,
        "af_target_99": af_target_robust,
        "cst_efficiency": af_cst_robust / UNIT_DIDT,
        "target_efficiency": eff_target,
    }


def estimate_rmt_threshold_contiguous(
    target_af_list: list,
    target_len_list: list,
    roi_masks: list,
    measured_rmt_mso: float,
    didt_max: float,
    activation_length_mm: float = 4.0,
    percentile_streamlines: float = 10.0,
):
    """
    Estimates intensity required to activate a continuous segment on X% of streamlines.

    Args:
        target_af_list: List of AF arrays per streamline
        target_len_list: List of length arrays per streamline
        roi_masks: List of boolean masks from tractography.get_roi_masks
        measured_rmt_mso: Measured RMT percentage
        didt_max: Device max dI/dt
        activation_length_mm: Required contiguous length (mm)
        percentile_streamlines: Percentage of fibers to activate

    Returns:
        Target AF threshold value
    """
    streamline_thresholds = []

    for af, length, mask in zip(target_af_list, target_len_list, roi_masks):
        # Apply ROI mask
        if len(mask) == len(af):
            af_roi = af[mask]
            len_roi = length[mask]
        elif len(mask) == len(af) + 1:
            af_roi = af[mask[:-1]]
            len_roi = length[mask[:-1]]
        else:
            continue

        if len(af_roi) == 0:
            streamline_thresholds.append(0.0)
            continue

        # AF is signed upstream; thresholding operates on magnitude.
        s_thresh = get_max_contiguous_threshold(np.abs(af_roi), len_roi, activation_length_mm)
        streamline_thresholds.append(s_thresh)

    streamline_thresholds = np.array(streamline_thresholds)

    # Determine population threshold
    target_percentile = 100.0 - percentile_streamlines
    if len(streamline_thresholds) > 0:
        af_target_robust = np.percentile(streamline_thresholds, target_percentile)
    else:
        af_target_robust = 0.0

    return af_target_robust
