"""
Unified TMS Target Intensity Estimation Module
==============================================
Estimates target intensity using the Contiguous Activating Function method.

Supports 4 modes:
1. Baseline: Spherical ROI only
2. Surface-Constrained (GWI): Restricted to grey-white interface
3. Weighted: Uses SIFT2/TOM streamline weights
4. Hybrid: Combines GWI and weights
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
from nibabel.streamlines.trk import TrkFile
from scipy.spatial import cKDTree

from tide.core.physics import (
    AGGREGATOR_KEYS,
    cross_streamline_aggregates,
    get_max_contiguous_threshold,
    median_of_top_percentile,
    weighted_percentile,
)

log = logging.getLogger(__name__)

# Definition of the %MSO scale: no stimulator can be programmed above its own
# maximum output, whatever the configured safety ratio allows.
MAX_STIMULATOR_OUTPUT_PCT = 100.0


@dataclass
class AnalysisConfig:
    """Configuration for bundle analysis."""

    cst_trk: str
    target_trk: str
    rmt: float
    cst_coords: np.ndarray
    target_coords: np.ndarray
    surf_path: Optional[str] = None
    gwi_threshold: float = 3.0
    roi_radius: float = 20.0
    activation_len: float = 4.0
    min_seg_len: float = 10.0
    cst_weights: Optional[str] = None
    target_weights: Optional[str] = None
    out_dir: str = "viz_output"


@dataclass
class BundleResult:
    """Results from bundle analysis."""

    name: str
    metric_weighted: float
    metric_unweighted: float
    n_total: int
    n_analyzed: int
    weight_source: str
    roi_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    roi_radius: float = 0.0
    aggregates_weighted: Dict[str, float] = field(default_factory=dict)
    aggregates_unweighted: Dict[str, float] = field(default_factory=dict)


def load_surface_tree(surface_path: str) -> cKDTree:
    """Load FreeSurfer surface and build KDTree for distance queries."""
    log.debug(f"Loading surface: {surface_path}")
    try:
        coords, _ = nib.freesurfer.read_geometry(surface_path)
        return cKDTree(coords)
    except Exception as e:
        log.error(f"Failed to load surface: {e}")
        raise


def _validate_weights(weights: np.ndarray, source: str) -> np.ndarray:
    """Validate SIFT2 streamline weights (audit S-004).

    Weights are per-streamline cross-sectional multipliers, so an explicitly
    requested weight set must be usable: fail loudly rather than silently
    fall back to a fabricated uniform weighting.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size == 0:
        raise ValueError(f"Weight source '{source}' is empty.")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"Weight source '{source}' contains non-finite values.")
    if np.any(weights < 0):
        raise ValueError(f"Weight source '{source}' contains negative values.")
    if weights.sum() <= 0:
        raise ValueError(f"Weight source '{source}' has zero total mass.")
    return weights


def load_weights(weight_path: str, streamlines: List[np.ndarray]) -> np.ndarray:
    """Load and validate streamline weights from a text file or NIfTI volume.

    A weight file is only loaded when it was explicitly configured; per audit
    S-004 a load or validation failure raises rather than silently substituting
    uniform weights.
    """
    if weight_path.endswith(".txt"):
        log.debug(f"Loading text weights: {weight_path}")
        try:
            w = np.loadtxt(weight_path)
        except Exception as e:
            raise ValueError(f"Failed to load weight file '{weight_path}': {e}")
        return _validate_weights(w, weight_path)

    elif weight_path.endswith((".nii", ".nii.gz")):
        log.debug(f"Sampling volume weights: {weight_path}")
        try:
            img = nib.load(weight_path)
            data = img.get_fdata()
            inv_affine = np.linalg.inv(img.affine)
            if data.ndim == 4:
                data = np.linalg.norm(data, axis=3)

            weights = []
            for sl in streamlines:
                vox_idx = nib.affines.apply_affine(inv_affine, sl).astype(int)
                x = np.clip(vox_idx[:, 0], 0, data.shape[0] - 1)
                y = np.clip(vox_idx[:, 1], 0, data.shape[1] - 1)
                z = np.clip(vox_idx[:, 2], 0, data.shape[2] - 1)
                vals = data[x, y, z]
                weights.append(np.mean(vals))
        except Exception as e:
            raise ValueError(f"Failed to sample weight volume '{weight_path}': {e}")
        return _validate_weights(np.array(weights), weight_path)

    raise ValueError(f"Unsupported weight file format: {weight_path}")


def apply_intensity_bounds(
    raw_intensity: float,
    rmt: float,
    floor_ratio: float = 0.70,
    ceiling_ratio: float = 1.40,
) -> dict:
    """
    Apply physiological bounds to a raw intensity estimate.

    Reports both the raw model value and a clamped "best estimate" constrained
    within [floor, ceiling]. The ceiling is the safety ratio limited by the
    stimulator maximum, and the floor may not exceed that ceiling, so the
    clamped value is always a programmable stimulator intensity.

    The two ceiling mechanisms carry distinct flags: CLAMPED_HIGH means the
    safety ratio bound the estimate, DEVICE_LIMITED means the estimate exceeds
    the stimulator maximum and cannot be delivered at any setting.

    Args:
        raw_intensity: Raw model-estimated intensity (% MSO).
        rmt: Measured RMT (% MSO).
        floor_ratio: Min intensity as fraction of RMT (default 0.70).
        ceiling_ratio: Max intensity as fraction of RMT (default 1.40).

    Returns:
        Dict with keys: model_raw, best_estimate, flag, deviation_pct.
    """
    # A non-finite raw value means the estimate failed (e.g. zero target
    # metric); report it as such rather than clamping it into a plausible range.
    if not np.isfinite(raw_intensity):
        return {
            "model_raw": raw_intensity,
            "best_estimate": float("nan"),
            "flag": "ESTIMATION_FAILED",
            "deviation_pct": float("nan"),
        }

    safety_ceil = rmt * ceiling_ratio
    intensity_ceil = min(safety_ceil, MAX_STIMULATOR_OUTPUT_PCT)
    # A floor ratio above 100/RMT would otherwise yield an unprogrammable floor.
    intensity_floor = min(rmt * floor_ratio, intensity_ceil)

    if raw_intensity < intensity_floor:
        best_estimate = intensity_floor
        flag = "CLAMPED_LOW"
    elif raw_intensity > intensity_ceil:
        best_estimate = intensity_ceil
        flag = "DEVICE_LIMITED" if safety_ceil > MAX_STIMULATOR_OUTPUT_PCT else "CLAMPED_HIGH"
    else:
        best_estimate = raw_intensity
        flag = "WITHIN_RANGE"

    deviation_pct = abs(raw_intensity - rmt) / rmt * 100.0 if rmt > 0 else 0.0

    return {
        "model_raw": raw_intensity,
        "best_estimate": best_estimate,
        "flag": flag,
        "deviation_pct": deviation_pct,
    }


def validate_calibration_metrics(metric_weighted: float, metric_unweighted: float) -> None:
    """Require usable weighted and unweighted CST calibration metrics."""
    invalid = []
    if not np.isfinite(metric_weighted) or metric_weighted <= 0:
        invalid.append(f"weighted={metric_weighted}")
    if not np.isfinite(metric_unweighted) or metric_unweighted <= 0:
        invalid.append(f"unweighted={metric_unweighted}")
    if invalid:
        values = ", ".join(invalid)
        raise ValueError(
            "CST calibration is unusable; weighted and unweighted metrics must be "
            f"finite and greater than zero ({values})."
        )


def format_weight_sources(cst_source: str, target_source: str) -> str:
    """Preserve the legacy value when sources match, otherwise report both."""
    if cst_source == target_source:
        return cst_source
    return f"CST: {cst_source}; Target: {target_source}"


def analyze_bundle(
    name: str,
    trk_path: str,
    roi_center: np.ndarray,
    config: AnalysisConfig,
    surface_tree: Optional[cKDTree] = None,
    weight_path: Optional[str] = None,
    orig_indices: Optional[np.ndarray] = None,
) -> BundleResult:
    """
    Analyze a bundle to compute robust activation metrics.

    Args:
        name: Bundle name for logging
        trk_path: Path to TRK file with AF data
        roi_center: ROI center coordinates
        config: Analysis configuration
        surface_tree: Optional surface KDTree for GWI filtering
        weight_path: Optional path to weight file
        orig_indices: Optional original streamline ids for the TRK streamlines,
            in TRK order (audit C-002). When provided, the weight file (indexed
            by original bundle id) is re-aligned to TRK order before use so the
            SIFT2 weight of each surviving streamline stays attached to it after
            upstream drops. When None, weights are indexed positionally
            (legacy/standalone behaviour).

    Returns:
        BundleResult with metrics
    """
    log.debug(f"Analyzing bundle: {name}")

    try:
        trk = TrkFile.load(trk_path)
        streamlines = list(trk.tractogram.streamlines)

        # Get AF scalar data
        if "AF" in trk.tractogram.data_per_point:
            af_data = trk.tractogram.data_per_point["AF"]
        elif "af" in trk.tractogram.data_per_point:
            af_data = trk.tractogram.data_per_point["af"]
        else:
            raise ValueError(f"Scalar 'AF' not found in {trk_path}")

        af_values = [np.array(a).flatten() for a in af_data]
        log.debug(f"Loaded {len(streamlines)} streamlines")
    except Exception as e:
        log.error(f"Failed to load TRK: {e}")
        raise

    # Load weights if available
    all_weights = None
    if weight_path:
        if not Path(weight_path).exists():
            raise FileNotFoundError(f"Configured weight file does not exist: {weight_path}")
        all_weights = load_weights(weight_path, streamlines)
        weight_source = f"External ({Path(weight_path).name})"

        # Re-align the weight vector (indexed by original bundle id) to the TRK
        # streamline order, so each surviving streamline keeps its own SIFT2
        # weight after upstream angular/AF drops (audit C-002). Only text weight
        # files are indexed by original streamline id; volume-sampled (NIfTI)
        # weights are read at the TRK streamline positions and are already
        # TRK-aligned, so they are left untouched.
        if orig_indices is not None and weight_path.endswith(".txt"):
            orig_indices = np.asarray(orig_indices, dtype=int)
            if len(orig_indices) != len(streamlines):
                raise ValueError(
                    f"{name}: orig_indices length ({len(orig_indices)}) does not "
                    f"match TRK streamlines ({len(streamlines)})."
                )
            if all_weights.size <= int(orig_indices.max(initial=-1)):
                raise ValueError(
                    f"{name}: weight file '{Path(weight_path).name}' has "
                    f"{all_weights.size} weights but the bundle references index "
                    f"{int(orig_indices.max())} (tractogram/weight provenance mismatch)."
                )
            all_weights = all_weights[orig_indices]
    else:
        weight_source = "Uniform"

    thresholds = []
    valid_indices = []
    n_nonfinite = 0

    for idx, (sl, af) in enumerate(zip(streamlines, af_values)):
        # ROI filter
        dists = np.linalg.norm(sl - roi_center, axis=1)
        sphere_mask = dists <= config.roi_radius

        if not np.any(sphere_mask):
            continue

        points_roi = sl[sphere_mask]
        af_roi = af[sphere_mask]

        # Surface filter (GWI)
        if surface_tree:
            surf_dists, _ = surface_tree.query(points_roi, k=1)
            gwi_mask = surf_dists <= config.gwi_threshold
            if not np.any(gwi_mask):
                continue
            points_roi = points_roi[gwi_mask]
            af_roi = af_roi[gwi_mask]

        if len(points_roi) < 2:
            continue

        # Skip streamlines whose ROI carries a non-finite AF (e.g. an
        # unmatched E-field sample); they would poison the aggregation.
        if not np.all(np.isfinite(af_roi)):
            n_nonfinite += 1
            continue

        # Contiguous threshold calculation
        steps = np.linalg.norm(np.diff(points_roi, axis=0), axis=1)
        if np.sum(steps) < config.min_seg_len:
            continue

        af_abs = np.abs(af_roi)
        af_seg = (af_abs[:-1] + af_abs[1:]) / 2.0

        thresh = get_max_contiguous_threshold(af_seg, steps, config.activation_len)
        thresholds.append(thresh)
        valid_indices.append(idx)

    n_valid = len(thresholds)
    log.debug(f"Valid streamlines: {n_valid}/{len(streamlines)}")
    if n_nonfinite:
        log.warning(f"{name}: skipped {n_nonfinite} streamlines with non-finite AF")

    thresholds = np.array(thresholds)

    if n_valid == 0:
        empty_aggregates = cross_streamline_aggregates(thresholds)
        return BundleResult(
            name,
            0.0,
            0.0,
            len(streamlines),
            0,
            weight_source,
            aggregates_weighted=dict(empty_aggregates),
            aggregates_unweighted=dict(empty_aggregates),
        )

    if all_weights is not None:
        final_weights = all_weights[valid_indices]
    else:
        final_weights = np.ones(n_valid)

    # Weighted metric: median of top 5%
    p95_w = weighted_percentile(thresholds, final_weights, 95.0)
    top_mask_w = thresholds >= p95_w
    metric_w = 0.0
    if np.any(top_mask_w):
        metric_w = weighted_percentile(thresholds[top_mask_w], final_weights[top_mask_w], 50.0)

    # Unweighted metric: median of top 5%
    metric_u = median_of_top_percentile(thresholds, 95.0)

    log.debug(f"{name}: Weighted={metric_w:.2f}, Unweighted={metric_u:.2f} V/m²")

    # Diagnostic alternatives on the same threshold distribution. The primary
    # entry is overwritten with the reported metric so the sensitivity table can
    # never drift from the value that drives the dose.
    aggregates_weighted = cross_streamline_aggregates(thresholds, final_weights)
    aggregates_unweighted = cross_streamline_aggregates(thresholds)
    aggregates_weighted["median_top5"] = metric_w
    aggregates_unweighted["median_top5"] = metric_u

    return BundleResult(
        name=name,
        metric_weighted=metric_w,
        metric_unweighted=metric_u,
        n_total=len(streamlines),
        n_analyzed=n_valid,
        weight_source=weight_source,
        roi_center=roi_center,
        roi_radius=config.roi_radius,
        aggregates_weighted=aggregates_weighted,
        aggregates_unweighted=aggregates_unweighted,
    )


def build_aggregator_sensitivity(
    *,
    cst_weighted: Dict[str, float],
    cst_unweighted: Dict[str, float],
    target_weighted: Dict[str, float],
    target_unweighted: Dict[str, float],
    rmt: float,
) -> Dict[str, Dict[str, float]]:
    """
    Calibration ratio and raw intensity under each cross-streamline aggregator.

    Diagnostic only: the reported dose always uses the primary aggregator
    (median of the top 5%). Every entry is derived from the per-streamline
    threshold distributions already computed by :func:`analyze_bundle`, so no
    additional field solve or AF pass is involved.

    Args:
        cst_weighted: Weighted aggregates of the calibration bundle.
        cst_unweighted: Unweighted aggregates of the calibration bundle.
        target_weighted: Weighted aggregates of the target bundle.
        target_unweighted: Unweighted aggregates of the target bundle.
        rmt: Measured RMT percentage.

    Returns:
        Mapping of aggregator key to AF_CST, AF_target, SEI and raw intensity,
        weighted and unweighted.
    """
    sensitivity: Dict[str, Dict[str, float]] = {}
    for key in AGGREGATOR_KEYS:
        row: Dict[str, float] = {}
        for suffix, cst_agg, tgt_agg in (
            ("weighted", cst_weighted, target_weighted),
            ("unweighted", cst_unweighted, target_unweighted),
        ):
            af_cst = float(cst_agg.get(key, 0.0))
            af_tgt = float(tgt_agg.get(key, 0.0))
            row[f"af_cst_{suffix}"] = af_cst
            row[f"af_target_{suffix}"] = af_tgt
            row[f"sei_{suffix}"] = af_tgt / af_cst if af_cst > 0 else 0.0
            row[f"intensity_raw_{suffix}"] = rmt * (af_cst / af_tgt) if af_tgt > 0 else float("nan")
        sensitivity[key] = row
    return sensitivity


def run_unified_estimation(
    cst_trk: Path,
    target_trk: Path,
    cst_coords: List[float],
    target_coords: List[float],
    rmt: float,
    weights_cst: Optional[Path] = None,
    weights_target: Optional[Path] = None,
    surface_path: Optional[Path] = None,
    gwi_threshold: float = 3.0,
    roi_radius: float = 20.0,
    activation_len: float = 4.0,
    mso_floor_ratio: float = 0.70,
    mso_ceiling_ratio: float = 1.40,
    cst_orig_indices: Optional[np.ndarray] = None,
    target_orig_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Main entry point for unified intensity estimation.

    Args:
        cst_trk: Path to CST TRK with AF
        target_trk: Path to target TRK with AF
        cst_coords: CST ROI center coordinates
        target_coords: Target ROI center coordinates
        rmt: Measured RMT percentage
        weights_cst: Optional path to CST weights
        weights_target: Optional path to target weights
        surface_path: Optional path to FreeSurfer surface
        gwi_threshold: Max distance from the GWI surface (mm); used only when
            surface_path is supplied
        roi_radius: ROI radius (mm)
        activation_len: Required contiguous length (mm)
        mso_floor_ratio: Min MSO as fraction of RMT (default 0.70)
        mso_ceiling_ratio: Max MSO as fraction of RMT (default 1.40)
        cst_orig_indices: Original streamline ids for the CST TRK streamlines
            (audit C-002); keeps SIFT2 weights aligned after drops.
        target_orig_indices: Original streamline ids for the target TRK streamlines.

    Returns:
        Dictionary with estimation results
    """
    config = AnalysisConfig(
        cst_trk=str(cst_trk),
        target_trk=str(target_trk),
        rmt=rmt,
        cst_coords=np.array(cst_coords),
        target_coords=np.array(target_coords),
        surf_path=str(surface_path) if surface_path and surface_path.exists() else None,
        gwi_threshold=gwi_threshold,
        cst_weights=str(weights_cst) if weights_cst is not None else None,
        target_weights=str(weights_target) if weights_target is not None else None,
        roi_radius=roi_radius,
        activation_len=activation_len,
    )

    # Determine mode
    mode = "Baseline"
    if config.surf_path:
        mode = "Surface-Constrained (GWI)"
    if config.cst_weights or config.target_weights:
        mode += " + Weighted"

    log.debug(f"Estimation mode: {mode}")

    # Load surface if available
    surf_tree = None
    if config.surf_path:
        surf_tree = load_surface_tree(config.surf_path)

    # Analyze bundles
    cst_res = analyze_bundle(
        "CST",
        config.cst_trk,
        config.cst_coords,
        config,
        surf_tree,
        config.cst_weights,
        orig_indices=cst_orig_indices,
    )
    validate_calibration_metrics(cst_res.metric_weighted, cst_res.metric_unweighted)

    tgt_res = analyze_bundle(
        "Target",
        config.target_trk,
        config.target_coords,
        config,
        surf_tree,
        config.target_weights,
        orig_indices=target_orig_indices,
    )

    cst_metric = cst_res.metric_weighted
    tgt_metric = tgt_res.metric_weighted

    # Calculate intensity
    if tgt_metric > 0:
        intensity_est = rmt * (cst_metric / tgt_metric)
    else:
        intensity_est = float("nan")
        log.error("Target metric is zero. Intensity cannot be estimated.")

    # Unweighted fallback
    if tgt_res.metric_unweighted > 0:
        intensity_est_u = rmt * (cst_res.metric_unweighted / tgt_res.metric_unweighted)
    else:
        intensity_est_u = float("nan")

    sei_weighted = tgt_metric / cst_metric if cst_metric > 0 else 0.0
    sei_unweighted = (
        tgt_res.metric_unweighted / cst_res.metric_unweighted
        if cst_res.metric_unweighted > 0
        else 0.0
    )

    # Multiplier k = M_CST / M_target (intensity-invariant geometric ratio).
    # I_raw = RMT * k → user can rescale dose by multiplying k by any RMT
    # without rerunning the simulation.
    multiplier_weighted = cst_metric / tgt_metric if tgt_metric > 0 else 0.0
    multiplier_unweighted = (
        cst_res.metric_unweighted / tgt_res.metric_unweighted
        if tgt_res.metric_unweighted > 0
        else 0.0
    )

    # Apply physiological bounds
    bounded_w = apply_intensity_bounds(
        intensity_est,
        rmt,
        floor_ratio=mso_floor_ratio,
        ceiling_ratio=mso_ceiling_ratio,
    )
    bounded_u = apply_intensity_bounds(
        intensity_est_u,
        rmt,
        floor_ratio=mso_floor_ratio,
        ceiling_ratio=mso_ceiling_ratio,
    )

    # Log summary
    log.debug(f"CST AF: {cst_metric:.2f} V/m², Target AF: {tgt_metric:.2f} V/m²")
    log.debug(f"Raw intensity: {intensity_est:.1f}% (Unweighted: {intensity_est_u:.1f}%)")
    log.debug(
        f"Clamped intensity: {bounded_w['best_estimate']:.1f}% (Unweighted: {bounded_u['best_estimate']:.1f}%)"
    )
    if bounded_w["flag"] != "WITHIN_RANGE":
        log.debug(f"Intensity flag (weighted): {bounded_w['flag']}")
    if bounded_u["flag"] != "WITHIN_RANGE":
        log.debug(f"Intensity flag (unweighted): {bounded_u['flag']}")

    return {
        "cst_metric": cst_metric,
        "tgt_metric": tgt_metric,
        # Weighted intensity
        "intensity_est": bounded_w["best_estimate"],
        "intensity_est_raw": bounded_w["model_raw"],
        "intensity_est_clamped": bounded_w["best_estimate"],
        "intensity_est_flag": bounded_w["flag"],
        # Unweighted intensity
        "intensity_est_u": bounded_u["best_estimate"],
        "intensity_est_u_raw": bounded_u["model_raw"],
        "intensity_est_u_clamped": bounded_u["best_estimate"],
        "intensity_est_u_flag": bounded_u["flag"],
        # SEI — Stimulation Efficiency Index: AF_target / AF_CST = RMT / I_raw
        # SEI > 1: target more efficient than CST (less stimulation needed)
        # SEI = 1: same efficiency as CST (M1 identity case)
        # SEI < 1: target less efficient than CST (more stimulation needed)
        "sei_weighted": sei_weighted,
        "sei_unweighted": sei_unweighted,
        # Multiplier k = M_CST / M_target (intensity-invariant; I_raw = RMT * k)
        "multiplier_weighted": multiplier_weighted,
        "multiplier_unweighted": multiplier_unweighted,
        "mode": mode,
        "weight_source": cst_res.weight_source,
        "weight_source_cst": cst_res.weight_source,
        "weight_source_target": tgt_res.weight_source,
        "cst_unweighted": cst_res.metric_unweighted,
        "tgt_unweighted": tgt_res.metric_unweighted,
        "mso_floor_ratio": mso_floor_ratio,
        "mso_ceiling_ratio": mso_ceiling_ratio,
        "cst_aggregates_weighted": cst_res.aggregates_weighted,
        "cst_aggregates_unweighted": cst_res.aggregates_unweighted,
        "target_aggregates_weighted": tgt_res.aggregates_weighted,
        "target_aggregates_unweighted": tgt_res.aggregates_unweighted,
        "aggregator_sensitivity": build_aggregator_sensitivity(
            cst_weighted=cst_res.aggregates_weighted,
            cst_unweighted=cst_res.aggregates_unweighted,
            target_weighted=tgt_res.aggregates_weighted,
            target_unweighted=tgt_res.aggregates_unweighted,
            rmt=rmt,
        ),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Unified TMS Target Intensity Estimation")
    parser.add_argument("--cst", required=True, help="CST TRK file")
    parser.add_argument("--target", required=True, help="Target TRK file")
    parser.add_argument("--rmt", required=True, type=float, help="Measured RMT (%)")
    parser.add_argument("--cst_coords", type=float, nargs=3, required=True, help="CST ROI center")
    parser.add_argument(
        "--target_coords", type=float, nargs=3, required=True, help="Target ROI center"
    )
    parser.add_argument("--surf", help="FreeSurfer surface file")
    parser.add_argument("--cst_weights", help="CST weights file")
    parser.add_argument("--target_weights", help="Target weights file")
    args = parser.parse_args()

    results = run_unified_estimation(
        Path(args.cst),
        Path(args.target),
        args.cst_coords,
        args.target_coords,
        args.rmt,
        Path(args.cst_weights) if args.cst_weights else None,
        Path(args.target_weights) if args.target_weights else None,
        Path(args.surf) if args.surf else None,
    )

    print(f"\nEstimated Target I: {results['intensity_est']:.1f}% MSO")
