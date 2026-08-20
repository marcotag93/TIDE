"""
Unit Tests for the Active Unified Estimation Path
=================================================
Covers the functions that produce every published MSO/SEI number:
``weighted_percentile``, ``analyze_bundle`` (ROI + GWI masking, top-5% median
aggregation) and ``run_unified_estimation`` (SEI, multiplier, ΔMSO identity).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

# ruff: noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

nib_streamlines = pytest.importorskip("nibabel.streamlines")
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile
from scipy.spatial import cKDTree

from tide.interfaces.unified_estimation import (
    AnalysisConfig,
    analyze_bundle,
    format_weight_sources,
    run_unified_estimation,
    validate_calibration_metrics,
    weighted_percentile,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_bundle(af_values, center=(0.0, 0.0, 0.0), n_points=17, offsets=None):
    """Build straight streamlines centred on ``center`` with constant per-fibre AF.

    ``af_values`` gives one constant AF magnitude per streamline. Points span
    ±8 mm along X at 1 mm spacing so the bundle sits well inside a 20 mm ROI.
    """
    if offsets is None:
        offsets = np.linspace(-2.0, 2.0, len(af_values))
    cx, cy, cz = center
    streamlines = []
    af_per_point = []
    x = np.linspace(-8.0, 8.0, n_points)
    for af, dy in zip(af_values, offsets):
        sl = np.column_stack([x + cx, np.full_like(x, cy + dy), np.full_like(x, cz)]).astype(
            np.float32
        )
        streamlines.append(sl)
        af_per_point.append(np.full((n_points, 1), af, dtype=np.float32))
    return streamlines, af_per_point


def _write_trk(path, streamlines, af_per_point):
    tractogram = Tractogram(
        streamlines,
        data_per_point={"AF": af_per_point},
        affine_to_rasmm=np.eye(4),
    )
    TrkFile(tractogram).save(str(path))
    return path


def _config(cst_trk, tgt_trk, rmt=50.0):
    return AnalysisConfig(
        cst_trk=str(cst_trk),
        target_trk=str(tgt_trk),
        rmt=rmt,
        cst_coords=np.zeros(3),
        target_coords=np.zeros(3),
        roi_radius=20.0,
        activation_len=4.0,
    )


# =============================================================================
# weighted_percentile
# =============================================================================


class TestWeightedPercentile:
    def test_uniform_weights_match_searchsorted(self):
        values = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
        weights = np.ones_like(values)
        # cum weights [.2,.4,.6,.8,1.0]; searchsorted(0.5) -> index 2.
        assert weighted_percentile(values, weights, 50.0) == 100.0

    def test_empty_returns_zero(self):
        assert weighted_percentile(np.array([]), np.array([]), 50.0) == 0.0

    def test_zero_weights_returns_zero(self):
        values = np.array([1.0, 2.0, 3.0])
        assert weighted_percentile(values, np.zeros(3), 50.0) == 0.0

    def test_weight_shifts_percentile(self):
        values = np.array([10.0, 100.0])
        # Almost all mass on the small value -> median sits at the small value.
        assert weighted_percentile(values, np.array([99.0, 1.0]), 50.0) == 10.0


# =============================================================================
# analyze_bundle — ROI and GWI masking
# =============================================================================


class TestAnalyzeBundleMasking:
    def test_constant_af_metric_equals_value(self, tmp_path):
        streamlines, af = _make_bundle([100.0] * 5)
        trk = _write_trk(tmp_path / "b.trk", streamlines, af)
        cfg = _config(trk, trk)
        res = analyze_bundle("B", str(trk), np.zeros(3), cfg)
        assert res.n_analyzed == 5
        assert np.isclose(res.metric_weighted, 100.0)
        assert np.isclose(res.metric_unweighted, 100.0)

    def test_roi_excludes_distant_streamlines(self, tmp_path):
        near, af_near = _make_bundle([100.0] * 5, center=(0.0, 0.0, 0.0))
        far, af_far = _make_bundle([100.0] * 2, center=(500.0, 0.0, 0.0))
        trk = _write_trk(tmp_path / "b.trk", near + far, af_near + af_far)
        cfg = _config(trk, trk)
        res = analyze_bundle("B", str(trk), np.zeros(3), cfg)
        assert res.n_total == 7
        assert res.n_analyzed == 5

    def test_gwi_surface_filter_keeps_near_streamlines(self, tmp_path):
        streamlines, af = _make_bundle([100.0] * 5)
        trk = _write_trk(tmp_path / "b.trk", streamlines, af)
        cfg = _config(trk, trk)
        near_tree = cKDTree(np.vstack(streamlines))
        res = analyze_bundle("B", str(trk), np.zeros(3), cfg, surface_tree=near_tree)
        assert res.n_analyzed == 5

    def test_gwi_surface_filter_excludes_far_streamlines(self, tmp_path):
        streamlines, af = _make_bundle([100.0] * 5)
        trk = _write_trk(tmp_path / "b.trk", streamlines, af)
        cfg = _config(trk, trk)
        far_tree = cKDTree(np.array([[1000.0, 1000.0, 1000.0]]))
        res = analyze_bundle("B", str(trk), np.zeros(3), cfg, surface_tree=far_tree)
        assert res.n_analyzed == 0

    def test_configured_gwi_threshold_governs_surface_mask(self, tmp_path):
        streamlines, af = _make_bundle([100.0] * 5)
        trk = _write_trk(tmp_path / "b.trk", streamlines, af)
        # Surface sits 5 mm above every streamline point.
        tree = cKDTree(np.vstack(streamlines) + np.array([0.0, 0.0, 5.0]))

        strict = _config(trk, trk)
        relaxed = _config(trk, trk)
        relaxed.gwi_threshold = 6.0

        assert analyze_bundle("B", str(trk), np.zeros(3), strict, surface_tree=tree).n_analyzed == 0
        assert (
            analyze_bundle("B", str(trk), np.zeros(3), relaxed, surface_tree=tree).n_analyzed == 5
        )

    def test_missing_explicit_weight_path_raises(self, tmp_path):
        streamlines, af = _make_bundle([100.0] * 5)
        trk = _write_trk(tmp_path / "b.trk", streamlines, af)
        cfg = _config(trk, trk)
        missing = tmp_path / "missing_weights.txt"

        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            analyze_bundle("B", str(trk), np.zeros(3), cfg, weight_path=str(missing))


# =============================================================================
# run_unified_estimation — SEI, multiplier, ΔMSO identity
# =============================================================================


class TestRunUnifiedEstimation:
    def _run(self, tmp_path, cst_af, tgt_af, rmt=50.0, **kwargs):
        cst_sl, cst_data = _make_bundle([cst_af] * 5)
        tgt_sl, tgt_data = _make_bundle([tgt_af] * 5)
        cst_trk = _write_trk(tmp_path / "cst.trk", cst_sl, cst_data)
        tgt_trk = _write_trk(tmp_path / "tgt.trk", tgt_sl, tgt_data)
        return run_unified_estimation(
            cst_trk=cst_trk,
            target_trk=tgt_trk,
            cst_coords=[0.0, 0.0, 0.0],
            target_coords=[0.0, 0.0, 0.0],
            rmt=rmt,
            roi_radius=20.0,
            activation_len=4.0,
            **kwargs,
        )

    def test_identity_case(self, tmp_path):
        res = self._run(tmp_path, 100.0, 100.0, rmt=50.0)
        assert np.isclose(res["intensity_est_raw"], 50.0)
        assert np.isclose(res["sei_weighted"], 1.0)
        assert np.isclose(res["multiplier_weighted"], 1.0)

    def test_lower_target_efficiency_raises_mso(self, tmp_path):
        res = self._run(tmp_path, 100.0, 50.0, rmt=50.0)
        assert res["intensity_est_raw"] > 50.0
        assert np.isclose(res["intensity_est_raw"], 100.0)
        assert np.isclose(res["sei_weighted"], 0.5)

    def test_ceiling_ratio_clamps_estimate(self, tmp_path):
        res = self._run(
            tmp_path,
            100.0,
            50.0,
            rmt=50.0,
            mso_ceiling_ratio=1.40,
        )
        assert np.isclose(res["intensity_est_raw"], 100.0)
        assert np.isclose(res["intensity_est_clamped"], 70.0)
        assert res["intensity_est_flag"] == "CLAMPED_HIGH"

    def test_mso_monotonic_in_target_efficiency(self, tmp_path):
        high = self._run(tmp_path, 100.0, 80.0, rmt=50.0)["intensity_est_raw"]
        low = self._run(tmp_path, 100.0, 40.0, rmt=50.0)["intensity_est_raw"]
        assert low > high

    def test_multiplier_reconstructs_mso(self, tmp_path):
        rmt = 50.0
        res = self._run(tmp_path, 100.0, 50.0, rmt=rmt)
        assert np.isclose(res["intensity_est_raw"], rmt * res["multiplier_weighted"])

    def test_sei_minus_one_equals_relative_mso_gap(self, tmp_path):
        # With MSO_standard == RMT, |SEI - 1| == |RMT - MSO_raw| / MSO_raw.
        rmt = 50.0
        res = self._run(tmp_path, 100.0, 50.0, rmt=rmt)
        raw = res["intensity_est_raw"]
        rel_gap = abs(rmt - raw) / raw
        assert np.isclose(abs(res["sei_weighted"] - 1.0), rel_gap)

    def test_uniform_weights_file_matches_no_weights(self, tmp_path):
        cst_sl, cst_data = _make_bundle([60.0, 80.0, 100.0, 120.0, 140.0])
        tgt_sl, tgt_data = _make_bundle([40.0, 60.0, 80.0, 100.0, 120.0])
        cst_trk = _write_trk(tmp_path / "cst.trk", cst_sl, cst_data)
        tgt_trk = _write_trk(tmp_path / "tgt.trk", tgt_sl, tgt_data)
        w_cst = tmp_path / "w_cst.txt"
        w_tgt = tmp_path / "w_tgt.txt"
        np.savetxt(w_cst, np.ones(5))
        np.savetxt(w_tgt, np.ones(5))

        common = dict(
            cst_coords=[0.0, 0.0, 0.0],
            target_coords=[0.0, 0.0, 0.0],
            rmt=50.0,
            roi_radius=20.0,
            activation_len=4.0,
        )
        no_w = run_unified_estimation(cst_trk=cst_trk, target_trk=tgt_trk, **common)
        with_w = run_unified_estimation(
            cst_trk=cst_trk,
            target_trk=tgt_trk,
            weights_cst=w_cst,
            weights_target=w_tgt,
            **common,
        )
        assert np.isclose(no_w["intensity_est_raw"], with_w["intensity_est_raw"])
        assert with_w["weight_source"] == "External (w_cst.txt)"
        assert with_w["weight_source_cst"] == "External (w_cst.txt)"
        assert with_w["weight_source_target"] == "External (w_tgt.txt)"

    @pytest.mark.parametrize("weight_arg", ["weights_cst", "weights_target"])
    def test_missing_weight_path_is_not_dropped(self, tmp_path, weight_arg):
        missing = tmp_path / f"missing_{weight_arg}.txt"

        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            self._run(tmp_path, 100.0, 100.0, **{weight_arg: missing})

    def test_zero_target_metric_yields_estimation_failed(self, tmp_path):
        res = self._run(tmp_path, 100.0, 0.0, rmt=50.0)
        assert np.isnan(res["intensity_est_raw"])
        assert res["intensity_est_flag"] == "ESTIMATION_FAILED"

    def test_zero_cst_metric_raises_before_inversion(self, tmp_path):
        with pytest.raises(ValueError, match="CST calibration"):
            self._run(tmp_path, 0.0, 100.0, rmt=50.0)


class TestCalibrationValidation:
    @pytest.mark.parametrize(
        ("weighted", "unweighted"),
        [
            (0.0, 1.0),
            (-1.0, 1.0),
            (float("nan"), 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (1.0, float("inf")),
        ],
    )
    def test_rejects_unusable_metric(self, weighted, unweighted):
        with pytest.raises(ValueError, match="CST calibration"):
            validate_calibration_metrics(weighted, unweighted)

    def test_accepts_positive_finite_metrics(self):
        validate_calibration_metrics(100.0, 101.0)


class TestWeightSourceFormatting:
    def test_matching_sources_keep_historical_value(self):
        assert format_weight_sources("Uniform", "Uniform") == "Uniform"

    def test_different_sources_report_both(self):
        assert format_weight_sources("Uniform", "External (target.txt)") == (
            "CST: Uniform; Target: External (target.txt)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
