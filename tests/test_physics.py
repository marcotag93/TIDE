"""
Unit Tests for TIDE Physics Module
===================================
Tests for activating function calculation, threshold estimation, and RMT estimation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tide.core.physics import (
    AGGREGATOR_KEYS,
    PRIMARY_AGGREGATOR,
    calculate_scalar_map,
    cross_streamline_aggregates,
    get_max_contiguous_threshold,
    median_of_top_percentile,
    weighted_mean,
    weighted_median_of_top_percentile,
    weighted_percentile,
)

# =============================================================================
# Test Fixtures (local to this file)
# =============================================================================


@pytest.fixture
def straight_bundle_with_uniform_efield():
    """
    Bundle of straight streamlines with uniform E-field aligned with tangent.
    Expected: High E_parallel, moderate AF (low gradient on uniform field).
    """
    streamlines = []
    e_fields = []

    for i in range(5):
        # Straight line along X-axis
        x = np.arange(0, 25, 0.5)
        sl = np.column_stack([x, np.ones_like(x) * i * 2, np.zeros_like(x)])

        # Uniform E-field aligned with tangent (X direction)
        ef = np.tile([100.0, 0.0, 0.0], (len(sl), 1))

        streamlines.append(sl)
        e_fields.append(ef)

    return streamlines, e_fields


@pytest.fixture
def curved_bundle_with_varying_efield():
    """
    Curved streamlines with varying E-field.
    Expected: Non-zero AF due to the gradient term d(E·T)/ds.
    """
    streamlines = []
    e_fields = []

    for i in range(3):
        # Curved path (quarter circle)
        t = np.linspace(0, np.pi / 2, 30)
        radius = 20.0
        sl = np.column_stack([radius * np.cos(t), radius * np.sin(t) + i * 5, np.zeros_like(t)])

        # E-field with gradient (increasing magnitude)
        magnitude = 50 + t * 100  # 50-200 V/m gradient
        tangent = np.gradient(sl, axis=0)
        tangent = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9)
        ef = tangent * magnitude[:, np.newaxis]

        streamlines.append(sl)
        e_fields.append(ef)

    return streamlines, e_fields


# =============================================================================
# Tests for calculate_scalar_map
# =============================================================================


class TestCalculateScalarMap:
    """Tests for the main AF calculation function."""

    def test_returns_correct_structure(self, straight_bundle_with_uniform_efield):
        """Test that function returns expected tuple structure."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        new_sl, af_vals, lengths = calculate_scalar_map(streamlines, e_fields, mode="af")

        assert isinstance(new_sl, list)
        assert isinstance(af_vals, list)
        assert isinstance(lengths, list)
        assert len(new_sl) == len(af_vals) == len(lengths)

    def test_midpoint_streamlines_have_correct_length(self, straight_bundle_with_uniform_efield):
        """Midpoint streamlines should have N-1 points for N-point input."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        new_sl, _, _ = calculate_scalar_map(streamlines, e_fields, mode="af")

        for orig, midpt in zip(streamlines, new_sl):
            # Should have one fewer point (midpoints between segments)
            assert len(midpt) == len(orig) - 1

    def test_af_values_finite(self, straight_bundle_with_uniform_efield):
        """Signed AF values should be finite (no NaN/Inf)."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        _, af_vals, _ = calculate_scalar_map(streamlines, e_fields, mode="af")

        for af in af_vals:
            assert np.all(np.isfinite(af)), "AF values must be finite"

    def test_af_values_abs_when_signed_false(self, straight_bundle_with_uniform_efield):
        """signed=False returns non-negative magnitudes."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        _, af_vals, _ = calculate_scalar_map(streamlines, e_fields, mode="af", signed=False)

        for af in af_vals:
            assert np.all(af >= 0), "signed=False must return |AF|"

    def test_uniform_field_produces_low_gradient_term(self, straight_bundle_with_uniform_efield):
        """Uniform E-field on straight fiber should have near-zero gradient term."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        _, af_vals, _ = calculate_scalar_map(streamlines, e_fields, mode="af")

        # For uniform field, |AF| should be low (numerical noise only)
        for af in af_vals:
            assert np.mean(np.abs(af)) < 50, "Uniform field should produce low |AF|"

    def test_varying_field_produces_higher_af(self, curved_bundle_with_varying_efield):
        """Varying E-field should produce measurable AF."""
        streamlines, e_fields = curved_bundle_with_varying_efield

        _, af_vals, _ = calculate_scalar_map(streamlines, e_fields, mode="af")

        # Check that |AF| has meaningful magnitude (signed values may be +/-)
        all_af = np.concatenate(af_vals)
        assert np.max(np.abs(all_af)) > 10, "Varying field should produce measurable AF"

    def test_e_parallel_mode(self, straight_bundle_with_uniform_efield):
        """Test e_parallel mode returns projected field (signed)."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        _, e_par_vals, _ = calculate_scalar_map(streamlines, e_fields, mode="e_parallel")

        # For aligned field, |E_parallel| should be close to field magnitude
        for e_par in e_par_vals:
            assert np.mean(np.abs(e_par)) > 90, "|E_parallel| should be ~100 V/m for aligned field"

    def test_skips_short_streamlines(self):
        """Streamlines with < 4 points should be skipped."""
        short_sl = [np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])]  # Only 3 points
        short_ef = [np.array([[100, 0, 0], [100, 0, 0], [100, 0, 0]])]

        new_sl, af_vals, lengths = calculate_scalar_map(short_sl, short_ef, mode="af")

        assert len(new_sl) == 0, "Streamlines with <4 points should be skipped"

    def test_empty_input_returns_empty(self):
        """Empty input should return empty lists."""
        new_sl, af_vals, lengths = calculate_scalar_map([], [], mode="af")

        assert new_sl == []
        assert af_vals == []
        assert lengths == []

    def test_adaptive_sigma_used_by_default(self, straight_bundle_with_uniform_efield):
        """Test that adaptive sigma is computed when smooth_sigma=None."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        # Should not raise, and should compute adaptive sigma internally
        new_sl, af_vals, lengths = calculate_scalar_map(
            streamlines, e_fields, mode="af", smooth_sigma=None
        )

        assert len(new_sl) > 0

    def test_custom_sigma_accepted(self, straight_bundle_with_uniform_efield):
        """Test that custom sigma can be provided."""
        streamlines, e_fields = straight_bundle_with_uniform_efield

        new_sl, af_vals, _ = calculate_scalar_map(
            streamlines, e_fields, mode="af", smooth_sigma=5.0
        )

        assert len(new_sl) > 0


# =============================================================================
# Tests for get_max_contiguous_threshold
# =============================================================================


class TestGradientTermAnalytic:
    """Analytic checks on the gradient term d(E·T)/ds with exact SI units."""

    @staticmethod
    def _straight_fiber(step_mm: float = 0.5, n_points: int = 50) -> np.ndarray:
        x = np.arange(0, n_points * step_mm, step_mm)[:n_points]
        return np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])

    def test_linear_field_yields_si_gradient(self):
        """E·T rising 1 (V/m) per mm gives d(E·T)/ds = 1000 V/m² (mm→m)."""
        sl = self._straight_fiber()
        slope_per_mm = 1.0
        ex = slope_per_mm * sl[:, 0]
        ef = np.column_stack([ex, np.zeros_like(ex), np.zeros_like(ex)])

        # Tiny sigma keeps smoothing an identity so the gradient is isolated.
        _, af_vals, _ = calculate_scalar_map([sl], [ef], mode="af", smooth_sigma=1e-6)

        assert len(af_vals) == 1
        assert np.allclose(np.median(af_vals[0]), slope_per_mm * 1000.0, rtol=1e-3)

    def test_gradient_sign_flips_with_field_reversal(self):
        """Signed AF flips sign when the field gradient reverses."""
        sl = self._straight_fiber()
        ex = sl[:, 0]
        ef_pos = np.column_stack([ex, np.zeros_like(ex), np.zeros_like(ex)])
        ef_neg = -ef_pos

        _, af_pos, _ = calculate_scalar_map([sl], [ef_pos], mode="af", smooth_sigma=1e-6)
        _, af_neg, _ = calculate_scalar_map([sl], [ef_neg], mode="af", smooth_sigma=1e-6)

        np.testing.assert_allclose(af_pos[0], -af_neg[0], rtol=1e-6, atol=1e-6)

    def test_uniform_field_yields_zero_gradient(self):
        """Constant E·T on a straight fiber gives a vanishing gradient term."""
        sl = self._straight_fiber()
        ef = np.tile([100.0, 0.0, 0.0], (len(sl), 1))

        _, af_vals, _ = calculate_scalar_map([sl], [ef], mode="af", smooth_sigma=1e-6)

        assert np.allclose(af_vals[0], 0.0, atol=1e-6)

    @pytest.mark.parametrize("step_mm", [0.25, 0.5, 1.0, 2.0])
    def test_af_is_sampling_invariant_at_boundaries(self, step_mm: float):
        sl = self._straight_fiber(step_mm=step_mm, n_points=int(24.0 / step_mm) + 1)
        ex = sl[:, 0]
        ef = np.column_stack((ex, np.zeros_like(ex), np.zeros_like(ex)))

        midpoint_sl, af_vals, lengths = calculate_scalar_map(
            [sl],
            [ef],
            mode="af",
        )

        np.testing.assert_allclose(af_vals[0], 1000.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(midpoint_sl[0], (sl[:-1] + sl[1:]) / 2.0)
        np.testing.assert_allclose(lengths[0], np.linalg.norm(np.diff(sl, axis=0), axis=1))

    def test_af_preserves_gradient_under_streamline_reversal(self):
        sl = self._straight_fiber(step_mm=0.75, n_points=41)
        ex = sl[:, 0]
        ef = np.column_stack((ex, np.zeros_like(ex), np.zeros_like(ex)))

        _, forward_af, _ = calculate_scalar_map(
            [sl],
            [ef],
            mode="af",
        )
        _, reversed_af, _ = calculate_scalar_map(
            [sl[::-1]],
            [ef[::-1]],
            mode="af",
        )

        np.testing.assert_allclose(forward_af[0], reversed_af[0][::-1], rtol=1e-6, atol=1e-6)

    def test_af_is_sampling_invariant_on_curved_fiber(self):
        radius_mm = 30.0
        total_length_mm = radius_mm * np.pi / 2.0
        comparison_s = np.linspace(5.0, total_length_mm - 5.0, 80)
        results = []

        for step_mm in (0.25, 1.0):
            s = np.arange(0.0, total_length_mm, step_mm)
            s = np.append(s, total_length_mm)
            theta = s / radius_mm
            sl = np.column_stack(
                (
                    radius_mm * np.cos(theta),
                    radius_mm * np.sin(theta),
                    np.zeros_like(theta),
                )
            )
            tangent = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros_like(theta)))
            ef = tangent * (50.0 + s)[:, np.newaxis]

            _, af_vals, _ = calculate_scalar_map(
                [sl],
                [ef],
                mode="af",
            )
            input_s = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(sl, axis=0), axis=1)))
            )
            midpoint_s = (input_s[:-1] + input_s[1:]) / 2.0
            results.append(np.interp(comparison_s, midpoint_s, af_vals[0]))

        np.testing.assert_allclose(results[0], results[1], rtol=5e-3, atol=1.0)


class TestGetMaxContiguousThreshold:
    """Tests for the contiguous segment threshold algorithm."""

    def test_simple_case(self):
        """Test with simple uniform values."""
        values = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        lengths = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        thresh = get_max_contiguous_threshold(values, lengths, target_length=2.0)

        # A contiguous segment of length 2 can have min value of 20 (indices 1-2 or 2-3)
        assert thresh == 20.0

    def test_target_length_exceeds_total(self):
        """Returns 0 if target length exceeds total streamline length."""
        values = np.array([10.0, 20.0, 30.0])
        lengths = np.array([1.0, 1.0, 1.0])  # Total = 3mm

        thresh = get_max_contiguous_threshold(values, lengths, target_length=10.0)

        assert thresh == 0.0

    def test_empty_input(self):
        """Empty input returns 0."""
        values = np.array([])
        lengths = np.array([])

        thresh = get_max_contiguous_threshold(values, lengths, target_length=1.0)

        assert thresh == 0.0

    def test_all_same_values(self):
        """Uniform values should return that value."""
        values = np.array([50.0, 50.0, 50.0, 50.0])
        lengths = np.array([1.0, 1.0, 1.0, 1.0])

        thresh = get_max_contiguous_threshold(values, lengths, target_length=2.0)

        assert thresh == 50.0

    def test_finds_optimal_window(self):
        """Should find the window with highest minimum."""
        values = np.array([5.0, 100.0, 100.0, 100.0, 5.0])
        lengths = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Best 3mm window is indices 1-3 with min=100
        thresh = get_max_contiguous_threshold(values, lengths, target_length=3.0)

        assert thresh == 100.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_streamline(self):
        """Test with single streamline."""
        sl = [np.column_stack([np.arange(0, 10, 0.5), np.zeros(20), np.zeros(20)])]
        ef = [np.tile([100.0, 0.0, 0.0], (20, 1))]

        new_sl, af_vals, lengths = calculate_scalar_map(sl, ef, mode="af")

        assert len(new_sl) == 1

    def test_mismatched_lengths_skipped(self):
        """Streamlines with mismatched E-field lengths should be skipped."""
        sl = [np.column_stack([np.arange(0, 10, 0.5), np.zeros(20), np.zeros(20)])]
        ef = [np.tile([100.0, 0.0, 0.0], (15, 1))]  # Wrong length

        new_sl, af_vals, lengths = calculate_scalar_map(sl, ef, mode="af")

        assert len(new_sl) == 0

    def test_nan_handling(self):
        """Test behavior with NaN values in input."""
        values = np.array([10.0, np.nan, 30.0])
        lengths = np.array([1.0, 1.0, 1.0])

        # Should handle gracefully (may return 0 or propagate NaN)
        thresh = get_max_contiguous_threshold(values, lengths, target_length=1.0)

        # Result should be finite or 0
        assert np.isfinite(thresh) or thresh == 0.0


class TestApplyMSOBounds:
    """Tests for the MSO physiological floor/ceiling bounding function."""

    def setup_method(self):
        """Import apply_intensity_bounds for each test."""
        from tide.interfaces.unified_estimation import apply_intensity_bounds

        self.apply_intensity_bounds = apply_intensity_bounds

    def test_within_range_returns_raw(self):
        """MSO within bounds should pass through unchanged."""
        result = self.apply_intensity_bounds(raw_intensity=45.0, rmt=50.0, floor_ratio=0.70)
        assert result["best_estimate"] == 45.0
        assert result["model_raw"] == 45.0
        assert result["flag"] == "WITHIN_RANGE"

    def test_clamped_low(self):
        """MSO below floor should be clamped to floor."""
        result = self.apply_intensity_bounds(raw_intensity=25.0, rmt=50.0, floor_ratio=0.70)
        assert result["best_estimate"] == 35.0  # 50 * 0.70
        assert result["model_raw"] == 25.0
        assert result["flag"] == "CLAMPED_LOW"

    def test_clamped_high(self):
        """MSO above ceiling should be capped."""
        result = self.apply_intensity_bounds(
            raw_intensity=100.0, rmt=50.0, floor_ratio=0.70, ceiling_ratio=1.50
        )
        assert result["best_estimate"] == 75.0  # 50 * 1.50
        assert result["model_raw"] == 100.0
        assert result["flag"] == "CLAMPED_HIGH"

    def test_ceiling_capped_at_100(self):
        """Ceiling never exceeds 100% MSO even when 1.5 * RMT would."""
        # 1.5 * 80 = 120, must be capped to 100, and the device cap is the
        # binding constraint, so the flag names the device rather than safety.
        result = self.apply_intensity_bounds(raw_intensity=130.0, rmt=80.0, ceiling_ratio=1.50)
        assert result["best_estimate"] == 100.0
        assert result["model_raw"] == 130.0
        assert result["flag"] == "DEVICE_LIMITED"

    def test_safety_ceiling_exactly_at_device_limit_flags_clamped_high(self):
        """The safety ratio still binds when it lands exactly on 100% MSO."""
        result = self.apply_intensity_bounds(raw_intensity=120.0, rmt=50.0, ceiling_ratio=2.00)
        assert result["best_estimate"] == 100.0
        assert result["flag"] == "CLAMPED_HIGH"

    def test_floor_never_exceeds_device_limit(self):
        """A floor ratio above 100/RMT must not produce an unprogrammable floor."""
        # 1.20 * 85 = 102, which no stimulator can deliver.
        result = self.apply_intensity_bounds(
            raw_intensity=60.0, rmt=85.0, floor_ratio=1.20, ceiling_ratio=1.40
        )
        assert result["best_estimate"] == 100.0
        assert result["model_raw"] == 60.0
        assert result["flag"] == "CLAMPED_LOW"

    def test_exact_floor_is_within_range(self):
        """MSO exactly at floor should be WITHIN_RANGE (not clamped)."""
        result = self.apply_intensity_bounds(raw_intensity=35.0, rmt=50.0, floor_ratio=0.70)
        assert result["flag"] == "WITHIN_RANGE"

    def test_deviation_pct(self):
        """Deviation percentage should be correct."""
        result = self.apply_intensity_bounds(raw_intensity=25.0, rmt=50.0, floor_ratio=0.70)
        assert np.isclose(result["deviation_pct"], 50.0)  # |25-50|/50 * 100

    def test_rmt_equal_mso(self):
        """When raw_mso equals RMT, deviation should be zero."""
        result = self.apply_intensity_bounds(raw_intensity=50.0, rmt=50.0)
        assert result["deviation_pct"] == 0.0
        assert result["flag"] == "WITHIN_RANGE"
        assert result["best_estimate"] == 50.0


# =============================================================================
# Streamline identity through the drop chain (audit C-002)
# =============================================================================


def _short_and_long_bundle():
    """Six streamlines; ids 0 (first), 3 (middle), 5 (last) are <4-point stubs
    that calculate_scalar_map drops, so only ids 1, 2, 4 survive."""
    streamlines, e_fields = [], []
    for i in range(6):
        if i in (0, 3, 5):
            x = np.arange(0, 1.0, 0.5)  # 2 points -> dropped (< 4)
        else:
            x = np.arange(0, 25, 0.5)
        sl = np.column_stack([x, np.ones_like(x) * i * 2, np.zeros_like(x)])
        streamlines.append(sl)
        e_fields.append(np.tile([100.0, 0.0, 0.0], (len(sl), 1)))
    return streamlines, e_fields


class TestStreamlineIdentity:
    """calculate_scalar_map / filter_by_angular_deviation index passthrough."""

    def test_default_return_is_three_tuple(self, straight_bundle_with_uniform_efield):
        """Without indices the return arity is unchanged (backward compatible)."""
        streamlines, e_fields = straight_bundle_with_uniform_efield
        out = calculate_scalar_map(streamlines, e_fields, mode="af")
        assert len(out) == 3

    def test_tracks_surviving_ids_after_drops(self):
        """Surviving ids identify the kept originals, not compacted positions."""
        streamlines, e_fields = _short_and_long_bundle()
        idx0 = np.arange(len(streamlines))
        new_sl, af_vals, lengths, surviving = calculate_scalar_map(
            streamlines,
            e_fields,
            mode="af",
            indices=idx0,
        )
        assert list(surviving) == [1, 2, 4]
        assert len(new_sl) == len(af_vals) == len(lengths) == len(surviving)

    def test_weights_stay_attached_after_drops(self):
        """The C-002 fix: weights[surviving] reattaches each streamline's own
        weight; positional indexing (the bug) would mis-assign."""
        streamlines, e_fields = _short_and_long_bundle()
        weights = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        _, _, _, surviving = calculate_scalar_map(
            streamlines, e_fields, mode="af", indices=np.arange(len(streamlines))
        )
        # Correct (re-aligned) mapping
        assert list(weights[surviving]) == [11.0, 12.0, 14.0]
        # Positional (buggy) mapping would have been the wrong originals
        assert list(weights[: len(surviving)]) == [10.0, 11.0, 12.0]

    def test_empty_input_returns_four_tuple_with_indices(self):
        out = calculate_scalar_map([], [], mode="af", indices=np.array([], dtype=int))
        assert len(out) == 4
        assert out[3].size == 0

    def test_filter_tracks_indices_in_sync(self):
        """filter_by_angular_deviation filters the id array in lockstep."""
        from tide.core.tractography import filter_by_angular_deviation

        streamlines, e_fields = _short_and_long_bundle()
        idx0 = np.arange(len(streamlines))
        # No ROI, generous angle: only the <4-point stubs (0, 3, 5) are removed.
        filtered_sl, filtered_ev, n_removed, filtered_idx = filter_by_angular_deviation(
            streamlines,
            e_field_vectors=e_fields,
            max_angle_deg=90.0,
            indices=idx0,
        )
        assert list(filtered_idx) == [1, 2, 4]
        assert len(filtered_sl) == len(filtered_ev) == len(filtered_idx)
        assert n_removed == 3


# =============================================================================
# SIFT2 weight validation (audit S-004)
# =============================================================================


class TestWeightValidation:
    """load_weights rejects unusable weight files instead of silent fallback."""

    def _load(self, tmp_path, values):
        from tide.interfaces.unified_estimation import load_weights

        p = tmp_path / "weights.txt"
        np.savetxt(p, np.asarray(values))
        return load_weights(str(p), streamlines=[None] * len(values))

    def test_loads_valid_weights(self, tmp_path):
        w = self._load(tmp_path, [1.0, 2.0, 0.0, 3.5])
        assert list(w) == [1.0, 2.0, 0.0, 3.5]

    def test_rejects_negative(self, tmp_path):
        with pytest.raises(ValueError):
            self._load(tmp_path, [1.0, -2.0, 3.0])

    def test_rejects_non_finite(self, tmp_path):
        with pytest.raises(ValueError):
            self._load(tmp_path, [1.0, np.nan, 3.0])

    def test_rejects_zero_mass(self, tmp_path):
        with pytest.raises(ValueError):
            self._load(tmp_path, [0.0, 0.0, 0.0])


class TestMedianOfTopPercentile:
    """Shared top-5% aggregator used by bundle analysis and M1 validation (C-003)."""

    @staticmethod
    def _legacy(values: np.ndarray, pct: float = 95.0) -> float:
        """The inline formula the helper replaces, kept here as an oracle."""
        if values.size == 0:
            return 0.0
        cutoff = np.percentile(values, pct)
        top = values[values >= cutoff]
        return float(np.median(top)) if top.size else 0.0

    def test_empty_returns_zero(self):
        assert median_of_top_percentile(np.array([])) == 0.0

    def test_single_value(self):
        assert median_of_top_percentile(np.array([7.0])) == 7.0

    def test_matches_legacy_formula(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            values = rng.uniform(0.0, 1000.0, size=rng.integers(1, 500))
            assert median_of_top_percentile(values) == self._legacy(values)

    def test_ties_at_cutoff_are_included(self):
        # All-equal input: cutoff equals the value, every element is in the tail.
        values = np.full(10, 5.0)
        assert median_of_top_percentile(values) == 5.0

    def test_known_top_tail(self):
        values = np.arange(1.0, 101.0)  # 1..100, p95 cutoff = 95.05 -> {96..100}
        assert median_of_top_percentile(values) == 98.0


class TestConsoleGridPointReporter:
    """The console adapter forwards grid-point stages to UI phases (C-003)."""

    def test_hooks_map_to_worker_phases(self):
        # Importing the console package eagerly pulls the SimNIBS-backed
        # reporters, so this env-checks like the other integration tests.
        pytest.importorskip("simnibs")
        from tide.console.ipc import WorkerPhase
        from tide.console.worker_reporter import _ConsoleGridPointReporter

        calls = []

        class _FakeReporter:
            def phase(self, phase, pct):
                calls.append(("phase", phase, pct))

            def progress(self, pct):
                calls.append(("progress", pct))

        adapter = _ConsoleGridPointReporter(_FakeReporter())
        adapter.optimization()
        adapter.simulation()
        adapter.sampling()
        adapter.activating_function()
        adapter.bundle_analysis()
        adapter.saving_results()
        adapter.progress(50)

        assert calls == [
            ("phase", WorkerPhase.OPTIMIZATION, 0),
            ("phase", WorkerPhase.FEM_SIMULATION, 0),
            ("phase", WorkerPhase.EFIELD_SAMPLING, 0),
            ("phase", WorkerPhase.ACTIVATING_FUNCTION, 0),
            ("phase", WorkerPhase.BUNDLE_ANALYSIS, 0),
            ("phase", WorkerPhase.SAVING_RESULTS, 0),
            ("progress", 50),
        ]


class TestCrossStreamlineAggregates:
    """Alternative cross-streamline aggregators reported alongside the primary one."""

    thresholds = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])

    def test_keys_and_order_follow_the_declared_inventory(self):
        aggregates = cross_streamline_aggregates(self.thresholds)
        assert tuple(aggregates.keys()) == AGGREGATOR_KEYS
        assert PRIMARY_AGGREGATOR in aggregates

    def test_unweighted_primary_matches_the_committed_statistic(self):
        aggregates = cross_streamline_aggregates(self.thresholds)
        assert aggregates["median_top5"] == median_of_top_percentile(self.thresholds, 95.0)

    def test_unweighted_alternatives_match_numpy_definitions(self):
        aggregates = cross_streamline_aggregates(self.thresholds)
        assert aggregates["mean"] == pytest.approx(float(np.mean(self.thresholds)))
        assert aggregates["median"] == pytest.approx(float(np.median(self.thresholds)))
        assert aggregates["q90"] == pytest.approx(float(np.percentile(self.thresholds, 90.0)))
        assert aggregates["q95"] == pytest.approx(float(np.percentile(self.thresholds, 95.0)))

    def test_weighted_branch_uses_the_weighted_definitions(self):
        weights = np.linspace(0.5, 2.0, self.thresholds.size)
        aggregates = cross_streamline_aggregates(self.thresholds, weights)
        assert aggregates["median_top5"] == pytest.approx(
            weighted_median_of_top_percentile(self.thresholds, weights, 95.0)
        )
        assert aggregates["mean"] == pytest.approx(weighted_mean(self.thresholds, weights))
        assert aggregates["q90"] == pytest.approx(
            weighted_percentile(self.thresholds, weights, 90.0)
        )

    def test_weights_shift_the_distribution(self):
        top_heavy = np.where(self.thresholds >= 80.0, 10.0, 0.1)
        weighted = cross_streamline_aggregates(self.thresholds, top_heavy)
        unweighted = cross_streamline_aggregates(self.thresholds)
        assert weighted["median"] > unweighted["median"]
        assert weighted["mean"] > unweighted["mean"]

    def test_empty_input_returns_zeros_for_every_aggregator(self):
        aggregates = cross_streamline_aggregates(np.array([]))
        assert aggregates == {key: 0.0 for key in AGGREGATOR_KEYS}

    def test_zero_total_weight_mean_is_zero(self):
        assert weighted_mean(self.thresholds, np.zeros_like(self.thresholds)) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
