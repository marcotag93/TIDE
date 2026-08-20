"""
Unit Tests for TIDE Tractography Module
========================================
Tests for ROI masking, data extraction, and medoid calculation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tide.core.tractography import (
    extract_grid_endpoints,
    get_bundle_cortical_medoid,
    get_data_in_roi,
    get_roi_masks,
)

# =============================================================================
# Test Fixtures (local)
# =============================================================================


@pytest.fixture
def simple_streamlines():
    """Simple parallel streamlines for testing."""
    streamlines = []
    for i in range(5):
        x = np.arange(0, 25, 0.5)
        sl = np.column_stack([x, np.ones_like(x) * i * 5, np.zeros_like(x)])
        streamlines.append(sl)
    return streamlines


@pytest.fixture
def streamlines_with_tips():
    """Streamlines with distinct endpoints for medoid testing."""
    streamlines = []
    for i in range(10):
        # Start clustered around origin, end clustered around (50, 0, 30)
        start = np.array([i * 0.5, i * 0.5, 0])
        end = np.array([50 + i * 0.5, i * 0.5, 30 + i * 0.2])

        n_points = 20
        t = np.linspace(0, 1, n_points)
        sl = start[np.newaxis, :] * (1 - t[:, np.newaxis]) + end[np.newaxis, :] * t[:, np.newaxis]
        streamlines.append(sl)

    return streamlines


# =============================================================================
# Tests for get_roi_masks
# =============================================================================


class TestGetRoiMasks:
    """Tests for ROI mask generation."""

    def test_returns_two_lists(self, simple_streamlines):
        """Should return point masks and segment masks."""
        point_masks, segment_masks = get_roi_masks(
            simple_streamlines, roi_size_mm=10.0, target_coords=None
        )

        assert isinstance(point_masks, list)
        assert isinstance(segment_masks, list)
        assert len(point_masks) == len(simple_streamlines)
        assert len(segment_masks) == len(simple_streamlines)

    def test_segment_mask_length(self, simple_streamlines):
        """Segment masks should have N-1 elements for N-point streamlines."""
        point_masks, segment_masks = get_roi_masks(simple_streamlines, roi_size_mm=10.0)

        for sl, p_mask, s_mask in zip(simple_streamlines, point_masks, segment_masks):
            assert len(p_mask) == len(sl)
            assert len(s_mask) == len(sl) - 1

    def test_mask_with_target_coords(self, simple_streamlines):
        """Masks should respect target coordinate center."""
        target = np.array([12.5, 2.5, 0.0])  # Near middle of first streamline

        point_masks, segment_masks = get_roi_masks(
            simple_streamlines, roi_size_mm=5.0, target_coords=target
        )

        # First streamline should have some points in ROI
        assert np.any(point_masks[0]), "Should have points in ROI near target"

    def test_mask_without_target_uses_tips(self, simple_streamlines):
        """Without target, should mask near streamline endpoints."""
        point_masks, segment_masks = get_roi_masks(
            simple_streamlines, roi_size_mm=3.0, target_coords=None
        )

        # Should have masks at tips
        for p_mask in point_masks:
            # First or last few points should be in mask
            assert p_mask[0] or p_mask[-1], "Tips should be in ROI"

    def test_empty_streamlines(self):
        """Empty input should return empty lists."""
        point_masks, segment_masks = get_roi_masks([], roi_size_mm=10.0)

        assert point_masks == []
        assert segment_masks == []


# =============================================================================
# Tests for get_data_in_roi
# =============================================================================


class TestGetDataInRoi:
    """Tests for extracting data within ROI."""

    def test_returns_array(self, simple_streamlines):
        """Should return numpy array of values."""
        values = [np.random.rand(len(sl)) for sl in simple_streamlines]

        result = get_data_in_roi(simple_streamlines, values, roi_size_mm=5.0)

        assert isinstance(result, np.ndarray)

    def test_with_lengths(self, simple_streamlines):
        """Should return tuple when lengths provided."""
        values = [np.random.rand(len(sl) - 1) for sl in simple_streamlines]
        lengths = [np.ones(len(sl) - 1) * 0.5 for sl in simple_streamlines]

        result_values, result_lengths = get_data_in_roi(
            simple_streamlines, values, roi_size_mm=5.0, lengths=lengths
        )

        assert isinstance(result_values, np.ndarray)
        assert isinstance(result_lengths, np.ndarray)

    def test_filters_to_roi(self, simple_streamlines):
        """Should only return values within ROI."""
        # Create values that increase along streamline
        values = [np.arange(len(sl)) for sl in simple_streamlines]

        # Target at start of streamlines
        target = np.array([0.0, 0.0, 0.0])

        result = get_data_in_roi(simple_streamlines, values, roi_size_mm=3.0, target_coords=target)

        # Should only get low values (near start)
        if len(result) > 0:
            assert np.mean(result) < np.mean(np.concatenate(values))


# =============================================================================
# Tests for get_bundle_cortical_medoid
# =============================================================================


class TestGetBundleCorticalMedoid:
    """Tests for medoid calculation (with mocked load)."""

    @patch("tide.core.tractography.load_tract")
    def test_returns_3d_coordinate(self, mock_load, streamlines_with_tips, tmp_path):
        """Should return a 3D coordinate array."""
        # Setup mock
        mock_sft = MagicMock()
        mock_sft.streamlines = streamlines_with_tips
        mock_load.return_value = mock_sft

        result = get_bundle_cortical_medoid(
            trk_path=tmp_path / "dummy.trk",
            anat_path=tmp_path / "dummy.nii.gz",
            cortex_thickness_mm=4.0,
        )

        assert result.shape == (3,)
        assert np.isfinite(result).all()

    @patch("tide.core.tractography.load_tract")
    def test_medoid_is_on_streamline(self, mock_load, streamlines_with_tips, tmp_path):
        """Medoid should be an actual streamline endpoint."""
        mock_sft = MagicMock()
        mock_sft.streamlines = streamlines_with_tips
        mock_load.return_value = mock_sft

        result = get_bundle_cortical_medoid(
            trk_path=tmp_path / "dummy.trk", anat_path=tmp_path / "dummy.nii.gz"
        )

        # Collect all endpoints
        all_endpoints = []
        for sl in streamlines_with_tips:
            all_endpoints.append(sl[0])
            all_endpoints.append(sl[-1])
        all_endpoints = np.array(all_endpoints)

        # Medoid should match one of the endpoints
        distances = np.linalg.norm(all_endpoints - result, axis=1)
        assert np.min(distances) < 0.01, "Medoid should be an actual endpoint"

    @patch("tide.core.tractography.load_tract")
    def test_uses_reference_coord(self, mock_load, streamlines_with_tips, tmp_path):
        """Reference coordinate should influence cluster selection."""
        mock_sft = MagicMock()
        mock_sft.streamlines = streamlines_with_tips
        mock_load.return_value = mock_sft

        # Reference near the high-Z end
        ref = [50.0, 0.0, 30.0]

        result = get_bundle_cortical_medoid(
            trk_path=tmp_path / "dummy.trk",
            anat_path=tmp_path / "dummy.nii.gz",
            reference_coord=ref,
        )

        # Result should be near the high-Z end
        assert result[2] > 20.0, "Should select cluster near reference"


# =============================================================================
# Tests for extract_grid_endpoints
# =============================================================================


class TestExtractGridEndpoints:
    """Tests for grid point extraction."""

    @patch("tide.core.tractography.load_tract")
    def test_returns_list_of_coordinates(self, mock_load, streamlines_with_tips, tmp_path):
        """Should return list of [x, y, z] coordinates."""
        mock_sft = MagicMock()
        mock_sft.streamlines = streamlines_with_tips
        mock_load.return_value = mock_sft

        result = extract_grid_endpoints(
            trk_path=tmp_path / "dummy.trk",
            anat_path=tmp_path / "dummy.nii.gz",
            step_mm=4.0,
            cortex_thickness_mm=4.0,
        )

        assert isinstance(result, list)
        if len(result) > 0:
            assert len(result[0]) == 3

    @patch("tide.core.tractography.load_tract")
    def test_respects_step_size(self, mock_load, streamlines_with_tips, tmp_path):
        """Grid points should be spaced by step_mm."""
        mock_sft = MagicMock()
        mock_sft.streamlines = streamlines_with_tips
        mock_load.return_value = mock_sft

        step = 4.0
        result = extract_grid_endpoints(
            trk_path=tmp_path / "dummy.trk",
            anat_path=tmp_path / "dummy.nii.gz",
            step_mm=step,
            cortex_thickness_mm=4.0,
        )

        if len(result) > 1:
            # Check that points are on grid
            points = np.array(result)
            remainder = np.mod(points, step)
            # Should be very close to 0 (on grid)
            assert np.allclose(remainder, 0, atol=0.01) or np.allclose(remainder, step, atol=0.01)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestTractographyEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_streamlines_roi_mask(self):
        """Empty streamlines should return empty masks."""
        p_masks, s_masks = get_roi_masks([], roi_size_mm=10.0)
        assert p_masks == []
        assert s_masks == []

    def test_single_point_streamline(self):
        """Single-point streamlines should be handled."""
        streamlines = [np.array([[0.0, 0.0, 0.0]])]

        p_masks, s_masks = get_roi_masks(streamlines, roi_size_mm=10.0)

        # Should have a point mask but empty segment mask
        assert len(p_masks) == 1
        assert len(s_masks) == 1
        assert len(s_masks[0]) == 0  # No segments in single-point streamline

    @patch("tide.core.tractography.load_tract")
    def test_empty_tractogram(self, mock_load, tmp_path):
        """Empty tractogram should raise or return empty."""
        mock_sft = MagicMock()
        mock_sft.streamlines = []
        mock_load.return_value = mock_sft

        # extract_grid_endpoints should return empty list
        result = extract_grid_endpoints(
            trk_path=tmp_path / "dummy.trk",
            anat_path=tmp_path / "dummy.nii.gz",
            step_mm=4.0,
            cortex_thickness_mm=4.0,
        )

        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
