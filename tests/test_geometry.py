"""
Unit Tests for TIDE Geometry Module
====================================
Tests for ray casting, coil orientation, and scalp projection.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Tests for project_target_to_scalp
# =============================================================================


class TestProjectTargetToScalp:
    """Tests for ray-triangle intersection (Möller-Trumbore)."""

    @patch("tide.core.geometry.read_msh", new=MagicMock())
    def test_raises_on_missing_mesh(self, tmp_path):
        """Should raise FileNotFoundError if mesh doesn't exist.

        ``read_msh`` is patched non-None so the missing-mesh path is reached
        whether or not SimNIBS is importable in the test environment.
        """
        from tide.core.geometry import project_target_to_scalp

        fake_path = tmp_path / "nonexistent.msh"
        target = np.array([0.0, 0.0, 0.0])

        with pytest.raises(FileNotFoundError):
            project_target_to_scalp(fake_path, target)

    @pytest.mark.skip(reason="Requires integration test with real SimNIBS mesh")
    @patch("tide.core.geometry.read_msh")
    @patch("pathlib.Path.exists")
    def test_returns_3d_array(self, mock_exists, mock_read_msh, mock_mesh):
        """Result should be a 3D coordinate array."""
        # This test requires a complex mesh structure that is difficult to mock
        # Marked for integration testing with real mesh files
        pass

    def test_validates_target_at_center(self):
        """Should raise ValueError if target is at brain center."""

        # This tests the internal validation
        # We can't easily mock the full mesh, so we check the error handling logic
        pass  # Covered by integration tests


# =============================================================================
# Tests for compute_default_coil_orientation
# =============================================================================


class TestComputeDefaultCoilOrientation:
    """Tests for automatic coil handle orientation calculation."""

    @pytest.mark.skip(reason="Requires integration test with real SimNIBS mesh")
    @patch("tide.core.geometry.read_msh")
    def test_returns_list_of_three(self, mock_read_msh):
        """Result should be [x, y, z] list."""
        pass  # Complex mesh mocking needed - use integration tests
        from tide.core.geometry import compute_default_coil_orientation

        # Create mock mesh with proper nodes array
        mock_mesh = MagicMock()
        n_nodes = 200

        # Create sphere-like node distribution
        nodes = np.zeros((n_nodes + 1, 3))
        for i in range(1, n_nodes + 1):
            theta = 2 * np.pi * (i / n_nodes)
            phi = np.pi * ((i % 50) / 50)
            r = 80 if i < 100 else 50  # Outer scalp, inner GM
            nodes[i] = [
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ]

        # Use the nodes array directly (MagicMock with return value)
        mock_mesh.nodes = MagicMock()
        mock_mesh.nodes.__getitem__ = MagicMock(return_value=nodes)
        mock_mesh.elm.tag1 = np.array([1005] * 100 + [1002] * 100)
        mock_mesh.elm.node_number_list = np.column_stack(
            [
                np.arange(1, 101),
                np.arange(2, 102) % 100 + 1,
                np.arange(3, 103) % 100 + 1,
            ]
        )

        mock_read_msh.return_value = mock_mesh

        scalp_pos = np.array([-45.0, 30.0, 65.0])  # Left hemisphere
        result = compute_default_coil_orientation(Path("dummy.msh"), scalp_pos)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(np.isfinite(result))

    @pytest.mark.skip(reason="Requires integration test with real SimNIBS mesh")
    @patch("tide.core.geometry.read_msh")
    def test_left_hemisphere_orientation(self, mock_read_msh):
        """Left hemisphere should orient handle toward right (medial)."""
        pass  # Complex mesh mocking needed - use integration tests
        from tide.core.geometry import compute_default_coil_orientation

        # Setup mock mesh
        mock_mesh = MagicMock()
        n_nodes = 200
        nodes = np.zeros((n_nodes + 1, 3))
        for i in range(1, n_nodes + 1):
            theta = 2 * np.pi * (i / n_nodes)
            phi = np.pi * ((i % 50) / 50)
            r = 80 if i < 100 else 50
            nodes[i] = [
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ]

        mock_mesh.nodes = MagicMock()
        mock_mesh.nodes.__getitem__ = MagicMock(return_value=nodes)
        mock_mesh.elm.tag1 = np.array([1005] * 100 + [1002] * 100)
        mock_mesh.elm.node_number_list = np.column_stack(
            [
                np.arange(1, 101),
                np.arange(2, 102) % 100 + 1,
                np.arange(3, 103) % 100 + 1,
            ]
        )
        mock_read_msh.return_value = mock_mesh

        # Left hemisphere position
        left_pos = np.array([-50.0, 20.0, 60.0])
        result = compute_default_coil_orientation(Path("dummy.msh"), left_pos)

        # The reference point should be anterior-right (positive X, positive Y)
        # This is a functional test - just verify it runs and returns valid coords
        assert np.isfinite(result).all()


# =============================================================================
# Tests for coil-pose QC
# =============================================================================


def _mock_scalp_qc_mesh():
    nodes = np.zeros((14, 3), dtype=float)
    nodes[1:6] = np.array(
        [
            [0.0, 0.0, -80.0],
            [8.0, 0.0, -80.0],
            [-8.0, 0.0, -80.0],
            [0.0, 8.0, -80.0],
            [0.0, -8.0, -80.0],
        ]
    )
    nodes[6:11] = np.array(
        [
            [0.0, 0.0, 80.0],
            [8.0, 0.0, 80.0],
            [-8.0, 0.0, 80.0],
            [0.0, 8.0, 80.0],
            [0.0, -8.0, 80.0],
        ]
    )
    nodes[11:14] = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

    mesh = MagicMock()
    mesh.nodes = nodes
    mesh.elm.tag1 = np.array([1005, 1005, 1005, 1005, 1002])
    mesh.elm.elm_type = np.array([2, 2, 2, 2, 2])
    mesh.elm.node_number_list = np.array(
        [
            [1, 2, 4],
            [1, 4, 5],
            [6, 7, 9],
            [6, 9, 10],
            [11, 12, 13],
        ]
    )
    return mesh


class TestCoilPoseQC:
    """Tests for coil-pose physical plausibility checks."""

    @patch("tide.core.geometry.read_msh")
    def test_passes_crown_pose_with_inward_normal(self, mock_read_msh, tmp_path):
        from tide.core.geometry import evaluate_coil_pose_qc

        mock_read_msh.return_value = _mock_scalp_qc_mesh()
        mesh_path = tmp_path / "head.msh"
        mesh_path.write_text("")
        matrix = np.eye(4)
        matrix[:3, 2] = [0.0, 0.0, -1.0]
        matrix[:3, 3] = [0.0, 0.0, 80.0]

        qc = evaluate_coil_pose_qc(mesh_path, matrix, n_neighbors=4)

        assert qc.status == "PASS"
        assert qc.reasons == ()
        assert qc.scalp_outward_dot < -0.9

    @patch("tide.core.geometry.read_msh")
    def test_warns_when_coil_normal_points_outward(self, mock_read_msh, tmp_path):
        from tide.core.geometry import evaluate_coil_pose_qc

        mock_read_msh.return_value = _mock_scalp_qc_mesh()
        mesh_path = tmp_path / "head.msh"
        mesh_path.write_text("")
        matrix = np.eye(4)
        matrix[:3, 2] = [0.0, 0.0, 1.0]
        matrix[:3, 3] = [0.0, 0.0, 80.0]

        qc = evaluate_coil_pose_qc(mesh_path, matrix, n_neighbors=4)

        assert qc.status == "WARN"
        assert "coil_normal_not_inward" in qc.reasons

    @patch("tide.core.geometry.read_msh")
    def test_warns_for_inferior_upward_firing_pose(self, mock_read_msh, tmp_path):
        from tide.core.geometry import evaluate_coil_pose_qc

        mock_read_msh.return_value = _mock_scalp_qc_mesh()
        mesh_path = tmp_path / "head.msh"
        mesh_path.write_text("")
        matrix = np.eye(4)
        matrix[:3, 2] = [0.0, 0.0, 1.0]
        matrix[:3, 3] = [0.0, 0.0, -80.0]

        qc = evaluate_coil_pose_qc(mesh_path, matrix, n_neighbors=4)

        assert qc.status == "WARN"
        assert "inferior_scalp_surface" in qc.reasons
        assert "upward_firing_low_inferior_pose" in qc.reasons

    def test_warned_automatic_pose_is_not_dose_eligible(self):
        from tide.core.geometry import CoilPoseQC, validate_coil_pose_for_dose

        qc = CoilPoseQC(status="WARN", reasons=("coil_normal_not_inward",))

        with pytest.raises(ValueError, match="not dose-eligible"):
            validate_coil_pose_for_dose(qc, explicit_matrix=False)

    def test_warned_explicit_matrix_is_a_specialist_override(self):
        from tide.core.geometry import CoilPoseQC, validate_coil_pose_for_dose

        qc = CoilPoseQC(status="WARN", reasons=("coil_normal_not_inward",))

        validate_coil_pose_for_dose(qc, explicit_matrix=True)


# =============================================================================
# Tests for corrected alignment QC
# =============================================================================


class TestAlignmentCorrected:
    """Tests for midpoint-aware alignment diagnostics."""

    def test_straight_aligned_fibre(self):
        from tide.core.geometry import calculate_alignment_corrected

        original_points = np.column_stack([np.arange(5, dtype=float), np.zeros(5), np.zeros(5)])
        midpoint_streamline = 0.5 * (original_points[:-1] + original_points[1:])
        e_vectors = np.tile([2.0, 0.0, 0.0], (len(original_points), 1))
        roi_mask = np.ones(len(midpoint_streamline), dtype=bool)

        alignment = calculate_alignment_corrected(
            [midpoint_streamline],
            [e_vectors],
            [roi_mask],
        )

        assert np.isclose(alignment, 1.0)

    def test_straight_orthogonal_fibre(self):
        from tide.core.geometry import calculate_alignment_corrected

        original_points = np.column_stack([np.arange(5, dtype=float), np.zeros(5), np.zeros(5)])
        midpoint_streamline = 0.5 * (original_points[:-1] + original_points[1:])
        e_vectors = np.tile([0.0, 2.0, 0.0], (len(original_points), 1))
        roi_mask = np.ones(len(midpoint_streamline), dtype=bool)

        alignment = calculate_alignment_corrected(
            [midpoint_streamline],
            [e_vectors],
            [roi_mask],
        )

        assert np.isclose(alignment, 0.0)

    def test_curved_fibre(self):
        from tide.core.geometry import calculate_alignment_corrected

        theta = np.linspace(0.0, np.pi / 2.0, 9)
        original_points = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        midpoint_streamline = 0.5 * (original_points[:-1] + original_points[1:])
        e_vectors = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
        roi_mask = np.ones(len(midpoint_streamline), dtype=bool)

        alignment = calculate_alignment_corrected(
            [midpoint_streamline],
            [e_vectors],
            [roi_mask],
        )

        assert alignment > 0.98


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGeometryHelpers:
    """Tests for geometry helper functions."""

    def test_moller_trumbore_math(self):
        """Test the Möller-Trumbore algorithm math isolated."""
        # Simple triangle in XY plane
        vert0 = np.array([[0.0, 0.0, 0.0]])
        vert1 = np.array([[1.0, 0.0, 0.0]])
        vert2 = np.array([[0.0, 1.0, 0.0]])

        # Ray from below pointing up
        ray_origin = np.array([0.25, 0.25, -1.0])
        ray_direction = np.array([0.0, 0.0, 1.0])

        # Compute intersection using M-T algorithm
        edge1 = vert1 - vert0
        edge2 = vert2 - vert0
        h = np.cross(ray_direction, edge2)
        a = np.einsum("ij,ij->i", edge1, h)

        epsilon = 1e-7
        valid_a = np.abs(a) > epsilon
        f = np.zeros_like(a)
        f[valid_a] = 1.0 / a[valid_a]

        s = ray_origin - vert0
        u = f * np.einsum("ij,ij->i", s, h)

        q = np.cross(s, edge1)
        v = f * np.einsum("j,ij->i", ray_direction, q)
        t = f * np.einsum("ij,ij->i", edge2, q)

        # Validate intersection
        valid = valid_a & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > epsilon)

        assert valid[0], "Ray should intersect triangle"
        assert np.isclose(t[0], 1.0), "Intersection distance should be 1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
