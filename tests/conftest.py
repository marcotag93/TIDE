"""
Pytest Configuration and Fixtures for TIDE Pipeline Tests
==========================================================
Provides mock data generators and shared fixtures for testing.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

# =============================================================================
# Mock Data Generators
# =============================================================================


def generate_mock_streamline(
    n_points: int = 50,
    start: np.ndarray = None,
    direction: np.ndarray = None,
    step_size: float = 0.5,
    curvature: float = 0.0,
) -> np.ndarray:
    """
    Generate a mock streamline with optional curvature.

    Args:
        n_points: Number of points in the streamline
        start: Starting coordinate [x, y, z]
        direction: Initial direction vector
        step_size: Distance between points (mm)
        curvature: Curvature factor (0 = straight, higher = more curved)

    Returns:
        Nx3 numpy array of streamline coordinates
    """
    if start is None:
        start = np.array([0.0, 0.0, 0.0])
    if direction is None:
        direction = np.array([1.0, 0.0, 0.0])

    direction = direction / np.linalg.norm(direction)
    points = [start.copy()]
    current_pos = start.copy()
    current_dir = direction.copy()

    for i in range(n_points - 1):
        # Add curvature by rotating direction
        if curvature > 0:
            angle = curvature * step_size
            # Rotate in XY plane
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
            current_dir = rot @ current_dir

        current_pos = current_pos + current_dir * step_size
        points.append(current_pos.copy())

    return np.array(points)


def generate_mock_efield(
    streamline: np.ndarray,
    magnitude: float = 100.0,
    alignment: float = 1.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Generate mock E-field vectors along a streamline.

    Args:
        streamline: Nx3 streamline coordinates
        magnitude: E-field magnitude (V/m)
        alignment: 0-1, how aligned E-field is with tangent
        noise_std: Standard deviation of noise to add

    Returns:
        Nx3 array of E-field vectors
    """
    _ = len(streamline)

    # Calculate tangent vectors
    tangents = np.gradient(streamline, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (tangent_norms + 1e-9)

    # Create perpendicular component
    perp = np.zeros_like(tangents)
    perp[:, 0] = -tangents[:, 1]
    perp[:, 1] = tangents[:, 0]
    perp_norms = np.linalg.norm(perp, axis=1, keepdims=True)
    perp = perp / (perp_norms + 1e-9)

    # Combine aligned and perpendicular components
    e_field = alignment * tangents + (1 - alignment) * perp
    e_field = e_field / np.linalg.norm(e_field, axis=1, keepdims=True)
    e_field = e_field * magnitude

    # Add noise if requested
    if noise_std > 0:
        e_field += np.random.normal(0, noise_std, e_field.shape)

    return e_field


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def single_straight_streamline():
    """Single straight streamline along X-axis."""
    return generate_mock_streamline(n_points=50, step_size=0.5, curvature=0.0)


@pytest.fixture
def single_curved_streamline():
    """Single curved streamline with moderate curvature."""
    return generate_mock_streamline(n_points=50, step_size=0.5, curvature=0.1)


@pytest.fixture
def bundle_straight(n_fibers: int = 10):
    """Bundle of parallel straight streamlines."""
    streamlines = []
    for i in range(n_fibers):
        offset = np.array([0.0, i * 2.0, 0.0])
        sl = generate_mock_streamline(n_points=50, start=offset, step_size=0.5)
        streamlines.append(sl)
    return streamlines


@pytest.fixture
def bundle_with_efield():
    """Bundle of streamlines with corresponding E-field vectors."""
    streamlines = []
    e_fields = []

    for i in range(5):
        offset = np.array([0.0, i * 2.0, 0.0])
        sl = generate_mock_streamline(n_points=30, start=offset, step_size=0.5)
        ef = generate_mock_efield(sl, magnitude=100.0, alignment=0.8)
        streamlines.append(sl)
        e_fields.append(ef)

    return streamlines, e_fields


@pytest.fixture
def short_streamline():
    """Very short streamline (edge case)."""
    return generate_mock_streamline(n_points=3, step_size=0.5)


@pytest.fixture
def empty_streamlines():
    """Empty list of streamlines (edge case)."""
    return []


@pytest.fixture
def mock_roi_center():
    """ROI center coordinate."""
    return np.array([10.0, 5.0, 0.0])


@pytest.fixture
def mock_scalp_coords():
    """Mock scalp coordinates."""
    return np.array([-45.0, 30.0, 65.0])


@pytest.fixture
def mock_brain_center():
    """Mock brain center coordinates."""
    return np.array([0.0, 0.0, 0.0])


# =============================================================================
# Mock SimNIBS Mesh Fixture
# =============================================================================


@pytest.fixture
def mock_mesh():
    """
    Create a simple mock mesh with scalp and GM surfaces.
    Returns a mock object that mimics SimNIBS mesh structure.
    """
    mesh = MagicMock()

    # Create simple sphere-like node positions
    n_nodes = 100
    theta = np.linspace(0, 2 * np.pi, n_nodes)
    phi = np.linspace(0, np.pi, n_nodes)

    # Dummy nodes (not physically accurate, just for testing)
    nodes = np.zeros((n_nodes + 1, 3))  # +1 for 1-indexed access
    for i in range(1, n_nodes + 1):
        nodes[i] = [
            50 * np.sin(phi[i - 1]) * np.cos(theta[i - 1]),
            50 * np.sin(phi[i - 1]) * np.sin(theta[i - 1]),
            50 * np.cos(phi[i - 1]),
        ]

    mesh.nodes.__getitem__ = lambda s: nodes
    mesh.elm.tag1 = np.array([1005] * 50 + [1002] * 50)  # Scalp + GM tags
    mesh.elm.node_number_list = np.column_stack(
        [np.arange(1, 51), np.arange(2, 52) % 50 + 1, np.arange(3, 53) % 50 + 1]
    )
    mesh.elm.elm_type = np.array([2] * 100)  # Triangle elements

    return mesh


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_t1w_path(tmp_path):
    """Mock T1w path (doesn't need to exist for unit tests)."""
    return tmp_path / "mock_t1w.nii.gz"
