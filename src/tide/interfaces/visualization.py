"""
Basic Visualization Module (FURY/DIPY)
======================================
Provides basic streamline visualization using FURY/DIPY.
For advanced 3D visualization, use visualization_3d module.
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Check for visualization libraries
try:
    from dipy.viz import actor, colormap, window

    FURY_AVAILABLE = True
except ImportError:
    try:
        from fury import actor, colormap, window

        FURY_AVAILABLE = True
    except ImportError:
        FURY_AVAILABLE = False
        window = None
        actor = None
        colormap = None


def save_af_visualization(
    streamlines: list,
    values: list,
    roi_center: list,
    roi_radius: float,
    output_dir: Path,
    prefix: str,
):
    """
    Generates basic PNG snapshots of the bundle colored by AF values.

    Args:
        streamlines: List of streamline coordinates
        values: List of AF values per streamline
        roi_center: ROI center coordinates
        roi_radius: ROI radius
        output_dir: Output directory
        prefix: Filename prefix
    """
    if not FURY_AVAILABLE:
        log.debug("FURY/DIPY visualization not available. Skipping basic visualization.")
        return

    if not streamlines or not values:
        log.debug(f"No streamlines to visualize for {prefix}")
        return

    try:
        # AF is signed upstream (polarity preserved); colour scaling uses
        # magnitude so peak activation of either polarity maps to the top.
        abs_values = [np.abs(v) for v in values]
        all_vals = np.concatenate(abs_values)
        _ = np.max(all_vals) if len(all_vals) > 0 else 1.0

        # Create streamline actor
        lut = colormap.create_colormap(np.linspace(0, 1, 256), name="jet")
        streamlines_obj = np.array(streamlines, dtype=object)
        values_obj = np.array(abs_values, dtype=object)
        streamline_actor = actor.line(
            streamlines_obj, values_obj, linewidth=0.5, lookup_colormap=lut
        )

        # Create ROI sphere
        sphere_actor = None
        if roi_center is not None:
            sphere_actor = actor.sphere(
                centers=np.array([roi_center]),
                radii=roi_radius,
                colors=np.array([1, 1, 1]),
                opacity=0.3,
            )

        # Setup scene
        scene = window.Scene()
        scene.add(streamline_actor)
        if sphere_actor:
            scene.add(sphere_actor)
        scene.add(actor.scalar_bar(lut, title="AF (V/m²)"))

        # Center camera
        center = (
            np.array(roi_center)
            if roi_center is not None
            else np.mean(np.concatenate(streamlines), axis=0)
        )

        views = {
            "axial": (0, 0, 1),
            "coronal": (0, 1, 0),
            "sagittal": (1, 0, 0),
            "oblique": (1, 1, 1),
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for view_name, cam_vec in views.items():
            scene.set_camera(position=None, focal_point=center, view_up=(0, 0, 1))
            cam_pos = center + np.array(cam_vec) * 200
            scene.set_camera(position=cam_pos, focal_point=center, view_up=(0, 0, 1))
            scene.zoom(1.0)

            out_path = output_dir / f"{prefix}_view_{view_name}.png"
            window.record(scene, out_path=str(out_path), size=(1024, 768))

        log.debug(f"Saved basic visualizations to {output_dir}")

    except Exception as e:
        log.debug(f"Basic visualization failed: {e}")
