#!/usr/bin/env python3
"""
Professional 3D Visualization Module for TIDE Pipeline
========================================================
Creates publication-quality 3D visualizations of E-field and AF along bundles
with brain anatomical context using PyVista.

Features:
- Brain surface rendering with E-field colormap
- Bundle tubes colored by AF values
- ROI boundary visualization
- Multiple camera angles
- Interactive HTML export
- Multi-view composite figures
- Depth analysis visualizations

Author: TIDE Pipeline
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _drop_dead_display() -> bool:
    """Unset ``DISPLAY`` when the X server is unreachable.

    Stale ``DISPLAY`` env vars (e.g. from a closed SSH X11 forward) make
    VTK abort with ``bad X server connection`` before Python can catch
    the failure. Returns ``True`` if a usable DISPLAY remains afterwards.
    """
    import os
    import subprocess

    display = os.environ.get("DISPLAY")
    if not display:
        return False
    try:
        result = subprocess.run(
            ["xset", "q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    log.debug(
        "Dropping unreachable DISPLAY=%s; will use Xvfb if available.",
        display,
    )
    os.environ.pop("DISPLAY", None)
    return False


def _ensure_offscreen_display() -> None:
    """Provide a working virtual display for VTK when none is available.

    SimNIBS-bundled VTK on Linux requires an X RenderWindow even with
    ``OFF_SCREEN=True``. When no usable DISPLAY exists, spawn Xvfb so
    rendering proceeds without aborting the pipeline.
    """
    import os
    import shutil

    if os.environ.get("DISPLAY"):
        return
    if shutil.which("Xvfb") is None:
        log.warning(
            "No DISPLAY and Xvfb not installed; 3D visualizations will be "
            "skipped. Install xvfb (Debian/Ubuntu: 'sudo apt install xvfb') "
            "or run on a workstation with a working display."
        )
        return
    try:
        import pyvista as _pv

        _pv.start_xvfb()
        log.debug("Started Xvfb at DISPLAY=%s", os.environ.get("DISPLAY"))
    except Exception as exc:
        log.warning("Failed to start Xvfb (3D viz disabled): %s", exc)


# Force offscreen rendering before importing pyvista; VTK reads DISPLAY
# at first RenderWindow creation, so the env must be set up first.
import os as _os  # noqa: E402

_os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
_drop_dead_display()
_ensure_offscreen_display()

# Check for PyVista
try:
    import pyvista as pv

    pv.OFF_SCREEN = True
    import vtk

    vtk.vtkObject.GlobalWarningDisplayOff()
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    log.debug("PyVista not available. 3D visualization disabled.")

# Check for matplotlib
try:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# SimNIBS mesh reading
try:
    from simnibs.mesh_tools import mesh_io

    read_msh = mesh_io.read_msh
except ImportError:
    try:
        from simnibs import read_msh
    except ImportError:
        read_msh = None


@dataclass
class VisualizationConfig:
    """Configuration for 3D visualizations."""

    # E-field colormap
    efield_cmap: str = "YlOrRd"
    efield_vmin: float = 0.0
    efield_vmax: float = 100.0

    # AF colormap
    af_cmap: str = "plasma"
    af_vmin: float = 0.0
    af_vmax: float = None  # Auto-scale if None

    # ROI visualization
    roi_color: str = "lime"
    roi_line_width: float = 4.0
    roi_opacity: float = 0.3

    # Bundle visualization
    bundle_tube_radius: float = 0.3
    bundle_tube_sides: int = 8
    bundle_opacity: float = 0.9

    # Brain surface
    brain_opacity: float = 0.4
    brain_color: str = "lightgray"

    # Output settings
    window_size: Tuple[int, int] = (1920, 1080)
    dpi: int = 300
    background_color: str = "white"
    interactive_max_streamlines: int = 1500
    interactive_max_points_per_streamline: int = 80
    interactive_decimals: int = 2


class Visualization3D:
    """
    Professional 3D visualization class for TIDE pipeline.

    Handles brain surface extraction, bundle rendering, and multi-view exports.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualization with optional config."""
        self.config = config or VisualizationConfig()
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        if not PYVISTA_AVAILABLE:
            log.warning("PyVista not installed. 3D visualization disabled.")
            return False
        return True

    # =========================================================================
    # BRAIN SURFACE EXTRACTION
    # =========================================================================

    def extract_brain_surface(
        self, mesh_path: Path, surface_tag: int = 1002, extract_efield: bool = True
    ) -> Optional[Tuple[pv.PolyData, Optional[np.ndarray]]]:
        """
        Extract brain (GM) surface from SimNIBS mesh.

        Args:
            mesh_path: Path to SimNIBS .msh file
            surface_tag: Tag for GM surface (1002 in SimNIBS v4)
            extract_efield: Whether to extract E-field values

        Returns:
            Tuple of (PyVista surface, E-field values) or None if failed
        """
        if not PYVISTA_AVAILABLE or read_msh is None:
            return None

        try:
            mesh = read_msh(str(mesh_path))
            vertices = mesh.nodes.node_coord
            elm_types = mesh.elm.elm_type
            elm_tags = mesh.elm.tag1

            # Find surface triangles
            surface_mask = (elm_types == 2) & (elm_tags == surface_tag)
            surface_indices = np.where(surface_mask)[0]

            if len(surface_indices) == 0:
                # Try alternative tags
                for alt_tag in [1002, 2, 1001, 1]:
                    surface_mask = (elm_types == 2) & (elm_tags == alt_tag)
                    surface_indices = np.where(surface_mask)[0]
                    if len(surface_indices) > 0:
                        break

            if len(surface_indices) == 0:
                log.warning("No brain surface found in mesh")
                return None

            triangles = mesh.elm.node_number_list[surface_indices, :3] - 1
            unique_nodes = np.unique(triangles)

            # Node mapping
            node_mapping = np.zeros(len(vertices), dtype=int) - 1
            for new_idx, old_idx in enumerate(unique_nodes):
                node_mapping[old_idx] = new_idx

            surface_vertices = vertices[unique_nodes]
            surface_faces = node_mapping[triangles]

            # Create PyVista mesh
            faces_pv = np.column_stack([np.full(len(surface_faces), 3), surface_faces]).ravel()
            surface = pv.PolyData(surface_vertices, faces_pv)

            # Extract E-field if available (vectorized element->vertex average)
            efield_values = None
            if extract_efield and hasattr(mesh, "elmdata") and mesh.elmdata:
                for ed in mesh.elmdata:
                    if "magne" in ed.field_name.lower():
                        elm_efield = np.asarray(ed.value)

                        valid_mask = surface_indices < len(elm_efield)
                        elm_vals = elm_efield[surface_indices[valid_mask]]
                        tri_local = surface_faces[valid_mask]  # (N, 3) → 0..M-1 or -1

                        # Replicate per-element scalar across its 3 vertices
                        tri_flat = tri_local.ravel()
                        val_flat = np.repeat(elm_vals, 3)
                        node_mask = tri_flat >= 0

                        node_efield = np.zeros(len(unique_nodes), dtype=np.float64)
                        node_counts = np.zeros(len(unique_nodes), dtype=np.float64)
                        np.add.at(node_efield, tri_flat[node_mask], val_flat[node_mask])
                        np.add.at(node_counts, tri_flat[node_mask], 1.0)

                        node_counts[node_counts == 0] = 1
                        efield_values = node_efield / node_counts
                        surface.point_data["E-field"] = efield_values
                        break

            return surface, efield_values

        except Exception as e:
            log.warning(f"Failed to extract brain surface: {e}")
            return None

    # =========================================================================
    # BUNDLE VISUALIZATION
    # =========================================================================

    def create_bundle_tubes(
        self,
        streamlines: List[np.ndarray],
        scalar_values: Optional[List[np.ndarray]] = None,
        scalar_name: str = "AF",
        radius: Optional[float] = None,
        subsample: int = 1,
    ) -> Optional[pv.PolyData]:
        """
        Create tube mesh from streamlines with optional scalar coloring.

        Args:
            streamlines: List of streamline coordinates
            scalar_values: Optional list of scalar values per streamline
            scalar_name: Name for the scalar data
            radius: Tube radius (uses config default if None)
            subsample: Use every Nth streamline

        Returns:
            Combined PyVista tube mesh or None
        """
        if not PYVISTA_AVAILABLE:
            return None

        radius = radius or self.config.bundle_tube_radius
        n_sides = self.config.bundle_tube_sides

        tubes = []
        n_streamlines = len(streamlines)

        for i in range(0, n_streamlines, subsample):
            sl = streamlines[i]
            if len(sl) < 2:
                continue

            try:
                spline = pv.Spline(sl, n_points=len(sl))

                if scalar_values is not None and i < len(scalar_values):
                    sv = scalar_values[i]
                    if len(sv) == len(sl):
                        spline.point_data[scalar_name] = sv
                    elif len(sv) == len(sl) - 1:
                        # Interpolate segment values to points
                        sv_interp = np.concatenate([[sv[0]], (sv[:-1] + sv[1:]) / 2, [sv[-1]]])
                        if len(sv_interp) == len(sl):
                            spline.point_data[scalar_name] = sv_interp

                tube = spline.tube(radius=radius, n_sides=n_sides)
                tubes.append(tube)

            except Exception:
                continue

        if not tubes:
            return None

        # Batch merge for efficiency
        combined = self._batch_merge(tubes)
        return combined

    def _batch_merge(self, meshes: List[pv.PolyData], batch_size: int = 100) -> pv.PolyData:
        """Efficiently merge multiple meshes using batch processing."""
        if len(meshes) == 0:
            return None
        if len(meshes) == 1:
            return meshes[0]

        current = meshes
        while len(current) > 1:
            merged = []
            for i in range(0, len(current), batch_size):
                batch = current[i : i + batch_size]
                if len(batch) == 1:
                    merged.append(batch[0])
                else:
                    m = batch[0]
                    for mesh in batch[1:]:
                        m = m.merge(mesh)
                    merged.append(m)
            current = merged

        return current[0]

    # =========================================================================
    # ROI VISUALIZATION
    # =========================================================================

    def create_roi_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: Optional[str] = None,
        opacity: Optional[float] = None,
    ) -> pv.PolyData:
        """Create a sphere mesh for ROI visualization."""
        if not PYVISTA_AVAILABLE:
            return None

        sphere = pv.Sphere(radius=radius, center=center, theta_resolution=30, phi_resolution=30)
        return sphere

    def find_roi_boundary_on_surface(
        self, surface: pv.PolyData, roi_center: np.ndarray, roi_radius: float
    ) -> Optional[pv.PolyData]:
        """
        Find the ROI boundary on the brain surface.

        Returns edges where surface crosses ROI boundary.
        """
        if not PYVISTA_AVAILABLE:
            return None

        try:
            # Calculate distance from ROI center
            points = surface.points
            distances = np.linalg.norm(points - roi_center, axis=1)

            # Find vertices inside ROI
            roi_mask = distances <= roi_radius

            # Get faces
            faces = surface.faces.reshape(-1, 4)[:, 1:4]

            # Find boundary edges
            boundary_edges = []
            for face in faces:
                in_roi = [roi_mask[v] for v in face]
                edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
                for i, (va, vb) in enumerate(edges):
                    if in_roi[i % 3] != in_roi[(i + 1) % 3]:
                        boundary_edges.append((va, vb))

            if not boundary_edges:
                return None

            # Create line mesh
            edge_points = []
            lines = []
            for i, (v0, v1) in enumerate(boundary_edges):
                edge_points.append(points[v0])
                edge_points.append(points[v1])
                lines.extend([2, 2 * i, 2 * i + 1])

            edge_points = np.array(edge_points)
            lines = np.array(lines)

            boundary_mesh = pv.PolyData(edge_points, lines=lines)
            return boundary_mesh

        except Exception as e:
            log.debug(f"Failed to extract ROI boundary: {e}")
            return None

    # =========================================================================
    # MULTI-VIEW RENDERING
    # =========================================================================

    def render_multiview(
        self,
        output_dir: Path,
        prefix: str,
        brain_surface: Optional[pv.PolyData] = None,
        bundle_mesh: Optional[pv.PolyData] = None,
        roi_center: Optional[np.ndarray] = None,
        roi_radius: Optional[float] = None,
        roi_boundary: Optional[pv.PolyData] = None,
        scalar_name: str = "AF",
        scalar_range: Optional[Tuple[float, float]] = None,
        create_composite: bool = True,
        create_html: bool = True,
        interactive_streamlines: Optional[List[np.ndarray]] = None,
        interactive_scalars: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Path]:
        """
        Render multiple views and create outputs.

        Args:
            output_dir: Directory for output files
            prefix: Filename prefix
            brain_surface: Brain surface mesh
            bundle_mesh: Bundle tube mesh
            roi_center: ROI center coordinates
            roi_radius: ROI radius
            roi_boundary: ROI boundary lines on surface
            scalar_name: Name of scalar data to visualize
            scalar_range: (min, max) for scalar colormap
            create_composite: Create multi-view composite image
            create_html: Create interactive HTML file
            interactive_streamlines: Streamlines for the lightweight HTML preview
            interactive_scalars: Scalar arrays for the lightweight HTML preview

        Returns:
            Dictionary of output file paths
        """
        if not PYVISTA_AVAILABLE:
            log.warning("PyVista not available for rendering")
            return {}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Determine center and camera distance
        if bundle_mesh is not None:
            center = bundle_mesh.center
            bounds = bundle_mesh.bounds
        elif brain_surface is not None:
            center = brain_surface.center
            bounds = brain_surface.bounds
        else:
            return outputs

        max_extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        cam_distance = max_extent * 1.5

        # Auto-scale scalar range
        if scalar_range is None and bundle_mesh is not None:
            if scalar_name in bundle_mesh.point_data:
                data = bundle_mesh.point_data[scalar_name]
                scalar_range = (0, np.percentile(data, 99))

        # Define views
        views = {
            "lateral_left": {
                "position": (center[0] - cam_distance, center[1], center[2]),
                "focal": center,
                "up": (0, 0, 1),
                "title": "Left Lateral",
            },
            "lateral_right": {
                "position": (center[0] + cam_distance, center[1], center[2]),
                "focal": center,
                "up": (0, 0, 1),
                "title": "Right Lateral",
            },
            "anterior": {
                "position": (center[0], center[1] + cam_distance, center[2]),
                "focal": center,
                "up": (0, 0, 1),
                "title": "Anterior",
            },
            "posterior": {
                "position": (center[0], center[1] - cam_distance, center[2]),
                "focal": center,
                "up": (0, 0, 1),
                "title": "Posterior",
            },
            "superior": {
                "position": (center[0], center[1], center[2] + cam_distance),
                "focal": center,
                "up": (0, 1, 0),
                "title": "Superior",
            },
            "oblique": {
                "position": (
                    center[0] - cam_distance * 0.7,
                    center[1] + cam_distance * 0.5,
                    center[2] + cam_distance * 0.5,
                ),
                "focal": center,
                "up": (0, 0, 1),
                "title": "Oblique",
            },
        }

        view_files = []

        for view_name, view_params in views.items():
            try:
                plotter = pv.Plotter(off_screen=True, window_size=self.config.window_size)
                plotter.set_background(self.config.background_color)

                # Add brain surface
                if brain_surface is not None:
                    if "E-field" in brain_surface.point_data:
                        plotter.add_mesh(
                            brain_surface,
                            scalars="E-field",
                            cmap=self.config.efield_cmap,
                            clim=[self.config.efield_vmin, self.config.efield_vmax],
                            opacity=self.config.brain_opacity,
                            show_scalar_bar=False,
                        )
                    else:
                        plotter.add_mesh(
                            brain_surface,
                            color=self.config.brain_color,
                            opacity=self.config.brain_opacity,
                        )

                # Add bundle
                if bundle_mesh is not None:
                    if scalar_name in bundle_mesh.point_data:
                        plotter.add_mesh(
                            bundle_mesh,
                            scalars=scalar_name,
                            cmap=self.config.af_cmap,
                            clim=scalar_range,
                            opacity=self.config.bundle_opacity,
                            scalar_bar_args={
                                "title": f"{scalar_name} (V/m²)",
                                "title_font_size": 14,
                                "label_font_size": 12,
                                "vertical": True,
                                "position_x": 0.85,
                                "position_y": 0.2,
                                "width": 0.08,
                                "height": 0.5,
                            },
                        )
                    else:
                        plotter.add_mesh(
                            bundle_mesh, color="steelblue", opacity=self.config.bundle_opacity
                        )

                # Add ROI boundary
                if roi_boundary is not None:
                    plotter.add_mesh(
                        roi_boundary,
                        color=self.config.roi_color,
                        line_width=self.config.roi_line_width,
                        render_lines_as_tubes=True,
                    )

                # Add ROI sphere (semi-transparent)
                if roi_center is not None and roi_radius is not None:
                    sphere = self.create_roi_sphere(roi_center, roi_radius)
                    if sphere is not None:
                        plotter.add_mesh(
                            sphere,
                            color=self.config.roi_color,
                            opacity=self.config.roi_opacity,
                            style="wireframe",
                            line_width=1,
                        )

                # Set camera
                plotter.camera.position = view_params["position"]
                plotter.camera.focal_point = view_params["focal"]
                plotter.camera.up = view_params["up"]

                # Add title
                plotter.add_text(
                    view_params["title"], position="upper_left", font_size=12, color="black"
                )

                # Save
                out_path = output_dir / f"{prefix}_{view_name}.png"
                plotter.screenshot(str(out_path), scale=2)
                view_files.append(out_path)
                outputs[view_name] = out_path

                plotter.close()

            except Exception as e:
                log.debug(f"Failed to render {view_name}: {e}")

        # Create composite
        if create_composite and MATPLOTLIB_AVAILABLE and len(view_files) >= 4:
            try:
                composite_path = output_dir / f"{prefix}_composite.png"
                self._create_composite_figure(view_files[:6], composite_path)
                outputs["composite"] = composite_path
            except Exception as e:
                log.debug(f"Failed to create composite: {e}")

        # Create interactive HTML
        if create_html:
            try:
                html_path = output_dir / f"{prefix}_interactive.html"
                self._create_interactive_html(
                    html_path,
                    brain_surface,
                    bundle_mesh,
                    roi_center,
                    roi_radius,
                    scalar_name,
                    scalar_range,
                    interactive_streamlines,
                    interactive_scalars,
                )
                outputs["html"] = html_path
            except Exception as e:
                log.warning(f"Interactive HTML export failed: {e}")

        return outputs

    def _create_composite_figure(self, image_paths: List[Path], output_path: Path):
        """Create a multi-view composite figure."""
        n_images = min(6, len(image_paths))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, img_path in enumerate(image_paths[:n_images]):
            if img_path.exists():
                img = mpimg.imread(str(img_path))
                axes[i].imshow(img)
            axes[i].axis("off")

        for i in range(n_images, 6):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
        plt.close()

    def _interactive_payload(
        self,
        streamlines: Optional[List[np.ndarray]],
        scalar_values: Optional[List[np.ndarray]],
    ) -> List[Dict[str, Any]]:
        if streamlines is None or len(streamlines) == 0:
            return []

        total = len(streamlines)
        stride = max(1, int(np.ceil(total / self.config.interactive_max_streamlines)))
        payload: List[Dict[str, Any]] = []

        for original_idx in list(range(0, total, stride))[
            : self.config.interactive_max_streamlines
        ]:
            sl = np.asarray(streamlines[original_idx], dtype=float)
            if sl.ndim != 2 or sl.shape[1] != 3 or len(sl) < 2:
                continue

            values = None
            if scalar_values is not None and original_idx < len(scalar_values):
                values = np.asarray(scalar_values[original_idx], dtype=float)
                if len(values) != len(sl):
                    values = None

            if len(sl) > self.config.interactive_max_points_per_streamline:
                indices = np.linspace(
                    0,
                    len(sl) - 1,
                    self.config.interactive_max_points_per_streamline,
                    dtype=int,
                )
                sl = sl[indices]
                if values is not None:
                    values = values[indices]

            if values is None:
                values = np.zeros(len(sl), dtype=float)

            values = np.nan_to_num(np.abs(values), nan=0.0, posinf=0.0, neginf=0.0)
            payload.append(
                {
                    "pts": np.round(sl, self.config.interactive_decimals).tolist(),
                    "values": np.round(values, self.config.interactive_decimals).tolist(),
                }
            )

        return payload

    def _create_interactive_html(
        self,
        output_path: Path,
        brain_surface: Optional[pv.PolyData],
        bundle_mesh: Optional[pv.PolyData],
        roi_center: Optional[np.ndarray],
        roi_radius: Optional[float],
        scalar_name: str,
        scalar_range: Optional[Tuple[float, float]],
        interactive_streamlines: Optional[List[np.ndarray]] = None,
        interactive_scalars: Optional[List[np.ndarray]] = None,
    ):
        payload = self._interactive_payload(interactive_streamlines, interactive_scalars)
        if bundle_mesh is not None:
            center = [round(float(v), 2) for v in bundle_mesh.center]
            bounds = [round(float(v), 2) for v in bundle_mesh.bounds]
        elif roi_center is not None:
            center = [round(float(v), 2) for v in roi_center]
            bounds = [
                center[0] - 40,
                center[0] + 40,
                center[1] - 40,
                center[1] + 40,
                center[2] - 40,
                center[2] + 40,
            ]
        else:
            center = [0.0, 0.0, 0.0]
            bounds = [-50, 50, -50, 50, -50, 50]

        vmax = scalar_range[1] if scalar_range is not None else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        roi = {
            "center": [round(float(v), 2) for v in roi_center] if roi_center is not None else None,
            "radius": float(roi_radius) if roi_radius is not None else None,
        }
        html = _build_lightweight_bundle_html(
            streamlines_json=json.dumps(payload, allow_nan=False),
            center_json=json.dumps(center),
            bounds_json=json.dumps(bounds),
            roi_json=json.dumps(roi),
            scalar_name=scalar_name,
            scalar_vmax=round(float(vmax), 2),
            sampled_streamlines=len(payload),
            full_streamlines=(
                len(interactive_streamlines) if interactive_streamlines is not None else 0
            ),
            has_brain_surface=brain_surface is not None,
        )
        output_path.write_text(html, encoding="utf-8")

    # =========================================================================
    # DEPTH ANALYSIS VISUALIZATION
    # =========================================================================

    def render_depth_analysis(
        self,
        output_dir: Path,
        prefix: str,
        streamlines: List[np.ndarray],
        scalar_values: List[np.ndarray],
        scalp_point: np.ndarray,
        scalar_name: str = "AF",
    ) -> Dict[str, Path]:
        """
        Create depth analysis visualizations.

        Args:
            output_dir: Output directory
            prefix: Filename prefix
            streamlines: List of streamline coordinates
            scalar_values: List of scalar values per streamline
            scalp_point: Reference scalp point for depth calculation
            scalar_name: Name of scalar metric

        Returns:
            Dictionary of output file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            return {}

        output_dir = Path(output_dir)
        outputs = {}

        try:
            # Compute depth for all points
            all_points = []
            all_values = []
            all_depths = []

            for sl, sv in zip(streamlines, scalar_values):
                # Align lengths
                n = min(len(sl), len(sv))
                if n < 2:
                    continue

                depths = np.linalg.norm(sl[:n] - scalp_point, axis=1)
                all_points.extend(sl[:n])
                all_values.extend(sv[:n])
                all_depths.extend(depths)

            if not all_depths:
                return outputs

            all_values = np.array(all_values)
            all_depths = np.array(all_depths)

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. Scatter plot
            ax = axes[0, 0]
            scatter = ax.scatter(
                all_depths, all_values, c=all_values, cmap="plasma", alpha=0.3, s=1
            )
            ax.set_xlabel("Depth from scalp (mm)", fontsize=11)
            ax.set_ylabel(f"{scalar_name} (V/m²)", fontsize=11)
            ax.set_title(f"{scalar_name} vs Depth", fontsize=12, fontweight="bold")
            plt.colorbar(scatter, ax=ax, label=scalar_name)
            ax.grid(True, alpha=0.3)

            # 2. Binned statistics
            ax = axes[0, 1]
            depth_bins = np.linspace(np.min(all_depths), np.max(all_depths), 20)
            bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
            bin_indices = np.digitize(all_depths, depth_bins)

            means = []
            stds = []
            maxs = []
            for i in range(1, len(depth_bins)):
                mask = bin_indices == i
                if np.any(mask):
                    means.append(np.mean(all_values[mask]))
                    stds.append(np.std(all_values[mask]))
                    maxs.append(np.max(all_values[mask]))
                else:
                    means.append(0)
                    stds.append(0)
                    maxs.append(0)

            means = np.array(means)
            stds = np.array(stds)

            ax.fill_between(bin_centers, means - stds, means + stds, alpha=0.3, color="steelblue")
            ax.plot(bin_centers, means, "b-", linewidth=2, label="Mean ± SD")
            ax.plot(bin_centers, maxs, "r--", linewidth=1.5, label="Max")
            ax.set_xlabel("Depth from scalp (mm)", fontsize=11)
            ax.set_ylabel(f"{scalar_name} (V/m²)", fontsize=11)
            ax.set_title("Depth Profile", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 3. Histogram
            ax = axes[1, 0]
            ax.hist(all_values, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(
                np.mean(all_values),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(all_values):.1f}",
            )
            ax.axvline(
                np.median(all_values),
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(all_values):.1f}",
            )
            ax.set_xlabel(f"{scalar_name} (V/m²)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"{scalar_name} Distribution", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # 4. Box plot by depth bands
            ax = axes[1, 1]
            depth_bands = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
            band_data = []
            band_labels = []

            for d_min, d_max in depth_bands:
                mask = (all_depths >= d_min) & (all_depths < d_max)
                if np.any(mask):
                    band_data.append(all_values[mask])
                    band_labels.append(f"{d_min}-{d_max}")

            if band_data:
                bp = ax.boxplot(band_data, labels=band_labels, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor("steelblue")
                    patch.set_alpha(0.6)

            ax.set_xlabel("Depth band (mm)", fontsize=11)
            ax.set_ylabel(f"{scalar_name} (V/m²)", fontsize=11)
            ax.set_title("Distribution by Depth", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            plt.suptitle(f"{scalar_name} Depth Analysis", fontsize=14, fontweight="bold")
            plt.tight_layout()

            out_path = output_dir / f"{prefix}_depth_analysis.png"
            plt.savefig(str(out_path), dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            plt.close()

            outputs["depth_analysis"] = out_path

        except Exception as e:
            log.debug(f"Failed to create depth analysis: {e}")

        return outputs

    # =========================================================================
    # ROI-FOCUSED VISUALIZATION
    # =========================================================================

    def render_roi_focused(
        self,
        output_dir: Path,
        prefix: str,
        bundle_mesh: pv.PolyData,
        roi_center: np.ndarray,
        roi_radius: float,
        scalar_name: str = "AF",
        scalar_range: Optional[Tuple[float, float]] = None,
        stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Path]:
        """
        Create ROI-focused close-up visualizations.
        """
        if not PYVISTA_AVAILABLE:
            return {}

        output_dir = Path(output_dir)
        outputs = {}

        cam_distance = roi_radius * 3

        views = {
            "roi_front": (
                (roi_center[0], roi_center[1] + cam_distance, roi_center[2]),
                roi_center,
                (0, 0, 1),
            ),
            "roi_side": (
                (roi_center[0] - cam_distance, roi_center[1], roi_center[2]),
                roi_center,
                (0, 0, 1),
            ),
            "roi_top": (
                (roi_center[0], roi_center[1], roi_center[2] + cam_distance),
                roi_center,
                (0, 1, 0),
            ),
        }

        for view_name, (position, focal, up) in views.items():
            try:
                plotter = pv.Plotter(off_screen=True, window_size=self.config.window_size)
                plotter.set_background(self.config.background_color)

                # Add bundle
                if scalar_name in bundle_mesh.point_data:
                    plotter.add_mesh(
                        bundle_mesh,
                        scalars=scalar_name,
                        cmap=self.config.af_cmap,
                        clim=scalar_range,
                        opacity=1.0,
                        scalar_bar_args={
                            "title": f"{scalar_name} (V/m²)",
                            "vertical": True,
                            "position_x": 0.85,
                        },
                    )
                else:
                    plotter.add_mesh(bundle_mesh, color="steelblue")

                # Add ROI sphere
                sphere = self.create_roi_sphere(roi_center, roi_radius)
                if sphere is not None:
                    plotter.add_mesh(
                        sphere, color=self.config.roi_color, opacity=0.15, style="wireframe"
                    )

                # Set camera
                plotter.camera.position = position
                plotter.camera.focal_point = focal
                plotter.camera.up = up

                # Add stats annotation
                if stats:
                    stats_text = "\n".join([f"{k}: {v:.1f}" for k, v in stats.items()])
                    plotter.add_text(
                        stats_text, position="lower_right", font_size=10, color="black"
                    )

                out_path = output_dir / f"{prefix}_{view_name}.png"
                plotter.screenshot(str(out_path), scale=2)
                outputs[view_name] = out_path
                plotter.close()

            except Exception as e:
                log.debug(f"Failed to render {view_name}: {e}")

        return outputs


def _build_lightweight_bundle_html(
    *,
    streamlines_json: str,
    center_json: str,
    bounds_json: str,
    roi_json: str,
    scalar_name: str,
    scalar_vmax: float,
    sampled_streamlines: int,
    full_streamlines: int,
    has_brain_surface: bool,
) -> str:
    surface_note = (
        "Static PNGs include brain-surface context. "
        if has_brain_surface
        else "Brain-surface context was not available. "
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TIDE Bundle Preview</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    height: 100vh;
    overflow: hidden;
    background: #f4f6f8;
    color: #17202a;
    font-family: Arial, Helvetica, sans-serif;
  }}
  #header {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 0 18px;
    background: rgba(255, 255, 255, 0.96);
    border-bottom: 1px solid #d9dee7;
    z-index: 10;
  }}
  h1 {{
    margin: 0;
    font-size: 16px;
    letter-spacing: 0;
  }}
  .stats {{
    color: #5d6a79;
    font-size: 12px;
  }}
  #viewer {{
    position: fixed;
    top: 56px;
    left: 0;
    right: 0;
    bottom: 0;
  }}
  #legend {{
    position: fixed;
    left: 18px;
    bottom: 18px;
    width: 240px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d9dee7;
    border-radius: 8px;
    z-index: 10;
    font-size: 12px;
    color: #5d6a79;
  }}
  #bar {{
    height: 12px;
    margin: 8px 0 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #37108a, #b6258f, #f47d4a, #f6dd35);
  }}
  #note {{
    position: fixed;
    right: 18px;
    bottom: 18px;
    max-width: 360px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #d9dee7;
    border-radius: 8px;
    z-index: 10;
    color: #5d6a79;
    font-size: 12px;
  }}
</style>
</head>
<body>
<div id="header">
  <h1>TIDE Bundle Interactive Preview</h1>
  <div class="stats">{sampled_streamlines} sampled streamlines from {full_streamlines} total</div>
</div>
<div id="viewer"></div>
<div id="legend">
  <strong>{scalar_name} magnitude</strong>
  <div id="bar"></div>
  <div>0 to {scalar_vmax}</div>
</div>
<div id="note">{surface_note}This preview is sampled for browser performance. Full data are in the TRK, NIfTI, TXT and JSON outputs.</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const streamlines = {streamlines_json};
const center = {center_json};
const bounds = {bounds_json};
const roi = {roi_json};
const scalarVmax = {scalar_vmax};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f6f8);

const viewer = document.getElementById('viewer');
const camera = new THREE.PerspectiveCamera(
  48,
  viewer.clientWidth / viewer.clientHeight,
  0.1,
  5000
);
camera.up.set(0, 0, 1);

const extent = Math.max(
  bounds[1] - bounds[0],
  bounds[3] - bounds[2],
  bounds[5] - bounds[4],
  60
);
camera.position.set(center[0] + extent, center[1] - extent * 0.8, center[2] + extent * 0.65);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(center[0], center[1], center[2]);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const light = new THREE.DirectionalLight(0xffffff, 0.7);
light.position.set(center[0] + 80, center[1] - 40, center[2] + 80);
scene.add(light);

function colorFor(value) {{
  const t = Math.max(0, Math.min(1, value / scalarVmax));
  const color = new THREE.Color();
  color.setHSL(0.72 - t * 0.56, 0.86, 0.34 + t * 0.22);
  return color;
}}

const lineMaterial = new THREE.LineBasicMaterial({{ vertexColors: true, opacity: 0.74, transparent: true }});

for (const sl of streamlines) {{
  const geom = new THREE.BufferGeometry();
  const verts = new Float32Array(sl.pts.flat());
  const colors = new Float32Array(sl.pts.length * 3);
  for (let i = 0; i < sl.pts.length; i += 1) {{
    const color = colorFor(sl.values[i] || 0);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }}
  geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  scene.add(new THREE.Line(geom, lineMaterial));
}}

if (roi.center && roi.radius) {{
  const sphere = new THREE.SphereGeometry(roi.radius, 36, 18);
  const wire = new THREE.WireframeGeometry(sphere);
  const mat = new THREE.LineBasicMaterial({{ color: 0x2e7d32, opacity: 0.42, transparent: true }});
  const mesh = new THREE.LineSegments(wire, mat);
  mesh.position.set(roi.center[0], roi.center[1], roi.center[2]);
  scene.add(mesh);
}}

window.addEventListener('resize', () => {{
  camera.aspect = viewer.clientWidth / viewer.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_bundle_visualization(
    mesh_path: Path,
    streamlines: List[np.ndarray],
    af_values: List[np.ndarray],
    roi_center: np.ndarray,
    roi_radius: float,
    output_dir: Path,
    prefix: str,
    config: Optional[VisualizationConfig] = None,
    scalp_point: Optional[np.ndarray] = None,
) -> Dict[str, Path]:
    """
    High-level function to generate all visualizations for a bundle.

    Args:
        mesh_path: Path to SimNIBS mesh
        streamlines: List of streamline coordinates
        af_values: List of AF values per streamline
        roi_center: ROI center coordinates
        roi_radius: ROI radius
        output_dir: Output directory
        prefix: Filename prefix
        config: Optional visualization config
        scalp_point: Optional scalp point for depth analysis

    Returns:
        Dictionary of all output file paths
    """
    if not PYVISTA_AVAILABLE:
        log.warning("PyVista not available. Skipping 3D visualization.")
        return {}

    viz = Visualization3D(config)
    all_outputs = {}

    # Extract brain surface
    result = viz.extract_brain_surface(mesh_path)
    brain_surface = result[0] if result else None

    # Create bundle tubes
    bundle_mesh = viz.create_bundle_tubes(streamlines, af_values, scalar_name="AF")

    if bundle_mesh is None:
        log.warning("Failed to create bundle mesh")
        return {}

    # Get ROI boundary
    roi_boundary = None
    if brain_surface is not None:
        roi_boundary = viz.find_roi_boundary_on_surface(brain_surface, roi_center, roi_radius)

    # Determine scalar range. AF is signed (polarity preserved); the colour bar
    # uses activation magnitude so peaks of either polarity map to the upper end.
    nonempty = [v for v in af_values if len(v) > 0] if af_values else []
    if nonempty:
        all_af_signed = np.concatenate(nonempty)
        all_af = np.abs(all_af_signed)
        scalar_range = (0.0, float(np.percentile(all_af, 99)))
    else:
        all_af_signed = None
        all_af = None
        scalar_range = None

    # Render multi-view
    outputs = viz.render_multiview(
        output_dir,
        prefix,
        brain_surface=brain_surface,
        bundle_mesh=bundle_mesh,
        roi_center=roi_center,
        roi_radius=roi_radius,
        roi_boundary=roi_boundary,
        scalar_name="AF",
        scalar_range=scalar_range,
        interactive_streamlines=streamlines,
        interactive_scalars=af_values,
    )
    all_outputs.update(outputs)

    # ROI-focused views — summary statistics use |AF| magnitude
    stats = {
        "Mean AF": float(np.mean(all_af)) if all_af is not None else 0,
        "Max AF": float(np.max(all_af)) if all_af is not None else 0,
        "Median AF": float(np.median(all_af)) if all_af is not None else 0,
    }

    roi_outputs = viz.render_roi_focused(
        output_dir,
        prefix,
        bundle_mesh,
        roi_center,
        roi_radius,
        scalar_name="AF",
        scalar_range=scalar_range,
        stats=stats,
    )
    all_outputs.update(roi_outputs)

    # Depth analysis
    if scalp_point is not None:
        depth_outputs = viz.render_depth_analysis(
            output_dir, prefix, streamlines, af_values, scalp_point, scalar_name="AF"
        )
        all_outputs.update(depth_outputs)

    return all_outputs
