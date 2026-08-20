"""
Configuration Module for TIDE Pipeline
======================================
Handles YAML configuration loading and validation.
"""

import ast
import logging
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from tide.utils import simnibs_env
from tide.utils.artifacts import CACHE_DISABLE_TOKENS

log = logging.getLogger(__name__)


def orientation_is_matrix(orientation: Any) -> bool:
    """True if an orientation value is a full 4x4 matsimnibs matrix."""
    return (
        isinstance(orientation, list)
        and len(orientation) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in orientation)
        and all(
            isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
            for row in orientation
            for value in row
        )
    )


@dataclass
class SubjectConfig:
    """Subject-specific paths and identifiers."""

    id: str
    derivatives_path: Path
    m2m_path: Path
    t1w_path: Path
    weights_cst_path: Optional[Path] = None
    weights_target_path: Optional[Path] = None
    surface_path: Optional[Path] = None
    # Fixed-pose cache base directory; None -> default (~/.cache/tide).
    cache_dir: Optional[Path] = None
    # Optional fixed-pose cache size cap in GB (LRU eviction); None -> unlimited.
    cache_max_size_gb: Optional[float] = None
    # True when `cache_dir: no` disables the fixed-pose cache (same as --no-cache).
    cache_disabled: bool = False

    @property
    def mesh_path(self) -> Path:
        """Return the path to the .msh file inside the m2m folder."""
        msh_files = sorted(self.m2m_path.glob("*.msh"))
        if len(msh_files) == 1:
            return msh_files[0]
        if len(msh_files) > 1:
            raise ValueError(f"Head-model directory contains multiple .msh files: {self.m2m_path}")
        return self.m2m_path / f"{self.id}.msh"


@dataclass
class CoilConfig:
    """TMS coil configuration."""

    coil_model: str
    coil_path: Path
    coil_distance_mm: float
    device_didt_max: float


@dataclass
class TargetConfig:
    """Target/calibration region configuration."""

    label: str
    bundle_path: Path
    coords: Optional[List[float]] = None
    scalp_coords: Optional[List[float]] = None
    orientation: Optional[Union[str, List[float], List[List[float]]]] = None
    medoid_endpoint: bool = False
    measured_rmt_mso: Optional[float] = None
    didt: Optional[float] = None
    mso: Optional[float] = None


@dataclass
class OptionsConfig:
    """Processing options and parameters."""

    roi_size_mm: float
    activation_length_mm: float
    field_mode: str
    adm_optimization: bool
    opt_spatial_resolution: float
    opt_angle_resolution: float
    opt_search_angle: float
    opt_search_radius: float
    # Visualization options
    generate_visualizations: bool = True  # Generate output images (2D plots, NIfTI masks)
    generate_3d_visualization: bool = True  # Generate 3D PyVista visualization
    visualization_dpi: int = 300
    # Streamline quality filter
    max_angular_deviation_deg: float = (
        0.0  # Max angle between consecutive tangents (degrees). 0 = disabled.
    )
    # Surface-constrained (GWI) filter; applies only when subject.files.surface is set
    gwi_threshold_mm: float = 3.0  # Max distance from the GWI surface (mm)
    # Intensity bounds
    mso_floor_ratio: float = 0.70  # Min intensity as fraction of RMT (0.70 = 70%)
    mso_ceiling_ratio: float = 1.40  # Max intensity as fraction of RMT (1.40 = 140%)
    # Parallelization options (grid search)
    max_workers: Optional[int] = None  # Max parallel processes, None = auto
    no_parallel: bool = False  # Force sequential processing
    stmpx_dataset_name: Optional[str] = None


@dataclass
class GridConfig:
    """Grid search configuration."""

    coords: List[float]
    search_radius_mm: float
    step_size_mm: float
    cortex_depth_mm: float
    scalp_coords: Optional[List[float]] = None
    orientation: Optional[Union[str, List[float], List[List[float]]]] = None


@dataclass
class SimNIBSConfig:
    """Main configuration container."""

    subject: SubjectConfig
    coil: CoilConfig
    calibration: TargetConfig
    target: TargetConfig
    options: OptionsConfig
    grid: GridConfig
    workflow: Optional[str] = None

    def get_orientation(self) -> Optional[Union[str, List[float], List[List[float]]]]:
        """Returns orientation or None if empty/not specified."""
        orientation = self.target.orientation

        if orientation is None:
            return None
        if isinstance(orientation, str) and orientation.strip() == "":
            return None
        if isinstance(orientation, list) and len(orientation) == 0:
            return None

        return orientation

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "SimNIBSConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a YAML mapping.")

        # Top-level workflow selection (blank/whitespace treated as unset).
        wf_raw = raw.get("workflow")
        workflow = wf_raw.strip() if isinstance(wf_raw, str) and wf_raw.strip() else None
        metadata = raw.get("_metadata")
        if workflow is None and isinstance(metadata, dict):
            metadata_workflow_raw = metadata.get("workflow")
            metadata_workflow = (
                metadata_workflow_raw.strip()
                if isinstance(metadata_workflow_raw, str) and metadata_workflow_raw.strip()
                else None
            )
            if (
                metadata_workflow == "grid_search"
                and metadata.get("source") == "grid_point_reproducibility"
            ):
                workflow = "estimation"
            elif metadata_workflow == "grid_search":
                workflow = "grid"
            else:
                workflow = metadata_workflow

        def resolve_path(base: Path, p: str) -> Optional[Path]:
            if not p:
                return None
            path_obj = Path(p)
            return path_obj if path_obj.is_absolute() else base / path_obj

        # --- 1. Subject ---
        s_data = raw.get("subject", {})
        der_path = Path(s_data["derivatives_path"])
        cache_raw = s_data.get("cache_dir")
        cache_disabled = (
            isinstance(cache_raw, str) and cache_raw.strip().lower() in CACHE_DISABLE_TOKENS
        )
        cache_dir = None if cache_disabled or not cache_raw else Path(cache_raw).expanduser()
        cache_max_raw = s_data.get("cache_max_size_gb")
        cache_max_size_gb = float(cache_max_raw) if cache_max_raw else None
        files = s_data.get("files", {})

        # Resolve m2m_path - required input
        m2m_raw = s_data.get("m2m_path")
        if not m2m_raw:
            raise ValueError("m2m_path is required in the configuration.")
        m2m_path = Path(m2m_raw) if Path(m2m_raw).is_absolute() else der_path / m2m_raw

        subject_conf = SubjectConfig(
            id=s_data.get("id", der_path.name),
            derivatives_path=der_path,
            m2m_path=m2m_path,
            t1w_path=resolve_path(der_path, files["t1w"]),
            weights_cst_path=resolve_path(der_path, files.get("weights_cst")),
            weights_target_path=resolve_path(der_path, files.get("weights_target")),
            surface_path=resolve_path(der_path, files.get("surface")),
            cache_dir=cache_dir,
            cache_max_size_gb=cache_max_size_gb,
            cache_disabled=cache_disabled,
        )

        # --- 2. Coil ---
        c_data = raw.get("coil", {})
        raw_coil_path = c_data.get("coil_path")
        coil_model = c_data["coil_model"]
        if raw_coil_path:
            configured_coil_path = Path(raw_coil_path)
            if configured_coil_path.suffix.lower() == ".ccd":
                coil_path = configured_coil_path
                coil_model = configured_coil_path.name
            else:
                coil_path = configured_coil_path / coil_model
        else:
            log.debug("Coil path not set. Auto-detecting...")
            coil_dir = _detect_simnibs_coil_path()
            if not coil_dir:
                raise ValueError("Could not auto-detect SimNIBS coil path.")
            coil_path = coil_dir / coil_model

        coil_conf = CoilConfig(
            coil_model=coil_model,
            coil_path=coil_path,
            coil_distance_mm=float(c_data["coil_distance_mm"]),
            device_didt_max=float(c_data["device_didt_max"]),
        )

        # --- 3. Experiment ---
        exp_data = raw.get("experiment", {})

        def parse_val(val):
            """Parse string representations of lists."""
            if isinstance(val, str) and ("[" in val):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    return val
            return val

        # Calibration (M1/CST)
        cal_data = exp_data.get("calibration", {})
        for section_name, section_data in (
            ("calibration", cal_data),
            ("target", exp_data.get("target", {})),
        ):
            if section_data.get("stmpx_file") not in (None, ""):
                raise ValueError(
                    f"experiment.{section_name}.stmpx_file is not supported; convert the "
                    "pose to a SimNIBS orientation matrix before running TIDE."
                )
        if cal_data.get("didt") not in (None, ""):
            raise ValueError(
                "experiment.calibration.didt is not supported; dose calibration uses a "
                "unit field scaled by measured_rmt_mso."
            )
        calibration_conf = TargetConfig(
            label=cal_data.get("label", "M1"),
            bundle_path=resolve_path(der_path, cal_data.get("bundle_path")),
            measured_rmt_mso=float(cal_data.get("measured_rmt_mso", 0)),
            coords=parse_val(cal_data.get("coords")),
            scalp_coords=parse_val(cal_data.get("scalp_coords")),
            orientation=parse_val(cal_data.get("orientation")),
        )

        # Target
        tgt_data = exp_data.get("target", {})
        target_conf = TargetConfig(
            label=tgt_data.get("label", "Target"),
            bundle_path=resolve_path(der_path, tgt_data.get("bundle_path")),
            coords=parse_val(tgt_data.get("coords")),
            scalp_coords=parse_val(tgt_data.get("scalp_coords")),
            orientation=parse_val(tgt_data.get("orientation")),
            medoid_endpoint=bool(tgt_data.get("cortical_medoid", False)),
            didt=float(tgt_data.get("didt")) if tgt_data.get("didt") else None,
            mso=float(tgt_data.get("mso")) if tgt_data.get("mso") else None,
        )

        # Options
        opt_data = raw.get("options", {})
        stmpx_dataset_name = opt_data.get("stmpx_dataset_name")
        if stmpx_dataset_name is not None and not isinstance(stmpx_dataset_name, str):
            raise ValueError("options.stmpx_dataset_name must be a string.")
        if stmpx_dataset_name == "":
            stmpx_dataset_name = None
        options_conf = OptionsConfig(
            roi_size_mm=float(opt_data.get("roi_size_mm", 30.0)),
            activation_length_mm=float(opt_data.get("activation_length_mm", 4.0)),
            field_mode=opt_data.get("field_mode", "af"),
            adm_optimization=opt_data.get("adm_optimization", True),
            opt_spatial_resolution=float(opt_data.get("opt_spatial_resolution", 2.0)),
            opt_angle_resolution=float(opt_data.get("opt_angle_resolution", 10.0)),
            opt_search_angle=float(opt_data.get("opt_search_angle", 30.0)),
            opt_search_radius=float(opt_data.get("opt_search_radius", 10.0)),
            generate_visualizations=(
                opt_data["generate_visualizations"]
                if "generate_visualizations" in opt_data
                else opt_data.get("generate_visualization", True)
            ),
            generate_3d_visualization=opt_data.get("generate_3d_visualization", True),
            visualization_dpi=int(opt_data.get("visualization_dpi", 300)),
            max_workers=(
                int(opt_data["max_workers"]) if opt_data.get("max_workers") is not None else None
            ),
            no_parallel=opt_data.get("no_parallel", False),
            max_angular_deviation_deg=float(opt_data.get("max_angular_deviation_deg", 0.0)),
            gwi_threshold_mm=float(opt_data.get("gwi_threshold_mm", 3.0)),
            mso_floor_ratio=float(opt_data.get("mso_floor_ratio", 0.70)),
            mso_ceiling_ratio=float(opt_data.get("mso_ceiling_ratio", 1.40)),
            stmpx_dataset_name=stmpx_dataset_name,
        )

        # Grid — geometry is nested under the target block. The grid search reuses
        # the target's coords / scalp_coords / orientation as its center and
        # per-point seed; only the search geometry (radius / step / depth) is
        # grid-specific. A legacy top-level experiment.grid block, which carries
        # its own coords / scalp_coords / orientation, is still accepted.
        nested_grid = tgt_data.get("grid")
        if isinstance(nested_grid, dict):
            grid_conf = GridConfig(
                coords=target_conf.coords,
                scalp_coords=target_conf.scalp_coords,
                orientation=target_conf.orientation,
                search_radius_mm=float(nested_grid.get("search_radius_mm", 20.0)),
                step_size_mm=float(nested_grid.get("step_size_mm", 4.0)),
                cortex_depth_mm=float(nested_grid.get("cortex_depth_mm", 2.0)),
            )
        else:
            g_data = exp_data.get("grid", {})
            grid_conf = GridConfig(
                coords=parse_val(g_data.get("coords")),
                scalp_coords=parse_val(g_data.get("scalp_coords")),
                orientation=parse_val(g_data.get("orientation")),
                search_radius_mm=float(g_data.get("search_radius_mm", 20.0)),
                step_size_mm=float(g_data.get("step_size_mm", 4.0)),
                cortex_depth_mm=float(g_data.get("cortex_depth_mm", 2.0)),
            )

        return cls(
            subject=subject_conf,
            coil=coil_conf,
            calibration=calibration_conf,
            target=target_conf,
            options=options_conf,
            grid=grid_conf,
            workflow=workflow,
        )


def _validate_number(
    value: Any,
    key: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number.")
    if minimum is not None:
        below_minimum = value < minimum if minimum_inclusive else value <= minimum
        if below_minimum:
            comparator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{key} must be {comparator} {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}.")


def _validate_vector(value: Any, key: str, *, required: bool = False) -> None:
    if value is None:
        if required:
            raise ValueError(f"{key} is required.")
        return
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{key} must contain exactly three coordinates.")
    for coordinate in value:
        if (
            not isinstance(coordinate, Real)
            or isinstance(coordinate, bool)
            or not math.isfinite(float(coordinate))
        ):
            raise ValueError(f"{key} must contain only finite numbers.")


def _validate_orientation(value: Any, key: str) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{key} cannot be blank.")
        return
    if orientation_is_matrix(value):
        matrix = [[float(element) for element in row] for row in value]
        if any(
            not math.isclose(matrix[3][index], expected, abs_tol=1e-6)
            for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
        ):
            raise ValueError(f"{key} matrix must end with [0, 0, 0, 1].")

        rotation = [row[:3] for row in matrix[:3]]
        for row_index in range(3):
            for column_index in range(3):
                dot_product = sum(
                    rotation[axis][row_index] * rotation[axis][column_index] for axis in range(3)
                )
                expected = 1.0 if row_index == column_index else 0.0
                if not math.isclose(dot_product, expected, abs_tol=1e-4):
                    raise ValueError(f"{key} rotation block must be orthonormal.")
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if not math.isclose(determinant, 1.0, abs_tol=1e-4):
            raise ValueError(f"{key} rotation block must have determinant +1.")
        return
    if isinstance(value, list) and not any(isinstance(item, list) for item in value):
        _validate_vector(value, key)
        return
    raise ValueError(f"{key} must be an EEG label, a three-vector, or a rigid 4x4 matrix.")


def _validate_label(value: Any, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{key} cannot contain path separators or traversal components.")


def _require_file(path: Optional[Path], key: str) -> None:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Required file does not exist ({key}): {path}")


def validate_workflow_config(config: SimNIBSConfig, workflow: str) -> None:
    """Validate workflow-specific inputs before any output is created."""
    if workflow in {"estimation", "grid"} and config.options.field_mode != "af":
        raise ValueError(
            f"options.field_mode '{config.options.field_mode}' is not supported by the "
            f"{workflow} dose workflow; use 'af'. The 'e_parallel' mode is available "
            "only for standard simulation mapping."
        )

    weight_paths = []
    if workflow in {"estimation", "grid"}:
        weight_paths.extend(
            [
                ("subject.files.weights_cst", config.subject.weights_cst_path),
                ("subject.files.weights_target", config.subject.weights_target_path),
            ]
        )
    elif workflow == "simulation":
        weight_paths.append(("subject.files.weights_target", config.subject.weights_target_path))

    for config_key, weight_path in weight_paths:
        if weight_path is not None and not weight_path.exists():
            raise FileNotFoundError(
                f"Configured weight file does not exist ({config_key}): {weight_path}"
            )

    if config.options.field_mode not in {"af", "e_parallel"}:
        raise ValueError("options.field_mode must be 'af' or 'e_parallel'.")

    _validate_label(config.subject.id, "subject.id")
    _validate_label(config.calibration.label, "experiment.calibration.label")
    _validate_label(config.target.label, "experiment.target.label")

    _validate_vector(
        config.calibration.coords,
        "experiment.calibration.coords",
        required=workflow in {"estimation", "grid"},
    )
    _validate_vector(config.calibration.scalp_coords, "experiment.calibration.scalp_coords")
    _validate_vector(
        config.target.coords,
        "experiment.target.coords",
        required=workflow in {"estimation", "grid", "optimization"}
        or (workflow == "simulation" and config.target.bundle_path is not None),
    )
    _validate_vector(config.target.scalp_coords, "experiment.target.scalp_coords")
    _validate_orientation(config.calibration.orientation, "experiment.calibration.orientation")
    _validate_orientation(config.target.orientation, "experiment.target.orientation")
    _validate_vector(
        config.grid.coords, "experiment.target.grid.coords", required=workflow == "grid"
    )
    _validate_vector(config.grid.scalp_coords, "experiment.target.grid.scalp_coords")
    _validate_orientation(config.grid.orientation, "experiment.target.grid.orientation")
    if workflow == "grid" and orientation_is_matrix(config.grid.orientation):
        raise ValueError("experiment.target.orientation cannot be a 4x4 matrix for grid search.")

    _validate_number(
        config.coil.coil_distance_mm,
        "coil.coil_distance_mm",
        minimum=0.0,
    )
    _validate_number(
        config.coil.device_didt_max,
        "coil.device_didt_max",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _validate_number(
        config.options.roi_size_mm,
        "options.roi_size_mm",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _validate_number(
        config.options.activation_length_mm,
        "options.activation_length_mm",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _validate_number(
        config.options.gwi_threshold_mm,
        "options.gwi_threshold_mm",
        minimum=0.0,
        minimum_inclusive=False,
    )
    for key, value in (
        ("options.opt_spatial_resolution", config.options.opt_spatial_resolution),
        ("options.opt_angle_resolution", config.options.opt_angle_resolution),
    ):
        _validate_number(value, key, minimum=0.0, minimum_inclusive=False)
    _validate_number(
        config.options.opt_search_radius,
        "options.opt_search_radius",
        minimum=0.0,
    )
    _validate_number(
        config.options.opt_search_angle,
        "options.opt_search_angle",
        minimum=0.0,
        maximum=360.0,
    )
    _validate_number(
        config.options.max_angular_deviation_deg,
        "options.max_angular_deviation_deg",
        minimum=0.0,
        maximum=180.0,
    )
    _validate_number(config.options.mso_floor_ratio, "options.mso_floor_ratio", minimum=0.0)
    _validate_number(
        config.options.mso_ceiling_ratio,
        "options.mso_ceiling_ratio",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if config.options.mso_floor_ratio > config.options.mso_ceiling_ratio:
        raise ValueError("options.mso_floor_ratio must be <= options.mso_ceiling_ratio.")
    _validate_number(
        config.options.visualization_dpi,
        "options.visualization_dpi",
        minimum=1.0,
    )
    if config.options.max_workers is not None:
        _validate_number(config.options.max_workers, "options.max_workers", minimum=1.0)
    for key, value in (
        ("options.adm_optimization", config.options.adm_optimization),
        ("options.generate_visualizations", config.options.generate_visualizations),
        ("options.generate_3d_visualization", config.options.generate_3d_visualization),
        ("options.no_parallel", config.options.no_parallel),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false.")

    if workflow in {"estimation", "grid"}:
        _validate_number(
            config.calibration.measured_rmt_mso,
            "experiment.calibration.measured_rmt_mso",
            minimum=0.0,
            maximum=100.0,
            minimum_inclusive=False,
        )
    if config.target.didt is not None:
        _validate_number(
            config.target.didt,
            "experiment.target.didt",
            minimum=0.0,
            minimum_inclusive=False,
        )
    if config.target.mso is not None:
        _validate_number(
            config.target.mso,
            "experiment.target.mso",
            minimum=0.0,
            maximum=100.0,
            minimum_inclusive=False,
        )

    if workflow == "grid":
        _validate_number(
            config.grid.search_radius_mm,
            "experiment.target.grid.search_radius_mm",
            minimum=0.0,
        )
        _validate_number(
            config.grid.step_size_mm,
            "experiment.target.grid.step_size_mm",
            minimum=0.0,
            minimum_inclusive=False,
        )
        _validate_number(
            config.grid.cortex_depth_mm,
            "experiment.target.grid.cortex_depth_mm",
            minimum=0.0,
        )

    _require_file(config.subject.t1w_path, "subject.files.t1w")
    if not config.subject.m2m_path.is_dir():
        raise FileNotFoundError(
            f"Head-model directory does not exist (subject.m2m_path): {config.subject.m2m_path}"
        )
    _require_file(config.subject.mesh_path, "subject.m2m_path/*.msh")
    _require_file(config.coil.coil_path, "coil.coil_path")

    if workflow in {"estimation", "grid"}:
        _require_file(config.calibration.bundle_path, "experiment.calibration.bundle_path")
        _require_file(config.target.bundle_path, "experiment.target.bundle_path")
    elif config.target.bundle_path is not None:
        _require_file(config.target.bundle_path, "experiment.target.bundle_path")

    if workflow in {"estimation", "grid", "simulation"} and config.subject.surface_path:
        _require_file(config.subject.surface_path, "subject.files.surface")


def _detect_simnibs_coil_path() -> Optional[Path]:
    """Auto-detect SimNIBS coil models directory."""
    return simnibs_env.find_coil_models_dir()


class FlowStyleDumper(yaml.SafeDumper):
    """
    Custom YAML dumper that uses flow style for lists.

    Makes coordinates and matrices display horizontally for better readability:
        coords: [-13.28, -26.71, 63.0]
    """

    pass


def _represent_list(dumper: FlowStyleDumper, data: list) -> yaml.Node:
    """Represent lists in flow style (horizontal)."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _represent_none(dumper: FlowStyleDumper, data: None) -> yaml.Node:
    """Represent None as empty string for cleaner output."""
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


# Register custom representers
FlowStyleDumper.add_representer(list, _represent_list)
FlowStyleDumper.add_representer(type(None), _represent_none)


def _path_to_str(p: Optional[Path]) -> Optional[str]:
    """Convert Path to string, or return None."""
    return str(p) if p else None


def _subject_output(config: SimNIBSConfig) -> Dict[str, Any]:
    return {
        "id": config.subject.id,
        "derivatives_path": _path_to_str(config.subject.derivatives_path),
        "cache_dir": (
            "no" if config.subject.cache_disabled else _path_to_str(config.subject.cache_dir)
        ),
        "cache_max_size_gb": config.subject.cache_max_size_gb,
        "m2m_path": _path_to_str(config.subject.m2m_path),
        "files": {
            "t1w": _path_to_str(config.subject.t1w_path),
            "weights_cst": _path_to_str(config.subject.weights_cst_path),
            "weights_target": _path_to_str(config.subject.weights_target_path),
            "surface": _path_to_str(config.subject.surface_path),
        },
    }


def _coil_output(config: SimNIBSConfig) -> Dict[str, Any]:
    return {
        "coil_model": config.coil.coil_model,
        "coil_path": _path_to_str(config.coil.coil_path.parent) if config.coil.coil_path else None,
        "coil_distance_mm": config.coil.coil_distance_mm,
        "device_didt_max": config.coil.device_didt_max,
    }


def _options_output(config: SimNIBSConfig) -> Dict[str, Any]:
    options = {
        "roi_size_mm": config.options.roi_size_mm,
        "activation_length_mm": config.options.activation_length_mm,
        "field_mode": config.options.field_mode,
        "adm_optimization": config.options.adm_optimization,
        "opt_search_radius": config.options.opt_search_radius,
        "opt_search_angle": config.options.opt_search_angle,
        "opt_angle_resolution": config.options.opt_angle_resolution,
        "opt_spatial_resolution": config.options.opt_spatial_resolution,
        "generate_visualizations": config.options.generate_visualizations,
        "generate_3d_visualization": config.options.generate_3d_visualization,
        "visualization_dpi": config.options.visualization_dpi,
        "max_angular_deviation_deg": config.options.max_angular_deviation_deg,
        "gwi_threshold_mm": config.options.gwi_threshold_mm,
        "mso_floor_ratio": config.options.mso_floor_ratio,
        "mso_ceiling_ratio": config.options.mso_ceiling_ratio,
        "max_workers": config.options.max_workers,
        "no_parallel": config.options.no_parallel,
    }
    if config.options.stmpx_dataset_name is not None:
        options["stmpx_dataset_name"] = config.options.stmpx_dataset_name
    return options


def _grid_output(config: SimNIBSConfig) -> Dict[str, float]:
    return {
        "search_radius_mm": config.grid.search_radius_mm,
        "step_size_mm": config.grid.step_size_mm,
        "cortex_depth_mm": config.grid.cortex_depth_mm,
    }


def _write_config_output(
    config_dict: Dict[str, Any],
    config_path: Path,
    success_message: str,
    failure_message: str,
) -> None:
    try:
        with open(config_path, "w") as f:
            yaml.dump(
                config_dict,
                f,
                Dumper=FlowStyleDumper,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=1000,
            )
        log.info(f"{success_message}: {config_path}")
    except Exception as e:
        log.warning(f"{failure_message}: {e}")
        raise


def save_config_to_output(
    config: SimNIBSConfig,
    output_dir: Path,
    workflow: str,
    generated_calibration_matrix: Optional[List[List[float]]] = None,
    generated_target_matrix: Optional[List[List[float]]] = None,
    generated_calibration_scalp_coords: Optional[List[float]] = None,
    generated_target_scalp_coords: Optional[List[float]] = None,
    medoid_resolved: bool = False,
) -> Path:
    """
    Save a copy of the configuration to the output directory.

    Creates a YAML file in the output directory containing all configuration
    parameters used for the workflow run. This provides reproducibility and
    documentation of the exact parameters used.

    When an orientation matrix is generated during optimization (from a 3D vector
    input), the output config will:
    - Store the generated 4x4 matrix as the 'orientation' value
    - Preserve the original user input in '_original_orientation_input' field

    For the 'optimization' workflow, generated matrices are NOT included since
    the purpose of that workflow is to generate them.

    Args:
        config: The SimNIBSConfig object to save.
        output_dir: The output directory for the workflow.
        workflow: Name of the workflow (e.g., 'simulation', 'optimization',
                  'estimation', 'grid_search').
        generated_calibration_matrix: Optional 4x4 matrix generated during
                                      calibration/M1 optimization.
        generated_target_matrix: Optional 4x4 matrix generated during target
                                 optimization.

    Returns:
        Path to the saved configuration file.
    """
    from datetime import datetime

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"config_{workflow}_{timestamp}.yml"
    config_path = output_dir / config_filename

    # Helper to build orientation field with optional generated matrix
    def _build_orientation_field(
        original_orientation: Optional[Union[str, List[float], List[List[float]]]],
        generated_matrix: Optional[List[List[float]]],
    ) -> Dict[str, Any]:
        """
        Build orientation field dict.

        If a matrix was generated, returns both the matrix (as 'orientation')
        and the original input (as '_original_orientation_input').
        Otherwise, returns just the original orientation.
        """
        result = {}

        if generated_matrix is not None:
            # Use generated matrix as the orientation value
            result["orientation"] = generated_matrix
            # Preserve original input for reference
            if original_orientation is not None:
                result["_original_orientation_input"] = original_orientation
        else:
            # No matrix generated, use original orientation
            result["orientation"] = original_orientation

        return result

    # Build calibration orientation
    cal_orientation_data = _build_orientation_field(
        config.calibration.orientation,
        generated_calibration_matrix,
    )

    # Build target orientation
    tgt_orientation_data = _build_orientation_field(
        config.target.orientation,
        generated_target_matrix,
    )

    # Resolve scalp coordinates: prefer optimization outputs so the saved
    # config is fully re-runnable without re-triggering coil optimization.
    cal_scalp_coords_final = (
        generated_calibration_scalp_coords
        if generated_calibration_scalp_coords is not None
        else config.calibration.scalp_coords
    )
    tgt_scalp_coords_final = (
        generated_target_scalp_coords
        if generated_target_scalp_coords is not None
        else config.target.scalp_coords
    )

    # If medoid was applied upstream, `config.target.coords` already holds the
    # resolved coordinates. Flip the flag off so replay uses them directly
    # rather than recomputing the medoid.
    cortical_medoid_final = False if medoid_resolved else config.target.medoid_endpoint

    # Build configuration dictionary
    config_dict = {
        "subject": _subject_output(config),
        "coil": _coil_output(config),
        "options": _options_output(config),
        "experiment": {
            "calibration": {
                "label": config.calibration.label,
                "bundle_path": _path_to_str(config.calibration.bundle_path),
                **cal_orientation_data,
                "coords": config.calibration.coords,
                "scalp_coords": cal_scalp_coords_final,
                "measured_rmt_mso": config.calibration.measured_rmt_mso,
            },
            "target": {
                "label": config.target.label,
                "bundle_path": _path_to_str(config.target.bundle_path),
                **tgt_orientation_data,
                "coords": config.target.coords,
                "scalp_coords": tgt_scalp_coords_final,
                "cortical_medoid": cortical_medoid_final,
                "didt": config.target.didt,
                "mso": config.target.mso,
                "grid": _grid_output(config),
            },
        },
        "_metadata": {
            "workflow": workflow,
            "generated_at": datetime.now().isoformat(),
            "output_directory": str(output_dir),
        },
    }

    _write_config_output(
        config_dict,
        config_path,
        "Configuration saved to",
        "Failed to save configuration",
    )

    return config_path


def save_grid_point_config(
    config: SimNIBSConfig,
    output_dir: Path,
    point_label: str,
    cortex_coords: List[float],
    scalp_coords: List[float],
    orientation_matrix: List[List[float]],
    fixed_scalp_coords: List[float],
    grid_orientation_ref: List[float],
    calibration_orientation: Optional[Union[str, List[float], List[List[float]]]] = None,
) -> Path:
    """
    Save a fully self-contained configuration for a specific grid point.

    The emitted YAML mirrors the structure produced by
    :func:`save_config_to_output` so it can be fed directly to
    ``main.py --workflow estimation`` to reproduce this grid point's
    simulation without any manual editing. The ``target`` section is
    pre-populated with the optimized 4x4 coil matrix (skipping target
    optimization on re-run), and the ``calibration`` section carries the
    finalized M1 orientation from the originating grid run (4x4 matrix when
    available) so that M1 optimization is also skipped.

    Args:
        config: The base SimNIBSConfig object used for the grid run.
        output_dir: The grid point output directory.
        point_label: Label for this grid point (e.g., ``"grid_P00"``).
        cortex_coords: Target cortex coordinates for this point.
        scalp_coords: Optimized scalp coordinates for this point.
        orientation_matrix: 4x4 coil orientation matrix from optimization.
        fixed_scalp_coords: Fixed scalp center used across all grid points.
        grid_orientation_ref: Orientation reference (``pos_ydir``) used for
            optimization.
        calibration_orientation: Final calibration orientation from the grid
            run (4x4 matrix when M1 optimization was performed, otherwise the
            original user input). If ``None``, falls back to
            ``config.calibration.orientation``.

    Returns:
        Path to the saved configuration file.
    """
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"config_{point_label}_{timestamp}.yml"
    config_path = output_dir / config_filename

    target_orientation = [list(row) for row in orientation_matrix]

    if calibration_orientation is not None:
        if (
            isinstance(calibration_orientation, list)
            and len(calibration_orientation) == 4
            and calibration_orientation
            and isinstance(calibration_orientation[0], list)
        ):
            cal_orientation_final: Union[str, List[float], List[List[float]]] = [
                list(row) for row in calibration_orientation
            ]
        else:
            cal_orientation_final = calibration_orientation
    else:
        cal_orientation_final = config.calibration.orientation

    config_dict = {
        "subject": _subject_output(config),
        "coil": _coil_output(config),
        "options": _options_output(config),
        "experiment": {
            "calibration": {
                "label": config.calibration.label,
                "bundle_path": _path_to_str(config.calibration.bundle_path),
                "orientation": cal_orientation_final,
                "coords": config.calibration.coords,
                "scalp_coords": config.calibration.scalp_coords,
                "measured_rmt_mso": config.calibration.measured_rmt_mso,
            },
            "target": {
                "label": config.target.label,
                "bundle_path": _path_to_str(config.target.bundle_path),
                "orientation": target_orientation,
                "coords": list(cortex_coords),
                "scalp_coords": list(scalp_coords),
                # Medoid must be disabled so the grid-point coords are used
                # as-is when the config is replayed via `--workflow estimation`.
                "cortical_medoid": False,
                "didt": config.target.didt,
                "mso": config.target.mso,
                "grid": _grid_output(config),
            },
        },
        "grid_point": {
            "label": point_label,
            "cortex_coords": list(cortex_coords),
            "optimized_scalp_coords": list(scalp_coords),
            "orientation_matrix": target_orientation,
            "fixed_scalp_center": list(fixed_scalp_coords),
            "orientation_reference": (
                list(grid_orientation_ref)
                if hasattr(grid_orientation_ref, "__iter__")
                and not isinstance(grid_orientation_ref, str)
                else grid_orientation_ref
            ),
        },
        "_metadata": {
            "workflow": "grid_search",
            "source": "grid_point_reproducibility",
            "generated_at": datetime.now().isoformat(),
            "output_directory": str(output_dir),
            "reproduce_with": (
                "python main.py --no-gui --config " f"{config_filename} --workflow estimation"
            ),
        },
    }

    _write_config_output(
        config_dict,
        config_path,
        "Grid point configuration saved to",
        "Failed to save grid point configuration",
    )

    return config_path
