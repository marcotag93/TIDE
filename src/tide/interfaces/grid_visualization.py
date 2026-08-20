"""
Grid Search Visualization (in-package)
======================================

Library port of ``scripts/visualize_grid.py``. Produces:

    1. Scalar NIfTI  — intensity-valued voxels (% MSO) overlayable in FSLeyes / MRIcroGL,
       written as a raw map, a clamped map, and a categorical clamp-regime map
    2. Interactive 3D — Self-contained HTML with tractography + clickable points

The raw map carries the unbounded output of the calibration inversion and is the
quantity to use for any quantitative or cross-subject comparison. The clamped map
carries the programmable intensity and saturates at the floor and ceiling bounds,
so uniform patches in it may reflect the clamp rather than uniform anatomy; the
clamp-regime map identifies exactly which points saturated.

Public entry point: :func:`run_grid_visualization`. Called in-process from
``workflows.grid_search`` Step 8.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Voxel codes of the clamp-regime map. 0 marks background and any unlabelled point.
CLAMP_FLAG_CODES: Dict[str, int] = {
    "WITHIN_RANGE": 1,
    "CLAMPED_LOW": 2,
    "CLAMPED_HIGH": 3,
    "DEVICE_LIMITED": 4,
}

# Colour-scale modes of the interactive viewer, mapped to their record field.
VIEW_MODES: Dict[str, str] = {
    "raw": "weighted_mso_raw",
    "clamped": "weighted_mso_clamped",
}

VIEW_MODE_TITLES: Dict[str, str] = {
    "raw": "Weighted I, raw (%)",
    "clamped": "Weighted I, clamped (%)",
}

DEFAULT_VIEW_MODE = "raw"


# =============================================================================
# CSV PARSING
# =============================================================================


def parse_grid_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse the TIDE grid results CSV into a list of structured records."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    records: List[Dict[str, Any]] = []

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)

        expected_cols = {"grid_point_labels", "grid_point_coords"}
        if not expected_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Unexpected CSV header. Expected at least {expected_cols}, "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            try:
                record = {
                    "label": row["grid_point_labels"],
                    "coords": ast.literal_eval(row["grid_point_coords"]),
                    "scalp_coords": _safe_literal_eval(row.get("fixed_scalp_start_coords", "")),
                    "opt_scalp_coords": _safe_literal_eval(
                        row.get("optimized_scalp_point_coords", "")
                    ),
                    "matrix4x4": _parse_matrix(row.get("matrix4x4", "")),
                    "measured_m1_mso": float(row.get("measured_m1_mso", 0)),
                    "unweighted_mso_raw": float(row.get("unweighted_mso_raw", 0)),
                    "weighted_mso_raw": float(row.get("weighted_mso_raw", 0)),
                    "unweighted_mso_clamped": float(row.get("unweighted_mso_clamped", 0)),
                    "weighted_mso_clamped": float(row.get("weighted_mso_clamped", 0)),
                    "unweighted_mso_flag": row.get("unweighted_mso_flag", "N/A"),
                    "weighted_mso_flag": row.get("weighted_mso_flag", "N/A"),
                    "sei_weighted": float(row.get("sei_weighted", 0)),
                    "sei_unweighted": float(row.get("sei_unweighted", 0)),
                    "sei_rank_pct": float(row.get("sei_rank_pct", 0)),
                }
                intensity_values = (
                    record["unweighted_mso_raw"],
                    record["weighted_mso_raw"],
                    record["unweighted_mso_clamped"],
                    record["weighted_mso_clamped"],
                )
                if (
                    "ESTIMATION_FAILED"
                    in (record["unweighted_mso_flag"], record["weighted_mso_flag"])
                    or not np.isfinite(intensity_values).all()
                ):
                    log.warning("Skipping failed grid point '%s'", record["label"])
                    continue
                records.append(record)
            except (ValueError, SyntaxError) as exc:
                log.warning(
                    "Skipping malformed row '%s': %s",
                    row.get("grid_point_labels", "?"),
                    exc,
                )

    log.info("Parsed %d grid point records from %s", len(records), csv_path)
    return records


def _safe_literal_eval(value: str) -> Optional[list]:
    if not value or value.strip() in ("", "None"):
        return None
    try:
        return ast.literal_eval(value.strip())
    except (ValueError, SyntaxError):
        return None


def _parse_matrix(value: str) -> Optional[np.ndarray]:
    if not value or value.strip() in ("", "None"):
        return None
    try:
        parsed = ast.literal_eval(value.strip())
        arr = np.array(parsed, dtype=float)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
        log.warning("Matrix has unexpected shape %s", arr.shape)
        return None
    except (ValueError, SyntaxError):
        return None


# =============================================================================
# SCALAR NIFTI MAP
# =============================================================================


def _scatter_to_volume(
    points: np.ndarray,
    values: np.ndarray,
    shape: tuple,
    inv_affine: np.ndarray,
) -> np.ndarray:
    """Paint per-point scalar values into a volume of the reference shape."""
    data = np.zeros(shape, dtype=np.float32)

    voxel_coords = (points @ inv_affine[:3, :3].T) + inv_affine[:3, 3]
    voxel_indices = np.rint(voxel_coords).astype(int)

    valid = (
        (voxel_indices[:, 0] >= 0)
        & (voxel_indices[:, 0] < data.shape[0])
        & (voxel_indices[:, 1] >= 0)
        & (voxel_indices[:, 1] < data.shape[1])
        & (voxel_indices[:, 2] >= 0)
        & (voxel_indices[:, 2] < data.shape[2])
    )

    vi = voxel_indices[valid]
    data[vi[:, 0], vi[:, 1], vi[:, 2]] = values[valid]
    return data


def generate_scalar_nifti(
    records: List[Dict[str, Any]],
    t1w_path: Path,
    output_dir: Path,
) -> None:
    """Write the clamped, raw, and clamp-regime NIfTI maps + a TSV label sidecar."""
    import nibabel as nib

    ref_img = nib.load(str(t1w_path))
    affine = ref_img.affine
    inv_affine = np.linalg.inv(affine)
    shape = ref_img.shape[:3]

    points = np.array([r["coords"] for r in records])

    volumes = {
        "grid_mso_map.nii.gz": [r["weighted_mso_clamped"] for r in records],
        "grid_mso_raw_map.nii.gz": [r["weighted_mso_raw"] for r in records],
        "grid_mso_flag_map.nii.gz": [
            CLAMP_FLAG_CODES.get(r["weighted_mso_flag"], 0) for r in records
        ],
    }

    for filename, values in volumes.items():
        data = _scatter_to_volume(points, np.array(values, dtype=np.float32), shape, inv_affine)
        nifti_path = output_dir / filename
        nib.save(nib.Nifti1Image(data, affine, ref_img.header), str(nifti_path))
        log.info("Scalar NIfTI saved: %s", nifti_path)

    tsv_path = output_dir / "grid_mso_labels.tsv"
    with open(tsv_path, "w") as fh:
        fh.write(
            "label\tx_ras\ty_ras\tz_ras\t"
            "weighted_mso_clamped\tweighted_mso_raw\t"
            "unweighted_mso_clamped\tsei_weighted\t"
            "unweighted_mso_raw\tweighted_mso_flag\tunweighted_mso_flag\n"
        )
        for r in records:
            fh.write(
                f"{r['label']}\t"
                f"{r['coords'][0]:.2f}\t{r['coords'][1]:.2f}\t"
                f"{r['coords'][2]:.2f}\t"
                f"{r['weighted_mso_clamped']:.2f}\t"
                f"{r['weighted_mso_raw']:.2f}\t"
                f"{r['unweighted_mso_clamped']:.2f}\t"
                f"{r['sei_weighted']:.4f}\t"
                f"{r['unweighted_mso_raw']:.2f}\t"
                f"{r['weighted_mso_flag']}\t"
                f"{r['unweighted_mso_flag']}\n"
            )
    log.info("Label sidecar saved: %s", tsv_path)


# =============================================================================
# INTERACTIVE 3D HTML
# =============================================================================


def _mso_to_hex(value: float, vmin: float, vmax: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.RdYlGn_r
    t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    r, g, b, _ = cmap(t)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _serialize_streamlines_for_html(
    streamlines: List[np.ndarray],
    max_streamlines: int = 1500,
    max_points_per_streamline: int = 80,
    decimals: int = 2,
) -> List[List[List[float]]]:
    total = len(streamlines)
    if total == 0:
        return []

    stride = max(1, int(np.ceil(total / max_streamlines)))
    selected = list(streamlines[::stride])[:max_streamlines]

    serialized: List[List[List[float]]] = []
    for sl in selected:
        arr = np.asarray(sl, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 2:
            continue
        if len(arr) > max_points_per_streamline:
            indices = np.linspace(0, len(arr) - 1, max_points_per_streamline, dtype=int)
            arr = arr[indices]
        serialized.append(np.round(arr, decimals).tolist())
    return serialized


def generate_interactive_html(
    records: List[Dict[str, Any]],
    trk_path: Path,
    output_dir: Path,
    streamline_subsample: int = 3,
) -> None:
    """Generate a self-contained interactive 3D HTML viewer."""
    import nibabel as nib

    log.info("Loading tractogram for 3D viewer: %s", trk_path.name)
    trk_file = nib.streamlines.load(str(trk_path))
    streamlines = trk_file.streamlines

    total_sl = len(streamlines)
    step = max(1, streamline_subsample)
    subsampled = list(streamlines[::step])
    sl_data = _serialize_streamlines_for_html(subsampled)
    log.info(
        "Using %d / %d streamlines in HTML preview (subsample=%d)",
        len(sl_data),
        total_sl,
        step,
    )

    scales = {}
    for mode, field in VIEW_MODES.items():
        values = [r[field] for r in records]
        scales[mode] = {"vmin": min(values), "vmax": max(values)}

    points_js = []
    for r in records:
        colors = {
            f"color_{mode}": _mso_to_hex(r[field], scales[mode]["vmin"], scales[mode]["vmax"])
            for mode, field in VIEW_MODES.items()
        }
        points_js.append(
            {
                "label": r["label"],
                "coords": [round(c, 2) for c in r["coords"]],
                **colors,
                "weighted_mso_clamped": round(r["weighted_mso_clamped"], 2),
                "weighted_mso_raw": round(r["weighted_mso_raw"], 2),
                "unweighted_mso_clamped": round(r["unweighted_mso_clamped"], 2),
                "unweighted_mso_raw": round(r["unweighted_mso_raw"], 2),
                "weighted_mso_flag": r["weighted_mso_flag"],
                "unweighted_mso_flag": r["unweighted_mso_flag"],
                "sei_weighted": round(r["sei_weighted"], 4),
                "sei_unweighted": round(r["sei_unweighted"], 4),
                "sei_rank_pct": round(r["sei_rank_pct"], 1),
                "matrix4x4": (r["matrix4x4"].tolist() if r["matrix4x4"] is not None else None),
            }
        )

    all_coords = np.array([r["coords"] for r in records])
    centroid = all_coords.mean(axis=0).tolist()

    n_stops = 8
    for mode, scale in scales.items():
        vmin, vmax = scale["vmin"], scale["vmax"]
        scale["stops"] = [
            {
                "value": round(vmin + (i / n_stops) * (vmax - vmin), 1),
                "color": _mso_to_hex(vmin + (i / n_stops) * (vmax - vmin), vmin, vmax),
            }
            for i in range(n_stops + 1)
        ]
        scale["vmin"] = round(vmin, 1)
        scale["vmax"] = round(vmax, 1)
        scale["title"] = VIEW_MODE_TITLES[mode]

    html_content = _build_html_template(
        streamlines_json=json.dumps(sl_data),
        points_json=json.dumps(points_js),
        centroid_json=json.dumps(centroid),
        scales_json=json.dumps(scales),
        default_mode=DEFAULT_VIEW_MODE,
        num_points=len(records),
        num_streamlines=len(sl_data),
    )

    out_path = output_dir / "grid_interactive.html"
    out_path.write_text(html_content, encoding="utf-8")
    log.info("Interactive HTML saved: %s", out_path)


def _build_html_template(
    *,
    streamlines_json: str,
    points_json: str,
    centroid_json: str,
    scales_json: str,
    default_mode: str,
    num_points: int,
    num_streamlines: int,
) -> str:
    """Return the complete self-contained HTML string for the 3D viewer."""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TIDE Grid Search - Interactive 3D Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f0f13;
    color: #e0e0e0;
    overflow: hidden;
    height: 100vh;
  }}

  #header {{
    position: fixed; top: 0; left: 0; right: 0;
    height: 48px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    display: flex; align-items: center; padding: 0 20px;
    z-index: 100;
    border-bottom: 1px solid #2a2a4a;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
  }}
  #header h1 {{
    font-size: 15px; font-weight: 600; color: #a8b4ff;
    letter-spacing: 0.5px;
  }}
  #header .stats {{
    margin-left: auto; font-size: 12px; color: #8888aa;
  }}

  #viewer {{ position: fixed; top: 48px; left: 0; right: 0; bottom: 0; }}

  #colorbar {{
    position: fixed; bottom: 20px; left: 20px;
    background: rgba(20,20,35,0.92);
    border: 1px solid #333355;
    border-radius: 8px;
    padding: 12px 16px;
    z-index: 50;
    backdrop-filter: blur(8px);
  }}
  #colorbar .title {{
    font-size: 11px; color: #aaa; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 1px;
  }}
  #colorbar .bar {{
    width: 200px; height: 14px;
    border-radius: 3px;
    margin-bottom: 4px;
  }}
  #colorbar .labels {{
    display: flex; justify-content: space-between;
    font-size: 10px; color: #999;
  }}
  #colorbar .modes {{
    display: flex; gap: 6px; margin-top: 10px;
  }}
  #colorbar .modes button {{
    flex: 1;
    font: inherit; font-size: 10px; letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 4px 0;
    color: #99a; background: #1c1c30;
    border: 1px solid #333355; border-radius: 4px;
    cursor: pointer;
  }}
  #colorbar .modes button.active {{
    color: #0f0f13; background: #6fc3df; border-color: #6fc3df;
  }}

  #panel {{
    position: fixed; top: 48px; left: 0; bottom: 0;
    width: 340px;
    background: rgba(18,18,30,0.96);
    border-right: 1px solid #2a2a4a;
    transform: translateX(-100%);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 80;
    overflow-y: auto;
    backdrop-filter: blur(12px);
  }}
  #panel.open {{ transform: translateX(0); }}
  #panel .close-btn {{
    position: absolute; top: 10px; right: 14px;
    background: none; border: none; color: #888;
    font-size: 20px; cursor: pointer;
  }}
  #panel .close-btn:hover {{ color: #fff; }}

  #panel .content {{ padding: 20px; }}
  #panel .point-label {{
    font-size: 20px; font-weight: 700;
    color: #a8b4ff; margin-bottom: 4px;
  }}
  #panel .section-title {{
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 1.2px; color: #666;
    margin-top: 18px; margin-bottom: 8px;
    border-bottom: 1px solid #2a2a4a;
    padding-bottom: 4px;
  }}

  .kv {{ display: flex; justify-content: space-between;
         padding: 3px 0; font-size: 13px; align-items: center; }}
  .kv .k {{ color: #999; }}
  .kv .v {{ color: #e0e0e0; font-family: 'JetBrains Mono', monospace;
            font-weight: 500; }}

  .flag {{
    display: inline-block; padding: 1px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600;
  }}
  .flag-within {{ background: #1a3a2a; color: #4caf50; }}
  .flag-clamped {{ background: #3a2a1a; color: #ff9800; }}

  .copyable {{
    display: flex; align-items: center; gap: 6px;
    margin-top: 4px;
  }}
  .copyable code {{
    flex: 1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: #c8d0e0;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    padding: 6px 10px;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.5;
  }}
  .copy-btn {{
    flex-shrink: 0;
    width: 28px; height: 28px;
    background: rgba(255,255,255,0.06);
    border: 1px solid #3a3a5a;
    border-radius: 4px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: #888;
    transition: all 0.15s ease;
  }}
  .copy-btn:hover {{ background: rgba(255,255,255,0.12); color: #ccc; }}
  .copy-btn.copied {{ background: #1a3a2a; border-color: #4caf50; color: #4caf50; }}

  #tooltip {{
    position: fixed;
    padding: 4px 10px;
    background: rgba(20,20,40,0.9);
    border: 1px solid #4a4a6a;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    z-index: 200;
    display: none;
    color: #ddd;
  }}

  #instructions {{
    position: fixed; bottom: 20px; right: 20px;
    font-size: 11px; color: #555;
    text-align: right; z-index: 50;
    line-height: 1.6;
  }}
</style>
</head>
<body>

<div id="header">
  <h1>TIDE Grid Search - 3D Viewer</h1>
  <span class="stats">{num_points} grid points &nbsp;·&nbsp; {num_streamlines} streamlines</span>
</div>

<div id="viewer"></div>

<div id="colorbar">
  <div class="title" id="colorbar-title"></div>
  <div class="bar" id="colorbar-bar"></div>
  <div class="labels">
    <span id="colorbar-min"></span><span id="colorbar-max"></span>
  </div>
  <div class="modes">
    <button id="mode-raw" onclick="setViewMode('raw')">Raw</button>
    <button id="mode-clamped" onclick="setViewMode('clamped')">Clamped</button>
  </div>
</div>

<div id="panel">
  <button class="close-btn" onclick="closePanel()">&times;</button>
  <div class="content" id="panel-content"></div>
</div>

<div id="tooltip"></div>

<div id="instructions">
  <b>Click</b> a sphere to inspect &nbsp;|&nbsp;
  <b>Drag</b> to rotate &nbsp;|&nbsp;
  <b>Scroll</b> to zoom
</div>

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
const gridPoints  = {points_json};
const centroid    = {centroid_json};
const scales      = {scales_json};

let viewMode = '{default_mode}';

const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x0f0f13);

const viewer = document.getElementById('viewer');
const camera = new THREE.PerspectiveCamera(
  50, viewer.clientWidth / viewer.clientHeight, 0.1, 5000
);
camera.up.set(0, 0, 1);
camera.position.set(
  centroid[0] + 80, centroid[1] - 40, centroid[2] + 60
);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(centroid[0], centroid[1], centroid[2]);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(50, 80, 60);
scene.add(dirLight);

const slMaterial = new THREE.LineBasicMaterial({{
  color: 0x8ea4bf, opacity: 0.38, transparent: true
}});

for (const sl of streamlines) {{
  const geom = new THREE.BufferGeometry();
  const verts = new Float32Array(sl.flat());
  geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  scene.add(new THREE.Line(geom, slMaterial));
}}

const sphereGeo = new THREE.SphereGeometry(0.8, 16, 16);
const sphereGroup = new THREE.Group();
const raycaster = new THREE.Raycaster();
const pointer   = new THREE.Vector2();
const sphereMap = new Map();
const sphereMeshes = [];

for (const pt of gridPoints) {{
  const color = new THREE.Color(pt['color_' + viewMode]);
  const mat = new THREE.MeshPhysicalMaterial({{
    color: color,
    roughness: 0.25,
    metalness: 0.15,
    emissive: color.clone(),
    emissiveIntensity: 0.12,
    clearcoat: 0.3,
    clearcoatRoughness: 0.2,
  }});
  const mesh = new THREE.Mesh(sphereGeo, mat);
  mesh.position.set(pt.coords[0], pt.coords[1], pt.coords[2]);
  sphereGroup.add(mesh);
  sphereMap.set(mesh.uuid, pt);
  sphereMeshes.push(mesh);
}}
scene.add(sphereGroup);

const barEl   = document.getElementById('colorbar-bar');
const titleEl = document.getElementById('colorbar-title');
const minEl   = document.getElementById('colorbar-min');
const maxEl   = document.getElementById('colorbar-max');

window.setViewMode = function(mode) {{
  viewMode = mode;
  const scale = scales[mode];

  sphereMeshes.forEach((mesh, i) => {{
    const color = new THREE.Color(gridPoints[i]['color_' + mode]);
    mesh.material.color.copy(color);
    mesh.material.emissive.copy(color);
  }});

  const stops = scale.stops.map(
    (s, i) => s.color + ' ' + (i / (scale.stops.length - 1) * 100) + '%'
  );
  barEl.style.background = 'linear-gradient(to right, ' + stops.join(', ') + ')';
  titleEl.textContent = scale.title;
  minEl.textContent = scale.vmin;
  maxEl.textContent = scale.vmax;

  for (const key of Object.keys(scales)) {{
    document.getElementById('mode-' + key).classList.toggle('active', key === mode);
  }}
}};

setViewMode(viewMode);

let selectedMesh = null;

function flagClass(flag) {{
  return flag === 'WITHIN_RANGE' ? 'flag-within' : 'flag-clamped';
}}

function fmtCoords(c) {{
  return '[' + c.map(v => v.toFixed(2)).join(', ') + ']';
}}

function fmtMatrix(mat) {{
  if (!mat) return 'N/A';
  const rows = mat.map(r => '[' + r.map(v => v.toFixed(8)).join(', ') + ']');
  return '[' + rows.join(', ') + ']';
}}

const COPY_SVG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
const CHECK_SVG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;

window.copyField = function(btn, text) {{
  navigator.clipboard.writeText(text).then(() => {{
    btn.innerHTML = CHECK_SVG;
    btn.classList.add('copied');
    setTimeout(() => {{ btn.innerHTML = COPY_SVG; btn.classList.remove('copied'); }}, 1500);
  }});
}};

function copyableField(value) {{
  const escaped = value.replace(/'/g, "\\'");
  return `<div class="copyable"><code>${{value}}</code><button class="copy-btn" onclick="copyField(this, '${{escaped}}')">${{COPY_SVG}}</button></div>`;
}}

function openPanel(pt) {{
  const coordsStr = fmtCoords(pt.coords);
  const matrixStr = fmtMatrix(pt.matrix4x4);
  const el = document.getElementById('panel-content');
  el.innerHTML = `
    <div class="point-label">${{pt.label}}</div>

    <div class="section-title">Coordinates (RAS mm)</div>
    ${{copyableField(coordsStr)}}

    <div class="section-title">Intensity Estimates (%)</div>
    <div class="kv"><span class="k">Weighted (raw)</span><span class="v">${{pt.weighted_mso_raw}}</span></div>
    <div class="kv"><span class="k">Unweighted (raw)</span><span class="v">${{pt.unweighted_mso_raw}}</span></div>
    <div class="kv"><span class="k">Weighted (clamped)</span><span class="v">${{pt.weighted_mso_clamped}}</span></div>
    <div class="kv"><span class="k">Unweighted (clamped)</span><span class="v">${{pt.unweighted_mso_clamped}}</span></div>

    <div class="section-title">Flags</div>
    <div class="kv"><span class="k">Weighted</span><span class="flag ${{flagClass(pt.weighted_mso_flag)}}">${{pt.weighted_mso_flag}}</span></div>
    <div class="kv"><span class="k">Unweighted</span><span class="flag ${{flagClass(pt.unweighted_mso_flag)}}">${{pt.unweighted_mso_flag}}</span></div>

    <div class="section-title">Stimulation Efficiency</div>
    <div class="kv"><span class="k">SEI (weighted)</span><span class="v">${{pt.sei_weighted}}</span></div>
    <div class="kv"><span class="k">SEI (unweighted)</span><span class="v">${{pt.sei_unweighted}}</span></div>
    <div class="kv"><span class="k">SEI rank</span><span class="v">${{pt.sei_rank_pct}}%</span></div>

    <div class="section-title">Optimised Coil Matrix (4×4)</div>
    ${{copyableField(matrixStr)}}
  `;
  document.getElementById('panel').classList.add('open');
}}

window.closePanel = function() {{
  document.getElementById('panel').classList.remove('open');
  if (selectedMesh) {{
    selectedMesh.material.emissiveIntensity = 0.15;
    selectedMesh.scale.set(1, 1, 1);
    selectedMesh = null;
  }}
}};

renderer.domElement.addEventListener('click', (e) => {{
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  pointer.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(sphereGroup.children);
  if (hits.length > 0) {{
    if (selectedMesh) {{
      selectedMesh.material.emissiveIntensity = 0.15;
      selectedMesh.scale.set(1, 1, 1);
    }}
    selectedMesh = hits[0].object;
    selectedMesh.material.emissiveIntensity = 0.5;
    selectedMesh.scale.set(1.3, 1.3, 1.3);
    openPanel(sphereMap.get(selectedMesh.uuid));
  }}
}});

const tooltip = document.getElementById('tooltip');
renderer.domElement.addEventListener('mousemove', (e) => {{
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  pointer.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(sphereGroup.children);
  if (hits.length > 0) {{
    const pt = sphereMap.get(hits[0].object.uuid);
    const shown = viewMode === 'raw' ? pt.weighted_mso_raw : pt.weighted_mso_clamped;
    const suffix = pt.weighted_mso_flag === 'WITHIN_RANGE'
      ? '' : ' (' + pt.weighted_mso_flag + ')';
    tooltip.textContent = pt.label + ' - I: ' + shown + '%' + suffix;
    tooltip.style.left = e.clientX + 14 + 'px';
    tooltip.style.top  = e.clientY + 14 + 'px';
    tooltip.style.display = 'block';
    renderer.domElement.style.cursor = 'pointer';
  }} else {{
    tooltip.style.display = 'none';
    renderer.domElement.style.cursor = 'grab';
  }}
}});

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
# PUBLIC ENTRY POINT
# =============================================================================


def run_grid_visualization(
    csv_path: Path,
    t1w_path: Path,
    trk_path: Path,
    output_dir: Path,
    streamline_subsample: int = 3,
    generate_interactive: bool = True,
) -> None:
    """Run all grid-search visualizations in-process.

    Equivalent to invoking ``scripts/visualize_grid.py`` but without spawning
    a subprocess. Produces ``grid_mso_map.nii.gz``, ``grid_mso_raw_map.nii.gz``,
    ``grid_mso_flag_map.nii.gz``, ``grid_mso_labels.tsv``, and, when requested,
    ``grid_interactive.html`` in *output_dir*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_grid_csv(Path(csv_path))
    if not records:
        log.error("No valid grid points found in CSV: %s", csv_path)
        return

    generate_scalar_nifti(records, Path(t1w_path), output_dir)
    if generate_interactive:
        generate_interactive_html(
            records,
            Path(trk_path),
            output_dir,
            streamline_subsample=streamline_subsample,
        )

    log.info("Grid visualizations written to: %s", output_dir)
