"""Softaxic STMPX export for completed TIDE estimations."""

import ast
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import numpy as np
from defusedxml import ElementTree as DefusedET
from defusedxml.common import DefusedXmlException

COIL_CODE = "8700449"

FP_ATTR_ORDER = [
    "m00",
    "m10",
    "y",
    "m21",
    "m02",
    "m22",
    "m01",
    "m12",
    "ts",
    "z",
    "id",
    "x",
    "m20",
    "m11",
]


def _parse_matrix_string(matrix_str: str) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(matrix_str))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Failed to parse Target Estimation matrix: {exc}") from exc


def _parse_coords_string(coords_str: str) -> list[float]:
    try:
        return [float(coordinate) for coordinate in ast.literal_eval(coords_str)]
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Failed to parse Target Estimation coordinates: {exc}") from exc


def _extract_target_data(results_path: Path) -> dict[str, Any]:
    content = results_path.read_text()
    target_match = re.search(
        r"--- Target Estimation \((.*?)\) ---(.*?)--- Geometric Analysis",
        content,
        re.DOTALL,
    )
    if not target_match:
        raise ValueError("Could not find Target Estimation section in TIDE results.")

    target_text = target_match.group(2)
    data: dict[str, Any] = {}

    cortex_match = re.search(r"Target Coords \(Cortex\):\s*(\[.*?\])", target_text)
    if cortex_match:
        data["cortex_coords"] = _parse_coords_string(cortex_match.group(1))

    scalp_match = re.search(r"Optimized Scalp Position:\s*(\[.*?\])", target_text)
    if scalp_match:
        data["scalp_coords"] = _parse_coords_string(scalp_match.group(1))

    matrix_match = re.search(r"Optimized Matrix:\s*(\[\[.*?\]\])", target_text, re.DOTALL)
    if not matrix_match:
        raise ValueError("Could not find Target Estimation matrix in TIDE results.")
    data["matrix"] = _parse_matrix_string(matrix_match.group(1))

    return data


def simnibs_to_softaxic_rotation(matrix: np.ndarray) -> dict[str, float]:
    """Apply the laboratory-verified SimNIBS-to-Softaxic rotation mapping."""
    return {
        "m00": float(matrix[0, 1]),
        "m01": float(matrix[0, 0]),
        "m02": float(-matrix[0, 2]),
        "m10": float(matrix[1, 1]),
        "m11": float(matrix[1, 0]),
        "m12": float(-matrix[1, 2]),
        "m20": float(matrix[2, 1]),
        "m21": float(matrix[2, 0]),
        "m22": float(-matrix[2, 2]),
    }


def _create_fp_element(matrix: np.ndarray, point_id: str) -> ET.Element:
    rotation = simnibs_to_softaxic_rotation(matrix)
    fp = ET.Element("fp")
    attributes = {
        "m00": f"{rotation['m00']:.6f}",
        "m10": f"{rotation['m10']:.6f}",
        "y": f"{float(matrix[1, 3]):.4f}",
        "m21": f"{rotation['m21']:.6f}",
        "m02": f"{rotation['m02']:.6f}",
        "m22": f"{rotation['m22']:.6f}",
        "m01": f"{rotation['m01']:.6f}",
        "m12": f"{rotation['m12']:.6f}",
        "ts": str(int(time.time() * 1000)),
        "z": f"{float(matrix[2, 3]):.4f}",
        "id": point_id,
        "x": f"{float(matrix[0, 3]):.4f}",
        "m20": f"{rotation['m20']:.6f}",
        "m11": f"{rotation['m11']:.6f}",
    }
    for attribute_name in FP_ATTR_ORDER:
        fp.set(attribute_name, attributes[attribute_name])
    return fp


def _create_target_element(data: dict[str, Any]) -> ET.Element:
    target = ET.Element("fmp")
    target.set("global", "0")
    target.set("id", "Target_Estimation")

    fp = _create_fp_element(data["matrix"], point_id=COIL_CODE)
    cortex_coords = data.get("cortex_coords")
    if cortex_coords:
        cortex = ET.SubElement(fp, "b")
        cortex.set("x", f"{cortex_coords[0]:.4f}")
        cortex.set("y", f"{cortex_coords[1]:.4f}")
        cortex.set("z", f"{cortex_coords[2]:.4f}")

    scalp_coords = data.get("scalp_coords")
    if scalp_coords:
        scalp = ET.SubElement(fp, "f")
        scalp.set("x", f"{scalp_coords[0]:.4f}")
        scalp.set("y", f"{scalp_coords[1]:.4f}")
        scalp.set("z", f"{scalp_coords[2]:.4f}")

    target.append(fp)
    return target


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    spacer = "\n" + level * " "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = spacer + " "
        if not element.tail or not element.tail.strip():
            element.tail = spacer
        for child in element:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = spacer


def _load_stmpx(stmpx_path: Path) -> tuple[ET.ElementTree, ET.Element]:
    if not stmpx_path.is_file():
        raise FileNotFoundError(f"STMPX file not found: {stmpx_path}")
    try:
        tree = DefusedET.parse(stmpx_path)
    except (ET.ParseError, DefusedXmlException) as exc:
        raise ValueError(f"Invalid STMPX XML ({stmpx_path}): {exc}") from exc
    fmpm = tree.getroot().find("fmpm")
    if fmpm is None:
        raise ValueError(f"STMPX file has no <fmpm> element: {stmpx_path}")
    return tree, fmpm


def validate_stmpx_input(stmpx_path: Path) -> None:
    """Validate the STMPX template before TIDE creates derivative output."""
    _load_stmpx(stmpx_path)


def export_target_to_stmpx(
    stmpx_path: Path,
    results_path: Path,
    dataset_name: Optional[str] = None,
) -> Path:
    """Append the TIDE Target Estimation pose and write ``*_updated.stmpx``."""
    target_data = _extract_target_data(results_path)
    tree, fmpm = _load_stmpx(stmpx_path)
    if dataset_name is not None:
        fmpm.set("dataset", dataset_name)
    fmpm.append(_create_target_element(target_data))

    output_path = stmpx_path.parent / f"{stmpx_path.stem}_updated{stmpx_path.suffix}"
    _indent_xml(tree.getroot())
    with open(output_path, "wb") as file_handle:
        file_handle.write(b"<!DOCTYPE stmp>\n")
        tree.write(file_handle, encoding="utf-8", xml_declaration=False)
    return output_path
