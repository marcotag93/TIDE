import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tide.core import io
from tide.core.physics import AGGREGATOR_KEYS
from tide.utils.config import (
    CoilConfig,
    GridConfig,
    OptionsConfig,
    SimNIBSConfig,
    SubjectConfig,
    TargetConfig,
    save_config_to_output,
    save_grid_point_config,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_pypi_release_is_gated_by_reusable_ci() -> None:
    root = Path(__file__).parent.parent
    ci = yaml.load(
        (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    release = yaml.load(
        (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )

    assert "workflow_call" in ci["on"]
    assert release["jobs"]["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert release["jobs"]["build"]["needs"] == "ci"


def _fake_aggregates(base: float) -> dict:
    """Per-aggregator stand-in values, distinct per key so ordering is contract-checked."""
    return {key: base - 10.0 * index for index, key in enumerate(AGGREGATOR_KEYS)}


def _normalize_artifact(text: str, root: Path) -> str:
    root_text = str(root)
    normalized = text.replace(root_text.replace("\\", "\\\\"), "<ROOT>")
    normalized = normalized.replace(root_text, "<ROOT>")
    normalized = re.sub(
        r'<ROOT>[^"\r\n]*',
        lambda match: match.group(0).replace("\\\\", "/").replace("\\", "/"),
        normalized,
    )
    normalized = re.sub(r"\d{8}_\d{6}", "<STAMP>", normalized)
    normalized = re.sub(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?",
        "<ISO_TIMESTAMP>",
        normalized,
    )
    normalized = re.sub(r"TIDE \d+\.\d+\.\d+", "TIDE <VERSION>", normalized)
    return normalized


def test_artifact_normalization_handles_windows_paths() -> None:
    root = Path(r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\test_contract")
    text = (
        r"plain: C:\Users\runneradmin\AppData\Local\Temp\pytest-0\test_contract\nested\out.txt"
        "\n"
        r"json: C:\\Users\\runneradmin\\AppData\\Local\\Temp\\pytest-0\\test_contract\\nested\\out.txt"
    )

    assert _normalize_artifact(text, root) == (
        "plain: <ROOT>/nested/out.txt\njson: <ROOT>/nested/out.txt"
    )


def test_report_relative_paths_use_url_separators() -> None:
    class ResolvedPath:
        def __init__(self, value: PureWindowsPath) -> None:
            self.value = value

        def resolve(self) -> PureWindowsPath:
            return self.value

    root = PureWindowsPath(r"C:\Users\runneradmin\report")
    image = root / "visualizations" / "target_composite.png"

    assert io._reporting._relative_path(ResolvedPath(image), ResolvedPath(root)) == (
        "visualizations/target_composite.png"
    )


def _make_config(root: Path) -> SimNIBSConfig:
    return SimNIBSConfig(
        subject=SubjectConfig(
            id="sub-CONTRACT",
            derivatives_path=root / "derivatives",
            m2m_path=root / "derivatives" / "m2m_sub-CONTRACT",
            t1w_path=root / "derivatives" / "t1w.nii.gz",
            weights_cst_path=root / "derivatives" / "cst_weights.txt",
            weights_target_path=root / "derivatives" / "target_weights.txt",
            surface_path=root / "derivatives" / "surface.gii",
            cache_dir=root / "cache",
            cache_max_size_gb=4.5,
        ),
        coil=CoilConfig(
            coil_model="contract.ccd",
            coil_path=root / "coils" / "contract.ccd",
            coil_distance_mm=4.0,
            device_didt_max=161e6,
        ),
        calibration=TargetConfig(
            label="M1",
            bundle_path=root / "derivatives" / "cst.trk",
            coords=[-1.0, 2.0, 3.0],
            scalp_coords=[-2.0, 3.0, 4.0],
            orientation=[0.0, 1.0, 0.0],
            measured_rmt_mso=50.0,
        ),
        target=TargetConfig(
            label="Target",
            bundle_path=root / "derivatives" / "target.trk",
            coords=[10.0, 20.0, 30.0],
            scalp_coords=[11.0, 21.0, 31.0],
            orientation=[1.0, 0.0, 0.0],
            medoid_endpoint=True,
            didt=80e6,
            mso=60.0,
        ),
        options=OptionsConfig(
            roi_size_mm=20.0,
            activation_length_mm=6.0,
            field_mode="af",
            adm_optimization=True,
            opt_spatial_resolution=2.0,
            opt_angle_resolution=5.0,
            opt_search_angle=30.0,
            opt_search_radius=10.0,
            generate_visualizations=True,
            generate_3d_visualization=False,
            visualization_dpi=200,
            max_angular_deviation_deg=45.0,
            gwi_threshold_mm=3.0,
            mso_floor_ratio=0.7,
            mso_ceiling_ratio=1.4,
            max_workers=2,
            no_parallel=False,
        ),
        grid=GridConfig(
            coords=[10.0, 20.0, 30.0],
            scalp_coords=[11.0, 21.0, 31.0],
            orientation=[1.0, 0.0, 0.0],
            search_radius_mm=12.0,
            step_size_mm=3.0,
            cortex_depth_mm=2.0,
        ),
        workflow="estimation",
    )


def test_report_sidecars_preserve_normalized_bytes(tmp_path: Path) -> None:
    txt_path = tmp_path / "TIDE_Results_Target.txt"
    lines = [
        "===========================================",
        "--- TIDE Contract Report ---",
        "===========================================",
        "",
        "Subject: sub-CONTRACT",
        "Weighted MSO: 48.25",
        "Unweighted MSO: 49.75",
        "Status: PASS",
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = io.save_report_json(
        txt_path,
        "estimation",
        data={"values": np.array([48.25, 49.75]), "output": tmp_path / "result.trk"},
        text_lines=lines,
    )

    assert json_path is not None
    html_path = txt_path.with_suffix(".html")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert list(payload) == [
        "schema_version",
        "report_type",
        "source_txt",
        "generated_at",
        "text",
        "sections",
        "data",
    ]
    assert payload["text"]["lines"] == lines
    assert _digest(_normalize_artifact(txt_path.read_text(), tmp_path)) == (
        "7caee9021554f7623d7301d5f954c4f21f8e60bab1bb3ba230ddac863189be60"
    )
    assert _digest(_normalize_artifact(json_path.read_text(), tmp_path)) == (
        "6fa16d9ea897df8208b5e848446f20720c35661f482c816071ceb40145803cb7"
    )
    assert _digest(_normalize_artifact(html_path.read_text(), tmp_path)) == (
        "eeaa38302bbcbd380ce265326d2b7a010e0bd13a5dc824505cc9f5a6e7340ddd"
    )


def test_report_sidecar_write_failure_propagates(tmp_path, monkeypatch):
    txt_path = tmp_path / "report.txt"

    def fail_html(*args, **kwargs):
        raise OSError("html write failed")

    monkeypatch.setattr(io, "save_report_html", fail_html)

    with pytest.raises(OSError, match="html write failed"):
        io.save_report_json(txt_path, "test", text_lines=["content"])


def test_saved_configs_preserve_normalized_yaml_bytes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    matrix = np.eye(4).tolist()

    workflow_path = save_config_to_output(
        config,
        tmp_path / "workflow",
        "estimation",
        generated_calibration_matrix=matrix,
        generated_target_matrix=matrix,
        generated_calibration_scalp_coords=[1.0, 2.0, 3.0],
        generated_target_scalp_coords=[4.0, 5.0, 6.0],
        medoid_resolved=True,
    )
    grid_path = save_grid_point_config(
        config,
        tmp_path / "grid_P00",
        "grid_P00",
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        matrix,
        [13.0, 14.0, 15.0],
        [0.0, 1.0, 0.0],
        calibration_orientation=matrix,
    )

    workflow_text = _normalize_artifact(workflow_path.read_text(), tmp_path)
    grid_text = _normalize_artifact(grid_path.read_text(), tmp_path)
    assert _digest(workflow_text) == (
        "29035ad2218901cca3a8dcd00a6f8733904405d0a96c1dbdb1b67371f6f1e86e"
    )
    assert _digest(grid_text) == (
        "82d2463333ef575c7f2c807c3ec546cf5e71c78fde74f626c79a0c9a6c1e4984"
    )


def test_stmpx_dataset_name_round_trips_only_when_configured(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.options.stmpx_dataset_name = "20260717-TIDE-SUB_01"

    config_path = save_config_to_output(
        config,
        tmp_path / "workflow",
        "estimation",
    )

    raw = yaml.safe_load(config_path.read_text())
    replay_config = SimNIBSConfig.from_yaml(config_path)
    assert raw["options"]["stmpx_dataset_name"] == "20260717-TIDE-SUB_01"
    assert replay_config.options.stmpx_dataset_name == "20260717-TIDE-SUB_01"


@pytest.mark.parametrize(
    ("saved_workflow", "replay_workflow"),
    [
        ("estimation", "estimation"),
        ("grid_search", "grid"),
        ("simulation", "simulation"),
        ("optimization", "optimization"),
    ],
)
def test_saved_workflow_metadata_selects_replay_workflow(
    tmp_path: Path,
    saved_workflow: str,
    replay_workflow: str,
) -> None:
    config_path = save_config_to_output(
        _make_config(tmp_path),
        tmp_path / saved_workflow,
        saved_workflow,
    )

    replay_config = SimNIBSConfig.from_yaml(config_path)

    assert replay_config.workflow == replay_workflow


def test_saved_grid_point_metadata_selects_estimation_replay(tmp_path: Path) -> None:
    config_path = save_grid_point_config(
        _make_config(tmp_path),
        tmp_path / "grid_P00",
        "grid_P00",
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        np.eye(4).tolist(),
        [13.0, 14.0, 15.0],
        [0.0, 1.0, 0.0],
    )

    replay_config = SimNIBSConfig.from_yaml(config_path)

    assert replay_config.workflow == "estimation"


def test_top_level_workflow_overrides_saved_metadata(tmp_path: Path) -> None:
    config_path = save_config_to_output(
        _make_config(tmp_path),
        tmp_path / "estimation",
        "estimation",
    )
    raw = yaml.safe_load(config_path.read_text())
    raw["workflow"] = "simulation"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    replay_config = SimNIBSConfig.from_yaml(config_path)

    assert replay_config.workflow == "simulation"


def test_config_write_failure_propagates(tmp_path, monkeypatch):
    config = _make_config(tmp_path)

    def fail_dump(*args, **kwargs):
        raise OSError("config write failed")

    monkeypatch.setattr(yaml, "dump", fail_dump)

    with pytest.raises(OSError, match="config write failed"):
        save_config_to_output(config, tmp_path, "estimation")


def test_workflow_support_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    from tide.workflows._shared import (
        SINGLE_THREAD_ENV,
        calculate_target_in_field_metric,
        configure_worker_environment,
        single_thread_child_environment,
        split_vectors_by_streamline,
    )

    vectors = np.arange(24, dtype=float).reshape(8, 3)
    streamlines = [np.zeros((3, 3)), np.zeros((1, 3)), np.zeros((4, 3))]
    split = split_vectors_by_streamline(vectors, streamlines)
    assert [len(item) for item in split] == [3, 1, 4]
    assert np.shares_memory(split[0], vectors)
    assert np.array_equal(np.concatenate(split), vectors)

    for key in SINGLE_THREAD_ENV:
        monkeypatch.setenv(key, "7")
    with single_thread_child_environment():
        assert all(__import__("os").environ[key] == "1" for key in SINGLE_THREAD_ENV)
    assert all(__import__("os").environ[key] == "7" for key in SINGLE_THREAD_ENV)

    configure_worker_environment()
    assert all(__import__("os").environ[key] == "1" for key in SINGLE_THREAD_ENV)

    points = np.column_stack((np.arange(7, dtype=float), np.zeros((7, 2))))
    e_field = np.column_stack((np.arange(7, dtype=float), np.zeros((7, 2))))
    metric = calculate_target_in_field_metric(
        [points],
        [e_field],
        roi_center=[3.0, 0.0, 0.0],
        roi_size_mm=10.0,
        activation_length_mm=2.0,
        max_angular_deviation_deg=0.0,
    )
    assert metric == 1000.0


def _grid_reporting_context(
    root: Path,
    config: SimNIBSConfig,
) -> Any:
    from tide.workflows._grid_reporting import GridReportingContext

    return GridReportingContext(
        config=config,
        out_dir=root,
        sims_dir=root / "simulations",
        results_csv=root / "TIDE_grid_results.csv",
        fixed_scalp_coords=np.array([13.0, 14.0, 15.0]),
        grid_orientation_ref=[0.0, 1.0, 0.0],
        calibration_orientation=np.eye(4).tolist(),
        target_streamlines_full=[],
        target_vectors_in_m1=[],
        cst_result=SimpleNamespace(
            weight_source="contract weights",
            metric_unweighted=1200.0,
            aggregates_weighted=_fake_aggregates(1250.0),
            aggregates_unweighted=_fake_aggregates(1200.0),
        ),
        af_cst_calibration=1250.0,
        cst_align=0.5,
        cst_align_corrected=0.6,
        cst_depth=14.0,
        intensity_rmt=80e6,
        biological_threshold=6250.0,
        m1_matrix_str=str(np.eye(4).tolist()),
        spatial_mode="SIFT2 Weighted",
        num_workers=2,
        calibration_pose_qc={"status": "PASS", "reasons": []},
        start_time=90.0,
        worker_memory_model={
            "workers": 2,
            "requested_workers": 2,
            "num_grid_points": 1,
            "cpu_count": 8,
            "available_memory_gb": 40.0,
            "memory_worker_limit": 3,
            "memory_per_worker_gb": 12.0,
            "memory_reserve_gb": 4.0,
            "forced": False,
            "solver": "PARDISO",
        },
    )


def test_grid_result_rows_preserve_success_and_failure_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tide.workflows import _grid_reporting

    config = _make_config(tmp_path)
    context = _grid_reporting_context(tmp_path, config)
    context.sims_dir.mkdir()
    _grid_reporting.initialize_grid_results_csv(context.results_csv)
    monkeypatch.setattr(_grid_reporting, "save_grid_point_config", lambda **kwargs: None)
    monkeypatch.setattr(_grid_reporting, "_write_point_summary_txt", lambda **kwargs: None)

    success = SimpleNamespace(
        point_label="grid_P00",
        success=True,
        cortex_coord=[7.0, 8.0, 9.0],
        opt_scalp_coords=[10.0, 11.0, 12.0],
        opt_matrix=np.eye(4).tolist(),
        unweighted_mso_raw=49.75,
        weighted_mso_raw=48.25,
        unweighted_mso=49.75,
        weighted_mso=48.25,
        unweighted_mso_flag="WITHIN_RANGE",
        weighted_mso_flag="WITHIN_RANGE",
        sei_weighted=1.0363,
        sei_unweighted=1.005,
        multiplier_weighted=0.96497,
        multiplier_unweighted=0.99502,
        target_metric_weighted=1300.0,
        target_metric_unweighted=1260.0,
        target_aggregates_weighted=_fake_aggregates(1300.0),
        target_aggregates_unweighted=_fake_aggregates(1260.0),
        pose_qc={"status": "PASS", "reasons": []},
        tgt_align=0.7,
        tgt_align_corrected=0.8,
        tgt_depth=12.0,
    )
    failure = SimpleNamespace(point_label="grid_P01", success=False, sei_weighted=0.0)

    records = _grid_reporting.write_grid_results([success, failure], context)

    assert _digest(context.results_csv.read_text()) == (
        "93413d086506e4d3b2f7bd3a74af47498817515445fad4da27110dd756e9d88b"
    )
    assert records[0]["weighted_mso"] == 48.25
    assert records[0]["unweighted_mso"] == 49.75
    assert records[1] == {
        "label": "grid_P01",
        "weighted_mso": 999.9,
        "unweighted_mso": 999.9,
        "weighted_mso_raw": 999.9,
        "unweighted_mso_raw": 999.9,
        "weighted_mso_flag": "N/A",
        "unweighted_mso_flag": "N/A",
        "sei_weighted": None,
        "sei_unweighted": None,
        "sei_rank_pct": None,
        "multiplier_weighted": None,
        "multiplier_unweighted": None,
        "target_pose_qc": None,
        "target_align": None,
        "target_align_corrected": None,
        "target_depth": None,
    }


def _summary_kwargs(tmp_path: Path) -> dict:
    return {
        "subject_id": "sub-CONTRACT",
        "timestamp_str": "2026-01-01 00:00:00",
        "out_dir": tmp_path,
        "num_workers": 2,
        "t1w_path": tmp_path / "t1w.nii.gz",
        "cst_bundle_path": tmp_path / "cst.trk",
        "target_bundle_path": tmp_path / "target.trk",
        "spatial_mode": "Sphere",
        "weight_source": "Uniform",
        "roi_size_mm": 40.0,
        "activation_length_mm": 6.0,
        "calibration_label": "M1",
        "measured_rmt_mso": 50.0,
        "m1_matrix_str": "[[1.0]]",
        "af_cst_w": 1250.0,
        "af_cst_u": 1200.0,
        "intensity_rmt": 80.5,
        "biological_threshold": 2242.24,
        "target_label": "Target",
        "target_coords": [1.0, 2.0, 3.0],
        "opt_scalp_str": "N/A",
        "tgt_matrix_str": "[[1.0]]",
        "af_tgt_w": 1300.0,
        "af_tgt_u": 1260.0,
        "cst_align": 0.6,
        "tgt_align": 0.7,
        "cst_depth": 11.0,
        "tgt_depth": 12.0,
        "optimization_gain": 1.35,
        "ratio_at_m1": 0.806,
        "intensity_from_m1_position": 62.1,
        "intensity_raw_w": 48.25,
        "intensity_raw_u": 49.75,
        "intensity_clamped_w": 48.25,
        "intensity_clamped_u": 49.75,
        "intensity_flag_w": "WITHIN_RANGE",
        "intensity_flag_u": "WITHIN_RANGE",
        "mso_floor_ratio": 0.70,
        "sei_w": 1.0363,
        "sei_u": 1.005,
        "multiplier_w": 0.96497,
        "multiplier_u": 0.99502,
    }


def test_aggregator_sensitivity_block_is_purely_additive(tmp_path: Path) -> None:
    from tide.interfaces.unified_estimation import build_aggregator_sensitivity

    kwargs = _summary_kwargs(tmp_path)
    baseline = io.build_estimation_summary_lines(**kwargs)

    assert io.build_aggregator_sensitivity_lines(None) == []
    assert io.build_estimation_summary_lines(**kwargs, aggregator_sensitivity=None) == baseline

    sensitivity = build_aggregator_sensitivity(
        cst_weighted=_fake_aggregates(1250.0),
        cst_unweighted=_fake_aggregates(1200.0),
        target_weighted=_fake_aggregates(1300.0),
        target_unweighted=_fake_aggregates(1260.0),
        rmt=50.0,
    )
    extended = io.build_estimation_summary_lines(**kwargs, aggregator_sensitivity=sensitivity)

    assert extended[: len(baseline)] == baseline
    appended = extended[len(baseline) :]
    assert appended == io.build_aggregator_sensitivity_lines(sensitivity)
    assert "--- Aggregator Sensitivity ---" in appended
    for key in AGGREGATOR_KEYS:
        assert any(io.AGGREGATOR_LABELS[key] in line for line in appended)


def test_grid_results_csv_header_appends_aggregator_columns(tmp_path: Path) -> None:
    from tide.workflows import _grid_reporting

    frozen_columns = [
        "grid_point_labels",
        "grid_point_coords",
        "fixed_scalp_start_coords",
        "optimized_scalp_point_coords",
        "matrix4x4",
        "measured_m1_mso",
        "unweighted_mso_raw",
        "weighted_mso_raw",
        "unweighted_mso_clamped",
        "weighted_mso_clamped",
        "unweighted_mso_flag",
        "weighted_mso_flag",
        "sei_weighted",
        "sei_unweighted",
        "sei_rank_pct",
        "multiplier_weighted",
        "multiplier_unweighted",
    ]
    results_csv = tmp_path / "TIDE_grid_results.csv"
    _grid_reporting.initialize_grid_results_csv(results_csv)
    header = results_csv.read_text().splitlines()[0].split(",")

    assert header[: len(frozen_columns)] == frozen_columns
    assert header[len(frozen_columns) :] == [
        name
        for key in AGGREGATOR_KEYS
        for name in (f"intensity_raw_{key}_unweighted", f"intensity_raw_{key}_weighted")
    ]


def test_grid_point_report_exposes_both_weight_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tide.workflows import _grid_reporting

    config = _make_config(tmp_path)
    context = _grid_reporting_context(tmp_path, config)
    point_dir = tmp_path / "simulations" / "grid_P00"
    point_dir.mkdir(parents=True)
    result = SimpleNamespace(
        point_label="grid_P00",
        cortex_coord=[7.0, 8.0, 9.0],
        opt_scalp_coords=[10.0, 11.0, 12.0],
        opt_matrix=np.eye(4).tolist(),
        target_metric_weighted=1300.0,
        target_metric_unweighted=1260.0,
        target_aggregates_weighted=_fake_aggregates(1300.0),
        target_aggregates_unweighted=_fake_aggregates(1260.0),
        target_weight_source="External (target_weights.txt)",
        tgt_align=0.7,
        tgt_align_corrected=0.8,
        tgt_depth=12.0,
        weighted_mso_raw=48.25,
        unweighted_mso_raw=49.75,
        weighted_mso=48.25,
        unweighted_mso=49.75,
        weighted_mso_flag="WITHIN_RANGE",
        unweighted_mso_flag="WITHIN_RANGE",
        sei_weighted=1.0363,
        sei_unweighted=1.005,
        multiplier_weighted=0.96497,
        multiplier_unweighted=0.99502,
        pose_qc={"status": "PASS", "reasons": []},
    )
    captured = {}
    monkeypatch.setattr(
        _grid_reporting,
        "calculate_target_in_field_metric",
        lambda *args, **kwargs: 1000.0,
    )
    monkeypatch.setattr(
        _grid_reporting.io,
        "save_report_json",
        lambda *args, **kwargs: captured.update(kwargs["data"]),
    )

    _grid_reporting._write_point_summary_txt(
        config=config,
        point_dir=point_dir,
        result=result,
        tgt_streamlines_full=[],
        e_vecs_list_tgt_in_m1_full=[],
        cst_res=context.cst_result,
        af_cst_calibration=context.af_cst_calibration,
        cst_align=context.cst_align,
        cst_align_corrected=context.cst_align_corrected,
        cst_depth=context.cst_depth,
        intensity_rmt=context.intensity_rmt,
        biological_threshold=context.biological_threshold,
        m1_matrix_str=context.m1_matrix_str,
        spatial_mode=context.spatial_mode,
        num_workers=context.num_workers,
        out_dir=context.out_dir,
        calibration_pose_qc=context.calibration_pose_qc,
    )

    report = (point_dir / f"TIDE_Results_{config.target.label}.txt").read_text()
    assert "Weight Source: CST: contract weights; Target: External (target_weights.txt)" in report
    assert captured["weight_source_cst"] == "contract weights"
    assert captured["weight_source_target"] == "External (target_weights.txt)"


def test_grid_summary_preserves_normalized_text_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tide.workflows import _grid_reporting

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: Optional[Any] = None) -> "FixedDateTime":
            return cls(2026, 7, 13, 12, 0, 0, tzinfo=tz)

    config = _make_config(tmp_path)
    context = _grid_reporting_context(tmp_path, config)
    context.results_csv.write_text("header\n")
    result = SimpleNamespace(
        success=True,
        weighted_mso=48.25,
        unweighted_mso=49.75,
        weighted_mso_raw=48.25,
        unweighted_mso_raw=49.75,
        weighted_mso_flag="WITHIN_RANGE",
        unweighted_mso_flag="WITHIN_RANGE",
        multiplier_weighted=0.96497,
        multiplier_unweighted=0.99502,
    )
    records = [
        {
            "label": "grid_P00",
            "weighted_mso": 48.25,
            "unweighted_mso": 49.75,
            "weighted_mso_raw": 48.25,
            "unweighted_mso_raw": 49.75,
            "weighted_mso_flag": "WITHIN_RANGE",
            "unweighted_mso_flag": "WITHIN_RANGE",
            "sei_weighted": 1.0363,
            "sei_unweighted": 1.005,
            "sei_rank_pct": 100.0,
            "multiplier_weighted": 0.96497,
            "multiplier_unweighted": 0.99502,
            "target_pose_qc": {"status": "PASS", "reasons": []},
            "target_align": 0.7,
            "target_align_corrected": 0.8,
            "target_depth": 12.0,
        }
    ]
    monkeypatch.setattr(_grid_reporting.time, "time", lambda: 100.0)
    monkeypatch.setattr(_grid_reporting, "datetime", FixedDateTime)
    captured = {}
    monkeypatch.setattr(
        _grid_reporting.io,
        "save_report_json",
        lambda *args, **kwargs: captured.update(kwargs["data"]),
    )

    summary = _grid_reporting.write_grid_summary([result], records, context)

    normalized = _normalize_artifact(summary.summary_path.read_text(), tmp_path)
    assert _digest(normalized) == (
        "12a21a15a56057ef8a817f2b5f75c4b2d588a0545458fc8ac42c7b668f1eb8fd"
    )
    assert captured["weight_source_cst"] == "contract weights"
    assert captured["weight_source_target"] == "External (target_weights.txt)"
    assert captured["worker_memory_model"]["memory_per_worker_gb"] == 12.0
    assert summary.elapsed_time == 10.0


def test_grid_summary_separates_raw_and_clamped_statistics_and_excludes_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tide.workflows import _grid_reporting

    config = _make_config(tmp_path)
    context = _grid_reporting_context(tmp_path, config)
    context.results_csv.write_text("header\n")
    results = [
        SimpleNamespace(
            success=True,
            weighted_mso=35.0,
            unweighted_mso=35.0,
            weighted_mso_raw=20.0,
            unweighted_mso_raw=25.0,
            weighted_mso_flag="CLAMPED_LOW",
            unweighted_mso_flag="CLAMPED_LOW",
            multiplier_weighted=0.4,
            multiplier_unweighted=0.5,
        ),
        SimpleNamespace(
            success=True,
            weighted_mso=50.0,
            unweighted_mso=52.0,
            weighted_mso_raw=50.0,
            unweighted_mso_raw=52.0,
            weighted_mso_flag="WITHIN_RANGE",
            unweighted_mso_flag="WITHIN_RANGE",
            multiplier_weighted=1.0,
            multiplier_unweighted=1.04,
        ),
        SimpleNamespace(
            success=True,
            weighted_mso=float("nan"),
            unweighted_mso=float("nan"),
            weighted_mso_raw=float("nan"),
            unweighted_mso_raw=float("nan"),
            weighted_mso_flag="ESTIMATION_FAILED",
            unweighted_mso_flag="ESTIMATION_FAILED",
            multiplier_weighted=float("nan"),
            multiplier_unweighted=float("nan"),
        ),
        SimpleNamespace(
            success=False,
            weighted_mso=999.9,
            unweighted_mso=999.9,
            weighted_mso_raw=999.9,
            unweighted_mso_raw=999.9,
            weighted_mso_flag="N/A",
            unweighted_mso_flag="N/A",
            multiplier_weighted=None,
            multiplier_unweighted=None,
        ),
    ]
    monkeypatch.setattr(_grid_reporting.time, "time", lambda: 100.0)
    captured = {}
    monkeypatch.setattr(
        _grid_reporting.io,
        "save_report_json",
        lambda *args, **kwargs: captured.update(kwargs["data"]),
    )

    summary = _grid_reporting.write_grid_summary(results, [], context)

    assert summary.weighted_statistics["mean"] == 42.5
    assert summary.unweighted_statistics["mean"] == 43.5
    assert summary.weighted_raw_statistics["mean"] == 35.0
    assert summary.unweighted_raw_statistics["mean"] == 38.5
    assert summary.weighted_multiplier_statistics["mean"] == 0.7
    assert summary.unweighted_multiplier_statistics["mean"] == 0.77
    assert summary.status_counts == {
        "total_points": 4,
        "processing_failed": 1,
        "weighted": {
            "included": 2,
            "within_range": 1,
            "clamped_low": 1,
            "clamped_high": 0,
            "estimation_failed": 1,
        },
        "unweighted": {
            "included": 2,
            "within_range": 1,
            "clamped_low": 1,
            "clamped_high": 0,
            "estimation_failed": 1,
        },
    }
    assert captured["statistics"]["weighted_raw"]["mean"] == 35.0
    assert captured["status_counts"] == summary.status_counts

    report = summary.summary_path.read_text()
    assert "--- Raw Statistical Summary ---" in report
    assert "Processing Failures: 1" in report
    assert "Estimation Failed       | 1" in report
