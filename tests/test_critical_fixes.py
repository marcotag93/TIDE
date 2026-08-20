"""
Regression tests for the applied critical fixes.

Critical 1 (M1 optimization gating): a calibration block with only cortical
coords must take the optimization branch in both workflows. The gating reduces
to ``needs_optimization = not orientation_is_matrix(orientation)``; this is the
SimNIBS-free seam the workflows share, so it is tested directly here.

Critical 2 (E-field realignment): unmatched points must be filled with NaN
(not silently zeroed), perturbed-but-close points must still match, and a loss
above 1% must raise.
"""

import importlib
import json
import re
import sys
import textwrap
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Critical 1 — M1 optimization gating
# =============================================================================


class TestM1OptimizationGating:
    """The gating predicate shared by the estimation and grid workflows."""

    def test_coords_only_calibration_needs_optimization(self):
        from tide.utils.config import orientation_is_matrix

        # No orientation supplied (cortical coords only) -> must optimize.
        assert orientation_is_matrix(None) is False
        assert (not orientation_is_matrix(None)) is True

    def test_vector_orientation_needs_optimization(self):
        from tide.utils.config import orientation_is_matrix

        # A 3-vector pos_ydir reference is not a finished pose -> must optimize.
        assert orientation_is_matrix([10.0, 20.0, 30.0]) is False

    def test_eeg_label_needs_optimization(self):
        from tide.utils.config import orientation_is_matrix

        # An EEG label is not a finished pose -> must optimize.
        assert orientation_is_matrix("F8") is False

    def test_full_matrix_skips_optimization(self):
        from tide.utils.config import orientation_is_matrix

        matrix = [
            [1.0, 0.0, 0.0, -13.0],
            [0.0, 1.0, 0.0, -26.0],
            [0.0, 0.0, 1.0, 85.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        assert orientation_is_matrix(matrix) is True
        assert (not orientation_is_matrix(matrix)) is False

    def test_empty_list_is_not_a_matrix(self):
        from tide.utils.config import orientation_is_matrix

        assert orientation_is_matrix([]) is False


# =============================================================================
# Critical 2 — E-field realignment
# =============================================================================


def _diagonal_coords(n: int) -> np.ndarray:
    """n points on the (x=y=z) diagonal, spaced sqrt(3) mm apart."""
    return (np.arange(n).reshape(-1, 1) * np.ones((1, 3))).astype(float)


class TestRealignSampledField:
    """Tests for _realign_sampled_field (Critical 2)."""

    def test_perturbed_points_still_match(self):
        from tide.interfaces.sampling import _realign_sampled_field

        coords = _diagonal_coords(5)
        # 0.01 mm perturbation (norm ~0.017 mm) is well within the 0.1 mm tol.
        out_coords = coords + 1e-2
        out_vals = np.array(
            [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )

        result = _realign_sampled_field(coords, out_coords, out_vals)

        assert np.isfinite(result).all()
        assert np.allclose(result, out_vals)

    def test_unmatched_point_filled_with_nan(self):
        from tide.interfaces.sampling import _realign_sampled_field

        coords = _diagonal_coords(100)
        # Drop the first point: 1% loss (not > 1%, so no raise).
        out_coords = coords[1:].copy()
        out_vals = np.tile([1.0, 2.0, 3.0], (99, 1))

        result = _realign_sampled_field(coords, out_coords, out_vals)

        assert np.isnan(result[0]).all()
        assert np.isfinite(result[1:]).all()
        assert np.allclose(result[1:], out_vals)

    def test_raises_above_one_percent_loss(self):
        from tide.interfaces.sampling import _realign_sampled_field

        coords = _diagonal_coords(100)
        # Drop five points: 5% loss > 1% threshold -> raise.
        out_coords = coords[5:].copy()
        out_vals = np.tile([1.0, 2.0, 3.0], (95, 1))

        with pytest.raises(RuntimeError):
            _realign_sampled_field(coords, out_coords, out_vals)

    def test_no_zero_fill_for_unmatched(self):
        from tide.interfaces.sampling import _realign_sampled_field

        coords = _diagonal_coords(100)
        out_coords = coords[1:].copy()
        out_vals = np.tile([1.0, 2.0, 3.0], (99, 1))

        result = _realign_sampled_field(coords, out_coords, out_vals)

        # The unmatched row must be NaN, never a silent zero vector.
        assert not np.allclose(result[0], 0.0)
        assert np.isnan(result[0]).all()

    def test_equal_length_permutation_is_reordered(self, monkeypatch, tmp_path):
        from tide.interfaces import sampling

        coords = _diagonal_coords(4)
        permutation = np.array([2, 0, 3, 1])
        values = np.column_stack((np.arange(4), np.arange(4) + 10, np.arange(4) + 20))

        monkeypatch.setattr(
            sampling.simnibs_env,
            "find_get_fields_at_coordinates",
            lambda: "get_fields_at_coordinates",
        )

        def run_cli(*args, **kwargs):
            output = np.column_stack((coords[permutation], values[permutation]))
            np.savetxt(tmp_path / "bundle_coords_E.csv", output, delimiter=",")
            return SimpleNamespace(stdout="")

        monkeypatch.setattr(sampling.subprocess, "run", run_cli)

        result = sampling.sample_field_at_coordinates(
            tmp_path / "field.msh", coords, output_dir=tmp_path
        )

        assert np.array_equal(result, values)

    def test_duplicate_nearest_neighbor_assignment_raises(self):
        from tide.interfaces.sampling import _realign_sampled_field

        coords = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
        out_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        out_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(RuntimeError, match="duplicate nearest-neighbour"):
            _realign_sampled_field(coords, out_coords, out_vals)

    def test_coordinate_free_drop_cannot_return_wrong_length(self, monkeypatch, tmp_path):
        from tide.interfaces import sampling

        coords = _diagonal_coords(3)
        monkeypatch.setattr(
            sampling.simnibs_env,
            "find_get_fields_at_coordinates",
            lambda: "get_fields_at_coordinates",
        )

        def run_cli(*args, **kwargs):
            np.savetxt(tmp_path / "bundle_coords_E.csv", np.ones((2, 3)), delimiter=",")
            return SimpleNamespace(stdout="")

        monkeypatch.setattr(sampling.subprocess, "run", run_cli)

        with pytest.raises(RuntimeError, match="omitted coordinates"):
            sampling.sample_field_at_coordinates(
                tmp_path / "field.msh", coords, output_dir=tmp_path
            )


class TestSamplingCommandSafety:
    def test_windows_cmd_uses_literal_argument_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tide.interfaces import sampling

        coords = _diagonal_coords(3)
        cli_cmd = str(tmp_path / "get_fields_at_coordinates.cmd")
        mesh_path = tmp_path / "field with spaces.msh"
        file_prefix = "bundle with spaces"
        calls: list[tuple[object, dict[str, object]]] = []

        monkeypatch.setattr(
            sampling.simnibs_env,
            "find_get_fields_at_coordinates",
            lambda: cli_cmd,
        )

        def run_cli(args: object, **kwargs: object) -> SimpleNamespace:
            calls.append((args, kwargs))
            output = np.column_stack((coords, np.ones((3, 3))))
            np.savetxt(
                tmp_path / f"{file_prefix}_coords_E.csv",
                output,
                delimiter=",",
            )
            return SimpleNamespace(stdout="")

        monkeypatch.setattr(sampling.subprocess, "run", run_cli)

        sampling.sample_field_at_coordinates(
            mesh_path,
            coords,
            output_dir=tmp_path,
            file_prefix=file_prefix,
        )

        coords_csv = tmp_path / f"{file_prefix}_coords.csv"
        assert calls == [
            (
                [
                    cli_cmd,
                    "--mesh",
                    str(mesh_path),
                    "--csv",
                    str(coords_csv),
                ],
                {
                    "check": True,
                    "cwd": str(tmp_path),
                    "shell": False,
                    "stdout": sampling.subprocess.PIPE,
                    "stderr": sampling.subprocess.PIPE,
                    "text": True,
                },
            )
        ]

    def test_windows_cmd_rejects_shell_metacharacters(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tide.interfaces import sampling

        monkeypatch.setattr(
            sampling.simnibs_env,
            "find_get_fields_at_coordinates",
            lambda: str(tmp_path / "get_fields_at_coordinates.cmd"),
        )

        with pytest.raises(ValueError, match="shell metacharacters"):
            sampling.sample_field_at_coordinates(
                tmp_path / "field&echo injected.msh",
                _diagonal_coords(3),
                output_dir=tmp_path,
            )


class TestFreshSamplingArtifacts:
    """Sampling must consume only output produced by the current CLI call."""

    def test_stale_expected_csv_is_ignored_for_fresh_fallback(self, monkeypatch, tmp_path):
        from tide.interfaces import sampling

        coords = _diagonal_coords(3)
        stale_values = np.full((3, 3), -1.0)
        fresh_values = np.arange(9, dtype=float).reshape(3, 3)
        np.savetxt(
            tmp_path / "bundle_coords_E.csv",
            np.column_stack((coords, stale_values)),
            delimiter=",",
        )
        monkeypatch.setattr(
            sampling.simnibs_env,
            "find_get_fields_at_coordinates",
            lambda: "get_fields_at_coordinates",
        )

        def run_cli(*args, **kwargs):
            np.savetxt(
                tmp_path / "bundle_coords_vector.csv",
                np.column_stack((coords, fresh_values)),
                delimiter=",",
            )
            return SimpleNamespace(stdout="")

        monkeypatch.setattr(sampling.subprocess, "run", run_cli)

        result = sampling.sample_field_at_coordinates(
            tmp_path / "field.msh", coords, output_dir=tmp_path
        )

        assert np.array_equal(result, fresh_values)
        manifest = json.loads((tmp_path / ".tide_run_manifest.json").read_text())
        assert manifest["artifacts"]["sampling:bundle:E"]["selected"] == (
            "bundle_coords_vector.csv"
        )

    def test_updated_expected_csv_is_accepted(self, monkeypatch, tmp_path):
        from tide.interfaces import sampling

        coords = _diagonal_coords(3)
        expected_out = tmp_path / "bundle_coords_E.csv"
        expected_out.write_text("stale")
        fresh_values = np.arange(9, dtype=float).reshape(3, 3)
        monkeypatch.setattr(
            sampling.simnibs_env,
            "find_get_fields_at_coordinates",
            lambda: "get_fields_at_coordinates",
        )

        def run_cli(*args, **kwargs):
            np.savetxt(
                expected_out,
                np.column_stack((coords, fresh_values)),
                delimiter=",",
            )
            return SimpleNamespace(stdout="")

        monkeypatch.setattr(sampling.subprocess, "run", run_cli)

        result = sampling.sample_field_at_coordinates(
            tmp_path / "field.msh", coords, output_dir=tmp_path
        )

        assert np.array_equal(result, fresh_values)

    def test_ambiguous_fresh_fallback_csvs_raise(self, monkeypatch, tmp_path):
        from tide.interfaces import sampling

        coords = _diagonal_coords(3)
        monkeypatch.setattr(
            sampling.simnibs_env,
            "find_get_fields_at_coordinates",
            lambda: "get_fields_at_coordinates",
        )

        def run_cli(*args, **kwargs):
            output = np.column_stack((coords, np.ones((3, 3))))
            np.savetxt(tmp_path / "bundle_coords_a.csv", output, delimiter=",")
            np.savetxt(tmp_path / "bundle_coords_b.csv", output, delimiter=",")
            return SimpleNamespace(stdout="")

        monkeypatch.setattr(sampling.subprocess, "run", run_cli)

        with pytest.raises(RuntimeError, match="Ambiguous sampling output"):
            sampling.sample_field_at_coordinates(
                tmp_path / "field.msh", coords, output_dir=tmp_path
            )


class TestFreshSimulationArtifacts:
    """Simulation must not return meshes left by an earlier run."""

    @staticmethod
    def _load_interface(monkeypatch, run_simnibs):
        class Position:
            pass

        class TmsList:
            def add_position(self):
                return Position()

        class Session:
            def add_tmslist(self):
                return TmsList()

        fake_simnibs = SimpleNamespace(
            opt_struct=SimpleNamespace(TMSoptimize=object),
            run_simnibs=run_simnibs,
            sim_struct=SimpleNamespace(SESSION=Session),
        )
        monkeypatch.setitem(sys.modules, "simnibs", fake_simnibs)
        sys.modules.pop("tide.interfaces.simnibs_interface", None)
        module = importlib.import_module("tide.interfaces.simnibs_interface")
        monkeypatch.setattr(module, "run_simnibs", run_simnibs)
        return module

    def test_stale_mesh_is_ignored(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "sim"
        output_dir.mkdir()
        (output_dir / "stale_scalar.msh").write_text("stale")

        def run_simnibs(session):
            (output_dir / "fresh_scalar.msh").write_text("fresh")

        module = self._load_interface(monkeypatch, run_simnibs)

        result = module.SimNIBSInterface.run_simulation(
            mesh_path=tmp_path / "head.msh",
            output_dir=output_dir,
            coil_path=tmp_path / "coil.ccd",
            didt=1e6,
            coords=[0.0, 0.0, 0.0],
        )

        assert result == output_dir / "fresh_scalar.msh"
        manifest = json.loads((output_dir / ".tide_run_manifest.json").read_text())
        assert manifest["artifacts"]["simulation_mesh"]["selected"] == ("fresh_scalar.msh")
        sys.modules.pop("tide.interfaces.simnibs_interface", None)

    def test_ambiguous_fresh_meshes_raise(self, monkeypatch, tmp_path):
        output_dir = tmp_path / "sim"
        output_dir.mkdir()

        def run_simnibs(session):
            (output_dir / "first_scalar.msh").write_text("first")
            (output_dir / "second_scalar.msh").write_text("second")

        module = self._load_interface(monkeypatch, run_simnibs)

        with pytest.raises(RuntimeError, match="Ambiguous simulation output"):
            module.SimNIBSInterface.run_simulation(
                mesh_path=tmp_path / "head.msh",
                output_dir=output_dir,
                coil_path=tmp_path / "coil.ccd",
                didt=1e6,
                coords=[0.0, 0.0, 0.0],
            )
        sys.modules.pop("tide.interfaces.simnibs_interface", None)


# =============================================================================
# Bug 3 — failed estimate is flagged, not mislabeled as clamped
# =============================================================================


class TestEstimationFailedFlag:
    """apply_intensity_bounds must distinguish a failed estimate from a clamped one."""

    def test_nan_yields_estimation_failed(self):
        from tide.interfaces.unified_estimation import apply_intensity_bounds

        result = apply_intensity_bounds(float("nan"), rmt=50.0)

        assert result["flag"] == "ESTIMATION_FAILED"
        assert np.isnan(result["best_estimate"])
        assert np.isnan(result["model_raw"])

    def test_zero_is_still_clamped_low(self):
        from tide.interfaces.unified_estimation import apply_intensity_bounds

        # A real (finite) low estimate is clamped, not flagged as a failure.
        result = apply_intensity_bounds(0.0, rmt=50.0, floor_ratio=0.80)

        assert result["flag"] == "CLAMPED_LOW"
        assert result["best_estimate"] == 40.0


# =============================================================================
# Config defaults for optional stability knobs
# =============================================================================


class TestConfigOptionDefaults:
    """Missing optional config knobs must resolve to shipped defaults."""

    @staticmethod
    def _write_config(tmp_path, extra_options=""):
        cfg = tmp_path / "config.yml"
        cfg.write_text(textwrap.dedent(f"""
                subject:
                  id: sub-TEST
                  derivatives_path: {tmp_path}
                  m2m_path: m2m_sub-TEST
                  files:
                    t1w: t1.nii.gz
                coil:
                  coil_model: MagVenture_C-B60.ccd
                  coil_path: {tmp_path}
                  coil_distance_mm: 4.0
                  device_didt_max: 161e6
                experiment:
                  calibration:
                    label: M1
                    bundle_path: cst.trk
                    measured_rmt_mso: 50.0
                    coords: [0.0, 0.0, 0.0]
                  target:
                    label: Target
                    bundle_path: target.trk
                    coords: [0.0, 0.0, 0.0]
                options:
                  roi_size_mm: 20.0
                  activation_length_mm: 6.0
                """) + textwrap.indent(textwrap.dedent(extra_options), "  "))
        return cfg

    def test_missing_optional_bounds_and_angular_filter_defaults(self, tmp_path):
        from tide.utils.config import SimNIBSConfig

        config = SimNIBSConfig.from_yaml(self._write_config(tmp_path))

        assert config.options.max_angular_deviation_deg == 0.0
        assert config.options.mso_floor_ratio == 0.70
        assert config.options.mso_ceiling_ratio == 1.40
        assert config.options.gwi_threshold_mm == 3.0

    def test_configured_gwi_threshold_is_read(self, tmp_path):
        from tide.utils.config import SimNIBSConfig

        cfg = self._write_config(tmp_path, "gwi_threshold_mm: 5.0\n")

        assert SimNIBSConfig.from_yaml(cfg).options.gwi_threshold_mm == 5.0


def test_required_nifti_reference_failure_propagates(tmp_path):
    from tide.core.io import save_points_as_nifti

    with pytest.raises(FileNotFoundError, match="Reference image"):
        save_points_as_nifti(
            np.zeros((1, 3)),
            tmp_path / "missing.nii.gz",
            tmp_path / "output.nii.gz",
        )


class TestWorkflowPreflight:
    @staticmethod
    def _config(tmp_path, field_mode="af", weights_cst=None, weights_target=None):
        m2m_path = tmp_path / "m2m_sub-TEST"
        m2m_path.mkdir(exist_ok=True)
        mesh_path = m2m_path / "sub-TEST.msh"
        for path in (
            mesh_path,
            tmp_path / "t1.nii.gz",
            tmp_path / "coil.ccd",
            tmp_path / "cst.trk",
            tmp_path / "target.trk",
        ):
            path.write_text("")
        return SimpleNamespace(
            options=SimpleNamespace(
                field_mode=field_mode,
                roi_size_mm=30.0,
                activation_length_mm=6.0,
                adm_optimization=True,
                opt_spatial_resolution=2.0,
                opt_angle_resolution=10.0,
                opt_search_angle=30.0,
                opt_search_radius=10.0,
                generate_visualizations=True,
                generate_3d_visualization=False,
                visualization_dpi=300,
                max_angular_deviation_deg=0.0,
                gwi_threshold_mm=3.0,
                mso_floor_ratio=0.7,
                mso_ceiling_ratio=1.4,
                max_workers=None,
                no_parallel=False,
            ),
            subject=SimpleNamespace(
                id="sub-TEST",
                t1w_path=tmp_path / "t1.nii.gz",
                m2m_path=m2m_path,
                mesh_path=mesh_path,
                surface_path=None,
                weights_cst_path=weights_cst,
                weights_target_path=weights_target,
            ),
            coil=SimpleNamespace(
                coil_path=tmp_path / "coil.ccd",
                coil_distance_mm=4.0,
                device_didt_max=161e6,
            ),
            calibration=SimpleNamespace(
                label="M1",
                bundle_path=tmp_path / "cst.trk",
                coords=[0.0, 0.0, 0.0],
                scalp_coords=None,
                orientation=None,
                measured_rmt_mso=50.0,
            ),
            target=SimpleNamespace(
                label="Target",
                bundle_path=tmp_path / "target.trk",
                coords=[0.0, 0.0, 0.0],
                scalp_coords=None,
                orientation=[0.0, 1.0, 0.0],
                didt=None,
                mso=None,
            ),
            grid=SimpleNamespace(
                coords=[0.0, 0.0, 0.0],
                scalp_coords=None,
                orientation=[0.0, 1.0, 0.0],
                search_radius_mm=4.0,
                step_size_mm=4.0,
                cortex_depth_mm=2.0,
            ),
        )

    @pytest.mark.parametrize("workflow", ["estimation", "grid"])
    def test_dose_workflows_require_af(self, tmp_path, workflow):
        from tide.utils.config import validate_workflow_config

        config = self._config(tmp_path, field_mode="e_parallel")

        with pytest.raises(ValueError, match="field_mode"):
            validate_workflow_config(config, workflow)

    def test_standard_simulation_accepts_e_parallel(self, tmp_path):
        from tide.utils.config import validate_workflow_config

        config = self._config(tmp_path, field_mode="e_parallel")

        validate_workflow_config(config, "simulation")

    def test_omitted_weights_remain_valid(self, tmp_path):
        from tide.utils.config import validate_workflow_config

        validate_workflow_config(self._config(tmp_path), "estimation")

    @pytest.mark.parametrize("workflow", ["estimation", "grid"])
    def test_dose_workflows_require_both_configured_weights(self, tmp_path, workflow):
        from tide.utils.config import validate_workflow_config

        missing = tmp_path / "missing.txt"
        config = self._config(tmp_path, weights_target=missing)

        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            validate_workflow_config(config, workflow)

    def test_simulation_ignores_unused_cst_weight(self, tmp_path):
        from tide.utils.config import validate_workflow_config

        missing = tmp_path / "missing_cst.txt"
        config = self._config(tmp_path, weights_cst=missing)

        validate_workflow_config(config, "simulation")

    @pytest.mark.parametrize("threshold", [0.0, -1.0])
    def test_non_positive_gwi_threshold_is_rejected(self, tmp_path, threshold):
        from tide.utils.config import validate_workflow_config

        config = self._config(tmp_path)
        config.options.gwi_threshold_mm = threshold

        with pytest.raises(ValueError, match="options.gwi_threshold_mm"):
            validate_workflow_config(config, "estimation")


# =============================================================================
# Bug 5 — single source for the threshold/percentile helpers
# =============================================================================


class TestSingleSourceThreshold:
    """The bundle-analysis path must reuse the physics implementations."""

    def test_contiguous_threshold_is_shared(self):
        from tide.core import physics
        from tide.interfaces import unified_estimation

        assert (
            unified_estimation.get_max_contiguous_threshold is physics.get_max_contiguous_threshold
        )

    def test_weighted_percentile_is_shared(self):
        from tide.core import physics
        from tide.interfaces import unified_estimation

        assert unified_estimation.weighted_percentile is physics.weighted_percentile


# =============================================================================
# QC text reporting
# =============================================================================


class TestQcTextReporting:
    """QC additions must preserve existing text report content."""

    def test_optimization_result_adds_alignment_qc(self, tmp_path):
        from tide.core import io

        output_path = tmp_path / "opt_result.txt"
        matrix = np.eye(4)
        scalp_coords = np.array([1.0, 2.0, 3.0])

        io.save_optimization_result_txt(
            output_path,
            matrix,
            scalp_coords,
            pose_qc={"status": "PASS", "reasons": []},
            alignment_qc={
                "alignment": 0.0,
                "alignment_corrected": 0.5,
                "depth_mm": 12.3,
            },
        )

        text = output_path.read_text()

        assert "Optimized Scalp Position (x, y, z):" in text
        assert "Full 4x4 Transformation Matrix:" in text
        assert "--- COIL POSE QC ---" in text
        assert "Pose QC: PASS" in text
        assert "--- ALIGNMENT QC ---" in text
        assert "Alignment: 0.0000" in text
        assert "Alignment Corrected: 0.5000" in text
        assert "Depth: 12.3 mm" in text
        assert "--- FOR CONFIG FILE (copy/paste) ---" in text

        json_path = output_path.with_suffix(".json")
        payload = json.loads(json_path.read_text())
        assert payload["report_type"] == "optimization_result"
        assert payload["source_txt"] == str(output_path)
        assert payload["data"]["optimized_scalp_position"] == [1.0, 2.0, 3.0]
        assert "Alignment Corrected: 0.5000" in payload["text"]["content"]
        assert any(section["title"] == "ALIGNMENT QC" for section in payload["sections"])

    def test_mapping_summary_adds_qc_without_removing_existing_sections(self, tmp_path):
        from tide.core import io

        output_path = tmp_path / "mapping_summary.txt"
        image_dir = tmp_path / "visualizations"
        image_dir.mkdir()
        image_path = image_dir / "target_composite.png"
        image_path.write_bytes(b"png placeholder")

        io.save_mapping_summary(
            output_path,
            {
                "Timestamp": "2026-07-07 20:00:00",
                "Prefix": "target_af",
                "Mesh": "mesh.msh",
                "Bundle": "bundle.trk",
                "Anatomy": "t1w.nii.gz",
                "Mode": "af",
                "Threshold_Percent": "N/A",
                "Total_Streamlines": 10,
                "Max_Value": "2.0",
                "Min_Value": "-1.0",
                "Metrics": {"Robust Metric (Weighted)": "1.2345"},
                "QC": {
                    "pose_qc": {"status": "PASS", "reasons": []},
                    "alignment_qc": {
                        "alignment": 0.0,
                        "alignment_corrected": 0.25,
                        "depth_mm": 9.8,
                    },
                },
                "Output_Files": {"Tractogram": "target_af.trk"},
            },
        )

        text = output_path.read_text()

        assert "--- E-Field to Bundle Mapping Summary ---" in text
        assert "--- INPUTS ---" in text
        assert "--- PARAMETERS ---" in text
        assert "--- RESULTS ---" in text
        assert "--- ROBUST METRICS ---" in text
        assert "--- QC ---" in text
        assert "Target Coil Pose QC: PASS" in text
        assert "Target Alignment Corrected: 0.2500" in text
        assert "Target Depth: 9.8 mm" in text
        assert "--- OUTPUT FILES (in outdir) ---" in text

        json_path = output_path.with_suffix(".json")
        payload = json.loads(json_path.read_text())
        assert payload["report_type"] == "mapping_summary"
        assert payload["source_txt"] == str(output_path)
        assert payload["data"]["QC"]["alignment_qc"]["alignment_corrected"] == 0.25
        assert "Target Alignment Corrected: 0.2500" in payload["text"]["content"]
        assert any(section["title"] == "QC" for section in payload["sections"])

        html_path = output_path.with_suffix(".html")
        html = html_path.read_text()
        assert "TIDE Report" in html
        assert "mapping_summary.txt" in html
        assert "Target Coil Pose QC" in html
        assert "visualizations/target_composite.png" in html


class TestLightweightVisualizationPayloads:
    """Interactive report previews should stay bounded and compact."""

    def test_grid_streamline_payload_is_capped_and_has_no_rgb_arrays(self):
        from tide.interfaces.grid_visualization import _serialize_streamlines_for_html

        streamlines = [
            np.column_stack(
                [
                    np.linspace(i, i + 1, 12),
                    np.linspace(2 * i, 2 * i + 1, 12),
                    np.linspace(3 * i, 3 * i + 1, 12),
                ]
            )
            for i in range(10)
        ]

        payload = _serialize_streamlines_for_html(
            streamlines,
            max_streamlines=3,
            max_points_per_streamline=5,
            decimals=1,
        )

        assert len(payload) == 3
        assert all(len(points) <= 5 for points in payload)
        assert payload[0][0] == [0.0, 0.0, 0.0]
        assert "rgb" not in json.dumps(payload)


def _load_workflow_module(monkeypatch, module_name):
    simnibs = types.ModuleType("simnibs")
    simnibs.opt_struct = SimpleNamespace(TMSoptimize=object)
    simnibs.run_simnibs = lambda *args, **kwargs: None
    simnibs.sim_struct = SimpleNamespace(SESSION=object)
    simnibs.read_msh = lambda *args, **kwargs: None
    visualization_3d = types.ModuleType("tide.interfaces.visualization_3d")
    visualization_3d.PYVISTA_AVAILABLE = False
    visualization_3d.VisualizationConfig = object
    visualization_3d.generate_bundle_visualization = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "simnibs", simnibs)
    monkeypatch.setitem(sys.modules, "tide.interfaces.visualization_3d", visualization_3d)
    monkeypatch.delitem(sys.modules, "tide.interfaces.simnibs_interface", raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


class TestDoseEligibilityWorkflowGate:
    def test_estimation_worker_stops_before_simulation_for_warned_automatic_pose(
        self,
        tmp_path,
        monkeypatch,
    ):
        from tide.core.geometry import CoilPoseQC

        estimation = _load_workflow_module(monkeypatch, "tide.workflows.estimation")
        task = estimation.PipelineTask(
            task_type="target",
            label="Target",
            mesh_path=str(tmp_path / "head.msh"),
            m2m_path=str(tmp_path / "m2m"),
            output_dir=str(tmp_path / "out"),
            coil_path=str(tmp_path / "coil.ccd"),
            coil_distance_mm=4.0,
            target_coords=[0.0, 0.0, 0.0],
            scalp_coords=[0.0, 0.0, 1.0],
            orientation_ref=[0.0, 1.0, 0.0],
            needs_optimization=True,
            opt_didt=1e6,
            opt_search_radius=10.0,
            opt_spatial_resolution=2.0,
            opt_angle_resolution=10.0,
            opt_search_angle=30.0,
            use_adm=True,
            sim_didt=1e6,
            sim_coords=[0.0, 0.0, 0.0],
            sim_orientation=None,
            bundle_path=str(tmp_path / "target.trk"),
            t1w_path=str(tmp_path / "t1.nii.gz"),
            roi_coords=[0.0, 0.0, 0.0],
            roi_size_mm=20.0,
            field_mode="af",
        )
        qc = CoilPoseQC(status="WARN", reasons=("coil_normal_not_inward",))
        monkeypatch.setattr(estimation, "_configure_worker_environment", lambda: None)
        monkeypatch.setattr(
            estimation.SimNIBSInterface,
            "run_optimization",
            lambda **kwargs: (np.eye(4), np.array([0.0, 0.0, 1.0])),
        )
        monkeypatch.setattr(estimation, "evaluate_coil_pose_qc", lambda *args: qc)
        monkeypatch.setattr(
            estimation.io,
            "save_optimization_result_txt",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            estimation.SimNIBSInterface,
            "run_simulation",
            lambda **kwargs: pytest.fail("Simulation must not run for a rejected pose"),
        )

        result = estimation._run_pipeline_task(task)

        assert result.success is False
        assert "not dose-eligible" in result.error_message

    def test_grid_worker_excludes_warned_automatic_pose_before_simulation(
        self,
        tmp_path,
        monkeypatch,
    ):
        from tide.core import geometry
        from tide.core.geometry import CoilPoseQC

        grid_search = _load_workflow_module(monkeypatch, "tide.workflows.grid_search")
        task = grid_search.GridPointTask(
            index=0,
            cortex_coord=[0.0, 0.0, 0.0],
            point_label="grid_P00",
            point_dir=str(tmp_path / "grid_P00"),
            msh_file=str(tmp_path / "head.msh"),
            m2m_path=str(tmp_path / "m2m"),
            coil_path=str(tmp_path / "coil.ccd"),
            fixed_scalp_coords=[0.0, 0.0, 1.0],
            grid_orientation_ref=[0.0, 1.0, 0.0],
            target_bundle_path=str(tmp_path / "target.trk"),
            t1w_path=str(tmp_path / "t1.nii.gz"),
            surface_path=None,
            weights_target_path=None,
            adm_optimization=True,
            opt_spatial_resolution=2.0,
            opt_angle_resolution=10.0,
            opt_search_angle=30.0,
            opt_search_radius=10.0,
            field_mode="af",
            roi_size_mm=20.0,
            activation_length_mm=6.0,
            gwi_threshold_mm=3.0,
            max_angular_deviation_deg=0.0,
            measured_rmt_mso=50.0,
            af_cst_calibration=1.0,
            cst_metric_unweighted=1.0,
            mso_floor_ratio=0.7,
            mso_ceiling_ratio=1.4,
            generate_visualizations=False,
        )
        qc = CoilPoseQC(status="WARN", reasons=("coil_normal_not_inward",))
        monkeypatch.setattr(grid_search, "_configure_worker_environment", lambda: None)
        monkeypatch.setattr(
            grid_search.SimNIBSInterface,
            "run_optimization",
            lambda **kwargs: (np.eye(4), np.array([0.0, 0.0, 1.0])),
        )
        monkeypatch.setattr(geometry, "evaluate_coil_pose_qc", lambda *args: qc)
        monkeypatch.setattr(
            grid_search.io,
            "save_optimization_result_txt",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            grid_search.SimNIBSInterface,
            "run_simulation",
            lambda **kwargs: pytest.fail("Simulation must not run for a rejected pose"),
        )

        result = grid_search.process_grid_point(task)

        assert result.success is False
        assert "not dose-eligible" in result.error_message


class TestGridWorkerResourcePlan:
    def test_explicit_workers_are_capped_by_pardiso_memory(self, monkeypatch):
        grid_search = _load_workflow_module(monkeypatch, "tide.workflows.grid_search")
        monkeypatch.setattr(grid_search, "_get_available_memory_gb", lambda: 28.0)
        monkeypatch.setattr(grid_search.os, "cpu_count", lambda: 16)

        plan = grid_search._resolve_grid_worker_plan(4, 10)

        assert plan.workers == 2
        assert plan.memory_worker_limit == 2
        assert plan.memory_per_worker_gb == 12.0
        assert plan.memory_reserve_gb == 4.0
        assert plan.forced is False

    def test_force_override_retains_explicit_workers(self, monkeypatch):
        grid_search = _load_workflow_module(monkeypatch, "tide.workflows.grid_search")
        monkeypatch.setattr(grid_search, "_get_available_memory_gb", lambda: 28.0)
        monkeypatch.setenv("TIDE_GRID_FORCE_WORKERS", "1")

        plan = grid_search._resolve_grid_worker_plan(4, 10)

        assert plan.workers == 4
        assert plan.memory_worker_limit == 2
        assert plan.forced is True

    def test_unknown_memory_uses_one_worker_without_override(self, monkeypatch):
        grid_search = _load_workflow_module(monkeypatch, "tide.workflows.grid_search")
        monkeypatch.setattr(grid_search, "_get_available_memory_gb", lambda: None)

        plan = grid_search._resolve_grid_worker_plan(4, 10)

        assert plan.workers == 1
        assert plan.memory_worker_limit == 1


class TestGridProgressCallbacks:
    def test_parallel_completions_all_emit_callbacks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        grid_search = _load_workflow_module(monkeypatch, "tide.workflows.grid_search")
        tasks = [
            SimpleNamespace(index=0, point_label="grid_P00", cortex_coord=[0.0, 0.0, 0.0]),
            SimpleNamespace(index=1, point_label="grid_P01", cortex_coord=[1.0, 0.0, 0.0]),
        ]
        success_future = grid_search.Future()
        success_future.set_result(
            grid_search.GridPointResult(
                index=0,
                point_label="grid_P00",
                success=True,
                weighted_mso=50.0,
                unweighted_mso=51.0,
                cortex_coord=tasks[0].cortex_coord,
                opt_scalp_coords=None,
                opt_matrix=None,
            )
        )
        failed_future = grid_search.Future()
        failed_future.set_exception(RuntimeError("worker failed"))
        callbacks = []

        results, failed_count = grid_search._collect_parallel_grid_results(
            {
                success_future: tasks[0],
                failed_future: tasks[1],
            },
            num_points=2,
            use_parallel_ui=False,
            progress_callback=lambda completed, total, label: callbacks.append(
                (completed, total, label)
            ),
        )

        assert failed_count == 1
        assert {result.point_label for result in results} == {"grid_P00", "grid_P01"}
        assert [completed for completed, _, _ in callbacks] == [1, 2]
        assert {label for _, _, label in callbacks} == {"grid_P00", "grid_P01"}
        assert all(total == 2 for _, total, _ in callbacks)


class TestVisualizationControls:
    def test_grid_point_optional_artifacts_are_skipped(self, monkeypatch, tmp_path):
        grid_search = _load_workflow_module(monkeypatch, "tide.workflows.grid_search")
        task = SimpleNamespace(
            generate_visualizations=False,
            t1w_path=str(tmp_path / "t1.nii.gz"),
            point_dir=str(tmp_path),
            point_label="grid_P00",
            cortex_coord=[0.0, 0.0, 0.0],
            roi_size_mm=20.0,
        )
        monkeypatch.setattr(
            grid_search.io,
            "save_points_as_nifti",
            lambda *args, **kwargs: pytest.fail("NIfTI output must be disabled"),
        )
        monkeypatch.setattr(
            grid_search,
            "save_af_visualization",
            lambda *args, **kwargs: pytest.fail("plot output must be disabled"),
        )

        grid_search._save_grid_point_visualizations(
            task,
            [np.zeros((4, 3))],
            [np.ones(4)],
        )

    def test_grid_visualizer_can_skip_interactive_html(self, monkeypatch, tmp_path):
        from tide.interfaces import grid_visualization

        records = [{"label": "grid_P00"}]
        calls = []
        monkeypatch.setattr(grid_visualization, "parse_grid_csv", lambda path: records)
        monkeypatch.setattr(
            grid_visualization,
            "generate_scalar_nifti",
            lambda *args: calls.append("nifti"),
        )
        monkeypatch.setattr(
            grid_visualization,
            "generate_interactive_html",
            lambda *args, **kwargs: calls.append("html"),
        )

        grid_visualization.run_grid_visualization(
            csv_path=tmp_path / "results.csv",
            t1w_path=tmp_path / "t1.nii.gz",
            trk_path=tmp_path / "bundle.trk",
            output_dir=tmp_path / "visualization",
            generate_interactive=False,
        )

        assert calls == ["nifti"]

    def test_grid_visualizer_excludes_failed_nonfinite_rows(self, tmp_path):
        from tide.interfaces.grid_visualization import parse_grid_csv

        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "grid_point_labels,grid_point_coords,unweighted_mso_raw,"
            "weighted_mso_raw,unweighted_mso_clamped,weighted_mso_clamped,"
            "unweighted_mso_flag,weighted_mso_flag\n"
            'grid_P00,"[1, 2, 3]",50,49,50,49,WITHIN_RANGE,WITHIN_RANGE\n'
            'grid_P01,"[4, 5, 6]",nan,nan,nan,nan,ESTIMATION_FAILED,'
            "ESTIMATION_FAILED\n"
        )

        records = parse_grid_csv(csv_path)

        assert [record["label"] for record in records] == ["grid_P00"]

    @staticmethod
    def _clamp_map_records():
        return [
            {
                "label": "grid_P00",
                "coords": [1.0, 2.0, 3.0],
                "weighted_mso_raw": 42.0,
                "weighted_mso_clamped": 42.0,
                "unweighted_mso_raw": 41.0,
                "unweighted_mso_clamped": 41.0,
                "weighted_mso_flag": "WITHIN_RANGE",
                "unweighted_mso_flag": "WITHIN_RANGE",
                "sei_weighted": 1.1,
            },
            {
                "label": "grid_P01",
                "coords": [4.0, 5.0, 6.0],
                "weighted_mso_raw": 95.0,
                "weighted_mso_clamped": 70.0,
                "unweighted_mso_raw": 90.0,
                "unweighted_mso_clamped": 70.0,
                "weighted_mso_flag": "CLAMPED_HIGH",
                "unweighted_mso_flag": "CLAMPED_HIGH",
                "sei_weighted": 0.6,
            },
        ]

    @staticmethod
    def _write_reference_t1w(tmp_path):
        import nibabel as nib

        t1w_path = tmp_path / "t1.nii.gz"
        image = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), np.eye(4))
        nib.save(image, str(t1w_path))
        return t1w_path

    def test_grid_nifti_maps_separate_raw_clamped_and_flag(self, tmp_path):
        import nibabel as nib

        from tide.interfaces.grid_visualization import CLAMP_FLAG_CODES, generate_scalar_nifti

        records = self._clamp_map_records()
        generate_scalar_nifti(records, self._write_reference_t1w(tmp_path), tmp_path)

        clamped = nib.load(str(tmp_path / "grid_mso_map.nii.gz")).get_fdata()
        raw = nib.load(str(tmp_path / "grid_mso_raw_map.nii.gz")).get_fdata()
        flags = nib.load(str(tmp_path / "grid_mso_flag_map.nii.gz")).get_fdata()

        # grid_mso_map.nii.gz keeps its historical clamped meaning.
        assert clamped[1, 2, 3] == pytest.approx(42.0)
        assert clamped[4, 5, 6] == pytest.approx(70.0)
        assert raw[1, 2, 3] == pytest.approx(42.0)
        assert raw[4, 5, 6] == pytest.approx(95.0)
        assert flags[1, 2, 3] == CLAMP_FLAG_CODES["WITHIN_RANGE"]
        assert flags[4, 5, 6] == CLAMP_FLAG_CODES["CLAMPED_HIGH"]
        assert flags[0, 0, 0] == 0

    def test_grid_label_sidecar_keeps_frozen_column_prefix(self, tmp_path):
        from tide.interfaces.grid_visualization import generate_scalar_nifti

        records = self._clamp_map_records()
        generate_scalar_nifti(records, self._write_reference_t1w(tmp_path), tmp_path)

        header = (tmp_path / "grid_mso_labels.tsv").read_text().splitlines()[0].split("\t")
        frozen = [
            "label",
            "x_ras",
            "y_ras",
            "z_ras",
            "weighted_mso_clamped",
            "weighted_mso_raw",
            "unweighted_mso_clamped",
            "sei_weighted",
        ]

        assert header[: len(frozen)] == frozen
        assert header[len(frozen) :] == [
            "unweighted_mso_raw",
            "weighted_mso_flag",
            "unweighted_mso_flag",
        ]

    def test_grid_viewer_colours_both_raw_and_clamped_and_defaults_to_raw(self):
        from tide.interfaces.grid_visualization import DEFAULT_VIEW_MODE, VIEW_MODES, _mso_to_hex

        assert DEFAULT_VIEW_MODE == "raw"
        assert VIEW_MODES == {
            "raw": "weighted_mso_raw",
            "clamped": "weighted_mso_clamped",
        }

        records = self._clamp_map_records()
        scales = {
            mode: {
                "vmin": min(r[field] for r in records),
                "vmax": max(r[field] for r in records),
            }
            for mode, field in VIEW_MODES.items()
        }
        saturated = records[1]
        raw_hex = _mso_to_hex(
            saturated["weighted_mso_raw"],
            scales["raw"]["vmin"],
            scales["raw"]["vmax"],
        )
        clamped_hex = _mso_to_hex(
            saturated["weighted_mso_clamped"],
            scales["clamped"]["vmin"],
            scales["clamped"]["vmax"],
        )

        # Same point, different scale extents: colours must not be conflated.
        assert scales["raw"]["vmax"] > scales["clamped"]["vmax"]
        assert isinstance(raw_hex, str) and isinstance(clamped_hex, str)

    def test_standard_mapping_skips_optional_nifti(self, tmp_path, monkeypatch):
        standard = _load_workflow_module(monkeypatch, "tide.workflows.standard")
        streamline = np.column_stack((np.arange(4, dtype=float), np.zeros(4), np.zeros(4)))
        config = SimpleNamespace(
            subject=SimpleNamespace(
                t1w_path=tmp_path / "t1.nii.gz",
                surface_path=None,
                weights_target_path=None,
            ),
            target=SimpleNamespace(
                label="Target",
                bundle_path=tmp_path / "target.trk",
                coords=None,
            ),
            options=SimpleNamespace(
                field_mode="af",
                max_angular_deviation_deg=0.0,
                gwi_threshold_mm=3.0,
                roi_size_mm=20.0,
                activation_length_mm=6.0,
                generate_visualizations=False,
                generate_3d_visualization=False,
            ),
        )
        sft = SimpleNamespace(streamlines=[streamline])
        monkeypatch.setattr(standard.tractography, "load_tract", lambda *args: sft)
        monkeypatch.setattr(
            standard,
            "sample_field_at_coordinates",
            lambda *args, **kwargs: np.zeros((4, 3)),
        )
        monkeypatch.setattr(
            standard,
            "split_vectors_by_streamline",
            lambda *args: [np.zeros((4, 3))],
        )
        monkeypatch.setattr(
            standard.physics,
            "calculate_scalar_map",
            lambda *args, **kwargs: (
                [streamline],
                [np.ones(4)],
                [np.ones(4)],
                np.array([0]),
            ),
        )
        monkeypatch.setattr(standard.io, "save_tract_with_data", lambda *args: None)
        monkeypatch.setattr(
            standard.io,
            "save_points_as_nifti",
            lambda *args, **kwargs: pytest.fail("NIfTI output must be disabled"),
        )
        captured = {}
        monkeypatch.setattr(
            standard.io,
            "save_mapping_summary",
            lambda path, data: captured.update(data),
        )
        monkeypatch.setattr(
            standard,
            "analyze_bundle",
            lambda *args, **kwargs: SimpleNamespace(
                metric_weighted=1.0,
                metric_unweighted=1.0,
                aggregates_weighted={},
                aggregates_unweighted={},
            ),
        )
        monkeypatch.setattr(
            standard,
            "log",
            SimpleNamespace(
                debug=lambda *args: None,
                highlight=lambda *args: None,
                warning=lambda *args: None,
            ),
        )

        standard._process_bundle_mapping(config, tmp_path / "field.msh", tmp_path)

        assert captured["Output_Files"]["NIfTI map"] == "Disabled by configuration"


class TestSurfaceLoadFailures:
    def test_grid_surface_load_failure_propagates(self, tmp_path, monkeypatch):
        grid_search = _load_workflow_module(monkeypatch, "tide.workflows.grid_search")
        surface_path = tmp_path / "surface.white"
        surface_path.write_text("")
        config = SimpleNamespace(
            subject=SimpleNamespace(
                id="sub-TEST",
                mesh_path=tmp_path / "head.msh",
                derivatives_path=tmp_path,
                surface_path=surface_path,
                weights_cst_path=None,
                weights_target_path=None,
            ),
            target=SimpleNamespace(label="Target", medoid_endpoint=False),
            options=SimpleNamespace(no_parallel=True, max_workers=1),
        )
        monkeypatch.setattr(grid_search, "validate_workflow_config", lambda *args: None)
        monkeypatch.setattr(
            grid_search,
            "load_surface_tree",
            lambda *args: (_ for _ in ()).throw(ValueError("invalid surface")),
        )

        with pytest.raises(ValueError, match="invalid surface"):
            grid_search.run_grid_search_workflow(config, console_ui=False)

    def test_standard_mapping_surface_load_failure_propagates(self, tmp_path, monkeypatch):
        standard = _load_workflow_module(monkeypatch, "tide.workflows.standard")
        streamline = np.column_stack((np.arange(4, dtype=float), np.zeros(4), np.zeros(4)))
        config = SimpleNamespace(
            subject=SimpleNamespace(
                t1w_path=tmp_path / "t1.nii.gz",
                surface_path=tmp_path / "surface.white",
                weights_target_path=None,
            ),
            target=SimpleNamespace(
                label="Target",
                bundle_path=tmp_path / "target.trk",
                coords=None,
            ),
            options=SimpleNamespace(
                field_mode="af",
                max_angular_deviation_deg=0.0,
                gwi_threshold_mm=3.0,
                roi_size_mm=20.0,
                activation_length_mm=6.0,
                generate_visualizations=True,
                generate_3d_visualization=False,
            ),
        )
        sft = SimpleNamespace(streamlines=[streamline])
        monkeypatch.setattr(standard.tractography, "load_tract", lambda *args: sft)
        monkeypatch.setattr(
            standard,
            "sample_field_at_coordinates",
            lambda *args, **kwargs: np.zeros((4, 3)),
        )
        monkeypatch.setattr(
            standard,
            "split_vectors_by_streamline",
            lambda *args: [np.zeros((4, 3))],
        )
        monkeypatch.setattr(
            standard.physics,
            "calculate_scalar_map",
            lambda *args, **kwargs: (
                [streamline],
                [np.ones(4)],
                [np.ones(4)],
                np.array([0]),
            ),
        )
        monkeypatch.setattr(standard.io, "save_tract_with_data", lambda *args: None)
        monkeypatch.setattr(standard.io, "save_points_as_nifti", lambda *args, **kwargs: None)
        monkeypatch.setattr(standard.io, "save_mapping_summary", lambda *args: None)
        monkeypatch.setattr(
            standard,
            "load_surface_tree",
            lambda *args: (_ for _ in ()).throw(ValueError("invalid surface")),
        )
        monkeypatch.setattr(
            standard,
            "analyze_bundle",
            lambda *args, **kwargs: SimpleNamespace(
                metric_weighted=1.0,
                metric_unweighted=1.0,
                aggregates_weighted={},
                aggregates_unweighted={},
            ),
        )
        monkeypatch.setattr(
            standard,
            "log",
            SimpleNamespace(
                debug=lambda *args: None,
                highlight=lambda *args: None,
                warning=lambda *args: None,
            ),
        )

        with pytest.raises(standard.WorkflowError, match="invalid surface"):
            standard._process_bundle_mapping(
                config,
                tmp_path / "field.msh",
                tmp_path,
            )


def test_standard_optimization_forwards_eeg_orientation(tmp_path, monkeypatch):
    standard = _load_workflow_module(monkeypatch, "tide.workflows.standard")
    config = SimpleNamespace(
        subject=SimpleNamespace(
            id="sub-TEST",
            derivatives_path=tmp_path,
            mesh_path=tmp_path / "head.msh",
        ),
        target=SimpleNamespace(
            label="Target",
            medoid_endpoint=False,
            bundle_path=None,
            coords=[0.0, 0.0, 0.0],
            scalp_coords=[0.0, 0.0, 1.0],
            orientation="F8",
            didt=None,
            mso=None,
        ),
        coil=SimpleNamespace(
            coil_model="coil.ccd",
            coil_path=tmp_path / "coil.ccd",
            coil_distance_mm=4.0,
            device_didt_max=161e6,
        ),
        options=SimpleNamespace(
            roi_size_mm=20.0,
            activation_length_mm=6.0,
            field_mode="af",
            adm_optimization=True,
            opt_search_radius=10.0,
            opt_spatial_resolution=2.0,
            opt_angle_resolution=10.0,
            opt_search_angle=30.0,
        ),
    )
    captured = {}

    def run_optimization(**kwargs):
        captured.update(kwargs)
        return np.eye(4), np.array([0.0, 0.0, 1.0])

    pose_qc = SimpleNamespace(
        status="PASS",
        reasons=(),
        as_dict=lambda: {},
    )
    monkeypatch.setattr(standard.SimNIBSInterface, "run_optimization", run_optimization)
    monkeypatch.setattr(standard, "evaluate_coil_pose_qc", lambda *args: pose_qc)
    monkeypatch.setattr(standard.io, "save_optimization_result_txt", lambda *args, **kwargs: None)
    monkeypatch.setattr(standard, "save_config_to_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        standard,
        "log",
        SimpleNamespace(
            highlight=lambda *args: None,
            info=lambda *args: None,
            warning=lambda *args: None,
        ),
    )

    standard.run_standard_optimization(config)

    assert captured["orientation_ref"] == "F8"


# =============================================================================
# H-15 — laboratory-verified Softaxic STMPX export
# =============================================================================


_P01_R_TARGET_MATRIX = [
    [-0.8853008359570536, -0.32864786304693355, -0.32898785991040186, 28.00473127057423],
    [-0.4542258111548085, 0.7626956250015824, 0.4604066638138527, -70.37541724363432],
    [0.09960593523730635, 0.5570331818824887, -0.8244954165714761, 74.36249582644216],
    [0.0, 0.0, 0.0, 1.0],
]


def _write_stmpx_export_inputs(tmp_path: Path) -> tuple[Path, Path]:
    stmpx_path = tmp_path / "P01_R.stmpx"
    stmpx_path.write_text(
        '<!DOCTYPE stmp>\n<stmp><fmpm dataset="20260304-092048_SUB_01">'
        '<fmp global="0" id="P001"><fp id="8700449" /></fmp>'
        "</fmpm></stmp>"
    )
    results_path = tmp_path / "TIDE_Results_Target.txt"
    results_path.write_text(textwrap.dedent(f"""
            --- Target Estimation (Target) ---
              Target Coords (Cortex): [21.0, -59.0, 56.0]
              Optimized Scalp Position: [28.00, -70.38, 74.36]
              Optimized Matrix: {_P01_R_TARGET_MATRIX}
            --- Geometric Analysis ---
            """).strip())
    return stmpx_path, results_path


def test_stmpx_export_matches_laboratory_verified_p01_r(tmp_path, monkeypatch):
    from tide.interfaces import stmpx

    stmpx_path, results_path = _write_stmpx_export_inputs(tmp_path)
    original = stmpx_path.read_bytes()
    (tmp_path / "P01_R_updated.stmpx").write_text("stale output")
    monkeypatch.setattr(stmpx.time, "time", lambda: 1772752999.345)

    output_path = stmpx.export_target_to_stmpx(
        stmpx_path,
        results_path,
        dataset_name="20260717-TIDE-SUB_01",
    )

    assert output_path == tmp_path / "P01_R_updated.stmpx"
    assert stmpx_path.read_bytes() == original
    assert output_path.read_bytes().startswith(b"<!DOCTYPE stmp>\n")

    root = ET.parse(output_path).getroot()
    fmpm = root.find("fmpm")
    assert fmpm is not None
    assert fmpm.get("dataset") == "20260717-TIDE-SUB_01"
    assert [node.get("id") for node in fmpm.findall("fmp")] == [
        "P001",
        "Target_Estimation",
    ]

    fp = fmpm.findall("fmp")[-1].find("fp")
    assert fp is not None
    assert list(fp.attrib) == stmpx.FP_ATTR_ORDER
    assert fp.attrib == {
        "m00": "-0.328648",
        "m10": "0.762696",
        "y": "-70.3754",
        "m21": "0.099606",
        "m02": "0.328988",
        "m22": "0.824495",
        "m01": "-0.885301",
        "m12": "-0.460407",
        "ts": "1772752999345",
        "z": "74.3625",
        "id": "8700449",
        "x": "28.0047",
        "m20": "0.557033",
        "m11": "-0.454226",
    }
    assert fp.find("b").attrib == {"x": "21.0000", "y": "-59.0000", "z": "56.0000"}
    assert fp.find("f").attrib == {"x": "28.0000", "y": "-70.3800", "z": "74.3600"}


def test_stmpx_export_preserves_existing_dataset_when_unset(tmp_path, monkeypatch):
    from tide.interfaces import stmpx

    stmpx_path, results_path = _write_stmpx_export_inputs(tmp_path)
    monkeypatch.setattr(stmpx.time, "time", lambda: 1772752999.345)

    output_path = stmpx.export_target_to_stmpx(stmpx_path, results_path)

    fmpm = ET.parse(output_path).getroot().find("fmpm")
    assert fmpm is not None
    assert fmpm.get("dataset") == "20260304-092048_SUB_01"


def test_stmpx_rejects_xml_entities(tmp_path: Path) -> None:
    from tide.interfaces.stmpx import validate_stmpx_input

    stmpx_path = tmp_path / "entity.stmpx"
    stmpx_path.write_text(
        '<!DOCTYPE stmp [<!ENTITY payload "boom">]>' "<stmp><fmpm>&payload;</fmpm></stmp>"
    )

    with pytest.raises(ValueError, match="Invalid STMPX XML"):
        validate_stmpx_input(stmpx_path)
