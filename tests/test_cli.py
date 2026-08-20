"""
Subprocess Tests for the TIDE CLI Ordering Contract
====================================================
Verify that ``main.py`` performs SimNIBS environment discovery before parsing
the workflow config or creating any output, and that the informational
commands (``--version`` / ``--help``) bypass SimNIBS entirely. These tests are
environment-independent: they neither require SimNIBS to be installed nor
mutate the caller's environment.
"""

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = REPO_ROOT / "main.py"
SRC_DIR = REPO_ROOT / "src"
_RELAUNCH_MARKER = "_TIDE_SIMNIBS_RELAUNCHED"
sys.path.insert(0, str(SRC_DIR))


def _simnibs_importable() -> bool:
    proc = subprocess.run([sys.executable, "-c", "import simnibs"], capture_output=True)
    return proc.returncode == 0


SIMNIBS_PRESENT = _simnibs_importable()


def _run_cli(args, path_dir: Path):
    """Invoke the CLI with ``simnibs`` hidden from PATH and no relaunch marker."""
    env = os.environ.copy()
    env["PATH"] = str(path_dir)  # a dir with no 'simnibs' launcher
    env["PYTHONPATH"] = str(SRC_DIR)
    env.pop(_RELAUNCH_MARKER, None)
    return subprocess.run(
        [sys.executable, str(MAIN_PY), *args],
        env=env,
        capture_output=True,
        text=True,
    )


def _write_preflight_config(
    tmp_path: Path,
    *,
    field_mode: str = "af",
    weights_cst: Optional[Path] = None,
    weights_target: Optional[Path] = None,
) -> tuple[Path, Path]:
    out_dir = tmp_path / "derivatives_out"
    weight_lines = []
    if weights_cst is not None:
        weight_lines.append(f"                weights_cst: {weights_cst}")
    if weights_target is not None:
        weight_lines.append(f"                weights_target: {weights_target}")
    weights_yaml = "\n".join(weight_lines)
    if weights_yaml:
        weights_yaml = f"\n{weights_yaml}"

    cfg = tmp_path / "preflight.yml"
    cfg.write_text(textwrap.dedent(f"""
            subject:
              id: sub-TEST
              derivatives_path: {out_dir}
              m2m_path: {tmp_path / "m2m_sub-TEST"}
              files:
                t1w: {tmp_path / "t1.nii.gz"}{weights_yaml}
            coil:
              coil_model: MagVenture_C-B60.ccd
              coil_path: {tmp_path}
              coil_distance_mm: 4.0
              device_didt_max: 161e6
            experiment:
              calibration:
                label: M1
                bundle_path: {tmp_path / "cst.trk"}
                measured_rmt_mso: 50.0
                coords: [0.0, 0.0, 0.0]
              target:
                label: Target
                bundle_path: {tmp_path / "target.trk"}
                coords: [0.0, 0.0, 0.0]
                orientation: [0.0, 1.0, 0.0]
                grid:
                  search_radius_mm: 4.0
                  step_size_mm: 4.0
                  cortex_depth_mm: 2.0
            options:
              field_mode: {field_mode}
            """))
    return cfg, out_dir


def _materialize_preflight_inputs(tmp_path: Path) -> None:
    m2m_path = tmp_path / "m2m_sub-TEST"
    m2m_path.mkdir(exist_ok=True)
    (m2m_path / "sub-TEST.msh").write_text("")
    for filename in ("t1.nii.gz", "cst.trk", "target.trk", "MagVenture_C-B60.ccd"):
        (tmp_path / filename).write_text("")


def _write_stmpx_template(tmp_path: Path, contents: Optional[str] = None) -> Path:
    stmpx_path = tmp_path / "session.stmpx"
    stmpx_path.write_text(
        contents
        if contents is not None
        else '<!DOCTYPE stmp>\n<stmp><fmpm dataset="ORIGINAL"><fmp id="P001" /></fmpm></stmp>'
    )
    return stmpx_path


def _write_target_summary(config) -> None:
    output_dir = config.subject.derivatives_path / f"TIDE_{config.target.label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"TIDE_Results_{config.target.label}.txt").write_text(textwrap.dedent("""
            --- Target Estimation (Target) ---
              Target Coords (Cortex): [1.0, 2.0, 3.0]
              Optimized Scalp Position: [4.0, 5.0, 6.0]
              Optimized Matrix: [[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 6.0], [0.0, 0.0, 0.0, 1.0]]
            --- Geometric Analysis ---
            """).strip())


def test_version_bypasses_simnibs(tmp_path):
    """--version exits 0 without needing SimNIBS on PATH."""
    result = _run_cli(["--version"], tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip()


def test_help_bypasses_simnibs(tmp_path):
    """--help exits 0 and lists workflows without needing SimNIBS."""
    result = _run_cli(["--help"], tmp_path)
    assert result.returncode == 0
    assert "estimation" in result.stdout


def test_python_m_tide_help_bypasses_simnibs(tmp_path):
    """``python -m tide --help`` is an interpreter-explicit CLI fallback."""
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)
    env["PYTHONPATH"] = str(SRC_DIR)
    env.pop(_RELAUNCH_MARKER, None)
    result = subprocess.run(
        [sys.executable, "-m", "tide", "--help"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "estimation" in result.stdout


def test_init_config_bypasses_simnibs_and_matches_canonical_template(tmp_path):
    """--init-config works without SimNIBS and writes the canonical template."""
    destination = tmp_path / "generated.yml"
    result = _run_cli(["--init-config", str(destination)], tmp_path)

    assert result.returncode == 0
    assert destination.read_text() == (REPO_ROOT / "config_template.yml").read_text()
    assert "Wrote TIDE configuration template" in result.stdout


def test_init_config_refuses_to_overwrite_existing_file(tmp_path):
    """--init-config must never silently replace a user's configuration."""
    destination = tmp_path / "config.yml"
    destination.write_text("existing: true\n")

    result = _run_cli(["--init-config", str(destination)], tmp_path)

    assert result.returncode == 1
    assert destination.read_text() == "existing: true\n"
    assert "Refusing to overwrite" in result.stderr


@pytest.mark.parametrize("flag", ["-v", "--version", "-h", "--help"])
def test_research_use_statement_in_informational_output(tmp_path, flag):
    """Both informational CLI paths carry the Research-Use-Only statement."""
    from tide.banner import RESEARCH_USE_HEADING, RESEARCH_USE_LINES

    result = _run_cli([flag], tmp_path)
    assert result.returncode == 0
    assert RESEARCH_USE_HEADING in result.stdout
    for line in RESEARCH_USE_LINES:
        assert line in result.stdout


def test_bootstrap_uses_exact_simnibs_python_for_pip(monkeypatch):
    """Bootstrap must use ``simnibs_python -m pip``, never a standalone pip script."""
    import tide.cli as cli

    simnibs_python = Path("/opt/SimNIBS/simnibs_env/bin/python")
    src_dir = Path("/checkout/TIDE/src")
    calls = []
    probe_envs = []

    monkeypatch.setattr(cli, "_locate_simnibs_python", lambda: simnibs_python)
    monkeypatch.setattr(cli, "_verify_simnibs_deps", lambda python: (True, []))
    monkeypatch.setattr(cli, "_source_checkout_src_dir", lambda: src_dir)

    def fake_run(args, check=False):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    def fake_importable(python, env=None):
        probe_envs.append(env)
        return True

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_tide_importable_under", fake_importable)
    monkeypatch.setenv("PYTHONPATH", "/should/not/leak")

    cli.run_bootstrap(editable=True)

    assert calls == [
        [
            str(simnibs_python),
            "-m",
            "pip",
            "install",
            "-e",
            str(src_dir.parent),
        ]
    ]
    assert probe_envs and "PYTHONPATH" not in probe_envs[0]


def test_bootstrap_fails_if_tide_not_importable_after_install(monkeypatch):
    """A successful pip exit is insufficient if the target interpreter cannot import tide."""
    import tide.cli as cli

    simnibs_python = Path("/opt/SimNIBS/simnibs_env/bin/python")
    monkeypatch.setattr(cli, "_locate_simnibs_python", lambda: simnibs_python)
    monkeypatch.setattr(cli, "_verify_simnibs_deps", lambda python: (True, []))
    monkeypatch.setattr(cli, "_source_checkout_src_dir", lambda: Path("/checkout/TIDE/src"))
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda args, check=False: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(cli, "_tide_importable_under", lambda python, env=None: False)

    with pytest.raises(SystemExit) as exc_info:
        cli.run_bootstrap(editable=True)

    assert exc_info.value.code == 1


def test_argument_parser_accepts_stmpx_path(tmp_path):
    from tide.cli import create_argument_parser

    args = create_argument_parser().parse_args(
        ["--config", str(tmp_path / "config.yml"), "--stmpx", str(tmp_path / "session.stmpx")]
    )

    assert args.stmpx == tmp_path / "session.stmpx"


def test_stmpx_is_rejected_for_non_estimation_workflow_before_output(tmp_path, capsys):
    from tide.cli import run_headless

    cfg, out_dir = _write_preflight_config(tmp_path)
    stmpx_path = _write_stmpx_template(tmp_path)
    args = SimpleNamespace(
        config=cfg,
        workflow="grid",
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
        stmpx=stmpx_path,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_headless(args)

    assert exc_info.value.code == 1
    assert "--stmpx" in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.parametrize(
    "contents",
    [None, "<stmp>", "<stmp></stmp>"],
    ids=["missing", "malformed", "missing-fmpm"],
)
def test_stmpx_input_preflight_fails_before_output(tmp_path, capsys, contents):
    from tide.cli import run_headless

    _materialize_preflight_inputs(tmp_path)
    cfg, out_dir = _write_preflight_config(tmp_path)
    stmpx_path = tmp_path / "session.stmpx"
    if contents is not None:
        stmpx_path.write_text(contents)
    args = SimpleNamespace(
        config=cfg,
        workflow="estimation",
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
        stmpx=stmpx_path,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_headless(args)

    assert exc_info.value.code == 1
    assert "STMPX" in capsys.readouterr().err
    assert not out_dir.exists()


def test_headless_estimation_exports_stmpx_after_workflow(tmp_path, monkeypatch):
    from tide.cli import run_headless

    _materialize_preflight_inputs(tmp_path)
    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    raw["options"]["stmpx_dataset_name"] = "20260717-TIDE-SUB_TEST"
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    stmpx_path = _write_stmpx_template(tmp_path)

    estimation = types.ModuleType("tide.workflows.estimation")
    estimation.run_estimation_workflow = lambda config, console_ui: _write_target_summary(config)
    monkeypatch.setitem(sys.modules, "tide.workflows.estimation", estimation)

    args = SimpleNamespace(
        config=cfg,
        workflow="estimation",
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
        stmpx=stmpx_path,
    )

    run_headless(args)

    output_path = tmp_path / "session_updated.stmpx"
    assert output_path.exists()
    assert 'dataset="20260717-TIDE-SUB_TEST"' in output_path.read_text()
    assert 'id="Target_Estimation"' in output_path.read_text()


def test_stmpx_export_failure_is_a_pipeline_failure(tmp_path, monkeypatch, capsys):
    from tide.cli import run_headless

    _materialize_preflight_inputs(tmp_path)
    cfg, _ = _write_preflight_config(tmp_path)
    stmpx_path = _write_stmpx_template(tmp_path)

    def write_invalid_summary(config, console_ui):
        output_dir = config.subject.derivatives_path / f"TIDE_{config.target.label}"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"TIDE_Results_{config.target.label}.txt").write_text("invalid report")

    estimation = types.ModuleType("tide.workflows.estimation")
    estimation.run_estimation_workflow = write_invalid_summary
    monkeypatch.setitem(sys.modules, "tide.workflows.estimation", estimation)
    args = SimpleNamespace(
        config=cfg,
        workflow="estimation",
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
        stmpx=stmpx_path,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_headless(args)

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Target Estimation" in captured.err
    assert "PIPELINE COMPLETE" not in captured.out + captured.err
    assert not (tmp_path / "session_updated.stmpx").exists()


@pytest.mark.parametrize("command", ["--cache-info", "--cache-clear"])
@pytest.mark.parametrize(
    "contents",
    [None, "[unterminated", "[]", "subject: []", "subject:\n  cache_dir: []"],
)
def test_cache_command_rejects_invalid_explicit_config(
    tmp_path,
    monkeypatch,
    capsys,
    command,
    contents,
):
    from tide.cli import run_cache_command

    cache_root = tmp_path / "cache" / "fixed_pose" / "aa" / "entry"
    cache_root.mkdir(parents=True)
    metadata = cache_root / "metadata.json"
    metadata.write_text("{}")
    config = tmp_path / "invalid.yml"
    if contents is not None:
        config.write_text(contents)
    monkeypatch.setenv("TIDE_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(SystemExit) as exc_info:
        run_cache_command([command, "--config", str(config)])

    assert exc_info.value.code == 1
    assert "Could not read cache configuration" in capsys.readouterr().err
    assert metadata.exists()


@pytest.mark.parametrize("command", ["--cache-info", "--cache-clear"])
def test_cache_command_rejects_missing_config_argument(tmp_path, monkeypatch, capsys, command):
    from tide.cli import run_cache_command

    cache_root = tmp_path / "cache" / "fixed_pose" / "aa" / "entry"
    cache_root.mkdir(parents=True)
    metadata = cache_root / "metadata.json"
    metadata.write_text("{}")
    monkeypatch.setenv("TIDE_CACHE_DIR", str(tmp_path / "cache"))

    with pytest.raises(SystemExit) as exc_info:
        run_cache_command([command, "--config"])

    assert exc_info.value.code == 1
    assert "path is missing" in capsys.readouterr().err
    assert metadata.exists()


def test_grid_module_imports_without_fcntl(tmp_path):
    script = textwrap.dedent("""
        import sys
        import types

        sys.modules["fcntl"] = None

        simnibs = types.ModuleType("simnibs")
        simnibs.opt_struct = types.SimpleNamespace(TMSoptimize=object)
        simnibs.run_simnibs = lambda *args, **kwargs: None
        simnibs.sim_struct = types.SimpleNamespace(SESSION=object)
        simnibs.read_msh = lambda *args, **kwargs: None
        sys.modules["simnibs"] = simnibs

        import tide.workflows.grid_search
        """)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(SIMNIBS_PRESENT, reason="requires SimNIBS to be absent")
def test_missing_simnibs_exits_before_config(tmp_path):
    """Workflow run aborts at SimNIBS discovery, before config or output."""
    out_dir = tmp_path / "derivatives_out"
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(textwrap.dedent(f"""
            subject:
              id: sub-TEST
              derivatives_path: {out_dir}
              files:
                t1w: {tmp_path / "t1.nii.gz"}
            coil:
              coil_model: "MagVenture_C-B60.ccd"
            experiment:
              calibration:
                label: M1
                bundle_path: {tmp_path / "cst.trk"}
                coords: [0.0, 0.0, 0.0]
              target:
                label: TGT
                bundle_path: {tmp_path / "tgt.trk"}
                coords: [0.0, 0.0, 0.0]
            """))

    result = _run_cli(["--config", str(cfg), "--workflow", "estimation"], tmp_path)

    assert result.returncode == 1
    assert "'simnibs' command not found" in result.stderr
    # Ordering contract: discovery fails before the workflow touches the config
    # or creates the derivatives tree.
    assert not out_dir.exists()


@pytest.mark.parametrize("workflow", ["estimation", "grid", None])
def test_dose_preflight_rejects_e_parallel_before_output(tmp_path, capsys, workflow):
    from tide.cli import run_headless

    cfg, out_dir = _write_preflight_config(tmp_path, field_mode="e_parallel")
    args = SimpleNamespace(
        config=cfg,
        workflow=workflow,
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_headless(args)

    assert exc_info.value.code == 1
    assert "field_mode" in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.parametrize("weight_key", ["weights_cst", "weights_target"])
def test_weight_preflight_rejects_missing_file_before_output(
    tmp_path,
    capsys,
    weight_key,
):
    from tide.cli import run_headless

    missing = tmp_path / f"missing_{weight_key}.txt"
    kwargs = {weight_key: missing}
    cfg, out_dir = _write_preflight_config(tmp_path, **kwargs)
    args = SimpleNamespace(
        config=cfg,
        workflow="estimation",
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_headless(args)

    assert exc_info.value.code == 1
    assert str(missing) in capsys.readouterr().err
    assert not out_dir.exists()


def test_workflow_exception_exits_nonzero_without_completion(tmp_path, monkeypatch, capsys):
    from tide.cli import run_headless

    _materialize_preflight_inputs(tmp_path)
    cfg, _ = _write_preflight_config(tmp_path)

    estimation = types.ModuleType("tide.workflows.estimation")

    def fail_workflow(*args, **kwargs):
        raise RuntimeError("required stage failed")

    estimation.run_estimation_workflow = fail_workflow
    grid = types.ModuleType("tide.workflows.grid_search")
    grid.run_grid_search_workflow = lambda *args, **kwargs: None
    standard = types.ModuleType("tide.workflows.standard")
    standard.run_standard_simulation = lambda *args, **kwargs: None
    standard.run_standard_optimization = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "tide.workflows.estimation", estimation)
    monkeypatch.setitem(sys.modules, "tide.workflows.grid_search", grid)
    monkeypatch.setitem(sys.modules, "tide.workflows.standard", standard)

    args = SimpleNamespace(
        config=cfg,
        workflow="estimation",
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_headless(args)

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "required stage failed" in captured.err
    assert "PIPELINE COMPLETE" not in captured.out + captured.err


def test_prepare_anatomy_writes_compressed_and_legacy_names(tmp_path: Path) -> None:
    from tide.cli import _prepare_anatomy

    source = tmp_path / "source_T1w.nii.gz"
    source.write_bytes(b"compressed nifti")
    derivatives = tmp_path / "derivatives"
    derivatives.mkdir()
    config = SimpleNamespace(
        subject=SimpleNamespace(
            derivatives_path=derivatives,
            t1w_path=source,
        )
    )

    _prepare_anatomy(config)

    assert (derivatives / "t1w.nii.gz").read_bytes() == source.read_bytes()
    assert (derivatives / "t1w.gz").read_bytes() == source.read_bytes()


def test_prepare_anatomy_preserves_existing_legacy_name(tmp_path: Path) -> None:
    from tide.cli import _prepare_anatomy

    source = tmp_path / "source_T1w.nii.gz"
    source.write_bytes(b"current anatomy")
    derivatives = tmp_path / "derivatives"
    derivatives.mkdir()
    legacy = derivatives / "t1w.gz"
    legacy.write_bytes(b"existing legacy anatomy")
    config = SimpleNamespace(
        subject=SimpleNamespace(
            derivatives_path=derivatives,
            t1w_path=source,
        )
    )

    _prepare_anatomy(config)

    assert legacy.read_bytes() == b"existing legacy anatomy"
    assert (derivatives / "t1w.nii.gz").read_bytes() == source.read_bytes()


def test_custom_coil_file_path_is_used_directly(tmp_path):
    from tide.utils.config import SimNIBSConfig

    cfg, _ = _write_preflight_config(tmp_path)
    custom_coil = tmp_path / "custom.ccd"
    raw = yaml.safe_load(cfg.read_text())
    raw["coil"]["coil_path"] = str(custom_coil)
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = SimNIBSConfig.from_yaml(cfg)

    assert config.coil.coil_path == custom_coil
    assert config.coil.coil_model == custom_coil.name


def test_visualization_singular_key_is_a_compatibility_alias(tmp_path):
    from tide.utils.config import SimNIBSConfig

    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    raw["options"]["generate_visualization"] = False
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = SimNIBSConfig.from_yaml(cfg)

    assert config.options.generate_visualizations is False


def test_stmpx_dataset_name_must_be_a_string(tmp_path):
    from tide.utils.config import SimNIBSConfig

    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    raw["options"]["stmpx_dataset_name"] = 123
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValueError, match="stmpx_dataset_name"):
        SimNIBSConfig.from_yaml(cfg)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("calibration", "stmpx_file", "/tmp/pose.stmpx"),
        ("target", "stmpx_file", "/tmp/pose.stmpx"),
        ("calibration", "didt", 1e6),
    ],
)
def test_unsupported_scientific_config_fields_fail_explicitly(
    tmp_path,
    section,
    key,
    value,
):
    from tide.utils.config import SimNIBSConfig

    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    raw["experiment"][section][key] = value
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValueError, match=key):
        SimNIBSConfig.from_yaml(cfg)


def test_shipped_template_parses(tmp_path, monkeypatch):
    from tide.utils import config as config_module

    monkeypatch.setattr(config_module, "_detect_simnibs_coil_path", lambda: tmp_path)

    config = config_module.SimNIBSConfig.from_yaml(REPO_ROOT / "config_template.yml")

    assert config.subject.id


def test_preflight_rejects_malformed_orientation_before_output(tmp_path, capsys):
    from tide.cli import run_headless

    cfg, out_dir = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    raw["experiment"]["target"]["orientation"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    args = SimpleNamespace(
        config=cfg,
        workflow="estimation",
        verbosity="standard",
        no_console_ui=True,
        no_cache=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_headless(args)

    assert exc_info.value.code == 1
    assert "orientation" in capsys.readouterr().err
    assert not out_dir.exists()


def test_preflight_rejects_nonrigid_orientation_matrix(tmp_path):
    from tide.utils.config import SimNIBSConfig, validate_workflow_config

    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    raw["experiment"]["target"]["orientation"] = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = SimNIBSConfig.from_yaml(cfg)

    with pytest.raises(ValueError, match="orthonormal"):
        validate_workflow_config(config, "estimation")


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("options", "roi_size_mm", 0.0, "roi_size_mm"),
        ("options", "mso_floor_ratio", 1.5, "mso_floor_ratio"),
        ("grid", "step_size_mm", 0.0, "step_size_mm"),
    ],
)
def test_preflight_rejects_invalid_numeric_contracts(
    tmp_path,
    section,
    key,
    value,
    message,
):
    from tide.utils.config import SimNIBSConfig, validate_workflow_config

    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    if section == "grid":
        raw["experiment"]["target"]["grid"][key] = value
    else:
        raw[section][key] = value
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = SimNIBSConfig.from_yaml(cfg)

    with pytest.raises(ValueError, match=message):
        validate_workflow_config(config, "grid" if section == "grid" else "estimation")


def test_preflight_rejects_ambiguous_mesh_selection(tmp_path):
    from tide.utils.config import SimNIBSConfig, validate_workflow_config

    _materialize_preflight_inputs(tmp_path)
    (tmp_path / "m2m_sub-TEST" / "second.msh").write_text("")
    cfg, _ = _write_preflight_config(tmp_path)
    config = SimNIBSConfig.from_yaml(cfg)

    with pytest.raises(ValueError, match="multiple .msh"):
        validate_workflow_config(config, "estimation")


def test_preflight_rejects_unsafe_output_label(tmp_path):
    from tide.utils.config import SimNIBSConfig, validate_workflow_config

    _materialize_preflight_inputs(tmp_path)
    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    raw["experiment"]["target"]["label"] = "../outside"
    cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = SimNIBSConfig.from_yaml(cfg)

    with pytest.raises(ValueError, match="label"):
        validate_workflow_config(config, "estimation")


@pytest.mark.parametrize("missing_input", ["target_bundle", "surface"])
def test_preflight_rejects_missing_scientific_inputs(tmp_path, missing_input):
    from tide.utils.config import SimNIBSConfig, validate_workflow_config

    _materialize_preflight_inputs(tmp_path)
    cfg, _ = _write_preflight_config(tmp_path)
    raw = yaml.safe_load(cfg.read_text())
    if missing_input == "target_bundle":
        (tmp_path / "target.trk").unlink()
    else:
        raw["subject"]["files"]["surface"] = str(tmp_path / "missing.white")
        cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = SimNIBSConfig.from_yaml(cfg)

    with pytest.raises(FileNotFoundError):
        validate_workflow_config(config, "estimation")


class TestSimnibsDescriptor:
    """Environment-independent unit tests for the SimNIBS installation descriptor."""

    def _descriptor(self):
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        from tide.utils import simnibs_env

        return simnibs_env

    def test_python_candidates_posix_order(self, monkeypatch):
        simnibs_env = self._descriptor()
        monkeypatch.setattr(simnibs_env.sys, "platform", "linux")
        root = Path("/opt/SimNIBS")
        assert simnibs_env.python_candidates(root) == [
            root / "simnibs_env" / "bin" / "python3",
            root / "simnibs_env" / "bin" / "python",
            root / "bin" / "python3",
            root / "bin" / "python",
        ]

    def test_python_candidates_windows_order(self, monkeypatch):
        simnibs_env = self._descriptor()
        monkeypatch.setattr(simnibs_env.sys, "platform", "win32")
        root = Path("C:/SimNIBS")
        candidates = simnibs_env.python_candidates(root)
        assert candidates[0] == root / "simnibs_env" / "Scripts" / "python.exe"
        assert all(c.suffix == ".exe" for c in candidates)

    def test_select_python_returns_first_executable(self, tmp_path, monkeypatch):
        simnibs_env = self._descriptor()
        monkeypatch.setattr(simnibs_env.sys, "platform", "linux")
        missing = tmp_path / "missing"
        present = tmp_path / "python3"
        present.write_text("")
        present.chmod(0o755)
        assert simnibs_env.select_python([missing, present]) == present

    def test_select_python_none_when_absent(self, tmp_path):
        simnibs_env = self._descriptor()
        assert simnibs_env.select_python([tmp_path / "nope"]) is None

    def test_resolvers_none_without_launcher(self, monkeypatch):
        simnibs_env = self._descriptor()
        monkeypatch.setattr(simnibs_env.shutil, "which", lambda name: None)
        assert simnibs_env.simnibs_root() is None
        assert simnibs_env.find_get_fields_at_coordinates() is None
        assert simnibs_env.find_coil_models_dir() is None

    def test_coil_models_dir_resolves_with_fallback(self, tmp_path, monkeypatch):
        simnibs_env = self._descriptor()
        launcher = tmp_path / "bin" / "simnibs"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("")
        site = tmp_path / "simnibs_env" / "lib" / "python3.11" / "site-packages"
        coil_parent = site / "simnibs" / "resources" / "coil_models"
        coil_parent.mkdir(parents=True)
        monkeypatch.setattr(
            simnibs_env.shutil,
            "which",
            lambda name: str(launcher) if name == "simnibs" else None,
        )
        # Specific Drakaki subdir absent -> falls back to parent coil_models dir.
        result = simnibs_env.find_coil_models_dir()
        assert result is not None and result.name == "coil_models"
        (coil_parent / "Drakaki_BrainStim_2022").mkdir()
        result = simnibs_env.find_coil_models_dir()
        assert result is not None and result.name == "Drakaki_BrainStim_2022"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
