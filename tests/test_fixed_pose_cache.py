import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tide.utils.artifacts import (
    CACHE_DISABLE_TOKENS,
    cache_total_size,
    clear_cache,
    enforce_cache_limit,
    entry_size,
    fixed_pose_cache_enabled,
    fixed_pose_cache_key,
    iter_cache_entries,
    resolve_cache_max_bytes,
    restore_fixed_pose_artifacts,
    store_fixed_pose_artifacts,
)

MATRIX = [
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, 20.0],
    [0.0, 0.0, 1.0, 30.0],
    [0.0, 0.0, 0.0, 1.0],
]


def test_cache_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDE_FIXED_POSE_CACHE", "false")

    assert fixed_pose_cache_enabled() is False


def test_disable_tokens_shared_by_env_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "no" in CACHE_DISABLE_TOKENS
    for token in sorted(CACHE_DISABLE_TOKENS):
        monkeypatch.setenv("TIDE_FIXED_POSE_CACHE", token)
        assert fixed_pose_cache_enabled() is False
    monkeypatch.setenv("TIDE_FIXED_POSE_CACHE", "1")
    assert fixed_pose_cache_enabled() is True


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    mesh_path = tmp_path / "m2m_subject"
    mesh_path.mkdir()
    (mesh_path / "subject.msh").write_bytes(b"head-mesh")
    coil_path = tmp_path / "coil.ccd"
    coil_path.write_bytes(b"coil-model")
    return mesh_path, coil_path


def _cache_key(
    mesh_path: Path,
    coil_path: Path,
    matrix: Optional[list[list[float]]] = None,
) -> str:
    return fixed_pose_cache_key(
        mesh_path=mesh_path,
        coil_path=coil_path,
        orientation=matrix if matrix is not None else MATRIX,
        didt=1e6,
        distance_mm=4.0,
        fields="E",
        runtime_signature={"simnibs": "4.5.0", "numpy": "1.26.4"},
    )


def test_cache_key_tracks_every_numerical_input(tmp_path: Path) -> None:
    mesh_path, coil_path = _write_inputs(tmp_path)
    baseline = _cache_key(mesh_path, coil_path)

    changed_matrix = [row.copy() for row in MATRIX]
    changed_matrix[0][3] = 10.5

    assert _cache_key(mesh_path, coil_path, changed_matrix) != baseline
    assert (
        fixed_pose_cache_key(
            mesh_path=mesh_path,
            coil_path=coil_path,
            orientation=MATRIX,
            didt=2e6,
            distance_mm=4.0,
            fields="E",
            runtime_signature={"simnibs": "4.5.0", "numpy": "1.26.4"},
        )
        != baseline
    )
    assert (
        fixed_pose_cache_key(
            mesh_path=mesh_path,
            coil_path=coil_path,
            orientation=MATRIX,
            didt=1e6,
            distance_mm=5.0,
            fields="E",
            runtime_signature={"simnibs": "4.5.0", "numpy": "1.26.4"},
        )
        != baseline
    )
    assert (
        fixed_pose_cache_key(
            mesh_path=mesh_path,
            coil_path=coil_path,
            orientation=MATRIX,
            didt=1e6,
            distance_mm=4.0,
            fields="veEjJ",
            runtime_signature={"simnibs": "4.5.0", "numpy": "1.26.4"},
        )
        != baseline
    )
    assert (
        fixed_pose_cache_key(
            mesh_path=mesh_path,
            coil_path=coil_path,
            orientation=MATRIX,
            didt=1e6,
            distance_mm=4.0,
            fields="E",
            runtime_signature={"simnibs": "4.6.0", "numpy": "1.26.4"},
        )
        != baseline
    )

    coil_path.write_bytes(b"changed-coil-model")
    assert _cache_key(mesh_path, coil_path) != baseline
    coil_path.write_bytes(b"coil-model")

    (mesh_path / "subject.msh").write_bytes(b"changed-head-mesh")
    assert _cache_key(mesh_path, coil_path) != baseline


def test_cache_round_trip_preserves_all_artifact_bytes(tmp_path: Path) -> None:
    mesh_path, coil_path = _write_inputs(tmp_path)
    key = _cache_key(mesh_path, coil_path)
    cache_root = tmp_path / "cache"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    artifacts = [
        source_dir / "subject_scalar.msh",
        source_dir / "subject_scalar.msh.opt",
        source_dir / "subject_coil_pos.geo",
    ]
    contents = [b"mesh-bytes", b"option-bytes", b"geometry-bytes"]
    for path, content in zip(artifacts, contents):
        path.write_bytes(content)

    stored = store_fixed_pose_artifacts(key, artifacts, cache_root=cache_root)
    output_dir = tmp_path / "restored"
    output_dir.mkdir()
    restored = restore_fixed_pose_artifacts(key, output_dir, cache_root=cache_root)

    assert stored is True
    assert [path.name for path in restored] == [path.name for path in artifacts]
    assert [path.read_bytes() for path in restored] == contents


def test_corrupt_cache_entry_is_a_miss_and_does_not_touch_output(tmp_path: Path) -> None:
    mesh_path, coil_path = _write_inputs(tmp_path)
    key = _cache_key(mesh_path, coil_path)
    cache_root = tmp_path / "cache"
    source = tmp_path / "subject_scalar.msh"
    source.write_bytes(b"valid")
    store_fixed_pose_artifacts(key, [source], cache_root=cache_root)

    metadata_path = cache_root / key[:2] / key / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cached_name = metadata["artifacts"][0]["cached_name"]
    (metadata_path.parent / cached_name).write_bytes(b"corrupt")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / source.name
    existing.write_bytes(b"existing")

    restored = restore_fixed_pose_artifacts(key, output_dir, cache_root=cache_root)

    assert restored == []
    assert existing.read_bytes() == b"existing"


class TestFixedPoseSimulationCache:
    @staticmethod
    def _load_interface(
        monkeypatch: pytest.MonkeyPatch,
        run_simnibs: Callable[[object], None],
    ) -> Any:
        class Position:
            pass

        class TmsList:
            def add_position(self) -> Position:
                return Position()

        class Session:
            time_str = "20260712-120000"

            def add_tmslist(self) -> TmsList:
                return TmsList()

        def save_matlab_sim_struct(session: object, path: str) -> None:
            Path(path).write_bytes(b"current-session")

        fake_simnibs = SimpleNamespace(
            __version__="4.5.0",
            opt_struct=SimpleNamespace(TMSoptimize=object),
            run_simnibs=run_simnibs,
            sim_struct=SimpleNamespace(
                SESSION=Session,
                save_matlab_sim_struct=save_matlab_sim_struct,
            ),
        )
        monkeypatch.setitem(sys.modules, "simnibs", fake_simnibs)
        sys.modules.pop("tide.interfaces.simnibs_interface", None)
        module = importlib.import_module("tide.interfaces.simnibs_interface")
        monkeypatch.setattr(module, "run_simnibs", run_simnibs)
        return module

    def test_second_output_directory_uses_cache_without_simnibs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        mesh_path, coil_path = _write_inputs(tmp_path)
        cache_root = tmp_path / "cache"
        monkeypatch.setenv("TIDE_CACHE_DIR", str(cache_root))
        first_output = tmp_path / "first"
        second_output = tmp_path / "second"
        first_output.mkdir()
        second_output.mkdir()
        calls = []

        def run_first(session: object) -> None:
            calls.append(session)
            (first_output / "subject_scalar.msh").write_bytes(b"exact-mesh")
            (first_output / "subject_scalar.msh.opt").write_bytes(b"exact-options")
            (first_output / "subject_coil_pos.geo").write_bytes(b"exact-geometry")
            (first_output / "fields_summary.txt").write_bytes(b"exact-summary")

        module = self._load_interface(monkeypatch, run_first)
        first_result = module.SimNIBSInterface.run_simulation(
            mesh_path=mesh_path,
            output_dir=first_output,
            coil_path=coil_path,
            didt=1e6,
            orientation=MATRIX,
        )

        def fail_run(session: object) -> None:
            pytest.fail("SimNIBS must not run on a fixed-pose cache hit")

        monkeypatch.setattr(module, "run_simnibs", fail_run)
        second_result = module.SimNIBSInterface.run_simulation(
            mesh_path=mesh_path,
            output_dir=second_output,
            coil_path=coil_path,
            didt=1e6,
            orientation=MATRIX,
        )

        assert len(calls) == 1
        assert first_result.read_bytes() == second_result.read_bytes() == b"exact-mesh"
        assert (second_output / "subject_scalar.msh.opt").read_bytes() == b"exact-options"
        assert (second_output / "subject_coil_pos.geo").read_bytes() == b"exact-geometry"
        assert (second_output / "fields_summary.txt").read_bytes() == b"exact-summary"
        assert len(list(second_output.glob("simnibs_simulation_*.mat"))) == 1
        assert len(list(second_output.glob("simnibs_simulation_*.log"))) == 1
        first_manifest = json.loads(
            (first_output / ".tide_run_manifest.json").read_text(encoding="utf-8")
        )
        second_manifest = json.loads(
            (second_output / ".tide_run_manifest.json").read_text(encoding="utf-8")
        )
        assert first_manifest["artifacts"]["simulation_mesh"]["cache"]["status"] == "miss"
        assert second_manifest["artifacts"]["simulation_mesh"]["cache"]["status"] == "hit"

        sys.modules.pop("tide.interfaces.simnibs_interface", None)


def _store_entry(
    tmp_path: Path,
    cache_root: Path,
    mesh_path: Path,
    coil_path: Path,
    tag: int,
    payload: bytes,
) -> Path:
    """Store one synthetic cache entry with a distinct key; return its entry dir."""
    matrix = [row.copy() for row in MATRIX]
    matrix[0][3] = float(tag)
    key = _cache_key(mesh_path, coil_path, matrix)
    source = tmp_path / f"art{tag}.msh"
    source.write_bytes(payload)
    assert store_fixed_pose_artifacts(key, [source], cache_root=cache_root) is True
    return cache_root / key[:2] / key


def test_resolve_cache_max_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIDE_CACHE_MAX_GB", raising=False)
    assert resolve_cache_max_bytes(None) is None
    assert resolve_cache_max_bytes(0) is None
    assert resolve_cache_max_bytes(2) == 2 * 1024**3
    monkeypatch.setenv("TIDE_CACHE_MAX_GB", "3")
    assert resolve_cache_max_bytes(None) == 3 * 1024**3
    assert resolve_cache_max_bytes(1) == 1024**3  # config wins over env


def test_enforce_cache_limit_evicts_oldest(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    mesh_path, coil_path = _write_inputs(tmp_path)
    old = _store_entry(tmp_path, cache_root, mesh_path, coil_path, 1, b"a" * 4000)
    mid = _store_entry(tmp_path, cache_root, mesh_path, coil_path, 2, b"b" * 4000)
    new = _store_entry(tmp_path, cache_root, mesh_path, coil_path, 3, b"c" * 4000)

    import os

    for offset, entry in ((300, old), (200, mid), (100, new)):
        stamp = 1_000_000 - offset
        os.utime(entry, (stamp, stamp))

    max_bytes = entry_size(mid) + entry_size(new)
    freed_old = entry_size(old)
    evicted, freed = enforce_cache_limit(cache_root, max_bytes)

    assert evicted == 1
    assert freed == freed_old
    assert not old.exists()
    assert mid.exists() and new.exists()
    assert cache_total_size(cache_root) <= max_bytes


def test_touch_on_hit_spares_recently_used(tmp_path: Path) -> None:
    import os

    cache_root = tmp_path / "cache"
    mesh_path, coil_path = _write_inputs(tmp_path)
    older = _store_entry(tmp_path, cache_root, mesh_path, coil_path, 1, b"a" * 4000)
    newer = _store_entry(tmp_path, cache_root, mesh_path, coil_path, 2, b"b" * 4000)

    # Make `older` the more-recently-stored entry, then hit `newer` to bump it.
    os.utime(older, (2_000_000, 2_000_000))
    os.utime(newer, (1_000_000, 1_000_000))

    key = newer.name
    output_dir = tmp_path / "restored"
    output_dir.mkdir()
    restored = restore_fixed_pose_artifacts(key, output_dir, cache_root=cache_root)
    assert restored  # cache hit bumped `newer` mtime

    max_bytes = entry_size(newer)  # room for one entry only
    evicted, _ = enforce_cache_limit(cache_root, max_bytes)

    assert evicted == 1
    assert newer.exists()  # spared because it was just used
    assert not older.exists()


def test_clear_cache_empties_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    mesh_path, coil_path = _write_inputs(tmp_path)
    _store_entry(tmp_path, cache_root, mesh_path, coil_path, 1, b"a" * 100)
    _store_entry(tmp_path, cache_root, mesh_path, coil_path, 2, b"b" * 100)
    assert len(iter_cache_entries(cache_root)) == 2

    removed, freed = clear_cache(cache_root)

    assert removed == 2
    assert freed > 0
    assert iter_cache_entries(cache_root) == []
