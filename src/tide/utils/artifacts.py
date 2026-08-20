import hashlib
import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ArtifactState = Tuple[int, int, int, Optional[str]]
FIXED_POSE_CACHE_SCHEMA = 1

# Tokens that disable the fixed-pose cache, whether given via the
# TIDE_FIXED_POSE_CACHE env var or a `subject.cache_dir` config field.
CACHE_DISABLE_TOKENS = frozenset({"0", "false", "no", "off"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _sha256_for_state(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    return _sha256(Path(path))


def _input_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_for_state(
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _copy_with_sha256(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as source_obj, destination.open("wb") as destination_obj:
        for chunk in iter(lambda: source_obj.read(1024 * 1024), b""):
            destination_obj.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: Dict[str, object]) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def fixed_pose_cache_enabled() -> bool:
    value = os.environ.get("TIDE_FIXED_POSE_CACHE", "1").strip().lower()
    return value not in CACHE_DISABLE_TOKENS


def fixed_pose_cache_root() -> Path:
    configured = os.environ.get("TIDE_CACHE_DIR")
    if configured:
        base = Path(configured).expanduser()
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tide"
    return base / "fixed_pose"


def resolve_cache_max_bytes(config_gb: Optional[float] = None) -> Optional[int]:
    """Return the fixed-pose cache size cap in bytes, or None (unlimited).

    Config value wins over the ``TIDE_CACHE_MAX_GB`` env fallback. ``0``, a
    negative value, or an unparseable value all mean unlimited. GB is binary
    GiB (``1024**3``).
    """
    value = config_gb
    if value is None:
        env_value = os.environ.get("TIDE_CACHE_MAX_GB")
        if env_value is not None:
            try:
                value = float(env_value)
            except ValueError:
                value = None
    if value is None or value <= 0:
        return None
    return int(value * 1024**3)


def iter_cache_entries(root: Path) -> List[Path]:
    """Return fixed-pose cache entry dirs (``root/*/*`` with a metadata.json)."""
    if not root.is_dir():
        return []
    return [entry for entry in root.glob("*/*") if (entry / "metadata.json").is_file()]


def entry_size(entry: Path) -> int:
    """Return the total size in bytes of all files in a cache entry dir."""
    total = 0
    for path in entry.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def cache_total_size(root: Path) -> int:
    """Return the total size in bytes of all fixed-pose cache entries."""
    return sum(entry_size(entry) for entry in iter_cache_entries(root))


def enforce_cache_limit(root: Path, max_bytes: int) -> Tuple[int, int]:
    """Evict least-recently-used entries until the store is under ``max_bytes``.

    Recency is the entry dir mtime (bumped on every cache hit). Returns
    ``(evicted_count, freed_bytes)``. Best-effort: an entry removed concurrently
    is treated as already gone, not an error.
    """
    entries = iter_cache_entries(root)
    sized = []
    for entry in entries:
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        sized.append((mtime, entry_size(entry), entry))
    total = sum(size for _, size, _ in sized)
    if total <= max_bytes:
        return 0, 0

    sized.sort(key=lambda item: item[0])  # oldest first (LRU)
    evicted = 0
    freed = 0
    for _, size, entry in sized:
        if total <= max_bytes:
            break
        try:
            shutil.rmtree(entry)
        except OSError:
            continue
        evicted += 1
        freed += size
        total -= size
    return evicted, freed


def clear_cache(root: Path) -> Tuple[int, int]:
    """Remove all fixed-pose cache entries. Returns ``(removed_count, freed_bytes)``."""
    removed = 0
    freed = 0
    for entry in iter_cache_entries(root):
        size = entry_size(entry)
        try:
            shutil.rmtree(entry)
        except OSError:
            continue
        removed += 1
        freed += size
    return removed, freed


def fixed_pose_cache_key(
    mesh_path: Path,
    coil_path: Path,
    orientation: Sequence[Sequence[float]],
    didt: float,
    distance_mm: float,
    fields: str,
    runtime_signature: Dict[str, str],
) -> str:
    matrix = [list(row) for row in orientation]
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Fixed-pose cache requires a 4x4 orientation matrix")

    if mesh_path.is_file():
        mesh_files = [mesh_path]
    elif mesh_path.is_dir():
        mesh_files = sorted(
            (path for path in mesh_path.glob("*.msh") if path.is_file()),
            key=lambda path: path.name,
        )
    else:
        raise FileNotFoundError(f"Head mesh input not found: {mesh_path}")
    if not mesh_files:
        raise FileNotFoundError(f"No head mesh found in: {mesh_path}")
    if not coil_path.is_file():
        raise FileNotFoundError(f"Coil model not found: {coil_path}")

    payload = {
        "schema_version": FIXED_POSE_CACHE_SCHEMA,
        "head_meshes": [{"name": path.name, "sha256": _input_sha256(path)} for path in mesh_files],
        "coil": {"name": coil_path.name, "sha256": _input_sha256(coil_path)},
        "orientation": [[float(value).hex() for value in row] for row in matrix],
        "didt": float(didt).hex(),
        "distance_mm": float(distance_mm).hex(),
        "fields": fields,
        "anisotropy_type": "scalar",
        "solver_options": None,
        "runtime": dict(sorted(runtime_signature.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fixed_pose_entry(cache_key: str, cache_root: Optional[Path]) -> Path:
    root = cache_root if cache_root is not None else fixed_pose_cache_root()
    return root / cache_key[:2] / cache_key


def store_fixed_pose_artifacts(
    cache_key: str,
    artifacts: Iterable[Path],
    cache_root: Optional[Path] = None,
) -> bool:
    source_paths = list(artifacts)
    if not source_paths:
        return False
    output_names = [path.name for path in source_paths]
    if len(set(output_names)) != len(output_names):
        raise ValueError("Fixed-pose cache artifact names must be unique")

    entry_dir = _fixed_pose_entry(cache_key, cache_root)
    entry_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    for index, source in enumerate(source_paths):
        if not source.is_file():
            raise FileNotFoundError(f"Simulation artifact not found: {source}")
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=entry_dir,
            prefix=".artifact.",
            suffix=".tmp",
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            digest = _copy_with_sha256(source, temporary_path)
            cached_name = f"{index:02d}-{digest}"
            temporary_path.replace(entry_dir / cached_name)
        finally:
            temporary_path.unlink(missing_ok=True)
        records.append(
            {
                "output_name": source.name,
                "cached_name": cached_name,
                "sha256": digest,
                "size": source.stat().st_size,
            }
        )

    _atomic_json_write(
        entry_dir / "metadata.json",
        {
            "schema_version": FIXED_POSE_CACHE_SCHEMA,
            "cache_key": cache_key,
            "artifacts": records,
        },
    )
    return True


def restore_fixed_pose_artifacts(
    cache_key: str,
    output_dir: Path,
    cache_root: Optional[Path] = None,
) -> List[Path]:
    entry_dir = _fixed_pose_entry(cache_key, cache_root)
    metadata_path = entry_dir / "metadata.json"
    if not metadata_path.is_file():
        return []

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata["schema_version"] != FIXED_POSE_CACHE_SCHEMA
            or metadata["cache_key"] != cache_key
        ):
            return []
        records = metadata["artifacts"]
        if not isinstance(records, list) or not records:
            return []
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    pending: List[Tuple[Path, Path]] = []
    try:
        for record in records:
            output_name = record["output_name"]
            cached_name = record["cached_name"]
            expected_digest = record["sha256"]
            expected_size = record["size"]
            if (
                not isinstance(output_name, str)
                or Path(output_name).name != output_name
                or not isinstance(cached_name, str)
                or Path(cached_name).name != cached_name
                or not isinstance(expected_digest, str)
                or not isinstance(expected_size, int)
            ):
                return []

            cached_path = entry_dir / cached_name
            if not cached_path.is_file() or cached_path.stat().st_size != expected_size:
                return []
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=output_dir,
                prefix=f".{output_name}.",
                suffix=".tmp",
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)
            digest = _copy_with_sha256(cached_path, temporary_path)
            pending.append((temporary_path, output_dir / output_name))
            if digest != expected_digest or temporary_path.stat().st_size != expected_size:
                return []

        for temporary_path, destination in pending:
            temporary_path.replace(destination)
        # Bump entry mtime so LRU eviction treats a cache hit as recent use.
        try:
            os.utime(entry_dir, None)
        except OSError:
            pass
        return [destination for _, destination in pending]
    except (KeyError, TypeError):
        return []
    finally:
        for temporary_path, _ in pending:
            temporary_path.unlink(missing_ok=True)


def capture_artifacts(
    paths: Iterable[Path], hash_contents: bool = False
) -> Dict[Path, ArtifactState]:
    states = {}
    for path in sorted(set(paths), key=str):
        if not path.is_file():
            continue
        stat = path.stat()
        digest = _sha256(path) if hash_contents else None
        states[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, digest)
    return states


def fresh_artifacts(
    before: Dict[Path, ArtifactState],
    paths: Iterable[Path],
    hash_contents: bool = False,
) -> List[Path]:
    after = capture_artifacts(paths, hash_contents=hash_contents)
    return sorted((path for path, state in after.items() if before.get(path) != state), key=str)


def record_artifact(
    output_dir: Path,
    key: str,
    selected: Path,
    candidates: Iterable[Path],
    details: Optional[Dict[str, object]] = None,
) -> None:
    manifest_path = output_dir / ".tide_run_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = {"schema_version": 1, "artifacts": {}}

    artifact: Dict[str, object] = {
        "selected": str(selected.relative_to(output_dir)),
        "candidates": [str(path.relative_to(output_dir)) for path in sorted(candidates, key=str)],
    }
    if details:
        artifact.update(details)
    payload["artifacts"][key] = artifact

    temporary_path = output_dir / ".tide_run_manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
