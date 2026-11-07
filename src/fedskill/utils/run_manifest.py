"""Portable identities and atomic state files for corrected protocol reruns."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


PROTOCOL_VERSION = "corrected-v1"
TAU2_REVISION = "01e812d1d1c4df6d8d2299bf175e482527112244"


def public_endpoint(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def fingerprint(data: Any) -> str:
    encoded = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_identity(skills) -> str:
    return fingerprint(sorted(skill.id for skill in skills))


def atomic_write_json(path: str | Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind_state(path: str | Path, identity: Any) -> dict[str, Any]:
    """Refuse reuse of state from a different input/protocol."""
    path = Path(path)
    expected = fingerprint(identity)
    if path.exists():
        state = read_json(path)
        if state.get("fingerprint") != expected:
            raise ValueError(f"Input/configuration mismatch in {path}; use a new output directory.")
        return state
    state = {"protocol_version": PROTOCOL_VERSION, "fingerprint": expected, "identity": identity}
    atomic_write_json(path, state)
    return state


def prompt_digest(root: str | Path) -> str:
    root = Path(root)
    paths = [
        *sorted((root / "src/fedskill/extractor").glob("*.py")),
        root / "src/fedskill/server/evolver.py",
        *sorted((root / "src/fedskill/clustering").glob("*.py")),
        root / "integrations/tau2/fedskill_tau2_integration.patch",
    ]
    return fingerprint({str(p.relative_to(root)): file_digest(p) for p in paths if p.exists()})


def code_identity(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, env=env, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, env=env, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    files = {}
    for directory in ("src", "scripts", "audits", "configs", "integrations", "prompts"):
        for path in sorted((root / directory).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".yaml", ".json", ".patch"}
            ):
                files[str(path.relative_to(root))] = file_digest(path)
    for name in ("pyproject.toml", "requirements-paper.txt"):
        if (root / name).exists():
            files[name] = file_digest(root / name)
    return {"revision": revision, "dirty": dirty, "source_digest": fingerprint(files)}
