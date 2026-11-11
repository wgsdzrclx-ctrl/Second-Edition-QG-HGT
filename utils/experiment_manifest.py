"""Content fingerprints for isolated, reviewable input-representation runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_manifest(root: Path) -> dict[str, Any]:
    paths = {root / "main.py", root / "train.py", root / "requirements.txt"}
    for directory in ("data", "models", "eval", "utils", "scripts"):
        paths.update((root / directory).glob("*.py"))
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(root), *arguments],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {
        "files": files, "sha256": fingerprint(files),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain"),
    }


def assert_matching_manifest(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    expected_fingerprint = expected.get("experiment_fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint or existing.get("experiment_fingerprint") != expected_fingerprint:
        raise ValueError("Configuration, code, environment, selected data, or scaler changed; choose a new output directory.")
