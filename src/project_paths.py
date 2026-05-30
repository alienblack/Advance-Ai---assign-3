from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_NAMES = ("a1974524_a3", "a1974524_a3 2")
DEFAULT_DRIVE_PARENT = Path("/content/drive/MyDrive/Advance-AI-Assign-3")
PROJECT_ROOT_ENV = "A3_PROJECT_ROOT"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _looks_like_project_root(path: Path) -> bool:
    return (path / "src").exists()


def candidate_project_roots(start: Path | None = None) -> list[Path]:
    base = (start or Path.cwd()).resolve()
    candidates: list[Path] = []

    env_root = os.environ.get(PROJECT_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            base,
            base.parent,
            base / "Advance-AI-Assign-3" / "a1974524_a3",
            base.parent / "Advance-AI-Assign-3" / "a1974524_a3",
        ]
    )

    if base.name == "notebooks":
        candidates.append(base.parent)

    for name in DEFAULT_PROJECT_NAMES:
        candidates.append(base / name)
        candidates.append(base.parent / name)

    if in_colab():
        for name in DEFAULT_PROJECT_NAMES:
            candidates.append(DEFAULT_DRIVE_PARENT / name)

    return _unique_paths(candidates)


def resolve_project_root(start: Path | None = None) -> Path:
    for candidate in candidate_project_roots(start=start):
        if _looks_like_project_root(candidate):
            return candidate

    env_root = os.environ.get(PROJECT_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()

    return (start or Path.cwd()).resolve()


@dataclass(frozen=True)
class NotebookPaths:
    project_root: Path
    output_root: Path
    figures_dir: Path
    tables_dir: Path
    phase2_dir: Path
    phase2_tables_dir: Path


def build_notebook_paths(project_root: Path | str) -> NotebookPaths:
    project_root = Path(project_root).resolve()
    output_root = project_root / "output"
    phase2_dir = output_root / "phase2"
    return NotebookPaths(
        project_root=project_root,
        output_root=output_root,
        figures_dir=output_root / "figures",
        tables_dir=output_root / "tables",
        phase2_dir=phase2_dir,
        phase2_tables_dir=phase2_dir / "tables",
    )
