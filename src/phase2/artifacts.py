from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class Phase1Snapshot:
    project_root: Path
    snapshot_id: str
    topic_name: str
    latest_manifest_path: Path
    run_summary_path: Path
    run_summary: dict[str, Any]
    phase1_summary_path: Path
    phase1_summary: dict[str, Any]
    output_root: Path
    chunks_path: Path
    embeddings_dir: Path
    chroma_dir: Path
    chroma_collection_name: str


def build_phase1_snapshot(project_root: Path | str) -> Phase1Snapshot:
    project_root = Path(project_root).resolve()
    latest_manifest_path = project_root / "data" / "manifests" / "latest.json"
    latest_manifest = _read_json(latest_manifest_path)

    run_summary_path = Path(latest_manifest["latest_run_summary_path"])
    run_summary = _read_json(run_summary_path)

    output_root = project_root / "output"
    phase1_summary_path = output_root / "phase1_summary.json"
    phase1_summary = _read_json(phase1_summary_path)

    return Phase1Snapshot(
        project_root=project_root,
        snapshot_id=latest_manifest["latest_snapshot_id"],
        topic_name=latest_manifest["topic"],
        latest_manifest_path=latest_manifest_path,
        run_summary_path=run_summary_path,
        run_summary=run_summary,
        phase1_summary_path=phase1_summary_path,
        phase1_summary=phase1_summary,
        output_root=output_root,
        chunks_path=Path(run_summary["chunks_path"]),
        embeddings_dir=Path(run_summary["embeddings_dir"]),
        chroma_dir=Path(run_summary["chroma_dir"]),
        chroma_collection_name=run_summary["chroma_collection_name"],
    )


def load_chunks_df(snapshot: Phase1Snapshot) -> pd.DataFrame:
    rows = _read_jsonl(snapshot.chunks_path)
    frame = pd.DataFrame(rows)
    if "section_path" in frame.columns:
        frame["section_path_text"] = frame["section_path"].apply(
            lambda values: " > ".join(values) if isinstance(values, list) else ""
        )
    return frame


def load_embeddings_metadata_df(snapshot: Phase1Snapshot) -> pd.DataFrame:
    metadata_path = snapshot.embeddings_dir / "metadata.jsonl"
    return pd.DataFrame(_read_jsonl(metadata_path))


def load_embeddings_array(snapshot: Phase1Snapshot) -> np.ndarray:
    return np.load(snapshot.embeddings_dir / "embeddings.npy")


def open_chroma_collection(snapshot: Phase1Snapshot):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(snapshot.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(snapshot.chroma_collection_name)


def summarize_phase1_snapshot(snapshot: Phase1Snapshot) -> dict[str, Any]:
    summary = {
        "snapshot_id": snapshot.snapshot_id,
        "topic_name": snapshot.topic_name,
        "num_sources": snapshot.phase1_summary["num_sources"],
        "num_fetched_items": snapshot.phase1_summary["num_fetched_items"],
        "num_docs": snapshot.phase1_summary["num_docs"],
        "num_chunks": snapshot.phase1_summary["num_chunks"],
        "embedding_vectors": snapshot.phase1_summary["embedding_vectors"],
        "embedding_dimension": snapshot.phase1_summary["embedding_dimension"],
        "chroma_collection_name": snapshot.phase1_summary["chroma_collection_name"],
        "chunks_path": str(snapshot.chunks_path),
        "embeddings_dir": str(snapshot.embeddings_dir),
        "chroma_dir": str(snapshot.chroma_dir),
    }
    return summary
