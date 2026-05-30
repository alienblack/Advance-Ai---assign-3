from __future__ import annotations

import json
import re
from typing import Iterable

import pandas as pd


RAGAS_METRIC_COLUMNS = [
    "context_relevance",
    "answer_relevance",
    "faithfulness",
    "context_precision",
    "context_recall",
]

BERTSCORE_COLUMNS = [
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
]


def strip_inline_citations(text: str) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_metric_placeholder_frame(query_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_row in query_df.to_dict(orient="records"):
        for variant_row in variant_df.to_dict(orient="records"):
            rows.append(
                {
                    "query_id": query_row["query_id"],
                    "variant": variant_row["variant"],
                    **{column: None for column in RAGAS_METRIC_COLUMNS},
                    **{column: None for column in BERTSCORE_COLUMNS},
                }
            )
    return pd.DataFrame(rows)


def build_evaluation_cases(
    generation_results_df: pd.DataFrame,
    evaluation_queries_df: pd.DataFrame,
    retrieval_results_df: pd.DataFrame,
) -> pd.DataFrame:
    expected_columns = [
        "query_id",
        "variant",
        "model_name",
        "topic",
        "difficulty",
        "query_text",
        "reference_answer",
        "answer_text",
        "answer_text_plain",
        "citations",
        "retrieved_chunk_ids",
        "retrieved_source_ids",
        "retrieved_context_text",
    ]

    if generation_results_df.empty:
        return pd.DataFrame(columns=expected_columns)

    query_lookup = evaluation_queries_df.set_index("query_id").to_dict(orient="index")
    retrieval_groups = retrieval_results_df.groupby(["query_id", "variant"])

    rows = []
    for generation_row in generation_results_df.to_dict(orient="records"):
        query_meta = query_lookup[generation_row["query_id"]]
        retrieved_rows = (
            retrieval_groups.get_group((generation_row["query_id"], generation_row["variant"]))
            .sort_values("rank")
            .to_dict(orient="records")
        )

        context_parts = []
        chunk_ids = []
        source_ids = []
        for retrieved_row in retrieved_rows:
            chunk_ids.append(retrieved_row["chunk_id"])
            source_ids.append(retrieved_row["source_id"])
            context_parts.append(
                "\n".join(
                    [
                        f"Source: {retrieved_row['source_id']}",
                        f"Section: {retrieved_row['section_path_text']}",
                        f"Content: {retrieved_row['chunk_text']}",
                    ]
                )
            )

        rows.append(
            {
                "query_id": generation_row["query_id"],
                "variant": generation_row["variant"],
                "model_name": generation_row["model_name"],
                "topic": query_meta["topic"],
                "difficulty": query_meta["difficulty"],
                "query_text": query_meta["query_text"],
                "reference_answer": query_meta["reference_answer"],
                "answer_text": generation_row["answer_text"],
                "answer_text_plain": generation_row["answer_text_plain"],
                "citations": generation_row["citations"],
                "retrieved_chunk_ids": json.dumps(chunk_ids),
                "retrieved_source_ids": json.dumps(source_ids),
                "retrieved_context_text": "\n\n".join(context_parts),
            }
        )

    return pd.DataFrame(rows, columns=expected_columns)


def build_evaluation_run_plan(
    evaluation_cases_df: pd.DataFrame,
    *,
    run_bertscore: bool,
    run_ragas: bool,
    bertscore_model_type: str,
    ragas_judge_model: str,
) -> dict:
    return {
        "run_bertscore": run_bertscore,
        "run_ragas": run_ragas,
        "num_evaluation_cases": int(len(evaluation_cases_df)),
        "num_variants": int(evaluation_cases_df["variant"].nunique()) if not evaluation_cases_df.empty else 0,
        "bertscore_model_type": bertscore_model_type,
        "ragas_judge_model": ragas_judge_model,
    }


def compute_bertscore_metrics(
    evaluation_cases_df: pd.DataFrame,
    *,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 8,
    device: str | None = None,
) -> pd.DataFrame:
    if evaluation_cases_df.empty:
        return pd.DataFrame(columns=["query_id", "variant", *BERTSCORE_COLUMNS])

    try:
        from bert_score import score as bertscore_score
    except ImportError as exc:
        raise ImportError(
            "bert-score is not installed. Install it from requirements.txt before running BERTScore."
        ) from exc

    candidates = [strip_inline_citations(text) for text in evaluation_cases_df["answer_text_plain"].fillna("")]
    references = [text for text in evaluation_cases_df["reference_answer"].fillna("")]

    precision, recall, f1 = bertscore_score(
        candidates,
        references,
        model_type=model_type,
        lang="en",
        batch_size=batch_size,
        device=device,
        verbose=False,
    )

    return pd.DataFrame(
        {
            "query_id": evaluation_cases_df["query_id"].tolist(),
            "variant": evaluation_cases_df["variant"].tolist(),
            "bertscore_p": [float(value) for value in precision.tolist()],
            "bertscore_r": [float(value) for value in recall.tolist()],
            "bertscore_f1": [float(value) for value in f1.tolist()],
        }
    )


def merge_metric_frames(
    metric_placeholder_df: pd.DataFrame,
    metric_frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    merged = metric_placeholder_df.copy()
    for metric_frame in metric_frames:
        if metric_frame.empty:
            continue
        merged = merged.merge(metric_frame, on=["query_id", "variant"], how="left", suffixes=("", "_new"))
        for column in list(merged.columns):
            if column.endswith("_new"):
                base_column = column[:-4]
                merged[base_column] = merged[column].combine_first(merged[base_column])
                merged = merged.drop(columns=[column])
    return merged
