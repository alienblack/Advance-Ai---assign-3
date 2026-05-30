from __future__ import annotations

import pandas as pd


FAILURE_LABELS = [
    "retrieval_miss",
    "source_imbalance",
    "incomplete_answer",
    "hallucinated_or_weakly_grounded_claim",
    "outdated_or_mismatched_context",
]


def build_failure_template(num_cases: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": f"F{i:02d}",
                "query_id": "",
                "variant": "",
                "failure_label": "",
                "notes": "",
            }
            for i in range(1, num_cases + 1)
        ]
    )


def summarize_metric_results(metric_results_df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    metric_columns = [
        column
        for column in [
            "context_relevance",
            "answer_relevance",
            "faithfulness",
            "context_precision",
            "context_recall",
            "bertscore_p",
            "bertscore_r",
            "bertscore_f1",
        ]
        if column in metric_results_df.columns
    ]

    if metric_results_df.empty:
        return pd.DataFrame(columns=[group_column, "num_cases", *metric_columns])

    summary = (
        metric_results_df.groupby(group_column, dropna=False)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = metric_results_df.groupby(group_column, dropna=False).size().reset_index(name="num_cases")
    return counts.merge(summary, on=group_column, how="left")
