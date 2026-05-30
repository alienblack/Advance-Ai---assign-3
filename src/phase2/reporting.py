from __future__ import annotations

import math

import pandas as pd

from .extended_evaluation import parse_listish


FAILURE_LABELS = [
    "retrieval_miss",
    "source_imbalance",
    "topic_drift",
    "incomplete_answer",
    "hallucinated_or_weakly_grounded_claim",
    "outdated_or_unsafe_guidance",
    "enterprise_bias_or_context_gap",
]

PAIRWISE_COMPARISONS = [
    ("V2", "V1", "rq1_small_mmr_vs_dense"),
    ("V5", "V4", "rq1_strong_mmr_vs_dense"),
    ("V3", "V2", "rq2_small_intent_vs_mmr"),
    ("V6", "V5", "rq2_strong_intent_vs_mmr"),
    ("V4", "V1", "rq3_baseline_strong_vs_small"),
    ("V5", "V2", "rq3_mmr_strong_vs_small"),
    ("V6", "V3", "rq3_intent_strong_vs_small"),
]


def classify_source_family(source_id: str) -> str:
    source_id = str(source_id)
    if source_id == "k8s_cve_feed":
        return "official_vulnerability_feed"
    if source_id.startswith("k8s_"):
        return "official_kubernetes_docs"
    if source_id.startswith("nsa_cisa") or source_id.startswith("nist_"):
        return "government_guidance"
    if source_id.startswith("owasp_") or source_id.startswith("cncf_") or source_id.startswith("kubescape_"):
        return "community_industry_guidance"
    if "book" in source_id:
        return "books_and_handbooks"
    if "thesis" in source_id or "academic" in source_id or "aalto" in source_id:
        return "academic_reference"
    return "curated_reference"


def compute_source_family_share(retrieval_results_df: pd.DataFrame) -> pd.DataFrame:
    if retrieval_results_df.empty:
        return pd.DataFrame(columns=["variant", "source_family", "num_rows", "share_of_variant"])
    working_df = retrieval_results_df.copy()
    working_df["source_family"] = working_df["source_id"].map(classify_source_family)
    counts_df = (
        working_df.groupby(["variant", "source_family"], as_index=False)
        .agg(num_rows=("chunk_id", "count"))
        .sort_values(["variant", "num_rows", "source_family"], ascending=[True, False, True])
    )
    totals = counts_df.groupby("variant")["num_rows"].transform("sum")
    counts_df["share_of_variant"] = counts_df["num_rows"] / totals
    return counts_df


def _compute_balanced_difficulty_quota(evaluation_cases_df: pd.DataFrame, target_n: int) -> dict[str, int]:
    difficulty_counts = evaluation_cases_df["difficulty"].value_counts().sort_index()
    raw_targets = (difficulty_counts / difficulty_counts.sum()) * target_n
    floor_targets = raw_targets.apply(math.floor).to_dict()
    remainder = target_n - sum(floor_targets.values())
    fractional_order = sorted(
        raw_targets.index.tolist(),
        key=lambda key: raw_targets[key] - floor_targets[key],
        reverse=True,
    )
    for key in fractional_order[:remainder]:
        floor_targets[key] += 1
    return {str(key): int(value) for key, value in floor_targets.items()}


def build_manual_audit_sample(
    metric_results_df: pd.DataFrame,
    evaluation_cases_df: pd.DataFrame,
    *,
    sample_size: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    if metric_results_df.empty or evaluation_cases_df.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in ["context_relevance", "answer_relevance", "faithfulness", "context_precision", "context_recall", "bertscore_f1"]
        if column in metric_results_df.columns
    ]
    merged_df = metric_results_df.merge(
        evaluation_cases_df[
            [
                "query_id",
                "variant",
                "model_name",
                "query_text",
                "reference_answer",
                "answer_text",
                "answer_text_plain",
                "citations",
                "retrieved_chunk_ids",
                "retrieved_source_ids",
                "retrieved_contexts",
                "retrieved_context_text",
            ]
        ],
        on=["query_id", "variant"],
        how="left",
    ).copy()

    merged_df["citation_valid_auto"] = merged_df.apply(
        lambda row: bool(parse_listish(row.get("citations"))) and set(parse_listish(row.get("citations"))).issubset(set(parse_listish(row.get("retrieved_chunk_ids")))),
        axis=1,
    )
    merged_df = merged_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    merged_df["selection_order"] = range(len(merged_df))

    variants = sorted(merged_df["variant"].dropna().unique().tolist())
    topics = sorted(merged_df["topic"].dropna().unique().tolist())
    variant_quota = {variant: sample_size // len(variants) for variant in variants}
    for variant in variants[: sample_size % len(variants)]:
        variant_quota[variant] += 1
    topic_quota = {topic: sample_size // len(topics) for topic in topics}
    for topic in topics[: sample_size % len(topics)]:
        topic_quota[topic] += 1
    difficulty_quota = _compute_balanced_difficulty_quota(merged_df, sample_size)

    selected_indices = []
    selected_set = set()
    variant_remaining = variant_quota.copy()
    topic_remaining = topic_quota.copy()
    difficulty_remaining = difficulty_quota.copy()

    def choose_best(candidate_df: pd.DataFrame) -> int | None:
        if candidate_df.empty:
            return None
        ranked_df = candidate_df.copy()
        ranked_df["difficulty_priority"] = ranked_df["difficulty"].map(lambda value: difficulty_remaining.get(value, 0))
        ranked_df = ranked_df.sort_values(["difficulty_priority", "selection_order"], ascending=[False, True])
        return int(ranked_df.iloc[0].name)

    for topic in topics:
        for variant in variants:
            if len(selected_indices) >= sample_size:
                break
            if topic_remaining.get(topic, 0) <= 0 or variant_remaining.get(variant, 0) <= 0:
                continue
            cell_df = merged_df[
                (merged_df["topic"] == topic)
                & (merged_df["variant"] == variant)
                & (~merged_df.index.isin(selected_set))
            ]
            pick = choose_best(cell_df)
            if pick is None:
                continue
            selected_indices.append(pick)
            selected_set.add(pick)
            variant_remaining[variant] -= 1
            topic_remaining[topic] -= 1
            difficulty_value = str(merged_df.loc[pick, "difficulty"])
            difficulty_remaining[difficulty_value] = max(0, difficulty_remaining.get(difficulty_value, 0) - 1)

    while len(selected_indices) < sample_size:
        remaining_df = merged_df[
            (~merged_df.index.isin(selected_set))
            & (merged_df["variant"].map(lambda value: variant_remaining.get(value, 0)) > 0)
            & (merged_df["topic"].map(lambda value: topic_remaining.get(value, 0)) > 0)
        ].copy()
        if remaining_df.empty:
            break
        remaining_df["priority"] = (
            remaining_df["topic"].map(lambda value: topic_remaining.get(value, 0)) * 100
            + remaining_df["variant"].map(lambda value: variant_remaining.get(value, 0)) * 10
            + remaining_df["difficulty"].map(lambda value: difficulty_remaining.get(value, 0))
        )
        remaining_df = remaining_df.sort_values(["priority", "selection_order"], ascending=[False, True])
        pick = int(remaining_df.iloc[0].name)
        selected_indices.append(pick)
        selected_set.add(pick)
        variant_value = str(merged_df.loc[pick, "variant"])
        topic_value = str(merged_df.loc[pick, "topic"])
        difficulty_value = str(merged_df.loc[pick, "difficulty"])
        variant_remaining[variant_value] = max(0, variant_remaining.get(variant_value, 0) - 1)
        topic_remaining[topic_value] = max(0, topic_remaining.get(topic_value, 0) - 1)
        difficulty_remaining[difficulty_value] = max(0, difficulty_remaining.get(difficulty_value, 0) - 1)

    audit_df = merged_df.loc[selected_indices].copy().reset_index(drop=True)
    rename_map = {
        "context_relevance": "context_relevance_score",
        "answer_relevance": "answer_relevance_score",
        "faithfulness": "faithfulness_score",
        "context_precision": "context_precision_score",
        "context_recall": "context_recall_score",
    }
    audit_df = audit_df.rename(columns=rename_map)
    audit_df.insert(0, "manual_audit_id", [f"A{i:03d}" for i in range(1, len(audit_df) + 1)])
    audit_df["retrieval_relevant"] = ""
    audit_df["answer_relevant"] = ""
    audit_df["faithful_to_context"] = ""
    audit_df["actionable_guidance"] = ""
    audit_df["citation_valid"] = ""
    audit_df["topic_drift"] = ""
    audit_df["bias_or_context_gap"] = ""
    audit_df["notes"] = ""
    ordered_columns = [
        "manual_audit_id",
        "query_id",
        "variant",
        "model_name",
        "topic",
        "difficulty",
        "query_text",
        "reference_answer",
        "retrieved_chunk_ids",
        "retrieved_source_ids",
        "retrieved_contexts",
        "answer_text",
        "answer_text_plain",
        "citations",
        "citation_valid_auto",
        "context_relevance_score",
        "answer_relevance_score",
        "faithfulness_score",
        "context_precision_score",
        "context_recall_score",
        "bertscore_f1",
        "retrieval_relevant",
        "answer_relevant",
        "faithful_to_context",
        "actionable_guidance",
        "citation_valid",
        "topic_drift",
        "bias_or_context_gap",
        "notes",
    ]
    return audit_df[ordered_columns]


def build_failure_analysis_frame(
    metric_results_df: pd.DataFrame,
    evaluation_cases_df: pd.DataFrame,
    *,
    num_cases: int = 12,
) -> pd.DataFrame:
    if metric_results_df.empty or evaluation_cases_df.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in ["context_relevance", "answer_relevance", "faithfulness", "context_precision", "context_recall", "bertscore_f1"]
        if column in metric_results_df.columns
    ]
    working_df = metric_results_df.merge(
        evaluation_cases_df[
            [
                "query_id",
                "variant",
                "model_name",
                "query_text",
                "reference_answer",
                "answer_text",
                "answer_text_plain",
                "citations",
                "retrieved_chunk_ids",
                "retrieved_source_ids",
                "retrieved_contexts",
                "retrieved_context_text",
            ]
        ],
        on=["query_id", "variant"],
        how="left",
    ).copy()
    working_df["automated_quality_score"] = working_df[metric_columns].mean(axis=1, skipna=True)
    failure_df = (
        working_df.sort_values(["automated_quality_score", "query_id", "variant"], ascending=[True, True, True])
        .drop_duplicates("query_id", keep="first")
        .head(num_cases)
        .copy()
        .reset_index(drop=True)
    )
    failure_df.insert(0, "failure_case_id", [f"F{i:02d}" for i in range(1, len(failure_df) + 1)])
    failure_df["failure_label"] = ""
    failure_df["notes"] = ""
    return failure_df[
        [
            "failure_case_id",
            "query_id",
            "variant",
            "model_name",
            "topic",
            "difficulty",
            "automated_quality_score",
            *metric_columns,
            "query_text",
            "reference_answer",
            "retrieved_chunk_ids",
            "retrieved_source_ids",
            "retrieved_contexts",
            "answer_text",
            "answer_text_plain",
            "citations",
            "failure_label",
            "notes",
        ]
    ]


def build_bias_audit_frame(
    evaluation_cases_df: pd.DataFrame,
    retrieval_results_df: pd.DataFrame,
    *,
    num_cases: int = 20,
    random_seed: int = 7,
) -> pd.DataFrame:
    if evaluation_cases_df.empty or retrieval_results_df.empty:
        return pd.DataFrame()

    merged_df = evaluation_cases_df.copy()
    merged_df = merged_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    variant_quota = {variant: num_cases // merged_df["variant"].nunique() for variant in sorted(merged_df["variant"].unique())}
    for variant in list(variant_quota)[: num_cases % len(variant_quota)]:
        variant_quota[variant] += 1

    selected_rows = []
    for variant in sorted(merged_df["variant"].unique()):
        variant_df = merged_df[merged_df["variant"] == variant].head(variant_quota[variant])
        selected_rows.append(variant_df)
    audit_df = pd.concat(selected_rows, ignore_index=True).head(num_cases).copy()
    audit_df.insert(0, "bias_audit_id", [f"B{i:02d}" for i in range(1, len(audit_df) + 1)])
    audit_df["dominant_source_family"] = audit_df["retrieved_source_ids"].apply(
        lambda value: ", ".join(sorted({classify_source_family(source_id) for source_id in parse_listish(value)}))
    )
    audit_df["enterprise_bias_flag"] = ""
    audit_df["under_resourced_context_gap"] = ""
    audit_df["outdated_guidance_risk"] = ""
    audit_df["human_review_required"] = ""
    audit_df["notes"] = ""
    return audit_df[
        [
            "bias_audit_id",
            "query_id",
            "variant",
            "model_name",
            "topic",
            "difficulty",
            "query_text",
            "retrieved_source_ids",
            "dominant_source_family",
            "answer_text",
            "enterprise_bias_flag",
            "under_resourced_context_gap",
            "outdated_guidance_risk",
            "human_review_required",
            "notes",
        ]
    ]


def build_pairwise_comparison_df(
    metric_results_df: pd.DataFrame,
    evaluation_cases_df: pd.DataFrame,
    better_variant: str,
    baseline_variant: str,
    label: str,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in ["context_relevance", "answer_relevance", "faithfulness", "context_precision", "context_recall", "bertscore_f1"]
        if column in metric_results_df.columns
    ]
    working_df = metric_results_df[metric_results_df["variant"].isin([better_variant, baseline_variant])].copy()
    if working_df.empty:
        return pd.DataFrame()

    pivot_frames = []
    for metric_name in metric_columns:
        pivot_df = working_df.pivot_table(
            index="query_id",
            columns="variant",
            values=metric_name,
            aggfunc="first",
        ).reset_index()
        rename_map = {
            better_variant: f"{metric_name}_{better_variant}",
            baseline_variant: f"{metric_name}_{baseline_variant}",
        }
        pivot_frames.append(pivot_df.rename(columns=rename_map))

    comparison_df = pivot_frames[0]
    for extra_df in pivot_frames[1:]:
        comparison_df = comparison_df.merge(extra_df, on="query_id", how="left")

    evaluation_meta = evaluation_cases_df.drop_duplicates("query_id")[
        ["query_id", "topic", "difficulty", "query_text", "reference_answer"]
    ]
    comparison_df = comparison_df.merge(evaluation_meta, on="query_id", how="left")

    better_answer_df = (
        evaluation_cases_df[evaluation_cases_df["variant"] == better_variant][
            ["query_id", "answer_text_plain"]
        ]
        .rename(columns={"answer_text_plain": f"answer_text_plain_{better_variant}"})
    )
    baseline_answer_df = (
        evaluation_cases_df[evaluation_cases_df["variant"] == baseline_variant][
            ["query_id", "answer_text_plain"]
        ]
        .rename(columns={"answer_text_plain": f"answer_text_plain_{baseline_variant}"})
    )
    comparison_df = comparison_df.merge(better_answer_df, on="query_id", how="left").merge(baseline_answer_df, on="query_id", how="left")

    delta_columns = []
    for metric_name in metric_columns:
        better_column = f"{metric_name}_{better_variant}"
        baseline_column = f"{metric_name}_{baseline_variant}"
        delta_column = f"{metric_name}_delta"
        comparison_df[delta_column] = comparison_df[better_column] - comparison_df[baseline_column]
        delta_columns.append(delta_column)

    comparison_df["composite_delta"] = comparison_df[delta_columns].mean(axis=1, skipna=True)
    comparison_df["comparison_label"] = label
    comparison_df["better_variant"] = better_variant
    comparison_df["baseline_variant"] = baseline_variant
    return comparison_df.sort_values("composite_delta", ascending=False).reset_index(drop=True)


def summarize_pairwise_comparisons(pairwise_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for pairwise_df in pairwise_frames:
        if pairwise_df.empty:
            continue
        rows.append(
            {
                "comparison_label": pairwise_df["comparison_label"].iloc[0],
                "better_variant": pairwise_df["better_variant"].iloc[0],
                "baseline_variant": pairwise_df["baseline_variant"].iloc[0],
                "num_queries": int(pairwise_df["query_id"].nunique()),
                "better_wins": int((pairwise_df["composite_delta"] > 1e-9).sum()),
                "baseline_wins": int((pairwise_df["composite_delta"] < -1e-9).sum()),
                "ties": int((pairwise_df["composite_delta"].abs() <= 1e-9).sum()),
                "mean_composite_delta": float(pairwise_df["composite_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)
