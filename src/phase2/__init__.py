"""Phase 2 support helpers for retrieval, generation, and evaluation."""

from .artifacts import (
    Phase1Snapshot,
    build_phase1_snapshot,
    load_chunks_df,
    load_embeddings_array,
    load_embeddings_metadata_df,
    open_chroma_collection,
    summarize_phase1_snapshot,
)
from .benchmark_content import build_reviewed_benchmark
from .generation import build_generation_cases, run_generation_variants
from .evaluation import (
    build_evaluation_cases,
    build_evaluation_run_plan,
    build_metric_placeholder_frame,
    compute_bertscore_metrics,
    merge_metric_frames,
)
from .queries import build_query_blueprint, build_query_distribution_tables
from .retrieval import (
    RETRIEVAL_ENCODER_NAME,
    build_retrieval_runtime,
    build_variant_table,
    run_retrieval_variants,
    summarize_retrieval_results,
)
from .analysis import build_failure_template, summarize_metric_results

__all__ = [
    "Phase1Snapshot",
    "build_phase1_snapshot",
    "load_chunks_df",
    "load_embeddings_array",
    "load_embeddings_metadata_df",
    "open_chroma_collection",
    "summarize_phase1_snapshot",
    "build_reviewed_benchmark",
    "build_evaluation_cases",
    "build_evaluation_run_plan",
    "build_metric_placeholder_frame",
    "compute_bertscore_metrics",
    "merge_metric_frames",
    "build_generation_cases",
    "build_query_blueprint",
    "build_query_distribution_tables",
    "RETRIEVAL_ENCODER_NAME",
    "build_retrieval_runtime",
    "build_failure_template",
    "build_variant_table",
    "run_generation_variants",
    "run_retrieval_variants",
    "summarize_retrieval_results",
    "summarize_metric_results",
]
