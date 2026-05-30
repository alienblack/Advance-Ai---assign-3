from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SMALL_MODEL_NAME = "Gemma-2-2B-IT"
STRONG_MODEL_NAME = "Gemma-2-9B-IT"
RETRIEVAL_ENCODER_NAME = "all-MiniLM-L6-v2"

OFFICIAL_SOURCE_IDS = {
    "k8s_security_docs",
    "k8s_access_control_docs",
    "k8s_operational_security_docs",
    "k8s_cve_feed",
    "nsa_cisa_k8s_hardening",
    "nist_sp_800_190",
    "owasp_k8s_cheatsheet",
}

TOPIC_KEYWORDS = {
    "authentication_and_identity": [
        "authentication",
        "identity",
        "oidc",
        "tls",
        "certificate",
        "api server",
    ],
    "rbac_and_service_accounts": [
        "rbac",
        "role",
        "rolebinding",
        "service account",
        "least privilege",
        "tokenreview",
    ],
    "pod_security_and_admission_control": [
        "pod security",
        "admission",
        "restricted",
        "privileged",
        "capabilities",
        "hostpath",
    ],
    "secrets_handling": [
        "secret",
        "encryption at rest",
        "plaintext",
        "kms",
        "sealed secret",
        "external secret",
    ],
    "network_policy_and_traffic_isolation": [
        "networkpolicy",
        "ingress",
        "egress",
        "default deny",
        "traffic isolation",
        "namespace selector",
    ],
    "etcd_and_data_protection": [
        "etcd",
        "encryption at rest",
        "backup",
        "confidential data",
        "key management",
        "data protection",
    ],
    "image_and_supply_chain_security": [
        "image",
        "digest",
        "signing",
        "sbom",
        "supply chain",
        "vulnerability scan",
    ],
    "logging_auditing_and_detection": [
        "audit log",
        "logging",
        "detection",
        "siem",
        "event",
        "forensics",
    ],
    "multi_tenancy_and_isolation": [
        "multi tenancy",
        "tenant isolation",
        "namespace",
        "node isolation",
        "workload separation",
        "sandbox",
    ],
    "vulnerability_management_and_incident_response": [
        "vulnerability",
        "incident response",
        "inventory",
        "triage",
        "patching",
        "containment",
    ],
}


@dataclass(frozen=True)
class RetrievalRuntime:
    collection: object
    embeddings_array: np.ndarray
    chunk_lookup_df: pd.DataFrame
    query_encoder_name: str = RETRIEVAL_ENCODER_NAME


def build_variant_table() -> pd.DataFrame:
    rows = [
        {
            "variant": "V1",
            "retrieval_mode": "baseline_dense_top3",
            "retrieval_description": "Chroma dense similarity search with final top-3 chunks",
            "model_name": SMALL_MODEL_NAME,
            "model_tier": "smaller_open_source_llm",
        },
        {
            "variant": "V2",
            "retrieval_mode": "improved_dense_mmr_top3",
            "retrieval_description": "Fetch top-10, apply MMR, limit repeated sources to 2, keep final top-3",
            "model_name": SMALL_MODEL_NAME,
            "model_tier": "smaller_open_source_llm",
        },
        {
            "variant": "V3",
            "retrieval_mode": "intent_enriched_dense_mmr_top3",
            "retrieval_description": "Rewrite query with topic cues, fetch top-12, apply MMR, add topic-alignment bonus, limit repeated sources to 3, keep final top-3",
            "model_name": SMALL_MODEL_NAME,
            "model_tier": "smaller_open_source_llm",
        },
        {
            "variant": "V4",
            "retrieval_mode": "baseline_dense_top3",
            "retrieval_description": "Chroma dense similarity search with final top-3 chunks",
            "model_name": STRONG_MODEL_NAME,
            "model_tier": "stronger_open_source_llm",
        },
        {
            "variant": "V5",
            "retrieval_mode": "improved_dense_mmr_top3",
            "retrieval_description": "Fetch top-10, apply MMR, limit repeated sources to 2, keep final top-3",
            "model_name": STRONG_MODEL_NAME,
            "model_tier": "stronger_open_source_llm",
        },
        {
            "variant": "V6",
            "retrieval_mode": "intent_enriched_dense_mmr_top3",
            "retrieval_description": "Rewrite query with topic cues, fetch top-12, apply MMR, add topic-alignment bonus, limit repeated sources to 3, keep final top-3",
            "model_name": STRONG_MODEL_NAME,
            "model_tier": "stronger_open_source_llm",
        },
    ]
    return pd.DataFrame(rows)


def load_query_encoder(model_name: str = RETRIEVAL_ENCODER_NAME):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def encode_query_texts(query_encoder, query_texts: list[str], batch_size: int = 32) -> np.ndarray:
    return query_encoder.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def cosine_similarity(query_vector: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32)
    candidates = np.asarray(candidate_vectors, dtype=np.float32)
    return candidates @ query


def mmr_select(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    if len(candidate_vectors) == 0:
        return []

    similarities = cosine_similarity(query_vector, candidate_vectors)
    selected: list[int] = []
    remaining = list(range(len(candidate_vectors)))

    while remaining and len(selected) < top_k:
        if not selected:
            best_idx = max(remaining, key=lambda idx: float(similarities[idx]))
            selected.append(best_idx)
            remaining.remove(best_idx)
            continue

        def mmr_score(candidate_idx: int) -> float:
            relevance = float(similarities[candidate_idx])
            redundancy = max(
                float(candidate_vectors[candidate_idx] @ candidate_vectors[selected_idx])
                for selected_idx in selected
            )
            return lambda_mult * relevance - (1.0 - lambda_mult) * redundancy

        best_idx = max(remaining, key=mmr_score)
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def apply_official_source_bonus(
    candidates_df: pd.DataFrame,
    official_source_ids: Iterable[str] = OFFICIAL_SOURCE_IDS,
    bonus: float = 0.02,
) -> pd.DataFrame:
    adjusted = candidates_df.copy()
    adjusted["official_bonus"] = adjusted["source_id"].isin(set(official_source_ids)).astype(float) * bonus
    adjusted["rerank_score"] = adjusted["similarity_score"] + adjusted["official_bonus"]
    return adjusted


def topic_keywords(topic: str) -> list[str]:
    return TOPIC_KEYWORDS.get(topic, topic.replace("_", " ").split())


def build_intent_enriched_query(query_text: str, topic: str, difficulty: str) -> str:
    topic_label = topic.replace("_", " ")
    keyword_text = ", ".join(topic_keywords(topic))
    return (
        "Kubernetes security hardening question. "
        f"Topic: {topic_label}. "
        f"Difficulty: {difficulty}. "
        f"Relevant concepts: {keyword_text}. "
        f"Question: {query_text}"
    )


def apply_topic_alignment_bonus(
    candidates_df: pd.DataFrame,
    topic: str,
    bonus_per_hit: float = 0.01,
    max_bonus: float = 0.03,
) -> pd.DataFrame:
    adjusted = candidates_df.copy()
    keywords = [keyword.lower() for keyword in topic_keywords(topic)]
    bonuses = []

    for row in adjusted.to_dict(orient="records"):
        searchable_text = " ".join(
            [
                str(row.get("source_id", "")),
                str(row.get("section_path_text", "")),
                str(row.get("text", ""))[:800],
            ]
        ).lower()
        hits = sum(1 for keyword in keywords if keyword in searchable_text)
        bonuses.append(min(hits * bonus_per_hit, max_bonus))

    adjusted["topic_bonus"] = bonuses
    adjusted["rerank_score"] = adjusted["similarity_score"] + adjusted["topic_bonus"]
    return adjusted


def enforce_source_diversity(
    ranked_df: pd.DataFrame,
    top_k: int = 3,
    max_per_source: int = 2,
) -> pd.DataFrame:
    chosen_rows = []
    skipped_rows = []
    source_counts: dict[str, int] = {}

    for row in ranked_df.to_dict(orient="records"):
        source_id = row["source_id"]
        if source_counts.get(source_id, 0) >= max_per_source:
            skipped_rows.append(row)
            continue
        chosen_rows.append(row)
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        if len(chosen_rows) == top_k:
            break

    # If diversity constraints leave us short of top_k, backfill from the
    # highest-ranked skipped rows rather than returning fewer than 3 chunks.
    if len(chosen_rows) < top_k:
        already_selected = {row["chunk_id"] for row in chosen_rows}
        for row in skipped_rows:
            if row["chunk_id"] in already_selected:
                continue
            chosen_rows.append(row)
            already_selected.add(row["chunk_id"])
            if len(chosen_rows) == top_k:
                break

    return pd.DataFrame(chosen_rows)


def build_chunk_lookup_df(chunks_df: pd.DataFrame, embeddings_metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata_df = embeddings_metadata_df.copy().reset_index().rename(columns={"index": "embedding_row_index"})
    lookup_df = metadata_df.merge(
        chunks_df[
            [
                "chunk_id",
                "doc_id",
                "source_id",
                "text",
                "word_count",
                "section_path_text",
                "source_url",
                "trust_level",
                "chunk_index",
            ]
        ],
        on=["chunk_id", "doc_id", "source_id", "text", "word_count", "section_path_text", "source_url", "trust_level", "chunk_index"],
        how="left",
    )
    return lookup_df


def build_retrieval_runtime(
    collection,
    embeddings_array: np.ndarray,
    chunks_df: pd.DataFrame,
    embeddings_metadata_df: pd.DataFrame,
    query_encoder_name: str = RETRIEVAL_ENCODER_NAME,
) -> RetrievalRuntime:
    chunk_lookup_df = build_chunk_lookup_df(chunks_df, embeddings_metadata_df)
    return RetrievalRuntime(
        collection=collection,
        embeddings_array=embeddings_array,
        chunk_lookup_df=chunk_lookup_df,
        query_encoder_name=query_encoder_name,
    )


def _query_chroma_ids(collection, query_vector: np.ndarray, n_results: int) -> list[str]:
    query_result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )
    ids = query_result.get("ids", [[]])[0]
    return ids


def _candidate_rows_from_ids(runtime: RetrievalRuntime, chunk_ids: list[str], query_vector: np.ndarray) -> pd.DataFrame:
    if not chunk_ids:
        return pd.DataFrame()

    candidates_df = runtime.chunk_lookup_df[runtime.chunk_lookup_df["chunk_id"].isin(chunk_ids)].copy()
    chunk_order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    candidates_df["chroma_rank"] = candidates_df["chunk_id"].map(chunk_order)
    candidates_df = candidates_df.sort_values("chroma_rank").reset_index(drop=True)
    candidate_vectors = runtime.embeddings_array[candidates_df["embedding_row_index"].astype(int).to_numpy()]
    candidates_df["similarity_score"] = cosine_similarity(query_vector, candidate_vectors)
    return candidates_df


def baseline_retrieve(runtime: RetrievalRuntime, query_id: str, query_vector: np.ndarray) -> pd.DataFrame:
    top_ids = _query_chroma_ids(runtime.collection, query_vector, n_results=3)
    candidates_df = _candidate_rows_from_ids(runtime, top_ids, query_vector)
    if candidates_df.empty:
        return candidates_df

    result_df = candidates_df.copy()
    result_df["variant_retrieval_mode"] = "baseline_dense_top3"
    result_df["rerank_score"] = result_df["similarity_score"]
    result_df["rank"] = range(1, len(result_df) + 1)
    result_df["query_id"] = query_id
    return result_df[
        [
            "query_id",
            "rank",
            "chunk_id",
            "source_id",
            "similarity_score",
            "rerank_score",
            "text",
            "section_path_text",
            "source_url",
            "word_count",
            "variant_retrieval_mode",
        ]
    ]


def improved_retrieve(
    runtime: RetrievalRuntime,
    query_id: str,
    query_vector: np.ndarray,
    fetch_k: int = 10,
    final_k: int = 3,
    mmr_lambda: float = 0.7,
    official_bonus: float = 0.0,
    max_per_source: int = 2,
) -> pd.DataFrame:
    top_ids = _query_chroma_ids(runtime.collection, query_vector, n_results=fetch_k)
    candidates_df = _candidate_rows_from_ids(runtime, top_ids, query_vector)
    if candidates_df.empty:
        return candidates_df

    candidate_vectors = runtime.embeddings_array[candidates_df["embedding_row_index"].astype(int).to_numpy()]
    mmr_order = mmr_select(
        query_vector=query_vector,
        candidate_vectors=candidate_vectors,
        top_k=len(candidates_df),
        lambda_mult=mmr_lambda,
    )
    rerank_df = candidates_df.iloc[mmr_order].copy().reset_index(drop=True)
    rerank_df["mmr_rank"] = range(1, len(rerank_df) + 1)
    rerank_df = apply_official_source_bonus(rerank_df, bonus=official_bonus)
    rerank_df = rerank_df.sort_values(
        ["rerank_score", "similarity_score", "mmr_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rerank_df = enforce_source_diversity(rerank_df, top_k=final_k, max_per_source=max_per_source)
    rerank_df = rerank_df.reset_index(drop=True)
    rerank_df["rank"] = range(1, len(rerank_df) + 1)
    rerank_df["query_id"] = query_id
    rerank_df["variant_retrieval_mode"] = "improved_dense_mmr_top3"
    return rerank_df[
        [
            "query_id",
            "rank",
            "chunk_id",
            "source_id",
            "similarity_score",
            "rerank_score",
            "text",
            "section_path_text",
            "source_url",
            "word_count",
            "variant_retrieval_mode",
        ]
    ]


def intent_enriched_retrieve(
    runtime: RetrievalRuntime,
    query_id: str,
    query_vector: np.ndarray,
    topic: str,
    fetch_k: int = 12,
    final_k: int = 3,
    mmr_lambda: float = 0.75,
    max_per_source: int = 3,
) -> pd.DataFrame:
    top_ids = _query_chroma_ids(runtime.collection, query_vector, n_results=fetch_k)
    candidates_df = _candidate_rows_from_ids(runtime, top_ids, query_vector)
    if candidates_df.empty:
        return candidates_df

    candidate_vectors = runtime.embeddings_array[candidates_df["embedding_row_index"].astype(int).to_numpy()]
    mmr_order = mmr_select(
        query_vector=query_vector,
        candidate_vectors=candidate_vectors,
        top_k=len(candidates_df),
        lambda_mult=mmr_lambda,
    )
    rerank_df = candidates_df.iloc[mmr_order].copy().reset_index(drop=True)
    rerank_df["mmr_rank"] = range(1, len(rerank_df) + 1)
    rerank_df = apply_topic_alignment_bonus(rerank_df, topic=topic)
    rerank_df = rerank_df.sort_values(
        ["rerank_score", "similarity_score", "mmr_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rerank_df = enforce_source_diversity(rerank_df, top_k=final_k, max_per_source=max_per_source)
    rerank_df = rerank_df.reset_index(drop=True)
    rerank_df["rank"] = range(1, len(rerank_df) + 1)
    rerank_df["query_id"] = query_id
    rerank_df["variant_retrieval_mode"] = "intent_enriched_dense_mmr_top3"
    return rerank_df[
        [
            "query_id",
            "rank",
            "chunk_id",
            "source_id",
            "similarity_score",
            "rerank_score",
            "text",
            "section_path_text",
            "source_url",
            "word_count",
            "variant_retrieval_mode",
        ]
    ]


def run_retrieval_variants(
    runtime: RetrievalRuntime,
    evaluation_queries_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    query_encoder_name: str = RETRIEVAL_ENCODER_NAME,
    batch_size: int = 32,
) -> pd.DataFrame:
    query_encoder = load_query_encoder(query_encoder_name)
    query_embeddings = encode_query_texts(
        query_encoder=query_encoder,
        query_texts=evaluation_queries_df["query_text"].tolist(),
        batch_size=batch_size,
    )
    query_embedding_map = {
        query_id: query_embeddings[index]
        for index, query_id in enumerate(evaluation_queries_df["query_id"].tolist())
    }
    enriched_query_texts = [
        build_intent_enriched_query(row["query_text"], row["topic"], row["difficulty"])
        for row in evaluation_queries_df.to_dict(orient="records")
    ]
    enriched_query_embeddings = encode_query_texts(
        query_encoder=query_encoder,
        query_texts=enriched_query_texts,
        batch_size=batch_size,
    )
    enriched_query_embedding_map = {
        query_id: enriched_query_embeddings[index]
        for index, query_id in enumerate(evaluation_queries_df["query_id"].tolist())
    }

    rows = []
    retrieval_mode_map = dict(zip(variant_df["variant"], variant_df["retrieval_mode"]))

    for query_row in evaluation_queries_df.to_dict(orient="records"):
        query_id = query_row["query_id"]
        query_vector = query_embedding_map[query_id]
        enriched_query_vector = enriched_query_embedding_map[query_id]

        baseline_df = baseline_retrieve(runtime, query_id=query_id, query_vector=query_vector)
        improved_df = improved_retrieve(runtime, query_id=query_id, query_vector=query_vector)
        intent_enriched_df = intent_enriched_retrieve(
            runtime,
            query_id=query_id,
            query_vector=enriched_query_vector,
            topic=query_row["topic"],
        )

        for variant in variant_df["variant"]:
            retrieval_mode = retrieval_mode_map[variant]
            if retrieval_mode == "baseline_dense_top3":
                source_df = baseline_df
            elif retrieval_mode == "improved_dense_mmr_top3":
                source_df = improved_df
            elif retrieval_mode == "intent_enriched_dense_mmr_top3":
                source_df = intent_enriched_df
            else:
                raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
            source_records = source_df.to_dict(orient="records")
            for record in source_records:
                rows.append(
                    {
                        "query_id": query_id,
                        "query_text": query_row["query_text"],
                        "topic": query_row["topic"],
                        "difficulty": query_row["difficulty"],
                        "variant": variant,
                        "retrieval_mode": retrieval_mode,
                        "rank": int(record["rank"]),
                        "chunk_id": record["chunk_id"],
                        "source_id": record["source_id"],
                        "similarity_score": float(record["similarity_score"]),
                        "rerank_score": float(record["rerank_score"]),
                        "chunk_text": record["text"],
                        "section_path_text": record["section_path_text"],
                        "source_url": record["source_url"],
                        "word_count": int(record["word_count"]),
                    }
                )

    retrieval_results_df = pd.DataFrame(rows)
    return retrieval_results_df


def summarize_retrieval_results(retrieval_results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_variant_summary = (
        retrieval_results_df.groupby(["variant", "retrieval_mode"], as_index=False)
        .agg(
            num_rows=("chunk_id", "count"),
            num_queries=("query_id", "nunique"),
            avg_similarity=("similarity_score", "mean"),
            avg_rerank=("rerank_score", "mean"),
            unique_sources=("source_id", "nunique"),
        )
        .sort_values("variant")
    )

    source_concentration = (
        retrieval_results_df.groupby(["variant", "source_id"], as_index=False)
        .agg(num_rows=("chunk_id", "count"))
        .sort_values(["variant", "num_rows", "source_id"], ascending=[True, False, True])
    )
    return per_variant_summary, source_concentration
