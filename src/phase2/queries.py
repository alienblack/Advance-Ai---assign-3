from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TOPIC_BUCKETS = [
    "authentication_and_identity",
    "rbac_and_service_accounts",
    "pod_security_and_admission_control",
    "secrets_handling",
    "network_policy_and_traffic_isolation",
    "etcd_and_data_protection",
    "image_and_supply_chain_security",
    "logging_auditing_and_detection",
    "multi_tenancy_and_isolation",
    "vulnerability_management_and_incident_response",
]

DIFFICULTY_PATTERN_PER_TOPIC = [
    "basic",
    "basic",
    "basic",
    "intermediate",
    "intermediate",
    "intermediate",
    "intermediate",
    "intermediate",
    "advanced",
    "advanced",
]


@dataclass(frozen=True)
class QueryBenchmarkConfig:
    topics: list[str]
    difficulty_pattern_per_topic: list[str]


DEFAULT_QUERY_CONFIG = QueryBenchmarkConfig(
    topics=TOPIC_BUCKETS,
    difficulty_pattern_per_topic=DIFFICULTY_PATTERN_PER_TOPIC,
)


def build_query_blueprint(config: QueryBenchmarkConfig = DEFAULT_QUERY_CONFIG) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    query_counter = 1

    for topic in config.topics:
        for slot_index, difficulty in enumerate(config.difficulty_pattern_per_topic, start=1):
            rows.append(
                {
                    "query_id": f"Q{query_counter:03d}",
                    "topic": topic,
                    "difficulty": difficulty,
                    "topic_slot": f"{topic}__{slot_index:02d}",
                    "query_text": "",
                    "reference_answer": "",
                    "review_status": "draft",
                    "draft_source": "llm_draft_then_manual_review",
                    "notes": "",
                }
            )
            query_counter += 1

    frame = pd.DataFrame(rows)
    return frame


def build_query_distribution_tables(query_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_distribution = (
        query_df.groupby("topic", as_index=False)
        .agg(num_queries=("query_id", "count"))
        .sort_values("topic")
    )
    difficulty_distribution = (
        query_df.groupby("difficulty", as_index=False)
        .agg(num_queries=("query_id", "count"))
        .sort_values("difficulty")
    )
    return topic_distribution, difficulty_distribution

