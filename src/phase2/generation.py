from __future__ import annotations

import json
import os
import re
from typing import Iterable

import pandas as pd

from .retrieval import SMALL_MODEL_NAME, STRONG_MODEL_NAME


SMALL_MODEL_HF_ID = "google/gemma-2-2b-it"
STRONG_MODEL_HF_ID = "google/gemma-2-9b-it"

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant focused on Kubernetes security hardening. "
    "Use only the retrieved context. If the context is incomplete, say that the context is incomplete. "
    "Write concise practitioner-style guidance. "
    "The final line of the answer must be a citation list in square brackets."
)

USER_PROMPT_TEMPLATE = """User question:
{query_text}

Allowed citation chunk IDs:
{citation_list}

Output requirements:
- Answer only from the retrieved context.
- If the retrieved context is incomplete, say that clearly.
- Keep the answer concise and practitioner-focused.
- The final line must be exactly one bracketed citation list using only the allowed chunk IDs.
- Example final line: [chunk_id_1, chunk_id_2]

Retrieved context:
{context_block}
"""


def build_context_block(retrieved_rows: Iterable[dict]) -> str:
    blocks = []
    for row in retrieved_rows:
        blocks.append(
            "\n".join(
                [
                    f"Chunk ID: {row['chunk_id']}",
                    f"Source ID: {row['source_id']}",
                    f"Section: {row.get('section_path_text', '')}",
                    f"Text: {row['chunk_text']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_generation_prompt(query_text: str, retrieved_rows: Iterable[dict]) -> str:
    citation_ids = [row["chunk_id"] for row in retrieved_rows]
    return USER_PROMPT_TEMPLATE.format(
        query_text=query_text,
        citation_list=", ".join(citation_ids),
        context_block=build_context_block(retrieved_rows),
    )


def variant_to_model_name(variant: str) -> str:
    if variant in {"V1", "V2", "V3"}:
        return SMALL_MODEL_NAME
    if variant in {"V4", "V5", "V6"}:
        return STRONG_MODEL_NAME
    raise ValueError(f"Unknown variant: {variant}")


def display_model_name_to_hf_id(model_name: str) -> str:
    if model_name == SMALL_MODEL_NAME:
        return SMALL_MODEL_HF_ID
    if model_name == STRONG_MODEL_NAME:
        return STRONG_MODEL_HF_ID
    raise ValueError(f"Unknown model display name: {model_name}")


def build_generation_cases(retrieval_results_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    if retrieval_results_df.empty:
        return pd.DataFrame()

    variant_map = dict(zip(variant_df["variant"], variant_df["model_name"]))
    allowed_variants = set(variant_map)
    filtered_retrieval_df = retrieval_results_df[retrieval_results_df["variant"].isin(allowed_variants)].copy()
    if filtered_retrieval_df.empty:
        return pd.DataFrame(
            columns=[
                "query_id",
                "variant",
                "model_name",
                "topic",
                "difficulty",
                "query_text",
                "chunk_ids",
                "source_ids",
                "prompt_text",
            ]
        )

    grouped = filtered_retrieval_df.sort_values(["query_id", "variant", "rank"]).groupby(
        ["query_id", "variant", "query_text", "topic", "difficulty"], as_index=False
    )

    rows = []
    for (query_id, variant, query_text, topic, difficulty), group_df in grouped:
        retrieved_rows = group_df.sort_values("rank").to_dict(orient="records")
        model_name = variant_map[variant]
        rows.append(
            {
                "query_id": query_id,
                "variant": variant,
                "model_name": model_name,
                "topic": topic,
                "difficulty": difficulty,
                "query_text": query_text,
                "chunk_ids": json.dumps(group_df.sort_values("rank")["chunk_id"].tolist()),
                "source_ids": json.dumps(group_df.sort_values("rank")["source_id"].tolist()),
                "prompt_text": build_generation_prompt(query_text, retrieved_rows),
            }
        )
    return pd.DataFrame(rows)


def extract_citations(answer_text: str) -> list[str]:
    matches = re.findall(r"\[([^\]]+)\]", answer_text)
    if not matches:
        return []
    last_match = matches[-1]
    return [item.strip() for item in last_match.split(",") if item.strip()]


def strip_inline_citations(answer_text: str) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", "", answer_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def enforce_case_citations(answer_text: str, allowed_chunk_ids: list[str]) -> tuple[str, str, list[str]]:
    extracted = extract_citations(answer_text)
    filtered = [citation for citation in extracted if citation in allowed_chunk_ids]

    if not filtered:
        filtered = allowed_chunk_ids

    answer_body = strip_inline_citations(answer_text)
    citation_suffix = f"[{', '.join(filtered)}]"
    final_answer_text = f"{answer_body}\n\n{citation_suffix}".strip()
    final_answer_text_plain = strip_inline_citations(final_answer_text)
    return final_answer_text, final_answer_text_plain, filtered


def load_generation_components(
    model_name: str,
    prefer_local_files: bool = True,
    use_4bit_if_cuda: bool = True,
):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model_id = display_model_name_to_hf_id(model_name)
    local_only_attempts = [True, False] if prefer_local_files else [False]

    tokenizer = None
    model = None
    last_error = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for local_only in local_only_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, local_files_only=local_only)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if device == "cuda":
                model_kwargs = {"device_map": "auto"}
                if use_4bit_if_cuda:
                    try:
                        from transformers import BitsAndBytesConfig

                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                    except Exception:
                        model_kwargs["torch_dtype"] = torch.float16
                else:
                    model_kwargs["torch_dtype"] = torch.float16

                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    local_files_only=local_only,
                    **model_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    local_files_only=local_only,
                    torch_dtype=torch.float32,
                )
                model.to(device)
            return tokenizer, model, device
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            last_error = exc

    raise last_error  # pragma: no cover - runtime environment dependent


def generate_single_answer(
    tokenizer,
    model,
    device: str,
    prompt_text: str,
    max_new_tokens: int = 220,
):
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception as exc:
        if "System role not supported" not in str(exc):
            raise

        fallback_messages = [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n\n{prompt_text}",
            }
        ]
        input_ids = tokenizer.apply_chat_template(
            fallback_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    attention_mask = torch.ones_like(input_ids)

    if device == "cuda":
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
    else:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][input_ids.shape[-1] :]
    answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    answer_text_plain = strip_inline_citations(answer_text)
    citations = extract_citations(answer_text)
    return answer_text, answer_text_plain, citations


def run_generation_variants(
    generation_cases_df: pd.DataFrame,
    max_new_tokens: int = 220,
    case_limit: int | None = None,
    prefer_local_files: bool = True,
    use_4bit_if_cuda: bool = True,
) -> pd.DataFrame:
    if generation_cases_df.empty:
        return pd.DataFrame()

    working_df = generation_cases_df.copy()
    if case_limit is not None:
        working_df = working_df.head(case_limit).copy()

    rows = []
    for model_name in working_df["model_name"].drop_duplicates().tolist():
        tokenizer, model, device = load_generation_components(
            model_name=model_name,
            prefer_local_files=prefer_local_files,
            use_4bit_if_cuda=use_4bit_if_cuda,
        )

        model_cases_df = working_df[working_df["model_name"] == model_name].copy()
        for case in model_cases_df.to_dict(orient="records"):
            answer_text, answer_text_plain, citations = generate_single_answer(
                tokenizer=tokenizer,
                model=model,
                device=device,
                prompt_text=case["prompt_text"],
                max_new_tokens=max_new_tokens,
            )
            allowed_chunk_ids = json.loads(case["chunk_ids"]) if isinstance(case["chunk_ids"], str) else list(case["chunk_ids"])
            final_answer_text, final_answer_text_plain, final_citations = enforce_case_citations(
                answer_text=answer_text,
                allowed_chunk_ids=allowed_chunk_ids,
            )
            rows.append(
                {
                    "query_id": case["query_id"],
                    "variant": case["variant"],
                    "model_name": case["model_name"],
                    "answer_text": final_answer_text,
                    "answer_text_plain": final_answer_text_plain,
                    "citations": json.dumps(final_citations),
                }
            )
    return pd.DataFrame(rows)
