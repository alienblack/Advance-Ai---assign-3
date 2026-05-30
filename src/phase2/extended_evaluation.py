from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path

import pandas as pd

from .evaluation import RAGAS_METRIC_COLUMNS, strip_inline_citations
from .generation import (
    SMALL_MODEL_HF_ID,
    SMALL_MODEL_NAME,
    STRONG_MODEL_HF_ID,
    STRONG_MODEL_NAME,
)


def parse_listish(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            continue
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return [text]


def _ragas_value(result) -> float | None:
    if result is None:
        return None
    if hasattr(result, "value"):
        return float(result.value)
    return float(result)


def _run_async_in_notebook(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass
    return loop.run_until_complete(coro)


def _invoke_metric_score(metric, **kwargs):
    sample = kwargs.pop("sample", None)
    if sample is not None:
        if hasattr(metric, "single_turn_ascore"):
            return _run_async_in_notebook(metric.single_turn_ascore(sample))
        if hasattr(metric, "single_turn_score"):
            async def _single_turn_score_in_thread():
                return await asyncio.to_thread(metric.single_turn_score, sample)

            return _run_async_in_notebook(_single_turn_score_in_thread())
    if hasattr(metric, "ascore"):
        return _run_async_in_notebook(metric.ascore(**kwargs))
    if hasattr(metric, "score"):
        async def _score_in_thread():
            return await asyncio.to_thread(metric.score, **kwargs)

        return _run_async_in_notebook(_score_in_thread())
    raise AttributeError(f"{type(metric).__name__} does not expose score() or ascore()")


def _resolve_hf_model_id(model_name: str) -> str:
    if model_name == SMALL_MODEL_NAME:
        return SMALL_MODEL_HF_ID
    if model_name == STRONG_MODEL_NAME:
        return STRONG_MODEL_HF_ID
    return model_name


def _setup_openai_ragas_clients(
    ragas_judge_model: str,
    ragas_embedding_model: str,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI-backed RAGAS run.")

    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    last_error = None
    for client in (AsyncOpenAI(api_key=api_key), OpenAI(api_key=api_key)):
        try:
            llm = llm_factory(ragas_judge_model, client=client)
            embeddings = embedding_factory("openai", model=ragas_embedding_model, client=client)
            return llm, embeddings
        except Exception as exc:
            last_error = exc
    raise last_error


def _setup_hf_local_ragas_clients(
    ragas_judge_model: str,
    ragas_embedding_model: str,
    *,
    use_4bit_if_cuda: bool = False,
):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    import torch
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception as exc:
        raise ImportError(
            "langchain-huggingface is required for local Hugging Face RAGAS embeddings."
        ) from exc

    try:
        from langchain_huggingface import HuggingFacePipeline
    except Exception:
        try:
            from langchain_community.llms import HuggingFacePipeline
        except Exception as exc:
            raise ImportError(
                "langchain-huggingface or langchain-community is required for local Hugging Face RAGAS judge models."
            ) from exc

    hf_model_id = _resolve_hf_model_id(ragas_judge_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict = {}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
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
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(hf_model_id, **model_kwargs)
    if device != "cuda":
        model.to(device)

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
    )

    langchain_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    llm = LangchainLLMWrapper(langchain_llm)

    langchain_embeddings = HuggingFaceEmbeddings(model_name=ragas_embedding_model)
    embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    return llm, embeddings


def setup_ragas_clients(
    ragas_judge_model: str = "gpt-4.1-mini",
    ragas_embedding_model: str = "text-embedding-3-small",
    *,
    ragas_provider: str = "openai",
    use_4bit_if_cuda: bool = False,
):
    provider = ragas_provider.strip().lower()
    if provider == "openai":
        return _setup_openai_ragas_clients(ragas_judge_model, ragas_embedding_model)
    if provider in {"hf_local", "huggingface_local", "local_hf", "huggingface"}:
        return _setup_hf_local_ragas_clients(
            ragas_judge_model,
            ragas_embedding_model,
            use_4bit_if_cuda=use_4bit_if_cuda,
        )
    raise ValueError(f"Unsupported ragas_provider: {ragas_provider}")


def build_ragas_scorers(llm, embeddings):
    try:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            ContextRelevance,
            Faithfulness,
        )

        return {
            "context_relevance": ContextRelevance(llm=llm),
            "answer_relevance": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "faithfulness": Faithfulness(llm=llm),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }
    except Exception:
        from ragas.metrics import (
            ContextRelevance,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        return {
            "context_relevance": ContextRelevance(llm=llm),
            "answer_relevance": ResponseRelevancy(llm=llm, embeddings=embeddings),
            "faithfulness": Faithfulness(llm=llm),
            "context_precision": LLMContextPrecisionWithReference(llm=llm),
            "context_recall": LLMContextRecall(llm=llm),
        }


def compute_ragas_metrics(
    evaluation_cases_df: pd.DataFrame,
    *,
    ragas_judge_model: str = "gpt-4.1-mini",
    ragas_embedding_model: str = "text-embedding-3-small",
    ragas_provider: str = "openai",
    use_4bit_if_cuda: bool = False,
    existing_results_df: pd.DataFrame | None = None,
    save_path: Path | None = None,
    checkpoint_every: int = 5,
) -> pd.DataFrame:
    output_columns = ["query_id", "variant", *RAGAS_METRIC_COLUMNS, "ragas_error"]
    if evaluation_cases_df.empty:
        return pd.DataFrame(columns=output_columns)

    if existing_results_df is None:
        existing_results_df = pd.DataFrame(columns=output_columns)
    else:
        existing_results_df = existing_results_df.copy()
        for column in output_columns:
            if column not in existing_results_df.columns:
                existing_results_df[column] = None
        existing_results_df = existing_results_df[output_columns].drop_duplicates(["query_id", "variant"], keep="last")
        completion_mask = existing_results_df["ragas_error"].isna()
        for metric_column in RAGAS_METRIC_COLUMNS:
            completion_mask &= existing_results_df[metric_column].notna()
        existing_results_df = existing_results_df[completion_mask].copy()

    completed_pairs = set(zip(existing_results_df["query_id"], existing_results_df["variant"]))
    pending_rows = [
        row
        for row in evaluation_cases_df.to_dict(orient="records")
        if (row["query_id"], row["variant"]) not in completed_pairs
    ]

    if not pending_rows:
        return existing_results_df.reset_index(drop=True)

    llm, embeddings = setup_ragas_clients(
        ragas_judge_model=ragas_judge_model,
        ragas_embedding_model=ragas_embedding_model,
        ragas_provider=ragas_provider,
        use_4bit_if_cuda=use_4bit_if_cuda,
    )
    scorers = build_ragas_scorers(llm, embeddings)
    from ragas import SingleTurnSample

    result_rows = existing_results_df.to_dict(orient="records")
    for index, row in enumerate(pending_rows, start=1):
        contexts = parse_listish(row.get("retrieved_contexts"))
        response_text = row.get("answer_text_plain") or strip_inline_citations(row.get("answer_text", ""))
        sample = SingleTurnSample(
            user_input=row["query_text"],
            retrieved_contexts=contexts,
            response=response_text,
            reference=row["reference_answer"],
        )
        metrics_row = {
            "query_id": row["query_id"],
            "variant": row["variant"],
            "ragas_error": None,
        }
        try:
            metrics_row["context_relevance"] = _ragas_value(
                _invoke_metric_score(
                    scorers["context_relevance"],
                    sample=sample,
                )
            )
            metrics_row["answer_relevance"] = _ragas_value(
                _invoke_metric_score(
                    scorers["answer_relevance"],
                    sample=sample,
                )
            )
            metrics_row["faithfulness"] = _ragas_value(
                _invoke_metric_score(
                    scorers["faithfulness"],
                    sample=sample,
                )
            )
            metrics_row["context_precision"] = _ragas_value(
                _invoke_metric_score(
                    scorers["context_precision"],
                    sample=sample,
                )
            )
            metrics_row["context_recall"] = _ragas_value(
                _invoke_metric_score(
                    scorers["context_recall"],
                    sample=sample,
                )
            )
        except Exception as exc:
            metrics_row.update({column: None for column in RAGAS_METRIC_COLUMNS})
            metrics_row["ragas_error"] = f"{type(exc).__name__}: {exc}"

        result_rows.append(metrics_row)
        if save_path is not None and (index % checkpoint_every == 0 or index == len(pending_rows)):
            pd.DataFrame(result_rows, columns=output_columns).to_csv(save_path, index=False)

    return pd.DataFrame(result_rows, columns=output_columns)


def compute_citation_validity_frame(evaluation_cases_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in evaluation_cases_df.to_dict(orient="records"):
        retrieved_chunk_ids = parse_listish(row.get("retrieved_chunk_ids"))
        citations = parse_listish(row.get("citations"))
        citation_valid = bool(citations) and set(citations).issubset(set(retrieved_chunk_ids))
        rows.append(
            {
                "query_id": row["query_id"],
                "variant": row["variant"],
                "num_retrieved_chunk_ids": len(retrieved_chunk_ids),
                "num_citations": len(citations),
                "citation_valid": citation_valid,
            }
        )
    return pd.DataFrame(rows)


def compute_duplicate_answer_summary(generation_results_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if generation_results_df.empty:
        empty_df = pd.DataFrame(columns=["query_id", "num_answers", "num_unique_answers", "duplicate_answers"])
        return empty_df, {"num_queries": 0, "overall_duplicate_rate": None}

    working_df = generation_results_df.copy()
    working_df["normalized_answer"] = (
        working_df["answer_text_plain"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    duplicate_df = (
        working_df.groupby("query_id", as_index=False)
        .agg(
            num_answers=("variant", "count"),
            num_unique_answers=("normalized_answer", "nunique"),
        )
    )
    duplicate_df["duplicate_answers"] = duplicate_df["num_answers"] - duplicate_df["num_unique_answers"]
    total_duplicates = int(duplicate_df["duplicate_answers"].sum())
    total_answers = int(duplicate_df["num_answers"].sum())
    summary = {
        "num_queries": int(duplicate_df["query_id"].nunique()),
        "overall_duplicate_rate": (total_duplicates / total_answers) if total_answers else None,
    }
    return duplicate_df, summary
