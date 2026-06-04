"""RAGAS judge LLM & embedder wiring.

- Judge LLM: MiniMax-M2.7 via Anthropic-compatible API
- Embedder: DashScope (text-embedding-v4) — same provider the main
  QueryDocs pipeline uses for retrieval embeddings, so RAGAS metric
  scores are computed in the same vector space as the retrieval system.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(
            f"Environment variable {name} is required for RAGAS evaluation. "
            f"Set it in .env or your shell."
        )
    return val


def build_minimax_judge() -> Any:
    """Build a RAGAS-wrapped judge LLM pointing at MiniMax (Anthropic-compatible)."""
    load_dotenv()
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    api_key = _require_env("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")
    model = os.getenv("MINIMAX_JUDGE_MODEL", "MiniMax-M2.7")

    chat = ChatAnthropic(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=2,
        max_tokens=8192,
    )
    logger.info("RAGAS judge LLM ready: model=%s base_url=%s", model, base_url)
    return LangchainLLMWrapper(chat)


def build_dashscope_embedder() -> Any:
    """Build a RAGAS-wrapped embedder using DashScope text-embedding-v4."""
    load_dotenv()
    from langchain_community.embeddings import DashScopeEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    api_key = _require_env("DASHSCOPE_API_KEY")
    model = os.getenv("RAGAS_EMBED_MODEL", "text-embedding-v4")

    embeddings = DashScopeEmbeddings(model=model, dashscope_api_key=api_key)
    logger.info("RAGAS embedder ready: provider=dashscope model=%s", model)
    return LangchainEmbeddingsWrapper(embeddings)


def build_default_judge_and_embedder() -> Tuple[Any, Any]:
    """Convenience: build both and return."""
    return build_minimax_judge(), build_dashscope_embedder()
