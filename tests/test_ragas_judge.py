"""Unit tests for src.evaluation.ragas_judge.

We mock langchain/ragas wrapper classes so the tests don't require real
network credentials. The real keys are only checked at construction time.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import ragas_judge


# ---------- _require_env ----------

def test_require_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(EnvironmentError) as exc:
        ragas_judge._require_env("MISSING_KEY")
    assert "MISSING_KEY" in str(exc.value)


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "abc")
    assert ragas_judge._require_env("PRESENT_KEY") == "abc"


# ---------- build_minimax_judge ----------

def test_build_minimax_judge_uses_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("MINIMAX_JUDGE_MODEL", "MiniMax-X")

    fake_chat = MagicMock()
    fake_wrapper_cls = MagicMock(return_value="WRAPPED")
    with patch.dict(sys.modules, {
        "langchain_anthropic": MagicMock(ChatAnthropic=fake_chat),
        "ragas.llms": MagicMock(LangchainLLMWrapper=fake_wrapper_cls),
    }):
        out = ragas_judge.build_minimax_judge()

    assert out == "WRAPPED"
    fake_chat.assert_called_once()
    kwargs = fake_chat.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["base_url"] == "https://example.test/anthropic"
    assert kwargs["model"] == "MiniMax-X"
    fake_wrapper_cls.assert_called_once_with(fake_chat.return_value)


def test_build_minimax_judge_missing_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://example.test/anthropic")
    monkeypatch.setenv("MINIMAX_JUDGE_MODEL", "MiniMax-X")

    with patch.dict(sys.modules, {
        "langchain_anthropic": MagicMock(),
        "ragas.llms": MagicMock(),
    }), patch.object(ragas_judge, "load_dotenv", lambda: None):
        with pytest.raises(EnvironmentError):
            ragas_judge.build_minimax_judge()


# ---------- build_dashscope_embedder ----------

def test_build_dashscope_embedder_uses_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ds")
    monkeypatch.setenv("RAGAS_EMBED_MODEL", "text-embedding-v4")

    fake_embeddings_cls = MagicMock()
    fake_wrapper_cls = MagicMock(return_value="WRAPPED-EMB")
    with patch.dict(sys.modules, {
        "langchain_community": MagicMock(),
        "langchain_community.embeddings": MagicMock(DashScopeEmbeddings=fake_embeddings_cls),
        "ragas.embeddings": MagicMock(LangchainEmbeddingsWrapper=fake_wrapper_cls),
    }):
        out = ragas_judge.build_dashscope_embedder()

    assert out == "WRAPPED-EMB"
    fake_embeddings_cls.assert_called_once_with(model="text-embedding-v4", dashscope_api_key="sk-ds")


def test_build_dashscope_embedder_missing_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with patch.dict(sys.modules, {
        "langchain_community": MagicMock(),
        "langchain_community.embeddings": MagicMock(),
        "ragas.embeddings": MagicMock(),
    }), patch.object(ragas_judge, "load_dotenv", lambda: None):
        with pytest.raises(EnvironmentError):
            ragas_judge.build_dashscope_embedder()


# ---------- build_default_judge_and_embedder ----------

def test_build_default_returns_both(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://x")
    monkeypatch.setenv("MINIMAX_JUDGE_MODEL", "MiniMax-M2.7")
    monkeypatch.setenv("RAGAS_EMBED_MODEL", "text-embedding-v4")

    with patch.object(ragas_judge, "build_minimax_judge", return_value="J") as j, \
         patch.object(ragas_judge, "build_dashscope_embedder", return_value="E") as e:
        judge, embedder = ragas_judge.build_default_judge_and_embedder()
    assert judge == "J"
    assert embedder == "E"
    j.assert_called_once()
    e.assert_called_once()
