"""Unit tests for src.evaluation.evaluate.

We mock the pipeline (answer_question) and ragas.evaluate so tests run
without any LLM calls and without network access.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import evaluate as evaluate_mod


# ---------- load_questions ----------

def test_load_questions_top_level_list(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([
        {"text": "Q1", "kind": "string"},
        {"text": "Q2"},
    ]), encoding="utf-8")
    out = evaluate_mod.load_questions(p)
    assert len(out) == 2
    assert out[0]["text"] == "Q1"


def test_load_questions_dict_with_questions_key(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"questions": [{"text": "Q1"}]}), encoding="utf-8")
    assert evaluate_mod.load_questions(p)[0]["text"] == "Q1"


def test_load_questions_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_mod.load_questions(tmp_path / "absent.json")


def test_load_questions_invalid_shape(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"kind": "string"}]), encoding="utf-8")  # missing 'text'
    with pytest.raises(ValueError):
        evaluate_mod.load_questions(p)


def test_load_questions_unsupported_payload(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps("a string"), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_mod.load_questions(p)


# ---------- build_samples ----------

def test_build_samples_collects_question_answer_contexts():
    questions = [{"text": "Q1", "kind": "string"}, {"text": "Q2", "kind": "string"}]
    fake_answer = MagicMock(side_effect=[
        {"final_answer": "A1", "contexts": ["ctx-1", "ctx-2"]},
        {"final_answer": "A2", "contexts": ["ctx-3"]},
    ])
    samples = evaluate_mod.build_samples(questions, answer_fn=fake_answer)
    assert len(samples) == 2
    assert samples[0].question == "Q1"
    assert samples[0].answer == "A1"
    assert samples[0].contexts == ["ctx-1", "ctx-2"]
    assert samples[1].contexts == ["ctx-3"]
    assert fake_answer.call_count == 2


def test_build_samples_handles_pipeline_error():
    questions = [{"text": "Q1"}]

    def boom(_q, _k):
        raise RuntimeError("pipeline down")

    samples = evaluate_mod.build_samples(questions, answer_fn=boom)
    assert samples[0].error == "pipeline down"
    assert samples[0].answer == ""
    assert samples[0].contexts == []


def test_build_samples_filters_non_string_contexts():
    questions = [{"text": "Q1"}]
    fake = MagicMock(return_value={"final_answer": "A", "contexts": ["ok", "", None, "also-ok"]})
    samples = evaluate_mod.build_samples(questions, answer_fn=fake)
    assert samples[0].contexts == ["ok", "also-ok"]


# ---------- run_ragas filtering ----------

def test_filter_metrics_keeps_metrics_when_some_answers_present():
    """Partial coverage is fine; metrics are dropped only when *all* samples are empty."""
    samples = [
        evaluate_mod.Sample(question="q1", answer="", contexts=["c"]),
        evaluate_mod.Sample(question="q2", answer="real answer", contexts=["c"]),
    ]
    faith = MagicMock()
    faith.name = "faithfulness"
    rel = MagicMock()
    rel.name = "answer_relevancy"
    out = evaluate_mod._filter_metrics_for_data([faith, rel], samples)
    assert faith in out
    assert rel in out


def test_filter_metrics_drops_when_no_answers_at_all():
    samples = [evaluate_mod.Sample(question="q", answer="", contexts=["c"])]
    faith = MagicMock()
    faith.name = "faithfulness"
    out = evaluate_mod._filter_metrics_for_data([faith], samples)
    assert faith not in out


def test_filter_metrics_drops_relevancy_when_no_contexts():
    samples = [evaluate_mod.Sample(question="q", answer="a", contexts=[])]
    rel = MagicMock()
    rel.name = "answer_relevancy"
    out = evaluate_mod._filter_metrics_for_data([rel], samples)
    assert rel not in out


# ---------- save_report ----------

class _FakeResult:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def test_save_report_writes_csv_and_md(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "question": ["q1", "q2"],
        "answer": ["a1", "a2"],
        "contexts": [["c1"], ["c2"]],
        "faithfulness": [0.9, 0.8],
        "answer_relevancy": [0.7, 0.6],
    })
    result = _FakeResult(df)
    samples = [
        evaluate_mod.Sample(question="q1", answer="a1", contexts=["c1"]),
        evaluate_mod.Sample(question="q2", answer="a2", contexts=["c2"]),
    ]
    paths = evaluate_mod.save_report(result, samples, tmp_path, elapsed_sec=1.23)
    csv_path = paths["csv"]
    md_path = paths["summary"]
    assert csv_path.exists()
    assert md_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert "faithfulness" in csv_text
    md_text = md_path.read_text(encoding="utf-8")
    assert "RAGAS Evaluation Summary" in md_text
    assert "faithfulness" in md_text
    assert "0.8500" in md_text  # mean of 0.9 and 0.8


# ---------- run_evaluation (skip_judge path) ----------

def test_run_evaluation_skip_judge_dumps_samples(tmp_path):
    cfg = evaluate_mod.EvalConfig(
        questions_path=str(tmp_path / "absent.json"),
        output_dir=str(tmp_path / "out"),
        skip_judge=True,
        limit=None,
    )
    qpath = tmp_path / "q.json"
    qpath.write_text(json.dumps([{"text": "Q1", "kind": "string"}]), encoding="utf-8")
    cfg.questions_path = str(qpath)

    with patch.object(evaluate_mod, "answer_question", return_value={
        "final_answer": "A1",
        "contexts": ["c1", "c2"],
    }):
        result = evaluate_mod.run_evaluation(cfg)

    assert result["n_samples"] == 1
    samples_path = Path(result["samples_path"])
    assert samples_path.exists()
    line = samples_path.read_text(encoding="utf-8").strip().splitlines()[0]
    obj = json.loads(line)
    assert obj["question"] == "Q1"
    assert obj["contexts"] == ["c1", "c2"]


# ---------- run_ragas smoke (mocked) ----------

def test_run_ragas_invokes_ragas_evaluate(monkeypatch):
    samples = [
        evaluate_mod.Sample(question="q1", answer="a1", contexts=["c1"]),
        evaluate_mod.Sample(question="q2", answer="a2", contexts=["c2", "c3"]),
    ]
    captured = {}

    def fake_evaluate(dataset, metrics, llm, embeddings=None):
        captured["n_rows"] = len(dataset)
        captured["n_metrics"] = len(metrics)
        captured["llm"] = llm
        return "fake-result"

    fake_metric = MagicMock()
    fake_metric.name = "faithfulness"
    monkeypatch.setattr(evaluate_mod, "_available_metrics", lambda: [fake_metric])
    monkeypatch.setitem(sys.modules, "ragas", MagicMock(evaluate=fake_evaluate))

    out = evaluate_mod.run_ragas(samples, judge_llm="JUDGE", embedder="EMB")
    assert out == "fake-result"
    assert captured["n_rows"] == 2
    assert captured["n_metrics"] == 1


def test_run_ragas_skips_samples_without_contexts(monkeypatch):
    samples = [
        evaluate_mod.Sample(question="q1", answer="a1", contexts=[]),
        evaluate_mod.Sample(question="q2", answer="a2", contexts=["c"]),
    ]
    captured = {}

    def fake_evaluate(dataset, metrics, llm, embeddings=None):
        captured["n_rows"] = len(dataset)
        return "ok"

    fake_metric = MagicMock()
    fake_metric.name = "faithfulness"
    monkeypatch.setattr(evaluate_mod, "_available_metrics", lambda: [fake_metric])
    monkeypatch.setitem(sys.modules, "ragas", MagicMock(evaluate=fake_evaluate))

    out = evaluate_mod.run_ragas(samples, judge_llm="J", embedder=None)
    assert out == "ok"
    assert captured["n_rows"] == 1  # the empty-context one dropped


# ---------- _available_metrics ----------

def test_available_metrics_handles_missing_modules(monkeypatch):
    """When ragas is not installed, _available_metrics should raise clearly."""
    monkeypatch.setitem(sys.modules, "ragas", None)
    monkeypatch.setitem(sys.modules, "ragas.metrics", None)
    with pytest.raises(RuntimeError):
        evaluate_mod._available_metrics()


# ---------- CLI ----------

def test_main_skip_judge_prints_paths(capsys, tmp_path):
    qpath = tmp_path / "q.json"
    qpath.write_text(json.dumps([{"text": "Q1"}]), encoding="utf-8")
    out_dir = tmp_path / "out"
    with patch.object(evaluate_mod, "answer_question", return_value={
        "final_answer": "A", "contexts": ["c"]
    }):
        rc = evaluate_mod.main([
            "--questions", str(qpath),
            "--output", str(out_dir),
            "--skip-judge",
        ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "samples.jsonl" in captured.out
