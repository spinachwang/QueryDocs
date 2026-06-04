"""Run RAGAS evaluation against the QueryDocs pipeline.

CLI usage:
    python -m src.evaluation.evaluate --questions questions.json --output data/eval_results/run1
    python -m src.evaluation.evaluate --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import Dataset

from src.api.pipeline_wrapper import answer_question

logger = logging.getLogger(__name__)


# ---------- Data loading ----------

REQUIRED_QUESTION_FIELDS = ("text",)


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load questions from JSON file.

    Accepts either a top-level list of objects, each with at least a 'text' field,
    or a dict containing a 'questions' / 'data' key with such a list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        questions = payload
    elif isinstance(payload, dict):
        for key in ("questions", "data", "items"):
            if key in payload and isinstance(payload[key], list):
                questions = payload[key]
                break
        else:
            raise ValueError(
                f"Could not locate a list of questions in {p}. "
                f"Expected top-level list or a 'questions'/'data' key."
            )
    else:
        raise ValueError(f"Unsupported questions payload type: {type(payload).__name__}")

    for i, q in enumerate(questions):
        if not isinstance(q, dict) or not all(k in q for k in REQUIRED_QUESTION_FIELDS):
            raise ValueError(
                f"Question at index {i} missing required field(s) {REQUIRED_QUESTION_FIELDS}: {q}"
            )
    return questions


# ---------- Sample collection ----------

@dataclass
class Sample:
    question: str
    answer: str
    contexts: list[str]
    kind: str | None = None
    error: str | None = None


def build_samples(
    questions: Sequence[dict[str, Any]],
    answer_fn: Any = None,
    progress_every: int = 1,
) -> list[Sample]:
    """Run each question through the pipeline and capture (question, answer, contexts).

    `answer_fn` is resolved lazily so tests can monkeypatch the import.
    """
    if answer_fn is None:
        answer_fn = answer_question
    samples: list[Sample] = []
    total = len(questions)
    for i, q in enumerate(questions, start=1):
        text = q.get("text", "")
        kind = q.get("kind")
        logger.info("[%d/%d] running: %s", i, total, text[:60])
        try:
            result = answer_fn(text, kind) if kind else answer_fn(text, "string")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pipeline failed for question %d: %s", i, exc)
            samples.append(Sample(question=text, answer="", contexts=[], kind=kind, error=str(exc)))
            continue

        answer = result.get("final_answer", "") if isinstance(result, dict) else str(result)
        contexts = result.get("contexts", []) if isinstance(result, dict) else []
        contexts = [c for c in contexts if isinstance(c, str) and c.strip()]
        samples.append(Sample(question=text, answer=answer, contexts=contexts, kind=kind))
        if i % progress_every == 0:
            logger.info("  -> %d contexts, answer=%d chars", len(contexts), len(answer))
    return samples


# ---------- RAGAS metrics ----------

def _available_metrics() -> list[Any]:
    """Import the no-ground-truth metrics that exist in the installed ragas version.

    The user explicitly opted out of supervised metrics, so we prefer the
    `*WithoutReference` variants / no-GT metrics. We silently skip names that
    don't exist in the installed ragas version.
    """
    candidates_module_level: list[tuple[str, str]] = [
        ("ragas.metrics", "faithfulness"),
        ("ragas.metrics", "answer_relevancy"),
    ]
    candidates_class_based: list[tuple[str, str]] = [
        ("ragas.metrics", "LLMContextPrecisionWithoutReference"),
        ("ragas.metrics", "ContextRelevance"),
        ("ragas.metrics", "ResponseGroundedness"),
    ]
    metrics: list[Any] = []
    for mod, name in candidates_module_level:
        try:
            metric = __import__(mod, fromlist=[name]).__dict__[name]
        except (ImportError, KeyError, AttributeError):
            logger.debug("metric %s not available in this ragas version", name)
            continue
        metrics.append(metric)
    for mod, name in candidates_class_based:
        try:
            cls = __import__(mod, fromlist=[name]).__dict__[name]
        except (ImportError, KeyError, AttributeError):
            continue
        try:
            instance = cls()
        except TypeError:
            continue
        instance_name = getattr(instance, "name", name)
        if not any(getattr(m, "name", None) == instance_name for m in metrics):
            metrics.append(instance)
    if not metrics:
        raise RuntimeError(
            "No supported RAGAS metrics could be imported. "
            "Check that ragas is installed and that you have at least one of: "
            "faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference."
        )
    return metrics


def _filter_metrics_for_data(metrics: list[Any], samples: list[Sample]) -> list[Any]:
    """Drop metrics whose required inputs we don't have across the dataset.

    Drops a metric only when *no* sample could satisfy it; partial coverage is
    fine (RAGAS produces NaN for the missing rows).
    """
    usable: list[Any] = []
    has_any_answer = any(s.answer for s in samples)
    has_any_contexts = any(s.contexts for s in samples)
    for m in metrics:
        name = getattr(m, "name", repr(m))
        needs_answer = name in {"answer_relevancy", "faithfulness", "answer_correctness"}
        if needs_answer and not has_any_answer:
            logger.warning("Skipping %s: no sample has a non-empty answer.", name)
            continue
        if name == "answer_relevancy" and not has_any_contexts:
            logger.warning("Skipping %s: no sample has any contexts.", name)
            continue
        usable.append(m)
    return usable


def run_ragas(
    samples: Iterable[Sample],
    judge_llm: Any,
    embedder: Any | None = None,
) -> Any:
    """Run ragas.evaluate() on the collected samples."""
    from ragas import evaluate

    samples_list = list(samples)
    if not samples_list:
        raise ValueError("No samples to evaluate.")

    rows = [
        {
            "question": s.question,
            "answer": s.answer,
            "contexts": s.contexts,
        }
        for s in samples_list
        if not s.error and s.contexts and s.answer
    ]
    skipped = len(samples_list) - len(rows)
    if skipped:
        logger.warning("Skipping %d samples (no contexts/answer or pipeline error).", skipped)
    if not rows:
        raise ValueError("All samples lack contexts/answer; nothing to evaluate.")

    dataset = Dataset.from_list(rows)
    metrics = _filter_metrics_for_data(_available_metrics(), samples_list)

    logger.info("Running RAGAS with %d metrics on %d samples: %s",
                len(metrics), len(rows), [getattr(m, 'name', m) for m in metrics])

    kwargs: dict[str, Any] = {"dataset": dataset, "metrics": metrics, "llm": judge_llm}
    if embedder is not None:
        kwargs["embeddings"] = embedder

    return evaluate(**kwargs)


# ---------- Reporting ----------

def _result_to_dataframe(result: Any) -> "Any":  # type: ignore[name-defined]
    """ragas returns an EvaluationResult with .to_pandas(); we wrap it."""
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    raise TypeError(f"Unsupported ragas result type: {type(result).__name__}")


def save_report(
    result: Any,
    samples: Sequence[Sample],
    output_dir: Path,
    elapsed_sec: float,
) -> dict[str, Path]:
    """Write per-sample CSV and aggregated Markdown summary. Return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _result_to_dataframe(result)
    csv_path = output_dir / "report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = output_dir / "summary.md"
    metric_cols = [c for c in df.columns if c not in {"question", "answer", "contexts", "user_input", "retrieved_contexts", "response", "reference"}]
    means = {col: df[col].mean(skipna=True) for col in metric_cols if df[col].dtype.kind in "fi"}

    md_lines = [
        "# RAGAS Evaluation Summary",
        "",
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- Samples evaluated: {len(df)}",
        f"- Elapsed: {elapsed_sec:.1f}s",
        "",
        "## Aggregate metrics",
        "",
        "| metric | mean |",
        "|---|---|",
    ]
    for name, val in means.items():
        md_lines.append(f"| {name} | {val:.4f} |")
    md_lines.append("")
    md_lines.append("## Per-sample scores")
    md_lines.append("")
    md_lines.append("See `report.csv` for the full per-question breakdown.")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {"csv": csv_path, "summary": md_path}


# ---------- Orchestration ----------

@dataclass
class EvalConfig:
    questions_path: str = "questions.json"
    output_dir: str = ""
    limit: int | None = None
    skip_judge: bool = False
    skip_embedder: bool = False


def run_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    """End-to-end driver. Returns a dict with report paths and aggregate metrics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    questions = load_questions(cfg.questions_path)
    if cfg.limit:
        questions = questions[: cfg.limit]
    logger.info("Loaded %d questions from %s", len(questions), cfg.questions_path)

    t0 = time.time()
    samples = build_samples(questions)
    logger.info("Collected %d samples in %.1fs", len(samples), time.time() - t0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("data/eval_results") / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.skip_judge:
        logger.warning("skip_judge=True: NOT running RAGAS. Samples will be persisted only.")
        raw_path = output_dir / "samples.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({
                    "question": s.question,
                    "answer": s.answer,
                    "contexts": s.contexts,
                    "kind": s.kind,
                    "error": s.error,
                }, ensure_ascii=False) + "\n")
        return {"samples_path": str(raw_path), "n_samples": len(samples), "elapsed_sec": time.time() - t0}

    from src.evaluation.ragas_judge import build_minimax_judge, build_dashscope_embedder

    judge = build_minimax_judge()
    embedder = None if cfg.skip_embedder else build_dashscope_embedder()

    t0 = time.time()
    result = run_ragas(samples, judge_llm=judge, embedder=embedder)
    elapsed = time.time() - t0
    paths = save_report(result, samples, output_dir, elapsed)
    logger.info("RAGAS evaluation done in %.1fs. Report: %s", elapsed, paths["summary"])
    return {"paths": {k: str(v) for k, v in paths.items()}, "n_samples": len(samples), "elapsed_sec": elapsed}


# ---------- CLI ----------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluate", description="RAGAS evaluation for QueryDocs")
    p.add_argument("--questions", default="questions.json", help="Path to questions.json")
    p.add_argument("--output", default="", help="Output directory (default: data/eval_results/{ts})")
    p.add_argument("--limit", type=int, default=None, help="Limit number of questions (smoke test)")
    p.add_argument("--skip-judge", action="store_true", help="Skip RAGAS LLM judge; dump samples only")
    p.add_argument("--skip-embedder", action="store_true", help="Skip embedder (drops answer_relevancy)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = EvalConfig(
        questions_path=args.questions,
        output_dir=args.output,
        limit=args.limit,
        skip_judge=args.skip_judge,
        skip_embedder=args.skip_embedder,
    )
    result = run_evaluation(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
