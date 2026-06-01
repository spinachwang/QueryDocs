from pathlib import Path
from pyprojroot import here
from src.pipeline import Pipeline, RunConfig
import logging

logger = logging.getLogger(__name__)

_pipeline_instance = None


def get_pipeline() -> Pipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        root_path = here() / "data" / "stock_data"
        # Use MiniMax config - same as configs["minimax"] in pipeline.py
        api_config = RunConfig(
            use_serialized_tables=False,
            parent_document_retrieval=False,
            llm_reranking=True,
            parallel_requests=4,
            submission_file=True,
            pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = MiniMax-M2.7",
            api_provider="minimax",
            answering_model="MiniMax-M2.7",
            config_suffix="_minimax",
            vector_db_dir_override="data/stock_data/debug_data/databases/vector_dbs",
            documents_dir_override="data/stock_data/debug_data/chunked_reports"
        )
        _pipeline_instance = Pipeline(root_path, run_config=api_config)
        logger.info("Pipeline initialized")
    return _pipeline_instance


def answer_question(question: str, kind: str = "string") -> dict:
    pipeline = get_pipeline()
    result = pipeline.answer_single_question(question, kind=kind)
    return result