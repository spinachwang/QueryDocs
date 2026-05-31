from fastapi import APIRouter, HTTPException
from src.api.models import QARequest, QAResponse, Reference
from src.api.pipeline_wrapper import answer_question
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    try:
        logger.info(f"Received question: {request.question}")
        result = answer_question(request.question, request.kind)

        references = []
        if "references" in result and result["references"]:
            for ref in result["references"]:
                if isinstance(ref, dict):
                    references.append(Reference(
                        pdf_sha1=ref.get("pdf_sha1", ""),
                        page_index=ref.get("page_index", 0)
                    ))

        return QAResponse(
            step_by_step_analysis=result.get("step_by_step_analysis", ""),
            reasoning_summary=result.get("reasoning_summary", ""),
            relevant_pages=result.get("relevant_pages", []),
            final_answer=result.get("final_answer", ""),
            references=references
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))