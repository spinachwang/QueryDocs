from pydantic import BaseModel
from typing import Optional


class QARequest(BaseModel):
    question: str
    kind: Optional[str] = "string"


class Reference(BaseModel):
    pdf_sha1: str
    page_index: int


class QAResponse(BaseModel):
    step_by_step_analysis: str
    reasoning_summary: str
    relevant_pages: list[int]
    final_answer: str
    references: list[Reference]
    contexts: list[str] = []


class HealthResponse(BaseModel):
    status: str