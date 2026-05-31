export interface Reference {
  pdf_sha1: string;
  page_index: number;
}

export interface QAResponse {
  step_by_step_analysis: string;
  reasoning_summary: string;
  relevant_pages: number[];
  final_answer: string;
  references: Reference[];
}