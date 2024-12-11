from typing import Optional

from pydantic import BaseModel

from app.models import EvaluationTask


class CreateEvaluationTask(BaseModel):
    name: str
    evaluation_dataset_id: int
    chat_engine: str = "default"
    run_size: Optional[int] = None


class EvaluationTaskSummary(BaseModel):
    task: EvaluationTask
    not_start: int
    succeed: int
    errored: int
    progressing: int
    avg_factual_correctness: Optional[float]
    avg_semantic_similarity: Optional[float]
    min_factual_correctness: Optional[float]
    min_semantic_similarity: Optional[float]
    max_factual_correctness: Optional[float]
    max_semantic_similarity: Optional[float]
    std_factual_correctness: Optional[float]
    std_semantic_similarity: Optional[float]


class UpdateEvaluationDataset(BaseModel):
    name: str


class CreateEvaluationDataset(BaseModel):
    name: str
    upload_id: Optional[int]


class ModifyEvaluationDatasetItem(BaseModel):
    query: str
    reference: str
    retrieved_contexts: list[str]
    extra: dict
    evaluation_dataset_id: int
