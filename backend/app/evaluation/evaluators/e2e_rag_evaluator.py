import time
from typing import Optional, Sequence, Mapping
from llama_index.core.evaluation.base import EvaluationResult
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)

max_retries = 3
retry_delay = 2


class E2ERagEvaluator:
    def __init__(self, model="gpt-4o", threshold=0.7) -> None:
        self._model = model
        self._threshold = threshold

        self._contextual_precision = ContextualPrecisionMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._contextual_recall = ContextualRecallMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._contextual_relevancy = ContextualRelevancyMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._answer_relevancy = AnswerRelevancyMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )
        self._faithfulness = FaithfulnessMetric(
            threshold=self._threshold, model=self._model, include_reason=True
        )

    def evaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
    ) -> Mapping[str, EvaluationResult]:
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=reference,
            retrieval_context=contexts,
        )

        evalution_results = []
        for attempt in range(max_retries):
            try:
                evalution_results = evaluate(
                    test_cases=[test_case],
                    metrics=[
                        self._contextual_precision,
                        self._contextual_recall,
                        self._contextual_relevancy,
                        self._answer_relevancy,
                        self._faithfulness,
                    ],
                    print_results=False,
                    show_indicator=False,
                )
            except ValueError as e:
                print(f"Caught ValueError: {e}")
                print(f"Retrying {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)

        if len(evalution_results) == 0:
            return {}

        metrics_results = {}
        for eval_result in evalution_results:
            for score in eval_result.metrics_metadata:
                metrics_results[score.metric] = EvaluationResult(
                    query=query,
                    response=response,
                    contexts=contexts,
                    passing=score.success,
                    score=score.score or 0.0,
                    feedback=score.reason or score.error,
                )

        return metrics_results
