import logging
import traceback

import httpx
from app.celery import app as celery_app
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import FactualCorrectness, SemanticSimilarity
from sqlmodel import Session, select
from celery.utils.log import get_task_logger
from tenacity import retry, stop_after_attempt, wait_fixed

from app.core.config import settings, Environment
from app.core.db import engine
from app.models import (
    EvaluationTask,
    EvaluationStatus,
    EvaluationTaskItem,
)
from dotenv import load_dotenv

load_dotenv()

logger = get_task_logger(__name__)

if settings.ENVIRONMENT == Environment.LOCAL:
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)


@celery_app.task
def add_evaluation_task(evaluation_task_id: int):
    logger.info(
        f"[add_evaluation_task] Enter with evaluation task #{evaluation_task_id}"
    )

    with Session(engine, expire_on_commit=False) as session:
        evaluation_task = session.get(EvaluationTask, evaluation_task_id)
        if evaluation_task is None:
            logger.error(f"Evaluation task #{evaluation_task_id} is not found")
            return

        # get eval items
        eval_item_stmt = select(EvaluationTaskItem).where(
            EvaluationTaskItem.evaluation_task_id == evaluation_task_id
        )
        eval_item_list = session.exec(eval_item_stmt).all()
        logger.info(f"[add_evaluation_task] get {len(eval_item_list)} evaluation items")
        for eval_item in eval_item_list:
            logger.debug(type(eval_item))
            logger.debug(
                f"[add_evaluation_task] deal with evaluation item #{eval_item.id}"
            )
            add_evaluation_task_item.delay(eval_item.id)


@celery_app.task
def add_evaluation_task_item(evaluation_task_item_id: int):
    logger.info(
        f"Enter add_evaluation_task_item with evaluation item #{evaluation_task_item_id}"
    )

    with Session(engine, expire_on_commit=False) as session:
        evaluation_task_item = session.get(EvaluationTaskItem, evaluation_task_item_id)
        if evaluation_task_item is None:
            logger.error(f"Evaluation item #{evaluation_task_item_id} is not found")
            return
        if evaluation_task_item.status != EvaluationStatus.NOT_START:
            logger.error(
                f"Evaluation item #{evaluation_task_item_id} is not in not start state"
            )
            return
    messages = [{"role": "user", "content": evaluation_task_item.query}]

    try:
        if evaluation_task_item.response is None or evaluation_task_item.response == "":
            response, _ = generate_answer_by_autoflow(
                messages, evaluation_task_item.chat_engine
            )
            if response is None or response == "":
                raise Exception("Autoflow response is empty")

            response = response.replace("\n", "\\n").replace("\r", "\\r")

            logger.info(
                f"Got response from autoflow for evaluation item #{evaluation_task_item_id}, {response}"
            )
            evaluation_task_item.response = response
            logger.info(f"Successfully get response item #{evaluation_task_item_id}")

            with Session(engine, expire_on_commit=False) as session:
                session.merge(evaluation_task_item)
                session.commit()

        evaluate_task(evaluation_task_item)

    except Exception as e:
        logger.error(f"Failed to evaluate item #{evaluation_task_item_id}, error: {e}")
        evaluation_task_item.error_msg = traceback.format_exc()
        evaluation_task_item.status = EvaluationStatus.ERROR

        with Session(engine, expire_on_commit=False) as session:
            session.merge(evaluation_task_item)
            session.commit()


def evaluate_task(evaluation_task_item: EvaluationTaskItem):
    logger.info(f"Enter evaluate_task with evaluation item #{evaluation_task_item.id}")
    ragas_list = [
        {
            "user_input": evaluation_task_item.query,
            "reference": evaluation_task_item.reference,
            "response": evaluation_task_item.response,
        }
    ]
    logger.debug(f"Response data {evaluation_task_item.response}")

    ragas_dataset = EvaluationDataset.from_list(ragas_list)
    logger.debug(f"Dataset {ragas_dataset.to_pandas().head()}")

    evaluator_llm = LlamaIndexLLMWrapper(
        OpenAI(model="gpt-4o", api_key=settings.EVALUATION_OPENAI_API_KEY)
    )
    evaluator_embeddings = LlamaIndexEmbeddingsWrapper(
        OpenAIEmbedding(
            model="text-embedding-3-large", api_key=settings.EVALUATION_OPENAI_API_KEY
        )
    )

    metrics = [
        # LLMContextRecall(llm=evaluator_llm),  # retrieved_contexts required
        FactualCorrectness(llm=evaluator_llm),
        # Faithfulness(llm=evaluator_llm),  # retrieved_contexts required
        SemanticSimilarity(embeddings=evaluator_embeddings),
    ]

    try:
        eval_result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics,
            raise_exceptions=True,
            show_progress=False,
        )

        logger.debug(f"eval_result to_pandas")
        result_list = eval_result.to_pandas().to_dict(orient="records")
        logger.debug(f"result list {result_list}")
        if len(result_list) != 1:
            raise Exception(
                f"Item {evaluation_task_item.id} cannot get evaluation from ragas"
            )

        logger.debug(f"result {result_list[0]}")
        evaluation_task_item.factual_correctness = result_list[0][
            FactualCorrectness.name
        ]
        evaluation_task_item.semantic_similarity = result_list[0][
            SemanticSimilarity.name
        ]
        evaluation_task_item.status = EvaluationStatus.DONE

        logger.info(f"Result evaluation item #{evaluation_task_item}")
        with Session(engine, expire_on_commit=False) as session:
            session.merge(evaluation_task_item)
            session.commit()
    except Exception as e:
        logger.error(f"Failed to evaluate item #{evaluation_task_item.id}, error: {e}")
        evaluation_task_item.error_msg = traceback.format_exc()
        evaluation_task_item.status = EvaluationStatus.ERROR

        with Session(engine, expire_on_commit=False) as session:
            session.merge(evaluation_task_item)
            session.commit()


@retry(stop=stop_after_attempt(2), wait=wait_fixed(5))
def generate_answer_by_autoflow(messages: list, chat_engine: str) -> (str, str):
    api_key = settings.TIDB_AI_API_KEY.get_secret_value() if settings.TIDB_AI_API_KEY else None
    response = httpx.post(
        url=settings.TIDB_AI_CHAT_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Origin": "evaluation",
        },
        json={
            "messages": messages,
            "index": "default",
            "chat_engine": chat_engine,
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()
    data = response.json()

    trace_id = None
    if data.get("trace"):
        trace_url = data["trace"].get("langfuse_url")
        if trace_url:
            trace_id = parse_langfuse_trace_id_from_url(trace_url)

    answer = data["content"]
    return answer, trace_id


def parse_langfuse_trace_id_from_url(trace_url: str) -> str:
    # Example trace_url: https://us.cloud.langfuse.com/trace/87e7eb2e-b789-4b23-af60-fbcf0fd517a1
    return trace_url.split("/")[-1]
