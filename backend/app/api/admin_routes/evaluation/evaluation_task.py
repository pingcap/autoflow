import logging
from typing import Optional, List

import sqlmodel
from fastapi import APIRouter, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy import func, update
from sqlalchemy.orm import Session
from sqlmodel import select, case, desc

from app.api.admin_routes.evaluation.models import (
    CreateEvaluationTask,
    EvaluationTaskSummary,
    ParamsWithKeyword, EvaluationTaskOverview,
)
from app.api.admin_routes.evaluation.tools import must_get_and_belong
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.models import (
    EvaluationTask,
    EvaluationTaskItem,
    EvaluationStatus,
    EvaluationDataset,
)
from app.tasks.evaluate import add_evaluation_task

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/admin/evaluation/tasks")
def create_evaluation_task(
    evaluation_task: CreateEvaluationTask,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> Optional[EvaluationTask]:
    """
    Create an evaluation task from the evaluation dataset.

    Args:
        evaluation_task.name: The name of the evaluation task.
        evaluation_task.evaluation_dataset_id: The ID of the uploaded evaluation dataset.
        evaluation_task.chat_engine: The chat engine to evaluate the queries against. Default is "default".
        evaluation_task.run_size: The number of queries to evaluate. Default is None, which means all queries in the CSV file.

    Returns:
        True if the evaluation task is created successfully.
    """

    name = evaluation_task.name
    evaluation_dataset_id = evaluation_task.evaluation_dataset_id
    chat_engine = evaluation_task.chat_engine
    run_size = evaluation_task.run_size

    dataset = must_get_and_belong(
        session, EvaluationDataset, evaluation_dataset_id, user.id
    )

    if run_size is not None and run_size < len(dataset.evaluation_data_list):
        dataset.evaluation_data_list = dataset.evaluation_data_list[:run_size]

    # create evaluation items
    # caveat: Do the deep copy on purpose to avoid the side effect of the original dataset modification
    evaluation_task_items = [
        EvaluationTaskItem(
            status=EvaluationStatus.NOT_START,
            chat_engine=chat_engine,
            query=item.query,
            reference=item.reference,
            retrieved_contexts=item.retrieved_contexts,
            extra=item.extra,
        )
        for item in dataset.evaluation_data_list
    ]

    evaluation_task = EvaluationTask(
        name=name,
        user_id=user.id,
        evaluation_task_items=evaluation_task_items,
        dataset_id=evaluation_dataset_id,
    )

    session.add(evaluation_task)
    session.commit()
    session.refresh(evaluation_task)

    add_evaluation_task.delay(evaluation_task.id)

    return evaluation_task


@router.delete("/admin/evaluation/tasks/{evaluation_task_id}")
def cancel_evaluation_task(
    evaluation_task_id: int, session: SessionDep, user: CurrentSuperuserDep
) -> Optional[bool]:
    must_get_and_belong(session, EvaluationTask, evaluation_task_id, user.id)

    session.exec(
        update(EvaluationTaskItem)
        .where(EvaluationTaskItem.evaluation_task_id == evaluation_task_id)
        .values(status=EvaluationStatus.CANCEL)
    )
    session.commit()

    return True


@router.get("/admin/evaluation/tasks/{evaluation_task_id}")
def list_evaluation_task(
    session: SessionDep,
    user: CurrentSuperuserDep,
    evaluation_task_id: int,
) -> EvaluationTask:
    return must_get_and_belong(session, EvaluationTask, evaluation_task_id, user.id)


@router.get("/admin/evaluation/tasks/{evaluation_task_id}/summary")
def get_evaluation_task_summary(
        evaluation_task_id: int, session: SessionDep, user: CurrentSuperuserDep
) -> EvaluationTaskSummary:
    task = must_get_and_belong(session, EvaluationTask, evaluation_task_id, user.id)
    return get_evaluation_task_summary(task, session)


@router.get("/admin/evaluation/tasks")
def list_evaluation_task(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: ParamsWithKeyword = Depends(),
) -> Page[EvaluationTaskSummary]:
    stmt = (
        select(EvaluationTask)
        .where(EvaluationTask.user_id == user.id)
        .order_by(desc(EvaluationTask.id))
    )
    if params.keyword:
        stmt = stmt.where(EvaluationTask.name.ilike(f"%{params.keyword}%"))

    task_page: Page[EvaluationTask] = paginate(session, stmt, params)
    summaries: List[EvaluationTaskSummary] = []
    for task in task_page.items:
        summaries.append(get_evaluation_task_summary(task, session))

    return Page[EvaluationTaskSummary](
        items=summaries,
        total=task_page.total,
        page=task_page.page,
        size=task_page.size,
        pages=task_page.pages
    )


@router.get("/admin/evaluation/tasks/{evaluation_task_id}/items")
def list_evaluation_task_items(
    evaluation_task_id: int,
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: ParamsWithKeyword = Depends(),
) -> Page[EvaluationTaskItem]:
    must_get_and_belong(session, EvaluationTask, evaluation_task_id, user.id)
    stmt = select(EvaluationTaskItem).where(
        EvaluationTaskItem.evaluation_task_id == evaluation_task_id
    )
    if params.keyword:
        stmt = stmt.where(
            sqlmodel.or_(
                EvaluationTaskItem.query.ilike(f"%{params.keyword}%"),
                EvaluationTaskItem.reference.ilike(f"%{params.keyword}%"),
            )
        )

    return paginate(session, stmt, params)


def get_evaluation_task_summary(
    evaluation_task: EvaluationTask, session: Session
) -> EvaluationTaskSummary:
    status_counts = (
        session.query(
            func.count(
                case(
                    (EvaluationTaskItem.status == EvaluationStatus.NOT_START, 1),
                    else_=None,
                )
            ).label("not_start"),
            func.count(
                case(
                    (EvaluationTaskItem.status == EvaluationStatus.EVALUATING, 1),
                    else_=None,
                )
            ).label("evaluating"),
            func.count(
                case(
                    (EvaluationTaskItem.status == EvaluationStatus.DONE, 1), else_=None
                )
            ).label("done"),
            func.count(
                case(
                    (EvaluationTaskItem.status == EvaluationStatus.ERROR, 1), else_=None
                )
            ).label("error"),
            func.count(
                case(
                    (EvaluationTaskItem.status == EvaluationStatus.CANCEL, 1),
                    else_=None,
                )
            ).label("cancel"),
        )
        .filter(EvaluationTaskItem.evaluation_task_id == evaluation_task.id)
        .one()
    )

    stats = {}
    if status_counts.not_start == 0 and status_counts.evaluating == 0:
        stats_tuple = session.query(
            func.avg(EvaluationTaskItem.factual_correctness).label(
                "avg_factual_correctness"
            ),
            func.avg(EvaluationTaskItem.semantic_similarity).label(
                "avg_semantic_similarity"
            ),
            func.min(EvaluationTaskItem.factual_correctness).label(
                "min_factual_correctness"
            ),
            func.min(EvaluationTaskItem.semantic_similarity).label(
                "min_semantic_similarity"
            ),
            func.max(EvaluationTaskItem.factual_correctness).label(
                "max_factual_correctness"
            ),
            func.max(EvaluationTaskItem.semantic_similarity).label(
                "max_semantic_similarity"
            ),
            func.stddev(EvaluationTaskItem.factual_correctness).label(
                "std_factual_correctness"
            ),
            func.stddev(EvaluationTaskItem.semantic_similarity).label(
                "std_semantic_similarity"
            ),
        ).filter(
            EvaluationTaskItem.evaluation_task_id == evaluation_task.id,
            EvaluationTaskItem.status == EvaluationStatus.DONE,
            EvaluationTaskItem.factual_correctness.isnot(None),
            EvaluationTaskItem.semantic_similarity.isnot(None),
        ).one()

        stats = dict(stats_tuple._mapping)
        logger.info(stats)

    return EvaluationTaskSummary(
        summary=EvaluationTaskOverview(
            not_start=status_counts.not_start,
            succeed=status_counts.done,
            errored=status_counts.error,
            progressing=status_counts.evaluating,
            cancel=status_counts.cancel,
            avg_factual_correctness=stats.get("avg_factual_correctness", 0),
            avg_semantic_similarity=stats.get("avg_semantic_similarity", 0),
            min_factual_correctness=stats.get("min_factual_correctness", 0),
            min_semantic_similarity=stats.get("min_semantic_similarity", 0),
            max_factual_correctness=stats.get("max_factual_correctness", 0),
            max_semantic_similarity=stats.get("max_semantic_similarity", 0),
            std_factual_correctness=stats.get("std_factual_correctness", 0),
            std_semantic_similarity=stats.get("std_semantic_similarity", 0),
        ),
        **evaluation_task.model_dump(),
    )
