import pandas as pd
from fastapi import APIRouter, status, HTTPException, Depends
from fastapi_pagination import Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlmodel import select, desc

from app.api.admin_routes.evaluation.models import (
    CreateEvaluationDataset,
    UpdateEvaluationDataset,
    ModifyEvaluationDatasetItem,
    ParamsWithKeyword,
)
from app.api.admin_routes.evaluation.tools import must_get, must_get_and_belong
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.file_storage import default_file_storage
from app.models import Upload, EvaluationDataset, EvaluationDatasetItem
from app.types import MimeTypes

router = APIRouter()


@router.post("/admin/evaluation/datasets")
def create_evaluation_dataset(
    evaluation_dataset: CreateEvaluationDataset,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> EvaluationDataset:
    """
    Create a dataset for a given question and chat engine.
    This API depends on the /admin/uploads API to upload the evaluation data.
    The evaluation data is expected to be a CSV file with the following columns:

    - query: The query to evaluate
    - reference: The expected response to the query

    You can add more columns to the CSV file, and the extra columns will adhere to the results.

    Args:
        evaluation_dataset.name: The name of the evaluation dataset.
        evaluation_dataset.upload_id: The ID of the uploaded CSV file of the evaluation dataset.

    Returns:
        True if the evaluation dataset is created successfully.
    """
    name = evaluation_dataset.name
    evaluation_data_list = []
    if evaluation_dataset.upload_id is not None:
        # If the evaluation_file_id is provided, validate the uploaded file
        evaluation_file_id = evaluation_dataset.upload_id
        upload = must_get_and_belong(session, Upload, evaluation_file_id, user.id)

        if upload.mime_type != MimeTypes.CSV:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file must be a CSV file.",
            )

        with default_file_storage.open(upload.path) as f:
            df = pd.read_csv(f)

            # check essential columns
            must_have_columns = ["query", "reference"]
            if not set(must_have_columns).issubset(df.columns):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"The uploaded file must have the following columns: {must_have_columns}",
                )

            eval_list = df.to_dict(orient="records")
            # create evaluation dataset items
            evaluation_data_list = [
                EvaluationDatasetItem(
                    query=item["query"],
                    reference=item["reference"],
                    retrieved_contexts=[],  # TODO: implement this after we can retrieve contexts
                    extra={k: item[k] for k in item if k not in must_have_columns},
                )
                for item in eval_list
            ]

    evaluation_dataset = EvaluationDataset(
        name=name,
        user_id=user.id,
        evaluation_data_list=evaluation_data_list,
    )

    session.add(evaluation_dataset)
    session.commit()
    session.refresh(evaluation_dataset)

    return evaluation_dataset


@router.delete("/admin/evaluation/datasets/{evaluation_dataset_id}")
def delete_evaluation_dataset(
    evaluation_dataset_id: int, session: SessionDep, user: CurrentSuperuserDep
) -> bool:
    evaluation_dataset = must_get_and_belong(
        session, EvaluationDataset, evaluation_dataset_id, user.id
    )

    session.delete(evaluation_dataset)
    session.commit()

    return True


@router.put("/admin/evaluation/datasets/{evaluation_dataset_id}")
def update_evaluation_dataset(
    evaluation_dataset_id: int,
    updated_evaluation_dataset: UpdateEvaluationDataset,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> EvaluationDataset:
    evaluation_dataset = must_get_and_belong(
        session, EvaluationDataset, evaluation_dataset_id, user.id
    )

    evaluation_dataset.name = updated_evaluation_dataset.name

    session.merge(evaluation_dataset)
    session.commit()
    session.refresh(evaluation_dataset)

    return evaluation_dataset


@router.get("/admin/evaluation/datasets")
def list_evaluation_dataset(
    session: SessionDep,
    user: CurrentSuperuserDep,
    params: ParamsWithKeyword = Depends(),
) -> Page[EvaluationDataset]:
    stmt = (
        select(EvaluationDataset)
        .where(EvaluationDataset.user_id == user.id)
        .order_by(desc(EvaluationDataset.id))
    )

    if params.keyword:
        stmt = stmt.where(EvaluationDataset.name.ilike(f"%{params.keyword}%"))

    return paginate(session, stmt, params)


@router.post("/admin/evaluation/dataset-items")
def create_evaluation_dataset_item(
    modify_evaluation_dataset_item: ModifyEvaluationDatasetItem,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> EvaluationDatasetItem:
    evaluation_dataset_item = EvaluationDatasetItem(
        query=modify_evaluation_dataset_item.query,
        reference=modify_evaluation_dataset_item.reference,
        retrieved_contexts=modify_evaluation_dataset_item.retrieved_contexts,
        extra=modify_evaluation_dataset_item.extra,
        evaluation_dataset_id=modify_evaluation_dataset_item.evaluation_dataset_id,
    )

    session.add(evaluation_dataset_item)
    session.commit()
    session.refresh(evaluation_dataset_item)

    return evaluation_dataset_item


@router.delete("/admin/evaluation/dataset-items/{evaluation_dataset_item_id}")
def delete_evaluation_dataset_item(
    evaluation_dataset_item_id: int, session: SessionDep, user: CurrentSuperuserDep
) -> bool:
    evaluation_dataset_item = must_get(
        session, EvaluationDataset, evaluation_dataset_item_id
    )

    session.delete(evaluation_dataset_item)
    session.commit()

    return True


@router.put("/admin/evaluation/dataset-items/{evaluation_dataset_item_id}")
def update_evaluation_dataset_item(
    evaluation_dataset_item_id: int,
    updated_evaluation_dataset_item: ModifyEvaluationDatasetItem,
    session: SessionDep,
    user: CurrentSuperuserDep,
) -> EvaluationDatasetItem:
    evaluation_dataset_item = must_get(
        session, EvaluationDatasetItem, evaluation_dataset_item_id
    )

    evaluation_dataset_item.query = updated_evaluation_dataset_item.query
    evaluation_dataset_item.reference = updated_evaluation_dataset_item.reference
    evaluation_dataset_item.retrieved_contexts = (
        updated_evaluation_dataset_item.retrieved_contexts
    )
    evaluation_dataset_item.extra = updated_evaluation_dataset_item.extra
    evaluation_dataset_item.evaluation_dataset_id = (
        updated_evaluation_dataset_item.evaluation_dataset_id
    )
    session.merge(evaluation_dataset_item)
    session.commit()
    session.refresh(evaluation_dataset_item)

    return evaluation_dataset_item


@router.get("/admin/evaluation/datasets/{evaluation_dataset_id}/dataset-items")
def list_evaluation_dataset_item(
    session: SessionDep,
    user: CurrentSuperuserDep,
    evaluation_dataset_id: int,
    params: ParamsWithKeyword = Depends(),
) -> Page[EvaluationDatasetItem]:
    stmt = (
        select(EvaluationDatasetItem)
        .where(EvaluationDatasetItem.evaluation_dataset_id == evaluation_dataset_id)
        .order_by(EvaluationDatasetItem.id)
    )

    if params.keyword:
        stmt = stmt.where(EvaluationDatasetItem.query.ilike(f"%{params.keyword}%"))
    return paginate(session, stmt, params)


@router.get("/admin/evaluation/dataset-items/{evaluation_dataset_item_id}")
def get_evaluation_dataset_item(
    session: SessionDep,
    user: CurrentSuperuserDep,
    evaluation_dataset_item_id: int,
) -> EvaluationDatasetItem:
    return must_get(session, EvaluationDatasetItem, evaluation_dataset_item_id)
