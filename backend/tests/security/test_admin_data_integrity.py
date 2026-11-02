from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.admin_routes.evaluation import evaluation_task as tasks
from app.api.admin_routes.evaluation.models import CreateEvaluationTask
from app.models import EmbeddingModel
from app.repositories.embedding_model import embedding_model_repo
from app.site_settings import SiteSettingProxy


def test_subset_evaluation_leaves_source_dataset_relationship_and_json_intact(
    monkeypatch,
):
    items = [
        SimpleNamespace(
            query=str(i),
            reference="ref",
            retrieved_contexts=["context"],
            extra={"tags": []},
        )
        for i in range(3)
    ]
    dataset = SimpleNamespace(evaluation_data_list=items)
    monkeypatch.setattr(tasks, "must_get", Mock(return_value=dataset))
    monkeypatch.setattr(tasks.add_evaluation_task, "delay", Mock())
    result = tasks.create_evaluation_task(
        CreateEvaluationTask(name="subset", evaluation_dataset_id=1, run_size=1),
        Mock(),
        SimpleNamespace(id=uuid4()),
    )
    assert dataset.evaluation_data_list is items
    assert len(dataset.evaluation_data_list) == 3
    assert len(result.evaluation_task_items) == 1
    result.evaluation_task_items[0].extra["tags"].append("changed")
    result.evaluation_task_items[0].retrieved_contexts.append("changed")
    assert items[0].extra == {"tags": []}
    assert items[0].retrieved_contexts == ["context"]
    with pytest.raises(ValidationError):
        CreateEvaluationTask(name="bad", evaluation_dataset_id=1, run_size=-1)


def test_referenced_embedding_model_cannot_be_deleted_or_detached():
    session = Mock()
    session.exec.return_value.rowcount = 0
    with pytest.raises(HTTPException) as error:
        embedding_model_repo.delete(session, EmbeddingModel(id=7))
    assert error.value.status_code == 409
    session.delete.assert_not_called()
    session.commit.assert_not_called()
    statement = session.exec.call_args.args[0]
    sql = str(statement)
    assert sql.startswith("DELETE FROM embedding_models")
    assert "NOT (EXISTS" in sql
    assert "knowledge_bases.embedding_model_id" in sql


def test_unused_embedding_model_can_be_deleted():
    session = Mock()
    session.exec.return_value.rowcount = 1
    embedding_model_repo.delete(session, EmbeddingModel(id=7))
    session.commit.assert_called_once()


@pytest.mark.parametrize(
    "value",
    ["G-ABC123');alert(1)//", "</script><script>alert(1)</script>", "G-ABC123\n"],
)
def test_unsafe_analytics_value_is_rejected_before_database_write(value):
    session = Mock()
    with pytest.raises(ValueError, match="measurement ID"):
        SiteSettingProxy().update_setting(session, "ga_id", value)
    session.exec.assert_not_called()


def test_existing_unsafe_analytics_setting_is_not_published(monkeypatch):
    settings = SiteSettingProxy()
    monkeypatch.setattr(
        settings,
        "get_all_settings",
        lambda: {
            "ga_id": SimpleNamespace(
                value="</script><script>alert(1)</script>", client=True
            )
        },
    )
    assert settings.get_client_settings()["ga_id"] == ""
