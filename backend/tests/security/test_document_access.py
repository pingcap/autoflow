from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.routes import document
from app.models import Chat, ChatEngine, ChatVisibility
from app.file_storage.local import LocalFileStorage


@pytest.fixture
def download(monkeypatch):
    chat = Chat(
        id=uuid4(), title="q", browser_id="owner", visibility=ChatVisibility.PRIVATE
    )
    chat.engine = ChatEngine(
        name="public", is_public=True, engine_options={"hide_sources": False}
    )
    session = Mock()
    session.exec.return_value.first.return_value = 10
    request = SimpleNamespace(state=SimpleNamespace(browser_id="owner"))
    monkeypatch.setattr(document.chat_repo, "must_get", Mock(return_value=chat))
    doc = Mock(source_uri="uploads/example.html", mime_type="text/html")
    get_doc = Mock(return_value=doc)
    monkeypatch.setattr(document.document_repo, "must_get", get_doc)
    storage = Mock()
    storage.exists.return_value = True
    storage.size.return_value = 4
    storage.open.return_value = BytesIO(b"test")
    monkeypatch.setattr(document, "get_file_storage", Mock(return_value=storage))
    return chat, request, session, get_doc, storage


@pytest.mark.parametrize(
    "denial",
    ["no-chat", "other-owner", "hidden", "private-engine", "uncited", "deleted"],
)
def test_download_denies_before_opening_or_looking_up_document(download, denial):
    chat, request, session, get_doc, storage = download
    chat_id = chat.id
    if denial == "no-chat":
        chat_id = None
    elif denial == "other-owner":
        request.state.browser_id = "visitor"
    elif denial == "hidden":
        chat.engine.engine_options = {"hide_sources": True}
    elif denial == "private-engine":
        chat.engine.is_public = False
    elif denial == "uncited":
        session.exec.return_value.first.return_value = None
    elif denial == "deleted":
        from datetime import datetime, UTC

        chat.deleted_at = datetime.now(UTC)
    with pytest.raises(HTTPException) as error:
        document.download_file(request, 42, session, None, chat_id)
    assert error.value.status_code == 404
    get_doc.assert_not_called()
    storage.open.assert_not_called()


@pytest.mark.parametrize("public", [False, True])
def test_cited_download_is_available_to_owner_or_explicit_public_share(
    download, public
):
    chat, request, session, get_doc, _ = download
    if public:
        chat.visibility = ChatVisibility.PUBLIC
        request.state.browser_id = "visitor"
    response = document.download_file(request, 42, session, None, chat.id)
    get_doc.assert_called_once_with(session, 42)
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, no-store"
    query = session.exec.call_args.args[0]
    params = query.compile().params
    assert chat.id in params.values()
    assert '{"id": 42}' in params.values()


def test_admin_download_remains_available_without_a_chat(download):
    _, request, session, _, _ = download
    response = document.download_file(
        request, 42, session, SimpleNamespace(is_superuser=True)
    )
    assert response.status_code == 200
    session.exec.assert_not_called()


def test_local_storage_rejects_escape_and_symlink_targets(monkeypatch, tmp_path):
    root = tmp_path / "storage"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("outside")
    (root / "link").symlink_to(outside)
    from app.file_storage.local import settings

    monkeypatch.setattr(settings, "LOCAL_FILE_STORAGE_PATH", str(root))
    storage = LocalFileStorage()
    for name in ["../secret", str(outside), "link", "."]:
        with pytest.raises(ValueError, match="Invalid storage path"):
            storage.open(name)
    storage.save("uploads/file.txt", BytesIO(b"allowed"))
    with storage.open("uploads/file.txt") as result:
        assert result.read() == b"allowed"
