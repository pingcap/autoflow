import asyncio
from unittest.mock import AsyncMock, Mock

import click
import pytest

import bootstrap
from app.auth import users
from app.models import User


def test_bootstrap_and_reset_never_print_the_supplied_password(monkeypatch, capsys):
    password = "fixture-admin-password-do-not-log"
    session = Mock()
    session.exec = AsyncMock(return_value=Mock(first=lambda: None))
    create = AsyncMock(
        return_value=User(email="admin@example.test", hashed_password="hash")
    )
    monkeypatch.setattr(users, "create_user", create)
    asyncio.run(bootstrap.ensure_admin_user(session, password=password))
    assert create.call_args.kwargs["password"] == password
    user = User(email="admin@example.test", hashed_password="hash")
    session.exec = AsyncMock(return_value=Mock(first=lambda: user))
    update = AsyncMock(return_value=user)
    monkeypatch.setattr(users, "update_user_password", update)
    asyncio.run(bootstrap.reset_admin_password(session, password))
    assert update.call_args.kwargs["new_password"] == password
    assert password not in capsys.readouterr().out


def test_noninteractive_bootstrap_requires_an_explicit_password(monkeypatch):
    monkeypatch.setattr(bootstrap.sys.stdin, "isatty", lambda: False)
    with pytest.raises(click.ClickException, match="password-file"):
        bootstrap.require_password(None)
