import asyncio
from pathlib import Path
import sys
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from colorama import Fore, Style
import click

from app.core.db import get_db_async_session_context
from app.models import User, ChatEngine


def require_password(value: str | None) -> str:
    if value is None and sys.stdin.isatty():
        value = click.prompt(
            "Administrator password", hide_input=True, confirmation_prompt=True
        )
    if not value or not 16 <= len(value) <= 1024:
        raise click.ClickException(
            "Supply a 16–1024 character password using --password-file or an interactive terminal"
        )
    return value


def read_private_password(path: Path) -> str:
    import stat

    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 4096:
        raise click.ClickException(
            "Password file must be a private regular file (chmod 600)"
        )
    return require_password(path.read_text().rstrip("\r\n"))


async def ensure_admin_user(
    session: AsyncSession, email: str | None = None, password: str | None = None
) -> None:
    result = await session.exec(select(User).where(User.is_superuser == True))
    user = result.first()
    if not user:
        from app.auth.users import create_user

        admin_email = email or "admin@example.com"
        admin_password = require_password(password)
        user = await create_user(
            session,
            email=admin_email,
            password=admin_password,
            is_active=True,
            is_verified=True,
            is_superuser=True,
        )
        print(
            Fore.GREEN
            + "Admin user created. Use the supplied password to sign in."
            + Style.RESET_ALL
        )
    else:
        print(Fore.YELLOW + "Admin user already exists, skipping...")


async def reset_admin_password(
    session: AsyncSession, new_password: str | None = None
) -> None:
    result = await session.exec(select(User).where(User.is_superuser == True))
    user = result.first()
    if not user:
        print(Fore.YELLOW + "Admin user does not exist, skipping reset password...")
    else:
        from app.auth.users import update_user_password

        admin_password = require_password(new_password)
        await update_user_password(
            session,
            user_id=user.id,
            new_password=admin_password,
        )
        print(
            Fore.GREEN
            + "Admin password reset; previous sessions and API keys revoked."
            + Style.RESET_ALL
        )


async def ensure_default_chat_engine(session: AsyncSession) -> None:
    result = await session.scalar(func.count(ChatEngine.id))
    if result == 0:
        from app.rag.chat.config import ChatEngineConfig

        chat_engine = ChatEngine(
            name="default",
            engine_options=ChatEngineConfig().model_dump(),
            is_default=True,
        )
        session.add(chat_engine)
        await session.commit()
        print("Default chat engine created.")
    else:
        print(Fore.YELLOW + "Default chat engine already exists, skipping...")


async def bootstrap(
    email: str | None = None, password: str | None = None, reset_password: bool = False
) -> None:
    async with get_db_async_session_context() as session:
        await ensure_admin_user(session, email, password)
        await ensure_default_chat_engine(session)
        if reset_password:
            await reset_admin_password(session, password)


@click.command()
@click.option(
    "--email", default=None, help="Admin user email, default=admin@example.com"
)
@click.option(
    "--password-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read the administrator password from a private file (never print it)",
)
@click.option("--reset-password", "-r", is_flag=True, help="Reset admin user password.")
def main(email: str | None, password_file: Path | None, reset_password: bool):
    """Bootstrap the application with optional admin credentials."""
    print(Fore.GREEN + "Bootstrapping the application..." + Style.RESET_ALL)
    password = read_private_password(password_file) if password_file else None
    asyncio.run(bootstrap(email, password, reset_password))
    print(Fore.GREEN + "Bootstrapping completed." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
