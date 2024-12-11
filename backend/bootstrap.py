import secrets
import asyncio
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from colorama import Fore, Style
import click

from app.core.db import get_db_async_session_context
from app.models import User, ChatEngine


async def ensure_admin_user(session: AsyncSession, email: str | None = None, password: str | None = None) -> None:
    result = await session.exec(select(User).where(User.is_superuser == True))
    user = result.first()
    if not user:
        from app.auth.users import create_user

        admin_email = email or "admin@example.com"
        admin_password = password or secrets.token_urlsafe(16)
        user = await create_user(
            session,
            email=admin_email,
            password=admin_password,
            is_active=True,
            is_verified=True,
            is_superuser=True,
        )
        print(Fore.RED + "\n" + "!" * 80)
        print(
            Fore.RED + "[IMPORTANT] Admin user created with email: "
            f"{admin_email} and password: {admin_password}"
        )
        print(Fore.RED + "!" * 80 + "\n" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "Admin user already exists, skipping...")


async def ensure_default_chat_engine(session: AsyncSession) -> None:
    result = await session.scalar(func.count(ChatEngine.id))
    if result == 0:
        from app.rag.chat_config import ChatEngineConfig

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


async def bootstrap(email: str | None = None, password: str | None = None) -> None:
    async with get_db_async_session_context() as session:
        await ensure_admin_user(session, email, password)
        await ensure_default_chat_engine(session)


@click.command()
@click.option("--email", default=None, help="Admin user email, default=admin@example.com")
@click.option("--password", default=None, help="Admin user password, default=random generated")
def main(email: str | None, password: str | None):
    """Bootstrap the application with optional admin credentials."""
    print(Fore.GREEN + "Bootstrapping the application..." + Style.RESET_ALL)
    asyncio.run(bootstrap(email, password))
    print(Fore.GREEN + "Bootstrapping completed." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
