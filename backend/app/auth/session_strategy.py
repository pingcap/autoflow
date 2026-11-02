from datetime import datetime, timedelta, UTC

from fastapi_users import exceptions
from fastapi_users.authentication.strategy import DatabaseStrategy


class VersionedDatabaseStrategy(DatabaseStrategy):
    def _create_access_token_dict(self, user):
        return {
            **super()._create_access_token_dict(user),
            "auth_version": user.auth_version,
        }

    async def read_token(self, token, user_manager):
        if not token or len(token) > 43:
            return None
        max_age = (
            datetime.now(UTC) - timedelta(seconds=self.lifetime_seconds)
            if self.lifetime_seconds
            else None
        )
        access_token = await self.database.get_by_token(token, max_age)
        if access_token is None:
            return None
        try:
            user = await user_manager.get(user_manager.parse_id(access_token.user_id))
        except (exceptions.UserNotExists, exceptions.InvalidID):
            return None
        return user if user.auth_version == access_token.auth_version else None
