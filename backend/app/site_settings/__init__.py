import time
import threading
import logging
from sqlmodel import Session, select

from app.models import SiteSetting as DBSiteSetting
from app.core.db import engine
from app.site_settings.default import default_settings
from app.site_settings.types import SettingValue, SettingType

logger = logging.getLogger(__name__)


def get_settings_from_db(session: Session):
    logger.debug("Getting settings from the database.")
    stmt = select(DBSiteSetting.name, DBSiteSetting.value)
    results = session.exec(stmt)
    return {name: value for name, value in results}


def get_db_last_updated_at(session: Session):
    logger.debug("Getting the last updated_at timestamp from the database.")
    stmt = (
        select(DBSiteSetting.updated_at)
        .order_by(DBSiteSetting.updated_at.desc())
        .limit(1)
    )
    result = session.exec(stmt).first()
    return result.timestamp() if result else 0


type_mapping = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
}


class SiteSettingProxy:
    __db_cache: dict = {}
    __last_updated_at_ts: float = 0
    __last_checked_at_ts: float = 0
    __mutex = threading.Lock()

    def update_db_cache(self, force_check=False):
        # Check if we need to update the cache every 6 seconds,
        # so it means settings will not be updated in real-time
        # which is acceptable for this project.
        # If we need real-time updates in the future, we can use
        # a message queue or a pub/sub system to notify the app.
        now = time.time()
        if force_check or (now - self.__last_checked_at_ts > 6):
            self.__last_checked_at_ts = now
            with Session(engine) as session:
                last_updated_at_ts = get_db_last_updated_at(session)

                if last_updated_at_ts > self.__last_updated_at_ts:
                    with self.__mutex:
                        if last_updated_at_ts > self.__last_updated_at_ts:
                            self.__db_cache = get_settings_from_db(session)
                            self.__last_updated_at_ts = last_updated_at_ts

    def get_db_cache(self) -> dict:
        # Should we use a lock here?
        with self.__mutex:
            return self.__db_cache

    def __getattr__(self, name: str) -> SettingType:
        return self.get_setting(name)

    def get_setting(self, name: str) -> SettingType:
        if hasattr(default_settings, name):
            default_setting = getattr(default_settings, name)
            self.update_db_cache()
            db_value = self.__db_cache.get(name)
            return db_value if db_value is not None else default_setting.default
        else:
            raise AttributeError(f"Setting {name} does not exist.")

    def get_all_settings(
        self, force_check_db_cache: bool = False
    ) -> dict[str, SettingValue]:
        self.update_db_cache(force_check_db_cache)

        result = {}
        for _, settings in default_settings.setting_groups.items():
            for default_setting in settings:
                db_value = self.__db_cache.get(default_setting.name)
                result[default_setting.name] = SettingValue(
                    name=default_setting.name,
                    default=default_setting.default,
                    value=db_value if db_value is not None else default_setting.default,
                    data_type=default_setting.data_type,
                    description=default_setting.description,
                    group=default_setting.group,
                    client=default_setting.client,
                )
        return result

    def get_client_settings(self) -> dict:
        # Retrieve all client settings utilized in the frontend.
        # These settings determine the behavior of the frontend and are accessible to all users.
        settings = self.get_all_settings()
        return {k: s.value for k, s in settings.items() if s.client}

    def setting_exists(self, name: str) -> bool:
        return hasattr(default_settings, name)

    def update_setting(self, session: Session, name: str, value: SettingType):
        if not self.setting_exists(name):
            raise AttributeError(f"Setting {name} does not exist.")

        _default_setting: SettingValue = getattr(default_settings, name)
        if not isinstance(value, type_mapping[_default_setting.data_type]):
            raise ValueError(f"{name} must be of type `{_default_setting.data_type}`.")

        db_setting_obj = session.exec(
            select(DBSiteSetting).filter(DBSiteSetting.name == name)
        ).first()
        if db_setting_obj:
            db_setting_obj.value = value
        else:
            db_setting_obj = DBSiteSetting(
                name=name, value=value, data_type=_default_setting.data_type
            )
            session.add(db_setting_obj)
        session.commit()

        self.update_db_cache(force_check=True)


SiteSetting = SiteSettingProxy()


__all__ = ["SiteSetting", "SettingValue", "SettingType"]
