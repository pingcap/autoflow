import sqlalchemy
from pytidb.utils import build_tidb_dsn
from sqlalchemy import Engine

from autoflow.configs.db import DatabaseConfig


def get_db_engine_from_config(db_config: DatabaseConfig) -> Engine:
    if db_config.database_url is not None:
        database_url = db_config.database_url
    else:
        database_url = str(
            build_tidb_dsn(
                host=db_config.host,
                port=db_config.port,
                username=db_config.username,
                password=db_config.password,
                database=db_config.database,
                enable_ssl=db_config.enable_ssl,
            )
        )

    # Notice:
    # In order to save resource consumption, the tidb serverless cluster will "pause" automatically if there
    # are no active connections for more than 5 minutes, it will close all connections on the server side,
    # so we also need to recycle the connections from the connection pool on the client side.
    db_engine = sqlalchemy.create_engine(
        database_url,
        pool_size=20,
        max_overflow=40,
        pool_recycle=300,
        pool_pre_ping=True,
    )

    return db_engine
