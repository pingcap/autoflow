import logging
from typing import List, Optional

import sqlalchemy
from pydantic import PrivateAttr
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from autoflow.storage.tidb.table import Table, TableModel

logger = logging.getLogger(__name__)


class TiDBClient:
    _db_engine: Engine = PrivateAttr()

    def __init__(self, db_engine: Engine):
        self._db_engine = db_engine
        self._inspector = sqlalchemy.inspect(self._db_engine)

    @classmethod
    def connect(cls, database_url: str) -> "TiDBClient":
        db_engine = create_engine(database_url)
        return cls(db_engine)

    # Notice: Since the Vector type is not in the type support list of mysql dialect, using the reflection API will cause an error.
    # https://github.com/sqlalchemy/sqlalchemy/blob/d6f11d9030b325d5afabf87869a6e3542edda54b/lib/sqlalchemy/dialects/mysql/base.py#L1199
    # def _load_table_metadata(self, table_names: Optional[List[str]] = None):
    #     if not table_names:
    #         Base.metadata.reflect(bind=self._db_engine)
    #     else:
    #         Base.metadata.reflect(bind=self._db_engine, only=table_names, extend_existing=True)

    @property
    def db_engine(self) -> Engine:
        return self._db_engine

    def create_table(
        self,
        *,
        schema: Optional[TableModel] = None,
    ) -> Table:
        table = Table(schema=schema, db_engine=self._db_engine)
        return table

    def open_table(self, schema: TableModel) -> Table:
        return Table(
            schema=schema,
            db_engine=self._db_engine,
        )

    def table_names(self) -> List[str]:
        return self._inspector.get_table_names()

    def has_table(self, table_name: str) -> bool:
        return self._inspector.has_table(table_name)

    def drop_table(self, table_name: str):
        return self.execute(f"DROP TABLE IF EXISTS {table_name}")

    def execute(self, sql: str, params: Optional[dict] = None) -> dict:
        """
        Execute an arbitrary SQL command and return execution status and result.

        This method can handle both DML (Data Manipulation Language) commands such as INSERT, UPDATE, DELETE,
        and DQL (Data Query Language) commands like SELECT. It returns a structured dictionary indicating
        the execution success status, result (for SELECT queries or affected rows count for DML), and any
        error message if the execution failed.

        Args:
            sql (str): The SQL command to execute.
            params (Optional[dict]): Parameters to bind to the SQL command, if any.

        Returns:
            dict: A dictionary containing 'success': boolean indicating if the execution was successful,
                'result': fetched results for SELECT or affected rows count for other statements,
                and 'error': error message if execution failed.

        Examples:
            - Creating a table:
            execute("CREATE TABLE users (id INT, username VARCHAR(50), email VARCHAR(50))")
            This would return: {'success': True, 'result': 0, 'error': None}

            - Executing a SELECT query:
            execute("SELECT * FROM users WHERE username = :username", {"username": "john_doe"})
            This would return: {'success': True, 'result': [(user data)], 'error': None}

            - Inserting data into a table:
            execute(
                "INSERT INTO users (username, email) VALUES (:username, :email)",
                {"username": "new_user", "email": "new_user@example.com"}
            )
            This would return: {'success': True, 'result': 1, 'error': None} if one row was affected.

            - Handling an error (e.g., table does not exist):
            execute("SELECT * FROM non_existing_table")
            This might return: {'success': False, 'result': None, 'error': '(Error message)'}
        """
        try:
            with Session(self._db_engine) as session, session.begin():
                result = session.execute(sqlalchemy.text(sql), params)
                session.commit()  # Ensure changes are committed for non-SELECT statements.
                if sql.strip().lower().startswith("select"):
                    return {"success": True, "result": result.fetchall(), "error": None}
                else:
                    return {"success": True, "result": result.rowcount, "error": None}
        except Exception as e:
            # Log the error or handle it as needed
            logger.error(f"SQL execution error: {str(e)}")
            return {"success": False, "result": None, "error": str(e)}

    def disconnect(self) -> None:
        self._db_engine.dispose()
