from typing import List
from sqlmodel import SQLModel

from autoflow.storage.table import Table


class Database:
    def create_table(self, table_name: str, schema: SQLModel) -> Table:
        raise NotImplementedError()

    def list_tables(self, schema: SQLModel) -> List[Table]:
        raise NotImplementedError()


database = Database()
