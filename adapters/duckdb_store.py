from __future__ import annotations
import duckdb
import pandas as pd
from typing import Iterable

### Implements the actual metadata opetations for DuckDB 

class DuckDBStore:
    def __init__(self, path: str):
        self.path = path
        self.con = duckdb.connect(path)

    def init_schema(self, schema_sql_path: str = "sql/schema_duckdb.sql") -> None:
        with open(schema_sql_path, "r", encoding="utf-8") as f:
            self.con.execute(f.read())

    def upsert_profiles(self, df: pd.DataFrame) -> None:
        # Simple first version: replace by id (DuckDB doesn't have full Postgres-style UPSERT everywhere)
        # Strategy: write temp then merge.
        self.con.register("incoming", df)

        self.con.execute("""
        CREATE TEMP TABLE incoming_tmp AS SELECT * FROM incoming;
        """)
        self.con.execute("""
        DELETE FROM profiles
        WHERE id IN (SELECT id FROM incoming_tmp);
        """)
        self.con.execute("""
        INSERT INTO profiles SELECT * FROM incoming_tmp;
        """)

    def get_metadata(self, ids: Iterable[str]) -> pd.DataFrame:
        # With a list of ids, return all matching metadata rows
        ids = list(ids)
        if not ids:
            return pd.DataFrame()

        # DuckDB parameterization: easiest is VALUES list
        placeholders = ",".join(["(?)"] * len(ids))
        query = f"""
        SELECT * FROM profiles
        WHERE id IN (SELECT * FROM (VALUES {placeholders}) AS t(id))
        """
        return self.con.execute(query, ids).df()

    def get_one(self, id_: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM profiles WHERE id = ?", [id_]).df()