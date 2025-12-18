import os
from dotenv import load_dotenv

load_dotenv()


## Implementing the metadata adapter (Duckdb) behind a stable interface

## It acts like a facotry to decide which metadata backend to use.
def get_store():
    backend = os.getenv("METADATA_BACKEND", "duckdb").lower()
    if backend == "duckdb":
        from adapters.duckdb_store import DuckDBStore
        return DuckDBStore(os.getenv("DUCKDB_PATH", "./data/metadata.duckdb"))
    raise ValueError(f"Unknown METADATA_BACKEND: {backend}")