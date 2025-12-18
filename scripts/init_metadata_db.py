from adapters.metadata_store import get_store
### Initialize the DuckDB metadata database schema using the schema file
if __name__ == "__main__":
    store = get_store()
    store.init_schema()
    print(" DuckDB metadata schema initialized.")