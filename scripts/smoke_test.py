from __future__ import annotations

import os
from dotenv import load_dotenv

from adapters.milvus_adapter import milvus_search, fetch_vector_by_id
from adapters.metadata_store import get_store

load_dotenv()


def main():
    # You must set a real existing ID in your .env for this to work
    query_id = os.getenv("SMOKE_QUERY_ID")
    dataset = os.getenv("SMOKE_DATASET", "orf")
    k = int(os.getenv("SMOKE_K", "10"))

    if not query_id:
        raise SystemExit(
            "Set SMOKE_QUERY_ID in your .env to a real profile ID, e.g.\n"
            "SMOKE_QUERY_ID=...\nSMOKE_DATASET=orf\nSMOKE_K=10"
        )

    print(f"🔎 Smoke test: dataset={dataset}, query_id={query_id}, k={k}")

    # 1) Fetch query vector from Milvus
    qvec = fetch_vector_by_id(dataset=dataset, id_=query_id)

    # 2) Search Milvus for neighbors
    hits = milvus_search(dataset=dataset, query_vector=qvec, k=k)
    ids = [hid for hid, _ in hits]
    print("Top IDs:", ids[:5])

    # 3) Fetch metadata for those IDs from DuckDB
    store = get_store()
    meta = store.get_metadata(ids)

    print("\n✅ Metadata rows fetched:", len(meta))
    print(meta.head(5))


if __name__ == "__main__":
    main()