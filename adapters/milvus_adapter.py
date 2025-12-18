from __future__ import annotations

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility

load_dotenv()

# ---------- Config helpers ----------

def _milvus_host() -> str:
    return os.getenv("MILVUS_HOST", "localhost")

def _milvus_port() -> str:
    return os.getenv("MILVUS_PORT", "19530")

def _vector_field() -> str:
    return os.getenv("MILVUS_VECTOR_FIELD", "vector")

def _id_field() -> str:
    return os.getenv("MILVUS_ID_FIELD", "id")

def _collection_name(dataset: str) -> str:
    ds = dataset.lower()
    if ds == "orf":
        return os.getenv("MILVUS_ORF_COLLECTION", "orf_profiles")
    if ds == "crispr":
        return os.getenv("MILVUS_CRISPR_COLLECTION", "crispr_profiles")
    if ds == "compound":
        return os.getenv("MILVUS_COMPOUND_COLLECTION", "compound_profiles")
    raise ValueError(f"Unknown dataset: {dataset!r}. Expected one of: orf, crispr, compound.")


# ---------- Connection management ----------

_CONNECTED = False

def connect(alias: str = "default") -> None:
    """
    Connects to Milvus.
    If MILVUS_MODE=lite, uses embedded Milvus Lite.
    Otherwise connects to a Milvus server using MILVUS_HOST/MILVUS_PORT.
    """
    global _CONNECTED
    if _CONNECTED:
        return

    mode = os.getenv("MILVUS_MODE", "server").lower()
    
    if mode == "lite":
        # Milvus Lite: connecting with a local .db uri starts embedded Milvus automatically
        db_path = os.getenv("MILVUS_LITE_PATH", "./data/milvus_lite.db")
        connections.connect(alias=alias, uri=db_path)
    else:
        connections.connect(alias=alias, host=_milvus_host(), port=_milvus_port())

    _CONNECTED = True


def get_collection(dataset: str) -> Collection:
    """
    Returns a loaded pymilvus Collection object for the given dataset.
    """
    connect()
    name = _collection_name(dataset)
    if not utility.has_collection(name):
        raise RuntimeError(f"Milvus collection not found: {name!r} (dataset={dataset!r})")

    col = Collection(name)
    # Load into memory for search; if already loaded this is cheap.
    col.load()
    return col


# ---------- Core operations ----------

def milvus_search(
    *,
    dataset: str,
    query_vector: np.ndarray,
    k: int,
    metric_type: str = "L2",
    nprobe: int = 16,
    expr: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Runs vector similarity search in a dataset's collection.
    Returns list of (id, distance).
    """
    col = get_collection(dataset)

    q = np.asarray(query_vector, dtype=np.float32)
    if q.ndim != 1:
        raise ValueError(f"query_vector must be 1D. Got shape={q.shape}")

    search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}

    results = col.search(
        data=[q.tolist()],
        anns_field=_vector_field(),
        param=search_params,
        limit=int(k),
        expr=expr,
        output_fields=[_id_field()],
    )

    hits = []
    for hit in results[0]:
        # Depending on Milvus schema, primary key may be in hit.id or hit.entity
        # We'll try both safely.
        pk = None
        try:
            pk = hit.entity.get(_id_field())
        except Exception:
            pk = None

        if pk is None:
            # fallback: hit.id sometimes is the primary key
            pk = str(hit.id)

        hits.append((str(pk), float(hit.distance)))

    return hits


def fetch_vectors(
    *,
    dataset: str,
    ids: List[str],
) -> np.ndarray:
    """
    Fetches vectors for the given IDs from Milvus.
    Returns array shape (len(ids), dim) aligned to the input ID order.

    NOTE: This assumes the collection stores a single vector field named by MILVUS_VECTOR_FIELD.
    """
    if not ids:
        return np.empty((0, 0), dtype=np.float32)

    col = get_collection(dataset)

    # Query in batches to avoid very long expressions
    out_vectors: Dict[str, np.ndarray] = {}

    batch_size = 200
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        expr = f'{_id_field()} in [{",".join([repr(x) for x in batch])}]'

        rows = col.query(
            expr=expr,
            output_fields=[_id_field(), _vector_field()],
        )

        for row in rows:
            rid = str(row[_id_field()])
            vec = np.asarray(row[_vector_field()], dtype=np.float32)
            out_vectors[rid] = vec

    # Reconstruct in input order; error if missing
    missing = [rid for rid in ids if rid not in out_vectors]
    if missing:
        raise RuntimeError(
            f"Some IDs were not found in Milvus collection {col.name!r}: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    return np.vstack([out_vectors[rid] for rid in ids]).astype(np.float32)


def fetch_vector_by_id(*, dataset: str, id_: str) -> np.ndarray:
    """
    Convenience: fetch one vector by ID.
    """
    return fetch_vectors(dataset=dataset, ids=[id_])[0]