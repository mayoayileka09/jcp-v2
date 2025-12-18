import numpy as np
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility
from adapters.milvus_adapter import connect

def main():
    connect()

    name = "orf_profiles"
    dim = 128  # demo dimension

    # If collection exists, drop it so we can recreate with an index
    if utility.has_collection(name):
        utility.drop_collection(name)
        print("♻️ Dropped existing collection:", name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Demo ORF collection")
    col = Collection(name, schema=schema)

    # Insert demo rows
    ids = [f"demo_{i}" for i in range(50)]
    vecs = np.random.randn(50, dim).astype(np.float32)
    col.insert([ids, vecs.tolist()])
    col.flush()

    # ✅ Create index BEFORE loading
    col.create_index(
        field_name="vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 64},
        },
    )

    # Load collection into memory for search
    col.load()

    print("✅ Created demo collection, built index, and inserted rows:", name)

if __name__ == "__main__":
    main()