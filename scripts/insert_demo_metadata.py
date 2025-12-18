import pandas as pd
from adapters.metadata_store import get_store

def main():
    store = get_store()

    rows = []
    for i in range(50):
        rows.append({
            "id": f"demo_{i}",
            "dataset": "orf",
            "name": f"DemoGene{i}",
            "perturbation_type": "demo",
            "plate": "P1",
            "well": f"A{(i % 12) + 1}",
            "batch": "B1",
            "cell_line": "U2OS",
            "timepoint": "24h",
            "pca_x": None,
            "pca_y": None,
            "umap_x": None,
            "umap_y": None,
        })

    df = pd.DataFrame(rows)
    store.upsert_profiles(df)
    print("✅ Inserted demo metadata into DuckDB")

if __name__ == "__main__":
    main()