-- Schema for DuckDB metadata store

CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY,
  dataset TEXT NOT NULL,            -- "orf" | "crispr" | "compound"
  name TEXT,                        -- gene/compound name, etc.
  perturbation_type TEXT,           -- optional
  plate TEXT,
  well TEXT,
  batch TEXT,
  cell_line TEXT,
  timepoint TEXT,

  -- optional precomputed coords for instant plotting later
  pca_x DOUBLE,
  pca_y DOUBLE,
  umap_x DOUBLE,
  umap_y DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_profiles_dataset ON profiles(dataset);