"""Precompute binary files for the client-side re-UMAP POC.

Reads existing pipeline outputs (embeddings, UMAP coords, repo metadata) and writes
optimized binary files for JS fetch() + TypedArray consumption.

Usage:
    python experiments/poc_reumap_precompute.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config import DATA_DIR, EMBEDDINGS_NPZ, REPOS_PARQUET, UMAP_COORDS_NPZ

# ── Constants ────────────────────────────────────────────────────────────────
KNN_K = 50


# ── Pure functions (importable for testing) ──────────────────────────────────


def remap_knn_to_subset(knn_indices, knn_distances, subset_indices):
    """Remap a global KNN graph to subset-local indices.

    For each point in subset_indices, extracts its k neighbors from the global
    KNN graph, drops neighbors not in the subset, and remaps remaining to
    subset-local indices. Pads with -1 / inf where neighbors are missing.

    Returns (remapped_indices, remapped_distances) with shape [len(subset), k].
    """
    k = knn_indices.shape[1]
    n_subset = len(subset_indices)

    # Build global-to-local index map
    global_to_local = np.full(knn_indices.max() + 1, -1, dtype=np.int32)
    for local_idx, global_idx in enumerate(subset_indices):
        global_to_local[global_idx] = local_idx

    remapped_indices = np.full((n_subset, k), -1, dtype=np.int32)
    remapped_distances = np.full((n_subset, k), np.inf, dtype=np.float32)

    for i, global_idx in enumerate(subset_indices):
        neighbors = knn_indices[global_idx]
        distances = knn_distances[global_idx]

        # Filter to neighbors present in the subset
        local_neighbors = global_to_local[neighbors]
        mask = local_neighbors >= 0
        valid_local = local_neighbors[mask]
        valid_dist = distances[mask]

        n_valid = len(valid_local)
        remapped_indices[i, :n_valid] = valid_local
        remapped_distances[i, :n_valid] = valid_dist

    return remapped_indices, remapped_distances


def rescale_coords(coords, target_range=(-9.0, 9.0)):
    """Rescale 2D coordinates to fit within target_range on both axes.

    Preserves aspect ratio by scaling uniformly based on the larger axis span.
    """
    lo, hi = target_range
    target_span = hi - lo

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = maxs - mins
    scale = target_span / max(spans[0], spans[1])

    centered = coords - (mins + maxs) / 2
    scaled = centered * scale
    return scaled.astype(np.float32)


def main():
    # ── Load inputs ──────────────────────────────────────────────────────────
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"].astype(np.float32)
    n, dim = embeddings.shape
    print(f"  {n} embeddings, {dim}D")

    print("Loading UMAP coords...")
    coords = np.load(UMAP_COORDS_NPZ)["coords"].astype(np.float32)
    print(f"  {coords.shape}")

    print("Loading metadata...")
    df = pd.read_parquet(REPOS_PARQUET, columns=["full_name", "stargazers_count", "language"])

    assert len(df) == n == len(coords), f"Length mismatch: {len(df)} repos, {n} embeddings, {len(coords)} coords"

    # ── Compute KNN graph ────────────────────────────────────────────────────
    print(f"Computing KNN graph (k={KNN_K}, cosine, brute)...")
    nn = NearestNeighbors(n_neighbors=KNN_K, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(embeddings)
    knn_distances, knn_indices = nn.kneighbors(embeddings)
    print("  Done.")

    # ── Write binary outputs ─────────────────────────────────────────────────
    out_embeddings = DATA_DIR / "poc_embeddings.bin"
    out_knn_indices = DATA_DIR / "poc_knn_indices.bin"
    out_knn_distances = DATA_DIR / "poc_knn_distances.bin"
    out_coords = DATA_DIR / "poc_coords.bin"
    out_metadata = DATA_DIR / "poc_metadata.json"

    print(f"Writing {out_embeddings}...")
    embeddings.tofile(out_embeddings)

    print(f"Writing {out_knn_indices}...")
    knn_indices.astype(np.uint16).tofile(out_knn_indices)

    print(f"Writing {out_knn_distances}...")
    knn_distances.astype(np.float32).tofile(out_knn_distances)

    print(f"Writing {out_coords}...")
    coords.tofile(out_coords)

    print(f"Writing {out_metadata}...")
    metadata = [
        {
            "name": row.full_name,
            "stars": int(row.stargazers_count),
            "language": row.language if pd.notna(row.language) else None,
        }
        for row in df.itertuples()
    ]
    with open(out_metadata, "w") as f:
        json.dump(metadata, f, separators=(",", ":"))

    # ── Summary ──────────────────────────────────────────────────────────────
    sizes = {
        out_embeddings: out_embeddings.stat().st_size,
        out_knn_indices: out_knn_indices.stat().st_size,
        out_knn_distances: out_knn_distances.stat().st_size,
        out_coords: out_coords.stat().st_size,
        out_metadata: out_metadata.stat().st_size,
    }
    total = sum(sizes.values())
    print("\nOutput files:")
    for path, size in sizes.items():
        print(f"  {path.name:30s} {size / 1024 / 1024:6.1f} MB")
    print(f"  {'TOTAL':30s} {total / 1024 / 1024:6.1f} MB")


if __name__ == "__main__":
    main()
