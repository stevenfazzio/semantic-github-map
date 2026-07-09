"""Generate topic labels using Toponymy with Claude LLM."""

import joblib

# Workaround: nested asyncio.run() calls fail with "Event loop is closed".
# nest_asyncio patches the loop to allow re-entrant calls.
import nest_asyncio
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

from config import (
    ANTHROPIC_API_KEY,
    CO_API_KEY,
    EMBEDDINGS_NPZ,
    LABELS_PARQUET,
    REPOS_PARQUET,
    TOPONYMY_MODEL_JOBLIB,
    UMAP_COORDS_NPZ,
)


def main():
    # Load data
    df = pd.read_parquet(REPOS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

    # Use LLM-generated summaries if available (from 03_summarize_readmes.py),
    # falling back to truncated READMEs
    MAX_README_CHARS = 2_000
    has_summary = "summary" in df.columns

    has_tagline = "tagline" in df.columns
    has_title = "project_title" in df.columns

    documents = []
    for _, row in df.iterrows():
        summary = ""
        if has_summary and isinstance(row.get("summary"), str):
            summary = row["summary"].strip()

        # Build prefix: "Title — Tagline" when both exist
        prefix_parts = []
        if has_title and isinstance(row.get("project_title"), str) and row["project_title"].strip():
            prefix_parts.append(row["project_title"].strip())
        if has_tagline and isinstance(row.get("tagline"), str) and row["tagline"].strip():
            prefix_parts.append(row["tagline"].strip())
        prefix = " — ".join(prefix_parts)

        if summary:
            text = f"{prefix}\n{summary}" if prefix else summary
        else:
            text = row["readme"].strip() if isinstance(row["readme"], str) else ""
            text = text[:MAX_README_CHARS]
            if prefix and text:
                text = f"{prefix}\n{text}"
            elif prefix:
                text = prefix

        if not text:
            text = row["description"].strip() if isinstance(row["description"], str) else ""
        if not text:
            text = row["full_name"]
        documents.append(text)

    print(f"Loaded {len(documents)} documents")

    # Set up Toponymy components
    llm = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model="claude-sonnet-4-20250514")
    embedder = CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0")
    clusterer = ToponymyClusterer(min_clusters=4)

    # Seed NumPy RNG for reproducible exemplar selection within Toponymy.
    # Note: LLM-generated topic names are inherently non-deterministic.
    np.random.seed(42)

    # Fit clusterer
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    # Fit Toponymy model
    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="GitHub repository descriptions",
        corpus_description="collection of the top 10,000 most-starred GitHub repositories",
        exemplar_delimiters=['    * """', '"""\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )

    # Extract per-document labels from all cluster layers.
    # cluster_layers_[0] is finest; the parquet stores label_layer_0 = coarsest.
    # NOTE: DataMapPlot itself expects FINEST-first -- stage 07 reverses at the
    # call site. (The old comment "DataMapPlot expects coarsest first" was wrong.)
    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("No cluster layers found")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    labels_dict = {"full_name": df["full_name"]}
    for i, layer in enumerate(reversed(topic_model.cluster_layers_)):
        labels_dict[f"label_layer_{i}"] = layer.topic_name_vector

    labels_df = pd.DataFrame(labels_dict)
    labels_df.to_parquet(LABELS_PARQUET, index=False)
    print(f"Saved labels to {LABELS_PARQUET}")

    # Save model (skip if unpicklable due to async client locks)
    try:
        joblib.dump(topic_model, TOPONYMY_MODEL_JOBLIB)
        print(f"Saved model to {TOPONYMY_MODEL_JOBLIB}")
    except TypeError:
        print("Skipped saving model (async client is not picklable)")


if __name__ == "__main__":
    main()
