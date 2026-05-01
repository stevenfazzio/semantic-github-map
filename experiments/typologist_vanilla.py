# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typologist[anthropic] @ file:///Users/stevenfazzio/repos/typologist",
#     "sentence-transformers",
#     "pandas",
#     "pyarrow",
#     "numpy",
#     "python-dotenv",
# ]
# ///
"""Vanilla Typologist run on top 500 GitHub repos by stars.

Run with:
    uv run experiments/typologist_vanilla.py

Why this is a PEP 723 script and not a regular pipeline module: Typologist
pins toponymy>=0.5, while this project's main venv pins toponymy==0.4 for
stage 06. Running under `uv run` puts this script in its own ephemeral env
so the main pipeline is untouched.

Inputs (read from the project's data/ dir):
  - data/repos.parquet      (sorted by stargazers_count, top 500 taken)
  - data/embeddings.npz     (positionally aligned with repos.parquet)

Outputs:
  - data/experiments/typologist_vanilla/schema.json
  - data/experiments/typologist_vanilla/labels.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typologist import AnthropicLLM, Typologist

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPOS_PARQUET = DATA_DIR / "repos.parquet"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
OUT_DIR = DATA_DIR / "experiments" / "typologist_vanilla"

N_DOCS = 500
N_FACETS = 3
RANDOM_STATE = 0


def build_documents(df: pd.DataFrame) -> list[str]:
    """Use stage 03 summaries when available, fall back to readme/description.

    Summaries are ~2-3 sentences, uniform-ish in length, and keep per-doc
    labeling cost bounded compared to feeding raw READMEs (which run 5-50x
    longer). The Cohere embeddings driving clustering were still computed
    against full READMEs, so the documents seen by the labeling LLM and the
    embeddings seen by Toponymy aren't from the same text — accepted for v1.
    """
    documents: list[str] = []
    for _, row in df.iterrows():
        text = ""
        if isinstance(row.get("summary"), str) and row["summary"].strip():
            text = row["summary"].strip()
        elif isinstance(row.get("readme"), str) and row["readme"].strip():
            text = row["readme"].strip()[:2000]
        elif isinstance(row.get("description"), str) and row["description"].strip():
            text = row["description"].strip()
        else:
            text = row["full_name"]
        documents.append(text)
    return documents


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_parquet(REPOS_PARQUET)
    embeddings_all = np.load(EMBEDDINGS_NPZ)["embeddings"]
    if embeddings_all.shape[0] != len(df_all):
        raise RuntimeError(
            f"Row mismatch: repos.parquet has {len(df_all)} rows, "
            f"embeddings.npz has {embeddings_all.shape[0]}"
        )

    order = df_all.sort_values("stargazers_count", ascending=False).index[:N_DOCS]
    df = df_all.loc[order].reset_index(drop=True)
    embeddings = embeddings_all[order.to_numpy()]

    documents = build_documents(df)
    print(f"Loaded {len(documents)} documents, embeddings shape {embeddings.shape}")

    topic_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    t = Typologist(
        n_facets=N_FACETS,
        topic_embedder=topic_embedder,
        object_description="GitHub repository READMEs",
        corpus_description="top 500 most-starred GitHub repositories",
        naming_llm=AnthropicLLM("claude-haiku-4-5"),
        schema_llm=AnthropicLLM("claude-opus-4-7"),
        labeling_llm=AnthropicLLM("claude-haiku-4-5"),
        random_state=RANDOM_STATE,
        verbose=True,
    )
    t.fit(documents, embeddings)

    schema_records = [
        {
            "facet_index": i,
            "name": facet["name"],
            "kind": facet.get("kind", ""),
            "definition": facet.get("definition", ""),
            "values": list(facet.get("values", [])),
        }
        for i, facet in enumerate(t.schema_)
    ]
    pd.DataFrame(schema_records).to_json(OUT_DIR / "schema.json", orient="records", indent=2)

    labels = t.labels_df_.copy()
    labels.insert(0, "full_name", df["full_name"].values)
    labels.to_parquet(OUT_DIR / "labels.parquet", index=False)

    print("\nDiscovered schema:")
    for facet in t.schema_:
        print(f"\nFacet: {facet['name']}")
        if facet.get("definition"):
            print(f"  Definition: {facet['definition']}")
        for v in facet.get("values", []):
            print(f"    - {v}")

    print(f"\nSaved schema  → {OUT_DIR / 'schema.json'}")
    print(f"Saved labels  → {OUT_DIR / 'labels.parquet'}")


if __name__ == "__main__":
    main()
