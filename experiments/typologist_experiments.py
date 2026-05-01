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
"""Run a matrix of Typologist erasure experiments on top 500 GitHub repos.

Each experiment fits Typologist with a different `metadata=` slate. Outputs
land in data/experiments/typologist_{name}/{schema.json, labels.parquet},
matching the layout that typologist_vanilla.py and typologist_visualize.py
already use.

Run:
    uv run experiments/typologist_experiments.py
    uv run experiments/typologist_experiments.py --resume
    uv run experiments/typologist_experiments.py --only lang_erased,owner_erased
"""

from __future__ import annotations

import argparse
import json
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
EXPERIMENTS_BASE = DATA_DIR / "experiments"

N_DOCS = 500
N_FACETS = 3
RANDOM_STATE = 0

EXPERIMENTS = [
    {"name": "lang_erased", "erase": ["language"]},
    {"name": "owner_erased", "erase": ["owner_type"]},
    {"name": "lang_owner_erased", "erase": ["language", "owner_type"]},
    {"name": "stage03_erased", "erase": ["project_type", "target_audience"]},
    {
        "name": "kitchen_sink",
        "erase": ["language", "owner_type", "project_type", "target_audience"],
    },
]


def build_documents(df: pd.DataFrame) -> list[str]:
    """Same fallback chain as typologist_vanilla.py."""
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


def build_metadata(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """LEACE expects no NaNs and stable category labels. Fill missing with
    'Unknown' so each row contributes a well-defined slate to the erasure."""
    out = df[columns].copy()
    for col in columns:
        out[col] = out[col].fillna("Unknown").astype(str).replace("", "Unknown")
    return out


def save_outputs(t: Typologist, df: pd.DataFrame, out_dir: Path) -> None:
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
    pd.DataFrame(schema_records).to_json(
        out_dir / "schema.json", orient="records", indent=2
    )

    labels = t.labels_df_.copy()
    labels.insert(0, "full_name", df["full_name"].values)
    labels.to_parquet(out_dir / "labels.parquet", index=False)


def run_one(
    name: str,
    erase: list[str],
    df: pd.DataFrame,
    embeddings: np.ndarray,
    documents: list[str],
    topic_embedder,
    resume: bool,
) -> None:
    out_dir = EXPERIMENTS_BASE / f"typologist_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_path = out_dir / "schema.json"
    labels_path = out_dir / "labels.parquet"

    if resume and schema_path.exists() and labels_path.exists():
        print(f"\n[{name}] Skipping (resume) — outputs exist at {out_dir}")
        return

    metadata = build_metadata(df, erase) if erase else None

    print(f"\n{'=' * 60}")
    print(f"[{name}] Erasing: {erase if erase else '(none — vanilla)'}")
    print(f"{'=' * 60}")

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
    t.fit(documents, embeddings, metadata=metadata)

    save_outputs(t, df, out_dir)

    print(f"\n[{name}] Discovered facets:")
    for facet in t.schema_:
        print(f"  {facet['name']}: {list(facet.get('values', []))}")
    print(f"[{name}] Saved → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments whose outputs already exist",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated experiment names to run (default: all)",
    )
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}

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

    needed_cols = {c for exp in EXPERIMENTS for c in exp["erase"]}
    missing = needed_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in repos.parquet: {sorted(missing)}")

    topic_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    selected = EXPERIMENTS if not only else [e for e in EXPERIMENTS if e["name"] in only]
    if only and not selected:
        print(f"No experiments match --only filter: {only}")
        print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
        return

    for exp in selected:
        run_one(
            exp["name"],
            exp["erase"],
            df,
            embeddings,
            documents,
            topic_embedder,
            args.resume,
        )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
