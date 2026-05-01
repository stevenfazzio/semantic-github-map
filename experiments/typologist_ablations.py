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
"""Ablation studies on Typologist's two mechanisms.

Tests the contribution of each piece of Typologist's iterative discovery
loop by toggling them off in isolation:

  * leace_only:    keep LEACE residualization between facets,
                   but do NOT tell synthesis_llm about prior facets.
  * priors_only:   tell synthesis_llm about prior facets,
                   but do NOT residualize embeddings between facets
                   (and skip per-doc labeling, which is only needed for LEACE).
  * single_pass:   one Toponymy run, one schema_llm call asking for all n_facets
                   at once. No iteration, no residualization, no per-doc labeling.

Each variant is run on two metadata-erasure conditions:
  * vanilla:       no metadata= (matches typologist_vanilla.py)
  * kitchen_sink:  metadata=[language, owner_type, project_type, target_audience]
                   (matches the kitchen_sink experiment from typologist_experiments.py)

Outputs land in data/experiments/typologist_abl_<variant>_<condition>/, in the
same shape (schema.json, labels.parquet) the existing visualize script consumes.
labels.parquet will be empty for priors_only and single_pass since per-doc
labeling is only run when LEACE needs the labels.

Run:
    uv run experiments/typologist_ablations.py
    uv run experiments/typologist_ablations.py --resume
    uv run experiments/typologist_ablations.py --only abl_single_pass_vanilla
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

KS_METADATA = ["language", "owner_type", "project_type", "target_audience"]

# Cheap variants first (no per-doc labeling), then the expensive one.
EXPERIMENTS = [
    {
        "name": "abl_priors_only_vanilla",
        "variant": "priors_only",
        "metadata": [],
    },
    {
        "name": "abl_priors_only_ks",
        "variant": "priors_only",
        "metadata": KS_METADATA,
    },
    {
        "name": "abl_single_pass_vanilla",
        "variant": "single_pass",
        "metadata": [],
    },
    {
        "name": "abl_single_pass_ks",
        "variant": "single_pass",
        "metadata": KS_METADATA,
    },
    {
        "name": "abl_leace_only_vanilla",
        "variant": "leace_only",
        "metadata": [],
    },
    {
        "name": "abl_leace_only_ks",
        "variant": "leace_only",
        "metadata": KS_METADATA,
    },
]


def build_documents(df: pd.DataFrame) -> list[str]:
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


def build_metadata(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame | None:
    if not columns:
        return None
    out = df[columns].copy()
    for col in columns:
        out[col] = out[col].fillna("Unknown").astype(str).replace("", "Unknown")
    return out


def make_typologist(variant: str, topic_embedder) -> Typologist:
    """Build a Typologist with toggles set per the ablation variant."""
    if variant == "leace_only":
        kwargs = {"concept_erasure": True, "inform_synthesis_of_prior_facets": False}
    elif variant == "priors_only":
        kwargs = {"concept_erasure": False, "inform_synthesis_of_prior_facets": True}
    elif variant == "single_pass":
        kwargs = {}  # toggles don't apply to fit_single_pass
    else:
        raise ValueError(f"unknown variant: {variant}")

    return Typologist(
        n_facets=N_FACETS,
        topic_embedder=topic_embedder,
        object_description="GitHub repository READMEs",
        corpus_description="top 500 most-starred GitHub repositories",
        naming_llm=AnthropicLLM("claude-haiku-4-5"),
        schema_llm=AnthropicLLM("claude-opus-4-7"),
        labeling_llm=AnthropicLLM("claude-haiku-4-5"),
        random_state=RANDOM_STATE,
        verbose=True,
        **kwargs,
    )


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
    if not labels.empty:
        labels.insert(0, "full_name", df["full_name"].values)
    else:
        # Stub labels parquet: just full_name, so the visualize script can
        # at least try to align (it'll skip facets that aren't present).
        labels = pd.DataFrame({"full_name": df["full_name"].values})
    labels.to_parquet(out_dir / "labels.parquet", index=False)


def run_one(
    name: str,
    variant: str,
    metadata_cols: list[str],
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

    metadata = build_metadata(df, metadata_cols)

    print(f"\n{'=' * 60}")
    print(f"[{name}] variant={variant} erasing={metadata_cols or '(none)'}")
    print(f"{'=' * 60}")

    t = make_typologist(variant, topic_embedder)
    if variant == "single_pass":
        t.fit_single_pass(documents, embeddings, metadata=metadata)
    else:
        t.fit(documents, embeddings, metadata=metadata)

    save_outputs(t, df, out_dir)

    print(f"\n[{name}] Discovered facets:")
    for facet in t.schema_:
        print(f"  {facet['name']}: {list(facet.get('values', []))}")
    print(f"[{name}] Saved → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments whose outputs already exist")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated experiment names to run (default: all)")
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

    needed = {c for exp in EXPERIMENTS for c in exp["metadata"]}
    missing = needed - set(df.columns)
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
            exp["variant"],
            exp["metadata"],
            df,
            embeddings,
            documents,
            topic_embedder,
            args.resume,
        )

    print("\nAll ablation experiments complete.")


if __name__ == "__main__":
    main()
