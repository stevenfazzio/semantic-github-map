# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typologist[anthropic] @ file:///Users/stevenfazzio/repos/typologist",
#     "pandas",
#     "pyarrow",
#     "numpy",
#     "python-dotenv",
# ]
# ///
"""Apply discovered schemas to the top-500 documents for variants that did not
run per-document labeling at fit time (priors_only and single_pass).

This reconstructs each facet's labeling_prompt_template from the saved schema
fields and calls typologist.apply_schema to produce per-doc labels, then
overwrites the variant's labels.parquet with the full label set.

Run:
    uv run experiments/typologist_apply_labels.py
    uv run experiments/typologist_apply_labels.py --only abl_priors_only_vanilla
    uv run experiments/typologist_apply_labels.py --resume   # skip already-labeled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typologist import AnthropicLLM, apply_schema
from typologist._prompts import render_labeling_template

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPOS_PARQUET = DATA_DIR / "repos.parquet"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
EXPERIMENTS_BASE = DATA_DIR / "experiments"

N_DOCS = 500
OBJECT_DESCRIPTION = "GitHub repository READMEs"

# Variants whose fit skipped per-doc labeling. labels.parquet for these
# currently has only a `full_name` column.
TARGET_VARIANTS = [
    "abl_priors_only_vanilla",
    "abl_priors_only_ks",
    "abl_single_pass_vanilla",
    "abl_single_pass_ks",
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


def reconstruct_schema(records: list[dict]) -> list[dict]:
    """Rebuild the labeling_prompt_template for each facet from its saved fields.

    The saved schema.json drops labeling_prompt_template and labeling_model to
    keep the file small and readable. apply_schema needs the template, so we
    rebuild it deterministically from name + definition + values.
    """
    schema: list[dict] = []
    for r in records:
        name = r["name"]
        values = list(r["values"])
        definition = r.get("definition", "")
        schema.append(
            {
                "name": name,
                "kind": r.get("kind", "categorical"),
                "values": values,
                "definition": definition,
                "labeling_prompt_template": render_labeling_template(
                    facet_name=name,
                    facet_definition=definition,
                    values=values,
                    object_description=OBJECT_DESCRIPTION,
                ),
            }
        )
    return schema


def label_one(variant: str, documents: list[str], full_names: np.ndarray, resume: bool) -> None:
    out_dir = EXPERIMENTS_BASE / f"typologist_{variant}"
    schema_path = out_dir / "schema.json"
    labels_path = out_dir / "labels.parquet"

    if not schema_path.exists():
        print(f"\n[{variant}] Missing schema.json at {schema_path} — skipping")
        return

    if resume:
        existing = pd.read_parquet(labels_path)
        if len(existing.columns) > 1:  # already has facet cols beyond full_name
            print(f"\n[{variant}] Skipping (resume) — labels.parquet already has facet columns")
            return

    records = json.loads(schema_path.read_text())
    schema = reconstruct_schema(records)

    print(f"\n{'=' * 60}")
    print(f"[{variant}] Labeling {len(documents)} docs across {len(schema)} facets")
    print(f"{'=' * 60}")

    llm = AnthropicLLM("claude-haiku-4-5")
    labels_df = apply_schema(
        schema=schema,
        documents=documents,
        llm=llm,
        max_concurrency=10,
        verbose=True,
    )

    labels_df.insert(0, "full_name", full_names)
    labels_df.to_parquet(labels_path, index=False)
    print(f"[{variant}] Saved labels → {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip variants whose labels.parquet already has facet columns")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated variant names to label (default: all targets)")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}

    df_all = pd.read_parquet(REPOS_PARQUET)
    order = df_all.sort_values("stargazers_count", ascending=False).index[:N_DOCS]
    df = df_all.loc[order].reset_index(drop=True)
    documents = build_documents(df)
    full_names = df["full_name"].values
    print(f"Loaded {len(documents)} documents")

    selected = TARGET_VARIANTS if not only else [v for v in TARGET_VARIANTS if v in only]
    if only and not selected:
        print(f"No variants match --only filter: {only}")
        print(f"Available: {TARGET_VARIANTS}")
        return

    for variant in selected:
        label_one(variant, documents, full_names, args.resume)

    print("\nAll labelings complete.")


if __name__ == "__main__":
    main()
