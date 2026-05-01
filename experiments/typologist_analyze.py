"""Compute per-facet diagnostics across all completed Typologist runs.

Metrics per variant:
  - Per facet:
      - balance: H(facet) / log2(|values|), where H is observed Shannon entropy.
        1.0 means uniform; 0 means all docs in a single bucket.
      - other_rate: fraction of docs labeled "Other" or "Unlabelled" (typologist's
        two catch-alls, summed).
  - Pairwise normalized mutual information between the three facets within
    a variant. NMI in [0,1]; 0 is independent, 1 is identical.

Reads:
  data/experiments/typologist_*/{schema.json, labels.parquet}

Run:
    uv run python experiments/typologist_analyze.py
    uv run python experiments/typologist_analyze.py --csv data/typologist_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import DATA_DIR, EMBEDDINGS_NPZ, REPOS_PARQUET  # noqa: E402

EXPERIMENTS_BASE = DATA_DIR / "experiments"
EXPERIMENT_PREFIX = "typologist_"
CATCH_ALL_LABELS = {"Other", "Unlabelled"}
N_DOCS = 500
PROBE_CV_FOLDS = 3

# Variant directory names mash two orthogonal dimensions together:
#   - mechanism: which Typologist code path was used
#   - condition: which metadata columns were pre-erased
# This map decomposes the directory suffix into those two dimensions so we can
# display the ablation matrices in their natural shape.
_NAME_TO_DIMENSIONS = {
    # Full mechanics, varying the erasure condition (the original "study 1")
    "vanilla":              ("full",         "vanilla"),
    "lang_erased":          ("full",         "lang_erased"),
    "owner_erased":         ("full",         "owner_erased"),
    "lang_owner_erased":    ("full",         "lang_owner_erased"),
    "stage03_erased":       ("full",         "stage03_erased"),
    "kitchen_sink":         ("full",         "kitchen_sink"),
    # Mechanism ablations on vanilla and kitchen_sink (the "study 2" matrix)
    "abl_leace_only_vanilla":     ("leace_only",  "vanilla"),
    "abl_leace_only_ks":          ("leace_only",  "kitchen_sink"),
    "abl_priors_only_vanilla":    ("priors_only", "vanilla"),
    "abl_priors_only_ks":         ("priors_only", "kitchen_sink"),
    "abl_single_pass_vanilla":    ("single_pass", "vanilla"),
    "abl_single_pass_ks":         ("single_pass", "kitchen_sink"),
}

_MECHANISM_ORDER = ["full", "leace_only", "priors_only", "single_pass"]
_CONDITION_ORDER = [
    "vanilla",
    "lang_erased",
    "owner_erased",
    "lang_owner_erased",
    "stage03_erased",
    "kitchen_sink",
]


def _balance(series: pd.Series, vocab_size: int) -> float:
    """Normalized entropy: H(observed) / log2(vocab_size)."""
    counts = series.value_counts()
    probs = counts.values / counts.sum()
    if probs.size <= 1:
        return 0.0
    h = float(-(probs * np.log2(probs)).sum())
    h_max = float(np.log2(vocab_size)) if vocab_size > 1 else 0.0
    return h / h_max if h_max > 0 else 0.0


def _other_rate(series: pd.Series) -> float:
    return float(series.isin(CATCH_ALL_LABELS).mean())


def _load_top_n_embeddings() -> dict[str, np.ndarray]:
    """Return {full_name: embedding} for the top-N-by-stars repos used at fit time."""
    df = pd.read_parquet(REPOS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    order = df.sort_values("stargazers_count", ascending=False).index[:N_DOCS]
    sub = df.loc[order].reset_index(drop=True)
    sub_emb = embeddings[order.to_numpy()]
    return dict(zip(sub["full_name"].values, sub_emb))


def _probe_facet(emb: np.ndarray, y: np.ndarray) -> float | None:
    """Cross-validated macro-F1 of a logistic probe on (emb, y).

    Drops classes too rare for stratified k-fold so a single rare label doesn't
    block the probe. Returns None if there's not enough data left after filtering.
    """
    counts = pd.Series(y).value_counts()
    keep_classes = counts[counts >= PROBE_CV_FOLDS].index.tolist()
    if len(keep_classes) < 2:
        return None
    keep = np.isin(y, keep_classes)
    if keep.sum() < 30:
        return None
    try:
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        scores = cross_val_score(clf, emb[keep], y[keep], cv=PROBE_CV_FOLDS, scoring="f1_macro")
        return float(scores.mean())
    except Exception:
        return None


def _analyze_variant(name: str, exp_dir: Path, emb_map: dict[str, np.ndarray] | None = None) -> dict | None:
    schema_path = exp_dir / "schema.json"
    labels_path = exp_dir / "labels.parquet"
    if not schema_path.exists() or not labels_path.exists():
        return None

    schema = json.loads(schema_path.read_text())
    labels_df = pd.read_parquet(labels_path)
    facet_names = [f["name"] for f in schema]
    missing = [n for n in facet_names if n not in labels_df.columns]
    if missing:
        return {"name": name, "status": f"missing facet columns: {missing}"}

    # Align embeddings (and labels) by full_name for the probe.
    emb_aligned = None
    labels_aligned = labels_df
    if emb_map is not None and "full_name" in labels_df.columns:
        keep = np.array([fn in emb_map for fn in labels_df["full_name"]])
        labels_aligned = labels_df[keep].reset_index(drop=True)
        emb_aligned = np.stack([emb_map[fn] for fn in labels_aligned["full_name"]])

    facets_info = []
    for f in schema:
        col = f["name"]
        series = labels_aligned[col].astype(str)
        vocab = list(f.get("values", []))
        probe_f1: float | None = None
        if emb_aligned is not None:
            probe_f1 = _probe_facet(emb_aligned, series.values)
        facets_info.append(
            {
                "name": col,
                "vocab_size": len(vocab),
                "balance": _balance(series, len(vocab)),
                "other_rate": _other_rate(series),
                "probe_f1": probe_f1,
            }
        )

    nmi_pairs: dict[str, float] = {}
    for i, j in combinations(range(len(facet_names)), 2):
        a = labels_df[facet_names[i]].astype(str).values
        b = labels_df[facet_names[j]].astype(str).values
        nmi = float(normalized_mutual_info_score(a, b))
        nmi_pairs[f"{i}-{j}"] = nmi

    mechanism, condition = _NAME_TO_DIMENSIONS.get(name, ("?", name))
    return {
        "name": name,
        "mechanism": mechanism,
        "condition": condition,
        "status": "ok",
        "facets": facets_info,
        "nmi_pairs": nmi_pairs,
        "mean_nmi": float(np.mean(list(nmi_pairs.values()))) if nmi_pairs else 0.0,
        "max_nmi": float(max(nmi_pairs.values())) if nmi_pairs else 0.0,
    }


def _print_variant(result: dict) -> None:
    name = result["name"]
    if result["status"] != "ok":
        print(f"\n[{name}] {result['status']}")
        return

    print(f"\n[{name}]  mean pairwise NMI = {result['mean_nmi']:.3f}")
    for i, f in enumerate(result["facets"]):
        bar = "#" * int(round(f["balance"] * 20))
        probe_str = f"  probe_f1={f['probe_f1']:.2f}" if f.get("probe_f1") is not None else "  probe_f1=  -  "
        print(
            f"  Facet {i}: {f['name']:<35}"
            f"  balance={f['balance']:.2f} |{bar:<20}|"
            f"  other={f['other_rate']:.0%}"
            f"  vocab={f['vocab_size']}"
            f"{probe_str}"
        )
    print(
        f"  pairwise NMI: "
        + ", ".join(f"{k}={v:.3f}" for k, v in result["nmi_pairs"].items())
    )


def _summary_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r["status"] != "ok":
            continue
        other_rates = [f["other_rate"] for f in r["facets"]]
        balances = [f["balance"] for f in r["facets"]]
        probe_f1s = [f["probe_f1"] for f in r["facets"] if f.get("probe_f1") is not None]
        row = {
            "mechanism": r["mechanism"],
            "condition": r["condition"],
            "mean_nmi": r["mean_nmi"],
            "max_nmi": r["max_nmi"],
            "mean_other_rate": float(np.mean(other_rates)) if other_rates else 0.0,
            "max_other_rate": float(max(other_rates)) if other_rates else 0.0,
            "mean_balance": float(np.mean(balances)) if balances else 0.0,
            "mean_probe_f1": float(np.mean(probe_f1s)) if probe_f1s else float("nan"),
            "min_probe_f1": float(np.min(probe_f1s)) if probe_f1s else float("nan"),
        }
        for i, f in enumerate(r["facets"]):
            row[f"f{i}_name"] = f["name"]
            row[f"f{i}_balance"] = f["balance"]
            row[f"f{i}_other_rate"] = f["other_rate"]
        for k, v in r["nmi_pairs"].items():
            row[f"nmi_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot to mechanism × condition, ordered. Cells without a run stay NaN."""
    pivot = df.pivot_table(
        index="mechanism", columns="condition", values=value_col, aggfunc="first"
    )
    mechs = [m for m in _MECHANISM_ORDER if m in pivot.index]
    conds = [c for c in _CONDITION_ORDER if c in pivot.columns]
    return pivot.loc[mechs, conds]


def _xvariant_matrix(
    a_name: str,
    b_name: str,
    aligned: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    """Pairwise NMI between every facet of variant a and every facet of variant b.

    Rows are a's facets, columns are b's facets. Cells are NMI(label_a, label_b).
    """
    a_facets = [(col, arr) for (var, col), arr in aligned.items() if var == a_name]
    b_facets = [(col, arr) for (var, col), arr in aligned.items() if var == b_name]
    if not a_facets or not b_facets:
        return pd.DataFrame()
    rows = []
    for col_a, arr_a in a_facets:
        row = {"facet": col_a}
        for col_b, arr_b in b_facets:
            row[col_b] = float(normalized_mutual_info_score(arr_a, arr_b))
        rows.append(row)
    return pd.DataFrame(rows).set_index("facet")


def _build_aligned_labels(
    results: list[dict],
    emb_map: dict[str, np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """Load every variant's labels and align them by full_name to a common index.

    Returns {(variant_name, facet_name): label_array} where every array has the
    same length and indexing (the full_names that appear in emb_map).
    """
    common = list(emb_map.keys())
    out: dict[tuple[str, str], np.ndarray] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        labels_path = EXPERIMENTS_BASE / f"{EXPERIMENT_PREFIX}{r['name']}" / "labels.parquet"
        labels_df = pd.read_parquet(labels_path).set_index("full_name")
        for f in r["facets"]:
            col = f["name"]
            if col in labels_df.columns:
                out[(r["name"], col)] = labels_df.reindex(common)[col].astype(str).values
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="",
                        help="Optional path to write the summary table as CSV")
    args = parser.parse_args()

    if not EXPERIMENTS_BASE.exists():
        print(f"No experiments dir at {EXPERIMENTS_BASE}")
        return

    variant_dirs = sorted(
        d for d in EXPERIMENTS_BASE.iterdir()
        if d.is_dir() and d.name.startswith(EXPERIMENT_PREFIX)
    )
    if not variant_dirs:
        print(f"No typologist_* experiments found under {EXPERIMENTS_BASE}")
        return

    print(f"Loading top-{N_DOCS} embeddings for probe analysis...")
    emb_map = _load_top_n_embeddings()

    results = []
    for d in variant_dirs:
        name = d.name[len(EXPERIMENT_PREFIX):]
        r = _analyze_variant(name, d, emb_map=emb_map)
        if r is None:
            print(f"\n[{name}] missing schema.json or labels.parquet — skipping")
            continue
        results.append(r)
        _print_variant(r)

    summary = _summary_table(results)

    print("\n" + "=" * 80)
    print("Mechanism × condition pivot — mean pairwise NMI (lower = more orthogonal)")
    print("=" * 80)
    print(_pivot(summary, "mean_nmi").to_string(float_format="{:.3f}".format, na_rep="  -  "))

    print("\n" + "=" * 80)
    print("Mechanism × condition pivot — MAX pairwise NMI (catches soft duplicates)")
    print("=" * 80)
    print(_pivot(summary, "max_nmi").to_string(float_format="{:.3f}".format, na_rep="  -  "))

    print("\n" + "=" * 80)
    print("Mechanism × condition pivot — mean Other-rate across the 3 facets")
    print("=" * 80)
    print(_pivot(summary, "mean_other_rate").to_string(float_format="{:.3f}".format, na_rep="  -  "))

    print("\n" + "=" * 80)
    print("Mechanism × condition pivot — MAX Other-rate (worst facet's miss rate)")
    print("=" * 80)
    print(_pivot(summary, "max_other_rate").to_string(float_format="{:.3f}".format, na_rep="  -  "))

    print("\n" + "=" * 80)
    print("Mechanism × condition pivot — mean linear-probe macro-F1")
    print("(higher = facets are decodable from embeddings, i.e. correspond to real structure)")
    print("=" * 80)
    print(_pivot(summary, "mean_probe_f1").to_string(float_format="{:.3f}".format, na_rep="  -  "))

    print("\n" + "=" * 80)
    print("Mechanism × condition pivot — MIN linear-probe macro-F1 (worst facet)")
    print("=" * 80)
    print(_pivot(summary, "min_probe_f1").to_string(float_format="{:.3f}".format, na_rep="  -  "))

    # ── Cross-variant agreement on shared axes ──────────────────────────────
    print("\n" + "=" * 80)
    print("Cross-variant agreement (NMI between facet labels on the same docs)")
    print("Higher NMI = the two facets are labeling the same docs the same way")
    print("=" * 80)

    aligned = _build_aligned_labels(results, emb_map)
    pairs_to_show = [
        ("vanilla", "abl_single_pass_vanilla"),
        ("vanilla", "abl_priors_only_vanilla"),
        ("vanilla", "abl_leace_only_vanilla"),
        ("kitchen_sink", "abl_single_pass_ks"),
    ]
    for a, b in pairs_to_show:
        m = _xvariant_matrix(a, b, aligned)
        if m.empty:
            continue
        print(f"\n{a}  vs  {b}")
        print(m.to_string(float_format="{:.3f}".format))

    print("\n" + "=" * 80)
    print("Long-form summary (sorted by mean pairwise NMI)")
    print("=" * 80)
    sorted_summary = summary.sort_values("mean_nmi")
    cols_to_show = [
        "mechanism", "condition", "mean_nmi", "max_nmi",
        "f0_balance", "f1_balance", "f2_balance",
        "f0_other_rate", "f1_other_rate", "f2_other_rate",
    ]
    cols_to_show = [c for c in cols_to_show if c in sorted_summary.columns]
    print(sorted_summary[cols_to_show].to_string(index=False, float_format="{:.3f}".format))

    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"\nSaved CSV → {args.csv}")


if __name__ == "__main__":
    main()
