"""Render a DataMapPlot of the top-500 docs that the typologist runs fit,
with Typologist's discovered facets as selectable colormaps and Toponymy's
hierarchical labels as the region labels.

Inputs:
  - data/repos.parquet, data/umap_coords.npz, data/labels.parquet  (main pipeline)
  - data/experiments/typologist_<name>/{schema.json, labels.parquet}

Output:
  - data/experiments/typologist_<name>/map.html

Runs in the main project venv (no typologist/toponymy import at runtime, just
DataMapPlot consuming the saved outputs):
    uv run python experiments/typologist_visualize.py                  # vanilla
    uv run python experiments/typologist_visualize.py --exp lang_erased
    uv run python experiments/typologist_visualize.py --exp all        # render every
                                                                       # experiment dir found
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (  # noqa: E402
    DATA_DIR,
    LABELS_PARQUET,
    REPOS_PARQUET,
    UMAP_COORDS_NPZ,
)

EXPERIMENTS_BASE = DATA_DIR / "experiments"
EXPERIMENT_PREFIX = "typologist_"

N_DOCS = 500


def _select_top_n(df_all: pd.DataFrame, coords_all: np.ndarray, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    order = df_all.sort_values("stargazers_count", ascending=False).index[:n]
    df = df_all.loc[order].reset_index(drop=True)
    coords = coords_all[order.to_numpy()]
    return df, coords


def _align_by_full_name(side: pd.DataFrame, full_names: pd.Series) -> pd.DataFrame:
    return side.set_index("full_name").loc[full_names.values].reset_index()


def _render_one(
    exp_name: str,
    df: pd.DataFrame,
    coords: np.ndarray,
    label_layers: list,
) -> None:
    exp_dir = EXPERIMENTS_BASE / f"{EXPERIMENT_PREFIX}{exp_name}"
    schema_path = exp_dir / "schema.json"
    labels_path = exp_dir / "labels.parquet"
    output_html = exp_dir / "map.html"

    if not schema_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing Typologist outputs in {exp_dir}. "
            "Run the matching fit script first (typologist_vanilla.py or "
            "typologist_experiments.py)."
        )

    typologist = _align_by_full_name(pd.read_parquet(labels_path), df["full_name"])
    schema = json.loads(schema_path.read_text())
    facet_names = [f["name"] for f in schema]
    missing = [n for n in facet_names if n not in typologist.columns]
    if missing:
        raise RuntimeError(f"[{exp_name}] Schema names not in labels.parquet: {missing}")

    # ── Colormaps: one per facet, glasbey palette per category set ──────────
    rawdata: list[np.ndarray] = []
    metadata: list[dict] = []
    for facet in schema:
        name = facet["name"]
        values = typologist[name].astype(str).fillna("Unlabelled").values
        unique = sorted(set(values))
        palette = glasbey.create_palette(palette_size=len(unique))
        rawdata.append(values)
        metadata.append(
            {
                "field": name,
                "description": name,
                "kind": "categorical",
                "color_mapping": dict(zip(unique, palette)),
            }
        )

    # ── Hover card data ─────────────────────────────────────────────────────
    project_titles = df["project_title"].fillna("").astype(str).values
    summaries = df["summary"].fillna("").astype(str).values
    taglines = (
        df["tagline"].fillna("").astype(str).values if "tagline" in df.columns else np.array([""] * len(df))
    )

    extra_cols = {
        "full_name": df["full_name"].values,
        "project_title": project_titles,
        "tagline": taglines,
        "summary": summaries,
    }
    for name in facet_names:
        extra_cols[name] = typologist[name].astype(str).fillna("").values
    extra_data = pd.DataFrame(extra_cols)

    # Build the per-facet block of the hover card. Each {facet_name} placeholder
    # gets filled by DataMapPlot from the matching column in extra_data.
    facet_block = "".join(
        f'<div class="hc-facet"><span class="hc-facet-name">{name}</span>'
        f"<span class=\"hc-facet-val\">{{{name}}}</span></div>"
        for name in facet_names
    )
    hover_template = (
        '<div class="hc">'
        '<div class="hc-title">{project_title}</div>'
        '<div class="hc-tagline">{tagline}</div>'
        '<div class="hc-summary">{summary}</div>'
        f'<div class="hc-facets">{facet_block}</div>'
        "</div>"
    )

    custom_css = """
    .hc { padding: 12px 14px; max-width: 340px;
          font-family: 'IBM Plex Sans', system-ui, sans-serif; color: #1a1a2e; }
    .hc-title { font-weight: 600; font-size: 14px; color: #0d1117;
                margin-bottom: 3px; line-height: 1.3; }
    .hc-tagline { font-style: italic; color: #656d76; font-size: 12px;
                  margin-bottom: 8px; line-height: 1.4; }
    .hc-tagline:empty { display: none; }
    .hc-summary { font-size: 12px; color: #3d4752; line-height: 1.5;
                  margin-bottom: 10px;
                  display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical;
                  overflow: hidden; }
    .hc-summary:empty { display: none; }
    .hc-facets { border-top: 1px solid rgba(0,0,0,0.08); padding-top: 8px;
                 font-size: 11.5px; color: #424a53; }
    .hc-facet { display: flex; gap: 8px; margin-top: 2px; }
    .hc-facet-name { font-weight: 600; min-width: 110px; color: #57606a; }
    .hc-facet-val { color: #1f2328; }
    """

    # ── Marker sizes: sqrt of stars, scaled to a sensible pixel range ───────
    star_counts = df["stargazers_count"].values.astype(float)
    sizes = np.sqrt(star_counts)
    marker_sizes = 3 + 15 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1.0)

    fig = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=df["full_name"].tolist(),
        hover_text_html_template=hover_template,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        on_click="window.open(`https://github.com/{full_name}`,'_blank')",
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        title=f"Typologist Facets ({exp_name})",
        sub_title=f"Top {N_DOCS} most-starred GitHub repositories, colored by discovered facets",
        enable_search=True,
        search_field="",
        custom_css=custom_css,
        font_family="IBM Plex Sans",
        darkmode=False,
    )
    fig.save(str(output_html))
    print(f"[{exp_name}] Saved → {output_html}")


def _discover_experiments() -> list[str]:
    """Find all data/experiments/typologist_*/ dirs that have a schema.json."""
    if not EXPERIMENTS_BASE.exists():
        return []
    found = []
    for child in sorted(EXPERIMENTS_BASE.iterdir()):
        if child.is_dir() and child.name.startswith(EXPERIMENT_PREFIX):
            if (child / "schema.json").exists() and (child / "labels.parquet").exists():
                found.append(child.name[len(EXPERIMENT_PREFIX):])
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="vanilla",
        help='Experiment name (e.g. "vanilla", "lang_erased") or "all" to render every '
        "experiment dir found under data/experiments/.",
    )
    args = parser.parse_args()

    df_all = pd.read_parquet(REPOS_PARQUET)
    coords_all = np.load(UMAP_COORDS_NPZ)["coords"]
    toponymy_all = pd.read_parquet(LABELS_PARQUET)

    df, coords = _select_top_n(df_all, coords_all, N_DOCS)
    toponymy = _align_by_full_name(toponymy_all, df["full_name"])
    layer_cols = sorted(
        (c for c in toponymy.columns if c.startswith("label_layer_")),
        key=lambda c: int(c.split("_")[-1]),
    )
    if not layer_cols:
        raise RuntimeError(f"No label_layer_* columns found in {LABELS_PARQUET}")
    label_layers = [toponymy[c].values for c in layer_cols]

    if args.exp == "all":
        names = _discover_experiments()
        if not names:
            print(f"No typologist experiments found under {EXPERIMENTS_BASE}")
            return
        print(f"Rendering {len(names)} experiment(s): {names}")
    else:
        names = [args.exp]

    for name in names:
        _render_one(name, df, coords, label_layers)


if __name__ == "__main__":
    main()
