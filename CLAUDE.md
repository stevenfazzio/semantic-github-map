# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python data pipeline that analyzes the top 10,000 most-starred GitHub repositories. It enumerates candidates via BigQuery, fetches repo data via GraphQL direct lookups, embeds READMEs, reduces dimensionality, labels topics via LLM-driven topic modeling, and produces an interactive 2D visualization map (`data/github_map.html`).

## Running the Pipeline

Each stage is a standalone script run in order. A Makefile provides shortcuts for common dev tasks.

```bash
uv sync                               # Install from lockfile
uv sync --extra bigquery              # (optional) for pipeline/00_enumerate_repos.py
# Or without uv: pip install -e . / pip install -e '.[bigquery]'

python pipeline/00_enumerate_repos.py          # BigQuery → data/candidates.csv (or use committed fallback)
python pipeline/01_fetch_repos.py              # GraphQL direct lookups → repos.parquet
python pipeline/02_select_top_repos.py         # Trim repos.parquet to top N by stars
python pipeline/03_summarize_readmes.py        # Add LLM summaries, taglines, and target audience via Claude Haiku → updates repos.parquet
python pipeline/04_embed_readmes.py            # Embed READMEs via Cohere → embeddings.npz
python pipeline/05_reduce_umap.py              # UMAP 512-dim → 2D → umap_coords.npz
python pipeline/06_label_topics.py             # Toponymy + Claude Sonnet topic labels → labels.parquet
python pipeline/07_visualize.py                # DataMapPlot interactive HTML → data/github_map.html + docs/index.html
python pipeline/08_fetch_dependencies.py       # GraphQL dep-graph crawl → data/dependencies.parquet + dependency_sources.parquet
python pipeline/09_enrich_external.py          # Cheap metadata for out-of-set targets → data/external_repos.parquet
python pipeline/10_build_graph.py              # Unified node + edge tables → data/nodes.parquet + edges.parquet
```

Stages 08-10 are an exploratory branch separate from the main visualization
pipeline (00-07): they produce a dependency graph dataset for offline analysis,
not visualization output.

**Two-phase fetch approach:** Step 00 queries GH Archive on BigQuery for repos with significant recent star activity, producing a generous candidate list (~25K). Step 01 then looks up each candidate via `repository(owner:, name:)` GraphQL queries (batched 50 per request), sorts by stars, and keeps the top 10K. This avoids the Search API's 1,000-result cap and non-deterministic ordering.

**Fallback for users without GCP:** A `candidates.csv` is committed to the repo root. If `data/candidates.csv` doesn't exist, `pipeline/01_fetch_repos.py` copies from the committed file automatically. Most contributors never need BigQuery access.

`experiments/compare_toponymy_configs.py` is a side branch for comparing Toponymy configurations, outputting to `data/experiments/`.

## Required Environment Variables

Set in `.env` (loaded by `python-dotenv`):
- `GITHUB_TOKEN` — used by `pipeline/01_fetch_repos.py`
- `CO_API_KEY` — Cohere API key, used by `pipeline/04_embed_readmes.py`, `pipeline/06_label_topics.py`, `experiments/compare_toponymy_configs.py`
- `ANTHROPIC_API_KEY` — used by `pipeline/03_summarize_readmes.py`, `pipeline/06_label_topics.py`, `experiments/compare_toponymy_configs.py`
- `GCP_PROJECT` — (optional) Google Cloud project ID, used by `pipeline/00_enumerate_repos.py`

## Architecture

**Sequential data pipeline** — each script reads outputs from previous stages:

```
candidates.csv ──> metadata.parquet ──> repos.parquet ──┬──> embeddings.npz ──> umap_coords.npz ──> labels.parquet
                                                        │                                                │
                                                        └────────────────────────────────────────────────┴──> github_map.html
```

`metadata.parquet` is the intermediate output of `pipeline/01_fetch_repos.py`'s metadata-only pass; `repos.parquet` is the final output after README fetching.

**`pipeline/config.py`** is the central configuration hub: all file paths, API keys, and constants (batch sizes, model names, target counts) are defined here. Every pipeline script imports from it.

**Key technology choices:**
- BigQuery (GH Archive) for reliable repo enumeration
- GraphQL `repository()` direct lookups for metadata + READMEs (batched 25–50/request)
- Cohere `embed-v4.0` (512-dim) for README embeddings
- UMAP (n_neighbors=15, min_dist=0.05, cosine) for dimensionality reduction (512D → 2D)
- Toponymy library for hierarchical topic clustering with LLM-generated labels
- DataMapPlot for the final interactive HTML visualization
- Claude Haiku for README summarization, Claude Sonnet for topic naming

## Data Pipeline Rules

- NEVER overwrite existing parquet/data files in-place. Always write to new files (e.g., with timestamp or version suffix) and only replace originals after verification. Treat fetched data as expensive/irreplaceable.
- For GitHub API calls (GraphQL or REST), always implement retry with exponential backoff, handling 502, ReadTimeout, and ChunkedEncodingError. Never assume a single request pattern will scale.

## Data Directory

All outputs go to `data/` (gitignored). Key files: `candidates.csv`, `metadata.parquet`, `repos.parquet`, `repos_pretrim.parquet` (backup of repos.parquet before trimming to top 10K), `embeddings.npz`, `umap_coords.npz`, `labels.parquet`, `toponymy_model.joblib`, `github_map.html`. Dependency graph (stages 08-10): `dependencies.parquet`, `dependency_sources.parquet`, `external_repos.parquet`, `nodes.parquet`, `edges.parquet`.

## Dependency Graph Data (stages 08-10)

Stage 08 caps each source repo at 30 manifests (`MAX_MANIFEST_PAGES = 10` * `MANIFEST_FIRST = 3`) to prevent monorepos like microsoft/vscode (220 manifests, mostly `.github/workflows/*.yml`) from blocking workers for many minutes. The cap drops mostly redundant ACTIONS workflow edges in the long tail. `dependency_sources.parquet` records per-source provenance: `was_truncated == True` means the cap fired; `status == "timedout"` with `manifest_pages_fetched > 0` means GitHub's index gave up partway. See the `pipeline/08_fetch_dependencies.py` docstring for the full coverage-state taxonomy.

## Visualization Details

`pipeline/07_visualize.py` produces the main output — an interactive HTML map with multiple colormaps (language, stars, license, age), hover tooltips with summaries, click-to-open-repo, and search. Toponymy's hierarchical cluster layers are passed to DataMapPlot for multi-level topic label display.

**Label layer order (fixed 2026-07-09):** `create_interactive_plot` expects label layers FINEST-first, coarsest last (its docstring). `labels.parquet` stores `label_layer_0` = coarsest, so stage 07 reverses at the call site. The repo originally passed coarsest-first (the old stage-06 comment asserting "DataMapPlot expects coarsest first" was wrong), which inverted label zoom gating so hyper-specific labels dominated the overview zoom.

## docs/ Directory: Source vs. Generated Files

- `docs/methodology.html` — **hand-authored source file** with a `<!-- DATA_AS_OF -->` placeholder. Step 07 replaces the placeholder with the actual date in-place. Edit it directly for methodology content changes.
- `docs/index.html` — **generated output** from `pipeline/07_visualize.py` (copy of `data/github_map.html` with adjusted links). Do not edit directly.
- `data/methodology.html` — **generated copy** of `docs/methodology.html` with nav links adjusted (`index.html` → `github_map.html`) and data date filled in for local use.
- `docs/filter_panel.html` — **hand-authored HTML snippet** injected by `pipeline/07_visualize.py` into the final map as the filter sidebar.

## Development

**Makefile targets:**
- `make install` — `uv sync --extra dev` (includes test and lint deps)
- `make lint` — run ruff check + format check
- `make format` — auto-format with ruff
- `make test` — run pytest suite
- `make pipeline` — run all pipeline stages in order

**Testing:** pytest tests live in `tests/`. They test config loading, fetch retry logic, summarization batching, and visualization output without requiring API keys or data files.

**Pre-commit hooks:** ruff check and ruff format run automatically on commit. Install with `pre-commit install`.

**CI:** GitHub Actions (`.github/workflows/ci.yml`) runs lint and test on every push and PR.
