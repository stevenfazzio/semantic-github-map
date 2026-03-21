"""Experiment: evaluate local open-source models for Toponymy topic naming.

Phase 1 — Swap only the LLM + Toponymy's internal embedder (reuse existing
Cohere embeddings and UMAP coords). Tests 5 Ollama LLMs with sentence-transformers.

Phase 2 — Re-embed all 10K READMEs with local embedding models, re-run UMAP,
and run Toponymy with the best LLM from Phase 1.

MLX mode (--mlx) — Run qwen2.5-0.5B via MLX on Apple Silicon using Toponymy's
AsyncOpenAINamer pointed at mlx_lm.server. Monkey-patches two bugs in the wrapper.
"""

import argparse
import copy
import itertools
import subprocess
import sys
import tempfile
import time

import nest_asyncio
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.audit import (
    create_comparison_df,
    create_keyphrase_analysis_df,
    create_layer_summary_df,
)
from toponymy.llm_wrappers import AsyncOllamaNamer, LLMWrapper

nest_asyncio.apply()

from pipeline.config import (
    EMBEDDINGS_NPZ,
    EXPERIMENTS_DIR,
    REPOS_PARQUET,
    UMAP_COORDS_NPZ,
)

# ── Output directory ─────────────────────────────────────────────────────────
LOCAL_MODELS_DIR = EXPERIMENTS_DIR / "local_models"
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Sentence-transformers embedder for Toponymy ─────────────────────────────


class SentenceTransformerEmbedder:
    """Drop-in replacement for CohereEmbedder using sentence-transformers.

    Toponymy calls ``embedder.encode(texts, show_progress_bar=...)`` on the
    text_embedding_model. We wrap SentenceTransformer to match that interface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name

    def encode(self, texts, show_progress_bar=False, **kwargs):
        return self._model.encode(texts, show_progress_bar=show_progress_bar)


# ── Experiment configs ───────────────────────────────────────────────────────

PHASE1_EMBEDDER_MODEL = "all-MiniLM-L6-v2"

PHASE1_EXPERIMENTS = [
    {"name": "smollm2_135m", "ollama_model": "smollm2:135m"},
    {"name": "smollm2_360m", "ollama_model": "smollm2:360m"},
    {"name": "qwen2.5_0.5b", "ollama_model": "qwen2.5:0.5b"},
    {"name": "smollm2_1.7b", "ollama_model": "smollm2:1.7b"},
    {"name": "gemma3_1b", "ollama_model": "gemma3:1b"},
]

PHASE2_EMBEDDING_MODELS = [
    {"name": "phase2_minilm", "model": "all-MiniLM-L6-v2"},
    {"name": "phase2_nomic", "model": "nomic-ai/nomic-embed-text-v1.5"},
]

MLX_EXPERIMENTS = [
    {"name": "mlx_qwen2.5_0.5b", "mlx_model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
]

# Toponymy settings (mirrors pipeline/06_label_topics.py)
TOPONYMY_DEFAULTS = {
    "min_clusters": 4,
    "object_description": "GitHub repository descriptions",
    "corpus_description": "collection of the top 10,000 most-starred GitHub repositories",
    "exemplar_delimiters": ['    * """', '"""\n'],
    "lowest_detail_level": 0.5,
    "highest_detail_level": 1.0,
}

# ── Data loading ─────────────────────────────────────────────────────────────


def load_data():
    """Load shared data used across all experiments."""
    df = pd.read_parquet(REPOS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

    MAX_README_CHARS = 2_000
    has_summary = "summary" in df.columns
    has_tagline = "tagline" in df.columns
    has_title = "project_title" in df.columns

    documents = []
    for _, row in df.iterrows():
        summary = ""
        if has_summary and isinstance(row.get("summary"), str):
            summary = row["summary"].strip()

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

    return df, embeddings, coords, documents


# ── Preflight checks ────────────────────────────────────────────────────────


def _ollama_is_running() -> bool:
    """Check if the Ollama server is reachable."""
    try:
        import ollama

        ollama.Client().list()
        return True
    except Exception:
        return False


def _get_available_ollama_models() -> set[str]:
    """Return set of locally available Ollama model names."""
    try:
        import ollama

        models = ollama.Client().list().models
        return {m.model for m in models}
    except Exception:
        return set()


def _pull_missing_models(required_models: list[str]) -> None:
    """Offer to pull any missing Ollama models."""
    available = _get_available_ollama_models()
    missing = [m for m in required_models if m not in available]

    if not missing:
        print("All required Ollama models are available.")
        return

    print(f"\nMissing Ollama models: {', '.join(missing)}")
    answer = input("Pull missing models now? [Y/n] ").strip().lower()
    if answer and answer != "y":
        print("Aborting — missing models are required.")
        sys.exit(1)

    for model in missing:
        print(f"Pulling {model}...")
        subprocess.run(["ollama", "pull", model], check=True)

    print("All models pulled.")


def validate_preflight(df, embeddings, coords, phase2: bool = False):
    """Run preflight checks before the experiment loop."""
    errors = []

    if embeddings.shape[0] != len(df):
        errors.append(f"embeddings rows ({embeddings.shape[0]}) != df rows ({len(df)})")
    if coords.shape[0] != len(df):
        errors.append(f"coords rows ({coords.shape[0]}) != df rows ({len(df)})")

    if not _ollama_is_running():
        errors.append("Ollama is not running — start it with `ollama serve`")

    try:
        with tempfile.NamedTemporaryFile(dir=LOCAL_MODELS_DIR, delete=True):
            pass
    except Exception as e:
        errors.append(f"Cannot write to {LOCAL_MODELS_DIR}: {e}")

    if errors:
        raise RuntimeError("Preflight checks failed:\n  " + "\n  ".join(errors))

    # Pull missing models
    required = [exp["ollama_model"] for exp in PHASE1_EXPERIMENTS]
    _pull_missing_models(required)

    # Quick connectivity test with the smallest model
    print("Testing Ollama connectivity...")
    try:
        wrapper = AsyncOllamaNamer(model=PHASE1_EXPERIMENTS[0]["ollama_model"])
        result = wrapper.test_llm_connectivity("Say hello in one word.")
        print(f"  Ollama test response: {result.strip()!r}")
    except Exception as e:
        raise RuntimeError(f"Ollama connectivity test failed: {e}")

    print("Preflight checks passed.")


# ── Experiment runner ────────────────────────────────────────────────────────


def extract_labels(model, documents):
    """Extract coarse/fine labels from a fitted Toponymy model."""
    n_layers = len(model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("No cluster layers found")

    coarse_layer = model.cluster_layers_[-1]
    fine_layer = model.cluster_layers_[0]

    coarse_labels = [str(coarse_layer.topic_name_vector[i]) for i in range(len(documents))]
    fine_labels = [str(fine_layer.topic_name_vector[i]) for i in range(len(documents))]
    return coarse_labels, fine_labels


def _sanitize_topic_names(model):
    """Coerce any non-string topic names (e.g. dicts from Ollama) to strings."""
    for layer in model.cluster_layers_:
        if hasattr(layer, "topic_names"):
            layer.topic_names = [str(t) if not isinstance(t, str) else t for t in layer.topic_names]
        if hasattr(layer, "topic_name_vector"):
            layer.topic_name_vector = [
                str(t) if not isinstance(t, str) else t for t in layer.topic_name_vector
            ]


def save_audit_csvs(model, exp_dir):
    """Save audit CSVs for a fitted model."""
    _sanitize_topic_names(model)

    try:
        layer_summary = create_layer_summary_df(model)
        layer_summary.to_csv(exp_dir / "audit_layer_summary.csv", index=False)
    except Exception as e:
        print(f"  Warning: could not create layer summary: {e}")

    n_layers = len(model.cluster_layers_)
    for i in range(n_layers):
        try:
            comp = create_comparison_df(model, layer_index=i)
            comp.to_csv(exp_dir / f"audit_comparison_layer{i}.csv", index=False)
        except Exception as e:
            print(f"  Warning: could not create comparison for layer {i}: {e}")

        try:
            kp = create_keyphrase_analysis_df(model, layer_index=i)
            kp.to_csv(exp_dir / f"audit_keyphrase_layer{i}.csv", index=False)
        except Exception as e:
            print(f"  Warning: could not create keyphrase analysis for layer {i}: {e}")


def run_single_experiment(
    name: str,
    llm_wrapper,
    embedder,
    df,
    embeddings,
    coords,
    documents,
    base_clusterer,
    exp_dir,
):
    """Fit Toponymy for a single experiment config and save outputs."""
    exp_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {name}")
    print(f"{'=' * 60}")

    clusterer = copy.deepcopy(base_clusterer)

    topic_model = Toponymy(
        llm_wrapper=llm_wrapper,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description=TOPONYMY_DEFAULTS["object_description"],
        corpus_description=TOPONYMY_DEFAULTS["corpus_description"],
        exemplar_delimiters=TOPONYMY_DEFAULTS["exemplar_delimiters"],
        lowest_detail_level=TOPONYMY_DEFAULTS["lowest_detail_level"],
        highest_detail_level=TOPONYMY_DEFAULTS["highest_detail_level"],
    )

    start = time.time()
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )
    elapsed = time.time() - start
    print(f"  Toponymy fit completed in {elapsed:.1f}s")

    _sanitize_topic_names(topic_model)

    # Save labels
    coarse_labels, fine_labels = extract_labels(topic_model, documents)
    labels_df = pd.DataFrame(
        {
            "full_name": df["full_name"],
            "coarse_label": coarse_labels,
            "fine_label": fine_labels,
        }
    )
    labels_df.to_parquet(exp_dir / "labels.parquet", index=False)
    print(f"  Saved labels to {exp_dir / 'labels.parquet'}")

    save_audit_csvs(topic_model, exp_dir)
    print(f"  Saved audit CSVs to {exp_dir}")

    # Save timing
    (exp_dir / "timing.txt").write_text(f"{elapsed:.1f}s\n")

    # Disambiguation stats
    disambig_rows = []
    for li, layer in enumerate(topic_model.cluster_layers_):
        indices = getattr(layer, "dismbiguation_topic_indices", None)
        if indices is not None:
            disambig_rows.append(
                {
                    "layer": li,
                    "num_groups": len(indices),
                    "topics_renamed": sum(len(g) for g in indices),
                    "total_topics": len(layer.topic_names),
                }
            )
    if disambig_rows:
        pd.DataFrame(disambig_rows).to_csv(exp_dir / "disambiguation.csv", index=False)

    return topic_model


def run_phase1(df, embeddings, coords, documents, resume=False):
    """Run Phase 1: swap LLM + Toponymy embedder, keep Cohere pipeline embeddings."""
    print("\n" + "=" * 60)
    print("PHASE 1: Local LLM experiments (reusing Cohere embeddings)")
    print("=" * 60)

    embedder = SentenceTransformerEmbedder(model_name=PHASE1_EMBEDDER_MODEL)
    print(f"Loaded sentence-transformers embedder: {PHASE1_EMBEDDER_MODEL}")

    base_clusterer = ToponymyClusterer(min_clusters=TOPONYMY_DEFAULTS["min_clusters"])
    base_clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    for exp in PHASE1_EXPERIMENTS:
        name = exp["name"]
        exp_dir = LOCAL_MODELS_DIR / name

        if resume and (exp_dir / "labels.parquet").exists():
            print(f"\n{'=' * 60}")
            print(f"Skipping (resume): {name}")
            print(f"{'=' * 60}")
            continue

        llm = AsyncOllamaNamer(model=exp["ollama_model"])
        run_single_experiment(
            name=name,
            llm_wrapper=llm,
            embedder=embedder,
            df=df,
            embeddings=embeddings,
            coords=coords,
            documents=documents,
            base_clusterer=base_clusterer,
            exp_dir=exp_dir,
        )


def run_phase2(df, documents, resume=False):
    """Run Phase 2: re-embed READMEs with local models, re-run UMAP, re-run Toponymy."""
    import umap

    print("\n" + "=" * 60)
    print("PHASE 2: Local embedding experiments")
    print("=" * 60)

    # Determine best LLM from Phase 1 (use phi4_mini as default / best expected)
    best_llm_name = "phi4-mini:3.8b"
    best_llm_exp = "phi4_mini"

    # Check if Phase 1 ran and pick the one with most unique fine topics as a proxy
    best_unique = 0
    for exp in PHASE1_EXPERIMENTS:
        labels_path = LOCAL_MODELS_DIR / exp["name"] / "labels.parquet"
        if labels_path.exists():
            ldf = pd.read_parquet(labels_path)
            n_unique = ldf["fine_label"].nunique()
            if n_unique > best_unique:
                best_unique = n_unique
                best_llm_name = exp["ollama_model"]
                best_llm_exp = exp["name"]

    print(f"Using best Phase 1 LLM: {best_llm_name} (from {best_llm_exp})")

    # Prepare texts for embedding (raw READMEs, matching pipeline/04)
    texts = []
    for _, row in df.iterrows():
        readme = row["readme"].strip() if isinstance(row["readme"], str) else ""
        texts.append(readme if readme else row.get("description", row["full_name"]))

    for emb_cfg in PHASE2_EMBEDDING_MODELS:
        name = emb_cfg["name"]
        exp_dir = LOCAL_MODELS_DIR / name

        if resume and (exp_dir / "labels.parquet").exists():
            print(f"\n{'=' * 60}")
            print(f"Skipping (resume): {name}")
            print(f"{'=' * 60}")
            continue

        exp_dir.mkdir(exist_ok=True)

        # Embed
        print(f"\nEmbedding {len(texts)} READMEs with {emb_cfg['model']}...")
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(emb_cfg["model"], trust_remote_code=True)
        start = time.time()
        local_embeddings = st_model.encode(texts, show_progress_bar=True, batch_size=64)
        embed_time = time.time() - start
        print(f"  Embedding completed in {embed_time:.1f}s, shape: {local_embeddings.shape}")

        np.savez(exp_dir / "embeddings.npz", embeddings=local_embeddings)

        # UMAP (same params as pipeline/05)
        print("  Running UMAP...")
        start = time.time()
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.05,
            metric="cosine",
            random_state=42,
        )
        local_coords = reducer.fit_transform(local_embeddings)
        umap_time = time.time() - start
        print(f"  UMAP completed in {umap_time:.1f}s")

        np.savez(exp_dir / "umap_coords.npz", coords=local_coords)

        # Toponymy
        embedder = SentenceTransformerEmbedder(model_name=emb_cfg["model"])
        llm = AsyncOllamaNamer(model=best_llm_name)

        base_clusterer = ToponymyClusterer(min_clusters=TOPONYMY_DEFAULTS["min_clusters"])
        base_clusterer.fit(clusterable_vectors=local_coords, embedding_vectors=local_embeddings)

        run_single_experiment(
            name=name,
            llm_wrapper=llm,
            embedder=embedder,
            df=df,
            embeddings=local_embeddings,
            coords=local_coords,
            documents=documents,
            base_clusterer=base_clusterer,
            exp_dir=exp_dir,
        )

        # Save extra timing info
        with open(exp_dir / "timing.txt", "a") as f:
            f.write(f"embedding: {embed_time:.1f}s\n")
            f.write(f"umap: {umap_time:.1f}s\n")


# ── MLX experiment runner ────────────────────────────────────────────────────


class MLXLocalNamer(LLMWrapper):
    """Synchronous LLM wrapper using mlx_lm.generate() in-process.

    Runs the model directly on Apple Silicon via MLX — no HTTP server needed.
    Avoids async deadlocks and server freezes that occur with mlx_lm.server.
    """

    supports_system_prompts = True

    def __init__(self, mlx_model_name: str):
        from mlx_lm import load

        print(f"  Loading MLX model: {mlx_model_name}...")
        self._model, self._tokenizer = load(mlx_model_name)
        self._model_name = mlx_model_name
        print("  MLX model loaded")

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._generate(messages, temperature, max_tokens)

    def _call_llm_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._generate(messages, temperature, max_tokens)

    def _generate(self, messages, temperature, max_tokens):
        from mlx_lm import generate

        prompt_str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            return generate(
                self._model,
                self._tokenizer,
                prompt=prompt_str,
                max_tokens=max_tokens,
            )
        except Exception as e:
            from warnings import warn

            warn(f"MLX generate failed: {e!s:.100}")
            return ""

    def test_llm_connectivity(self, prompt="Say hello in one word."):
        return self._call_llm(prompt, temperature=0.0, max_tokens=10)


def run_mlx_experiments(df, embeddings, coords, documents, resume=False):
    """Run MLX experiments: qwen2.5-0.5B via mlx_lm in-process on Apple Silicon."""
    print("\n" + "=" * 60)
    print("MLX EXPERIMENTS: Local models via mlx_lm (in-process)")
    print("=" * 60)

    # Check mlx_lm is available
    try:
        import mlx_lm  # noqa: F401

        print("  mlx_lm is installed")
    except ImportError:
        print("ERROR: mlx_lm not installed. Run: uv sync --extra local-models")
        sys.exit(1)

    embedder = SentenceTransformerEmbedder(model_name=PHASE1_EMBEDDER_MODEL)
    print(f"Loaded sentence-transformers embedder: {PHASE1_EMBEDDER_MODEL}")

    base_clusterer = ToponymyClusterer(min_clusters=TOPONYMY_DEFAULTS["min_clusters"])
    base_clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    for exp in MLX_EXPERIMENTS:
        name = exp["name"]
        exp_dir = LOCAL_MODELS_DIR / name

        if resume and (exp_dir / "labels.parquet").exists():
            print(f"\nSkipping (resume): {name}")
            continue

        llm = MLXLocalNamer(mlx_model_name=exp["mlx_model"])

        # Quick connectivity test
        test_result = llm.test_llm_connectivity()
        print(f"  MLX test response: {test_result.strip()!r}")

        run_single_experiment(
            name=name,
            llm_wrapper=llm,
            embedder=embedder,
            df=df,
            embeddings=embeddings,
            coords=coords,
            documents=documents,
            base_clusterer=base_clusterer,
            exp_dir=exp_dir,
        )


# ── Comparison ───────────────────────────────────────────────────────────────


def compare_experiments(include_phase2: bool = False, include_mlx: bool = False):
    """Compare all completed experiments and write summary."""
    all_experiments = list(PHASE1_EXPERIMENTS)
    if include_phase2:
        all_experiments += [{"name": e["name"]} for e in PHASE2_EMBEDDING_MODELS]
    if include_mlx:
        all_experiments += [{"name": e["name"]} for e in MLX_EXPERIMENTS]

    experiment_names = []
    for exp in all_experiments:
        exp_dir = LOCAL_MODELS_DIR / exp["name"]
        if (exp_dir / "labels.parquet").exists():
            experiment_names.append(exp["name"])

    if not experiment_names:
        print("No completed experiments found.")
        return

    print(f"\n{'=' * 60}")
    print("Comparison across experiments")
    print(f"{'=' * 60}")
    print(f"Experiments: {', '.join(experiment_names)}")

    md_lines = ["# Local Models Experiment Comparison\n"]

    # Timing
    md_lines.append("## Timing\n")
    md_lines.append("| Experiment | Toponymy fit time |")
    md_lines.append("|---|---|")
    for name in experiment_names:
        timing_path = LOCAL_MODELS_DIR / name / "timing.txt"
        timing = timing_path.read_text().splitlines()[0] if timing_path.exists() else "N/A"
        md_lines.append(f"| {name} | {timing} |")
        print(f"  {name}: {timing}")
    md_lines.append("")

    # Audit summaries
    md_lines.append("## Audit Summaries\n")
    for name in experiment_names:
        exp_dir = LOCAL_MODELS_DIR / name
        print(f"\n-- {name} --")
        md_lines.append(f"### {name}\n")

        summary_path = exp_dir / "audit_layer_summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            print(summary_df.to_string(index=False))
            md_lines.append("**Layer summary:**\n")
            md_lines.append(summary_df.to_markdown(index=False))
            md_lines.append("")

        layer_idx = 0
        while (exp_dir / f"audit_comparison_layer{layer_idx}.csv").exists():
            comp = pd.read_csv(exp_dir / f"audit_comparison_layer{layer_idx}.csv")
            if "Final LLM Topic Name" in comp.columns:
                lengths = comp["Final LLM Topic Name"].astype(str).str.len()
                print(
                    f"  Layer {layer_idx} avg topic name length: {lengths.mean():.1f} chars "
                    f"(min {lengths.min()}, max {lengths.max()})"
                )
            layer_idx += 1

        for li in [0]:
            kp_path = exp_dir / f"audit_keyphrase_layer{li}.csv"
            if kp_path.exists():
                kp_df = pd.read_csv(kp_path)
                if "keyphrase_in_topic" in kp_df.columns:
                    rate = kp_df["keyphrase_in_topic"].mean()
                    print(f"  Keyphrase-in-topic rate (layer {li}): {rate:.1%}")
                    md_lines.append(f"Keyphrase-in-topic rate (layer {li}): {rate:.1%}\n")

    if len(experiment_names) < 2:
        print("\nOnly one experiment — skipping pairwise comparison.")
        md_lines.append("\nOnly one experiment — no pairwise comparison.\n")
        (LOCAL_MODELS_DIR / "comparison_summary.md").write_text("\n".join(md_lines))
        return

    # Pairwise comparisons
    md_lines.append("\n## Pairwise Comparisons\n")
    for layer_idx, label in [(0, "fine"), (-1, "coarse")]:
        print(f"\n-- {label} layer comparison --")
        md_lines.append(f"### {label.title()} layer\n")

        rows = []
        for name in experiment_names:
            exp_dir = LOCAL_MODELS_DIR / name
            if layer_idx < 0:
                actual_idx = 0
                while (exp_dir / f"audit_comparison_layer{actual_idx + 1}.csv").exists():
                    actual_idx += 1
                idx = actual_idx
            else:
                idx = layer_idx
            comp_path = exp_dir / f"audit_comparison_layer{idx}.csv"
            if comp_path.exists():
                comp = pd.read_csv(comp_path)
                comp = comp.rename(columns={"Final LLM Topic Name": f"topic_{name}"})
                rows.append((name, comp))

        if not rows:
            continue

        keyphrases_col = "Extracted Keyphrases (Top 5)"
        base_cols = ["Cluster ID", "Document Count"]
        if keyphrases_col in rows[0][1].columns:
            base_cols.append(keyphrases_col)
        merged = rows[0][1][base_cols + [f"topic_{rows[0][0]}"]].copy()
        for exp_name, comp in rows[1:]:
            right = comp[["Cluster ID", f"topic_{exp_name}"]].copy()
            merged = merged.merge(right, on="Cluster ID", how="outer")

        merged.to_csv(LOCAL_MODELS_DIR / f"comparison_{label}.csv", index=False)
        print(f"  Saved {LOCAL_MODELS_DIR / f'comparison_{label}.csv'}")

        topic_cols = [f"topic_{n}" for n in experiment_names]
        for col in topic_cols:
            if col in merged.columns:
                n_unique = merged[col].nunique()
                avg_len = merged[col].astype(str).str.len().mean()
                line = f"  Unique topics ({col}): {n_unique}, avg name length: {avg_len:.1f} chars"
                print(line)
                md_lines.append(f"- {line.strip()}")

        md_lines.append("")

        for name_a, name_b in itertools.combinations(experiment_names, 2):
            col_a, col_b = f"topic_{name_a}", f"topic_{name_b}"
            if col_a in merged.columns and col_b in merged.columns:
                agree = (merged[col_a] == merged[col_b]).mean()
                line = f"  Topic name agreement ({name_a} vs {name_b}): {agree:.1%}"
                print(line)
                md_lines.append(f"- {line.strip()}")

        md_lines.append("")

    # Keyphrase overlap (Jaccard) for fine layer
    print("\n-- Keyphrase overlap (fine, Jaccard) --")
    md_lines.append("## Keyphrase Overlap (fine layer, Jaccard)\n")
    kp_dfs = {}
    for name in experiment_names:
        kp_path = LOCAL_MODELS_DIR / name / "audit_comparison_layer0.csv"
        if kp_path.exists():
            kp = pd.read_csv(kp_path)
            if "Extracted Keyphrases (Top 5)" in kp.columns:
                kp_dfs[name] = kp.set_index("Cluster ID")["Extracted Keyphrases (Top 5)"]

    for name_a, name_b in itertools.combinations(experiment_names, 2):
        if name_a in kp_dfs and name_b in kp_dfs:
            common_ids = kp_dfs[name_a].index.intersection(kp_dfs[name_b].index)
            jaccard_scores = []
            for cid in common_ids:
                set_a = set(str(kp_dfs[name_a].get(cid, "")).split(", "))
                set_b = set(str(kp_dfs[name_b].get(cid, "")).split(", "))
                if set_a or set_b:
                    jaccard = len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0
                    jaccard_scores.append(jaccard)

            if jaccard_scores:
                mean_jaccard = np.mean(jaccard_scores)
                line = f"  Mean Jaccard similarity ({name_a} vs {name_b}): {mean_jaccard:.3f}"
                print(line)
                md_lines.append(f"- {line.strip()}")

    md_lines.append("\n---\n*Generated by experiments/local_models_toponymy.py*\n")
    (LOCAL_MODELS_DIR / "comparison_summary.md").write_text("\n".join(md_lines))
    print(f"\nSaved comparison summary to {LOCAL_MODELS_DIR / 'comparison_summary.md'}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate local models for Toponymy topic naming")
    parser.add_argument("--resume", action="store_true", help="Skip experiments with existing labels.parquet")
    parser.add_argument("--phase2", action="store_true", help="Also run Phase 2: local embeddings + UMAP")
    parser.add_argument("--mlx", action="store_true", help="Run MLX experiments (Apple Silicon, mlx_lm.server)")
    args = parser.parse_args()

    df, embeddings, coords, documents = load_data()
    print(f"Loaded {len(documents)} documents")

    if args.mlx:
        # MLX-only mode: skip Ollama preflight, run MLX experiments
        run_mlx_experiments(df, embeddings, coords, documents, resume=args.resume)
        compare_experiments(include_mlx=True)
    else:
        validate_preflight(df, embeddings, coords, phase2=args.phase2)
        run_phase1(df, embeddings, coords, documents, resume=args.resume)

        if args.phase2:
            run_phase2(df, documents, resume=args.resume)

        compare_experiments(include_phase2=args.phase2)

    print("\nDone.")


if __name__ == "__main__":
    main()
