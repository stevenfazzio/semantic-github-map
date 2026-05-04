"""Query-answerability evaluation: can each schema answer canonical queries?

Three-phase pipeline:

  Phase 1 (--generate): Ask Opus for ~20 diverse questions/search tasks a
    user might want to perform on the corpus, saved to questions.json.
    Run once and reuse.

  Phase 2 (default): For each (variant, question) pair, ask Haiku whether
    the variant's schema can answer the question, with a filter expression
    if so. Saved to per-variant scoring JSONs.

  Phase 3 (--report): Aggregate per-variant scores and per-question
    category breakdowns into a summary table.

Run:
    uv run python experiments/typologist_query_eval.py --generate
    uv run python experiments/typologist_query_eval.py \\
        --variants vanilla,abl_leace_only_vanilla,abl_priors_only_vanilla,abl_single_pass_vanilla
    uv run python experiments/typologist_query_eval.py --report

Outputs:
    data/experiments/typologist_queries/questions.json
    data/experiments/typologist_queries/scoring_<variant>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import ANTHROPIC_API_KEY, DATA_DIR  # noqa: E402

EXPERIMENTS_BASE = DATA_DIR / "experiments"
QUERIES_OUT_DIR = EXPERIMENTS_BASE / "typologist_queries"
EXPERIMENT_PREFIX = "typologist_"

QUESTION_GEN_MODEL = "claude-opus-4-7"
EVAL_MODEL = "claude-haiku-4-5"

N_QUESTIONS = 20
EVAL_CONCURRENCY = 8

CORPUS_DESCRIPTION = (
    # Match what Typologist itself was given (object_description=
    # "GitHub repository READMEs", corpus_description=
    # "top 500 most-starred GitHub repositories"). Don't enumerate corpus
    # categories here; doing so primes the question generator to ask about
    # exactly the topics we listed, which artificially favors any schema
    # that picked up those topics.
    "GitHub repository READMEs from top 500 most-starred GitHub repositories."
)

# Same display map used elsewhere so output matches the rest of the analysis.
_NAME_TO_DIMENSIONS = {
    "vanilla":              ("full",         "vanilla"),
    "lang_erased":          ("full",         "lang_erased"),
    "owner_erased":         ("full",         "owner_erased"),
    "lang_owner_erased":    ("full",         "lang_owner_erased"),
    "stage03_erased":       ("full",         "stage03_erased"),
    "kitchen_sink":         ("full",         "kitchen_sink"),
    "abl_leace_only_vanilla":     ("leace_only",  "vanilla"),
    "abl_leace_only_ks":          ("leace_only",  "kitchen_sink"),
    "abl_priors_only_vanilla":    ("priors_only", "vanilla"),
    "abl_priors_only_ks":         ("priors_only", "kitchen_sink"),
    "abl_single_pass_vanilla":    ("single_pass", "vanilla"),
    "abl_single_pass_ks":         ("single_pass", "kitchen_sink"),
}


def _display_name(variant: str, all_variants: list[str]) -> str:
    mech, cond = _NAME_TO_DIMENSIONS.get(variant, (variant, ""))
    conds = {_NAME_TO_DIMENSIONS.get(v, (v, ""))[1] for v in all_variants}
    if len(conds) <= 1:
        return mech
    return f"{mech}@{cond}"


def _strip_fences(s: str) -> str:
    s = re.sub(r"^```(?:json)?\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s)
    return s


def _format_schema(records: list[dict]) -> str:
    lines = []
    for r in records:
        lines.append(f"  {r['name']}")
        defn = (r.get("definition") or "").strip()
        if defn:
            lines.append(f"    Definition: {defn}")
        lines.append("    Values:")
        for v in r.get("values", []):
            lines.append(f"      - {v}")
        lines.append("")
    return "\n".join(lines)


# ── Phase 1: Question generation ─────────────────────────────────────────────


def _build_question_gen_prompt(n: int) -> str:
    return f"""You are designing an evaluation set for faceted-search systems over a corpus of GitHub repositories.

Corpus: {CORPUS_DESCRIPTION}

Produce {n} diverse, concrete questions or search tasks a user might want to perform on this corpus. Aim for diversity: questions about what the projects ARE, who they're FOR, what kinds of things they SHIP, how they're written, how popular they are, what they're built with, and combinations of these.

Each question should be:
- Specific enough to have a definite "answer" or "matching set" within the corpus
- Phrased the way a real user would ask, not as a formal database query
- Independent (don't make a series of related questions; each should stand alone)

Respond with only valid JSON of this form:
{{
  "questions": [
    {{"id": 1, "question": "...", "category": "<one of: content_domain, audience, artifact_type, format_or_style, technical_attribute, popularity_or_age, cross_cutting, other>"}}
  ]
}}

Return exactly {n} questions.
"""


def _generate_questions(client: anthropic.Anthropic, n: int) -> list[dict]:
    prompt = _build_question_gen_prompt(n)
    resp = client.messages.create(
        model=QUESTION_GEN_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text
    parsed = json.loads(_strip_fences(raw))
    questions = parsed["questions"]
    if len(questions) != n:
        print(f"  Warning: requested {n} questions, got {len(questions)}")
    return questions


# ── Phase 2: Per-cell scoring ────────────────────────────────────────────────


def _build_eval_prompt(question: str, schema_records: list[dict]) -> str:
    return f"""You are evaluating whether a faceted-search schema can answer a specific user query.

Corpus context: {CORPUS_DESCRIPTION}

User query: {question}

Schema (a list of facets, each a categorical axis with named values; every repo gets one value per facet):

{_format_schema(schema_records)}

Can a user answer this query by filtering the corpus on this schema's facets? Be honest and strict: if the schema doesn't have a facet that captures what the query is asking about, the answer is no.

Respond with only valid JSON of this form:
{{
  "answerable": "<yes, partial, or no>",
  "reasoning": "<one sentence>",
  "example_filter": "<the facet=value filter combination that would answer it, or empty string if not answerable>"
}}

Definitions:
- "yes": one or more facet=value combinations express this query directly.
- "partial": the schema can narrow the result set but cannot fully answer the query without manual review of remaining repos.
- "no": no combination of facet values expresses the query.
"""


def _score_one(client: anthropic.Anthropic, question: str, schema_records: list[dict]) -> dict:
    prompt = _build_eval_prompt(question, schema_records)
    resp = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {"answerable": "?", "reasoning": "parse_error", "example_filter": "", "_raw": raw}


def _score_variant(client: anthropic.Anthropic, variant: str, questions: list[dict], resume: bool) -> dict:
    schema_path = EXPERIMENTS_BASE / f"{EXPERIMENT_PREFIX}{variant}" / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"No schema at {schema_path}")
    schema_records = json.loads(schema_path.read_text())

    out_path = QUERIES_OUT_DIR / f"scoring_{variant}.json"
    if resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        if len(existing.get("results", [])) == len(questions):
            print(f"  [{variant}] skipping (resume) — {len(questions)} questions already scored")
            return existing

    print(f"  [{variant}] scoring {len(questions)} questions across {len(schema_records)} facets...")

    results: list[dict] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as ex:
        future_to_idx = {
            ex.submit(_score_one, client, q["question"], schema_records): i
            for i, q in enumerate(questions)
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"answerable": "?", "reasoning": f"error: {e}", "example_filter": ""}

    output = {
        "variant": variant,
        "n_questions": len(questions),
        "results": [{**q, **r} for q, r in zip(questions, results)],
    }
    out_path.write_text(json.dumps(output, indent=2))
    return output


# ── Phase 3: Aggregation ─────────────────────────────────────────────────────


def _score_value(answerable: str) -> float:
    if answerable == "yes":
        return 1.0
    if answerable == "partial":
        return 0.5
    return 0.0


def _aggregate(scoring_files: list[Path]) -> tuple[list[dict], dict]:
    """Load all per-variant scoring files. Return (per_variant_summary, by_category)."""
    per_variant = []
    by_category: dict[str, dict[str, list[float]]] = {}  # category -> {variant -> [scores]}

    for path in scoring_files:
        data = json.loads(path.read_text())
        variant = data["variant"]
        scores = [_score_value(r.get("answerable", "?")) for r in data["results"]]
        n_yes = sum(1 for r in data["results"] if r.get("answerable") == "yes")
        n_partial = sum(1 for r in data["results"] if r.get("answerable") == "partial")
        n_no = sum(1 for r in data["results"] if r.get("answerable") == "no")
        n_other = len(scores) - n_yes - n_partial - n_no
        per_variant.append(
            {
                "variant": variant,
                "score": sum(scores) / max(len(scores), 1),
                "n_yes": n_yes,
                "n_partial": n_partial,
                "n_no": n_no,
                "n_other": n_other,
                "total": len(scores),
            }
        )
        for r in data["results"]:
            cat = r.get("category", "uncategorized")
            by_category.setdefault(cat, {}).setdefault(variant, []).append(
                _score_value(r.get("answerable", "?"))
            )

    return per_variant, by_category


def _report(scoring_files: list[Path]) -> None:
    if not scoring_files:
        print("No scoring files found. Run Phase 2 first.")
        return

    per_variant, by_category = _aggregate(scoring_files)
    variants = [v["variant"] for v in per_variant]

    print("\n" + "=" * 75)
    print("Per-variant query-answerability score")
    print("=" * 75)
    per_variant.sort(key=lambda v: -v["score"])
    print(f"  {'variant':<22}  {'score':>6}   yes / partial / no   ({'total':>5})")
    for v in per_variant:
        disp = _display_name(v["variant"], variants)
        print(
            f"  {disp:<22}  {v['score']:>6.2%}   "
            f"{v['n_yes']:>3} / {v['n_partial']:>7} / {v['n_no']:>3}   ({v['total']:>5})"
        )

    print("\n" + "=" * 75)
    print("Score by question category")
    print("=" * 75)
    cat_order = sorted(by_category.keys())
    print(f"  {'category':<22}  " + "  ".join(f"{_display_name(v, variants):>14}" for v in variants))
    for cat in cat_order:
        row = f"  {cat:<22}"
        for var in variants:
            scores = by_category[cat].get(var, [])
            if not scores:
                row += "  " + " " * 14
                continue
            mean = sum(scores) / len(scores)
            row += f"  {mean:>10.0%} (n={len(scores)})".rjust(16)
        print(row)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Phase 1: generate questions.json (overwrites if exists)")
    parser.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    parser.add_argument("--variants", default="",
                        help="Comma-separated variants to score (Phase 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip variants whose scoring file already exists with full results")
    parser.add_argument("--report", action="store_true",
                        help="Phase 3: aggregate and print summary across all scoring files")
    args = parser.parse_args()

    QUERIES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    questions_path = QUERIES_OUT_DIR / "questions.json"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if args.generate:
        print(f"Phase 1: generating {args.n_questions} questions via {QUESTION_GEN_MODEL}...")
        questions = _generate_questions(client, args.n_questions)
        questions_path.write_text(json.dumps({"questions": questions}, indent=2))
        print(f"  Saved → {questions_path}")
        for q in questions[:3]:
            print(f"  ex: [{q.get('category')}] {q.get('question')}")
        if len(questions) > 3:
            print(f"  ... ({len(questions) - 3} more)")
        return

    if args.report:
        scoring_files = sorted(QUERIES_OUT_DIR.glob("scoring_*.json"))
        _report(scoring_files)
        return

    # Phase 2
    if not args.variants:
        print("Specify --variants for Phase 2 scoring, or --generate / --report.")
        return
    if not questions_path.exists():
        print(f"No questions at {questions_path}. Run --generate first.")
        return

    questions = json.loads(questions_path.read_text())["questions"]
    variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]

    print(f"Phase 2: scoring {len(variants_list)} variants × {len(questions)} questions via {EVAL_MODEL}")
    for v in variants_list:
        _score_variant(client, v, questions, args.resume)

    print("\nDone. Run --report to see aggregate scores.")


if __name__ == "__main__":
    main()
