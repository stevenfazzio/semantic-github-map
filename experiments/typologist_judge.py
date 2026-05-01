"""Ask Opus to judge which of two (or more) discovered schemas is more useful.

Each pair is judged twice with the schema order swapped so position bias is
visible. With --variants, runs an all-pairs tournament. Saves per-pair JSON.

Usage:
    # Single pair
    uv run python experiments/typologist_judge.py --a vanilla --b abl_priors_only_vanilla

    # All-pairs tournament
    uv run python experiments/typologist_judge.py \\
        --variants vanilla,abl_leace_only_vanilla,abl_priors_only_vanilla,abl_single_pass_vanilla \\
        --resume

Output:
    data/experiments/typologist_judge/<a>_vs_<b>.json   (one per pair)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import ANTHROPIC_API_KEY, DATA_DIR  # noqa: E402

EXPERIMENTS_BASE = DATA_DIR / "experiments"
JUDGE_OUT_DIR = EXPERIMENTS_BASE / "typologist_judge"
EXPERIMENT_PREFIX = "typologist_"
MODEL = "claude-opus-4-7"

# Directory names mash mechanism × condition together. This map decomposes
# them so summaries can show only the dimension(s) that vary across the
# variants under comparison.
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
    """Format a variant for display.

    If every variant under comparison shares the same condition, suppress the
    condition and show only the mechanism. If conditions vary, show
    "mechanism@condition" so both axes are explicit.
    """
    mech, cond = _NAME_TO_DIMENSIONS.get(variant, (variant, ""))
    conds = {_NAME_TO_DIMENSIONS.get(v, (v, ""))[1] for v in all_variants}
    if len(conds) <= 1:
        return mech
    return f"{mech}@{cond}"

CORPUS_DESCRIPTION = (
    "A corpus of approximately 500 of the most-starred public GitHub repositories. "
    "Each entry has a 2-3 sentence summary describing the project: what it does, "
    "key features, and what makes it notable. The corpus is heterogeneous, spanning "
    "ML/AI libraries, web frameworks, developer tools, command-line utilities, "
    "learning resources, awesome lists, system infrastructure, security tools, and "
    "more, written in many programming languages and maintained by both individuals "
    "and organizations."
)


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


def _build_prompt(label_a: str, schema_a: list[dict], label_b: str, schema_b: list[dict]) -> str:
    return f"""You are evaluating two candidate facet-classification schemas for a corpus of GitHub repositories. Each schema describes the corpus as 3 categorical axes (facets), where every repository can be assigned exactly one value on each facet.

Corpus: {CORPUS_DESCRIPTION}

The goal of a faceted schema is to support understanding and retrieval: answering questions about what's in the corpus, and letting users filter or group repositories by combinations of facet values.

Schema {label_a}:
{_format_schema(schema_a)}
Schema {label_b}:
{_format_schema(schema_b)}

Which schema do you think is more useful for understanding and exploring this corpus? Consider what each schema captures well, what it might miss, and what kinds of questions a user could (or could not) answer with each.

Respond with only valid JSON of this form:
{{
  "winner": "<{label_a}, {label_b}, or tie>",
  "reasoning": "<2-4 sentences explaining your judgment>",
  "schema_{label_a}_strengths": "<what schema {label_a} captures well>",
  "schema_{label_a}_weaknesses": "<what schema {label_a} misses or does poorly>",
  "schema_{label_b}_strengths": "<what schema {label_b} captures well>",
  "schema_{label_b}_weaknesses": "<what schema {label_b} misses or does poorly>"
}}
"""


def _strip_fences(s: str) -> str:
    s = re.sub(r"^```(?:json)?\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s)
    return s


def _judge(client: anthropic.Anthropic, label_a: str, schema_a: list[dict], label_b: str, schema_b: list[dict]) -> dict:
    prompt = _build_prompt(label_a, schema_a, label_b, schema_b)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text
    cleaned = _strip_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        parsed = {"_parse_error": str(e), "_raw": raw}
    return {"prompt": prompt, "raw": raw, "parsed": parsed}


def _load_schema(variant: str) -> list[dict]:
    path = EXPERIMENTS_BASE / f"{EXPERIMENT_PREFIX}{variant}" / "schema.json"
    if not path.exists():
        raise FileNotFoundError(f"No schema at {path}")
    return json.loads(path.read_text())


def _judge_pair(client: anthropic.Anthropic, a: str, b: str, resume: bool) -> dict:
    """Run one pair (forward + swapped) with caching to JSON. Returns the saved dict."""
    out_path = JUDGE_OUT_DIR / f"{a}_vs_{b}.json"
    if resume and out_path.exists():
        print(f"\n[{a} vs {b}] skipping (resume) — exists at {out_path}")
        return json.loads(out_path.read_text())

    schema_a = _load_schema(a)
    schema_b = _load_schema(b)

    print(f"\n[{a} vs {b}] forward (A={a}, B={b})")
    forward = _judge(client, "A", schema_a, "B", schema_b)
    print(f"  → winner: {forward['parsed'].get('winner', 'parse error')}")

    print(f"[{a} vs {b}] swapped (A={b}, B={a})")
    swapped = _judge(client, "A", schema_b, "B", schema_a)
    print(f"  → winner: {swapped['parsed'].get('winner', 'parse error')}")

    output = {
        "model": MODEL,
        "variant_a": a,
        "variant_b": b,
        "forward": {"label_a": a, "label_b": b, **forward},
        "swapped": {"label_a": b, "label_b": a, **swapped},
    }
    out_path.write_text(json.dumps(output, indent=2))
    return output


def _resolve_choice(letter: str, label_a: str, label_b: str) -> str:
    if letter == "A":
        return label_a
    if letter == "B":
        return label_b
    return letter  # "tie" or unparseable


def _summarize_pair(saved: dict) -> dict:
    """Distill a saved pair JSON into a small summary of who won and how cleanly."""
    a, b = saved["variant_a"], saved["variant_b"]
    fwd = saved["forward"]["parsed"].get("winner", "?")
    swp = saved["swapped"]["parsed"].get("winner", "?")
    fwd_choice = _resolve_choice(fwd, a, b)
    swp_choice = _resolve_choice(swp, b, a)  # swapped roles
    if fwd_choice == swp_choice and fwd_choice in (a, b):
        outcome = f"consistent — {fwd_choice} wins"
        winner = fwd_choice
    elif fwd_choice == "tie" and swp_choice == "tie":
        outcome = "consistent tie"
        winner = "tie"
    elif fwd_choice in (a, b) and swp_choice in (a, b) and fwd_choice != swp_choice:
        outcome = "order-dependent (position bias)"
        winner = None
    else:
        outcome = f"ambiguous: forward={fwd_choice}, swapped={swp_choice}"
        winner = None
    return {"a": a, "b": b, "fwd": fwd_choice, "swp": swp_choice, "outcome": outcome, "winner": winner}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="vanilla", help='First variant (single-pair mode)')
    parser.add_argument("--b", default="abl_single_pass_vanilla", help='Second variant (single-pair mode)')
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variants for an all-pairs tournament. Overrides --a/--b.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse saved per-pair JSONs")
    args = parser.parse_args()

    JUDGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    variants_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    if variants_list:
        pairs = list(combinations(variants_list, 2))
    else:
        pairs = [(args.a, args.b)]

    saved_pairs = []
    for a, b in pairs:
        saved_pairs.append(_judge_pair(client, a, b, args.resume))

    # Compute the union of variants we're displaying so we can decide how to
    # format names (suppress condition if all share the same condition).
    display_universe = list({a for a, _ in pairs} | {b for _, b in pairs})

    # Summaries
    print("\n" + "=" * 70)
    print("Pairwise outcomes")
    print("=" * 70)
    summaries = [_summarize_pair(s) for s in saved_pairs]
    for s in summaries:
        a_disp = _display_name(s["a"], display_universe)
        b_disp = _display_name(s["b"], display_universe)
        # Re-render outcome string with display names if a winner is identified.
        if s["winner"] in (s["a"], s["b"]):
            outcome = f"consistent — {_display_name(s['winner'], display_universe)} wins"
        else:
            outcome = s["outcome"]
        print(f"  {a_disp:<18} vs {b_disp:<18}  {outcome}")

    if len(variants_list) > 2:
        print("\n" + "=" * 70)
        print("Tournament wins (consistent pairs only — position-biased pairs skipped)")
        print("=" * 70)
        wins = {v: 0 for v in variants_list}
        ties = 0
        skipped = 0
        for s in summaries:
            if s["winner"] in (None,):
                skipped += 1
            elif s["winner"] == "tie":
                ties += 1
            else:
                wins[s["winner"]] = wins.get(s["winner"], 0) + 1
        ranked = sorted(wins.items(), key=lambda kv: -kv[1])
        for v, w in ranked:
            print(f"  {_display_name(v, display_universe):<18}  {w} wins")
        if ties:
            print(f"  (ties: {ties})")
        if skipped:
            print(f"  (position-biased / ambiguous pairs: {skipped})")


if __name__ == "__main__":
    main()
