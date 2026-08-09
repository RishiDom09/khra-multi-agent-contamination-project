"""Entry point: the MINT contamination experiment.

ONE MODEL PER RUN, ALL QUESTIONS PER RUN. Select the model from the registry in
``config.MODELS`` and run the full question set (every verified winogrande item)
through every architecture x condition cell:

    python run_experiment.py --model opus       # Claude Opus 5 (AWS Bedrock)
    python run_experiment.py --model sol        # GPT-5.6 Sol   (AWS Bedrock)
    python run_experiment.py --model deepseek   # DeepSeek V4-Pro (Azure Foundry)
    python run_experiment.py --model grok       # Grok 4.20 reasoning (Azure Foundry)

Cross-model comparability is by construction: every model answers the identical
full set, with identical (seeded) option shuffles and contamination text — so
per-question results pair across models with no manifest bookkeeping.

Each run writes one JSON transcript per cell into ``logs/<date>_<model>/`` and
an accuracy + propagation analysis into ``results/<date>_<model>/``. Logging is
RESUMABLE: transcript filenames are deterministic and a rerun skips every cell
it already has, so an interrupted run picks up where it stopped. ``--new-run``
starts a fresh folder; ``--force`` re-runs cells in place.

Validation helpers: ``--smoke`` (first 4 questions), ``--limit N``, and
KHRA_MOCK=1 for a fully offline run with no API keys (isolated in ``_mock``
run folders). ``--analyze-only`` rebuilds results from transcripts with zero
API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Any, Callable, Optional

import analysis
import architectures
import config
import llm_client
import mint_loader
import tasks
import token_ledger
from agent import build_false_belief

# Each setup: (runner, seats ordered by structural influence), AUTO-DISCOVERED
# from architectures/ — every module there that satisfies the plug-in contract
# (see architectures/__init__.py) registers itself. Adding
# a compliant file makes it runnable with --architectures <name>; stub files
# carrying `INCOMPLETE = True` are parked in architectures.PENDING until their
# owner activates them.
SETUPS: dict[str, tuple[Callable[..., dict[str, Any]], list[str]]] = (
    architectures.REGISTRY
)

# How many agents each condition contaminates. Seats are taken from the front of
# the architecture's INFLUENCE_ORDER, i.e. the most structurally influential
# seats first (dynamic: the most-consulted agent; adversarial: a Worker, never
# the critic).
CONDITIONS: dict[str, int] = {"control": 0, "single": 1, "multiple": 2}

def _expected_cells(architectures: list[str], conditions: list[str]) -> int:
    """How many (architecture, condition) cells a single question should yield.

    A question counts as COMPLETED only when all of these ran without failure.
    baseline+multiple contributes 0 (it is N/A).
    """
    return sum(
        1
        for arch in architectures
        for cond in conditions
        if CONDITIONS[cond] <= len(SETUPS[arch][1])
    )


def contamination_for(
    architecture: str, condition: str, question: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Build the contamination spec, or None if the cell is not applicable.

    Returns None for baseline+multiple: a 1-agent architecture has no second seat
    to contaminate, so that cell is skipped rather than faked.
    """
    n_seeded = CONDITIONS[condition]
    influence = SETUPS[architecture][1]
    if n_seeded > len(influence):
        return None
    return {
        "seats": influence[:n_seeded],
        "false_belief": build_false_belief(question["misinfo"]) if n_seeded else None,
    }


# Resumable transcript logging
def _transcript_path(architecture: str, condition: str, question_id: str) -> str:
    """Deterministic path — this is what makes a run resumable."""
    return os.path.join(
        config.LOG_DIR, f"{architecture}__{condition}__{question_id}.json"
    )


def _load_transcript(path: str, expect: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            transcript = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None  # truncated by an interrupted write: redo the cell

    for key, wanted in expect.items():
        found = transcript.get(key)
        if found is not None and found != wanted:
            return None  # different model/strategy/options: not comparable
    return transcript


def _write_transcript(path: str, transcript: dict[str, Any]) -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(transcript, fh, indent=2)
    os.replace(tmp, path)  # atomic: a killed run never leaves a half file


def _update_run_metadata(args: argparse.Namespace, run_model: str) -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    path = os.path.join(config.LOG_DIR, "run_metadata.json")
    meta: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            meta = {}
    meta.setdefault("run", config.RUN_NAME)
    meta.setdefault("model_key", config.MODEL_KEY)
    meta.setdefault("provider", config.PROVIDER)
    meta.setdefault("model", run_model)
    meta.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    meta.setdefault("sessions", []).append(
        {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mock": config.MOCK_MODE,
            "architectures": args.architectures,
            "conditions": args.conditions,
            "categories": args.categories,
            "strategy": args.strategy,
            "seed": args.seed,
            "shuffle_options": not args.no_shuffle_options,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.MODEL_SPEC.get("max_tokens") or config.MAX_TOKENS,
            "thinking_off": config.THINKING_OFF,
            "limit": args.limit,
            "force": args.force,
        }
    )
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, path)


def _note_api_model(api_model: Optional[str]) -> None:
    if not api_model:
        return
    path = os.path.join(config.LOG_DIR, "run_metadata.json")
    if not os.path.exists(path):
        return
    try:
        with open(path) as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return
    meta["api_model"] = api_model
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, path)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(config.MODELS), default=config.MODEL_KEY,
                   help="which registry model to run (see config.MODELS). ONE "
                        "model per run; outputs are isolated per model. "
                        f"Default: {config.MODEL_KEY!r} (or KHRA_MODEL env)")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N questions (pipeline validation)")
    p.add_argument("--smoke", action="store_true",
                   help="shorthand for --limit 4")
    p.add_argument("--architectures", nargs="+", choices=list(SETUPS),
                   default=list(SETUPS),
                   help="which architectures to run (default: all three)")
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                   default=list(config.ACTIVE_CONDITIONS),
                   help="which conditions to run (default: control single; "
                        "'multiple' is off)")
    p.add_argument("--categories", nargs="+", choices=list(mint_loader.DATASETS),
                   default=list(config.ACTIVE_CATEGORIES),
                   help="which MINT subsets to draw from (default: "
                        f"{config.ACTIVE_CATEGORIES}; ethics is loaded but "
                        "excluded unless named here)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                   help="OPTION-SHUFFLE seed (default 42). There is no question "
                        "sampling — every run answers the full set — so this "
                        "only controls the (identical-across-models) option "
                        "permutations")
    p.add_argument("--strategy", default=config.MINT_STRATEGY,
                   help="MINT misinformation strategy (default 'other')")
    p.add_argument("--no-shuffle-options", action="store_true",
                   help="keep MINT's original option order. NOT recommended: "
                        "every winogrande item has answer=1, so without "
                        "shuffling the correct answer is 'A' for that entire "
                        "category and positional bias confounds accuracy.")
    p.add_argument("--force", action="store_true",
                   help="re-run cells even if a transcript already exists")
    p.add_argument("--new-run", action="store_true",
                   help="start a FRESH run folder (logs/<today>_<model>/) instead "
                        "of resuming the most recent one for this model")
    p.add_argument("--run", default=None, metavar="NAME",
                   help="target a specific run folder by name (e.g. "
                        "'2026-07-25_qwen'), mainly for --analyze-only on an "
                        "older run")
    p.add_argument("--analyze-only", action="store_true",
                   help="skip the model entirely; rebuild results from the "
                        "active run folder's transcripts (zero API calls)")
    args = p.parse_args(argv)
    if args.smoke and args.limit is None:
        args.limit = 4
    return args


# Run one question through the whole matrix (shared by both modes)
class QuestionOutcome:
    """What running one question through the matrix produced."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.tokens = 0            # tokens billed THIS run (fresh calls only)
        self.ran = 0               # cells freshly executed
        self.skipped = 0           # cells reused from an existing transcript
        self.failed = 0            # cells that raised
        self.rate_limited = False  # a fresh call hit the daily/rate cap


def run_one_question(
    question: dict[str, Any],
    args: argparse.Namespace,
    run_model: str,
) -> QuestionOutcome:
    """Run every (architecture, condition) cell for one question.

    Reuses cached transcripts, writes new ones, and accumulates this run's token
    spend. Metrics for every cell (cached or fresh) are collected either way.
    """
    out = QuestionOutcome()
    expect = {
        "model": run_model,
        "strategy": args.strategy,
        "options": question["options"],
    }
    for arch in args.architectures:
        for cond in args.conditions:
            spec = contamination_for(arch, cond, question)
            if spec is None:
                continue  # e.g. baseline+multiple is N/A
            path = _transcript_path(arch, cond, question["id"])
            transcript = (
                None
                if (args.force and not args.analyze_only)
                else _load_transcript(path, expect)
            )

            if transcript is not None:
                out.skipped += 1
            elif args.analyze_only:
                continue
            else:
                runner, _ = SETUPS[arch]
                try:
                    result = runner(question, contamination=spec)
                except Exception:  
                    out.failed += 1
                    print(f"  FAILED {arch}/{cond}/{question['id']}")
                    traceback.print_exc()
                    continue
                transcript = result["transcript"]
                transcript.update(
                    {
                        "condition": cond,
                        "strategy": args.strategy,
                        "model": run_model,
                        "category": question["category"],
                        "correct": question["correct"],
                        "options": question["options"],
                        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                    }
                )
                _write_transcript(path, transcript)
                out.ran += 1
                out.tokens += analysis.transcript_tokens(transcript)
                if any(
                    llm_client.is_rate_limit_error(r)
                    for r in analysis.final_round_answers(transcript)
                ):
                    out.rate_limited = True

            transcript.setdefault("condition", cond)
            out.rows.append(analysis.metrics_for_run(transcript, question))
    return out


def _reload_transcript(path: str) -> Optional[dict[str, Any]]:
    """Read a transcript straight off disk, with no reuse/compatibility filtering."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


_AUDIT_FIELDS = [
    "accuracy", "propagation", "spread", "final_answer_contaminated",
    "final_answer", "correct_letter", "misinfo_letter", "n_contaminated",
    "n_answering_agents", "n_abstain",
]


def audit_scoring(
    live_rows: list[dict[str, Any]], questions_by_id: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """SECOND PASS: re-score every transcript from disk and compare to the live score.

    The live score is computed in memory as each question finishes; this pass
    independently re-reads what was actually written to ``logs/`` and re-derives
    every metric from it. Agreement means the persisted transcripts fully support
    the reported numbers. Any disagreement is FLAGGED rather than silently
    resolved, because it means the in-memory result and the durable record differ
    — e.g. a transcript that failed to serialize, or ground truth that shifted
    mid-run.

    Returns ``(disk_scored_rows, flags)``.
    """
    audited: list[dict[str, Any]] = []
    flags: list[str] = []
    for row in live_rows:
        cell = f"{row['architecture']}/{row['condition']}/{row['question_id']}"
        path = _transcript_path(row["architecture"], row["condition"], row["question_id"])
        transcript = _reload_transcript(path)
        if transcript is None:
            flags.append(f"{cell}: transcript missing or unreadable at {path}")
            continue
        question = questions_by_id.get(row["question_id"])
        if question is None:
            flags.append(f"{cell}: question id is no longer in the dataset")
            continue
        transcript.setdefault("condition", row["condition"])
        fresh = analysis.metrics_for_run(transcript, question)
        diffs = [
            f"{k}: live={row.get(k)!r} disk={fresh.get(k)!r}"
            for k in _AUDIT_FIELDS
            if row.get(k) != fresh.get(k)
        ]
        if diffs:
            flags.append(f"{cell}: " + "; ".join(diffs))
        audited.append(fresh)
    return audited, flags


def _running_accuracy(rows: list[dict[str, Any]]) -> str:
    by_arch: dict[str, list[bool]] = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], []).append(r["accuracy"])
    parts = []
    for arch in ("baseline", "dynamic_team", "adversarial_team"):
        accs = by_arch.get(arch)
        if accs:
            short = {"baseline": "base", "dynamic_team": "dyn", "adversarial_team": "adv"}[arch]
            parts.append(f"{short} {sum(accs) / len(accs):.0%}")
    return "  ".join(parts)


# Main
def _load_all_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The FULL question set, in fixed file order (no sampling, no streaming).

    Every model answers this identical set — that is what makes cross-model
    comparisons paired. ``--limit N`` takes the first N (interleaved across
    categories when more than one is active, so smoke tests stay balanced).
    """
    questions = mint_loader.load_questions(
        strategy=args.strategy,
        shuffle_options=not args.no_shuffle_options,
        categories=args.categories,
        seed=args.seed,
    )
    if args.limit is not None:
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for q in questions:
            by_cat.setdefault(q["category"], []).append(q)
        interleaved: list[dict[str, Any]] = []
        for tier in zip(*by_cat.values()):
            interleaved.extend(tier)
        questions = interleaved[: args.limit]
    return questions


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)


    config.set_model(args.model)
    if args.run:
        config.set_run(args.run)
    elif args.new_run:
        config.start_new_run()

    questions = _load_all_questions(args)

    run_model = "mock" if config.MOCK_MODE else config.MODEL_ID
    mode = (
        "MOCK"
        if config.MOCK_MODE
        else f"REAL  {config.MODEL_KEY} = {config.MODEL_ID} via {config.PROVIDER}"
    )
    expected_cells = _expected_cells(args.architectures, args.conditions)

    print(f"Mode: {mode}")
    if architectures.PENDING:
        print("Pending architectures (INCOMPLETE stubs, not runnable yet): "
              + ", ".join(f"{n} [{o}]" for n, o in sorted(architectures.PENDING.items())))
    print(f"Run folder: {config.LOG_DIR}/  "
          f"({'targeted via --run' if args.run else 'fresh' if args.new_run else 'resuming latest for this model; --new-run for a fresh one'})")
    if not args.analyze_only:
        _update_run_metadata(args, run_model)
    print(f"strategy: {args.strategy!r}  |  option-shuffle seed: {args.seed}  "
          f"|  option shuffle: {not args.no_shuffle_options}  "
          f"|  thinking: {'off' if config.THINKING_OFF else 'on'}")
    print(f"Categories: {args.categories}  |  architectures: {args.architectures}  "
          f"|  conditions: {args.conditions}  |  {len(questions)} questions queued\n")

    run_model_for_ledger = config.MODEL_ID

    rows: list[dict[str, Any]] = []
    completed_ids: list[str] = []
    tokens_used = 0
    q_done = ran = skipped = failed = 0
    started = time.time()
    stop_reason = "all questions processed"

    for question in questions:
        outcome = run_one_question(question, args, run_model)
        rows.extend(outcome.rows)
        tokens_used += outcome.tokens
        ran += outcome.ran
        skipped += outcome.skipped
        failed += outcome.failed
        q_done += 1

        if outcome.tokens and not config.MOCK_MODE:
            token_ledger.record(
                outcome.tokens, model=run_model_for_ledger, note=question["id"]
            )
        # A question is COMPLETED only if every expected cell ran without failure.
        if outcome.failed == 0 and len(outcome.rows) == expected_cells:
            completed_ids.append(question["id"])

        if not args.analyze_only:
            print(f"[q{q_done:>3}/{len(questions)}] {question['id']:<16} "
                  f"{question['category']:<10} acc: {_running_accuracy(rows)}"
                  f"  tokens {tokens_used:,}")

        if outcome.rate_limited:
            stop_reason = "HIT a provider rate limit mid-run — stopping (rerun to resume)"
            break

    elapsed = time.time() - started
    print(f"\nStopped: {stop_reason}.")
    print(f"Questions processed: {q_done}  |  cells: {ran} run, {skipped} cached, "
          f"{failed} failed  |  tokens this run: {tokens_used:,}  |  {elapsed:.0f}s")
    if not rows:
        print("No results to analyze.")
        return

    print(f"Completed questions (all conditions ran): {len(completed_ids)}")

    # SECOND PASS — re-score every log from disk and cross-check the live score
    questions_by_id = {q["id"]: q for q in questions}
    audited_rows, flags = audit_scoring(rows, questions_by_id)
    print("\nSCORING AUDIT (re-scored every transcript from logs/)")
    print("-" * 60)
    print(f"  cells scored live      : {len(rows)}")
    print(f"  cells re-scored on disk: {len(audited_rows)}")
    if flags:
        print(f"  *** {len(flags)} MISMATCH(ES) FLAGGED — live score != disk score ***")
        for f in flags[:20]:
            print(f"    ! {f}")
        if len(flags) > 20:
            print(f"    ... and {len(flags) - 20} more")
        print("  Final metrics below are computed from the DISK re-score.")
        print("  Investigate the flags before trusting these numbers.")
    else:
        print("  no mismatches — live scores agree with the logs on every field.")
    rows = audited_rows or rows

    if not args.analyze_only:
        _note_api_model(llm_client.API_MODEL_SEEN)

    report = analysis.build_report(rows)
    analysis.print_report(report)
    analysis.write_results(
        rows,
        report,
        meta={
            "model_key": config.MODEL_KEY,
            "provider": config.PROVIDER,
            "model": run_model,
            "api_model": llm_client.API_MODEL_SEEN,
            "run": config.RUN_NAME,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.MODEL_SPEC.get("max_tokens") or config.MAX_TOKENS,
            "thinking_off": config.THINKING_OFF,
            "strategy": args.strategy,
            "seed": args.seed,
            "categories": args.categories,
            "shuffle_options": not args.no_shuffle_options,
            "architectures": args.architectures,
            "conditions": args.conditions,
            "questions_processed": q_done,
            "questions_completed": len(completed_ids),
            "tokens_used_this_run": tokens_used,
            "stop_reason": stop_reason,
            "label_source": "winogrande gold labels (data/winogrande_labels.json)",
            "scoring_audit_cells": len(audited_rows),
            "scoring_audit_mismatches": len(flags),
            "scoring_audit_flags": flags[:50],
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"Transcripts in ./{config.LOG_DIR}/")


if __name__ == "__main__":
    main()
