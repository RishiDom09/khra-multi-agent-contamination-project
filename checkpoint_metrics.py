"""Read-only mid-run checkpoint: accuracy/propagation + logging integrity.

Scores every transcript currently on disk in a run folder WITHOUT touching the
run or the results/ directory (safe to run while an experiment is writing) and
prints per-condition accuracy, propagation, spread, final-answer contamination,
abstains, plus integrity checks (model stamp, seeded seats, final-round
completeness) and the coordinator-overrules-unanimous-workers count.

Zero API calls. Usage:

    python checkpoint_metrics.py <run-folder-name>     # e.g. 2026-08-08_grok_hier
    python checkpoint_metrics.py                       # newest run folder in logs/
"""

from __future__ import annotations

import collections
import json
import os
import sys

import analysis
import mint_loader


def newest_run() -> str:
    runs = [d for d in os.listdir("logs") if os.path.isdir(os.path.join("logs", d))]
    if not runs:
        sys.exit("no run folders in logs/")
    return max(runs, key=lambda d: os.path.getmtime(os.path.join("logs", d)))


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    logdir = os.path.join("logs", run)

    meta_path = os.path.join(logdir, "run_metadata.json")
    expect_model = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            expect_model = json.load(fh).get("model")

    questions = {
        q["id"]: q
        for q in mint_loader.load_questions(
            strategy="other", shuffle_options=True,
            categories=["winogrande"], seed=42,
        )
    }

    rows: list[dict] = []
    bad: list[str] = []
    override: collections.Counter = collections.Counter()
    files = sorted(
        f for f in os.listdir(logdir)
        if f.endswith(".json") and f != "run_metadata.json"
    )
    for name in files:
        with open(os.path.join(logdir, name)) as fh:
            t = json.load(fh)
        q = questions.get(t["question_id"])
        if q is None:
            bad.append(f"{name}: unknown question id")
            continue
        finals = analysis.final_round_answers(t)
        problems = []
        if expect_model and t.get("model") != expect_model:
            problems.append("model mismatch")
        if sorted(r["agent"] for r in finals) != sorted(t["answering_agents"]):
            problems.append("final round != answering_agents")
        if t["condition"] == "control" and t["seeded_seats"]:
            problems.append("control cell has seeded seats")
        if t["condition"] != "control" and not t["seeded_seats"]:
            problems.append("seeded cell has no seeded seats")
        if problems:
            bad.append(f"{name}: {', '.join(problems)}")
        seat_answers = {r["agent"]: r["answer"] for r in finals}
        coord = seat_answers.pop("Coordinator", None)
        worker_answers = set(seat_answers.values())
        if coord is not None and len(worker_answers) == 1 and coord not in worker_answers:
            override[t["condition"]] += 1
        rows.append(analysis.metrics_for_run(t, q))

    print(f"run: {run}  |  model: {expect_model}")
    print(f"cells on disk: {len(rows)}  |  integrity problems: {len(bad)}")
    for b in bad[:10]:
        print(f"  BAD {b}")

    by_cond: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    for cond in sorted(by_cond):
        g = by_cond[cond]
        n = len(g)
        spread = [r["spread"] for r in g if r["spread"] is not None]
        print(
            f"{cond:>8}: n={n}  acc={sum(r['accuracy'] for r in g) / n:.0%}  "
            f"prop={sum(r['propagation'] for r in g) / n:.1%}  "
            f"spread={sum(spread) / len(spread):.1%}  "
            f"final_contam={sum(r['final_answer_contaminated'] for r in g) / n:.0%}  "
            f"abstains={sum(r['n_abstain'] for r in g)}"
        )
    print(f"coordinator overruled unanimous workers: {dict(override) or 0}")


if __name__ == "__main__":
    main()
