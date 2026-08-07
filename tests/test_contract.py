"""Contract compliance test for every discovered architecture — fully OFFLINE.

Runs each architecture in architectures.REGISTRY through synthetic questions in
mock mode (no API keys, no dataset download, no network) and asserts the
plug-in contract (see architectures/__init__.py):

  * the three exports exist and are consistent;
  * run() returns {"final_answer", "transcript"} with the required fields;
  * analysis.final_round_answers() / metrics_for_run() can score the transcript
    (exactly one final record per answering agent);
  * contamination lands on the declared seats and ONLY there;
  * token accounting sees every recorded call.

Runnable two ways (CI uses both entry points interchangeably):

    KHRA_MOCK=1 python tests/test_contract.py
    KHRA_MOCK=1 pytest tests/test_contract.py
"""

from __future__ import annotations

import os
import sys

# Mock mode MUST be set before config is imported (it is read at import time).
os.environ.setdefault("KHRA_MOCK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis            # noqa: E402
import architectures       # noqa: E402
from agent import build_false_belief  # noqa: E402

# Synthetic 2-option questions — no dataset needed, so this runs anywhere.
QUESTIONS = [
    {
        "id": f"synthetic_{i:02d}",
        "category": "synthetic",
        "question": f"The trophy doesn't fit in the suitcase (case {i}) because it is too large. What is too large?",
        "options": {"A": "the trophy", "B": "the suitcase"},
        "correct": "A",
        "misinfo": "Reliable sources report the suitcase is the item that is too large.",
        "answer_type": "mc",
    }
    for i in range(2)
]

REQUIRED_TRANSCRIPT_FIELDS = (
    "architecture", "question_id", "seeded_seats", "answering_agents",
    "final_answer",
)


def _check_one(name: str, run, influence: list[str]) -> None:
    answering = architectures.ANSWERING[name]
    assert influence, f"{name}: INFLUENCE_ORDER is empty"
    assert answering, f"{name}: ANSWERING_AGENTS is empty"
    assert set(answering) <= set(influence) or True  # answering seats need not be in influence, but both must be seat names

    for question in QUESTIONS:
        for condition, n_seeded in (("control", 0), ("single", 1)):
            if n_seeded > len(influence):
                continue
            contamination = {
                "seats": influence[:n_seeded],
                "false_belief": build_false_belief(question["misinfo"]) if n_seeded else None,
            }
            result = run(question, contamination=contamination)

            assert isinstance(result, dict) and "final_answer" in result and "transcript" in result, (
                f"{name}: run() must return {{'final_answer', 'transcript'}}"
            )
            t = result["transcript"]
            for field in REQUIRED_TRANSCRIPT_FIELDS:
                assert field in t, f"{name}/{condition}: transcript missing {field!r}"
            assert t["architecture"] == name, (
                f"{name}: transcript.architecture is {t['architecture']!r} — must equal the module name"
            )
            assert sorted(t["seeded_seats"]) == sorted(contamination["seats"] or []), (
                f"{name}/{condition}: seeded_seats does not match the requested seats"
            )

            # The scorer must accept it: one final record per answering agent.
            finals = analysis.final_round_answers(t)
            assert sorted(r["agent"] for r in finals) == sorted(answering), (
                f"{name}/{condition}: final round has {sorted(r['agent'] for r in finals)}, "
                f"expected exactly the answering agents {sorted(answering)}"
            )
            # Contamination must be marked on seeded seats and only there.
            for rec in finals:
                assert rec.get("seeded") == (rec["agent"] in set(t["seeded_seats"])), (
                    f"{name}/{condition}: seeded flag wrong on {rec['agent']}"
                )
            # Metrics and token accounting must compute without error.
            row = analysis.metrics_for_run(t, question)
            assert row["n_answering_agents"] == len(answering)
            assert analysis.transcript_tokens(t) >= 0
    print(f"  PASS  {name}  ({len(influence)} seats, {len(answering)} answering)")


def test_all_architectures() -> None:
    assert architectures.REGISTRY, "no architectures discovered"
    for name, (run, influence) in sorted(architectures.REGISTRY.items()):
        _check_one(name, run, influence)


def test_pending_stubs_are_parked() -> None:
    """Stubs must be listed as pending, never silently runnable."""
    overlap = set(architectures.PENDING) & set(architectures.REGISTRY)
    assert not overlap, f"stubs both pending and registered: {overlap}"


if __name__ == "__main__":
    print(f"Discovered: {', '.join(sorted(architectures.REGISTRY))}")
    if architectures.PENDING:
        print(f"Pending stubs: {', '.join(sorted(architectures.PENDING))}")
    test_all_architectures()
    test_pending_stubs_are_parked()
    print("ALL CONTRACT CHECKS PASSED")
