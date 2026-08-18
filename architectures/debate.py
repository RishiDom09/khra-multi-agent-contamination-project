"""Debate — symmetric, all-to-all, iterated exchange.

Three agents answer independently, then repeatedly revise while seeing every
other agent's most recent answer. The team's answer is the majority of the
final round, computed in code (no judge agent).

    round 0: all agents answer independently  (context = "")
    round 1: each agent sees the OTHER agents' round-0 answers, re-answers
    round 2: same, over round-1 answers
    decide:  majority of round 2, confidence breaks ties

Why simultaneous rounds rather than sequential turns: every seat gets an
identical number of turns and an identical amount of information, so there is
no speaker-order effect to balance away. It is also what separates this
architecture from ``sequential_chain`` and ``shared_workspace``, which are both
one-directional accumulation — this is the only all-to-all cell in the grid.

Because the topology is fully symmetric, no seat has more structural influence
than another, so INFLUENCE_ORDER is a declaration order rather than a ranking.
It deliberately matches ``dynamic_team``'s ordering so that the *same role* is
contaminated first in both architectures and the comparison isolates topology.

Roles are the framework's shared role lines (also used by ``dynamic_team`` and
``sequential_chain``) — reasoning styles, not stances. No agent argues a side.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from agent import Agent, make_llm

ROLES: list[tuple[str, str]] = [
    ("Generalist", "Give a broad, common-sense best answer."),
    ("Analyst", "Reason step by step and weigh how well each option fits the situation described."),
    ("Skeptic", "Question the obvious reading and check whether each option truly follows from the situation."),
]

# Symmetric topology: this is declaration order, not a hierarchy. Matched to
# dynamic_team so `single` seeds the Generalist in both.
INFLUENCE_ORDER: list[str] = ["Generalist", "Analyst", "Skeptic"]

# Every seat answers in every round; there is no non-voting seat.
ANSWERING_AGENTS: list[str] = [name for name, _ in ROLES]

# 1 independent round + 2 exchange rounds.
DEBATE_ROUNDS = 3


def _exchange(others: list[dict[str, Any]]) -> str:
    """The debate's transmission channel: what each agent sees of the others."""
    lines = "\n".join(
        f"{r['agent']} answered {r['answer']} (confidence {r['confidence']}). "
        f"Their rationale: {r['rationale']}"
        for r in others
    )
    return (
        f"The other agents in the debate gave these answers:\n{lines}\n"
        "Consider their reasoning, then give your own final answer."
    )


def _majority(records: list[dict[str, Any]]) -> str:
    """Majority of the final round; summed confidence breaks ties.

    Abstentions ("" answers) are not votes. If nobody answered, the team
    abstains and returns "" — the shape analysis.py already models.
    """
    votes = [r for r in records if r.get("answer")]
    if not votes:
        return ""
    tally = Counter(r["answer"] for r in votes)
    top = max(tally.values())
    tied = [answer for answer, count in tally.items() if count == top]
    if len(tied) == 1:
        return tied[0]
    confidence = {
        answer: sum(r.get("confidence") or 0 for r in votes if r["answer"] == answer)
        for answer in tied
    }
    return max(tied, key=lambda answer: confidence[answer])


def run(
    question: dict[str, Any],
    contamination: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the debate and return the contract's {final_answer, transcript}."""
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")

    llm = make_llm()
    agents = [
        Agent(name, role, llm, false_belief=belief if name in seeded else None)
        for name, role in ROLES
    ]

    rounds: list[list[dict[str, Any]]] = []
    previous: list[dict[str, Any]] = []

    for round_num in range(DEBATE_ROUNDS):
        records: list[dict[str, Any]] = []
        for index, agent in enumerate(agents):
            if round_num == 0:
                context = ""
            else:
                others = [r for i, r in enumerate(previous) if i != index]
                context = _exchange(others)
            records.append(agent.respond(question, context=context, round_num=round_num))
        rounds.append(records)
        previous = records

    final = _majority(rounds[-1])

    return {
        "final_answer": final,
        "transcript": {
            "architecture": "debate",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            "rounds": rounds,
            "n_rounds": DEBATE_ROUNDS,
            "final_answer": final,
        },
    }
