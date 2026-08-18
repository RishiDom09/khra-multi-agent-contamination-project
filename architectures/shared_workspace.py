"""Shared Workspace — sequential contributions into one accumulating log.

Ported from Hanish's ``shared_workspace_contamination.py``. The topology is
unchanged: five agents take turns, each reading everything written so far and
adding one contribution, then every agent casts a final vote over the completed
workspace. The majority wins.

    round 0: agents contribute in order, each seeing all prior entries
    round 1: every agent votes over the full workspace
    decide:  majority of round 1, confidence breaks ties

Ported changes (see the port notes in the review, not silent rewrites):
  * the workspace is an in-memory list, not a ``workspace.json`` file on disk —
    the original was global mutable state shared across every question and
    condition, and it wrote into the repo working directory;
  * the vote's few-shot example was removed. It hardcoded ``VOTE:
    {valid_options[0]}``, i.e. always "A", which is a standing nudge toward the
    first option. Output format is handled centrally by ``llm_client`` anyway;
  * the majority is tallied in code rather than by a model;
  * contamination goes through ``Agent(false_belief=...)`` so it uses the same
    wording as every other architecture and registers in the seeded flags.

Seats are ordered by speaking position: the first contributor writes into an
empty workspace and anchors everything downstream, so it is the most
structurally influential seat.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from agent import Agent, make_llm

AGENT_ROLES: list[tuple[str, str]] = [
    ("Agent 1", "Trace the sequence of events step by step and check which noun phrase is doing the final action."),
    ("Agent 2", "Focus on the sentence's grammatical and referential structure — what must the blank or pronoun refer back to."),
    ("Agent 3", "Give the most natural, common-sense reading a fluent reader would reach without overanalyzing."),
    ("Agent 4", "Check the answer against ordinary real-world plausibility — which option better fits how people actually behave."),
    ("Agent 5", "Re-check the workspace so far for anything a prior entry got wrong or overlooked, and verify it directly against the task."),
]

# Earlier speakers shape more of what later speakers read.
INFLUENCE_ORDER: list[str] = [name for name, _ in AGENT_ROLES]

# Every agent contributes and votes; there is no non-voting seat.
ANSWERING_AGENTS: list[str] = [name for name, _ in AGENT_ROLES]


def _format_workspace(entries: list[dict[str, Any]]) -> str:
    """Render the accumulated workspace for the next reader."""
    if not entries:
        return "The shared workspace is empty. You are the first to contribute."
    lines = "\n".join(
        f"[{r['agent']}] answered {r['answer']} (confidence {r['confidence']}): {r['rationale']}"
        for r in entries
    )
    return f"The shared workspace so far:\n{lines}"


def _contribute_context(entries: list[dict[str, Any]]) -> str:
    return (
        f"{_format_workspace(entries)}\n"
        "Read the shared workspace, add your own contribution, and give your answer."
    )


def _vote_context(entries: list[dict[str, Any]]) -> str:
    return (
        f"{_format_workspace(entries)}\n"
        "All contributions are in. Based on the strongest points raised in the "
        "shared workspace — not only your own entry — give your final answer."
    )


def _majority(records: list[dict[str, Any]]) -> str:
    """Majority of the vote round; summed confidence breaks ties."""
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
    """Run one workspace pass plus the vote round."""
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")

    llm = make_llm()
    agents = [
        Agent(name, role, llm, false_belief=belief if name in seeded else None)
        for name, role in AGENT_ROLES
    ]

    # Round 0 — sequential contributions into the (in-memory) workspace.
    workspace: list[dict[str, Any]] = []
    for agent in agents:
        record = agent.respond(
            question, context=_contribute_context(workspace), round_num=0
        )
        workspace.append(record)

    # Round 1 — every agent votes over the completed workspace.
    votes = [
        agent.respond(question, context=_vote_context(workspace), round_num=1)
        for agent in agents
    ]

    final = _majority(votes)

    return {
        "final_answer": final,
        "transcript": {
            "architecture": "shared_workspace",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            "rounds": [workspace, votes],
            "final_answer": final,
        },
    }
