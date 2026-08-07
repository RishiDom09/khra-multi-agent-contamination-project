"""Single-agent baseline to compare all results from test against 
"""

from __future__ import annotations

from typing import Any, Optional

from agent import Agent, make_llm

BASELINE_ROLE = "Give your single best answer."
BASELINE_AGENT = "Solo"

INFLUENCE_ORDER: list[str] = [BASELINE_AGENT]
ANSWERING_AGENTS: list[str] = [BASELINE_AGENT]


def run(
    question: dict[str, Any],
    contamination: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Single-agent baseline: one standardized agent answers alone."""
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")
    agent = Agent(
        BASELINE_AGENT,
        BASELINE_ROLE,
        make_llm(),
        false_belief=belief if BASELINE_AGENT in seeded else None,
    )
    out = agent.respond(question, round_num=0)
    return {
        "final_answer": out["answer"],
        "transcript": {
            "architecture": "baseline",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            "response": out,
            "final_answer": out["answer"],
        },
    }
