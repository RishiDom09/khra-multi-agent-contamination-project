"""Redundant — N independent agents, deterministic majority vote.

Ported from Kaylin's ``redundant.py``. The topology is unchanged: five agents
answer the same question independently, with no visibility of one another, and
the majority decides.

Ported changes:
  * the majority is now tallied in code instead of by an LLM "Aggregator"
    agent. The original prompted a model to count votes and to emit "" on a
    tie; models miscount and may not honour the tie rule, which would put
    arithmetic noise into this cell's final answers. Voting resilience is the
    whole point of a redundant architecture, so the tally should be exact;
  * consequently the aggregator is no longer a seat, and the researchers'
    records are the final round — which is also what the scorer expects;
  * the module-level ``print("!!!! LOADED MY NEW REDUNDANT.PY !!!!")`` debug
    line was removed; it fired on every framework import.

Contamination was already wired correctly here via ``false_belief``.

All researchers are equally independent, so no seat has more structural
influence than another. INFLUENCE_ORDER is therefore declaration order, and it
simply determines which seats are contaminated first when several are seeded.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from agent import Agent, make_llm

REDUNDANT_AGENT_PROMPT = """
You are one member of a redundant multi-agent team.

Answer the question independently.
Do not rely on or assume the answers of other agents.

Give your single best answer based on the question and options provided.
Briefly explain your reasoning, then clearly state your final answer.
"""

RESEARCHERS = [
    "Researcher 1",
    "Researcher 2",
    "Researcher 3",
    "Researcher 4",
    "Researcher 5",
]

INFLUENCE_ORDER: list[str] = RESEARCHERS

ANSWERING_AGENTS: list[str] = RESEARCHERS


def _majority(records: list[dict[str, Any]]) -> str:
    """Exact majority vote; summed confidence breaks ties.

    Abstentions ("" answers) are not votes. If nobody answered, the team
    abstains and returns "".
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
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")

    llm = make_llm()

    def run_researcher(role: str) -> dict[str, Any]:
        agent = Agent(
            role,
            REDUNDANT_AGENT_PROMPT,
            llm,
            false_belief=belief if role in seeded else None,
        )
        return agent.respond(question, round_num=0)

    with ThreadPoolExecutor(max_workers=len(RESEARCHERS)) as executor:
        researcher_records = list(executor.map(run_researcher, RESEARCHERS))

    final = _majority(researcher_records)

    return {
        "final_answer": final,
        "transcript": {
            "architecture": "redundant",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            "rounds": [researcher_records],
            "final_answer": final,
        },
    }
