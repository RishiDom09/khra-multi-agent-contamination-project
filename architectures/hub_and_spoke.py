"""Hub and Spoke — independent spokes, one coordinating hub.

Ported from Kaylin's ``hub_and_spoke.py`` with the architecture itself left
alone: same five independent researchers, same hub, same prompts, same flow
(researchers answer in parallel and never see each other; the hub reads their
answers and decides). Only the plug-in wiring was changed, because the original
shape could not be scored:

  * ``transcript["architecture"]`` now reports "hub_and_spoke" — it said
    "hub_spoke", which does not match the module name and split analysis groups;
  * contamination goes through ``Agent(false_belief=...)`` instead of being
    appended to the hub's context string. The contract test asserts that each
    final record's ``seeded`` flag matches ``seeded_seats``, and a hand-built
    context string never sets that flag;
  * the hub's record now sits in the same (final) round as the researchers,
    following ``hierarchical.py`` — the in-repo precedent for an architecture
    whose aggregator speaks last. Previously the last round held only the hub
    while ANSWERING_AGENTS listed only the researchers, so the scorer's filter
    returned nothing and every metric collapsed to zero.

Only the Hub is contaminable, so the "multiple" condition (two seats) is not
applicable here and that cell is skipped, exactly as it is for baseline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from agent import Agent, make_llm

HUB_EXPERT_PROMPT = """
You are an independent researcher on a multi-agent team.

Answer the question independently.
Do not rely on other researchers.

Give your single best answer based on the question and options provided.
Briefly explain your reasoning, then clearly state your final answer.
"""

HUB_PROMPT = """
You are the Hub coordinator of a multi-agent team.

You receive answers from several independent researchers.

Use their answers to determine the final answer.
Consider the researchers' responses carefully and resolve disagreements.

Give your single best answer.
"""

HUB = "Hub coordinator"

RESEARCHERS = [
    "Researcher 1",
    "Researcher 2",
    "Researcher 3",
    "Researcher 4",
    "Researcher 5",
]

# The Hub is the seat this architecture contaminates.
INFLUENCE_ORDER: list[str] = [HUB]

# The hub decides, so it counts alongside the researchers in the denominator —
# the same choice hierarchical.py makes for its coordinator.
ANSWERING_AGENTS: list[str] = RESEARCHERS + [HUB]


def run(
    question: dict[str, Any],
    contamination: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")

    llm = make_llm()

    # ------------------------------------------------------------------
    # Spokes: five independent researchers, no visibility of each other
    # ------------------------------------------------------------------
    def run_researcher(role: str) -> dict[str, Any]:
        agent = Agent(
            role,
            HUB_EXPERT_PROMPT,
            llm,
            false_belief=belief if role in seeded else None,
        )
        return agent.respond(question, round_num=0)

    with ThreadPoolExecutor(max_workers=len(RESEARCHERS)) as executor:
        researcher_records = list(executor.map(run_researcher, RESEARCHERS))

    # ------------------------------------------------------------------
    # Hub: reads the researchers' answers and decides
    # ------------------------------------------------------------------
    context = "\n\n".join(
        f"{record['agent']}: {record['answer']}" for record in researcher_records
    )

    hub = Agent(
        HUB,
        HUB_PROMPT,
        llm,
        false_belief=belief if HUB in seeded else None,
    )
    hub_record = hub.respond(question, context=context, round_num=0)

    final = hub_record["answer"]

    return {
        "final_answer": final,
        "transcript": {
            "architecture": "hub_and_spoke",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            "rounds": [researcher_records + [hub_record]],
            "final_answer": final,
        },
    }
