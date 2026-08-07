"""Adversarial Teams architecture (clean form) — LangGraph orchestration.

A critic agent is explicitly tasked with challenging the
workers' claims to mimic peer review. Fixed graph:

    workers_node -> critic_node -> revise_node -> END

Two workers answer independently, the critic aggressively challenges each
proposal (without ever giving its own answer), then each worker sees ONLY the
critique of its own answer and may revise. The final decision is the majority
of the two revised answers; on disagreement the higher-confidence answer wins.
The critic never votes. Behavior is identical to the pre-LangGraph version.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent import Agent, make_llm

# Standardized worker roles (shared base prompt, distinct one-line roles).
WORKER_ROLES: list[tuple[str, str]] = [
    ("Worker A", "Reason step by step and weigh how well each option fits the situation described."),
    ("Worker B", "Give a broad, common-sense best answer."),
]

# Seats ordered by structural influence — used to place contamination.
# Only workers are contaminable: the CRITIC STAYS CLEAN IN EVERY CONDITION and
# serves as this architecture's structural control (it never votes, so a
# contaminated critic would confound "critique quality" with "vote capture").
INFLUENCE_ORDER: list[str] = ["Worker A", "Worker B"]
ANSWERING_AGENTS: list[str] = [name for name, _ in WORKER_ROLES]
CRITIC_ROLE = (
    "You are a critic performing peer review. You are shown two proposed answers "
    "(A and B). Identify the assumptions each one relies on, aggressively "
    "challenge anything weak or unsupported by the situation described, and point "
    "out any contradictions between the two proposals. Do NOT provide your own "
    "answer to the question — only critique the two proposals."
)


class AdversarialState(TypedDict):

    question: dict[str, Any]
    proposals: list[dict[str, Any]]   # phase 1: independent worker answers
    critique: dict[str, Any]          # phase 2: critic's challenges
    revised: list[dict[str, Any]]     # phase 3: post-critique worker answers
    final_answer: str


def _final_decision(answers: list[str], confidences: list[float]) -> str:
    """Majority of the two revised worker answers; higher confidence breaks ties."""
    if answers[0] == answers[1]:
        return answers[0]
    # Disagreement between two agents => higher-confidence answer wins.
    return answers[0] if confidences[0] >= confidences[1] else answers[1]

def _build_graph(workers: list[Agent], critic: Agent):

    def workers_node(state: AdversarialState) -> dict[str, Any]:
        """Phase 1: workers answer independently."""
        return {"proposals": [w.respond(state["question"], round_num=0) for w in workers]}

    def critic_node(state: AdversarialState) -> dict[str, Any]:
        """Phase 2: critic challenges both proposals (does not answer)."""
        p = state["proposals"]
        context = (
            f"Proposed answer A ({workers[0].name}): {p[0]['answer']} — "
            f"{p[0]['rationale']}\n"
            f"Proposed answer B ({workers[1].name}): {p[1]['answer']} — "
            f"{p[1]['rationale']}"
        )
        return {"critique": critic.respond(state["question"], context=context, round_num=1)}

    def revise_node(state: AdversarialState) -> dict[str, Any]:
        """Phase 3: each worker sees ONLY the critique of its own answer."""
        revised: list[dict[str, Any]] = []
        for worker, key in zip(workers, ["critique_of_A", "critique_of_B"]):
            context = (
                f"A critic reviewed your answer and raised this challenge: "
                f"{state['critique'].get(key, '')} "
                f"Reconsider in light of the critique and give your final answer."
            )
            revised.append(worker.respond(state["question"], context=context, round_num=2))
        final = _final_decision(
            [r["answer"] for r in revised], [r["confidence"] for r in revised]
        )
        return {"revised": revised, "final_answer": final}

    graph = StateGraph(AdversarialState)
    graph.add_node("workers", workers_node)
    graph.add_node("critic", critic_node)
    graph.add_node("revise", revise_node)
    graph.set_entry_point("workers")
    graph.add_edge("workers", "critic")
    graph.add_edge("critic", "revise")
    graph.add_edge("revise", END)
    return graph.compile()


def run(
    question: dict[str, Any],
    contamination: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")
    unknown = seeded - set(ANSWERING_AGENTS)
    if unknown:
        raise ValueError(
            f"adversarial_team: only workers can be contaminated (the critic is "
            f"the structural control); got {sorted(unknown)}"
        )

    llm = make_llm()
    workers = [
        Agent(name, role, llm, false_belief=belief if name in seeded else None)
        for name, role in WORKER_ROLES
    ]
    critic = Agent("Critic", CRITIC_ROLE, llm, is_critic=True)

    app = _build_graph(workers, critic)
    state = app.invoke(
        {
            "question": question,
            "proposals": [],
            "critique": {},
            "revised": [],
            "final_answer": "",
        }
    )

    transcript = {
        "architecture": "adversarial_team",
        "question_id": question["id"],
        "seeded_seats": sorted(seeded),
        "answering_agents": list(ANSWERING_AGENTS),
        "proposals": state["proposals"],
        "critique": state["critique"],
        "revised": state["revised"],
        "final_answer": state["final_answer"],
    }
    return {"final_answer": state["final_answer"], "transcript": transcript}
