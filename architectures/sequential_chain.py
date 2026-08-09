"""Sequential Chain"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent import Agent, make_llm

ROLES: list[tuple[str, str]] = [
    ("Researcher", "Reason step by step and weigh how well each option fits the situation described."),
    ("Writer", "Give a broad, common-sense best answer."),
    ("Editor", "Question the obvious reading and check whether each option truly follows from the situation."),
]

# Middle seat first (see module docstring), then the chain head, then the tail.
INFLUENCE_ORDER: list[str] = ["Writer", "Researcher", "Editor"]

# Every seat answers as it hands off, so all three count in the denominator.
ANSWERING_AGENTS: list[str] = [name for name, _ in ROLES]


class ChainState(TypedDict):
    question: dict[str, Any]
    records: list[dict[str, Any]]  # chronological: one record per seat


def _handoff(prev: dict[str, Any]) -> str:
    """The chain's transmission channel: what the next seat is told."""
    return (
        f"The previous agent in the chain ({prev['agent']}) answered "
        f"{prev['answer']} (confidence {prev['confidence']}). "
        f"Their rationale: {prev['rationale']} "
        f"Consider their work, then give your own final answer."
    )


def _build_graph(agents: list[Agent]):
    graph = StateGraph(ChainState)

    def make_node(idx: int, agent: Agent):
        def node(state: ChainState) -> dict[str, Any]:
            context = _handoff(state["records"][-1]) if state["records"] else ""
            out = agent.respond(state["question"], context=context, round_num=idx)
            return {"records": state["records"] + [out]}
        return node

    names = [a.name for a in agents]
    for i, agent in enumerate(agents):
        graph.add_node(names[i], make_node(i, agent))
    graph.set_entry_point(names[0])
    for a, b in zip(names, names[1:]):
        graph.add_edge(a, b)
    graph.add_edge(names[-1], END)
    return graph.compile()


def run(
    question: dict[str, Any],
    contamination: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")

    llm = make_llm()
    agents = [
        Agent(name, role, llm, false_belief=belief if name in seeded else None)
        for name, role in ROLES
    ]

    app = _build_graph(agents)
    state = app.invoke({"question": question, "records": []})
    records = state["records"]

    # The tail of the chain decides for the team.
    final = records[-1]["answer"]

    return {
        "final_answer": final,
        "transcript": {
            "architecture": "sequential_chain",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            # One chronological round holding each seat's (only, hence final)
            # answer — exactly one record per answering agent, as the contract
            # requires of the last round.
            "rounds": [records],
            "final_answer": final,
        },
    }
