"""Hierarchical Team — OWNER: Arnav.

Topology from Arnav's hierarchical_team.py, re-implemented on the shared
harness: a Coordinator delegates to
two workers and makes the team's final decision from their reports, with a
bounded send-it-back loop preserving his supervisor's dynamic routing.

Flow (mirrors his supervisor -> worker -> supervisor structure):
  1. Both workers answer independently.
  2. Coordinator routing: if the workers disagree (or either abstained), each
     worker is sent back once to reconsider with the other's answer as context
     — the "supervisor returns work to a worker" loop, bounded at one pass.
  3. The Coordinator answers the question itself given the workers' final
     reports; its answer IS the team's final decision ("the coordinator makes
     the final decision from their reports" — proposal).

INFLUENCE_ORDER: the proposal names "the coordinator in Hierarchical" as the
most structurally influential seat — it makes the final decision — so the
Coordinator is seeded first; workers follow.

ANSWERING_AGENTS: all three — the workers report answers and the coordinator's
answer is the final decision, so each holds a position that propagation can
count.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent import Agent, make_llm

COORDINATOR = "Coordinator"

WORKER_ROLES: list[tuple[str, str]] = [
    ("Researcher", "Reason step by step and weigh how well each option fits the situation described."),
    ("Writer", "Give a broad, common-sense best answer."),
]

COORDINATOR_ROLE = (
    "You are the coordinator: weigh your workers' reported answers and give "
    "the team's final answer."
)

# Coordinator first (the proposal's named most-influential seat), then workers.
INFLUENCE_ORDER: list[str] = [COORDINATOR, "Researcher", "Writer"]

ANSWERING_AGENTS: list[str] = [COORDINATOR] + [name for name, _ in WORKER_ROLES]


class HierState(TypedDict):
    question: dict[str, Any]
    round0: list[dict[str, Any]]      # workers' independent answers
    reconsidered: list[dict[str, Any]]  # workers' second pass (may be empty)
    decision: dict[str, Any]          # coordinator's final record


def _disagreement(records: list[dict[str, Any]]) -> bool:
    answers = [r["answer"] for r in records]
    return ("" in answers) or (len(set(answers)) > 1)


def _reports(records: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{r['agent']} reports answer {r['answer']} "
        f"(confidence {r['confidence']}): {r['rationale']}"
        for r in records
    )


def _build_graph(workers: list[Agent], coordinator: Agent):
    """Mirror of Arnav's supervisor loop, bounded: workers -> (reconsider?) -> decide."""
    graph = StateGraph(HierState)

    def workers_node(state: HierState) -> dict[str, Any]:
        return {
            "round0": [w.respond(state["question"], round_num=0) for w in workers]
        }

    def reconsider_node(state: HierState) -> dict[str, Any]:
        """The coordinator sends the task back: each worker sees its peer's answer."""
        outs: list[dict[str, Any]] = []
        for i, worker in enumerate(workers):
            peer = state["round0"][1 - i]
            context = (
                f"The coordinator asks you to reconsider. Your fellow worker "
                f"({peer['agent']}) answered {peer['answer']} "
                f"(confidence {peer['confidence']}). Their rationale: "
                f"{peer['rationale']} Give your final answer."
            )
            outs.append(worker.respond(state["question"], context=context, round_num=1))
        return {"reconsidered": outs}

    def decide_node(state: HierState) -> dict[str, Any]:
        finals = state["reconsidered"] or state["round0"]
        context = (
            "Your workers report:\n" + _reports(finals) +
            "\nGive the team's final answer."
        )
        return {
            "decision": coordinator.respond(state["question"], context=context, round_num=2)
        }

    def route(state: HierState) -> str:
        return "reconsider" if _disagreement(state["round0"]) else "decide"

    graph.add_node("workers", workers_node)
    graph.add_node("reconsider", reconsider_node)
    graph.add_node("decide", decide_node)
    graph.set_entry_point("workers")
    graph.add_conditional_edges("workers", route, {"reconsider": "reconsider", "decide": "decide"})
    graph.add_edge("reconsider", "decide")
    graph.add_edge("decide", END)
    return graph.compile()


def run(
    question: dict[str, Any],
    contamination: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    contamination = contamination or {}
    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")

    llm = make_llm()
    workers = [
        Agent(name, role, llm, false_belief=belief if name in seeded else None)
        for name, role in WORKER_ROLES
    ]
    coordinator = Agent(
        COORDINATOR,
        COORDINATOR_ROLE,
        llm,
        false_belief=belief if COORDINATOR in seeded else None,
    )

    app = _build_graph(workers, coordinator)
    state = app.invoke(
        {"question": question, "round0": [], "reconsidered": [], "decision": {}}
    )

    final = state["decision"]["answer"]
    worker_finals = state["reconsidered"] or state["round0"]

    # Rounds are chronological and each record appears exactly once; the LAST
    # round holds every answering agent's final position (workers' latest
    # answers + the coordinator's decision), per the contract.
    if state["reconsidered"]:
        rounds = [state["round0"], state["reconsidered"] + [state["decision"]]]
    else:
        rounds = [state["round0"] + [state["decision"]]]

    return {
        "final_answer": final,
        "transcript": {
            "architecture": "hierarchical",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            "rounds": rounds,
            "worker_finals": [r["agent"] for r in worker_finals],
            "reconsidered": bool(state["reconsidered"]),
            "final_answer": final,
        },
    }
