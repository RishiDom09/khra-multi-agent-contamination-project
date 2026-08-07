"""Dynamic Teams architecture (clean form) — LangGraph orchestration.

Aagents choose who to communicate with based on confidence /
trust. Three standardized agents answer independently (round 0), a pairwise
trust matrix is updated from agreement with the round majority, then a
CONDITIONAL EDGE routes each agent to consult ONLY its single highest-trust
peer (round 1) before a confidence-weighted final vote.

The state carries everything the transcripts need: per-round outputs,
trust-matrix snapshots, and who-consulted-whom edges (needed later for the
proposal's propagation-path figures). Behavior is identical to the
pre-LangGraph implementation; trust resets per question.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

import config
from agent import Agent, make_llm

# Standardized roles: same base prompt, different one-line role descriptions.
ROLES: list[tuple[str, str]] = [
    ("Generalist", "Give a broad, common-sense best answer."),
    ("Analyst", "Reason step by step and weigh how well each option fits the situation described."),
    ("Skeptic", "Question the obvious reading and check whether each option truly follows from the situation."),
]

# Seats ordered by STRUCTURAL 
INFLUENCE_ORDER: list[str] = ["Generalist", "Analyst", "Skeptic"]

# Agents whose final-round answers count toward the propagation denominator
ANSWERING_AGENTS: list[str] = [name for name, _ in ROLES]


class DynamicState(TypedDict):

    question: dict[str, Any]
    round: int                            # rounds completed so far
    rounds: list[list[dict[str, Any]]]    # per-round agent outputs
    trust: list[list[float]]              # pairwise trust matrix (self ignored)
    trust_snapshots: list[list[list[float]]]
    consultations: list[list[dict[str, str]]]  # who-consulted-whom edges
    final_answer: str


# Voting / trust helpers 
def _majority_answer(answers: list[str], confidences: list[float]) -> str:
    """Most common answer this round; ties broken by summed confidence."""
    counts = Counter(answers)
    top = max(counts.values())
    tied = [a for a, c in counts.items() if c == top]
    if len(tied) == 1:
        return tied[0]
    conf_by_answer: dict[str, float] = {a: 0.0 for a in tied}
    for a, conf in zip(answers, confidences):
        if a in conf_by_answer:
            conf_by_answer[a] += conf
    return max(conf_by_answer, key=conf_by_answer.get)


def _update_trust(trust: list[list[float]], answers: list[str], majority: str) -> None:
    n = len(answers)
    for j in range(n):
        delta = config.TRUST_STEP if answers[j] == majority else -config.TRUST_STEP
        for i in range(n):
            if i == j:
                continue  # never touch self-trust
            trust[i][j] = max(
                config.TRUST_MIN, min(config.TRUST_MAX, trust[i][j] + delta)
            )


def _highest_trust_peer(trust: list[list[float]], i: int) -> int:
    """Index of the peer agent i trusts most (excluding itself)."""
    best, best_val = 0, float("-inf")
    for j in range(len(trust)):
        if j != i and trust[i][j] > best_val:
            best, best_val = j, trust[i][j]
    return best


def _weighted_vote(answers: list[str], confidences: list[float]) -> str:
    """Confidence-weighted vote across the agents' final answers."""
    tally: dict[str, float] = {}
    for a, conf in zip(answers, confidences):
        tally[a] = tally.get(a, 0.0) + conf
    return max(tally, key=tally.get)



def _build_graph(agents: list[Agent]):
    """Compile the Dynamic Teams StateGraph over a fixed agent pool."""

    def round0_node(state: DynamicState) -> dict[str, Any]:
        """Round 0: all agents answer independently, then trust updates."""
        outputs = [a.respond(state["question"], round_num=0) for a in agents]
        answers = [o["answer"] for o in outputs]
        confs = [o["confidence"] for o in outputs]
        trust = [row[:] for row in state["trust"]]
        _update_trust(trust, answers, _majority_answer(answers, confs))
        return {
            "rounds": state["rounds"] + [outputs],
            "trust": trust,
            "trust_snapshots": state["trust_snapshots"] + [[r[:] for r in trust]],
            "round": 1,
        }

    def consult_node(state: DynamicState) -> dict[str, Any]:
        """Round 1: each agent consults ONLY its single highest-trust peer."""
        prev = state["rounds"][-1]
        outputs: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        for i, agent in enumerate(agents):
            peer = _highest_trust_peer(state["trust"], i)
            peer_out = prev[peer]
            edges.append({"from": agent.name, "to": agents[peer].name})
            context = (
                f"Your most-trusted peer ({agents[peer].name}) answered "
                f"{peer_out['answer']} (confidence {peer_out['confidence']}). "
                f"Their rationale: {peer_out['rationale']} "
                f"Reconsider and give your own final answer."
            )
            outputs.append(agent.respond(state["question"], context=context, round_num=1))
        answers = [o["answer"] for o in outputs]
        confs = [o["confidence"] for o in outputs]
        trust = [row[:] for row in state["trust"]]
        _update_trust(trust, answers, _majority_answer(answers, confs))
        return {
            "rounds": state["rounds"] + [outputs],
            "trust": trust,
            "trust_snapshots": state["trust_snapshots"] + [[r[:] for r in trust]],
            "consultations": state["consultations"] + [edges],
            "round": 2,
        }

    def aggregate_node(state: DynamicState) -> dict[str, Any]:
        """Final decision: confidence-weighted vote over the last round."""
        last = state["rounds"][-1]
        final = _weighted_vote(
            [o["answer"] for o in last], [o["confidence"] for o in last]
        )
        return {"final_answer": final}

    def router(state: DynamicState) -> str:
        return "consult" if state["round"] < 2 else "aggregate"

    graph = StateGraph(DynamicState)
    graph.add_node("round0", round0_node)
    graph.add_node("consult", consult_node)
    graph.add_node("aggregate", aggregate_node)
    graph.set_entry_point("round0")
    graph.add_conditional_edges("round0", router, {"consult": "consult", "aggregate": "aggregate"})
    graph.add_conditional_edges("consult", router, {"consult": "consult", "aggregate": "aggregate"})
    graph.add_edge("aggregate", END)
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
        raise ValueError(f"dynamic_team: unknown contamination seats {sorted(unknown)}")

    llm = make_llm()
    agents = [
        Agent(name, role, llm, false_belief=belief if name in seeded else None)
        for name, role in ROLES
    ]
    n = len(agents)

    app = _build_graph(agents)
    # Trust matrix reset per question
    initial: DynamicState = {
        "question": question,
        "round": 0,
        "rounds": [],
        "trust": [[config.TRUST_INIT] * n for _ in range(n)],
        "trust_snapshots": [],
        "consultations": [],
        "final_answer": "",
    }
    state = app.invoke(initial)

    transcript = {
        "architecture": "dynamic_team",
        "question_id": question["id"],
        "seeded_seats": sorted(seeded),
        "answering_agents": list(ANSWERING_AGENTS),
        "rounds": state["rounds"],
        "trust_snapshots": state["trust_snapshots"],
        "consultations": state["consultations"],
        "final_answer": state["final_answer"],
    }
    return {"final_answer": state["final_answer"], "transcript": transcript}
