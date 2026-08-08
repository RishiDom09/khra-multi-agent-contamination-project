"""
Hierarchical Team
==================
One supervisor agent decides which worker should act next, based on
the current state. Workers do their job and report back to the
supervisor. The supervisor loops until it decides the task is done.

Order isn't fixed here, unlike Sequential Chain: the supervisor makes
that call dynamically, and could send work back to the same worker
again, skip a worker, or finish early.

If an agent's name appears in the misinformed list passed to run(),
that agent's prompt includes the MINT misinformation text as if it were
extra true information, following Becker et al. 2026's benign setup.

All model calls go through common.call_llm(), which retries with
backoff on rate limits instead of crashing the run.
"""

from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END

from common import (
    question_block, context_block, extract_answer,
    ANSWER_INSTRUCTION, call_llm,
)

# Only the workers can be contaminated. The supervisor just routes,
# so it is not a candidate for misinformation injection.
AGENTS = ["researcher", "writer"]

MAX_LOOPS = 6  # safety guard so a stuck supervisor cannot loop forever


class State(TypedDict):
    sample: dict
    misinformed: List[str]
    notes: str
    draft: str
    next_step: str
    loops: int


def _make_supervisor(llm):
    def node(state: State) -> State:
        state["loops"] += 1
        if state["loops"] > MAX_LOOPS:
            state["next_step"] = "FINISH"
            return state

        prompt = (
            "You are a supervisor managing two workers: 'researcher' and 'writer'.\n\n"
            + question_block(state["sample"])
            + f"\n\nResearch notes so far: {state['notes'] or '(none yet)'}\n"
            f"Draft so far: {state['draft'] or '(none yet)'}\n\n"
            "Decide what should happen next. Reply with exactly one word:\n"
            "'researcher' if research notes are missing or incomplete,\n"
            "'writer' if research is done but draft is missing,\n"
            "'FINISH' if the draft is complete and answers the question well."
        )
        decision = call_llm(llm, prompt).strip().lower()

        if "finish" in decision:
            state["next_step"] = "FINISH"
        elif "writer" in decision:
            state["next_step"] = "writer"
        else:
            state["next_step"] = "researcher"

        return state
    return node


def _make_researcher(llm):
    def node(state: State) -> State:
        prompt = (
            "You are a researcher. Gather key facts needed to answer this "
            "question. Be concise, bullet points only.\n\n"
            + question_block(state["sample"])
            + context_block(state["sample"], "researcher" in state["misinformed"])
        )
        state["notes"] = call_llm(llm, prompt)
        return state
    return node


def _make_writer(llm):
    def node(state: State) -> State:
        prompt = (
            "You are a writer. Using the research notes below, write a clear "
            "final answer to the question.\n\n"
            + question_block(state["sample"])
            + context_block(state["sample"], "writer" in state["misinformed"])
            + f"\n\nResearch notes:\n{state['notes']}\n\n"
            + ANSWER_INSTRUCTION
        )
        state["draft"] = call_llm(llm, prompt)
        return state
    return node


def _route(state: State) -> Literal["researcher", "writer", "__end__"]:
    if state["next_step"] == "FINISH":
        return END
    return state["next_step"]


def _build_graph(llm):
    graph = StateGraph(State)

    graph.add_node("supervisor", _make_supervisor(llm))
    graph.add_node("researcher", _make_researcher(llm))
    graph.add_node("writer", _make_writer(llm))

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", _route, {
        "researcher": "researcher",
        "writer": "writer",
        END: END,
    })

    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


def run(llm, sample, misinformed):
    """
    Run one sample through the hierarchical team.

    misinformed: list of agent names (subset of AGENTS) that receive the
    MINT misinformation text in their prompt.

    Returns {"final_answer": "A"/"B"/... extracted from the writer's draft}
    """
    app = _build_graph(llm)

    result = app.invoke({
        "sample": sample,
        "misinformed": misinformed,
        "notes": "",
        "draft": "",
        "next_step": "",
        "loops": 0,
    })

    return {"final_answer": extract_answer(result["draft"], sample)}


if __name__ == "__main__":
    # Quick manual check, not used by run_experiment.py
    from common import get_llm

    demo_sample = {
        "id": "demo",
        "question": "The trophy doesn't fit in the suitcase because _ is too big.",
        "options": ["trophy", "suitcase"],
        "gold": "1",
        "misinformation": "Recent reports suggest the suitcase was actually the larger item.",
    }

    llm = get_llm()
    out = run(llm, demo_sample, misinformed=["writer"])
    print(out)