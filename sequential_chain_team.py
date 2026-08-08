"""
Sequential Chain Team
======================
Agents run in a fixed order: Researcher -> Writer -> Editor -> END.
Each agent reads the shared state, does its job, and passes updated
state to the next node.

If an agent's name appears in the misinformed list passed to run(),
that agent's prompt includes the MINT misinformation text as if it were
extra true information, following Becker et al. 2026's benign setup.

All model calls go through common.call_llm(), which retries with
backoff on rate limits instead of crashing the run.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from common import (
    question_block, context_block, extract_answer,
    ANSWER_INSTRUCTION, call_llm,
)

AGENTS = ["researcher", "writer", "editor"]


class State(TypedDict):
    sample: dict
    misinformed: List[str]
    notes: str
    draft: str
    final_text: str


def _researcher_prompt(sample, misinformed):
    return (
        "You are a researcher. Gather key facts and considerations "
        "needed to answer this question. Be concise, bullet points only.\n\n"
        + question_block(sample)
        + context_block(sample, "researcher" in misinformed)
    )


def _writer_prompt(sample, misinformed, notes):
    return (
        "You are a writer. Using the research notes below, write a clear "
        "draft answer to the question.\n\n"
        + question_block(sample)
        + context_block(sample, "writer" in misinformed)
        + f"\n\nResearch notes:\n{notes}"
    )


def _editor_prompt(sample, misinformed, draft):
    return (
        "You are an editor. Tighten and polish this draft into a final "
        "answer. Fix any unclear phrasing.\n\n"
        + context_block(sample, "editor" in misinformed)
        + f"\n\nDraft:\n{draft}\n\n"
        + ANSWER_INSTRUCTION
    )


def _make_researcher(llm):
    def node(state: State) -> State:
        prompt = _researcher_prompt(state["sample"], state["misinformed"])
        state["notes"] = call_llm(llm, prompt)
        return state
    return node


def _make_writer(llm):
    def node(state: State) -> State:
        prompt = _writer_prompt(state["sample"], state["misinformed"], state["notes"])
        state["draft"] = call_llm(llm, prompt)
        return state
    return node


def _make_editor(llm):
    def node(state: State) -> State:
        prompt = _editor_prompt(state["sample"], state["misinformed"], state["draft"])
        state["final_text"] = call_llm(llm, prompt)
        return state
    return node


def _build_graph(llm):
    graph = StateGraph(State)

    graph.add_node("researcher", _make_researcher(llm))
    graph.add_node("writer", _make_writer(llm))
    graph.add_node("editor", _make_editor(llm))

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    return graph.compile()


def run(llm, sample, misinformed):
    """
    Run one sample through the sequential chain.

    misinformed: list of agent names (subset of AGENTS) that receive the
    MINT misinformation text in their prompt.

    Returns {"final_answer": "A"/"B"/... extracted from the editor's reply}
    """
    app = _build_graph(llm)

    result = app.invoke({
        "sample": sample,
        "misinformed": misinformed,
        "notes": "",
        "draft": "",
        "final_text": "",
    })

    return {"final_answer": extract_answer(result["final_text"], sample)}


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
    out = run(llm, demo_sample, misinformed=["researcher"])
    print(out)