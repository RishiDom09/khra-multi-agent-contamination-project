print("!!!!!!!! LOADED MY NEW REDUNDANT.PY !!!!!!!!")
from agent import Agent, make_llm
from concurrent.futures import ThreadPoolExecutor


# ============================================================
# Prompts
# ============================================================

REDUNDANT_AGENT_PROMPT = """
You are one member of a redundant multi-agent team.

Answer the question independently.
Do not rely on or assume the answers of other agents.

Give your single best answer based on the question and options provided.
Briefly explain your reasoning, then clearly state your final answer.
"""


REDUNDANT_AGGREGATOR_PROMPT = """
You are the aggregator for a redundant multi-agent team.

Several researchers independently answered the same question.

Determine the final answer using MAJORITY VOTE ONLY.

Rules:
1. Count the researchers' answers.
2. If more researchers answered A than B, output A.
3. If more researchers answered B than A, output B.
4. If there is a tie or no valid answers, output "".

Do NOT independently solve the question.
Do NOT override the majority.
Do NOT use information other than the researchers' answers to determine the final answer.

Give your single best answer.
"""


# ============================================================
# Architecture definition
# ============================================================

RESEARCHERS = [
    "Researcher 1",
    "Researcher 2",
    "Researcher 3",
    "Researcher 4",
    "Researcher 5",
]

ANSWERING_AGENTS = RESEARCHERS

# All researchers are equally independent, so there is no meaningful
# structural influence difference. This order simply determines which
# researchers are contaminated first when multiple seats are seeded.
INFLUENCE_ORDER = RESEARCHERS


def run(question, contamination=None):

    contamination = contamination or {}

    seeded = set(contamination.get("seats") or [])
    belief = contamination.get("false_belief")

    rounds = []

    # ============================================================
    # Round 1: Independent researchers
    # ============================================================

    def run_researcher(role):
        agent = Agent(
            role,
            REDUNDANT_AGENT_PROMPT,
            make_llm(),
            false_belief=belief if role in seeded else None,
        )

        return agent.respond(
            question,
            round_num=0,
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        researcher_records = list(
            executor.map(run_researcher, RESEARCHERS)
        )

    rounds.append(researcher_records)

    # ============================================================
    # Round 2: Aggregator
    # ============================================================

    context = "\n\n".join(
        f"{record['agent']}: {record['answer']}"
        for record in researcher_records
    )

    aggregator = Agent(
        "Aggregator",
        REDUNDANT_AGGREGATOR_PROMPT,
        make_llm(),
    )

    final_record = aggregator.respond(
        question,
        context=context,
        round_num=1,
    )

    rounds.append([final_record])

    # ============================================================
    # Standardized plug-in output
    # ============================================================

    return {
        "final_answer": final_record["answer"],
        "transcript": {
            "architecture": "redundant",
            "question_id": question["id"],
            "seeded_seats": sorted(seeded),
            "answering_agents": list(ANSWERING_AGENTS),
            "rounds": rounds,
            "final_answer": final_record["answer"],
        },
    }