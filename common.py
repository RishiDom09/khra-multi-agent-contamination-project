"""
Shared helpers for the Algoverse contamination experiments.

Everything the two architectures need lives here:
  - loading MINT from Hugging Face
  - building the uninformed vs misinformed prompt
  - stripping qwen3 thinking tags and pulling out the final answer letter
  - a retrying Groq client so free tier rate limits do not kill a run
"""

import os
import re
import time
import random

from datasets import load_dataset
from langchain_groq import ChatGroq


MODEL_NAME = "qwen/qwen3.6-27b"
MINT_REPO = "jonasbecker/MINT"

# MINT's three benchmark subsets live as separate files in the repo,
# not as HF "configs". Always pass one of these to data_files.
FILES = {
    "winogrande": "winogrande_misinformed.json",
    "cwq": "complex_web_questions_misinformed.json",
    "ethics": "ethics_commonsense_misinformed.json",
}

# The nine MINT intent categories. These are the exact keys inside
# each row's misinformation_by_strategy dict.
CATEGORIES = [
    "neutral", "clickbait", "hoax", "rumor", "satire",
    "propaganda", "framing", "conspiracy", "other",
]


# ----------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------

def get_llm(temperature=0.0):
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. In PowerShell run:\n"
            '  $env:GROQ_API_KEY="your_key_here"'
        )
    return ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        reasoning_effort="none",
    )


def call_llm(llm, prompt, max_retries=6):
    """Call the model, retrying with backoff on rate limits."""
    delay = 5
    for attempt in range(max_retries):
        try:
            return strip_thinking(llm.invoke(prompt).content)
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "quota" in msg:
                print(f"    rate limited, waiting {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 90)
            elif attempt == max_retries - 1:
                raise
            else:
                print(f"    error: {e}, retrying")
                time.sleep(delay)
    raise RuntimeError("gave up after repeated failures")


def strip_thinking(text):
    """qwen3 emits <think>...</think>. Remove it before parsing."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ----------------------------------------------------------------------
# MINT loading
# ----------------------------------------------------------------------

def inspect_mint(subset=None):
    """Print MINT's real schema. Run this once before anything else."""
    ds = _raw_mint(subset)
    print("Columns:", ds.column_names)
    print("Rows:", len(ds))
    print("\nFirst row:")
    row = ds[0]
    for k, v in row.items():
        preview = str(v)
        if len(preview) > 300:
            preview = preview[:300] + " ..."
        print(f"  {k}: {preview}")
    return ds


def _raw_mint(subset=None):
    """
    Load exactly one MINT subset file. subset must be one of the keys
    in FILES ("winogrande", "cwq", "ethics"). Defaults to winogrande.
    """
    filename = FILES.get(subset or "winogrande", FILES["winogrande"])
    try:
        return load_dataset(MINT_REPO, data_files=filename, split="train")
    except Exception:
        # Some repos expose "test" instead of "train" for a given file.
        return load_dataset(MINT_REPO, data_files=filename, split="test")


def load_mint(subset="winogrande", n=100, category="hoax", seed=42):
    """
    Return a list of dicts shaped like:
      {id, question, options: [..], gold, misinformation}

    Real MINT columns confirmed via inspect_mint():
      MINT_ID, id, sentence, answer, options,
      false_fact, misinformation_by_strategy,
      irrelevant_true_information, avg_misinformation_tokens,
      irrelevant_true_information_tokens, metadata

    - sentence -> the question (WinoGrande fill-in-the-blank)
    - answer   -> "1" or "2", which option is correct
    - options  -> the two choices
    - misinformation_by_strategy -> dict keyed by category
      ("hoax", "clickbait", etc.), matching CATEGORIES above
    """
    ds = _raw_mint(subset)

    rows = list(range(len(ds)))
    random.Random(seed).shuffle(rows)

    out = []
    for idx in rows:
        row = ds[idx]

        question = row.get("sentence")
        options = row.get("options")
        gold = row.get("answer")

        strategies = row.get("misinformation_by_strategy") or {}
        misinfo = strategies.get(category)

        if not question or not gold or not misinfo or not options:
            continue

        out.append({
            "id": row.get("MINT_ID", f"{subset}-{idx}"),
            "question": str(question),
            "options": [str(o) for o in options],
            "gold": str(gold),
            "misinformation": str(misinfo),
        })

        if len(out) >= n:
            break

    if not out:
        raise RuntimeError(
            "Could not map MINT fields. Columns are:\n"
            f"  {ds.column_names}\n"
            "Run  python inspect_mint.py  and tell Claude what it prints."
        )
    return out


# ----------------------------------------------------------------------
# Prompts and scoring
# ----------------------------------------------------------------------

def letters(sample):
    return [chr(ord("A") + i) for i in range(len(sample["options"]))]


def question_block(sample):
    text = sample["question"]
    if sample["options"]:
        lines = [
            f"{L}) {opt}"
            for L, opt in zip(letters(sample), sample["options"])
        ]
        text += "\n\nAnswer Choices:\n" + "\n".join(lines)
    return text


def context_block(sample, misinformed):
    """
    The only difference between an uninformed and a misinformed agent.
    The agent is never told the extra information might be false, which
    is the benign setup used by Becker et al. 2026.
    """
    if not misinformed:
        return ""
    return f"\n\nYou have this extra information:\n{sample['misinformation']}\n"


def gold_letter(sample):
    """Normalise the gold answer to a letter where possible."""
    g = sample["gold"].strip()
    if len(g) == 1 and g.upper() in letters(sample):
        return g.upper()
    if g in ("1", "2", "3", "4"):
        return chr(ord("A") + int(g) - 1)
    for L, opt in zip(letters(sample), sample["options"]):
        if opt.strip().lower() == g.lower():
            return L
    return g


ANSWER_RE = re.compile(r"\b(?:answer|choice)\b[^A-Za-z]{0,12}([A-D])\b", re.I)


def extract_answer(text, sample):
    """Pull the final answer letter out of an agent's reply."""
    valid = set(letters(sample)) or {"A", "B", "C", "D"}

    matches = ANSWER_RE.findall(text)
    if matches:
        cand = matches[-1].upper()
        if cand in valid:
            return cand

    # Otherwise take the last standalone capital letter in the reply.
    standalone = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
    for cand in reversed(standalone):
        if cand.upper() in valid:
            return cand.upper()

    return ""


ANSWER_INSTRUCTION = (
    "End your reply with a single line in exactly this format:\n"
    "ANSWER: X\n"
    "where X is the letter of your chosen option. Nothing after that line."
)