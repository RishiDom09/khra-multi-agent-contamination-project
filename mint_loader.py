"""Load the MINT dataset and normalize it into this project's question schema.

MINT (https://github.com/jonas-becker/mint) pairs each task instance with
generated misinformation about it, across eight rhetorical strategies. We use two
of its multiple-choice subsets:

* ``winogrande_misinformed.json``       — commonsense pronoun resolution
* ``ethics_commonsense_misinformed.json`` — moral judgement (Ethical/Unethical)

``complex_web_questions`` is intentionally excluded: it is free-form and needs a
different scorer. The loader is structured so a free-form dataset can be added
later (see :data:`DATASETS` and the ``answer_type`` field) without disturbing the
multiple-choice path.

Everything downstream consumes the internal schema, unchanged::

    {"id", "question", "options": {"A": .., "B": ..}, "correct": "<letter>",
     "category": "winogrande"|"ethics", "misinfo": "<strategy text>"}

so the architectures need no schema changes.

Run this module directly to verify the load and print a sample::

    python mint_loader.py
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from typing import Any, Callable, Optional

import config

RAW_BASE = (
    "https://raw.githubusercontent.com/jonas-becker/mint/main/MINT-dataset_v1.1"
)

LETTERS = "ABCDEFGH"


class DatasetSpec:
    """How to read one MINT subset.

    ``label_file`` names a GOLD-LABEL override keyed by ``MINT_ID``. It exists
    because **MINT's winogrande ``answer`` field is unusable**: it is the constant
    "1" for all 382 items, so it encodes nothing about which option is correct.
    Measured against the published WinoGrande gold labels, resolving it as
    1-indexed is right for 190 items and 0-indexed for 189 — i.e. either rule is a
    coin flip. Ground truth for winogrande therefore comes from the override file
    (built by joining the original allenai/winogrande dataset on sentence text),
    never from ``answer``.

    ``answer_base`` is used only for subsets WITHOUT an override file (currently
    ethics_commonsense, whose ``answer`` field does vary and is 0-indexed).
    """

    def __init__(
        self,
        category: str,
        filename: str,
        answer_base: int,
        answer_type: str = "multiple_choice",
        label_file: Optional[str] = None,
    ) -> None:
        self.category = category
        self.filename = filename
        self.answer_base = answer_base
        self.answer_type = answer_type
        self.label_file = label_file


DATASETS: dict[str, DatasetSpec] = {
    "winogrande": DatasetSpec(
        "winogrande", "winogrande_misinformed.json", answer_base=1,
        label_file="winogrande_labels.json",
    ),
    "ethics": DatasetSpec(
        "ethics", "ethics_commonsense_misinformed.json", answer_base=0
    ),
}


# --------------------------------------------------------------------------- #
# Gold-label overrides
# --------------------------------------------------------------------------- #
_GOLD_CACHE: dict[str, dict[str, str]] = {}


def gold_labels(spec: DatasetSpec) -> dict[str, str]:
    """Return ``{MINT_ID: correct option text}`` for a subset, or {} if none."""
    if not spec.label_file:
        return {}
    if spec.category in _GOLD_CACHE:
        return _GOLD_CACHE[spec.category]
    path = os.path.join(config.DATA_DIR, spec.label_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{spec.category}: gold-label file {path} is missing. MINT's own "
            f"'answer' field for this subset is unusable (constant), so the "
            f"experiment cannot be scored without it."
        )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    labels = payload.get("labels", payload)
    _GOLD_CACHE[spec.category] = labels
    return labels


# --------------------------------------------------------------------------- #
# Download / disk cache
# --------------------------------------------------------------------------- #
def _local_path(spec: DatasetSpec) -> str:
    return os.path.join(config.DATA_DIR, spec.filename)


def _ensure_downloaded(spec: DatasetSpec) -> str:
    """Fetch the subset into ``data/`` once; load from disk on every later run."""
    path = _local_path(spec)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(config.DATA_DIR, exist_ok=True)
    url = f"{RAW_BASE}/{spec.filename}"
    print(f"[mint_loader] downloading {spec.filename} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            payload = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download {url}: {exc}\n"
            f"Download it manually into {config.DATA_DIR}/ and rerun."
        ) from exc
    with open(path, "wb") as fh:
        fh.write(payload)
    return path


def load_raw(category: str) -> list[dict[str, Any]]:
    """Return the raw MINT items for one category."""
    spec = DATASETS[category]
    with open(_ensure_downloaded(spec), encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{spec.filename}: expected a JSON list, got {type(data)}")
    return data


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize(
    item: dict[str, Any],
    spec: DatasetSpec,
    index: int,
    strategy: str = "other",
    rng: Optional[random.Random] = None,
) -> dict[str, Any]:
    """Convert one raw MINT item into the internal question schema.

    ``rng`` (optional) deterministically shuffles the option order. See
    :func:`load_questions` for why that matters.
    """
    options_list = list(item["options"])
    if len(options_list) < 2:
        raise ValueError(f"{item.get('MINT_ID')}: fewer than 2 options")

    # Ground truth: the gold-label override when the subset has one (winogrande),
    # otherwise MINT's own answer field (ethics). Items with no gold label are
    # marked and dropped by load_category — but only AFTER the option shuffle
    # below, so excluding them never perturbs any other item's A/B arrangement.
    gold = gold_labels(spec)
    label_missing = False
    if gold:
        correct_text = gold.get(str(item.get("MINT_ID")))
        if correct_text is None or correct_text not in options_list:
            label_missing = True
            correct_text = options_list[0]  # placeholder; item will be dropped
        label_source = "gold"
    else:
        correct_idx = int(item["answer"]) - spec.answer_base
        if not 0 <= correct_idx < len(options_list):
            raise ValueError(
                f"{item.get('MINT_ID')}: answer {item['answer']!r} with answer_base "
                f"{spec.answer_base} resolves to out-of-range index {correct_idx}"
            )
        correct_text = options_list[correct_idx]
        label_source = "mint_answer_field"

    if rng is not None:
        rng.shuffle(options_list)
    correct_idx = options_list.index(correct_text)

    options = {LETTERS[i]: text for i, text in enumerate(options_list)}
    correct_letter = LETTERS[correct_idx]

    strategies = item.get("misinformation_by_strategy") or {}
    if strategy not in strategies:
        raise KeyError(
            f"{item.get('MINT_ID')}: strategy {strategy!r} not present "
            f"(available: {sorted(strategies)})"
        )

    return {
        "id": f"{spec.category}_{index:04d}",
        "mint_id": item.get("MINT_ID"),
        "question": item["sentence"],
        "options": options,
        "correct": correct_letter,
        "correct_text": correct_text,
        "category": spec.category,
        "answer_type": spec.answer_type,
        "strategy": strategy,
        "misinfo": strategies[strategy],
        "false_fact": item.get("false_fact", ""),
        "label_source": label_source,
        "label_missing": label_missing,
    }


def load_category(
    category: str,
    strategy: str = "other",
    rng: Optional[random.Random] = None,
    keep_unlabeled: bool = False,
) -> list[dict[str, Any]]:
    """Normalize every item in one category, dropping any with no gold label.

    Normalization runs over ALL raw items first so the option-shuffle RNG advances
    identically regardless of how many are dropped; the filter is applied only
    afterwards. That keeps every retained item's A/B arrangement stable, so
    transcripts recorded before an exclusion remain valid and re-usable.
    """
    spec = DATASETS[category]
    items = [
        normalize(item, spec, i, strategy=strategy, rng=rng)
        for i, item in enumerate(load_raw(category))
    ]
    if keep_unlabeled:
        return items
    return [q for q in items if not q["label_missing"]]


# --------------------------------------------------------------------------- #
# Verification — fail loudly rather than silently corrupt ground truth
# --------------------------------------------------------------------------- #
# Each check: (category, substring of the question, expected correct option TEXT).
# Asserting on the option *text* rather than the letter keeps these valid whether
# or not option order is shuffled.
# NOTE: the old winogrande spot-check ("A tortoise crawled..." -> William) has been
# REMOVED. It was one of the ~50% of items that happened to resolve correctly under
# the broken answer field, so it passed while half the dataset was mislabeled — the
# exact false assurance this list must never give again. Winogrande is now verified
# item-by-item against gold labels instead.
VERIFICATION_CASES: list[tuple[str, str, str]] = [
    ("ethics", "I suggested my mother-in-law move to Kansas", "Unethical"),
    ("ethics", "I gave the shivering child a thick blanket", "Ethical"),
]


def verify(strategy: str = "other", verbose: bool = False) -> None:
    """Check ground truth for EVERY item, not a hand-picked few. Raises on mismatch.

    The earlier version asserted three hand-picked items and passed while ~50% of
    winogrande labels were wrong — one of the three happened to be among the half
    that resolved correctly. This version checks all of them:

    * gold-label subsets: every retained item's ``correct`` letter must point at
      exactly the option text the gold file specifies;
    * non-gold subsets: the legacy spot-checks still apply.
    """
    for category, spec in DATASETS.items():
        gold = gold_labels(spec)
        items = load_category(category, strategy=strategy, keep_unlabeled=True)

        if gold:
            checked = dropped = 0
            for it in items:
                if it["label_missing"]:
                    dropped += 1
                    continue
                expected = gold[str(it["mint_id"])]
                actual = it["options"][it["correct"]]
                if actual != expected:
                    raise AssertionError(
                        f"GROUND TRUTH MISMATCH in {category}: item {it['mint_id']} "
                        f"resolved to {actual!r} but the gold label is {expected!r}. "
                        f"Do not run the experiment until this is fixed."
                    )
                checked += 1
            if verbose:
                print(f"[verify] {category}: {checked} items match gold labels "
                      f"({dropped} unlabeled, excluded)")
            continue

        # Legacy spot-checks for subsets that still use MINT's answer field.
        by_text = {it["question"]: it for it in items}
        for check_cat, needle, expected in VERIFICATION_CASES:
            if check_cat != category:
                continue
            matches = [it for q, it in by_text.items() if needle in q]
            if not matches:
                raise AssertionError(
                    f"MINT verification: no {category} item containing {needle!r}. "
                    f"The dataset file may have changed — recheck the index rule."
                )
            item = matches[0]
            actual = item["options"][item["correct"]]
            if actual != expected:
                raise AssertionError(
                    f"MINT index mapping is WRONG for {category}: item "
                    f"{item['mint_id']} ({needle!r}) resolved to {actual!r}, "
                    f"expected {expected!r}. "
                    f"answer_base={DATASETS[category].answer_base}. "
                    f"Fix DatasetSpec.answer_base before running the experiment."
                )
            if verbose:
                print(f"[verify] {category}: spot-check {needle!r} -> {actual!r} OK")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def load_questions(
    strategy: str = config.MINT_STRATEGY,
    shuffle_options: bool = True,
    categories: Optional[list[str]] = None,
    seed: int = config.RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Return ALL normalized MINT questions for the active categories, in file order.

    There is NO sampling: every run of every model answers the identical full
    set (currently the 379 verified winogrande items), which is what makes
    cross-model comparisons paired without any manifest bookkeeping.

    The only randomness left is the seeded OPTION-ORDER shuffle.
    **``shuffle_options`` is on by default because MINT's winogrande subset has
    ``answer == "1"`` for all 382 items** — i.e. the correct option is always
    listed first. With a fixed positional letter mapping the correct answer
    would be "A" for every winogrande question, and any positional bias in the
    model would masquerade as commonsense accuracy. The shuffle RNG is seeded
    per category (``f"{seed}:{category}:options"``) and ``id`` is tied to the
    item's position in the source file, so every model sees byte-identical
    options, correct letter, and misinfo text for every question. Pass
    ``shuffle_options=False`` for the literal unshuffled mapping.
    """
    verify(strategy=strategy)

    selected: list[dict[str, Any]] = []
    for category in categories or config.ACTIVE_CATEGORIES:
        opt_rng = random.Random(f"{seed}:{category}:options") if shuffle_options else None
        selected.extend(load_category(category, strategy=strategy, rng=opt_rng))
    return selected


def misinfo_aligned_letter(question: dict[str, Any]) -> str:
    """The option letter the misinformation steers toward.

    For these 2-option MINT items that is simply the non-correct option. Kept as
    a named function so a >2-option or free-form dataset can override it without
    touching the metric code.
    """
    wrong = [k for k in question["options"] if k != question["correct"]]
    if len(wrong) != 1:
        raise ValueError(
            f"{question['id']}: misinfo-aligned answer is only well defined for "
            f"2-option items (got {len(question['options'])})."
        )
    return wrong[0]


# --------------------------------------------------------------------------- #
# Free-form hook (complex_web_questions and friends, added later)
# --------------------------------------------------------------------------- #
SCORERS: dict[str, Callable[[str, dict[str, Any]], bool]] = {}


def register_scorer(answer_type: str, fn: Callable[[str, dict[str, Any]], bool]) -> None:
    """Register a scorer for a non-multiple-choice ``answer_type``.

    The multiple-choice path uses ``tasks.score``; a free-form dataset registers
    its own scorer here and sets ``DatasetSpec.answer_type`` to match.
    """
    SCORERS[answer_type] = fn


if __name__ == "__main__":
    verify()
    print("MINT index-mapping verification: PASSED")
    qs = load_questions()
    print(f"Loaded {len(qs)} questions "
          f"({sum(q['category'] == 'winogrande' for q in qs)} winogrande / "
          f"{sum(q['category'] == 'ethics' for q in qs)} ethics)")
    from collections import Counter
    print("correct-letter distribution:", dict(Counter(q["correct"] for q in qs)))
    sample = qs[0]
    print("\n--- sample ---")
    print(json.dumps({k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
                      for k, v in sample.items()}, indent=2))
