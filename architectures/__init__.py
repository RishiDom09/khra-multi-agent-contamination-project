"""Auto-discovery of architecture plug-ins — and THE CONTRACT they must satisfy.

Every ``architectures/<name>.py`` that satisfies the contract below is
discovered here and becomes runnable as
``python run_experiment.py --architectures <name>`` — no registration code,
the file IS the registration.

THE CONTRACT — your module must export:

  run(question, contamination) -> {"final_answer": str, "transcript": dict}
  INFLUENCE_ORDER: list[str]   seats by structural influence; contamination is
                               seeded from the front (justify your ordering)
  ANSWERING_AGENTS: list[str]  the propagation denominator (exclude non-voting
                               seats like a critic)

Hard rules: build every agent with the shared ``agent.Agent`` class (never
write your own prompts or call a model directly); contamination goes only to
``contamination["seats"]`` via ``false_belief``; record EVERY model call in the
transcript's ``rounds``.

Required transcript shape (what ``analysis.py`` scores)::

    {
      "architecture":     "<your module name>",
      "question_id":      question["id"],
      "seeded_seats":     sorted(seeded),
      "answering_agents": list(ANSWERING_AGENTS),
      "rounds":           [[rec, ...], ...],  # chronological; LAST round = one
                                              # final record per answering
                                              # agent (non-voting seats go in
                                              # earlier rounds)
      "final_answer":     "A" | "B" | "",     # "" = abstain
    }

Each ``rec`` is exactly what ``Agent.respond()`` returns — append it untouched.
Working examples: ``baseline.py`` (smallest), ``dynamic_team.py`` (LangGraph),
``adversarial_team.py`` (fixed pipeline with a non-voting critic). Validate
offline with ``KHRA_MOCK=1 python tests/test_contract.py``.

Discovery rules:
  * modules starting with ``_`` are skipped (reserved for helpers);
  * modules with a truthy module-level ``INCOMPLETE`` are skipped but listed in
    :data:`PENDING` — that is how a teammate's stub sits in the repo without
    breaking runs. Delete the ``INCOMPLETE = True`` line to activate the file;
  * everything else MUST export ``run``, ``INFLUENCE_ORDER``, and
    ``ANSWERING_AGENTS`` — a missing export fails loudly here rather than
    silently mid-run.

``REGISTRY`` maps ``name -> (run, INFLUENCE_ORDER)``, the tuple shape
``run_experiment.SETUPS`` has always used.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

REGISTRY: dict[str, tuple[Callable[..., dict[str, Any]], list[str]]] = {}
PENDING: dict[str, str] = {}  # name -> owner/TODO note, for CLI help text

_REQUIRED = ("run", "INFLUENCE_ORDER", "ANSWERING_AGENTS")

for _info in pkgutil.iter_modules(__path__):
    _name = _info.name
    if _name.startswith("_"):
        continue
    _mod = importlib.import_module(f"{__name__}.{_name}")
    if getattr(_mod, "INCOMPLETE", False):
        PENDING[_name] = getattr(_mod, "OWNER", "unassigned")
        continue
    _missing = [a for a in _REQUIRED if not hasattr(_mod, a)]
    if _missing:
        raise ImportError(
            f"architectures/{_name}.py does not satisfy the contract: missing "
            f"{', '.join(_missing)}. See the contract in architectures/__init__.py. "
            f"(If the file is still under construction, add "
            f"'INCOMPLETE = True' at module level to park it.)"
        )
    REGISTRY[_name] = (_mod.run, list(_mod.INFLUENCE_ORDER))

# Keep the propagation denominator discoverable per architecture too.
ANSWERING: dict[str, list[str]] = {
    name: list(importlib.import_module(f"{__name__}.{name}").ANSWERING_AGENTS)
    for name in REGISTRY
}

# Short display/folder labels per architecture. Plug-ins not listed here fall
# back to a truncated module name, so adding an architecture never requires
# touching this file — add an entry only if you want a nicer label.
ARCH_SHORT: dict[str, str] = {
    "baseline": "base",
    "dynamic_team": "dyn",
    "adversarial_team": "adv",
    "hierarchical": "hier",
    "sequential_chain": "seq",
    "hub_and_spoke": "hub",
    "shared_workspace": "shared",
    "redundant": "redun",
    "debate": "debate",
}


def arch_tag(names: list[str]) -> str:
    """Folder-name tag naming the multi-agent system(s) a run covers.

    E.g. ``['hierarchical']`` -> ``'hier'`` and the default three-architecture
    sweep -> ``'adv-base-dyn'``. Sorted so the same selection always maps to
    the same run folder — that is what lets an interrupted run resume.
    """
    return "-".join(ARCH_SHORT.get(n, n[:6]) for n in sorted(names))
