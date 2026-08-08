"""
Main experiment runner.

Runs both of your architectures across the contamination conditions and
writes one CSV plus a printed summary.

Usage:
    python run_experiment.py
    python run_experiment.py --n 20 --category hoax

Start with --n 5 to confirm everything works before burning quota.
"""

import csv
import random
import argparse
import traceback

from common import get_llm, load_mint, gold_letter

import sequential_chain_team as seq
import hierarchical_team as hier


ARCHITECTURES = {
    "sequential_chain": (seq, seq.AGENTS),
    "hierarchical": (hier, hier.AGENTS),
}


def conditions_for(agents, rng):
    """
    Four conditions per Tony's spec plus a randomised choice of which
    agent gets contaminated, matching Becker et al. so that position bias
    does not confound the result.
    """
    one = [rng.choice(agents)]
    two = rng.sample(agents, 2)
    return [
        ("control", []),
        ("one_contaminated", one),
        ("multi_contaminated", two),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--subset", default="winogrande")
    ap.add_argument("--category", default="hoax")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    llm = get_llm()
    samples = load_mint(subset=args.subset, n=args.n, category=args.category)
    print(f"Loaded {len(samples)} samples from MINT/{args.subset} "
          f"[{args.category}]\n")

    rng = random.Random(0)
    rows = []

    for arch_name, (module, agents) in ARCHITECTURES.items():
        for cond_name, misinformed in conditions_for(agents, rng):
            print(f"--- {arch_name} | {cond_name} "
                  f"| misinformed={misinformed or 'none'}")

            for i, sample in enumerate(samples, 1):
                gold = gold_letter(sample)
                try:
                    result = module.run(llm, sample, misinformed)
                    answer = result["final_answer"]
                    err = ""
                except Exception as e:
                    answer, err = "", str(e)[:200]
                    traceback.print_exc()

                rows.append({
                    "id": sample["id"],
                    "architecture": arch_name,
                    "condition": cond_name,
                    "n_misinformed": len(misinformed),
                    "misinformed_agents": "|".join(misinformed),
                    "category": args.category,
                    "final_answer": answer,
                    "gold_answer": gold,
                    "correct": int(bool(answer) and answer == gold),
                    "error": err,
                })

                if i % 10 == 0:
                    print(f"    {i}/{len(samples)}")

            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}\n")
    summarise(rows)


def summarise(rows):
    print(f"{'architecture':<20}{'condition':<22}{'acc':>8}{'contam%':>10}")
    print("-" * 60)
    seen = []
    for r in rows:
        key = (r["architecture"], r["condition"])
        if key not in seen:
            seen.append(key)
    for arch, cond in seen:
        group = [r for r in rows
                 if r["architecture"] == arch and r["condition"] == cond]
        acc = sum(r["correct"] for r in group) / len(group)
        contam = 1 - acc
        print(f"{arch:<20}{cond:<22}{acc:>8.3f}{contam * 100:>9.1f}%")


if __name__ == "__main__":
    main()
