"""
Run this FIRST, before anything else.

It downloads MINT and prints the real column names and one example row.
If load_mint() later complains it could not map fields, paste this output
to Claude and the loader gets fixed in one place.
"""

from common import inspect_mint

for subset in ["winogrande", None]:
    print("=" * 70)
    print(f"SUBSET: {subset}")
    print("=" * 70)
    try:
        inspect_mint(subset)
        break
    except Exception as e:
        print(f"failed: {e}\n")
