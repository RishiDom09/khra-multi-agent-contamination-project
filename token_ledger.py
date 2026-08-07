"""Cross-run token accounting against the provider's rolling daily cap.

The budget guard in ``run_experiment`` used to reason about a counter that was
reset to zero at the start of every process, so it assumed a full daily budget
was available even when a previous run had already spent most of it. That is how
the adversarial run of 2026-07-26 died on a provider 429 (Groq reported
``Used 199551`` of 200000) while its own guard still believed it had headroom.

This module keeps a small append-only ledger on disk so successive runs share one
view of what has been spent inside the provider's window.

Why a ledger rather than response headers: ``langchain_groq`` calls the Groq
SDK's ``client.create()`` and returns the parsed message, discarding the HTTP
response and therefore the ``x-ratelimit-remaining-tokens`` header. Rather than
reach around LangChain into the raw client, we track usage ourselves and
*calibrate* against ground truth whenever the provider does tell us — a 429 body
carries an exact ``Limit``/``Used`` pair.

The window is rolling, not calendar-day: observed usage of 228,506 tokens inside
one local calendar day against a 200,000 cap proves the provider does not reset
at local midnight.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import config

LEDGER_FILE = "token_usage.jsonl"


def _path() -> str:
    # Root, not the run folder: the ledger tracks the PROVIDER's rolling window,
    # which spans every run made against the same API key.
    return os.path.join(config.RESULTS_ROOT, LEDGER_FILE)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def record(tokens: int, model: Optional[str] = None, note: str = "") -> None:
    """Append a usage record. Cheap enough to call after every question."""
    if tokens <= 0:
        return
    os.makedirs(config.RESULTS_ROOT, exist_ok=True)
    entry = {
        "ts": time.time(),
        "tokens": int(tokens),
        "model": model or config.MODEL_ID,
        "note": note,
    }
    with open(_path(), "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _read() -> list[dict[str, Any]]:
    path = _path()
    if not os.path.exists(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line must not break accounting
    return out


# --------------------------------------------------------------------------- #
# Querying
# --------------------------------------------------------------------------- #
def used_in_window(
    window_hours: float = config.TOKEN_WINDOW_HOURS,
    model: Optional[str] = None,
) -> int:
    """Tokens recorded for ``model`` within the last ``window_hours``.

    A calibration record (written from a provider 429) supersedes everything at
    or before its timestamp, since the provider's own count is authoritative.
    """
    model = model or config.MODEL_ID
    cutoff = time.time() - window_hours * 3600
    rows = [r for r in _read() if r.get("model") == model]

    # Only a calibration INSIDE the window is meaningful: an older one describes
    # usage the provider has since aged out, and applying it would permanently
    # suppress the budget. Treating it as a hard cliff (rather than the
    # provider's gradual expiry) makes this estimate conservative near the
    # boundary — it errs toward not spending, which is the safe direction.
    calibrated_at, calibrated_used = 0.0, 0
    for r in rows:
        if (
            r.get("note") == "calibration"
            and r["ts"] >= cutoff
            and r["ts"] >= calibrated_at
        ):
            calibrated_at, calibrated_used = r["ts"], int(r["tokens"])

    total = calibrated_used if calibrated_at else 0
    for r in rows:
        if r.get("note") == "calibration":
            continue
        if r["ts"] <= calibrated_at:
            continue  # already included in the calibrated figure
        if r["ts"] < cutoff:
            continue  # aged out of the rolling window
        total += int(r["tokens"])
    return total


def remaining(
    cap: Optional[int] = None,
    window_hours: float = config.TOKEN_WINDOW_HOURS,
    model: Optional[str] = None,
) -> int:
    """Best estimate of tokens still available before the provider cap."""
    cap = cap if cap is not None else config.DAILY_TOKEN_CAP
    return max(0, cap - used_in_window(window_hours, model))


# --------------------------------------------------------------------------- #
# Calibration from a provider 429
# --------------------------------------------------------------------------- #
_LIMIT_USED = re.compile(
    r"Limit\s+(\d+)\s*,\s*Used\s+(\d+)", re.IGNORECASE
)


def parse_rate_limit(message: str) -> Optional[tuple[int, int]]:
    """Extract ``(limit, used)`` from a provider rate-limit message, if present."""
    m = _LIMIT_USED.search(message or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def calibrate(message: str, model: Optional[str] = None) -> Optional[tuple[int, int]]:
    """Record the provider's authoritative usage figure from a 429 body.

    Returns ``(limit, used)`` when the message carried one. Subsequent
    :func:`used_in_window` calls treat this as the new baseline, which corrects
    any drift between our per-call accounting and the provider's.
    """
    parsed = parse_rate_limit(message)
    if parsed is None:
        return None
    limit, used = parsed
    os.makedirs(config.RESULTS_ROOT, exist_ok=True)
    entry = {
        "ts": time.time(),
        "tokens": used,
        "model": model or config.MODEL_ID,
        "note": "calibration",
        "limit": limit,
    }
    with open(_path(), "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return limit, used


def summary(model: Optional[str] = None) -> str:
    """One-line human-readable state, for the run header."""
    used = used_in_window(model=model)
    left = remaining(model=model)
    return (
        f"{used:,} used / {config.DAILY_TOKEN_CAP:,} cap in the last "
        f"{config.TOKEN_WINDOW_HOURS:g}h -> ~{left:,} remaining"
    )


if __name__ == "__main__":
    print(summary())
    for r in _read()[-15:]:
        stamp = time.strftime("%m-%d %H:%M:%S", time.localtime(r["ts"]))
        tag = f" [{r['note']}]" if r.get("note") else ""
        print(f"  {stamp}  {r['tokens']:>8,}  {r['model']}{tag}")
