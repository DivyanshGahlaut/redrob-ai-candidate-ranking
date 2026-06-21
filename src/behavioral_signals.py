"""
behavioral_signals.py
=======================
Converts the 23 `redrob_signals` fields into a single "availability /
engageability" multiplier in (0, 1], applied on top of the fit score.

Rationale (straight from job_description.md's closing note):
    "a perfect-on-paper candidate who hasn't logged in for 6 months and has
    a 5% recruiter response rate is, for hiring purposes, not actually
    available. Down-weight them appropriately."

This is deliberately a *multiplier*, not an additive bonus: a candidate with
a great fit score but who is clearly unreachable/unavailable should still
rank below an equally-good candidate who is actually reachable, but a
behavioral-signal multiplier should never be able to rescue a poor fit score
into a top rank. Multiplication enforces that asymmetry naturally.
"""

from __future__ import annotations

from datetime import date
from typing import Any

TODAY = date(2026, 6, 21)


def _recency_weight(last_active_date: str | None, half_life_days: int = 45) -> float:
    """Exponential decay: a candidate active today scores 1.0; activity
    `half_life_days` ago scores 0.5, decaying further from there.
    """
    if not last_active_date:
        return 0.3
    try:
        d = date.fromisoformat(last_active_date)
    except ValueError:
        return 0.3
    days_ago = max(0, (TODAY - d).days)
    return 0.5 ** (days_ago / half_life_days)


def availability_multiplier(signals: dict[str, Any]) -> float:
    """Returns a multiplier in roughly [0.35, 1.05]."""
    if not signals:
        return 0.7  # missing signals block entirely -- mild, non-punitive default

    recency = _recency_weight(signals.get("last_active_date"))
    response_rate = float(signals.get("recruiter_response_rate", 0.0) or 0.0)
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.55
    interview_completion = float(signals.get("interview_completion_rate", 0.5) or 0.5)

    offer_accept = signals.get("offer_acceptance_rate", -1)
    # -1 means "no prior offer history" -- neutral, not negative.
    offer_accept_term = 0.75 if offer_accept == -1 else float(offer_accept)

    # Weighted blend; recency and response rate matter most because they're
    # the most direct evidence of "can we actually reach this person".
    raw = (
        0.35 * recency
        + 0.30 * response_rate
        + 0.15 * open_to_work
        + 0.10 * interview_completion
        + 0.10 * offer_accept_term
    )

    # Rescale from [0,1] raw blend into a multiplier band that can mildly
    # boost (engaged, responsive candidates) or meaningfully discount
    # (unreachable candidates) the underlying fit score, without ever letting
    # behavioral signal alone overturn a strong fit-score gap.
    return 0.4 + 0.65 * raw


def availability_explanation(signals: dict[str, Any]) -> str:
    """Short human-readable fact for the reasoning column."""
    if not signals:
        return "no platform activity data"
    rr = signals.get("recruiter_response_rate")
    last_active = signals.get("last_active_date", "unknown")
    otw = "open to work" if signals.get("open_to_work_flag") else "not flagged open-to-work"
    rr_str = f"{rr:.0%} recruiter response rate" if isinstance(rr, (int, float)) else "no response-rate data"
    return f"{rr_str}, last active {last_active}, {otw}"
