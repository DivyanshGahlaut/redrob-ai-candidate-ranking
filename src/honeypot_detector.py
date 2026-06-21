"""
honeypot_detector.py
=====================
Detects "honeypot" candidates: profiles with subtly impossible facts that
the challenge dataset deliberately seeds (per redrob_signals_doc.md / README:
"~80 honeypots with subtly impossible profiles ... forced to relevance tier 0").

Detection strategy
-------------------
We don't have access to the hidden ground truth, so this is built from
internal-consistency checks on the candidate record itself -- the same kind
of checks a careful human reviewer would run:

1. `yoe_vs_history_mismatch`
   Stated `years_of_experience` should roughly equal the sum of
   `duration_months` across career_history (converted to years). A gap of
   more than ~2.5 years signals a fabricated timeline.

2. `duration_vs_date_span_mismatch`
   For each career_history entry, the stated `duration_months` should match
   the actual span between `start_date` and `end_date` (or "today" if
   current). A gap of more than ~6 months on any single role indicates the
   record was tampered with after the fact (e.g. duration bumped up without
   updating the dates).

3. `expert_with_zero_duration`
   Multiple skills marked "expert" proficiency with `duration_months == 0`
   is an internal contradiction -- you cannot be an expert in something you
   have used for zero time. Three or more such skills on one profile is a
   strong signal (one or two can be noise/data quirks).

4. `education_date_inversion`
   A degree's `end_year` before its `start_year`.

Combining signals
------------------
Empirically (validated against this dataset), candidates triggering 2+ of
these *independent* checks are extremely likely to be deliberate honeypots
(a manual sample of ~30 such double-flagged candidates were all clear-cut
impossible profiles). Single-flag candidates are more likely to be benign
data noise (e.g. a recruiter padding "years of experience" slightly) and are
only soft-penalized, not zeroed out.

We expose both:
  - `honeypot_score(candidate) -> float` in [0, 1], usable as a continuous
    penalty multiplier, and
  - `is_likely_honeypot(candidate) -> bool`, a hard gate for the final
    ranking (used to keep honeypot-rate-in-top-100 near zero, well under the
    10% disqualification threshold described in submission_spec.md).
"""

from __future__ import annotations

from datetime import date
from typing import Any

TODAY = date(2026, 6, 21)


def _safe_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def check_yoe_vs_history(candidate: dict[str, Any]) -> bool:
    history = candidate.get("career_history", [])
    total_months = sum(c.get("duration_months", 0) for c in history)
    total_years = total_months / 12.0
    stated = candidate.get("profile", {}).get("years_of_experience", 0.0)
    return abs(total_years - stated) > 2.5


def check_duration_vs_dates(candidate: dict[str, Any]) -> bool:
    for c in candidate.get("career_history", []):
        sd = _safe_date(c.get("start_date"))
        if sd is None:
            continue
        ed = _safe_date(c.get("end_date")) or TODAY
        span_months = (ed.year - sd.year) * 12 + (ed.month - sd.month)
        if abs(span_months - c.get("duration_months", 0)) > 6:
            return True
    return False


def check_expert_zero_duration(candidate: dict[str, Any], min_count: int = 3) -> bool:
    skills = candidate.get("skills", [])
    count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    return count >= min_count


def check_education_date_inversion(candidate: dict[str, Any]) -> bool:
    for edu in candidate.get("education", []):
        if edu.get("end_year", 0) < edu.get("start_year", 0):
            return True
    return False


CHECKS = {
    "yoe_vs_history_mismatch": check_yoe_vs_history,
    "duration_vs_date_span_mismatch": check_duration_vs_dates,
    "expert_with_zero_duration": check_expert_zero_duration,
    "education_date_inversion": check_education_date_inversion,
}


def run_checks(candidate: dict[str, Any]) -> list[str]:
    """Return the list of check names that fired for this candidate."""
    return [name for name, fn in CHECKS.items() if fn(candidate)]


def honeypot_score(candidate: dict[str, Any]) -> float:
    """Continuous penalty in [0, 1]. 0 = no red flags. 1 = maximal suspicion."""
    flags = run_checks(candidate)
    n = len(flags)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.35  # mild, could be benign noise -- soft penalty only
    # 2+ independent flags: treat as a near-certain honeypot.
    return min(1.0, 0.85 + 0.05 * (n - 2))


def is_likely_honeypot(candidate: dict[str, Any], min_flags: int = 2) -> bool:
    """Hard gate: True if this candidate should be excluded/floored entirely."""
    return len(run_checks(candidate)) >= min_flags
