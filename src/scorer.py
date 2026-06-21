"""
scorer.py
==========
Combines structured features, semantic similarity, behavioral signals, and
honeypot detection into a single composite fit score per candidate, plus a
fact-grounded reasoning string.

Composite structure
--------------------
    base_fit = weighted_sum(
        semantic_similarity,       # topical/career-narrative match to JD meaning
        skill_group_scores,        # must-have / nice-to-have, trust-weighted
        experience_band_fit,       # soft 5-9yr band
        title_role_relevance,      # is the title/role-family even in the right universe
        location_fit,
        notice_period_fit,
    )

    base_fit *= disqualifier_multipliers(   # multiplicative, not additive --
        title_description_coherence,        # a disqualifier should be able to
        pure_services_penalty,               # crater an otherwise-good score,
        research_only_penalty,               # not just nudge it
        architecture_drift_penalty,
        excluded_domain_penalty,
        job_hopping_penalty,
    )

    final_score = base_fit * availability_multiplier   # behavioral signals

    if is_likely_honeypot: final_score is floored near zero (hard gate)

Why multiplicative disqualifiers
----------------------------------
The JD is explicit that certain things are near-hard "no"s (pure research
with zero production, pure consulting-only career, CV/speech-only without
NLP). An additive penalty risks a candidate with an otherwise enormous
skill/semantic score still landing in the top 10 despite a real
disqualifier. Multiplying by a small factor (e.g. 0.25-0.45) lets the
disqualifier dominate while still allowing the score to differentiate among
disqualified candidates (so the ranking among "rejected" candidates near
rank 100 isn't arbitrary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import feature_extractor as fx
from . import behavioral_signals as bx
from . import honeypot_detector as hp
from .jd_parser import JDRequirements

# Title-family relevance buckets, used as a coarse role-relevance gate before
# the fine-grained skill/semantic matching kicks in. This directly targets
# the keyword-stuffer trap: an "Accountant" with ten AI skill keywords still
# gets bucketed as "irrelevant_role" because the *title* -- corroborated by
# career-history substance, see title_description_coherence -- says so.
CORE_AI_TITLES = {
    "ml engineer", "ai research engineer", "senior software engineer (ml)",
    "computer vision engineer", "junior ml engineer", "ai specialist",
    "recommendation systems engineer", "machine learning engineer",
    "applied ml engineer", "search engineer", "ai engineer",
    "senior data scientist", "data scientist", "ai engineer — founding team",
}
ADJACENT_ENG_TITLES = {
    "data engineer", "senior data engineer", "analytics engineer",
    "backend engineer", "software engineer", "senior software engineer",
    "full stack developer", "cloud engineer",
}
GENERIC_SWE_TITLES = {
    "java developer", ".net developer", "devops engineer", "mobile developer",
    "frontend engineer", "qa engineer",
}


def title_relevance_bucket(title: str) -> float:
    t = title.lower().strip()
    if t in CORE_AI_TITLES:
        return 1.0
    if t in ADJACENT_ENG_TITLES:
        return 0.55
    if t in GENERIC_SWE_TITLES:
        return 0.3
    return 0.08  # business/ops/non-engineering roles (HR Manager, Accountant, etc.)


@dataclass
class ScoredCandidate:
    candidate_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    reasoning: str = ""
    is_honeypot: bool = False


def score_candidate(
    candidate: dict[str, Any],
    jd: JDRequirements,
    semantic_similarity: float,
) -> ScoredCandidate:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    cid = candidate["candidate_id"]

    # --- Honeypot hard gate -------------------------------------------------
    honeypot_flags = hp.run_checks(candidate)
    is_honeypot = len(honeypot_flags) >= 2

    # --- Structured component scores ---------------------------------------
    title = profile.get("current_title", "")
    title_bucket_score = title_relevance_bucket(title)

    skill_scores = fx.skill_group_match(candidate, jd.must_have_groups)
    must_have_avg = float(np.mean(list(skill_scores.values()))) if skill_scores else 0.0

    nice_scores = fx.skill_group_match(candidate, jd.nice_to_have_groups)
    nice_have_avg = float(np.mean(list(nice_scores.values()))) if nice_scores else 0.0

    exp_fit = fx.experience_band_fit(profile.get("years_of_experience", 0.0), jd.min_years, jd.max_years)
    loc_fit = fx.location_fit(candidate, jd.preferred_locations, jd.country_required)
    notice_fit = fx.notice_period_fit(
        signals.get("notice_period_days", 30), jd.notice_period_soft_cap_days, jd.notice_period_hard_concern_days
    )

    base_fit = (
        0.30 * semantic_similarity
        + 0.22 * title_bucket_score
        + 0.20 * must_have_avg
        + 0.06 * nice_have_avg
        + 0.11 * exp_fit
        + 0.08 * loc_fit
        + 0.03 * notice_fit
    )

    # --- Multiplicative disqualifier checks ---------------------------------
    disqualifier_mult = 1.0
    flags: list[str] = []

    coherence = fx.title_description_coherence(candidate)
    if coherence < 0.5:
        disqualifier_mult *= 0.65
        flags.append("title_description_mismatch")

    if fx.pure_services_career(candidate, jd.services_firms):
        disqualifier_mult *= 0.35
        flags.append("pure_services_only_career")

    if fx.research_only_no_production(candidate, jd.research_only_title_hints, jd.research_only_text_hints):
        disqualifier_mult *= 0.30
        flags.append("research_only_no_production")

    if fx.architecture_drift_no_recent_code(candidate, jd.non_coding_title_hints):
        disqualifier_mult *= 0.55
        flags.append("architecture_drift_stale_coding")

    if fx.excluded_primary_domain(candidate, jd.excluded_primary_domains):
        disqualifier_mult *= 0.40
        flags.append("excluded_primary_domain_cv_speech_robotics")

    # Geography / visa: JD doesn't sponsor work visas and lists India location
    # (or willingness to relocate within India) as one of five core "ideal
    # candidate" bullets. Treat a non-India location without sponsorship as
    # a meaningful, multiplicative practical hurdle rather than only a small
    # additive deduction -- an excellent skills match outside India is still
    # realistically much harder to convert into a hire than an equally good
    # match already based in (or relocatable within) India.
    if profile.get("country", "").lower() != jd.country_required:
        disqualifier_mult *= 0.55
        flags.append("outside_india_no_visa_sponsorship")

    hop_score = fx.job_hopping_score(candidate)
    if hop_score < 1.0:
        disqualifier_mult *= (0.7 + 0.3 * hop_score)  # gentle: title-chasing is a soft flag, not a hard one
        if hop_score <= 0.6:
            flags.append("frequent_job_hopping")

    fit_score = base_fit * disqualifier_mult

    # --- Behavioral signal multiplier ---------------------------------------
    avail_mult = bx.availability_multiplier(signals)
    final = fit_score * avail_mult

    # --- Honeypot floor -------------------------------------------------------
    if is_honeypot:
        final = min(final, 0.02)
        flags.append("honeypot_suspected:" + ",".join(honeypot_flags))

    components = {
        "semantic_similarity": round(semantic_similarity, 4),
        "title_bucket_score": round(title_bucket_score, 4),
        "must_have_skill_avg": round(must_have_avg, 4),
        "nice_to_have_skill_avg": round(nice_have_avg, 4),
        "experience_fit": round(exp_fit, 4),
        "location_fit": round(loc_fit, 4),
        "notice_fit": round(notice_fit, 4),
        "base_fit": round(base_fit, 4),
        "disqualifier_mult": round(disqualifier_mult, 4),
        "availability_mult": round(avail_mult, 4),
        "title_description_coherence": round(coherence, 4),
    }

    return ScoredCandidate(
        candidate_id=cid,
        score=final,
        components=components,
        flags=flags,
        is_honeypot=is_honeypot,
    )
