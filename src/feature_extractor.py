"""
feature_extractor.py
======================
Turns a raw candidate JSON record into the structured features the scorer
needs. Each function here is deliberately small and named after the exact
JD requirement it answers, so the scoring logic in scorer.py reads like a
checklist a human recruiter would tick through -- and so each one is
independently testable and explainable in the "reasoning" column / the
defend-your-work interview.
"""

from __future__ import annotations

from datetime import date
from typing import Any

TODAY = date(2026, 6, 21)


# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------

def candidate_full_text(candidate: dict[str, Any]) -> str:
    """Concatenate every free-text field into one string for TF-IDF embedding.

    Career-history descriptions carry the most signal (per the JD's own
    framing: "the gap between what the JD says and what the JD means" is
    read from career substance, not the skills list), so they're included
    in full. Skill names are added as light context.
    """
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    for c in candidate.get("career_history", []):
        parts.append(c.get("title", ""))
        parts.append(c.get("description", ""))
    skill_names = [s.get("name", "") for s in candidate.get("skills", [])]
    parts.append(" ".join(skill_names))
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Title / role-family coherence
# ---------------------------------------------------------------------------

# Crude lexical fingerprints for common titles in this dataset, used to
# detect the "title says X, description says Y" shuffle trap identified
# during data exploration (~10% of records have a current-role description
# that doesn't match the stated title at all).
TITLE_FINGERPRINTS: dict[str, list[str]] = {
    "mobile developer": ["mobile", "ios", "android", "flutter", "react native", "kotlin", "swift"],
    "devops engineer": ["devops", "ci/cd", "kubernetes", "terraform", "infra", "infrastructure", "docker"],
    ".net developer": [".net", "c#", "asp.net"],
    "qa engineer": ["qa", "test automation", "selenium", "quality", "testing"],
    "frontend engineer": ["frontend", "react", "css", "ui", "angular", "vue"],
    "java developer": ["java", "spring"],
    "backend engineer": [
        "backend", "api", "microservice", "database", "server",
        # Backend engineers legitimately drift into the ML/ranking side of
        # their product (this is exactly the "Tier 5, no jargon" profile the
        # JD wants surfaced) -- don't flag that as incoherent.
        "recommendation", "ranking", "model", "pipeline", "data",
    ],
    "full stack developer": ["full stack", "fullstack", "frontend", "backend"],
    "data engineer": ["data pipeline", "etl", "spark", "airflow", "warehouse"],
    "cloud engineer": ["cloud", "aws", "azure", "gcp", "kubernetes"],
    "ml engineer": ["machine learning", "ml model", "model", "training", "inference"],
    "data scientist": ["data scien", "model", "statistic", "analysis", "predictive"],
}


def title_description_coherence(candidate: dict[str, Any]) -> float:
    """Returns 1.0 if the current role's description plausibly matches its
    title, 0.3 if it clearly doesn't (the shuffle trap), 0.7 if the title
    isn't in our fingerprint dict (unknown -- don't penalize blindly).
    """
    history = candidate.get("career_history", [])
    current = next((c for c in history if c.get("is_current")), history[0] if history else None)
    if not current:
        return 0.7

    title = current.get("title", "").lower().strip()
    desc = current.get("description", "").lower()

    if title not in TITLE_FINGERPRINTS:
        return 0.7

    keywords = TITLE_FINGERPRINTS[title]
    return 1.0 if any(k in desc for k in keywords) else 0.3


# ---------------------------------------------------------------------------
# Experience band fit (soft, per JD's own instruction)
# ---------------------------------------------------------------------------

def experience_band_fit(years: float, min_years: float, max_years: float) -> float:
    """Soft band scoring: full credit inside [min, max], decaying outside.

    JD explicitly says "This is a range, not a requirement" -- so we use a
    smooth falloff rather than a hard cutoff. Candidates well outside the
    band (e.g. 1 year, or 20 years with no other strong signal) still score
    low, but a 4.5-year or 9.5-year candidate isn't zeroed out.
    """
    if min_years <= years <= max_years:
        return 1.0
    if years < min_years:
        gap = min_years - years
        return max(0.0, 1.0 - 0.30 * gap)
    gap = years - max_years
    return max(0.0, 1.0 - 0.12 * gap)


# ---------------------------------------------------------------------------
# Company-type / career-pattern signals
# ---------------------------------------------------------------------------

def pure_services_career(candidate: dict[str, Any], services_firms: set[str]) -> bool:
    companies = {c.get("company", "") for c in candidate.get("career_history", [])}
    companies_lower = {c.lower() for c in companies}
    return bool(companies_lower) and companies_lower.issubset({s.lower() for s in services_firms})


def currently_at_services_with_product_history(candidate: dict[str, Any], services_firms: set[str]) -> bool:
    """JD: 'If you're currently at one of these companies but have prior
    product-company experience, that's fine.' -- distinguishes this case
    from the pure-services disqualifier.
    """
    history = candidate.get("career_history", [])
    if not history:
        return False
    current = next((c for c in history if c.get("is_current")), history[0])
    is_current_services = current.get("company", "").lower() in {s.lower() for s in services_firms}
    if not is_current_services:
        return False
    other_companies = {c.get("company", "").lower() for c in history if c is not current}
    has_non_services_history = bool(other_companies - {s.lower() for s in services_firms})
    return has_non_services_history


def job_hopping_score(candidate: dict[str, Any]) -> float:
    """Detects the JD's 'title-chaser' pattern: average tenure well under
    ~1.5 years across a multi-role career. Returns a penalty multiplier in
    (0, 1] -- 1.0 means no concern, lower means more hopping.
    """
    history = candidate.get("career_history", [])
    if len(history) < 3:
        return 1.0  # not enough roles to call a pattern
    durations = [c.get("duration_months", 0) for c in history]
    avg_months = sum(durations) / len(durations)
    if avg_months >= 24:
        return 1.0
    if avg_months <= 12:
        return 0.55
    # linear interpolation between 12 and 24 months
    return 0.55 + 0.45 * (avg_months - 12) / 12


def research_only_no_production(candidate: dict[str, Any], title_hints: list[str], text_hints: list[str]) -> bool:
    history = candidate.get("career_history", [])
    if not history:
        return False
    titles = " ".join(c.get("title", "").lower() for c in history)
    descs = " ".join(c.get("description", "").lower() for c in history)
    summary = candidate.get("profile", {}).get("summary", "").lower()
    has_research_title = any(h in titles for h in title_hints)
    has_research_text = any(h in (descs + summary) for h in text_hints)
    has_production_signal = any(
        kw in descs for kw in ["production", "deployed", "shipped", "real users", "scale", "real-time"]
    )
    return (has_research_title or has_research_text) and not has_production_signal


def architecture_drift_no_recent_code(candidate: dict[str, Any], title_hints: list[str]) -> bool:
    """JD: senior engineer who hasn't written production code in 18+ months
    because they moved into 'architecture'/'tech lead' roles.
    """
    history = candidate.get("career_history", [])
    current = next((c for c in history if c.get("is_current")), history[0] if history else None)
    if not current:
        return False
    title = current.get("title", "").lower()
    if not any(h in title for h in title_hints):
        return False
    start = current.get("start_date")
    try:
        sd = date.fromisoformat(start)
    except (ValueError, TypeError):
        return False
    months_in_role = (TODAY.year - sd.year) * 12 + (TODAY.month - sd.month)
    return months_in_role >= 18


def excluded_primary_domain(candidate: dict[str, Any], excluded_domains: list[str]) -> bool:
    """JD: CV/speech/robotics-primary candidates without NLP/IR exposure
    are explicitly not a fit.
    """
    full_text = candidate_full_text(candidate).lower()
    has_excluded = any(d in full_text for d in excluded_domains)
    has_nlp_ir = any(
        kw in full_text for kw in [
            "nlp", "natural language", "information retrieval", "retrieval",
            "search ranking", "embeddings", "text classification",
        ]
    )
    return has_excluded and not has_nlp_ir


# ---------------------------------------------------------------------------
# Skills: must-have / nice-to-have structured matching, trust-weighted
# ---------------------------------------------------------------------------

def _skill_trust_weight(skill: dict[str, Any]) -> float:
    """A skill claim is more credible the longer it's been used and the more
    it's been endorsed. This is the direct countermeasure to the
    keyword-stuffer trap: a skill listed with 0 months' duration and 0
    endorsements is worth far less than one backed by 30+ months and
    endorsements, even if the skill *name* is identical.

    Critically, duration is a multiplicative gate, not just an additive
    modifier: a self-declared "expert" with zero months of actual use and
    zero endorsements is the textbook keyword-stuffer signature (an
    "expert" who has never used the skill is not credible at any
    proficiency level) and must score near zero, not get a flat 50% floor
    just for claiming the highest proficiency label.
    """
    duration = skill.get("duration_months", 0) or 0
    endorsements = skill.get("endorsements", 0) or 0
    proficiency = skill.get("proficiency", "beginner")
    prof_weight = {"beginner": 0.4, "intermediate": 0.65, "advanced": 0.85, "expert": 1.0}.get(proficiency, 0.5)

    duration_weight = min(1.0, duration / 18.0)  # full credit at 18+ months
    endorsement_weight = min(1.0, endorsements / 10.0)  # full credit at 10+ endorsements

    # Track-record term: zero duration AND zero endorsements collapses this
    # to (near) zero, regardless of proficiency. Any real usage time or
    # social corroboration (endorsements) lifts it back up.
    track_record = max(duration_weight, endorsement_weight * 0.6)
    if track_record <= 0.0:
        return 0.03  # negligible, but not a hard absolute zero (data noise tolerance)

    # Duration/endorsement track record gates the proficiency claim rather
    # than just discounting it -- this is what actually stops "expert,
    # 0 months" from getting meaningful credit.
    return prof_weight * (0.15 + 0.85 * track_record)


def _career_narrative_text(candidate: dict[str, Any]) -> str:
    """Career-history descriptions + summary only -- explicitly excludes the
    skills list. Used for the 'demonstrated in narrative, not just listed'
    fallback credit in skill_group_match, so a stuffed skills list can't
    inflate this score by definition (the skill *names* aren't in this text).
    """
    profile = candidate.get("profile", {})
    parts = [profile.get("summary", "")]
    for c in candidate.get("career_history", []):
        parts.append(c.get("description", ""))
    return " ".join(p for p in parts if p)


def skill_group_match(candidate: dict[str, Any], skill_groups: dict[str, list[str]]) -> dict[str, float]:
    """For each concept group (e.g. 'embeddings_retrieval'), return the best
    trust-weighted match score in [0, 1] found among the candidate's skills
    or career-history narrative text.
    """
    skills = candidate.get("skills", [])
    narrative_text = _career_narrative_text(candidate).lower()

    results: dict[str, float] = {}
    for group, keywords in skill_groups.items():
        best = 0.0
        for skill in skills:
            name = skill.get("name", "").lower()
            if any(kw in name for kw in keywords):
                best = max(best, _skill_trust_weight(skill))
        # Even without an exact skill-list entry, mentioning the concept in
        # career-history description/summary narrative (i.e. *demonstrated*,
        # not just *listed* as a skill) counts for a meaningful chunk of
        # credit. Deliberately excludes the skills list itself.
        if any(kw in narrative_text for kw in keywords):
            best = max(best, 0.6)
        results[group] = best
    return results


# ---------------------------------------------------------------------------
# Location / relocation fit
# ---------------------------------------------------------------------------

def location_fit(candidate: dict[str, Any], preferred_locations: list[str], country_required: str) -> float:
    profile = candidate.get("profile", {})
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    signals = candidate.get("redrob_signals", {})
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if country != country_required:
        # JD: "Outside India: case-by-case, but we don't sponsor work visas."
        # This is a real practical hurdle (visa), not just a mild preference,
        # so the penalty here is substantial even though willing_to_relocate
        # may be true -- relocating without sponsorship is a much bigger ask
        # than relocating within India.
        return 0.08

    if any(loc in location for loc in preferred_locations):
        return 1.0

    # In India, not in a preferred city, but open to relocating.
    if willing_to_relocate:
        return 0.75

    return 0.45


# ---------------------------------------------------------------------------
# Notice period fit
# ---------------------------------------------------------------------------

def notice_period_fit(notice_days: int, soft_cap: int, hard_concern: int) -> float:
    if notice_days <= soft_cap:
        return 1.0
    if notice_days <= hard_concern:
        # linear falloff between soft_cap and hard_concern
        span = max(1, hard_concern - soft_cap)
        return 1.0 - 0.3 * (notice_days - soft_cap) / span
    # Beyond hard_concern: still "in scope" per JD ("higher bar"), steeper falloff
    span = max(1, 180 - hard_concern)
    extra = min(notice_days - hard_concern, span)
    return max(0.35, 0.7 - 0.35 * extra / span)
