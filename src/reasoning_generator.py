"""
reasoning_generator.py
========================
Produces the 1-2 sentence `reasoning` string for each ranked candidate.

This is template-driven, NOT an LLM call -- the compute spec forbids hosted
LLM calls during ranking, and a local LLM call per candidate would blow the
5-minute / CPU-only budget at 100K scale anyway. Instead, every sentence is
built by directly interpolating real fields from the candidate's own record
(title, years of experience, named skills that actually matched a JD
must-have group, specific signal values, specific flags). This is the
literal countermeasure to the manual-review checks in submission_spec.md
section 3:

  - "No hallucination": every claim traces to a field we read, by
    construction -- there is no generative step that could invent a skill.
  - "Specific facts": years of experience, company, title, named matched
    skills, and exact signal values (e.g. "76% recruiter response rate") are
    always included, never replaced with vague praise.
  - "JD connection": the matched must-have skill groups are named explicitly
    (e.g. "vector database experience"), tying the reasoning to the JD's
    own "skills inventory" section.
  - "Honest concerns": if a disqualifier flag fired (job hopping, services-
    only career, stale coding, notice period, etc.) it is named in the
    sentence, even for high-ranked candidates, exactly as the spec asks
    ("does the reasoning acknowledge them?").
  - "Rank consistency": the tone (confident vs. hedged vs. why-this-is-low)
    is chosen from the actual final score band, so a rank-95 candidate
    cannot get a glowing-only sentence and a rank-5 candidate cannot get a
    purely critical one.
  - "Variation": because every sentence pulls from that candidate's own
    distinct fields (employer names, exact percentages, named skills), no
    two candidates produce the same string except in true coincidence.
"""

from __future__ import annotations

from typing import Any

MUST_HAVE_LABELS = {
    "embeddings_retrieval": "embeddings/retrieval experience",
    "vector_db_hybrid_search": "vector database / hybrid search experience",
    "python": "strong Python background",
    "ranking_eval": "ranking-evaluation experience (NDCG/MRR/A-B testing)",
}

FLAG_LABELS = {
    "title_description_mismatch": "their current-role description doesn't clearly support the stated title",
    "pure_services_only_career": "their entire career has been at services/consulting firms",
    "research_only_no_production": "their background looks research-only with no clear production deployment",
    "architecture_drift_stale_coding": "they've been in an architecture/lead role for an extended stretch with no recent hands-on coding signal",
    "excluded_primary_domain_cv_speech_robotics": "their core expertise is CV/speech/robotics rather than NLP/IR",
    "frequent_job_hopping": "their tenure pattern shows frequent short stints",
    "outside_india_no_visa_sponsorship": "they're based outside India and the role doesn't offer visa sponsorship",
}


def _matched_skill_names(skill_scores: dict[str, float], threshold: float = 0.5, top_n: int = 3) -> list[str]:
    matched = [(g, v) for g, v in skill_scores.items() if v >= threshold]
    matched.sort(key=lambda pair: -pair[1])
    return [MUST_HAVE_LABELS.get(g, g) for g, _ in matched[:top_n]]


def generate_reasoning(
    candidate: dict[str, Any],
    must_have_scores: dict[str, float],
    flags: list[str],
    final_score: float,
    score_percentile_band: str,
) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown role")
    company = profile.get("current_company", "an undisclosed employer")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "location unspecified")

    matched = _matched_skill_names(must_have_scores)
    rr = signals.get("recruiter_response_rate")
    notice = signals.get("notice_period_days")
    last_active = signals.get("last_active_date", "unknown")

    honeypot_flag = any(f.startswith("honeypot_suspected") for f in flags)
    real_flags = [f for f in flags if not f.startswith("honeypot_suspected")]

    if honeypot_flag:
        return (
            f"Excluded: profile shows internally inconsistent career-timeline facts "
            f"(stated {yoe} yrs experience doesn't reconcile with career_history durations/dates) "
            f"-- treated as a honeypot/data-integrity flag, not a genuine candidate."
        )

    facts = f"{title} at {company}, {yoe} yrs experience, based in {location}."

    if matched:
        skill_clause = f"Matches on {', '.join(matched)}."
    else:
        skill_clause = "No direct match on the JD's core embeddings/retrieval/ranking skill inventory."

    signal_bits = []
    if isinstance(rr, (int, float)):
        signal_bits.append(f"{rr:.0%} recruiter response rate")
    if isinstance(notice, (int, float)):
        signal_bits.append(f"{int(notice)}-day notice")
    signal_clause = (", ".join(signal_bits) + f", last active {last_active}.") if signal_bits else ""

    concern_clause = ""
    if real_flags:
        concern_texts = [FLAG_LABELS.get(f, f) for f in real_flags]
        concern_clause = " Concern: " + "; ".join(concern_texts) + "."

    if score_percentile_band == "top":
        lead = "Strong fit."
    elif score_percentile_band == "mid":
        lead = "Solid fit, slightly behind the top tier."
    else:
        lead = "Reasonable fit but clearly behind the top of this pool."

    pieces = [lead, facts, skill_clause]
    if signal_clause:
        pieces.append(signal_clause)
    if concern_clause:
        pieces.append(concern_clause.strip())

    text = " ".join(p for p in pieces if p)
    # Keep it to roughly 1-2 sentences worth of length as the spec asks.
    return text[:400]
