"""
test_scorer_end_to_end.py
===========================
The single most important behavioral test for this whole system: does it
actually pass the JD's own stated acceptance criterion?

    "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
    profile, but if their career history shows they built a recommendation
    system at a product company, they're a fit. A candidate who has all the
    AI keywords listed as skills but whose title is 'Marketing Manager' is
    not a fit, no matter how perfect their skill list looks."

This test constructs exactly those two archetypes synthetically (mirroring
the patterns found during real dataset exploration) and asserts the
keyword-stuffer scores lower than the substance-but-no-jargon candidate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scorer import score_candidate
from src.jd_parser import JDRequirements


def make_jd():
    return JDRequirements(
        semantic_text=(
            "own the intelligence layer of the product: ranking, retrieval, and matching "
            "systems. ship a v2 ranking system involving embeddings, hybrid retrieval, and "
            "LLM-based re-ranking. set up evaluation infrastructure -- offline benchmarks, "
            "online A/B testing. production experience with embeddings-based retrieval and "
            "vector databases or hybrid search infrastructure. strong python. hands-on "
            "experience designing evaluation frameworks for ranking systems -- NDCG, MRR, MAP."
        )
    )


def keyword_stuffer_candidate():
    """Marketing Manager with a skill list stuffed with AI keywords but zero
    relevant career substance -- the exact trap pattern found at scale
    (~5,200 candidates) during dataset exploration.
    """
    return {
        "candidate_id": "CAND_9999001",
        "profile": {
            "current_title": "Marketing Manager",
            "current_company": "Acme Corp",
            "years_of_experience": 7.0,
            "location": "Bangalore, Karnataka",
            "country": "India",
            "summary": "Marketing Manager with 7 years of experience driving outcomes in my domain. "
                       "Recently I've been excited about how AI and GenAI tools can augment marketing workflows.",
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Marketing Manager",
                "start_date": "2019-01-01",
                "end_date": None,
                "duration_months": 89,
                "is_current": True,
                "description": "Led brand campaigns, managed the marketing budget, and ran customer "
                                "segmentation analysis for quarterly campaigns.",
            }
        ],
        "education": [{"institution": "X", "degree": "MBA", "field_of_study": "Marketing", "start_year": 2015, "end_year": 2017}],
        "skills": [
            {"name": "RAG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "LangChain", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Fine-tuning LLMs", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.5,
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1,
            "notice_period_days": 30,
            "willing_to_relocate": True,
        },
    }


def substance_no_jargon_candidate():
    """A 'Tier 5' candidate who never says RAG/Pinecone but whose career
    history shows real recommendation-system / ranking work at a product
    company -- the JD's explicit description of what SHOULD be ranked highly.
    """
    return {
        "candidate_id": "CAND_9999002",
        "profile": {
            "current_title": "Backend Engineer",
            "current_company": "Flipkart",
            "years_of_experience": 6.5,
            "location": "Bangalore, Karnataka",
            "country": "India",
            "summary": "Backend engineer who's spent the last several years building the "
                       "personalization and recommendation system that powers our product feed. "
                       "Comfortable across the modeling and serving stack.",
        },
        "career_history": [
            {
                "company": "Flipkart",
                "title": "Backend Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 65,
                "is_current": True,
                "description": "Built and owned the recommendation system serving our product discovery "
                                "feed end to end -- candidate generation, feature pipeline, ranking model "
                                "(gradient-boosted trees), and the A/B testing harness used to validate "
                                "every change before full rollout. Also responsible for monitoring "
                                "offline-to-online metric correlation so we could trust offline evaluation.",
            },
            {
                "company": "Myntra",
                "title": "Software Engineer",
                "start_date": "2018-01-01",
                "end_date": "2020-12-01",
                "duration_months": 35,
                "is_current": False,
                "description": "Worked on search relevance -- improved the ranking function blending "
                                "text match scores with click-through-rate signals, and built the "
                                "offline evaluation pipeline (precision/recall at k) used to validate "
                                "ranking changes before shipping.",
            },
        ],
        "education": [{"institution": "X", "degree": "B.Tech", "field_of_study": "CS", "start_year": 2014, "end_year": 2018}],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 60},
            {"name": "XGBoost", "proficiency": "advanced", "endorsements": 8, "duration_months": 40},
            {"name": "A/B Testing", "proficiency": "advanced", "endorsements": 6, "duration_months": 48},
            {"name": "Search Ranking", "proficiency": "advanced", "endorsements": 5, "duration_months": 48},
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.7,
            "last_active_date": "2026-06-10",
            "open_to_work_flag": True,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": -1,
            "notice_period_days": 45,
            "willing_to_relocate": False,
        },
    }


def test_keyword_stuffer_scores_below_substance_candidate():
    jd = make_jd()
    stuffer = keyword_stuffer_candidate()
    substance = substance_no_jargon_candidate()

    # Use neutral (mid-range) semantic similarity for both so the test
    # isolates the structured-feature logic rather than depending on the
    # TF-IDF/SVD fit, which needs a larger corpus to be meaningful.
    stuffer_score = score_candidate(stuffer, jd, semantic_similarity=0.5)
    substance_score = score_candidate(substance, jd, semantic_similarity=0.5)

    print(f"Stuffer score: {stuffer_score.score:.4f}  flags={stuffer_score.flags}")
    print(f"Substance score: {substance_score.score:.4f}  flags={substance_score.flags}")

    assert substance_score.score > stuffer_score.score, (
        "The substance-but-no-jargon candidate must outrank the keyword-stuffer "
        "candidate -- this is the JD's core acceptance test."
    )
    # The stuffer's score should be clearly low (irrelevant title bucket),
    # not just marginally behind.
    assert stuffer_score.score < 0.35


if __name__ == "__main__":
    import inspect
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"PASS: {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL: {name} -- {e}")
    print(f"\n{'All tests passed.' if failures == 0 else f'{failures} test(s) failed.'}")
    sys.exit(1 if failures else 0)
