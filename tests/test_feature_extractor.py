"""Tests for feature_extractor.py -- the structured signal checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import feature_extractor as fx


def test_title_description_coherence_matches():
    cand = {
        "career_history": [
            {
                "is_current": True,
                "title": "DevOps Engineer",
                "description": "Owned Kubernetes infra, Terraform modules, CI/CD pipelines.",
            }
        ]
    }
    assert fx.title_description_coherence(cand) == 1.0


def test_title_description_coherence_mismatch_flagged():
    cand = {
        "career_history": [
            {
                "is_current": True,
                "title": "DevOps Engineer",
                "description": "Frontend engineering with React, TypeScript, Webpack, Jest, Cypress.",
            }
        ]
    }
    assert fx.title_description_coherence(cand) == 0.3


def test_title_description_coherence_unknown_title_neutral():
    cand = {
        "career_history": [
            {"is_current": True, "title": "Chief Vibes Officer", "description": "Various things."}
        ]
    }
    assert fx.title_description_coherence(cand) == 0.7


def test_experience_band_fit_inside_band_is_full_credit():
    assert fx.experience_band_fit(7.0, 5.0, 9.0) == 1.0
    assert fx.experience_band_fit(5.0, 5.0, 9.0) == 1.0
    assert fx.experience_band_fit(9.0, 5.0, 9.0) == 1.0


def test_experience_band_fit_below_band_decays():
    full = fx.experience_band_fit(5.0, 5.0, 9.0)
    one_under = fx.experience_band_fit(4.0, 5.0, 9.0)
    far_under = fx.experience_band_fit(1.0, 5.0, 9.0)
    assert full > one_under > far_under
    assert far_under >= 0.0


def test_pure_services_career_detected():
    cand = {"career_history": [{"company": "TCS"}, {"company": "Infosys"}]}
    assert fx.pure_services_career(cand, {"TCS", "Infosys", "Wipro"})


def test_mixed_services_and_product_not_flagged_as_pure():
    cand = {"career_history": [{"company": "TCS"}, {"company": "Google"}]}
    assert not fx.pure_services_career(cand, {"TCS", "Infosys", "Wipro"})


def test_currently_at_services_with_product_history_is_fine():
    cand = {
        "career_history": [
            {"company": "TCS", "is_current": True},
            {"company": "Google", "is_current": False},
        ]
    }
    assert fx.currently_at_services_with_product_history(cand, {"TCS"})


def test_job_hopping_score_penalizes_short_tenure():
    hopper = {"career_history": [
        {"duration_months": 10}, {"duration_months": 8}, {"duration_months": 9}, {"duration_months": 11}
    ]}
    stable = {"career_history": [
        {"duration_months": 36}, {"duration_months": 40}, {"duration_months": 30}
    ]}
    assert fx.job_hopping_score(hopper) < fx.job_hopping_score(stable)
    assert fx.job_hopping_score(stable) == 1.0


def test_excluded_primary_domain_without_nlp_flagged():
    cand = {
        "profile": {"summary": "Computer vision engineer focused on object detection and segmentation."},
        "career_history": [{"title": "CV Engineer", "description": "Built image classification pipelines."}],
        "skills": [{"name": "Computer Vision"}],
    }
    assert fx.excluded_primary_domain(cand, ["computer vision"])


def test_excluded_primary_domain_with_nlp_not_flagged():
    cand = {
        "profile": {"summary": "Computer vision background, now doing NLP and retrieval work."},
        "career_history": [{"title": "ML Engineer", "description": "Built embeddings-based search ranking."}],
        "skills": [{"name": "Computer Vision"}, {"name": "NLP"}],
    }
    assert not fx.excluded_primary_domain(cand, ["computer vision"])


def test_skill_group_match_rewards_duration_over_bare_listing():
    groups = {"python": ["python"]}
    well_used = {"skills": [{"name": "Python", "proficiency": "expert", "duration_months": 36, "endorsements": 10}], "profile": {}, "career_history": []}
    bare_listed = {"skills": [{"name": "Python", "proficiency": "expert", "duration_months": 0, "endorsements": 0}], "profile": {}, "career_history": []}
    well_score = fx.skill_group_match(well_used, groups)["python"]
    bare_score = fx.skill_group_match(bare_listed, groups)["python"]
    assert well_score > bare_score


def test_location_fit_preferred_city_full_credit():
    cand = {"profile": {"location": "Pune, Maharashtra", "country": "India"}, "redrob_signals": {"willing_to_relocate": False}}
    assert fx.location_fit(cand, ["pune", "noida"], "india") == 1.0


def test_location_fit_outside_india_heavily_penalized():
    cand = {"profile": {"location": "San Francisco", "country": "USA"}, "redrob_signals": {"willing_to_relocate": True}}
    score = fx.location_fit(cand, ["pune", "noida"], "india")
    assert score < 0.2


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
