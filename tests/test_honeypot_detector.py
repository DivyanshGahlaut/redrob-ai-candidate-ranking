"""Tests for honeypot_detector.py against known patterns found during dataset exploration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import honeypot_detector as hp


def make_candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000000",
        "profile": {"years_of_experience": 6.0},
        "career_history": [
            {
                "company": "Acme",
                "title": "ML Engineer",
                "start_date": "2022-01-01",
                "end_date": None,
                "duration_months": 53,  # ~ matches 2022-01 to 2026-06
                "is_current": True,
            }
        ],
        "education": [{"institution": "X", "degree": "B.E.", "start_year": 2016, "end_year": 2020}],
        "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 5, "duration_months": 36}],
    }
    base.update(overrides)
    return base


def test_clean_profile_is_not_honeypot():
    cand = make_candidate()
    assert hp.run_checks(cand) == []
    assert not hp.is_likely_honeypot(cand)
    assert hp.honeypot_score(cand) == 0.0


def test_yoe_vs_history_mismatch_detected():
    cand = make_candidate(profile={"years_of_experience": 20.0})  # history only sums to ~4.4 yrs
    flags = hp.run_checks(cand)
    assert "yoe_vs_history_mismatch" in flags


def test_duration_vs_dates_mismatch_detected():
    cand = make_candidate(career_history=[
        {
            "company": "Stark Industries",
            "title": "Graphic Designer",
            "start_date": "2024-09-04",
            "end_date": None,
            "duration_months": 171,  # wildly exceeds actual elapsed time
            "is_current": True,
        }
    ])
    flags = hp.run_checks(cand)
    assert "duration_vs_date_span_mismatch" in flags


def test_expert_zero_duration_detected():
    cand = make_candidate(skills=[
        {"name": "MLflow", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
        {"name": "Photoshop", "proficiency": "expert", "endorsements": 2, "duration_months": 0},
        {"name": "Docker", "proficiency": "expert", "endorsements": 1, "duration_months": 0},
    ])
    flags = hp.run_checks(cand)
    assert "expert_with_zero_duration" in flags


def test_education_date_inversion_detected():
    cand = make_candidate(education=[{"institution": "X", "degree": "B.E.", "start_year": 2020, "end_year": 2016}])
    flags = hp.run_checks(cand)
    assert "education_date_inversion" in flags


def test_multi_flag_is_hard_honeypot():
    cand = make_candidate(
        profile={"years_of_experience": 20.0},
        career_history=[
            {
                "company": "Stark Industries",
                "title": "Graphic Designer",
                "start_date": "2024-09-04",
                "end_date": None,
                "duration_months": 171,
                "is_current": True,
            }
        ],
    )
    assert hp.is_likely_honeypot(cand)
    assert hp.honeypot_score(cand) >= 0.85


def test_single_flag_is_soft_not_hard():
    cand = make_candidate(profile={"years_of_experience": 9.0})  # ~4.4yr history vs 9yr stated; one flag only
    flags = hp.run_checks(cand)
    assert len(flags) <= 1
    assert not hp.is_likely_honeypot(cand)


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
