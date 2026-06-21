#!/usr/bin/env python3
"""
parse_jd_llm.py
===============
Offline pre-computation script that uses Google's Gemini LLM API to parse
the Job Description markdown file (`docs/job_description.md`) and extract
a high-fidelity, structured JSON representation of the hiring requirements.

Saves structured requirements to `docs/job_description_structured.json`.
"""

import json
import os
import sys
from pathlib import Path

# Add workspace root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai is not installed. Please run `pip install -r requirements.txt` first.")
    sys.exit(1)


SYSTEM_PROMPT = """
You are an expert technical recruiting coordinator. Your job is to read a Job Description (JD) and extract the exact hiring constraints, skill groups, location preferences, experience ranges, and disqualifiers.

You must output a single, raw, valid JSON object with NO markdown formatting (do NOT wrap it in ```json blocks).

Here is the target schema for your JSON output:
{
  "role_title": "string",
  "min_years": float,
  "max_years": float,
  "preferred_locations": ["string"],
  "country_required": "string (e.g. 'india')",
  "visa_sponsorship": boolean,
  "notice_period_soft_cap_days": int,
  "notice_period_hard_concern_days": int,
  "must_have_groups": {
    "concept_name_1": ["keyword1", "keyword2", "keyword3"],
    "concept_name_2": ["keyword4", "keyword5"]
  },
  "nice_to_have_groups": {
    "concept_name_3": ["keyword6", "keyword7"]
  },
  "excluded_primary_domains": ["string"],
  "services_firms": ["string"],
  "research_only_title_hints": ["string"],
  "research_only_text_hints": ["string"],
  "non_coding_title_hints": ["string"]
}

Guidelines:
1. Extract must-have and nice-to-have skill concepts, map them to a small name, and list all technical keywords, libraries, databases, or frameworks associated with them.
2. Identify red flags / things explicitly NOT wanted (e.g. consulting firms, CV/speech only, research only with no production, architecture drift / EM roles with no recent coding).
3. If years of experience is described as a guideline or range, capture the start and end (e.g. "5-9 years" -> min=5.0, max=9.0).
4. Parse location terms carefully.
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse JD using Gemini LLM API.")
    parser.add_argument(
        "--jd",
        default="docs/job_description.md",
        help="Path to job_description.md"
    )
    parser.add_argument(
        "--out",
        default="docs/job_description_structured.json",
        help="Path to save structured json output"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    jd_path = Path(args.jd)
    if not jd_path.is_absolute():
        jd_path = repo_root / jd_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    if not jd_path.exists():
        print(f"Error: Job description file not found at {jd_path}")
        sys.exit(1)

    # Resolve API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it first, e.g. in Powershell: $env:GEMINI_API_KEY='your-key'")
        sys.exit(1)

    print(f"Loading Job Description text from {jd_path}...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("Configuring Gemini client...")
    genai.configure(api_key=api_key)
    
    # We use gemini-1.5-flash for fast and reliable extraction
    model = genai.GenerativeModel("gemini-1.5-flash")

    print("Sending Job Description to Gemini for extraction...")
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\nHere is the Job Description:\n\n{jd_text}"
    )

    result_text = response.text.strip()

    # Clean markdown formatting if model output wraps it
    if result_text.startswith("```"):
        # Strip ```json and ```
        lines = result_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        result_text = "\n".join(lines).strip()

    try:
        structured_data = json.loads(result_text)
    except json.JSONDecodeError as e:
        print("Error: Gemini output was not valid JSON.")
        print("Raw response from Gemini:")
        print(response.text)
        print(f"JSON Parsing Error: {e}")
        sys.exit(1)

    print(f"Parsed successfully! Saving structured requirements to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)

    print("Offline Job Description parsing completed successfully!")


if __name__ == "__main__":
    main()
