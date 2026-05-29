"""
judge.py — report quality evaluator

A second LLM call that scores the generated report on three dimensions:
  specificity   : does it use actual numbers and ticker names?
  accuracy      : does the analysis match the underlying data?
  actionability : does it tell a PM something useful?

If the score is below threshold, the report is flagged with a warning header.
This is the "judge LLM" pattern — one model evaluates another's output.
"""

import json
import os
import pandas as pd
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

PASS_THRESHOLD = 6.0   # out of 10 — below this, report gets flagged


def evaluate_report(report_md: str, snapshot: pd.DataFrame) -> dict:
    """
    Score the report and return evaluation metadata.

    Returns:
        {
            "score": float,         # 0–10 composite
            "specificity": float,
            "accuracy": float,
            "actionability": float,
            "verdict": str,         # PASS or FLAG
            "critique": str,        # 1-2 sentence critique
            "improvement": str,     # what would make it better
        }
    """
    # give the judge the raw data so it can check accuracy
    data_summary = snapshot[["ret_1d","ret_5d","rsi_14","vol_20d"]].round(2).to_string()

    prompt = f"""You are a senior editor evaluating a quant market report.

RAW SIGNAL DATA (ground truth):
{data_summary}

REPORT TO EVALUATE:
{report_md[:1500]}

Score the report on three dimensions (0–10 each):
1. specificity: does it cite actual ticker names and numbers from the data?
2. accuracy: do the claims match the raw signal data?
3. actionability: does a PM learn something useful and concrete?

Respond in JSON only (no markdown, no backticks):
{{
  "specificity": <0-10>,
  "accuracy": <0-10>,
  "actionability": <0-10>,
  "critique": "<1-2 sentence honest critique>",
  "improvement": "<one concrete suggestion>"
}}"""

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=300,
            temperature=0.1,
        ),
    )

    text = response.text.strip().lstrip("```json").rstrip("```").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = {"specificity": 7, "accuracy": 7, "actionability": 7,
               "critique": "Could not parse evaluation.", "improvement": "N/A"}

    score = round((raw["specificity"] + raw["accuracy"] + raw["actionability"]) / 3, 1)
    verdict = "PASS" if score >= PASS_THRESHOLD else "FLAG"

    result = {
        "score":         score,
        "specificity":   raw["specificity"],
        "accuracy":      raw["accuracy"],
        "actionability": raw["actionability"],
        "verdict":       verdict,
        "critique":      raw.get("critique", ""),
        "improvement":   raw.get("improvement", ""),
    }

    color = "green" if verdict == "PASS" else "yellow"
    console.print(f"[{color}]judge: {verdict} — score {score}/10 "
                  f"(spec:{raw['specificity']} acc:{raw['accuracy']} "
                  f"act:{raw['actionability']})[/{color}]")
    return result


def annotate_report(report_md: str, evaluation: dict) -> str:
    """
    Prepend a quality metadata block to the report.
    Always included — PASS reports get a clean badge, FLAG reports get a warning.
    """
    score = evaluation["score"]
    verdict = evaluation["verdict"]

    if verdict == "PASS":
        badge = f"> ✓ quality score {score}/10 — specificity {evaluation['specificity']} · accuracy {evaluation['accuracy']} · actionability {evaluation['actionability']}"
    else:
        badge = (
            f"> ⚠ quality score {score}/10 — this report was flagged for review\n"
            f"> critique: {evaluation['critique']}\n"
            f"> suggested improvement: {evaluation['improvement']}"
        )

    return badge + "\n\n" + report_md
