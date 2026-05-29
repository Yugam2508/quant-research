"""
steerer.py — narrative continuity injector

Reads the last N daily reports and extracts a context summary
so Gemini can track narrative continuity day-to-day.

This prevents the AI from treating each day as isolated —
it can say "continuing the energy sector weakness from Tuesday"
or "this reverses yesterday's risk-off tone".
"""

from pathlib import Path
from datetime import date
import os
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

REPORTS_DIR = Path(__file__).parents[2] / "reports" / "daily"


def build_context(n_days: int = 3) -> str:
    """
    Read the last n_days reports and return a compact context string
    to inject into the analyst prompt.

    Returns empty string if no prior reports exist.
    """
    reports = sorted(REPORTS_DIR.glob("*.md"), reverse=True)
    # exclude sample reports
    reports = [r for r in reports if "sample" not in r.stem]

    if not reports:
        return ""

    recent = reports[:n_days]
    if not recent:
        return ""

    combined = ""
    for r in reversed(recent):   # oldest first
        combined += f"\n\n--- {r.stem} ---\n{r.read_text()[:800]}"  # first 800 chars

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""You are summarising recent market reports for context injection.

Recent reports (oldest first):
{combined}

Extract a 3-5 bullet context summary covering:
- The dominant narrative / regime of the past few days
- Any trends that are continuing or reversing
- Any specific tickers or themes that keep appearing
- The overall tone (bullish / bearish / mixed)

Be extremely concise. Each bullet max 15 words. No headers.""",
        config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.1),
    )

    context = response.text.strip()
    console.print(f"[dim]context injected from {len(recent)} prior report(s)[/dim]")
    return context


def format_context_block(context: str) -> str:
    """Wrap context for injection into the analyst prompt."""
    if not context:
        return ""
    return f"""
## prior context (last {3} sessions)
{context}

Use this context to maintain narrative continuity — note what's continuing, reversing, or new.
"""
