"""
router.py — market regime classifier

Analyses the signal snapshot and classifies the current regime.
The regime is used to route to the appropriate analysis prompt style.

Regimes:
  RISK_ON       — broad rally, momentum positive, credit spreads tight
  RISK_OFF      — broad selloff, defensive rotation, vol elevated
  TRENDING      — strong directional momentum in a clear sector
  MEAN_REVERTING— choppy, low momentum, RSI extremes reverting
  CRISIS        — vol spike, correlations converging, broad breakdown
"""

import json
import os
import pandas as pd
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

REGIMES = ["RISK_ON", "RISK_OFF", "TRENDING", "MEAN_REVERTING", "CRISIS"]


def classify_regime(snapshot: pd.DataFrame) -> dict:
    """
    Classify the current market regime from a signal snapshot.

    Returns:
        {
            "regime": str,          # one of REGIMES
            "confidence": float,    # 0.0–1.0
            "reasoning": str,       # 1-2 sentence explanation
            "prompt_style": str     # guidance for the analyst prompt
        }
    """
    # build a compact summary for the router
    summary = {
        "avg_ret_1d":    round(snapshot["ret_1d"].mean(), 2),
        "avg_ret_5d":    round(snapshot["ret_5d"].mean(), 2),
        "avg_vol":       round(snapshot["vol_20d"].mean(), 2),
        "pct_positive_1d": round((snapshot["ret_1d"] > 0).mean() * 100, 1),
        "avg_rsi":       round(snapshot["rsi_14"].mean(), 1),
        "max_abs_zscore":round(snapshot["cs_zscore"].abs().max(), 2),
        "n_overbought":  int((snapshot["rsi_14"] > 70).sum()),
        "n_oversold":    int((snapshot["rsi_14"] < 30).sum()),
    }

    prompt = f"""You are a market regime classifier. Given this cross-asset signal summary:

{json.dumps(summary, indent=2)}

Classify the current market regime as exactly one of: {', '.join(REGIMES)}

Respond in JSON only (no markdown, no backticks):
{{
  "regime": "<REGIME>",
  "confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentences using the numbers>",
  "prompt_style": "<one of: momentum_focus | defensive_focus | dispersion_focus | reversion_focus | crisis_focus>"
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
        result = json.loads(text)
    except json.JSONDecodeError:
        # fallback if JSON parse fails
        result = {
            "regime": "RISK_ON",
            "confidence": 0.5,
            "reasoning": "Could not parse regime classification.",
            "prompt_style": "momentum_focus",
        }

    console.print(f"[cyan]regime: [bold]{result['regime']}[/bold] "
                  f"(confidence: {result['confidence']:.0%})[/cyan]")
    return result
