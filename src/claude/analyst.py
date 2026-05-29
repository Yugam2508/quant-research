"""
analyst.py — Gemini-powered market analysis layer

Uses google-genai with gemini-2.5-flash.
Accepts regime classification and prior context for richer analysis.
"""

import os
import json
from datetime import date

import pandas as pd
from rich.console import Console

console = Console()

_client = None

STYLE_INSTRUCTIONS = {
    "momentum_focus":   "Focus on momentum leaders and laggards. Identify which trends have legs.",
    "defensive_focus":  "Focus on defensive positioning. Highlight safe havens and risk-off signals.",
    "dispersion_focus": "Focus on cross-sectional dispersion. Identify rotation themes and sector divergences.",
    "reversion_focus":  "Focus on mean-reversion setups. Highlight overbought/oversold extremes.",
    "crisis_focus":     "Focus on risk management. Flag correlation breakdown, vol spikes, and tail risks.",
}


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set. Add it to your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_daily_pulse(
    snapshot: pd.DataFrame,
    ticker_meta: dict,
    as_of: date,
    regime: dict = None,
    prior_context: str = "",
) -> str:
    """
    Generate a daily market pulse memo.

    Args:
        snapshot      : DataFrame from signals.build_snapshot()
        ticker_meta   : dict mapping ticker → {name, sector}
        as_of         : date of the report
        regime        : output from router.classify_regime() — optional
        prior_context : output from steerer.build_context() — optional

    Returns:
        Markdown string — the full report.
    """
    rows = []
    for ticker, row in snapshot.iterrows():
        meta = ticker_meta.get(ticker, {})
        rows.append({
            "ticker":    ticker,
            "name":      meta.get("name", ticker),
            "sector":    meta.get("sector", "Unknown"),
            "price":     row.get("price"),
            "ret_1d":    row.get("ret_1d"),
            "ret_5d":    row.get("ret_5d"),
            "ret_20d":   row.get("ret_20d"),
            "ret_60d":   row.get("ret_60d"),
            "vol_20d":   row.get("vol_20d"),
            "rsi_14":    row.get("rsi_14"),
            "cs_zscore": row.get("cs_zscore"),
        })

    data_json = json.dumps(rows, indent=2, default=str)

    # build regime block
    regime_block = ""
    style_instruction = STYLE_INSTRUCTIONS["momentum_focus"]
    if regime:
        style_instruction = STYLE_INSTRUCTIONS.get(
            regime.get("prompt_style", ""), style_instruction
        )
        regime_block = f"""
## detected regime: {regime['regime']} (confidence: {regime['confidence']:.0%})
{regime['reasoning']}
"""

    system_prompt = (
        "You are a senior quantitative analyst writing a daily internal market memo. "
        "Your audience is experienced investors who want signal, not noise. "
        "Write with precision and dry wit. Avoid filler phrases. "
        "Use concrete numbers. Flag genuine anomalies. Be concise.\n\n"
        f"Analysis style for today: {style_instruction}"
    )

    user_prompt = f"""Today is {as_of.strftime('%A, %d %B %Y')}.
{regime_block}
{prior_context}

Cross-asset signal snapshot:
{data_json}

Write a daily market pulse memo in this exact markdown structure:

# Market Pulse — {as_of.strftime('%d %b %Y')}

## regime read
One paragraph (3–5 sentences). Characterise the broad risk environment. Name specific tickers and numbers. Note anything continuing or reversing from prior sessions if context was provided.

## top movers
Markdown table — 5 biggest movers by absolute 1D return: Ticker | Name | 1D % | 5D % | RSI | Signal
Signal column: 🔥 momentum | ❄️ oversold | ⚡ breakout | 🔻 breakdown | ➡️ neutral

## cross-asset radar
One bullet per sector. Each bullet: sector name, 1–2 sentence observation with numbers.

## signals worth watching
3 tickers with the most interesting setups. For each: ticker, the setup in one sentence, what to watch.

## quant footnote
One sentence of dry technical observation a PM would find useful."""

    console.print("[cyan]calling Gemini for market analysis...[/cyan]")

    from google.genai import types
    response = _get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1800,
            temperature=0.4,
        ),
    )

    return response.text


def generate_risk_comment(snapshot: pd.DataFrame, as_of: date) -> str:
    high_vol   = snapshot.nlargest(3, "vol_20d")[["vol_20d", "rsi_14"]]
    overbought = snapshot[snapshot["rsi_14"] > 70].index.tolist()
    oversold   = snapshot[snapshot["rsi_14"] < 30].index.tolist()

    prompt = (
        f"Date: {as_of}. "
        f"High vol: {high_vol.to_dict()}. "
        f"Overbought (RSI>70): {overbought}. "
        f"Oversold (RSI<30): {oversold}.\n\n"
        "Write one risk comment sentence (max 40 words) for the bottom of a daily report. "
        "Be specific and dry. No filler."
    )

    from google.genai import types
    response = _get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=100),
    )
    return response.text.strip()
