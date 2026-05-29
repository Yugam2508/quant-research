"""
analyst.py — Gemini-powered market analysis layer

Uses google-genai (the new SDK) with gemini-2.5-flash.
Set GEMINI_API_KEY in your .env file.
"""

import os
import json
from datetime import date

import pandas as pd
from rich.console import Console

console = Console()

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Add it to your .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _build_prompt(snapshot: pd.DataFrame, ticker_meta: dict, as_of: date) -> tuple[str, str]:
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

    system = (
        "You are a senior quantitative analyst writing a daily internal market memo. "
        "Your audience is experienced investors who want signal, not noise. "
        "Write with precision and dry wit. Avoid filler phrases. "
        "Use concrete numbers. Flag genuine anomalies. Be concise."
    )

    user = f"""Today is {as_of.strftime('%A, %d %B %Y')}.

Here is the cross-asset signal snapshot:

{data_json}

Write a daily market pulse memo in this exact markdown structure:

# Market Pulse — {as_of.strftime('%d %b %Y')}

## regime read
One paragraph (3–5 sentences). What is the market telling us today? Characterise the broad risk environment — risk-on/off, sector rotation, any divergences between equities / bonds / commodities / crypto. Name specific tickers and numbers.

## top movers
A compact markdown table with the 5 biggest movers (by absolute 1-day return), showing: Ticker | Name | 1D % | 5D % | RSI | Signal.
In the Signal column write one of: 🔥 momentum | ❄️ oversold | ⚡ breakout | 🔻 breakdown | ➡️ neutral

## cross-asset radar
Bullet list. One bullet per sector (US Equity, Fixed Income, Commodities, Crypto, International). Each bullet: sector name, 1–2 sentence observation using the data.

## signals worth watching
3 tickers with the most interesting signal setups right now (high RSI divergence, momentum extremes, vol compression, etc). For each: ticker, the setup in one sentence, what to watch.

## quant footnote
One sentence of dry, technical observation about the data — something a PM would actually find useful."""

    return system, user


def generate_daily_pulse(
    snapshot: pd.DataFrame,
    ticker_meta: dict,
    as_of: date,
) -> str:
    system, user = _build_prompt(snapshot, ticker_meta, as_of)

    console.print("[cyan]calling Gemini for market analysis...[/cyan]")

    from google.genai import types
    client = _get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1500,
        ),
    )

    return response.text


def generate_risk_comment(snapshot: pd.DataFrame, as_of: date) -> str:
    high_vol   = snapshot.nlargest(3, "vol_20d")[["vol_20d", "rsi_14"]]
    overbought = snapshot[snapshot["rsi_14"] > 70].index.tolist()
    oversold   = snapshot[snapshot["rsi_14"] < 30].index.tolist()

    prompt = (
        f"Date: {as_of}. "
        f"High vol names: {high_vol.to_dict()}. "
        f"Overbought (RSI>70): {overbought}. "
        f"Oversold (RSI<30): {oversold}.\n\n"
        "Write a single risk comment sentence (max 40 words) for the bottom of a daily market report. "
        "Be specific and dry. No filler."
    )

    from google.genai import types
    client = _get_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=100),
    )

    return response.text.strip()
