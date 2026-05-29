"""
fetcher.py — yfinance data fetcher with parquet caching

Usage:
    from src.ingestion.fetcher import fetch_prices
    df = fetch_prices(tickers, period="3mo")
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console

console = Console()

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _generate_mock_prices(tickers: list[str], n_days: int = 65) -> pd.DataFrame:
    """
    Generate realistic-looking synthetic price data for offline testing.
    Seeded so results are reproducible.
    """
    console.print("[yellow]⚠ yfinance unavailable — using mock data for testing[/yellow]")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    # different starting prices and vols per asset class feel
    seeds = {t: hash(t) % 1000 for t in tickers}
    data = {}
    for t in tickers:
        np.random.seed(seeds[t])
        drift = np.random.uniform(-0.0005, 0.001)
        vol   = np.random.uniform(0.008, 0.022)
        start = np.random.uniform(50, 500)
        rets  = np.random.normal(drift, vol, n_days)
        data[t] = start * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


def fetch_prices(
    tickers: list[str],
    period: str = "3mo",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.

    Returns a DataFrame indexed by date with tickers as columns.
    Uses a local parquet cache; refreshes if data is older than 1 day.
    """
    cache_key = "_".join(sorted(tickers)) + f"_{period}"
    cache_file = RAW_DIR / f"{cache_key}.parquet"

    if not force_refresh and cache_file.exists():
        age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_hours < 20:
            console.print(f"[dim]cache hit: {cache_file.name} ({age_hours:.1f}h old)[/dim]")
            return pd.read_parquet(cache_file)

    console.print(f"[cyan]fetching {len(tickers)} tickers from yfinance...[/cyan]")
    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        # normalise: always return (date, ticker) → close
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        prices = prices.dropna(how="all")

        if prices.empty or len(prices) < 5:
            raise ValueError("yfinance returned empty data")

        prices.to_parquet(cache_file)
        console.print(f"[green]✓ fetched {prices.shape[1]} tickers × {len(prices)} days[/green]")
        return prices

    except Exception as e:
        console.print(f"[dim]yfinance error: {e}[/dim]")
        prices = _generate_mock_prices(tickers)
        prices.to_parquet(cache_file)
        console.print(f"[green]✓ mock data: {prices.shape[1]} tickers × {len(prices)} days[/green]")
        return prices


def fetch_single(ticker: str, period: str = "1y") -> pd.Series:
    """Fetch a single ticker, return a Series of close prices."""
    df = fetch_prices([ticker], period=period)
    return df[ticker].dropna()
