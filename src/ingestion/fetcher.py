"""
fetcher.py — yfinance data fetcher with parquet caching
"""

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console

console = Console()

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _generate_mock_prices(tickers: list, n_days: int = 65) -> pd.DataFrame:
    console.print("[yellow]⚠ yfinance unavailable — using mock data for testing[/yellow]")
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    data = {}
    for t in tickers:
        np.random.seed(hash(t) % 10000)
        drift = np.random.uniform(-0.0005, 0.001)
        vol   = np.random.uniform(0.008, 0.022)
        start = np.random.uniform(50, 500)
        rets  = np.random.normal(drift, vol, n_days)
        data[t] = start * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


def _extract_close(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Handle the different column structures yfinance returns
    depending on version and number of tickers requested.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        # single ticker — flat columns like Close, High, Low...
        if "Close" in raw.columns:
            return raw[["Close"]].rename(columns={"Close": tickers[0]})
        return raw.iloc[:, :1].rename(columns={raw.columns[0]: tickers[0]})

    # multi-level columns — figure out which level has the metric names
    l0 = raw.columns.get_level_values(0).unique().tolist()
    l1 = raw.columns.get_level_values(1).unique().tolist()

    if "Close" in l0:
        # (metric, ticker) — standard older yfinance
        return raw["Close"]
    elif "Close" in l1:
        # (ticker, metric) — some versions flip it
        return raw.xs("Close", axis=1, level=1)
    elif "Price" in l0:
        # newest yfinance uses "Price" as top level
        sub = raw["Price"]
        if isinstance(sub, pd.Series):
            return sub.to_frame(name=tickers[0])
        return sub
    else:
        # fallback: just take the first metric level
        first = l0[0]
        console.print(f"[yellow]column guessing — using '{first}' as close proxy[/yellow]")
        result = raw[first]
        if isinstance(result, pd.Series):
            return result.to_frame(name=tickers[0])
        return result


def fetch_prices(
    tickers: list,
    period: str = "3mo",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers.
    Returns a DataFrame indexed by date with tickers as columns.
    """
    cache_key  = "_".join(sorted(tickers)) + f"_{period}"
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

        prices = _extract_close(raw, tickers)
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
    df = fetch_prices([ticker], period=period)
    return df[ticker].dropna()
