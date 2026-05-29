"""
factors.py — signal library

Each function takes a prices DataFrame (date × ticker) and returns
a DataFrame of the same shape with the signal values.

Signals are normalised to be interpretable by Claude:
  momentum  : trailing return (%)
  volatility: annualised rolling vol (%)
  rsi       : 0–100 RSI
  z_score   : cross-sectional z-score of 1-month return
"""

import numpy as np
import pandas as pd


# ── momentum ──────────────────────────────────────────────────────────────────

def momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Trailing n-day return (%)."""
    return prices.pct_change(window) * 100


def short_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def medium_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=20)


def long_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=60)


# ── volatility ────────────────────────────────────────────────────────────────

def rolling_vol(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Annualised rolling volatility (%)."""
    daily_returns = prices.pct_change()
    return daily_returns.rolling(window).std() * np.sqrt(252) * 100


# ── RSI ───────────────────────────────────────────────────────────────────────

def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index (0–100)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── cross-sectional z-score ───────────────────────────────────────────────────

def cross_sectional_zscore(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Cross-sectional z-score of trailing returns.
    Positive = outperforming the universe on a given day.
    """
    ret = momentum(prices, window=window)
    mean = ret.mean(axis=1)
    std = ret.std(axis=1)
    return ret.sub(mean, axis=0).div(std, axis=0)


# ── composite snapshot ────────────────────────────────────────────────────────

def build_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cross-asset signal snapshot for the latest date.

    Returns a DataFrame indexed by ticker with columns:
        ret_1d, ret_5d, ret_20d, ret_60d,
        vol_20d, rsi_14, cs_zscore_20d, price
    """
    prices = prices.dropna(how="all").dropna(axis=0, thresh=len(prices.columns)//2)
    latest = prices.iloc[-1]

    snap = pd.DataFrame(index=prices.columns)
    snap["price"]        = latest
    snap["ret_1d"]       = prices.pct_change(1).iloc[-1] * 100
    snap["ret_5d"]       = prices.pct_change(5).iloc[-1] * 100
    snap["ret_20d"]      = prices.pct_change(20).iloc[-1] * 100
    snap["ret_60d"]      = prices.pct_change(60).iloc[-1] * 100
    snap["vol_20d"]      = rolling_vol(prices, 20).iloc[-1]
    snap["rsi_14"]       = rsi(prices, 14).iloc[-1]
    snap["cs_zscore"]    = cross_sectional_zscore(prices, 20).iloc[-1]

    return snap.round(2)
