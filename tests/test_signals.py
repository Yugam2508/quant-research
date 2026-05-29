"""
test_signals.py — basic signal sanity checks

Run with:  python -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.signals.factors import (
    momentum, rolling_vol, rsi, cross_sectional_zscore, build_snapshot
)


@pytest.fixture
def prices():
    """Synthetic price series: 3 tickers, 90 days, random walk."""
    np.random.seed(42)
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for t in ["A", "B", "C"]:
        returns = np.random.normal(0.0003, 0.01, n)
        data[t] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


def test_momentum_shape(prices):
    m = momentum(prices, 20)
    assert m.shape == prices.shape


def test_rsi_bounds(prices):
    r = rsi(prices, 14).dropna()
    assert (r >= 0).all().all()
    assert (r <= 100).all().all()


def test_rolling_vol_positive(prices):
    v = rolling_vol(prices, 20).dropna()
    assert (v > 0).all().all()


def test_zscore_mean_near_zero(prices):
    z = cross_sectional_zscore(prices, 20).dropna()
    # cross-sectional mean should be ~0 each day
    assert z.mean(axis=1).abs().mean() < 0.01


def test_build_snapshot_columns(prices):
    snap = build_snapshot(prices)
    expected = {"price", "ret_1d", "ret_5d", "ret_20d", "ret_60d",
                "vol_20d", "rsi_14", "cs_zscore"}
    assert expected.issubset(set(snap.columns))
    assert set(snap.index) == {"A", "B", "C"}
