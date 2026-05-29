"""
engine.py — vectorised backtester scaffold

A simple, dependency-light backtester that works on daily price data.
Designed to be extended for the weekly factor report.

Basic usage:
    from src.backtest.engine import Backtest
    from src.signals.factors import momentum

    bt = Backtest(prices)
    results = bt.run(signal_fn=lambda p: momentum(p, 20))
    print(results.summary())
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResults:
    returns: pd.Series        # daily strategy returns
    positions: pd.DataFrame   # daily positions
    turnover: pd.Series       # daily portfolio turnover

    def summary(self) -> dict:
        r = self.returns.dropna()
        total = (1 + r).prod() - 1
        ann = (1 + total) ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (r.cumsum() - r.cumsum().cummax()).min()
        return {
            "total_return": round(total * 100, 2),
            "annualised_return": round(ann * 100, 2),
            "annualised_vol": round(vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(dd * 100, 2),
            "avg_daily_turnover": round(self.turnover.mean() * 100, 2),
        }


class Backtest:
    """
    Simple long/short equal-weight backtest.

    The signal_fn receives a price DataFrame and returns a signal
    DataFrame of the same shape. Positive signal = long, negative = short.
    Positions are cross-sectionally z-scored and then top/bottom n are held.
    """

    def __init__(self, prices: pd.DataFrame, n_longs: int = 3, n_shorts: int = 3):
        self.prices = prices
        self.n_longs = n_longs
        self.n_shorts = n_shorts

    def run(self, signal_fn) -> BacktestResults:
        signals = signal_fn(self.prices)
        daily_ret = self.prices.pct_change()

        positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            row = signals.loc[date].dropna()
            if len(row) < self.n_longs + self.n_shorts:
                continue
            longs  = row.nlargest(self.n_longs).index
            shorts = row.nsmallest(self.n_shorts).index
            positions.loc[date, longs]  =  1.0 / self.n_longs
            positions.loc[date, shorts] = -1.0 / self.n_shorts

        # shift positions by 1 day (no look-ahead)
        positions_shifted = positions.shift(1)
        strategy_returns = (positions_shifted * daily_ret).sum(axis=1)
        turnover = positions.diff().abs().sum(axis=1)

        return BacktestResults(
            returns=strategy_returns,
            positions=positions,
            turnover=turnover,
        )
