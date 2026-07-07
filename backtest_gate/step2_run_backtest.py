"""
STEP 2 — The gate check: does the strategy survive real-world costs?

What this simulates, week by week over ~2 years:
  1. Rank the ~25 tradeable coins by momentum (how much each rose or fell
     over the past month, ignoring the very latest day).
  2. Bet UP on the 5 strongest, bet DOWN on the 5 weakest, equal money
     each side. Because the up-bets and down-bets are the same size,
     the overall market direction roughly cancels out ("market-neutral").
  3. Subtract three real costs:
       - exchange fees   (paid every time we trade)
       - funding         (the 8-hourly payment perps charge; longs pay
                          when the rate is positive, shorts receive it)
       - slippage        (prices move slightly against you when you trade)

Output: a chart, a CSV of daily results, and a printed verdict.

THE NUMBER THAT MATTERS: "Net Sharpe" at the bottom.
  Sharpe ratio = average return divided by how bumpy the ride was.
  Rule of thumb here:  > 1.0 promising | 0.5–1.0 marginal | < 0.5 stop.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- settings ------------------------------------
LOOKBACK = 21        # momentum window: ~1 month of daily moves
SKIP = 1             # ignore the most recent day (short-term reversal noise)
N_SIDE = 5           # coins per side (top 5 long, bottom 5 short)
REBALANCE = "W-MON"  # trade once a week, Mondays
FEE = 0.0005         # 5 bps taker fee per trade (Binance futures standard)
SLIPPAGE = 0.0003    # 3 bps assumed price impact per trade
DATA_DIR = "data"
# ----------------------------------------------------------------------------

# ------------------------------ load data -----------------------------------
ohlcv = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet")
funding = pd.read_parquet(f"{DATA_DIR}/funding.parquet")
universe = pd.read_csv(f"{DATA_DIR}/universe.csv", parse_dates=["month"])
universe = universe.dropna(subset=["symbols"])

close = ohlcv.pivot_table(index="date", columns="symbol", values="close")
close.index = close.index.normalize()
rets = close.pct_change()

# funding: sum the (usually three) 8-hour rates into one daily rate per coin
funding["day"] = funding["date"].dt.normalize()
fund_daily = (
    funding.groupby(["day", "symbol"])["funding_rate"].sum().unstack()
)
fund_daily = fund_daily.reindex(index=close.index, columns=close.columns).fillna(0.0)

# month -> allowed coins that month (point-in-time universe)
universe["symbols"] = universe["symbols"].str.split(",")
uni_by_month = {
    row.month.to_period("M"): set(row.symbols) for row in universe.itertuples()
}

# ------------------------- signal: momentum z-score -------------------------
# "How much did this coin move over the last month?" then standardized
# across coins so they're comparable.
mom = close.shift(SKIP) / close.shift(SKIP + LOOKBACK) - 1.0
zscore = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)

# --------------------------- weekly rebalance loop --------------------------
rebal_dates = [d for d in pd.date_range(close.index[LOOKBACK + SKIP + 5],
                                        close.index[-1], freq=REBALANCE)
               if d in close.index]

weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
current = pd.Series(0.0, index=close.columns)
turnover = pd.Series(0.0, index=close.index)

for d in close.index:
    if d in rebal_dates:
        allowed = uni_by_month.get(d.to_period("M") - 1)  # last completed month
        z = zscore.loc[d].dropna()
        if allowed:
            z = z[z.index.isin(allowed)]
        if len(z) >= 2 * N_SIDE:
            longs = z.nlargest(N_SIDE).index
            shorts = z.nsmallest(N_SIDE).index
            target = pd.Series(0.0, index=close.columns)
            target[longs] = 0.5 / N_SIDE    # +50% of capital spread over longs
            target[shorts] = -0.5 / N_SIDE  # -50% spread over shorts
            turnover[d] = (target - current).abs().sum()
            current = target
    weights.loc[d] = current

# ------------------------------ daily P&L -----------------------------------
w_lag = weights.shift(1).fillna(0.0)          # yesterday's positions earn today
gross = (w_lag * rets).sum(axis=1)            # signal P&L before costs
fee_cost = turnover * FEE                     # fees on amount traded
slip_cost = turnover * SLIPPAGE               # slippage on amount traded
# funding: a LONG position (w>0) PAYS positive funding -> negative P&L,
# a SHORT position (w<0) RECEIVES it. Hence the minus sign:
fund_pnl = -(w_lag * fund_daily).sum(axis=1)

net = gross - fee_cost - slip_cost + fund_pnl
net = net.loc[rebal_dates[0]:]
gross = gross.loc[net.index]
fee_cost, slip_cost, fund_pnl = (
    fee_cost.loc[net.index], slip_cost.loc[net.index], fund_pnl.loc[net.index]
)

# ------------------------------- results ------------------------------------
def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0


def maxdd(x):
    curve = (1 + x).cumprod()
    return float((curve / curve.cummax() - 1).min())


summary = pd.DataFrame(
    {
        "total_%": [
            100 * ((1 + s).prod() - 1)
            for s in [gross, -fee_cost, -slip_cost, fund_pnl, net]
        ]
    },
    index=["gross signal", "fees", "slippage", "funding", "NET"],
).round(2)

print("\n================ COST DECOMPOSITION (whole period) ================")
print(summary.to_string())
print("====================================================================")
print(f"Period            : {net.index[0].date()} -> {net.index[-1].date()}")
print(f"Net Sharpe        : {sharpe(net):.2f}   (gross: {sharpe(gross):.2f})")
print(f"Max drawdown      : {maxdd(net) * 100:.1f}%")
print(f"Avg weekly turnover: {turnover[turnover > 0].mean():.2f}x of capital")

print("\nVERDICT GUIDE:")
print("  Net Sharpe > 1.0  -> promising, proceed to robustness checks (§3.3)")
print("  0.5 - 1.0         -> marginal, try longer holding / fewer trades")
print("  < 0.5             -> signal doesn't survive costs; do NOT deploy")

# ------------------------------ save outputs --------------------------------
out = pd.DataFrame(
    {"gross": gross, "fees": -fee_cost, "slippage": -slip_cost,
     "funding": fund_pnl, "net": net}
)
out.to_csv("backtest_results.csv")

fig, ax = plt.subplots(figsize=(10, 5))
(1 + net).cumprod().plot(ax=ax, label=f"NET (Sharpe {sharpe(net):.2f})", lw=2)
(1 + gross).cumprod().plot(ax=ax, label="gross (no costs)", ls="--", alpha=0.7)
ax.set_title("Momentum L/S on crypto perps — growth of $1, after costs")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("backtest_curve.png", dpi=120)
print("\nSaved: backtest_results.csv, backtest_curve.png")
