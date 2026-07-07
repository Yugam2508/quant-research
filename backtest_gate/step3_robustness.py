"""
STEP 3 — Robustness checks (spec section 3.3). PRE-REGISTERED: the grid below
was fixed before running. Run once, interpret honestly, no cherry-picking.

What this answers:
  A. GRID: does the strategy work across nearby settings, or only at the
     exact ones we happened to pick? (lookback 14/21/30 x weekly/biweekly)
     A real signal degrades gracefully; an artifact falls off a cliff.
  B. LEG SPLIT: is the profit from the up-bets, the down-bets, or both?
  C. FUNDING ZEROED: does the edge survive if funding income disappears?
  D. DROP TOP 3: is it all coming from 3 lucky coins?

Reads the same data/ folder as step 2. Runs in seconds.
"""

import numpy as np
import pandas as pd

# --------------------- PRE-REGISTERED GRID (do not extend) ------------------
LOOKBACKS = [14, 21, 30]
REBALANCES = {"weekly": "W-MON", "biweekly": "2W-MON"}
BASE = (21, "weekly")            # the config from step 2
SKIP, N_SIDE = 1, 5
FEE, SLIPPAGE = 0.0005, 0.0003
DATA_DIR = "data"
# ----------------------------------------------------------------------------

ohlcv = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet")
funding = pd.read_parquet(f"{DATA_DIR}/funding.parquet")
universe = pd.read_csv(f"{DATA_DIR}/universe.csv", parse_dates=["month"])
universe = universe.dropna(subset=["symbols"])
universe["symbols"] = universe["symbols"].str.split(",")
uni_by_month = {r.month.to_period("M"): set(r.symbols)
                for r in universe.itertuples()}

close = ohlcv.pivot_table(index="date", columns="symbol", values="close")
close.index = close.index.normalize()
rets = close.pct_change()

funding["day"] = funding["date"].dt.normalize()
fund_daily = (funding.groupby(["day", "symbol"])["funding_rate"]
              .sum().unstack()
              .reindex(index=close.index, columns=close.columns).fillna(0.0))


def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0


def maxdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


def run(lookback, rebal_key, exclude=frozenset(), use_funding=True):
    """One full backtest; returns stats + per-day series + weights."""
    mom = close.shift(SKIP) / close.shift(SKIP + lookback) - 1.0
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)

    rebal_dates = [d for d in pd.date_range(close.index[lookback + SKIP + 5],
                                            close.index[-1],
                                            freq=REBALANCES[rebal_key])
                   if d in close.index]

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current = pd.Series(0.0, index=close.columns)
    turnover = pd.Series(0.0, index=close.index)

    for d in close.index:
        if d in rebal_dates:
            allowed = uni_by_month.get(d.to_period("M") - 1)
            zd = z.loc[d].dropna()
            if allowed:
                zd = zd[zd.index.isin(allowed)]
            zd = zd[~zd.index.isin(exclude)]
            if len(zd) >= 2 * N_SIDE:
                target = pd.Series(0.0, index=close.columns)
                target[zd.nlargest(N_SIDE).index] = 0.5 / N_SIDE
                target[zd.nsmallest(N_SIDE).index] = -0.5 / N_SIDE
                turnover[d] = (target - current).abs().sum()
                current = target
        weights.loc[d] = current

    w_lag = weights.shift(1).fillna(0.0)
    gross = (w_lag * rets).sum(axis=1)
    costs = turnover * (FEE + SLIPPAGE)
    fpnl = -(w_lag * fund_daily).sum(axis=1) if use_funding else 0.0 * gross
    net = (gross - costs + fpnl).loc[rebal_dates[0]:]
    return {
        "net": net,
        "w_lag": w_lag.loc[net.index],
        "sharpe": sharpe(net),
        "total_%": 100 * ((1 + net).prod() - 1),
        "maxdd_%": 100 * maxdd(net),
    }


# ------------------------------- A. grid -------------------------------------
print("A. PARAMETER GRID (net Sharpe / total % / max DD %)")
print(f"{'':>10}" + "".join(f"{rk:>26}" for rk in REBALANCES))
for lb in LOOKBACKS:
    cells = []
    for rk in REBALANCES:
        r = run(lb, rk)
        tag = " <- base" if (lb, rk) == BASE else ""
        cells.append(f"{r['sharpe']:5.2f} / {r['total_%']:6.1f} / "
                     f"{r['maxdd_%']:6.1f}{tag}")
    print(f"lookback {lb:>2}: " + " | ".join(cells))

# ---------------------------- B/C/D. diagnostics -----------------------------
base = run(*BASE)
w = base["w_lag"]
long_pnl = (w.clip(lower=0) * rets.loc[w.index]).sum(axis=1)
short_pnl = (w.clip(upper=0) * rets.loc[w.index]).sum(axis=1)
print("\nB. LEG SPLIT (gross, base config)")
print(f"   long leg : {100 * ((1 + long_pnl).prod() - 1):7.1f}%  "
      f"(Sharpe {sharpe(long_pnl):.2f})")
print(f"   short leg: {100 * ((1 + short_pnl).prod() - 1):7.1f}%  "
      f"(Sharpe {sharpe(short_pnl):.2f})")

nf = run(*BASE, use_funding=False)
print("\nC. FUNDING ZEROED (base config)")
print(f"   net Sharpe {nf['sharpe']:.2f}, total {nf['total_%']:.1f}%, "
      f"maxDD {nf['maxdd_%']:.1f}%")

contrib = (w * rets.loc[w.index]).sum().sort_values(ascending=False)
top3 = contrib.head(3)
print("\nD. DROP TOP 3 CONTRIBUTORS")
print("   top contributors were: "
      + ", ".join(f"{s} (+{100 * v:.1f}%)" for s, v in top3.items()))
d3 = run(*BASE, exclude=frozenset(top3.index))
print(f"   without them: net Sharpe {d3['sharpe']:.2f}, "
      f"total {d3['total_%']:.1f}%, maxDD {d3['maxdd_%']:.1f}%")

print("""
INTERPRETATION GUIDE (written before running):
  A. PASS if most grid cells land ~0.6-1.2 Sharpe; FAIL if only the base
     cell works. Biweekly beating weekly = fee drag confirmed, prefer it.
  B. HEALTHY if both legs contribute; if one leg is ~all of it, the
     market-neutral framing needs a caveat in the writeup.
  C. PASS if Sharpe stays > ~0.5 with funding removed.
  D. PASS if Sharpe stays positive and above ~0.4 without the top 3.
Any single FAIL = investigate, not necessarily abandon. Two+ FAILs = the
signal is likely an artifact of this sample; do not build the live system.
""")
