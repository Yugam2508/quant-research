"""
STEP 4 — Iteration 2 (FINAL pre-registered iteration on this signal).

Changes vs iteration 1, chosen for structural reasons before seeing what
would have worked:
  - Universe: top 40 by rolling 30d median dollar volume (rebuilt here
    from ohlcv.parquet; no re-download needed)
  - Breadth: 8 names per side (16-position book) instead of 5
  - Weighting: inverse-volatility within each leg (calmer coins get more
    weight, wild ones less), hard cap 9% of gross per name
  - Same signal (21d momentum, skip 1), same weekly rebalance, gross 1x

DECISION RULE (fixed in advance):
  PASS  = base net Sharpe >= ~0.6  AND  drop-top-3 Sharpe >= ~0.4
  PASS -> gate cleared, build execution layer at half-gross sizing.
  FAIL -> stop iterating on this signal. Execution layer may still be
          built for paper trading; no real capital on this signal.
"""

import numpy as np
import pandas as pd

# ------------------------- pre-registered settings ---------------------------
N_UNIVERSE = 40
N_SIDE = 8
CAP = 0.09                 # max |weight| per name, as fraction of gross (=1.0)
VOL_WIN = 30               # days for the volatility estimate
LOOKBACKS = [14, 21, 30]   # grid re-check
REBALANCES = {"weekly": "W-MON", "biweekly": "2W-MON"}
BASE = (21, "weekly")
SKIP = 1
FEE, SLIPPAGE = 0.0005, 0.0003
DATA_DIR = "data"
# ------------------------------------------------------------------------------

ohlcv = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet")
funding = pd.read_parquet(f"{DATA_DIR}/funding.parquet")

close = ohlcv.pivot_table(index="date", columns="symbol", values="close")
close.index = close.index.normalize()
rets = close.pct_change()
vol = rets.rolling(VOL_WIN, min_periods=20).std()

funding["day"] = funding["date"].dt.normalize()
fund_daily = (funding.groupby(["day", "symbol"])["funding_rate"]
              .sum().unstack()
              .reindex(index=close.index, columns=close.columns).fillna(0.0))

# ---------------- rebuild point-in-time universe at top 40 -------------------
dv = ohlcv.pivot_table(index="date", columns="symbol", values="dollar_volume")
dv.index = dv.index.normalize()
med30 = dv.rolling(30, min_periods=20).median()
uni_by_month = {}
for month_end in med30.resample("ME").last().index:
    snap = med30.loc[:month_end]
    if len(snap) == 0:
        continue
    snap = snap.iloc[-1].dropna().sort_values(ascending=False)
    if len(snap) >= 2 * N_SIDE:
        uni_by_month[month_end.to_period("M")] = set(
            snap.head(N_UNIVERSE).index
        )


def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0


def maxdd(x):
    c = (1 + x).cumprod()
    return float((c / c.cummax() - 1).min())


def capped_inverse_vol(names, d, leg_gross):
    """Weights proportional to 1/vol, then capped at CAP and renormalized."""
    iv = 1.0 / vol.loc[d, list(names)]
    iv = iv.replace([np.inf, -np.inf], np.nan).dropna()
    if len(iv) == 0:
        return pd.Series(dtype=float)
    w = leg_gross * iv / iv.sum()
    for _ in range(5):  # cap-and-redistribute until stable
        over = w > CAP
        if not over.any():
            break
        excess = (w[over] - CAP).sum()
        w[over] = CAP
        under = ~over
        if w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
        else:
            break
    return w


def run(lookback, rebal_key, exclude=frozenset(), use_funding=True):
    mom = close.shift(SKIP) / close.shift(SKIP + lookback) - 1.0
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)

    rebal_dates = [d for d in pd.date_range(close.index[max(lookback, VOL_WIN)
                                                        + SKIP + 5],
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
            zd = zd[zd.index.isin(vol.loc[d].dropna().index)]
            if allowed:
                zd = zd[zd.index.isin(allowed)]
            zd = zd[~zd.index.isin(exclude)]
            if len(zd) >= 2 * N_SIDE:
                target = pd.Series(0.0, index=close.columns)
                wl = capped_inverse_vol(zd.nlargest(N_SIDE).index, d, 0.5)
                ws = capped_inverse_vol(zd.nsmallest(N_SIDE).index, d, 0.5)
                target[wl.index] = wl
                target[ws.index] = -ws
                turnover[d] = (target - current).abs().sum()
                current = target
        weights.loc[d] = current

    w_lag = weights.shift(1).fillna(0.0)
    gross = (w_lag * rets).sum(axis=1)
    costs = turnover * (FEE + SLIPPAGE)
    fpnl = -(w_lag * fund_daily).sum(axis=1) if use_funding else 0.0 * gross
    net = (gross - costs + fpnl).loc[rebal_dates[0]:]
    return {"net": net, "w_lag": w_lag.loc[net.index],
            "sharpe": sharpe(net),
            "total_%": 100 * ((1 + net).prod() - 1),
            "maxdd_%": 100 * maxdd(net)}


# --------------------------------- A. grid -----------------------------------
print("A. PARAMETER GRID, iteration 2 (net Sharpe / total % / max DD %)")
print(f"{'':>10}" + "".join(f"{rk:>26}" for rk in REBALANCES))
for lb in LOOKBACKS:
    cells = []
    for rk in REBALANCES:
        r = run(lb, rk)
        tag = " <- base" if (lb, rk) == BASE else ""
        cells.append(f"{r['sharpe']:5.2f} / {r['total_%']:6.1f} / "
                     f"{r['maxdd_%']:6.1f}{tag}")
    print(f"lookback {lb:>2}: " + " | ".join(cells))

base = run(*BASE)
w = base["w_lag"]

# ------------------------------ B. leg split ---------------------------------
long_pnl = (w.clip(lower=0) * rets.loc[w.index]).sum(axis=1)
short_pnl = (w.clip(upper=0) * rets.loc[w.index]).sum(axis=1)
print("\nB. LEG SPLIT (gross, base config)")
print(f"   long leg : {100 * ((1 + long_pnl).prod() - 1):7.1f}%  "
      f"(Sharpe {sharpe(long_pnl):.2f})")
print(f"   short leg: {100 * ((1 + short_pnl).prod() - 1):7.1f}%  "
      f"(Sharpe {sharpe(short_pnl):.2f})")

# ---------------------------- C. funding zeroed ------------------------------
nf = run(*BASE, use_funding=False)
print("\nC. FUNDING ZEROED (base config)")
print(f"   net Sharpe {nf['sharpe']:.2f}, total {nf['total_%']:.1f}%, "
      f"maxDD {nf['maxdd_%']:.1f}%")

# ----------------------------- D. drop top 3 ---------------------------------
contrib = (w * rets.loc[w.index]).sum().sort_values(ascending=False)
top3 = contrib.head(3)
print("\nD. DROP TOP 3 CONTRIBUTORS (of a 16-position book)")
print("   top contributors were: "
      + ", ".join(f"{s} (+{100 * v:.1f}%)" for s, v in top3.items()))
d3 = run(*BASE, exclude=frozenset(top3.index))
print(f"   without them: net Sharpe {d3['sharpe']:.2f}, "
      f"total {d3['total_%']:.1f}%, maxDD {d3['maxdd_%']:.1f}%")

# ------------------------------- decision ------------------------------------
passed = base["sharpe"] >= 0.6 and d3["sharpe"] >= 0.4
print("\n" + "=" * 60)
print(f"BASE net Sharpe: {base['sharpe']:.2f}   "
      f"DROP-TOP-3 Sharpe: {d3['sharpe']:.2f}")
print("DECISION (rule fixed in advance): "
      + ("GATE CLEARED — build execution layer at half-gross sizing."
         if passed else
         "GATE FAILED — stop iterating this signal. Paper-trade only; "
         "no real capital."))
print("=" * 60)
print("This was the final pre-registered iteration. Either way, the next "
      "step is the execution layer;\nthe result only decides whether real "
      "capital is ever attached to this particular signal.")
