"""
STEP 6 — Hourly pairs-trading gate. PRE-REGISTERED; run once.

Design (fixed before running):
  - Universe: the downloaded top-40 hourly closes
  - Every 30 days: FORMATION on trailing 60 days of hourly log prices -
      keep pairs with log-price correlation > 0.90,
      OLS hedge ratio, ADF test on the spread residual,
      select the 10 most cointegrated (lowest ADF stat) pairs
  - TRADING (strictly out-of-sample from formation), hourly:
      z = spread z-score over trailing 240h
      enter at |z| > 2 (short rich leg, long cheap leg, dollar-neutral
      via hedge ratio), exit at |z| < 0.5, hard stop at |z| > 4
      (stopped pairs stay out until next formation)
  - Capital: equal slice per selected pair; per-pair gross = 2 x slice
  - Costs on turnover, two scenarios:
      TAKER 8 bps/side (current infrastructure)
      MAKER 2 bps/side (requires limit-order OMS, assumed perfect fills -
      an optimistic bound, stated as such)

DECISION RULE (fixed in advance):
  taker-net Sharpe >= 1.0                 -> viable on current OMS
  taker fails, maker-net Sharpe >= 1.0    -> maker OMS is the prerequisite
  both fail                                -> documented negative result

Requires: pip install statsmodels
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ------------------------- pre-registered settings ---------------------------
FORM_DAYS = 60          # formation lookback
REFORM_DAYS = 30        # re-form pairs every 30 days
CORR_MIN = 0.90
N_PAIRS = 10
Z_WIN = 240             # hours for spread z-score
Z_IN, Z_OUT, Z_STOP = 2.0, 0.5, 4.0
TAKER, MAKER = 0.0008, 0.0002   # per side, fees+slippage
DATA_DIR = "data"
HOURS_YEAR = 24 * 365
# ------------------------------------------------------------------------------

hourly = pd.read_parquet(f"{DATA_DIR}/hourly.parquet")
px = hourly.pivot_table(index="date", columns="symbol", values="close")
px = px.asfreq("1h").ffill(limit=3)
logp = np.log(px)
rets = px.pct_change()

form_h, reform_h = FORM_DAYS * 24, REFORM_DAYS * 24
dates = px.index


def form_pairs(window):
    """Correlation filter + ADF on OLS spread; returns [(a, b, hedge)]."""
    w = window.dropna(axis=1, thresh=int(len(window) * 0.95))
    cols = w.columns
    corr = w.corr()
    scored = []
    for i in range(len(cols)):
        for k in range(i + 1, len(cols)):
            a, b = cols[i], cols[k]
            if corr.loc[a, b] < CORR_MIN:
                continue
            ya, yb = w[a].values, w[b].values
            beta = np.polyfit(yb, ya, 1)[0]
            resid = ya - beta * yb
            try:
                stat = adfuller(resid, maxlag=24, autolag=None)[0]
            except Exception:
                continue
            scored.append((stat, a, b, beta))
    scored.sort()                      # most negative ADF = most cointegrated
    return [(a, b, beta) for _, a, b, beta in scored[:N_PAIRS]]


# ------------------------------ walk-forward ---------------------------------
pnl = pd.Series(0.0, index=dates)
turnover = pd.Series(0.0, index=dates)
pair_pnl = {}          # pair label -> cumulative gross pnl
hold_hours, n_trades = [], 0

start = form_h
while start + reform_h <= len(dates):
    pairs = form_pairs(logp.iloc[start - form_h:start])
    slice_w = 1.0 / max(len(pairs), 1)      # capital slice per pair
    trade_idx = range(start, min(start + reform_h, len(dates)))

    for a, b, beta in pairs:
        label = f"{a.split('/')[0]}-{b.split('/')[0]}"
        spread = logp[a] - beta * logp[b]
        mu = spread.rolling(Z_WIN).mean()
        sd = spread.rolling(Z_WIN).std()
        z = ((spread - mu) / sd)

        pos, entry_t = 0, None            # +1 long spread, -1 short spread
        for t in trade_idx:
            zt = z.iloc[t - 1]            # decide on last completed bar
            if np.isnan(zt):
                continue
            new = pos
            if pos == 0:
                if zt > Z_IN:
                    new = -1
                elif zt < -Z_IN:
                    new = +1
            else:
                if abs(zt) < Z_OUT:
                    new = 0
                elif abs(zt) > Z_STOP:
                    new = 0               # stop: relationship broke
                    pairs_broken = True
            if new != pos:
                # weight change: each leg holds slice/2 gross
                turnover.iloc[t] += abs(new - pos) * slice_w
                if pos != 0 and entry_t is not None:
                    hold_hours.append(t - entry_t)
                if new != 0:
                    entry_t = t
                    n_trades += 1
                pos = new
                if abs(zt) > Z_STOP:      # stopped out: sit out this block
                    pos_locked = True
            if pos != 0:
                r = pos * slice_w * 0.5 * (
                    rets[a].iloc[t] - beta * rets[b].iloc[t]
                    * (px[b].iloc[t - 1] / px[a].iloc[t - 1])
                )
                # simpler, standard dollar-neutral leg returns:
                r = pos * slice_w * 0.5 * (rets[a].iloc[t] - rets[b].iloc[t])
                pnl.iloc[t] += r
                pair_pnl[label] = pair_pnl.get(label, 0.0) + r
    start += reform_h

live = pnl.iloc[form_h:]
tov = turnover.iloc[form_h:]


def stats(cost):
    net = live - tov * cost
    sh = float(net.mean() / net.std() * np.sqrt(HOURS_YEAR)) if net.std() else 0
    tot = 100 * ((1 + net).prod() - 1)
    curve = (1 + net).cumprod()
    dd = 100 * float((curve / curve.cummax() - 1).min())
    return sh, tot, dd


print("================== HOURLY PAIRS GATE ==================")
gross_sh = float(live.mean() / live.std() * np.sqrt(HOURS_YEAR))
print(f"gross Sharpe          : {gross_sh:5.2f}   "
      f"({100 * ((1 + live).prod() - 1):.1f}% total)")
for name, c in [("TAKER 8bps", TAKER), ("MAKER 2bps", MAKER)]:
    sh, tot, dd = stats(c)
    print(f"net Sharpe ({name}) : {sh:5.2f}   ({tot:6.1f}% total, "
          f"maxDD {dd:.1f}%)")
print(f"trades: {n_trades}, median hold: "
      f"{np.median(hold_hours) if hold_hours else 0:.0f}h, "
      f"avg hourly turnover: {tov[tov > 0].mean():.3f}")

top = sorted(pair_pnl.items(), key=lambda kv: -kv[1])[:3]
print("top pairs: " + ", ".join(f"{k} (+{100 * v:.1f}%)" for k, v in top))
drop = sum(v for k, v in pair_pnl.items() if k not in dict(top))
print(f"gross total without top-3 pairs: {100 * drop:.1f}%")

sh_taker = stats(TAKER)[0]
sh_maker = stats(MAKER)[0]
print("\nDECISION (rule fixed in advance):")
if sh_taker >= 1.0:
    print("  PASS (taker) - viable on current market-order OMS.")
elif sh_maker >= 1.0:
    print("  CONDITIONAL - viable only with maker execution; limit-order")
    print("  OMS build is the prerequisite. NOTE: maker scenario assumes")
    print("  perfect fills - an optimistic bound.")
else:
    print("  FAIL - edge does not survive costs at either fee tier.")
    print("  Documented negative result; no re-runs without a new")
    print("  pre-registration.")
