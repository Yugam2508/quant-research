# Live Deployment Spec — Market-Neutral Crypto Long/Short

Extension of `quant-research`. Deliverable is a live-deployed, market-neutral
cross-sectional strategy with full execution attribution (backtest vs. live).
Profitability is reported honestly with error bars; it is not the success metric.

**Governing rule: no real capital touches the exchange until every item in
Section 3 (Validation Gate) passes.**

---

## 1. Strategy Definition

- **Style:** Cross-sectional long/short on perpetual futures, dollar-neutral.
- **Universe:** Top 20–30 perps by 30-day median volume on the chosen exchange
  (Binance or Bybit). Recompute universe monthly. Exclude anything with
  median daily volume < $50M or spread > 5bps — at our size liquidity isn't
  the issue, but illiquid names have junk price data and erratic funding.
- **Signal:** Start with the existing cross-sectional momentum z-score from the
  signal library (e.g. 7–30d lookback, skip most recent 1d to avoid reversal
  contamination). One signal first. Composites only after the single-signal
  version survives the gate.
- **Portfolio construction:** Long top quintile, short bottom quintile,
  equal-weight within legs, both legs sized to equal dollar exposure.
  Net exposure ≈ 0 by construction.
- **Rebalance frequency:** Weekly (default). Fee math: ~10bps taker per side
  means a full rebalance costs up to ~20bps × turnover. Daily rebalancing at
  this fee tier will almost certainly eat the edge. Confirm with the turnover
  analysis in §3.4 before locking this in.
- **Leverage:** Effective gross leverage 1x per leg maximum (i.e. $500 capital
  → ≤ $250 long + $250 short notional... conservative; margin allows far more.
  We deliberately do not use it). Maintain margin ratio such that liquidation
  price is > 40% away on every position at all times.

## 2. Risk Controls (hard-coded, not discretionary)

These bound the maximum loss by construction:

1. **Capital cap:** $300–500 in a dedicated sub-account. Nothing else in it.
2. **Kill switch:** If account equity drops 15% below initial, the system
   flattens all positions and halts. Restart requires manual review + writeup
   of what happened. Max loss is therefore ~$75 on $500, plus gap risk.
3. **Per-name cap:** No single position > 15% of gross exposure.
4. **Net exposure band:** |net| must stay < 10% of gross; if drift pushes it
   outside the band, rebalance the hedge at next check, don't wait a week.
5. **Funding guard:** If aggregate funding cost run-rate exceeds 20% annualized
   against the book, flag for review (funding can silently dominate P&L on
   crowded shorts).
6. **Staleness guard:** If price data is stale > N minutes or the exchange API
   fails reconciliation twice, halt new orders (positions stay hedged).
7. **No manual overrides mid-drawdown.** Parameter changes only at scheduled
   monthly reviews, logged with rationale. (This is what "systematic" means,
   and it's the discipline interviewers probe for.)

## 3. Validation Gate (must pass before deployment)

### 3.1 Cost-realistic walk-forward backtest
Extend the existing JAX walk-forward harness with:
- Taker fees both sides (use actual exchange tier, ~10bps).
- **Funding rates:** historical funding is downloadable per perp; apply it to
  each leg. This is not optional — funding is a first-order cost for
  short legs and the most common reason paper crypto L/S dies live.
- Slippage: at our size, model as ½ spread + 2bps buffer per fill.
- Execution lag: signals computed on close of bar t, fills at open of t+1.

**Pass criterion:** positive net Sharpe over ≥ 2 years of walk-forward,
including 2022-style down regimes, with all costs on.

### 3.2 Overfitting discipline
- Fix the signal family and hyperparameter grid *in writing before* running
  the sweep. Log every configuration tested.
- Report the deflated Sharpe ratio (Bailey & López de Prado) given the number
  of trials — n trials means the best backtest Sharpe is inflated; correct
  for it.
- Hold out the most recent 6 months entirely until the final configuration
  is frozen. One look. If it fails there, the strategy goes back to research,
  not to a second peek.

### 3.3 Robustness checks
- Perturb lookback ±30%: Sharpe should degrade gracefully, not cliff.
  A strategy that only works at lookback=14 exactly is an artifact.
- Drop the top 3 contributing names: still positive?
- Long leg and short leg P&L separately: if all P&L is in one leg, the
  "market-neutral" framing is misleading and funding/borrow on the other
  leg needs scrutiny.
- Subperiod analysis: bull/bear/chop. Doesn't need to win everywhere;
  needs to not be one regime's fluke.

### 3.4 Turnover / capacity sanity
- Compute per-rebalance turnover; net-of-fee edge must survive at
  weekly frequency. If not, lengthen holding period or widen the
  entry/exit hysteresis (trade only when rank change is material).

### 3.5 Paper trading burn-in (2–4 weeks)
Run the full live stack against the exchange **testnet or paper mode**:
same code path, real-time data, simulated fills. Purpose is to shake out
infrastructure bugs (reconciliation, restarts, API edge cases), not to
validate the signal. Zero code changes between paper and live except the
API endpoint.

## 4. Execution Architecture

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐
│ Data layer   │→ │ Signal engine │→ │ Target portfolio│
│ (ccxt OHLCV, │   │ (existing     │   │ (quintile L/S, │
│  funding)    │   │  z-score lib) │   │  risk checks)  │
└─────────────┘   └──────────────┘   └───────┬───────┘
                                              ↓
┌─────────────┐   ┌──────────────┐   ┌───────────────┐
│ State store  │← │ Reconciler    │← │ Order manager  │
│ (SQLite:     │   │ (exchange vs  │   │ (diff current →│
│  fills, eq., │   │  local state) │   │  target, place │
│  intents)    │   │               │   │  orders)       │
└─────────────┘   └──────────────┘   └───────────────┘
```

Components (Python, lives in `quant-research/execution/`):

- **connector.py** — ccxt wrapper for the chosen exchange. Retry with
  exponential backoff; idempotent order placement via client order IDs.
- **signals_live.py** — imports the existing signal library; computes
  target ranks from latest data. No new signal logic here.
- **portfolio.py** — converts ranks → target weights → target notional per
  name; applies every §2 risk check; emits a target position vector.
- **oms.py** — diffs target vs. current positions, generates minimal order
  list, places orders, records **intended price at decision time** and
  fill price per order (slippage data is unrecoverable retroactively —
  this logging exists from day one).
- **reconcile.py** — after each cycle, pulls exchange positions/balances and
  asserts they match local state; mismatch → halt flag.
- **riskguard.py** — kill switch, exposure bands, staleness checks. Runs
  every cycle and independently on a faster heartbeat.
- **journal (SQLite)** — every intent, order, fill, funding payment, equity
  snapshot, and halt event, timestamped. This database *is* the project's
  headline artifact.
- **report.py** — weekly: live vs. backtest attribution (signal P&L, fees,
  funding, slippage decomposition), equity curve, exposure history.
  Publishes to the existing GitHub Pages dashboard.

**Scheduling:** weekly rebalance + hourly risk heartbeat. GitHub Actions cron
is workable (secrets for API keys, keys restricted to trade-only — **withdrawals
disabled** — and IP-whitelisted if the exchange supports it for Actions).
A $5 VPS is the more robust alternative if Actions latency/reliability annoys.
Since rebalances are weekly and the risk guard tolerates hour-level granularity,
uptime requirements are mild — that's a deliberate design choice.

**Security non-negotiables:** API key has trade permission only, never
withdrawal; keys in environment/secrets, never in the repo; sub-account
isolates the capital.

## 5. Reporting Standards (what goes on the CV / dashboard)

- Live equity curve with backtest overlay for the same period.
- Slippage: distribution of (fill − intent) in bps, by name and by side.
- Cost decomposition: gross signal P&L − fees − funding − slippage = net.
- Alpha reported **with standard error** and the sentence "N months is not a
  statistically significant sample" attached. This sentence is a feature.
- Every halt/incident written up briefly (what tripped, why, fix).

## 6. Timeline

1. Week 1–2: backtest extensions (funding, fees, slippage) + validation gate.
2. Week 3: execution layer build.
3. Week 4–6: paper burn-in.
4. Month 2–4: live, $300–500. Monthly review checkpoints.
5. End: writeup — backtest-vs-live attribution as the centerpiece.
