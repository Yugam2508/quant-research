# Incident Log — Live Execution System

Every halt, failure, or unexpected behavior of the paper-trading system,
per spec §5. Newest at the bottom.

---

**#1 — 2026-07-07 — Binance futures testnet deprecated**
First connection attempt failed: ccxt raised `NotSupported` because Binance
retired the legacy futures testnet (testnet.binancefuture.com) in favor of
the new Demo Trading environment. Fix: switched connector from
`set_sandbox_mode(True)` to `enable_demo_trading(True)` with API keys from
demo.binance.com. Lesson: exchange infrastructure changes under you; the
paper burn-in caught a breaking API migration on day one.

**#2 — 2026-07-07 — Clock skew rejected signed requests**
All private API calls failed with error -1021 ("Timestamp 1000ms ahead of
server time"). Local Windows clock had drifted ahead of Binance's server
clock; Binance rejects skewed timestamps as an anti-replay measure. Fix:
enabled ccxt option `adjustForTimeDifference`, which measures and corrects
the offset automatically. Lesson: production trading systems never trust
the local clock.

**#3 — 2026-07-07 — Dust threshold silently skipped the entire book**
First "successful" cycle placed zero orders: config assumed $500 capital at
half-gross across 16 positions (~$15/position), all below the $25 minimum
order threshold, so every order was skipped as dust. Fix for paper trading:
capital set to match the $5,000 demo account. Real finding: a 16-name
diversified book has a capacity floor (~$2–3k minimum) — the
diversification added to fix concentration risk is in direct tension with
small capital. This constraint would have blocked live deployment at the
originally planned size.

**#4 — 2026-07-07 — False reconciliation alarm from notional comparison**
Reconciler flagged RDNT: exchange -236.2 vs local -225.0 USDT. Not a
position break — the reconciler compared dollar notional, which is
contracts × mark price and therefore moves with the market and differs
between fill price and current mark. Fix: reconcile in contract quantities
(the invariant that only changes on fills) with a 2% relative tolerance.
Lesson: reconcile in units that don't move with the market, or real breaks
hide in false-alarm noise.

**#5 — 2026-07-07 — Persistent PERCENT_PRICE rejection on illiquid demo symbol**
SIREN/USDT market orders rejected every cycle with -4131: the demo
environment's order book for the symbol is too thin, so the best
counterparty price violates Binance's price-band filter. Fix: added a
config-level exclusion list; the ranking fills the slot with the
next-ranked name. Noted for production: naked market orders are fragile on
thin books; a limit-order-with-retry policy is the proper long-term fix.
