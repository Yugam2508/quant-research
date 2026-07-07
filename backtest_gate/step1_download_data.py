"""
STEP 1 — Download historical data from Binance (public data, no account needed).

What this does, in plain language:
  1. Finds all USDT-margined perpetual futures on Binance.
  2. Keeps the ~60 biggest by trading volume today (our candidate pool).
  3. Downloads 2 years of daily prices for each.
  4. Downloads 2 years of funding-rate history for each.
  5. Builds a month-by-month "universe": the top 25 most-traded coins
     *as of each month in the past* (so we don't cheat by using
     today's knowledge about which coins became big — that would be
     'survivorship bias').
  6. Saves everything into the data/ folder.

Run time: roughly 10–20 minutes (we deliberately go slow to respect
Binance's rate limits). You can walk away while it runs.

Honest caveat (also printed at the end): the candidate pool is chosen by
*today's* volume, so a coin that was big in 2024 but died since won't be
included. This slightly flatters the backtest. Acceptable for a first gate
check; noted for the writeup.
"""

import time
import ccxt
import pandas as pd

# ----------------------------- settings ------------------------------------
YEARS_BACK = 2.2          # a bit more than 2 years of history
N_CANDIDATES = 60         # coins we download data for
N_UNIVERSE = 25           # coins actually tradeable each month
DATA_DIR = "data"
# ----------------------------------------------------------------------------

ex = ccxt.binanceusdm({"enableRateLimit": True})  # Binance futures, public API

now_ms = ex.milliseconds()
since_ms = now_ms - int(YEARS_BACK * 365 * 24 * 60 * 60 * 1000)

print("Loading market list...")
ex.load_markets()
perps = [
    m for m in ex.markets.values()
    if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
    and m.get("active", True)
]
print(f"Found {len(perps)} active USDT perpetuals.")

print("Ranking by today's volume to pick candidate pool...")
tickers = ex.fetch_tickers([m["symbol"] for m in perps])


def _qvol(t):
    v = t.get("quoteVolume")
    return float(v) if v is not None else 0.0


ranked = sorted(tickers.items(), key=lambda kv: _qvol(kv[1]), reverse=True)
candidates = [sym for sym, _ in ranked[:N_CANDIDATES]]
print(f"Candidate pool ({len(candidates)}): {', '.join(candidates[:10])} ...")

# ------------------------- daily price download -----------------------------
print("\nDownloading daily prices (this is the slow part)...")
all_ohlcv = []
for i, sym in enumerate(candidates, 1):
    rows, since = [], since_ms
    while True:
        batch = ex.fetch_ohlcv(sym, timeframe="1d", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        since = batch[-1][0] + 1
    if rows:
        df = pd.DataFrame(
            rows, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["symbol"] = sym
        all_ohlcv.append(df)
    print(f"  [{i:>2}/{len(candidates)}] {sym}: {len(rows)} days")

ohlcv = pd.concat(all_ohlcv, ignore_index=True)
ohlcv["date"] = pd.to_datetime(ohlcv["ts"], unit="ms")
ohlcv["dollar_volume"] = ohlcv["close"] * ohlcv["volume"]

# ------------------------- funding-rate download ----------------------------
print("\nDownloading funding-rate history...")
all_funding = []
for i, sym in enumerate(candidates, 1):
    rows, since = [], since_ms
    while True:
        try:
            batch = ex.fetch_funding_rate_history(sym, since=since, limit=1000)
        except Exception as e:  # some symbols may lack history; skip cleanly
            print(f"  [{i:>2}] {sym}: skipped ({type(e).__name__})")
            batch = []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        since = batch[-1]["timestamp"] + 1
        time.sleep(0.2)
    if rows:
        df = pd.DataFrame(
            {
                "ts": [r["timestamp"] for r in rows],
                "funding_rate": [float(r["fundingRate"]) for r in rows],
            }
        )
        df["symbol"] = sym
        all_funding.append(df)
    print(f"  [{i:>2}/{len(candidates)}] {sym}: {len(rows)} funding records")

funding = pd.concat(all_funding, ignore_index=True)
funding["date"] = pd.to_datetime(funding["ts"], unit="ms")

# --------------------- point-in-time monthly universe -----------------------
print("\nBuilding point-in-time monthly universe...")
px = ohlcv.pivot_table(index="date", columns="symbol", values="dollar_volume")
med30 = px.rolling(30, min_periods=20).median()

universe_rows = []
for month_end in med30.resample("ME").last().index:
    snapshot = med30.loc[:month_end].iloc[-1].dropna().sort_values(
        ascending=False
    )
    top = snapshot.head(N_UNIVERSE).index.tolist()
    universe_rows.append({"month": month_end, "symbols": ",".join(top)})
universe = pd.DataFrame(universe_rows)

# ------------------------------- save ---------------------------------------
import os

os.makedirs(DATA_DIR, exist_ok=True)
ohlcv.to_parquet(f"{DATA_DIR}/ohlcv.parquet", index=False)
funding.to_parquet(f"{DATA_DIR}/funding.parquet", index=False)
universe.to_csv(f"{DATA_DIR}/universe.csv", index=False)

print(
    f"\nDone. Saved:\n"
    f"  {DATA_DIR}/ohlcv.parquet    ({len(ohlcv):,} rows)\n"
    f"  {DATA_DIR}/funding.parquet  ({len(funding):,} rows)\n"
    f"  {DATA_DIR}/universe.csv     ({len(universe)} monthly universes)\n\n"
    f"Caveat for the writeup: candidate pool uses today's volume ranking,\n"
    f"which introduces mild survivorship bias. Fine for a gate check."
)
