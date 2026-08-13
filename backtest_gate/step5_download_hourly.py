"""
STEP 5 — Download ~2 years of HOURLY closes from Binance (public data).

Feeds the pairs-trading gate (step 6). ~40 symbols x ~17,500 hourly bars.
Run time: roughly 20-30 minutes at polite rate limits.

Same survivorship caveat as step 1: candidates ranked by today's volume.
"""

import os
import ccxt
import pandas as pd

YEARS_BACK = 2.0
N_CANDIDATES = 40
DATA_DIR = "data"

ex = ccxt.binanceusdm({"enableRateLimit": True})
now_ms = ex.milliseconds()
since0 = now_ms - int(YEARS_BACK * 365 * 24 * 3600 * 1000)

print("Selecting candidates by today's volume...")
ex.load_markets()
perps = [m["symbol"] for m in ex.markets.values()
         if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
         and m.get("active", True)]
tickers = ex.fetch_tickers(perps)
ranked = sorted(tickers.items(),
                key=lambda kv: float(kv[1].get("quoteVolume") or 0),
                reverse=True)
candidates = [s for s, _ in ranked[:N_CANDIDATES]]

print(f"Downloading 1h bars for {len(candidates)} symbols "
      f"(~{int(YEARS_BACK * 8760)} bars each)...")
frames = []
for i, sym in enumerate(candidates, 1):
    rows, since = [], since0
    while True:
        batch = ex.fetch_ohlcv(sym, "1h", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        since = batch[-1][0] + 1
    if rows:
        df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "close", "vol"])
        frames.append(pd.DataFrame(
            {"ts": df["ts"], "close": df["close"], "symbol": sym}
        ))
    print(f"  [{i:>2}/{len(candidates)}] {sym}: {len(rows)} bars")

hourly = pd.concat(frames, ignore_index=True)
hourly["date"] = pd.to_datetime(hourly["ts"], unit="ms")
os.makedirs(DATA_DIR, exist_ok=True)
hourly.to_parquet(f"{DATA_DIR}/hourly.parquet", index=False)
print(f"\nSaved {DATA_DIR}/hourly.parquet ({len(hourly):,} rows).")
