"""
universe.py — cross-asset watchlist v2

Covers: US equities, sectors, international, fixed income,
        commodities futures, crypto, and SGX (Singapore).
"""

UNIVERSE = [
    # ── US equities ───────────────────────────────────────────────
    {"ticker": "SPY",    "name": "S&P 500 ETF",          "sector": "US Equity"},
    {"ticker": "QQQ",    "name": "Nasdaq 100 ETF",        "sector": "US Equity"},
    {"ticker": "IWM",    "name": "Russell 2000 ETF",      "sector": "US Equity"},
    # ── US sectors ────────────────────────────────────────────────
    {"ticker": "XLK",    "name": "Technology SPDR",       "sector": "US Sector"},
    {"ticker": "XLF",    "name": "Financials SPDR",       "sector": "US Sector"},
    {"ticker": "XLE",    "name": "Energy SPDR",           "sector": "US Sector"},
    {"ticker": "XLV",    "name": "Healthcare SPDR",       "sector": "US Sector"},
    # ── international ─────────────────────────────────────────────
    {"ticker": "EEM",    "name": "Emerging Markets ETF",  "sector": "International"},
    {"ticker": "EFA",    "name": "EAFE ETF",              "sector": "International"},
    # ── SGX / Asia ────────────────────────────────────────────────
    {"ticker": "ES3.SI", "name": "STI ETF (Singapore)",   "sector": "Asia"},
    {"ticker": "D05.SI", "name": "DBS Group",             "sector": "Asia"},
    {"ticker": "O39.SI", "name": "OCBC Bank",             "sector": "Asia"},
    # ── fixed income ──────────────────────────────────────────────
    {"ticker": "TLT",    "name": "20Y Treasury ETF",      "sector": "Fixed Income"},
    {"ticker": "HYG",    "name": "High Yield ETF",        "sector": "Fixed Income"},
    # ── commodities futures ───────────────────────────────────────
    {"ticker": "GC=F",   "name": "Gold Futures",          "sector": "Commodities"},
    {"ticker": "CL=F",   "name": "Crude Oil Futures",     "sector": "Commodities"},
    {"ticker": "SI=F",   "name": "Silver Futures",        "sector": "Commodities"},
    # ── crypto ────────────────────────────────────────────────────
    {"ticker": "BTC-USD","name": "Bitcoin",               "sector": "Crypto"},
    {"ticker": "ETH-USD","name": "Ethereum",              "sector": "Crypto"},
    {"ticker": "SOL-USD","name": "Solana",                "sector": "Crypto"},
]

TICKERS     = [x["ticker"] for x in UNIVERSE]
TICKER_META = {x["ticker"]: x for x in UNIVERSE}

# group by sector for structured prompts
SECTORS = {}
for item in UNIVERSE:
    SECTORS.setdefault(item["sector"], []).append(item)
