"""
universe.py — cross-asset watchlist

Edit UNIVERSE freely. Each entry is a dict with:
  ticker   : yfinance symbol
  name     : human-readable name
  sector   : broad grouping for the analysis prompt
"""

UNIVERSE = [
    # US equities — large cap
    {"ticker": "SPY",  "name": "S&P 500 ETF",         "sector": "US Equity"},
    {"ticker": "QQQ",  "name": "Nasdaq 100 ETF",       "sector": "US Equity"},
    {"ticker": "IWM",  "name": "Russell 2000 ETF",     "sector": "US Equity"},
    # sectors
    {"ticker": "XLK",  "name": "Technology SPDR",      "sector": "US Sector"},
    {"ticker": "XLF",  "name": "Financials SPDR",      "sector": "US Sector"},
    {"ticker": "XLE",  "name": "Energy SPDR",          "sector": "US Sector"},
    {"ticker": "XLV",  "name": "Healthcare SPDR",      "sector": "US Sector"},
    # international
    {"ticker": "EEM",  "name": "Emerging Markets ETF", "sector": "International"},
    {"ticker": "EFA",  "name": "EAFE ETF",             "sector": "International"},
    # fixed income
    {"ticker": "TLT",  "name": "20Y Treasury ETF",     "sector": "Fixed Income"},
    {"ticker": "HYG",  "name": "High Yield ETF",       "sector": "Fixed Income"},
    # commodities & alternatives
    {"ticker": "GLD",  "name": "Gold ETF",             "sector": "Commodities"},
    {"ticker": "USO",  "name": "Oil ETF",              "sector": "Commodities"},
    {"ticker": "BTC-USD", "name": "Bitcoin",           "sector": "Crypto"},
]

# convenience: flat list of tickers
TICKERS = [x["ticker"] for x in UNIVERSE]

# lookup by ticker
TICKER_META = {x["ticker"]: x for x in UNIVERSE}
