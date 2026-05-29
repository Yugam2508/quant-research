# quant-research

A quantitative research framework with Claude-powered market analysis. Pulls market data, computes signals, and generates natural-language research memos automatically.

## what it does

| run | schedule | description |
|-----|----------|-------------|
| `daily_pulse` | daily | EOD prices → momentum/vol signals → Claude market summary |
| `factor_report` | weekly | Factor backtest → Claude research memo |
| `risk_snapshot` | weekly | Drawdown/VaR/correlation → Claude risk narrative |
| `earnings_digest` | on-trigger | Earnings data → Claude 1-page brief |

## quick start

```bash
# install dependencies
pip install -r requirements.txt

# set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# run the daily market pulse
python runs/daily_pulse.py

# output lands in reports/daily/YYYY-MM-DD.md
```

## project structure

```
quant-research/
├── data/
│   ├── raw/          # cached parquet files from yfinance
│   └── processed/    # cleaned, feature-engineered data
├── src/
│   ├── ingestion/    # data fetchers (yfinance, FRED)
│   ├── signals/      # factor & signal library
│   ├── backtest/     # vectorised backtester
│   └── claude/       # AI analysis layer
├── runs/             # scheduled entry-point scripts
├── reports/
│   ├── daily/        # daily pulse markdown reports
│   └── weekly/       # weekly factor & risk reports
├── notebooks/        # exploratory research
└── tests/
```

## universe

Default universe is a cross-asset watchlist defined in `src/ingestion/universe.py`. Edit freely — the pipeline adapts automatically.

## scheduling

### cron (local)
```bash
# run daily at 6pm SGT (10:00 UTC)
0 10 * * 1-5 cd /path/to/quant-research && python runs/daily_pulse.py
```

### GitHub Actions
See `.github/workflows/daily_pulse.yml` — runs on schedule and commits the report to the repo automatically.

## adding a new signal

1. Add a function to `src/signals/factors.py`
2. Register it in `src/signals/__init__.py`
3. It will be picked up automatically by the pulse run

## dependencies

- `yfinance` — market data (free, no API key)
- `pandas` / `numpy` — data wrangling
- `anthropic` — Claude API for analysis
- `rich` — pretty terminal output
- `pyarrow` — parquet caching
