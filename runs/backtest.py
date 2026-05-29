#!/usr/bin/env python3
"""
backtest.py — differentiable portfolio optimisation backtest
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

import jax.numpy as jnp
from rich.console import Console
from rich.panel import Panel

from src.ingestion import fetch_prices, TICKERS, TICKER_META
from src.strategies.trainer import walk_forward_backtest, TrainConfig
from src.strategies.metrics import compute_metrics, print_report

console = Console()

RESULTS_DIR = Path(__file__).parents[1] / "reports" / "backtest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_signal_matrix(prices_window):
    tickers = prices_window.columns.tolist()
    signals = []
    for t in tickers:
        s = prices_window[t]
        ret1  = float(s.pct_change(1).iloc[-1]  or 0)
        ret5  = float(s.pct_change(5).iloc[-1]  or 0)
        ret20 = float(s.pct_change(20).iloc[-1] or 0)
        v     = float(s.pct_change().std() * np.sqrt(252) or 0)
        delta = s.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-9)
        r14   = float(100 - 100 / (1 + rs))
        rets  = s.pct_change(20)
        zs    = float((rets.iloc[-1] - rets.mean()) / (rets.std() + 1e-9))
        for val in [ret1, ret5, ret20, v, r14, zs]:
            signals.append(0.0 if (val != val) else val)

    arr = np.array(signals, dtype=np.float32)
    std = arr.std() + 1e-6
    return jnp.array((arr - arr.mean()) / std)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="deep", choices=["deep", "linear"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--period", default="2y")
    p.add_argument("--tc-bps", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()

    console.print(Panel(
        f"[bold]differentiable portfolio backtest[/bold]  •  model: {args.model}  •  epochs: {args.epochs}",
        border_style="purple"
    ))

    # 1. fetch
    console.print("\n[bold]1/4  fetching prices[/bold]")
    universe = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "EEM", "TLT", "GLD", "BTC-USD", "ETH-USD"]
    prices = fetch_prices(universe, period=args.period)
    # keep tickers with at least 50% data coverage, forward-fill gaps
    prices = prices.dropna(thresh=int(len(prices) * 0.5), axis=1)
    prices = prices.ffill().dropna()
    console.print(f"  using {prices.shape[1]} assets × {len(prices)} days")

    # 2. config
    config = TrainConfig(
        n_epochs=args.epochs,
        tc_bps=args.tc_bps,
    )

    # 3. walk-forward
    console.print("\n[bold]2/4  running walk-forward backtest[/bold]")
    results = walk_forward_backtest(
        prices=prices,
        signal_fn=build_signal_matrix,
        config=config,
        model_type=args.model,
    )

    # 4. performance
    console.print("\n[bold]3/4  performance attribution[/bold]")
    strat_metrics = compute_metrics(results.oos_returns)
    bench_metrics = compute_metrics(results.benchmark_returns)

    print_report(f"signal network ({args.model})", strat_metrics)
    print_report("equal-weight benchmark", bench_metrics)

    sr_diff = strat_metrics.sharpe - bench_metrics.sharpe
    color = "green" if sr_diff > 0 else "red"
    console.print(f"\n[bold]sharpe edge:[/bold] [{color}]{sr_diff:+.3f}[/{color}]")

    console.print("\n[bold]walk-forward OOS sharpe by window:[/bold]")
    for i, (ts, os) in enumerate(zip(results.train_sharpes, results.oos_sharpes)):
        date = results.weight_history[i]["date"]
        c = "green" if os > 0 else "red"
        console.print(f"  {str(date)[:10]}  train: {ts:+.2f}  oos: [{c}]{os:+.2f}[/{c}]")

    # 5. save
    console.print("\n[bold]4/4  saving results[/bold]")
    output = {
        "run_date":      datetime.now().isoformat(),
        "model":         args.model,
        "n_assets":      int(prices.shape[1]),
        "n_days":        int(len(prices)),
        "tickers":       prices.columns.tolist(),
        "config": {
            "epochs":       args.epochs,
            "tc_bps":       args.tc_bps,
            "train_window": config.train_window,
            "test_window":  config.test_window,
        },
        "strategy":      {k: float(v) for k, v in vars(strat_metrics).items()},
        "benchmark":     {k: float(v) for k, v in vars(bench_metrics).items()},
        "sharpe_edge":   float(sr_diff),
        "oos_sharpes":   [float(x) for x in results.oos_sharpes],
        "train_sharpes": [float(x) for x in results.train_sharpes],
        "oos_returns":   [float(x) for x in results.oos_returns],
        "bench_returns": [float(x) for x in results.benchmark_returns],
        "dates":         [str(d) for d in results.oos_dates],
        "window_dates":  [str(w["date"]) for w in results.weight_history],
    }

    fname = RESULTS_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    fname.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    console.print(f"  saved → {fname}")

    docs_out = Path(__file__).parents[1] / "docs" / "backtest.json"
    docs_out.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    console.print(f"  dashboard → {docs_out}")


if __name__ == "__main__":
    main()
