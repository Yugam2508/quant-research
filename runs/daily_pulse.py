#!/usr/bin/env python3
"""
daily_pulse.py — daily market pulse run (v2)

Pipeline:
  1. Fetch prices (yfinance + parquet cache)
  2. Compute signals (momentum, vol, RSI, z-score)
  3. Classify regime (router)
  4. Inject prior context (steerer)
  5. Generate analysis (Gemini analyst)
  6. Evaluate quality (judge LLM)
  7. Save annotated report + data.json for dashboard

Usage:
  python runs/daily_pulse.py
  python runs/daily_pulse.py --dry-run     # skip all LLM calls
  python runs/daily_pulse.py --force       # bypass data cache
  python runs/daily_pulse.py --no-judge    # skip judge (saves one API call)
"""

import argparse
import sys
import os
import json
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.ingestion import fetch_prices, TICKERS, TICKER_META
from src.signals import build_snapshot
from src.claude import (
    generate_daily_pulse, classify_regime,
    build_context, format_context_block,
    evaluate_report, annotate_report,
)

console = Console()

REPORT_DIR = Path(__file__).parents[1] / "reports" / "daily"
DOCS_DIR   = Path(__file__).parents[1] / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Daily market pulse v2")
    p.add_argument("--date",     type=str,  default=None)
    p.add_argument("--force",    action="store_true")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--no-judge", action="store_true")
    return p.parse_args()


def print_snapshot_table(snapshot):
    table = Table(title="signal snapshot", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("ticker",  style="bold")
    table.add_column("sector",  style="dim")
    table.add_column("1D %",    justify="right")
    table.add_column("5D %",    justify="right")
    table.add_column("20D %",   justify="right")
    table.add_column("vol",     justify="right")
    table.add_column("RSI",     justify="right")
    table.add_column("z",       justify="right")

    for ticker, row in snapshot.iterrows():
        meta = TICKER_META.get(ticker, {})

        def fmt(v, pct=True):
            if v is None or (isinstance(v, float) and v != v): return "[dim]—[/dim]"
            c = "green" if v > 0 else "red"
            return f"[{c}]{v:+.1f}[/{c}]" if pct else f"{v:.1f}"

        def fmt_rsi(v):
            if v is None or (isinstance(v, float) and v != v): return "[dim]—[/dim]"
            c = "red" if v > 70 else ("green" if v < 30 else "white")
            return f"[{c}]{v:.0f}[/{c}]"

        table.add_row(
            ticker,
            meta.get("sector", ""),
            fmt(row.get("ret_1d")),
            fmt(row.get("ret_5d")),
            fmt(row.get("ret_20d")),
            f"{row.get('vol_20d', 0):.1f}%",
            fmt_rsi(row.get("rsi_14")),
            f"{row.get('cs_zscore', 0):+.2f}",
        )
    console.print(table)


def save_data_json(snapshot, regime, evaluation, as_of):
    """Save structured data.json for the static dashboard."""
    rows = []
    for ticker, row in snapshot.iterrows():
        meta = TICKER_META.get(ticker, {})
        rows.append({
            "ticker":    ticker,
            "name":      meta.get("name", ticker),
            "sector":    meta.get("sector", "Unknown"),
            **{k: (None if (isinstance(v, float) and v != v) else round(float(v), 2))
               for k, v in row.items()},
        })

    payload = {
        "as_of":      as_of.isoformat(),
        "generated":  datetime.now().isoformat(),
        "regime":     regime,
        "evaluation": evaluation,
        "signals":    rows,
    }

    out = DOCS_DIR / "data.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    console.print(f"[dim]data.json saved → {out}[/dim]")


def main():
    args = parse_args()
    as_of = (datetime.strptime(args.date, "%Y-%m-%d").date()
             if args.date else date.today())

    console.print(Panel(
        f"[bold]daily market pulse v2[/bold]  •  {as_of.strftime('%A, %d %B %Y')}",
        border_style="cyan"
    ))

    # 1. fetch
    console.print("\n[bold]1/5  fetching prices[/bold]")
    prices = fetch_prices(TICKERS, period="3mo", force_refresh=args.force)

    # 2. signals
    console.print("\n[bold]2/5  computing signals[/bold]")
    snapshot = build_snapshot(prices)
    print_snapshot_table(snapshot)

    if args.dry_run:
        console.print("\n[yellow]dry-run: skipping all LLM calls[/yellow]")
        save_data_json(snapshot, None, None, as_of)
        return

    # 3. route
    console.print("\n[bold]3/5  classifying regime[/bold]")
    regime = classify_regime(snapshot)

    # 4. steer
    console.print("\n[bold]4/5  building context[/bold]")
    context_raw = build_context(n_days=3)
    context_block = format_context_block(context_raw)

    # 5. generate
    console.print("\n[bold]5/5  generating analysis[/bold]")
    report_md = generate_daily_pulse(
        snapshot, TICKER_META, as_of,
        regime=regime,
        prior_context=context_block,
    )

    # 6. judge
    evaluation = None
    if not args.no_judge:
        console.print("\n[bold]+   judging report quality[/bold]")
        evaluation = evaluate_report(report_md, snapshot)
        report_md  = annotate_report(report_md, evaluation)

    # 7. save
    report_path = REPORT_DIR / f"{as_of.isoformat()}.md"
    report_path.write_text(report_md)
    save_data_json(snapshot, regime, evaluation, as_of)

    console.print(f"\n[bold green]✓ report saved:[/bold green] {report_path}")
    console.print("\n" + "─" * 60)
    console.print(report_md)
    console.print("─" * 60)


if __name__ == "__main__":
    main()
