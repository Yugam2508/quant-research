#!/usr/bin/env python3
"""
daily_pulse.py — daily market pulse run

Usage:
    python runs/daily_pulse.py
    python runs/daily_pulse.py --date 2024-12-01   # backfill a specific date
    python runs/daily_pulse.py --force              # bypass cache
    python runs/daily_pulse.py --dry-run            # skip Claude, print data only

Output:
    reports/daily/YYYY-MM-DD.md
"""

import argparse
import sys
import os
from datetime import date, datetime
from pathlib import Path

# make src importable when running from repo root or runs/
sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.ingestion import fetch_prices, TICKERS, TICKER_META
from src.signals import build_snapshot
from src.claude import generate_daily_pulse

console = Console()

REPORT_DIR = Path(__file__).parents[1] / "reports" / "daily"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Daily market pulse")
    p.add_argument("--date",    type=str,  default=None, help="As-of date (YYYY-MM-DD)")
    p.add_argument("--force",   action="store_true",     help="Force refresh data cache")
    p.add_argument("--dry-run", action="store_true",     help="Skip Claude, print signals only")
    return p.parse_args()


def print_snapshot_table(snapshot):
    """Pretty-print the signal snapshot to terminal."""
    table = Table(title="signal snapshot", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("ticker",   style="bold")
    table.add_column("1D %",     justify="right")
    table.add_column("5D %",     justify="right")
    table.add_column("20D %",    justify="right")
    table.add_column("vol",      justify="right")
    table.add_column("RSI",      justify="right")
    table.add_column("z-score",  justify="right")

    for ticker, row in snapshot.iterrows():
        def fmt_ret(v):
            if v is None or (isinstance(v, float) and v != v):
                return "[dim]—[/dim]"
            color = "green" if v > 0 else "red"
            return f"[{color}]{v:+.1f}[/{color}]"

        def fmt_rsi(v):
            if v is None or (isinstance(v, float) and v != v):
                return "[dim]—[/dim]"
            color = "red" if v > 70 else ("green" if v < 30 else "white")
            return f"[{color}]{v:.0f}[/{color}]"

        table.add_row(
            ticker,
            fmt_ret(row.get("ret_1d")),
            fmt_ret(row.get("ret_5d")),
            fmt_ret(row.get("ret_20d")),
            f"{row.get('vol_20d', 0):.1f}%",
            fmt_rsi(row.get("rsi_14")),
            f"{row.get('cs_zscore', 0):+.2f}",
        )

    console.print(table)


def save_report(content: str, as_of: date) -> Path:
    output_path = REPORT_DIR / f"{as_of.isoformat()}.md"
    output_path.write_text(content)
    return output_path


def main():
    args = parse_args()

    as_of = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else date.today()
    )

    console.print(Panel(
        f"[bold]daily market pulse[/bold]  •  {as_of.strftime('%A, %d %B %Y')}",
        border_style="cyan"
    ))

    # ── 1. fetch prices ───────────────────────────────────────────────────────
    console.print("\n[bold]1/3  fetching prices[/bold]")
    prices = fetch_prices(TICKERS, period="3mo", force_refresh=args.force)

    # ── 2. compute signals ────────────────────────────────────────────────────
    console.print("\n[bold]2/3  computing signals[/bold]")
    snapshot = build_snapshot(prices)
    print_snapshot_table(snapshot)

    if args.dry_run:
        console.print("\n[yellow]dry-run: skipping Claude call[/yellow]")
        return

    # ── 3. generate Claude memo ───────────────────────────────────────────────
    console.print("\n[bold]3/3  generating analysis[/bold]")
    report_md = generate_daily_pulse(snapshot, TICKER_META, as_of)

    # ── 4. save report ────────────────────────────────────────────────────────
    output_path = save_report(report_md, as_of)
    console.print(f"\n[bold green]✓ report saved:[/bold green] {output_path}")
    console.print("\n" + "─" * 60)
    console.print(report_md)
    console.print("─" * 60)


if __name__ == "__main__":
    main()
