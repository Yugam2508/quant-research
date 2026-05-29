#!/usr/bin/env python3
"""
factor_report.py — weekly factor research memo

Runs a rolling momentum/value factor backtest and has Claude
write a research memo on what's working.

Usage:
    python runs/factor_report.py

Output:
    reports/weekly/factor_YYYY-WW.md

TODO: implement backtest engine in src/backtest/engine.py
      and wire up here (scaffold ready, see src/backtest/)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from rich.console import Console
console = Console()


def main():
    console.print("[yellow]factor_report: coming soon[/yellow]")
    console.print("See src/backtest/ for the engine scaffold.")
    console.print("Wire it up here once you've implemented your factor logic.")


if __name__ == "__main__":
    main()
