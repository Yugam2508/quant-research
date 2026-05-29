"""
metrics.py — performance attribution and statistics

All the standard quant performance metrics, computed properly.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    total_return:    float
    ann_return:      float
    ann_vol:         float
    sharpe:          float
    max_drawdown:    float
    calmar:          float
    win_rate:        float
    avg_win:         float
    avg_loss:        float
    profit_factor:   float


def compute_metrics(daily_returns: np.ndarray) -> PerformanceReport:
    r = daily_returns
    ann_ret  = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(252)
    sharpe   = ann_ret / (ann_vol + 1e-9)
    total    = (1 + r).prod() - 1

    # max drawdown
    cum      = (1 + r).cumprod()
    peak     = np.maximum.accumulate(cum)
    dd       = (cum - peak) / peak
    max_dd   = dd.min()

    calmar   = ann_ret / (abs(max_dd) + 1e-9)
    wins     = r[r > 0]
    losses   = r[r < 0]
    win_rate = len(wins) / len(r)
    pf       = wins.sum() / (abs(losses.sum()) + 1e-9)

    return PerformanceReport(
        total_return=round(total * 100, 2),
        ann_return=round(ann_ret * 100, 2),
        ann_vol=round(ann_vol * 100, 2),
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd * 100, 2),
        calmar=round(calmar, 3),
        win_rate=round(win_rate * 100, 2),
        avg_win=round(wins.mean() * 100, 3) if len(wins) else 0,
        avg_loss=round(losses.mean() * 100, 3) if len(losses) else 0,
        profit_factor=round(pf, 3),
    )


def print_report(name: str, metrics: PerformanceReport):
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    table = Table(title=f"[bold]{name}[/bold]", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="dim")
    table.add_column("value",  justify="right")

    def color(v, good_positive=True):
        if good_positive:
            return f"[green]{v}[/green]" if v > 0 else f"[red]{v}[/red]"
        else:
            return f"[green]{v}[/green]" if v < 0 else f"[red]{v}[/red]"

    table.add_row("total return",    color(metrics.total_return) + "%")
    table.add_row("ann. return",     color(metrics.ann_return) + "%")
    table.add_row("ann. volatility", f"{metrics.ann_vol}%")
    table.add_row("sharpe ratio",    color(metrics.sharpe))
    table.add_row("max drawdown",    color(metrics.max_drawdown, good_positive=False) + "%")
    table.add_row("calmar ratio",    color(metrics.calmar))
    table.add_row("win rate",        f"{metrics.win_rate}%")
    table.add_row("avg win",         f"{metrics.avg_win}%")
    table.add_row("avg loss",        f"{metrics.avg_loss}%")
    table.add_row("profit factor",   color(metrics.profit_factor - 1))

    console.print(table)
