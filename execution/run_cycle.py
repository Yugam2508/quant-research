"""One full rebalance cycle, v2 — reconciliation in contracts, not dollars.

Run weekly. Safe to re-run: idempotent client order ids per (cycle, symbol)."""

import datetime
import config
from exchange import Exchange
from strategy import build_targets
from journal import Journal


def main():
    j = Journal()
    ex = Exchange()
    cycle = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d-%H%M"
    )

    # ------------------------- 1. risk guard --------------------------------
    if j.halted():
        j.event("SKIP", "system is halted; manual review required")
        return
    equity = ex.equity_usdt()
    init = j.initial_equity()
    if init is None:
        j.equity(equity)
        init = equity
        j.event("INIT", f"initial equity {equity:.2f} USDT")
    if equity < init * (1 - config.KILL_DRAWDOWN):
        for sym, notional in ex.positions_notional().items():
            ex.market_order(sym, -notional, f"kill-{cycle}-{sym[:8]}")
        j.event(
            "HALT",
            f"kill switch: equity {equity:.2f} < "
            f"{init * (1 - config.KILL_DRAWDOWN):.2f}",
        )
        return

    # ----------------------- 2. target portfolio ----------------------------
    targets = build_targets(ex)
    current = ex.positions_notional()
    expected_contracts = ex.positions_contracts()
    j.event("CYCLE", f"{cycle}: {len(targets)} targets, equity {equity:.2f}")

    # -------------------- 3. diff and execute -------------------------------
    for i, sym in enumerate(sorted(set(targets) | set(current))):
        tgt = targets.get(sym, 0.0)
        cur = current.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < config.MIN_ORDER_USDT:
            continue
        intended = ex.mid_price(sym)
        j.intent(cycle, sym, tgt, intended, delta)
        client_id = f"qr-{cycle}-{i}"
        try:
            fill_price, fill_qty, fill_usdt = ex.market_order(
                sym, delta, client_id
            )
            if fill_price:
                j.fill(cycle, sym, client_id, fill_price, fill_usdt)
                expected_contracts[sym] = (
                    expected_contracts.get(sym, 0.0) + fill_qty
                )
        except Exception as e:
            j.event("ORDER_FAIL", f"{sym}: {type(e).__name__}: {e}")

    # ------------------------- 4. reconcile (contracts) ---------------------
    problems = ex.reconcile_contracts(
        {s: q for s, q in expected_contracts.items() if abs(q) > 0}
    )
    if problems:
        j.event("RECON_MISMATCH", "; ".join(problems))
    else:
        j.event("RECON_OK", "exchange matches expected positions (contracts)")

    # ------------------------ 5. equity snapshot ----------------------------
    j.equity(ex.equity_usdt())
    j.event("DONE", cycle)


if __name__ == "__main__":
    main()
