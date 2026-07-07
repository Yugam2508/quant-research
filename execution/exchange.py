"""Exchange connector, v2.

Changes vs v1 (see journal incidents 1-4):
  - Binance demo-trading endpoint (legacy futures testnet deprecated)
  - adjustForTimeDifference: absorbs local clock skew (-1021 errors)
  - reconciliation now compares CONTRACT QUANTITIES, not USDT notional.
    Notional = contracts x mark price moves with the market every second,
    so dollar-based reconciliation raises false alarms on every price tick.
    Contracts only change when an order fills - that's the invariant to check.
"""

import os
import time
import ccxt
import config


class Exchange:
    def __init__(self):
        key, secret = os.environ.get("BINANCE_KEY"), os.environ.get(
            "BINANCE_SECRET"
        )
        if not key or not secret:
            raise SystemExit(
                "Set BINANCE_KEY and BINANCE_SECRET environment variables "
                "(demo keys from demo.binance.com API management)."
            )
        self.x = ccxt.binanceusdm(
            {"apiKey": key, "secret": secret, "enableRateLimit": True,
             "options": {"adjustForTimeDifference": True}}
        )
        if config.TESTNET:
            self.x.enable_demo_trading(True)
        self.x.load_markets()

    # ------------------------------ market data ------------------------------
    def top_perp_candidates(self, n):
        perps = [
            m["symbol"]
            for m in self.x.markets.values()
            if m.get("swap") and m.get("linear") and m.get("quote") == "USDT"
            and m.get("active", True)
        ]
        tickers = self.x.fetch_tickers(perps)
        ranked = sorted(
            tickers.items(),
            key=lambda kv: float(kv[1].get("quoteVolume") or 0),
            reverse=True,
        )
        return [s for s, _ in ranked[:n]]

    def daily_closes(self, symbols, days):
        out = {}
        since = self.x.milliseconds() - days * 86_400_000
        for s in symbols:
            try:
                rows = self.x.fetch_ohlcv(s, "1d", since=since, limit=days + 5)
                out[s] = [(r[0], r[4], r[4] * r[5]) for r in rows]
            except Exception:
                continue
        return out

    def mid_price(self, symbol):
        t = self.x.fetch_ticker(symbol)
        bid, ask = t.get("bid"), t.get("ask")
        if bid and ask:
            return (bid + ask) / 2
        return t.get("last")

    # ------------------------------ account ----------------------------------
    def equity_usdt(self):
        bal = self.x.fetch_balance()
        return float(bal["info"].get("totalMarginBalance", bal["USDT"]["total"]))

    def positions_notional(self):
        """symbol -> signed USDT notional at current mark (for order sizing)."""
        pos = {}
        for p in self.x.fetch_positions():
            amt = float(p.get("contracts") or 0) * (
                1 if p.get("side") == "long" else -1
            )
            if amt != 0:
                price = float(p.get("markPrice") or p.get("entryPrice") or 0)
                pos[p["symbol"]] = amt * price
        return pos

    def positions_contracts(self):
        """symbol -> signed contract quantity (for reconciliation)."""
        pos = {}
        for p in self.x.fetch_positions():
            amt = float(p.get("contracts") or 0) * (
                1 if p.get("side") == "long" else -1
            )
            if amt != 0:
                pos[p["symbol"]] = amt
        return pos

    # ------------------------------- orders ----------------------------------
    def market_order(self, symbol, notional_usdt, client_id):
        """Signed notional -> market order.

        Returns (fill_price, signed_filled_contracts, signed_filled_notional),
        or (None, 0.0, 0.0) if nothing was sent.
        """
        price = self.mid_price(symbol)
        amount = abs(notional_usdt) / price
        amount = float(self.x.amount_to_precision(symbol, amount))
        if amount <= 0:
            return None, 0.0, 0.0
        side = "buy" if notional_usdt > 0 else "sell"
        sign = 1 if side == "buy" else -1
        try:
            self.x.set_leverage(3, symbol)
        except Exception:
            pass  # some symbols reject leverage changes; order attempt decides
        o = self.x.create_order(
            symbol, "market", side, amount,
            params={"newClientOrderId": client_id[:36]},
        )
        time.sleep(0.5)
        filled = self.x.fetch_order(o["id"], symbol)
        fp = float(filled.get("average") or filled.get("price") or price)
        qty = float(filled.get("filled") or amount)
        return fp, sign * qty, sign * qty * fp

    # ----------------------------- reconciliation ----------------------------
    def reconcile_contracts(self, local, rel_tol=0.02):
        """Compare exchange contract quantities vs expected. A position is
        flagged only if it differs by more than rel_tol of its own size
        (plus a tiny absolute floor for near-zero cases)."""
        live = self.positions_contracts()
        problems = []
        for sym in set(live) | set(local):
            a, b = live.get(sym, 0.0), local.get(sym, 0.0)
            tol = max(rel_tol * max(abs(a), abs(b)), 1e-6)
            if abs(a - b) > tol:
                problems.append(
                    f"{sym}: exchange {a:.6f} vs local {b:.6f} contracts"
                )
        return problems
