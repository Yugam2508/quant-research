"""Strategy: iteration-2 signal -> target portfolio (spec: signals_live +
portfolio). Same math as step4_iteration2.py, applied to live data."""

import numpy as np
import pandas as pd
import config


def build_targets(ex):
    """Returns dict symbol -> signed target notional in USDT."""
    candidates = ex.top_perp_candidates(config.N_CANDIDATES)
    days = max(config.LOOKBACK + config.SKIP, config.VOL_WIN) + 40
    raw = ex.daily_closes(candidates, days)

    closes = pd.DataFrame(
        {s: pd.Series({r[0]: r[1] for r in rows}) for s, rows in raw.items()}
    ).sort_index()
    dvol = pd.DataFrame(
        {s: pd.Series({r[0]: r[2] for r in rows}) for s, rows in raw.items()}
    ).sort_index()

    # point-in-time universe: top N by 30d median dollar volume, as of now
    med = dvol.rolling(30, min_periods=20).median().iloc[-1].dropna()
    universe = med.nlargest(config.N_UNIVERSE).index

    rets = closes.pct_change()
    vol = rets.rolling(config.VOL_WIN, min_periods=20).std().iloc[-1]
    mom = (
        closes.shift(config.SKIP)
        / closes.shift(config.SKIP + config.LOOKBACK)
        - 1.0
    ).iloc[-1]

    z = mom[universe].dropna()
    z = z[~z.index.isin(getattr(config, "EXCLUDE_SYMBOLS", set()))]
    z = z[vol[z.index].notna() & (vol[z.index] > 0)]
    z = (z - z.mean()) / z.std()
    if len(z) < 2 * config.N_SIDE:
        raise RuntimeError(f"Universe too small after filters: {len(z)}")

    gross_usdt = config.CAPITAL_USDT * config.GROSS

    def leg(names, sign):
        iv = 1.0 / vol[names]
        w = 0.5 * iv / iv.sum()
        for _ in range(5):  # cap and redistribute
            over = w > config.PER_NAME_CAP
            if not over.any():
                break
            excess = (w[over] - config.PER_NAME_CAP).sum()
            w[over] = config.PER_NAME_CAP
            rest = ~over
            if w[rest].sum() > 0:
                w[rest] += excess * w[rest] / w[rest].sum()
        return {s: sign * float(wi) * gross_usdt for s, wi in w.items()}

    targets = leg(z.nlargest(config.N_SIDE).index, +1)
    targets.update(leg(z.nsmallest(config.N_SIDE).index, -1))

    # risk assertions before anything is sent to the exchange
    net = sum(targets.values())
    gross = sum(abs(v) for v in targets.values())
    assert abs(net) <= config.NET_BAND * gross, f"net {net:.1f} outside band"
    assert all(
        abs(v) <= (config.PER_NAME_CAP * 1.01) * gross for v in targets.values()
    ), "per-name cap violated"
    return targets
