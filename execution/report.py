"""Builds docs/live.html — a self-contained dashboard page for the paper
system: equity curve, current book, slippage analysis, cost summary, and
event/incident feed. Run after each cycle (the Action does this).

Reads journal.db; fetches the current book from the exchange if API keys
are present, otherwise skips that section (so it also runs offline)."""

import json
import sqlite3
import datetime

import config

db = sqlite3.connect(config.DB_PATH)


def rows(q):
    return db.execute(q).fetchall()


# ------------------------------ equity curve --------------------------------
equity = [(datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"), v)
          for ts, v in rows("SELECT ts, equity_usdt FROM equity ORDER BY ts")]
init_eq = equity[0][1] if equity else 0.0
last_eq = equity[-1][1] if equity else 0.0
ret_pct = 100 * (last_eq / init_eq - 1) if init_eq else 0.0

# ------------------------------- slippage -----------------------------------
# join fills to intents on (cycle, symbol); cost in bps, signed so that
# positive = execution cost (filled worse than intended)
slip = rows("""
    SELECT f.symbol, i.order_usdt, i.intended_price, f.fill_price
    FROM fills f JOIN intents i ON f.cycle = i.cycle AND f.symbol = i.symbol
""")
slippage = []
for sym, order_usdt, intent, fill in slip:
    if not intent or not fill:
        continue
    sign = 1 if order_usdt > 0 else -1
    bps = sign * (fill - intent) / intent * 1e4
    slippage.append({"symbol": sym.split("/")[0], "bps": round(bps, 2),
                     "usdt": round(abs(order_usdt), 1)})
avg_slip = (sum(s["bps"] for s in slippage) / len(slippage)) if slippage else 0
n_fills = len(slippage)

# -------------------------------- events -------------------------------------
events = [
    {"time": datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
     "kind": k, "detail": d}
    for ts, k, d in rows(
        "SELECT ts, kind, detail FROM events ORDER BY ts DESC LIMIT 50")
]

# --------------------------- current book (optional) -------------------------
book = []
try:
    from exchange import Exchange
    ex = Exchange()
    for sym, notional in sorted(ex.positions_notional().items(),
                                key=lambda kv: -abs(kv[1])):
        book.append({"symbol": sym.split("/")[0],
                     "side": "LONG" if notional > 0 else "SHORT",
                     "usdt": round(abs(notional), 1)})
except Exception as e:
    print(f"(book skipped: {type(e).__name__})")

gross = sum(p["usdt"] for p in book)
net = sum(p["usdt"] if p["side"] == "LONG" else -p["usdt"] for p in book)

payload = {
    "generated": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    "equity": equity, "init": init_eq, "last": last_eq,
    "ret_pct": round(ret_pct, 2),
    "slippage": slippage, "avg_slip": round(avg_slip, 2), "n_fills": n_fills,
    "events": events, "book": book,
    "gross": round(gross, 1), "net": round(net, 1),
}

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Paper-Trading Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;
      padding:0 1rem;color:#1a1a2e;background:#fafafa}
 h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:2rem}
 .banner{background:#fff3cd;border:1px solid #ffec99;border-radius:8px;
         padding:.7rem 1rem;font-size:.9rem}
 .cards{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}
 .card{background:#fff;border:1px solid #e5e5e5;border-radius:10px;
       padding:.8rem 1.2rem;min-width:130px}
 .card .v{font-size:1.4rem;font-weight:700}
 .card .l{font-size:.75rem;color:#777;text-transform:uppercase}
 table{border-collapse:collapse;width:100%;font-size:.85rem;background:#fff}
 th,td{border:1px solid #e5e5e5;padding:.4rem .6rem;text-align:left}
 th{background:#f0f0f5} .neg{color:#c0392b}.pos{color:#27ae60}
 .muted{color:#888;font-size:.8rem}
</style></head><body>
<h1>Live Paper-Trading Dashboard</h1>
<div class="banner"><b>Paper trading (Binance demo, no real capital).</b>
The pre-registered research gate was failed on single-name concentration
risk; this system runs to validate execution infrastructure. Returns shown
are simulated and statistically insignificant by construction.</div>
<div class="cards">
 <div class="card"><div class="v" id="c-eq"></div><div class="l">Equity USDT</div></div>
 <div class="card"><div class="v" id="c-ret"></div><div class="l">Return since start</div></div>
 <div class="card"><div class="v" id="c-gross"></div><div class="l">Gross exposure</div></div>
 <div class="card"><div class="v" id="c-net"></div><div class="l">Net exposure</div></div>
 <div class="card"><div class="v" id="c-slip"></div><div class="l">Avg slippage (bps)</div></div>
</div>
<h2>Equity curve</h2><canvas id="chart" height="90"></canvas>
<h2>Current book</h2><table id="book"><tr><th>Symbol</th><th>Side</th>
<th>Notional (USDT)</th></tr></table>
<h2>Fill-level slippage (intended vs filled)</h2>
<table id="slip"><tr><th>Symbol</th><th>Order size</th><th>Cost (bps)</th></tr></table>
<h2>Event log</h2>
<table id="ev"><tr><th>Time</th><th>Kind</th><th>Detail</th></tr></table>
<p class="muted" id="gen"></p>
<script>
const D = __DATA__;
document.getElementById("c-eq").textContent = D.last.toFixed(2);
const r = document.getElementById("c-ret");
r.textContent = (D.ret_pct>=0?"+":"")+D.ret_pct+"%";
r.className = "v "+(D.ret_pct>=0?"pos":"neg");
document.getElementById("c-gross").textContent = D.gross||"–";
document.getElementById("c-net").textContent = D.net||"–";
document.getElementById("c-slip").textContent = D.avg_slip;
document.getElementById("gen").textContent =
  "Generated "+D.generated+" · "+D.n_fills+" fills recorded";
new Chart(document.getElementById("chart"),{type:"line",
 data:{labels:D.equity.map(e=>e[0]),
       datasets:[{data:D.equity.map(e=>e[1]),borderColor:"#4059ad",
                  backgroundColor:"rgba(64,89,173,.08)",fill:true,
                  pointRadius:2,tension:.2,label:"Equity (USDT)"}]},
 options:{plugins:{legend:{display:false}},
          scales:{x:{ticks:{maxTicksLimit:8}}}}});
function fill(id, rows, cols, cls){const t=document.getElementById(id);
 rows.forEach(o=>{const tr=t.insertRow();
  cols.forEach(c=>{const td=tr.insertCell();td.textContent=o[c];
   if(cls&&c===cls&&parseFloat(o[c])>0)td.className="neg";
   if(cls&&c===cls&&parseFloat(o[c])<0)td.className="pos";});});}
fill("book", D.book, ["symbol","side","usdt"]);
fill("slip", D.slippage, ["symbol","usdt","bps"], "bps");
fill("ev", D.events, ["time","kind","detail"]);
</script></body></html>"""

import os
os.makedirs("../docs", exist_ok=True)
with open("../docs/live.html", "w", encoding="utf-8") as f:
    f.write(HTML.replace("__DATA__", json.dumps(payload)))
print(f"Wrote docs/live.html  (equity {last_eq:.2f}, {n_fills} fills, "
      f"{len(events)} events)")
