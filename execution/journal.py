"""Journal: every intent, fill, equity snapshot and event, timestamped.
This database is the project's primary artifact (spec sections 4 and 5)."""

import sqlite3
import time
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS intents(
  ts REAL, cycle TEXT, symbol TEXT, target_usdt REAL,
  intended_price REAL, order_usdt REAL);
CREATE TABLE IF NOT EXISTS fills(
  ts REAL, cycle TEXT, symbol TEXT, client_id TEXT,
  fill_price REAL, fill_usdt REAL);
CREATE TABLE IF NOT EXISTS equity(ts REAL, equity_usdt REAL);
CREATE TABLE IF NOT EXISTS events(ts REAL, kind TEXT, detail TEXT);
"""


class Journal:
    def __init__(self, path=config.DB_PATH):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def intent(self, cycle, symbol, target, price, order_usdt):
        self.db.execute(
            "INSERT INTO intents VALUES(?,?,?,?,?,?)",
            (time.time(), cycle, symbol, target, price, order_usdt),
        )
        self.db.commit()

    def fill(self, cycle, symbol, client_id, price, usdt):
        self.db.execute(
            "INSERT INTO fills VALUES(?,?,?,?,?,?)",
            (time.time(), cycle, symbol, client_id, price, usdt),
        )
        self.db.commit()

    def equity(self, value):
        self.db.execute(
            "INSERT INTO equity VALUES(?,?)", (time.time(), value)
        )
        self.db.commit()

    def event(self, kind, detail=""):
        self.db.execute(
            "INSERT INTO events VALUES(?,?,?)", (time.time(), kind, detail)
        )
        self.db.commit()
        print(f"[{kind}] {detail}")

    def initial_equity(self):
        row = self.db.execute(
            "SELECT equity_usdt FROM equity ORDER BY ts LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def halted(self):
        row = self.db.execute(
            "SELECT COUNT(*) FROM events WHERE kind='HALT'"
        ).fetchone()
        return row[0] > 0
