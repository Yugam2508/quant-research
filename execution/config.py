"""All settings in one place. Change nothing mid-drawdown (spec section 2.7)."""

TESTNET = True            # HARD RULE: stays True. Gate failed -> no real capital.
CAPITAL_USDT = 5000.0      # notional account size the sizing logic assumes

# --- strategy (iteration-2 configuration, paper-traded for infrastructure) ---
N_CANDIDATES = 60
N_UNIVERSE = 40
N_SIDE = 8
LOOKBACK = 21
SKIP = 1
VOL_WIN = 30
GROSS = 0.5               # half-gross: total notional = 50% of capital
PER_NAME_CAP = 0.09       # max |weight| per name, fraction of gross book

EXCLUDE_SYMBOLS = {"SIREN/USDT:USDT"}   # demo book too thin, -4131 rejects

# --- risk limits (spec section 2) ---
KILL_DRAWDOWN = 0.15      # flatten + halt if equity falls 15% from start
NET_BAND = 0.10           # |net exposure| must stay under 10% of gross
STALENESS_MIN = 120       # halt new orders if price data older than this (min)
MIN_ORDER_USDT = 25.0     # skip dust orders below this notional

DB_PATH = "journal.db"
