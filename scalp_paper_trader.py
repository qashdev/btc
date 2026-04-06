"""
Scalping Paper Trader — Gold & Bitcoin
========================================
Runs the 5-signal scalping strategy in real-time using yfinance.
All trades are PAPER (no real money). Results saved to CSV live.

Run:   python scalp_paper_trader.py
Stop:  Ctrl+C  (prints final summary on exit)

Dependencies:
    pip install yfinance pandas numpy tabulate colorama
"""

import csv
import os
import sys
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ─────────────────────────────────────────────
# CONFIG  (edit freely)
# ─────────────────────────────────────────────
ASSET            = "BTC"       # "GOLD" or "BTC"
GOLD_TICKER      = "GC=F"
BTC_TICKER       = "BTC-USD"

# Signal params (match backtest)
EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.5
MIN_SCORE        = 2

# Trade params
TAKE_PROFIT_PCT  = 0.0025       # 0.25%
STOP_LOSS_PCT    = 0.0008       # 0.08%
MAX_HOLD_BARS    = 24           # 24 × 5 min = 120 min max hold

# Session filter (UTC)
SESSION_START    = 7
SESSION_END      = 21

# Paper account
ACCOUNT_SIZE     = 10_000.0
MAX_DRAWDOWN_LIM = 0.05
RISK_SAFE        = 0.95
RISK_WARN        = 0.60
RISK_DANGER      = 0.10

# Loop
POLL_SECONDS     = 60           # check every 60 s (yfinance rate limits)
CANDLE_INTERVAL  = "5m"
WARMUP_BARS      = 60           # bars needed to compute indicators

CSV_FILE         = f"paper_trades_{ASSET.lower()}.csv"
LOG_FILE         = f"paper_trader_{ASSET.lower()}.log"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
@dataclass
class OpenTrade:
    entry_time: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_usd: float
    bars_held: int = 0

equity      = ACCOUNT_SIZE
peak_equity = ACCOUNT_SIZE
open_trade: OpenTrade | None = None
all_trades  = []   # list of dicts

# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
CSV_HEADERS = [
    "trade_num", "entry_time", "exit_time", "direction",
    "entry_price", "exit_price", "stop_loss", "take_profit",
    "position_usd", "pnl", "win", "exit_reason",
    "bars_held", "equity_after"
]

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
        log.info(f"📄 Created {CSV_FILE}")
    else:
        log.info(f"📄 Appending to {CSV_FILE}")

def append_csv(row: dict):
    with open(CSV_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)

# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
def fetch_candles(ticker: str, n_bars: int = 150) -> pd.DataFrame | None:
    """Fetch last n_bars of 5-min candles."""
    try:
        df = yf.download(ticker, period="5d", interval=CANDLE_INTERVAL,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna().tail(n_bars)
        return df
    except Exception as e:
        log.warning(f"Fetch error: {e}")
        return None

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ema_fast"]  = d["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    d["ema_slow"]  = d["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    d["ema_cross"] = np.sign(d["ema_fast"] - d["ema_slow"])

    delta    = d["Close"].diff()
    gain     = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss     = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    d["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    sma           = d["Close"].rolling(BB_PERIOD).mean()
    std           = d["Close"].rolling(BB_PERIOD).std()
    d["bb_upper"] = sma + BB_STD * std
    d["bb_lower"] = sma - BB_STD * std
    d["bb_pct"]   = (d["Close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)

    tp         = (d["High"] + d["Low"] + d["Close"]) / 3
    vol        = d["Volume"].replace(0, 1)
    d["vwap"]  = (tp * vol).rolling(VWAP_PERIOD).sum() / vol.rolling(VWAP_PERIOD).sum()
    d["vwap_dist"] = (d["Close"] - d["vwap"]) / d["vwap"]

    d["vol_avg"]   = d["Volume"].rolling(20).mean()
    d["vol_spike"] = d["Volume"] / (d["vol_avg"] + 1e-9)
    d["mom3"]      = d["Close"].pct_change(3)

    try:
        idx = d.index.tz_convert("UTC") if d.index.tzinfo else d.index.tz_localize("UTC")
    except Exception:
        idx = d.index
    d["hour"]       = idx.hour
    d["weekday"]    = idx.weekday
    d["in_session"] = ((d["hour"] >= SESSION_START) &
                       (d["hour"] < SESSION_END) &
                       (d["weekday"] < 5)).astype(int)
    return d.dropna()

# ─────────────────────────────────────────────
# SIGNAL SCORER
# ─────────────────────────────────────────────
def score_row(row, prev_cross):
    score_l = score_s = 0.0

    cross = float(row["ema_cross"])
    if prev_cross <= 0 and cross > 0:   score_l += 1
    elif prev_cross >= 0 and cross < 0: score_s += 1
    elif cross > 0:  score_l += 0.5
    elif cross < 0:  score_s += 0.5

    rsi = float(row["rsi"])
    if rsi < 35:   score_l += 1
    elif rsi > 65: score_s += 1

    vd = float(row["vwap_dist"])
    if vd < -0.001:  score_l += 1
    elif vd > 0.001: score_s += 1

    bp = float(row["bb_pct"])
    if bp < 0.2:   score_l += 1
    elif bp > 0.8: score_s += 1

    vs = float(row["vol_spike"])
    if vs >= VOLUME_SPIKE_MIN:
        if float(row["mom3"]) > 0: score_l += 1
        else:                      score_s += 1

    return score_l, score_s

# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
def dynamic_position(confidence: float) -> float:
    global equity, peak_equity
    current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
    dd_ratio   = min(current_dd / MAX_DRAWDOWN_LIM, 1.0)
    if dd_ratio < 0.5:
        scale    = dd_ratio / 0.5
        max_risk = RISK_SAFE + scale * (RISK_WARN - RISK_SAFE)
    else:
        scale    = (dd_ratio - 0.5) / 0.5
        max_risk = RISK_WARN + scale * (RISK_DANGER - RISK_WARN)
    b     = TAKE_PROFIT_PCT / STOP_LOSS_PCT
    kelly = max(0, (confidence * (b + 1) - 1) / b) * 0.5
    return max(equity * 0.01, min(equity * kelly, equity * max_risk))

# ─────────────────────────────────────────────
# TRADE MANAGEMENT
# ─────────────────────────────────────────────
def check_open_trade(current_price: float, now_str: str) -> bool:
    """Returns True if trade was closed."""
    global open_trade, equity, peak_equity

    if open_trade is None:
        return False

    open_trade.bars_held += 1
    t = open_trade
    hit_sl = hit_tp = time_exit = False
    pnl = 0.0

    if t.direction == "LONG":
        if current_price <= t.stop_loss:
            hit_sl = True
            pnl = -t.position_usd * STOP_LOSS_PCT
        elif current_price >= t.take_profit:
            hit_tp = True
            pnl = t.position_usd * TAKE_PROFIT_PCT
    else:  # SHORT
        if current_price >= t.stop_loss:
            hit_sl = True
            pnl = -t.position_usd * STOP_LOSS_PCT
        elif current_price <= t.take_profit:
            hit_tp = True
            pnl = t.position_usd * TAKE_PROFIT_PCT

    if t.bars_held >= MAX_HOLD_BARS and not (hit_sl or hit_tp):
        time_exit = True
        if t.direction == "LONG":
            pnl = (current_price - t.entry_price) / t.entry_price * t.position_usd
        else:
            pnl = (t.entry_price - current_price) / t.entry_price * t.position_usd

    if hit_sl or hit_tp or time_exit:
        equity      = max(0.0, equity + pnl)
        peak_equity = max(peak_equity, equity)

        reason = "TP" if hit_tp else ("SL" if hit_sl else "TIME")
        win    = pnl > 0
        color  = Fore.GREEN if win else Fore.RED
        trade_num = len(all_trades) + 1

        row = {
            "trade_num":    trade_num,
            "entry_time":   t.entry_time,
            "exit_time":    now_str,
            "direction":    t.direction,
            "entry_price":  round(t.entry_price, 5),
            "exit_price":   round(current_price, 5),
            "stop_loss":    round(t.stop_loss, 5),
            "take_profit":  round(t.take_profit, 5),
            "position_usd": round(t.position_usd, 2),
            "pnl":          round(pnl, 2),
            "win":          win,
            "exit_reason":  reason,
            "bars_held":    t.bars_held,
            "equity_after": round(equity, 2),
        }
        all_trades.append(row)
        append_csv(row)

        print(color + f"\n  {'✅ WIN' if win else '❌ LOSS'}  #{trade_num} {t.direction}  "
              f"entry={t.entry_price:.4f} → exit={current_price:.4f}  "
              f"PnL=${pnl:+.2f}  reason={reason}  equity=${equity:,.2f}")
        open_trade = None
        return True
    return False

def try_open_trade(row, prev_cross, current_price: float, now_str: str):
    global open_trade, equity

    if open_trade is not None:
        return  # already in trade

    score_l, score_s = score_row(row, prev_cross)

    # LONGS ONLY (match backtest default)
    if score_l < MIN_SCORE:
        return

    if int(row["in_session"]) == 0:
        return

    vals = [row["ema_fast"], row["rsi"], row["bb_pct"], row["vwap_dist"], row["vol_spike"]]
    if not all(np.isfinite(v) for v in vals):
        return

    confidence   = min(score_l / 5.0, 1.0)
    position_usd = dynamic_position(confidence)
    if position_usd < 5:
        return

    sl = current_price * (1 - STOP_LOSS_PCT)
    tp = current_price * (1 + TAKE_PROFIT_PCT)

    open_trade = OpenTrade(
        entry_time=now_str,
        direction="LONG",
        entry_price=current_price,
        stop_loss=sl,
        take_profit=tp,
        position_usd=position_usd,
    )
    print(Fore.CYAN + f"\n  🔵 OPEN LONG  @ {current_price:.4f}  "
          f"SL={sl:.4f}  TP={tp:.4f}  size=${position_usd:.2f}  "
          f"score={score_l:.1f}/5  conf={confidence:.0%}")

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
def print_dashboard(ticker: str, current_price: float, df_last: pd.Series):
    os.system("cls" if os.name == "nt" else "clear")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    wins   = [t for t in all_trades if t["win"]]
    losses = [t for t in all_trades if not t["win"]]
    total_pnl = sum(t["pnl"] for t in all_trades)
    wr     = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    dd     = (peak_equity - equity) / peak_equity * 100 if peak_equity else 0.0

    eq_color = Fore.GREEN if equity >= ACCOUNT_SIZE else Fore.RED
    dd_color = Fore.GREEN if dd < 3 else (Fore.YELLOW if dd < 5 else Fore.RED)

    print(Fore.YELLOW + Style.BRIGHT + f"\n  ⚡ SCALP PAPER TRADER — {ASSET}  [{now}]")
    print(Style.DIM + "  " + "─"*60)

    # Market snapshot
    rsi = float(df_last["rsi"])
    rsi_color = Fore.GREEN if rsi < 40 else (Fore.RED if rsi > 60 else Fore.WHITE)
    print(f"  Price : {Fore.CYAN}{current_price:.4f}{Style.RESET_ALL}   "
          f"RSI: {rsi_color}{rsi:.1f}{Style.RESET_ALL}   "
          f"BB%: {df_last['bb_pct']:.2f}   "
          f"VWAP dist: {df_last['vwap_dist']:+.4f}   "
          f"Vol×: {df_last['vol_spike']:.1f}x")

    # Open trade
    if open_trade:
        dist_sl = (current_price - open_trade.stop_loss) / open_trade.entry_price * 100
        dist_tp = (open_trade.take_profit - current_price) / open_trade.entry_price * 100
        unrealised = (current_price - open_trade.entry_price) / open_trade.entry_price * open_trade.position_usd
        ur_color = Fore.GREEN if unrealised >= 0 else Fore.RED
        print(f"\n  🔵 OPEN TRADE:  {open_trade.direction}  entry={open_trade.entry_price:.4f}  "
              f"bars={open_trade.bars_held}/{MAX_HOLD_BARS}  "
              f"size=${open_trade.position_usd:.2f}")
        print(f"     SL dist: {dist_sl:.3f}%   TP dist: {dist_tp:.3f}%   "
              f"Unrealised: {ur_color}${unrealised:+.2f}")
    else:
        print(f"\n  ⚪ No open trade")

    # Account summary
    print(f"\n  {'─'*60}")
    print(f"  Equity  : {eq_color}${equity:,.2f}{Style.RESET_ALL}   "
          f"Start: ${ACCOUNT_SIZE:,.2f}   "
          f"PnL: {Fore.GREEN if total_pnl >= 0 else Fore.RED}${total_pnl:+,.2f}")
    print(f"  DrawDown: {dd_color}{dd:.2f}%{Style.RESET_ALL}   "
          f"Trades: {len(all_trades)}   "
          f"Win Rate: {wr:.1f}%   "
          f"Wins: {len(wins)}  Losses: {len(losses)}")
    print(f"  CSV log : {CSV_FILE}")

    # Recent trades table
    if all_trades:
        recent = all_trades[-6:]
        table  = []
        for t in reversed(recent):
            c = Fore.GREEN if t["win"] else Fore.RED
            table.append([
                t["trade_num"],
                t["entry_time"][11:16],
                t["exit_time"][11:16],
                t["direction"],
                f"{t['entry_price']:.4f}",
                f"{t['exit_price']:.4f}",
                c + f"${t['pnl']:+.2f}" + Style.RESET_ALL,
                t["exit_reason"],
                f"{t['bars_held']}",
            ])
        print(f"\n  {'─'*60}")
        print(tabulate(table,
                       headers=["#","Entry","Exit","Dir","Entry$","Exit$","PnL","Reason","Bars"],
                       tablefmt="simple", stralign="right"))
    print()

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    ticker = GOLD_TICKER if ASSET == "GOLD" else BTC_TICKER
    init_csv()

    print(Fore.YELLOW + Style.BRIGHT + f"""
  ╔══════════════════════════════════════════════╗
  ║   SCALP PAPER TRADER  —  {ASSET:<18}     ║
  ║   Ticker : {ticker:<20}            ║
  ║   Account: ${ACCOUNT_SIZE:,.0f}  |  Longs Only         ║
  ║   TP={TAKE_PROFIT_PCT*100:.2f}%  SL={STOP_LOSS_PCT*100:.2f}%  R:R={TAKE_PROFIT_PCT/STOP_LOSS_PCT:.0f}:1         ║
  ║   CSV log: {CSV_FILE:<20}     ║
  ║   Press Ctrl+C to stop                      ║
  ╚══════════════════════════════════════════════╝
    """)
    time.sleep(2)

    bar_count = 0

    try:
        while True:
            df_raw = fetch_candles(ticker, n_bars=150)
            if df_raw is None or len(df_raw) < WARMUP_BARS:
                log.warning("Not enough data, retrying...")
                time.sleep(POLL_SECONDS)
                continue

            df = build_indicators(df_raw)
            if df.empty or len(df) < 2:
                time.sleep(POLL_SECONDS)
                continue

            last      = df.iloc[-1]
            prev      = df.iloc[-2]
            price     = float(last["Close"])
            now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            bar_count += 1

            # Manage open trade first
            check_open_trade(price, now_str)

            # Try entry on latest bar
            try_open_trade(last, float(prev["ema_cross"]), price, now_str)

            # Refresh dashboard
            print_dashboard(ticker, price, last)

            log.info(f"Bar {bar_count} | price={price:.4f} | "
                     f"RSI={last['rsi']:.1f} | BB%={last['bb_pct']:.2f} | "
                     f"equity=${equity:.2f} | open={'YES' if open_trade else 'NO'}")

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n  ⚡ Stopped by user. Final summary:\n")
        if all_trades:
            wins   = [t for t in all_trades if t["win"]]
            losses = [t for t in all_trades if not t["win"]]
            pnls   = [t["pnl"] for t in all_trades]
            pf     = abs(sum(t["pnl"] for t in wins)) / (abs(sum(t["pnl"] for t in losses)) + 1e-9)
            print(f"  Trades     : {len(all_trades)}")
            print(f"  Win Rate   : {len(wins)/len(all_trades):.1%}")
            print(f"  Total PnL  : ${sum(pnls):+,.2f}")
            print(f"  Best trade : ${max(pnls):+.2f}")
            print(f"  Worst trade: ${min(pnls):+.2f}")
            print(f"  Prof Factor: {pf:.2f}")
            print(f"  Final Equity: ${equity:,.2f}")
        print(f"\n  Full trade log: {CSV_FILE}\n")


if __name__ == "__main__":
    main()
