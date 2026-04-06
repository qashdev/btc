"""
Scalping Paper Trader v2 — BTC / Gold  (A+ Setups Only)
=========================================================
Upgraded with filters from real trade analysis:

  Filter 1 — Session window   : Only trade 07:00-10:00 and 13:30-16:30 UTC
  Filter 2 — Minimum score    : Require score >= 3/5 (was 2/5)
  Filter 3 — Streak boost     : After 2+ consecutive stops, +20% confidence
  Filter 4 — Position cap     : Hard cap $1,800 max per trade
  Filter 5 — Max hold         : 7 bars (35 min) instead of 24 — exit slow trades

Run:   py scalp_paper_trader.py
Stop:  Ctrl+C

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
from dataclasses import dataclass
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ASSET            = "BTC"
GOLD_TICKER      = "GC=F"
BTC_TICKER       = "BTC-USD"

EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.5
MIN_SCORE        = 3              # upgraded from 2 → 3

TAKE_PROFIT_PCT  = 0.025
STOP_LOSS_PCT    = 0.08
MAX_HOLD_BARS    = 7              # 35 min max hold

# A+ session windows (UTC)
SESSION_WINDOWS  = [
    (7,  0,  10, 0),
    (13, 30, 16, 30),
]

ACCOUNT_SIZE         = 10_000
MAX_DRAWDOWN_LIMIT   = 0.05
RISK_WHEN_SAFE       = 0.95
RISK_WHEN_WARNING    = 0.60
RISK_WHEN_DANGER     = 0.10
MAX_POSITION_USD     = 1_800     # hard cap per trade

STREAK_BOOST_AFTER   = 2
STREAK_BOOST_AMOUNT  = 0.20

POLL_SECONDS     = 60
CANDLE_INTERVAL  = "5m"
WARMUP_BARS      = 60
CSV_FILE         = f"paper_trades_v2_{ASSET.lower()}.csv"
LOG_FILE         = f"paper_trader_v2_{ASSET.lower()}.log"

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

equity        = ACCOUNT_SIZE
peak_equity   = ACCOUNT_SIZE
open_trade    = None
all_trades    = []
consec_losses = 0

# ─────────────────────────────────────────────
# SESSION FILTER
# ─────────────────────────────────────────────
def in_aplus_session(dt_utc):
    mins = dt_utc.hour * 60 + dt_utc.minute
    for (sh, sm, eh, em) in SESSION_WINDOWS:
        if sh * 60 + sm <= mins < eh * 60 + em:
            return True
    return False

def session_label(dt_utc):
    mins = dt_utc.hour * 60 + dt_utc.minute
    if 7*60 <= mins < 10*60:
        return "London open"
    if 13*60+30 <= mins < 16*60+30:
        return "NY open"
    return "out of session"

# ─────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────
CSV_HEADERS = [
    "trade_num","entry_time","exit_time","direction",
    "entry_price","exit_price","stop_loss","take_profit",
    "position_usd","pnl","win","exit_reason",
    "bars_held","equity_after","session","score","streak_at_entry"
]

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()

def append_csv(row):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow({k: row.get(k, "") for k in CSV_HEADERS})

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def fetch_candles(ticker, n_bars=150):
    try:
        df = yf.download(ticker, period="5d", interval=CANDLE_INTERVAL,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna().tail(n_bars)
    except Exception as e:
        log.warning(f"Fetch error: {e}")
        return None

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def build_indicators(df):
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
    return d.dropna()

# ─────────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────────
def score_row(row, prev_cross):
    sl = ss = 0.0
    cross = float(row["ema_cross"])
    if prev_cross <= 0 and cross > 0:   sl += 1
    elif prev_cross >= 0 and cross < 0: ss += 1
    elif cross > 0:  sl += 0.5
    elif cross < 0:  ss += 0.5

    rsi = float(row["rsi"])
    if rsi < 38:   sl += 1
    elif rsi > 62: ss += 1

    vd = float(row["vwap_dist"])
    if vd < -0.001:  sl += 1
    elif vd > 0.001: ss += 1

    bp = float(row["bb_pct"])
    if bp < 0.2:   sl += 1
    elif bp > 0.8: ss += 1

    vs = float(row["vol_spike"])
    if vs >= VOLUME_SPIKE_MIN:
        if float(row["mom3"]) > 0: sl += 1
        else:                       ss += 1
    return sl, ss

# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
def dynamic_position(confidence):
    dd_ratio = min((peak_equity - equity) / peak_equity / MAX_DRAWDOWN_LIMIT, 1.0) if peak_equity > 0 else 0.0
    if dd_ratio < 0.5:
        max_risk = RISK_WHEN_SAFE + (dd_ratio / 0.5) * (RISK_WHEN_WARNING - RISK_WHEN_SAFE)
    else:
        max_risk = RISK_WHEN_WARNING + ((dd_ratio - 0.5) / 0.5) * (RISK_WHEN_DANGER - RISK_WHEN_WARNING)
    b     = TAKE_PROFIT_PCT / STOP_LOSS_PCT
    kelly = max(0, (confidence * (b + 1) - 1) / b) * 0.5
    raw   = max(equity * 0.01, min(equity * kelly, equity * max_risk))
    return min(raw, MAX_POSITION_USD)

# ─────────────────────────────────────────────
# TRADE MANAGEMENT
# ─────────────────────────────────────────────
def check_open_trade(price, now_str, sess):
    global open_trade, equity, peak_equity, consec_losses
    if open_trade is None:
        return False

    open_trade.bars_held += 1
    t = open_trade
    hit_sl = hit_tp = time_exit = False
    pnl = 0.0

    if price <= t.stop_loss:
        hit_sl = True
        pnl = -t.position_usd * STOP_LOSS_PCT
    elif price >= t.take_profit:
        hit_tp = True
        pnl = t.position_usd * TAKE_PROFIT_PCT

    if t.bars_held >= MAX_HOLD_BARS and not (hit_sl or hit_tp):
        time_exit = True
        pnl = (price - t.entry_price) / t.entry_price * t.position_usd

    if hit_sl or hit_tp or time_exit:
        equity      = max(0.0, equity + pnl)
        peak_equity = max(peak_equity, equity)
        win = pnl > 0
        consec_losses = 0 if win else consec_losses + 1

        reason    = "TP" if hit_tp else ("SL" if hit_sl else "TIME")
        color     = Fore.GREEN if win else Fore.RED
        trade_num = len(all_trades) + 1

        row = {
            "trade_num": trade_num, "entry_time": t.entry_time,
            "exit_time": now_str, "direction": t.direction,
            "entry_price": round(t.entry_price, 2), "exit_price": round(price, 2),
            "stop_loss": round(t.stop_loss, 2), "take_profit": round(t.take_profit, 2),
            "position_usd": round(t.position_usd, 2), "pnl": round(pnl, 2),
            "win": win, "exit_reason": reason, "bars_held": t.bars_held,
            "equity_after": round(equity, 2), "session": sess,
        }
        all_trades.append(row)
        append_csv(row)

        streak_note = f"  streak now {consec_losses}" if not win else ""
        print(color + f"\n  {'WIN' if win else 'LOSS'}  #{trade_num}  "
              f"{t.entry_price:.0f}→{price:.0f}  ${pnl:+.2f}  {reason}  eq=${equity:,.0f}{streak_note}")
        open_trade = None
        return True
    return False


def try_open_trade(row, prev_cross, price, now_str, dt_utc):
    global open_trade

    if open_trade is not None:
        return
    if not in_aplus_session(dt_utc):
        return

    score_l, _ = score_row(row, prev_cross)
    if score_l < MIN_SCORE:
        return

    vals = [row["ema_fast"], row["rsi"], row["bb_pct"], row["vwap_dist"], row["vol_spike"]]
    if not all(np.isfinite(float(v)) for v in vals):
        return

    streak_boost = STREAK_BOOST_AMOUNT if consec_losses >= STREAK_BOOST_AFTER else 0.0
    confidence   = min(score_l / 5.0 + streak_boost, 1.0)
    position_usd = dynamic_position(confidence)
    if position_usd < 5:
        return

    sl   = price * (1 - STOP_LOSS_PCT)
    tp   = price * (1 + TAKE_PROFIT_PCT)
    sess = session_label(dt_utc)
    boost_note = f"  [STREAK BOOST after {consec_losses} losses]" if streak_boost > 0 else ""

    print(Fore.CYAN + f"\n  OPEN  @ {price:.0f}  SL={sl:.0f}  TP={tp:.0f}  "
          f"${position_usd:.0f}  score={score_l:.1f}/5  {sess}{boost_note}")

    open_trade = OpenTrade(
        entry_time=now_str, direction="LONG",
        entry_price=price, stop_loss=sl, take_profit=tp, position_usd=position_usd,
    )

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
def print_dashboard(ticker, price, df_last, dt_utc):
    os.system("cls" if os.name == "nt" else "clear")
    now_str   = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    sess      = session_label(dt_utc)
    in_sess   = in_aplus_session(dt_utc)
    wins      = [t for t in all_trades if t["win"]]
    losses    = [t for t in all_trades if not t["win"]]
    total_pnl = sum(t["pnl"] for t in all_trades)
    wr        = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    dd        = (peak_equity - equity) / peak_equity * 100 if peak_equity else 0.0

    sc = Fore.GREEN if in_sess else Fore.YELLOW
    print(Fore.YELLOW + Style.BRIGHT + f"\n  SCALP PAPER TRADER v2  —  {ASSET}  [{now_str}]")
    print(Style.DIM + "  " + "─"*65)
    print(f"  Session : {sc}{sess}  {'LIVE' if in_sess else 'WAITING'}{Style.RESET_ALL}   "
          f"streak={consec_losses} losses  {'BOOST ACTIVE' if consec_losses >= STREAK_BOOST_AFTER else ''}")

    rsi = float(df_last["rsi"])
    rc  = Fore.GREEN if rsi < 40 else (Fore.RED if rsi > 60 else Fore.WHITE)
    print(f"  Price: {Fore.CYAN}{price:.2f}{Style.RESET_ALL}  "
          f"RSI:{rc}{rsi:.1f}{Style.RESET_ALL}  "
          f"BB%:{float(df_last['bb_pct']):.2f}  "
          f"VWAP:{float(df_last['vwap_dist']):+.4f}  "
          f"Vol:{float(df_last['vol_spike']):.1f}x")

    if open_trade:
        t  = open_trade
        ur = (price - t.entry_price) / t.entry_price * t.position_usd
        uc = Fore.GREEN if ur >= 0 else Fore.RED
        print(f"\n  OPEN LONG  @ {t.entry_price:.0f}  "
              f"bars={t.bars_held}/{MAX_HOLD_BARS}  size=${t.position_usd:.0f}  "
              f"unrealised:{uc}${ur:+.2f}")
    else:
        print(f"\n  No open trade")

    eq_color = Fore.GREEN if equity >= ACCOUNT_SIZE else Fore.RED
    pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
    print(f"\n  {'─'*65}")
    print(f"  Equity: {eq_color}${equity:,.2f}{Style.RESET_ALL}  "
          f"PnL:{pnl_color}${total_pnl:+,.2f}{Style.RESET_ALL}  "
          f"Trades:{len(all_trades)}  WR:{wr:.1f}%  DD:{dd:.2f}%")
    print(f"  W:{len(wins)}  L:{len(losses)}  CSV:{CSV_FILE}")

    if all_trades:
        recent = all_trades[-5:]
        table  = []
        for t in reversed(recent):
            c = Fore.GREEN if t["win"] else Fore.RED
            table.append([
                t["trade_num"],
                t["entry_time"][11:16],
                t["exit_time"][11:16],
                f"{t['entry_price']:.0f}",
                c + f"${t['pnl']:+.2f}" + Style.RESET_ALL,
                t["exit_reason"],
                t.get("session", "")[:10],
            ])
        print(f"\n  {'─'*65}")
        print(tabulate(table, headers=["#","Entry","Exit","Price","PnL","Why","Session"],
                       tablefmt="simple"))
    print()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    ticker = GOLD_TICKER if ASSET == "GOLD" else BTC_TICKER
    init_csv()

    print(Fore.YELLOW + Style.BRIGHT + f"""
  ╔══════════════════════════════════════════════════╗
  ║  SCALP PAPER TRADER v2  —  {ASSET}               ║
  ║                                                  ║
  ║  Sessions : 07:00-10:00  &  13:30-16:30 UTC     ║
  ║  Min score: {MIN_SCORE}/5 signals                         ║
  ║  Max pos  : ${MAX_POSITION_USD:,} hard cap                 ║
  ║  Max hold : {MAX_HOLD_BARS} bars = {MAX_HOLD_BARS*5} min                     ║
  ║  Streak   : +20% boost after {STREAK_BOOST_AFTER}+ losses         ║
  ║  CSV      : {CSV_FILE:<28} ║
  ║  Ctrl+C to stop                                  ║
  ╚══════════════════════════════════════════════════╝
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

            last    = df.iloc[-1]
            prev    = df.iloc[-2]
            price   = float(last["Close"])
            now_utc = datetime.now(timezone.utc)
            now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")
            bar_count += 1
            sess    = session_label(now_utc)

            check_open_trade(price, now_str, sess)
            try_open_trade(last, float(prev["ema_cross"]), price, now_str, now_utc)
            print_dashboard(ticker, price, last, now_utc)

            log.info(f"Bar {bar_count} | price={price:.2f} | RSI={float(last['rsi']):.1f} | "
                     f"sess={sess} | streak={consec_losses} | eq=${equity:.0f}")

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n  Final summary:\n")
        if all_trades:
            wins   = [t for t in all_trades if t["win"]]
            losses = [t for t in all_trades if not t["win"]]
            pnls   = [t["pnl"] for t in all_trades]
            pf = abs(sum(t["pnl"] for t in wins)) / (abs(sum(t["pnl"] for t in losses)) + 1e-9)
            print(f"  Trades     : {len(all_trades)}")
            print(f"  Win Rate   : {len(wins)/len(all_trades):.1%}")
            print(f"  Total PnL  : ${sum(pnls):+,.2f}")
            print(f"  Profit Fac : {pf:.2f}")
            print(f"  Final eq   : ${equity:,.2f}")

            for sn in ["London open", "NY open"]:
                st = [t for t in all_trades if t.get("session") == sn]
                if st:
                    sw = [t for t in st if t["win"]]
                    print(f"  {sn}: {len(sw)}/{len(st)} = {len(sw)/len(st):.1%}")
        print(f"\n  CSV: {CSV_FILE}\n")


if __name__ == "__main__":
    main()
