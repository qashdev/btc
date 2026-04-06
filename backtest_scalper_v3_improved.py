"""
Backtester v3 - WITH IMPROVEMENTS
Compare this against original results to see improvement
"""

import csv
import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ─────────────────────────────────────────────
# CONFIG - IMPROVED VERSION
# ─────────────────────────────────────────────
EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.8              # INCREASED from 1.5
MIN_SCORE        = 3.5              # INCREASED from 3.0

TAKE_PROFIT_PCT  = 0.0035           # INCREASED from 0.0025
STOP_LOSS_PCT    = 0.0008
MAX_HOLD_BARS    = 7

# Skip first 15 min of session opens
SESSION_WINDOWS  = [
    (7,  15,  10, 0),
    (13, 45, 16, 30),
]

ACCOUNT_SIZE         = 10_000
MAX_DRAWDOWN_LIMIT   = 0.05
RISK_WHEN_SAFE       = 0.95
RISK_WHEN_WARNING    = 0.60
RISK_WHEN_DANGER     = 0.10
MAX_POSITION_USD     = 1_800

STREAK_BOOST_AFTER   = 2
STREAK_BOOST_AMOUNT  = 0.20

# ─────────────────────────────────────────────
# NEW FILTERS
# ─────────────────────────────────────────────
def is_strong_trend(row):
    """Only trade when there's clear momentum"""
    ema_spread = (row["ema_fast"] - row["ema_slow"]) / row["ema_slow"]
    return abs(ema_spread) > 0.003

def has_volume_confirmation(row):
    """Volume must be strong"""
    return row["vol_spike"] >= VOLUME_SPIKE_MIN

def avoid_session_start(dt_utc):
    """Skip first 15 minutes of session opens"""
    mins = dt_utc.hour * 60 + dt_utc.minute
    if 7*60 <= mins < 7*60+15:
        return False
    if 13*60+30 <= mins < 13*60+45:
        return False
    return True

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_binance_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                  'close_time', 'quote_vol', 'trades', 'taker_buy_base',
                  'taker_buy_quote', 'ignore']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def load_all_historical_data(data_dir="./historical_data"):
    possible_dirs = [
        data_dir,
        f"{data_dir}/spot",
        f"{data_dir}/spot/monthly/klines/BTCUSDT/5m",
    ]
    
    csv_files = []
    found_dir = None
    
    for d in possible_dirs:
        if os.path.exists(d):
            files = sorted(glob.glob(f"{d}/**/BTCUSDT-5m-*.csv", recursive=True))
            if not files:
                files = sorted(glob.glob(f"{d}/BTCUSDT-5m-*.csv"))
            if files:
                csv_files = files
                found_dir = d
                break
    
    if not csv_files:
        print(Fore.RED + f"\n❌ No CSV files found!")
        return None
    
    print(f"\n📂 Found {len(csv_files)} CSV files in: {found_dir}")
    print("📊 Loading data...\n")
    
    dfs = []
    for f in csv_files:
        try:
            df = load_binance_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)} - Error: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    print(f"✅ Total rows: {len(combined):,}")
    print(f"📅 Date range: {combined.index[0]} to {combined.index[-1]}\n")
    
    return combined

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

def in_aplus_session(dt_utc):
    mins = dt_utc.hour * 60 + dt_utc.minute
    for (sh, sm, eh, em) in SESSION_WINDOWS:
        if sh * 60 + sm <= mins < eh * 60 + em:
            return True
    return False

def session_label(dt_utc):
    mins = dt_utc.hour * 60 + dt_utc.minute
    if 7*60+15 <= mins < 10*60:
        return "London open"
    if 13*60+45 <= mins < 16*60+30:
        return "NY open"
    return "out of session"

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

def dynamic_position(confidence, equity, peak_equity):
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
# BACKTESTING
# ─────────────────────────────────────────────
def run_backtest(df):
    equity = ACCOUNT_SIZE
    peak_equity = ACCOUNT_SIZE
    open_trade = None
    all_trades = []
    consec_losses = 0
    
    total_bars = len(df)
    
    print("\n" + "="*70)
    print("  🚀 STARTING IMPROVED BACKTEST")
    print("="*70)
    print(f"  Total bars: {total_bars:,}")
    print(f"  New filters: Trend + Volume + Skip session start")
    print(f"  Wider TP: 0.35% (was 0.25%)")
    print(f"  Higher MIN_SCORE: 3.5 (was 3.0)\n")
    
    filtered_out = {
        'session_start': 0,
        'weak_trend': 0,
        'low_volume': 0,
        'low_score': 0,
    }
    
    for i in range(1, total_bars):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        price = float(row["Close"])
        timestamp = row.name
        dt_utc = timestamp.to_pydatetime()
        now_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        sess = session_label(dt_utc)
        
        if i % 10000 == 0:
            pct = (i / total_bars) * 100
            print(f"  Progress: {pct:5.1f}% | {now_str} | Equity: ${equity:>10,.0f} | Trades: {len(all_trades):>4}")
        
        # Manage open trade
        if open_trade is not None:
            open_trade.bars_held += 1
            t = open_trade
            
            hit_sl = price <= t.stop_loss
            hit_tp = price >= t.take_profit
            time_exit = t.bars_held >= MAX_HOLD_BARS
            
            if hit_sl or hit_tp or time_exit:
                if hit_sl:
                    pnl = -t.position_usd * STOP_LOSS_PCT
                    reason = "SL"
                elif hit_tp:
                    pnl = t.position_usd * TAKE_PROFIT_PCT
                    reason = "TP"
                else:
                    pnl = (price - t.entry_price) / t.entry_price * t.position_usd
                    reason = "TIME"
                
                equity = max(0.0, equity + pnl)
                peak_equity = max(peak_equity, equity)
                win = pnl > 0
                consec_losses = 0 if win else consec_losses + 1
                
                all_trades.append({
                    "trade_num": len(all_trades) + 1,
                    "entry_time": t.entry_time,
                    "exit_time": now_str,
                    "direction": t.direction,
                    "entry_price": round(t.entry_price, 2),
                    "exit_price": round(price, 2),
                    "position_usd": round(t.position_usd, 2),
                    "pnl": round(pnl, 2),
                    "win": win,
                    "exit_reason": reason,
                    "bars_held": t.bars_held,
                    "session": sess,
                })
                
                open_trade = None
        
        # Look for new trade WITH FILTERS
        if open_trade is None and in_aplus_session(dt_utc):
            # Filter 1: Avoid session start
            if not avoid_session_start(dt_utc):
                filtered_out['session_start'] += 1
                continue
            
            # Filter 2: Strong trend only
            if not is_strong_trend(row):
                filtered_out['weak_trend'] += 1
                continue
            
            # Filter 3: Volume confirmation
            if not has_volume_confirmation(row):
                filtered_out['low_volume'] += 1
                continue
            
            score_l, _ = score_row(row, float(prev_row["ema_cross"]))
            
            # Filter 4: Higher MIN_SCORE
            if score_l < MIN_SCORE:
                filtered_out['low_score'] += 1
                continue
            
            vals = [row["ema_fast"], row["rsi"], row["bb_pct"], row["vwap_dist"], row["vol_spike"]]
            if all(np.isfinite(float(v)) for v in vals):
                streak_boost = STREAK_BOOST_AMOUNT if consec_losses >= STREAK_BOOST_AFTER else 0.0
                confidence = min(score_l / 5.0 + streak_boost, 1.0)
                position_usd = dynamic_position(confidence, equity, peak_equity)
                
                if position_usd >= 5:
                    sl = price * (1 - STOP_LOSS_PCT)
                    tp = price * (1 + TAKE_PROFIT_PCT)
                    
                    open_trade = OpenTrade(
                        entry_time=now_str,
                        direction="LONG",
                        entry_price=price,
                        stop_loss=sl,
                        take_profit=tp,
                        position_usd=position_usd,
                    )
    
    print(f"\n✅ Backtest complete!\n")
    
    # Show filter stats
    print("📊 FILTER STATISTICS:")
    print(f"  Filtered by session start: {filtered_out['session_start']:,}")
    print(f"  Filtered by weak trend:    {filtered_out['weak_trend']:,}")
    print(f"  Filtered by low volume:    {filtered_out['low_volume']:,}")
    print(f"  Filtered by low score:     {filtered_out['low_score']:,}")
    print(f"  Total filtered:            {sum(filtered_out.values()):,}")
    print(f"  Trades executed:           {len(all_trades):,}\n")
    
    return all_trades, equity

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
def compare_results(trades, final_equity):
    if not trades:
        print(Fore.RED + "\n❌ No trades executed!\n")
        return
    
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100
    
    tp_exits = [t for t in trades if t["exit_reason"] == "TP"]
    sl_exits = [t for t in trades if t["exit_reason"] == "SL"]
    time_exits = [t for t in trades if t["exit_reason"] == "TIME"]
    
    print("\n" + "="*70)
    print(Fore.YELLOW + Style.BRIGHT + "  📊 IMPROVED STRATEGY RESULTS vs ORIGINAL")
    print("="*70)
    
    print(f"\n  {'Metric':<30} {'Original':<15} {'Improved':<15} {'Change'}")
    print("  " + "-"*68)
    print(f"  {'Total Trades':<30} {'10,558':<15} {len(trades):<15,} {Fore.CYAN}{((len(trades)-10558)/10558)*100:+.1f}%{Style.RESET_ALL}")
    print(f"  {'Win Rate':<30} {'43.88%':<15} {f'{win_rate:.2f}%':<15} {Fore.GREEN if win_rate > 43.88 else Fore.RED}{win_rate-43.88:+.2f}%{Style.RESET_ALL}")
    print(f"  {'Total PnL':<30} {'$10,172':<15} {f'${total_pnl:,.0f}':<15} {Fore.GREEN if total_pnl > 10172 else Fore.RED}${total_pnl-10172:+,.0f}{Style.RESET_ALL}")
    print(f"  {'Final Equity':<30} {'$20,172':<15} {f'${final_equity:,.0f}':<15} {Fore.GREEN if final_equity > 20172 else Fore.RED}${final_equity-20172:+,.0f}{Style.RESET_ALL}")
    
    print(f"\n  {'EXIT BREAKDOWN':<30} {'Original':<15} {'Improved'}")
    print("  " + "-"*68)
    print(f"  {'Stop Loss hits':<30} {'5,734 (54.3%)':<15} {len(sl_exits):,} ({len(sl_exits)/len(trades)*100:.1f}%)")
    print(f"  {'Take Profit hits':<30} {'3,717 (35.2%)':<15} {len(tp_exits):,} ({len(tp_exits)/len(trades)*100:.1f}%)")
    print(f"  {'Time exits':<30} {'1,107 (10.5%)':<15} {len(time_exits):,} ({len(time_exits)/len(trades)*100:.1f}%)")
    
    print("\n" + "="*70)
    
    # Session breakdown
    london = [t for t in trades if t.get("session") == "London open"]
    ny = [t for t in trades if t.get("session") == "NY open"]
    
    if london:
        l_wins = [t for t in london if t["win"]]
        print(f"\n  London: {len(l_wins)}/{len(london)} = {len(l_wins)/len(london)*100:.1f}% (was 43.1%)")
    if ny:
        n_wins = [t for t in ny if t["win"]]
        print(f"  NY:     {len(n_wins)}/{len(ny)} = {len(n_wins)/len(ny)*100:.1f}% (was 43.0%)")
    
    print()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(Fore.YELLOW + Style.BRIGHT + """
  ╔════════════════════════════════════════════════════════════╗
  ║  BTC SCALPING STRATEGY BACKTESTER v3 (IMPROVED)          ║
  ║                                                            ║
  ║  New Filters:                                             ║
  ║  ✓ Trend confirmation (EMA spread > 0.3%)                 ║
  ║  ✓ Volume strength (spike >= 1.8x)                        ║
  ║  ✓ Skip session start volatility (first 15min)            ║
  ║  ✓ Wider TP: 0.35% (was 0.25%)                            ║
  ║  ✓ Higher MIN_SCORE: 3.5 (was 3.0)                        ║
  ╚════════════════════════════════════════════════════════════╝
    """)
    
    df = load_all_historical_data("./historical_data")
    
    if df is None or len(df) < 100:
        print(Fore.RED + "\n❌ Not enough data!\n")
        return
    
    print("🔧 Calculating indicators...")
    df = build_indicators(df)
    print(f"✅ Ready: {len(df):,} candles")
    
    trades, final_equity = run_backtest(df)
    compare_results(trades, final_equity)
    
    if trades:
        results_file = "backtest_results_v3_improved.csv"
        pd.DataFrame(trades).to_csv(results_file, index=False)
        print(f"💾 Results saved: {results_file}\n")


if __name__ == "__main__":
    main()
