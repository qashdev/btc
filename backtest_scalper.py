"""
Scalping Strategy Backtester - BTC Historical Data
Based on your live trading bot, adapted for historical backtesting
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
# CONFIG (Same as your live bot)
# ─────────────────────────────────────────────
EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.5
MIN_SCORE        = 3

TAKE_PROFIT_PCT  = 0.0025
STOP_LOSS_PCT    = 0.0008
MAX_HOLD_BARS    = 7

SESSION_WINDOWS  = [
    (7,  0,  10, 0),
    (13, 30, 16, 30),
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
# LOAD HISTORICAL DATA
# ─────────────────────────────────────────────
def load_binance_csv(csv_path):
    """Load Binance historical data CSV"""
    df = pd.read_csv(csv_path)
    
    # Binance columns: timestamp, open, high, low, close, volume, close_time, ...
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                  'close_time', 'quote_vol', 'trades', 'taker_buy_base',
                  'taker_buy_quote', 'ignore']
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df


def load_all_historical_data(data_dir="./historical_data"):
    """Load and combine all monthly CSV files"""
    
    # Check multiple possible locations
    possible_dirs = [
        data_dir,
        f"{data_dir}/spot",
        f"{data_dir}/spot/monthly/klines/BTCUSDT/5m",
        "./historical_data/spot",
        "./historical_data/spot/monthly/klines/BTCUSDT/5m"
    ]
    
    csv_files = []
    found_dir = None
    
    for d in possible_dirs:
        if os.path.exists(d):
            # Try to find files recursively
            files = sorted(glob.glob(f"{d}/**/BTCUSDT-5m-*.csv", recursive=True))
            if not files:
                # Try direct search
                files = sorted(glob.glob(f"{d}/BTCUSDT-5m-*.csv"))
            
            if files:
                csv_files = files
                found_dir = d
                break
    
    if not csv_files:
        print(Fore.RED + f"\n❌ No CSV files found!")
        print("\nChecked these locations:")
        for d in possible_dirs:
            exists = "✓" if os.path.exists(d) else "✗"
            print(f"  {exists} {d}")
        print("\nRun download_btc_data.py first!\n")
        return None
    
    print(f"\n📂 Found {len(csv_files)} CSV files in: {found_dir}")
    print("📊 Loading and combining data...\n")
    
    dfs = []
    for f in csv_files:
        try:
            df = load_binance_csv(f)
            dfs.append(df)
            print(f"  ✓ {os.path.basename(f):<30} {len(df):>8,} rows")
        except Exception as e:
            print(f"  ✗ {os.path.basename(f):<30} Error: {e}")
    
    if not dfs:
        print(Fore.RED + "\n❌ Failed to load any CSV files!\n")
        return None
    
    print(f"\n🔄 Combining all data...")
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
# INDICATORS (Same as your bot)
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
    if 7*60 <= mins < 10*60:
        return "London open"
    if 13*60+30 <= mins < 16*60+30:
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
# BACKTESTING ENGINE
# ─────────────────────────────────────────────
def run_backtest(df):
    """Run backtest on historical data"""
    
    equity = ACCOUNT_SIZE
    peak_equity = ACCOUNT_SIZE
    open_trade = None
    all_trades = []
    consec_losses = 0
    
    total_bars = len(df)
    
    print("\n" + "="*70)
    print("  🚀 STARTING BACKTEST")
    print("="*70)
    print(f"  Total bars to process: {total_bars:,}")
    print(f"  Estimated time: {total_bars // 50000} - {total_bars // 30000} minutes\n")
    
    for i in range(1, total_bars):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        price = float(row["Close"])
        timestamp = row.name
        dt_utc = timestamp.to_pydatetime()
        now_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        sess = session_label(dt_utc)
        
        # Progress indicator
        if i % 10000 == 0:
            pct = (i / total_bars) * 100
            print(f"  Progress: {pct:5.1f}% | {now_str} | Equity: ${equity:>10,.0f} | Trades: {len(all_trades):>4}")
        
        # ─── MANAGE OPEN TRADE ───
        if open_trade is not None:
            open_trade.bars_held += 1
            t = open_trade
            
            hit_sl = price <= t.stop_loss
            hit_tp = price >= t.take_profit
            time_exit = t.bars_held >= MAX_HOLD_BARS
            
            if hit_sl or hit_tp or time_exit:
                # Close trade
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
        
        # ─── LOOK FOR NEW TRADE ───
        if open_trade is None and in_aplus_session(dt_utc):
            score_l, _ = score_row(row, float(prev_row["ema_cross"]))
            
            if score_l >= MIN_SCORE:
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
    return all_trades, equity


# ─────────────────────────────────────────────
# RESULTS ANALYSIS
# ─────────────────────────────────────────────
def analyze_results(trades, final_equity):
    """Print comprehensive backtest results"""
    
    if not trades:
        print(Fore.RED + "\n❌ No trades executed!\n")
        print("Possible reasons:")
        print("  - MIN_SCORE too high (currently: {})".format(MIN_SCORE))
        print("  - Not enough data in trading sessions")
        print("  - Indicators filtering out all signals\n")
        return
    
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100
    
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    
    profit_factor = abs(sum(t["pnl"] for t in wins)) / (abs(sum(t["pnl"] for t in losses)) + 1e-9)
    
    max_dd = 0
    peak = ACCOUNT_SIZE
    equity_curve = []
    running_equity = ACCOUNT_SIZE
    
    for t in trades:
        running_equity += t["pnl"]
        equity_curve.append(running_equity)
        peak = max(peak, running_equity)
        dd = (peak - running_equity) / peak * 100
        max_dd = max(max_dd, dd)
    
    # Session breakdown
    london_trades = [t for t in trades if t.get("session") == "London open"]
    ny_trades = [t for t in trades if t.get("session") == "NY open"]
    
    london_wins = [t for t in london_trades if t["win"]]
    ny_wins = [t for t in ny_trades if t["win"]]
    
    # Exit reason breakdown
    tp_exits = [t for t in trades if t["exit_reason"] == "TP"]
    sl_exits = [t for t in trades if t["exit_reason"] == "SL"]
    time_exits = [t for t in trades if t["exit_reason"] == "TIME"]
    
    print("\n" + "="*70)
    print(Fore.YELLOW + Style.BRIGHT + "  📊 BACKTEST RESULTS")
    print("="*70)
    
    print(f"\n  {'Starting Equity:':<30} ${ACCOUNT_SIZE:,.2f}")
    print(f"  {'Final Equity:':<30} {Fore.GREEN if final_equity >= ACCOUNT_SIZE else Fore.RED}${final_equity:,.2f}{Style.RESET_ALL}")
    print(f"  {'Total PnL:':<30} {Fore.GREEN if total_pnl >= 0 else Fore.RED}${total_pnl:+,.2f}{Style.RESET_ALL}")
    print(f"  {'Return:':<30} {Fore.GREEN if total_pnl >= 0 else Fore.RED}{(total_pnl/ACCOUNT_SIZE)*100:+.2f}%{Style.RESET_ALL}")
    
    print(f"\n  {'─ Trade Statistics ─':<30}")
    print(f"  {'Total Trades:':<30} {len(trades)}")
    print(f"  {'Wins:':<30} {len(wins)} ({win_rate:.1f}%)")
    print(f"  {'Losses:':<30} {len(losses)} ({100-win_rate:.1f}%)")
    
    print(f"\n  {'Average Win:':<30} ${avg_win:+.2f}")
    print(f"  {'Average Loss:':<30} ${avg_loss:+.2f}")
    print(f"  {'Profit Factor:':<30} {profit_factor:.2f}")
    print(f"  {'Max Drawdown:':<30} {max_dd:.2f}%")
    
    print(f"\n  {'─ Exit Breakdown ─':<30}")
    print(f"  {'Take Profit:':<30} {len(tp_exits)} ({len(tp_exits)/len(trades)*100:.1f}%)")
    print(f"  {'Stop Loss:':<30} {len(sl_exits)} ({len(sl_exits)/len(trades)*100:.1f}%)")
    print(f"  {'Time Exit:':<30} {len(time_exits)} ({len(time_exits)/len(trades)*100:.1f}%)")
    
    print(f"\n  {'─ Session Breakdown ─':<30}")
    if london_trades:
        london_wr = len(london_wins)/len(london_trades)*100
        print(f"  {'London (7-10 UTC):':<30} {len(london_wins)}/{len(london_trades)} ({london_wr:.1f}%)")
    if ny_trades:
        ny_wr = len(ny_wins)/len(ny_trades)*100
        print(f"  {'NY (13:30-16:30 UTC):':<30} {len(ny_wins)}/{len(ny_trades)} ({ny_wr:.1f}%)")
    
    print("\n" + "="*70)
    
    # Last 10 trades
    print(f"\n  📋 Last 10 Trades:")
    print("="*70)
    
    recent = trades[-10:]
    table = []
    for t in reversed(recent):
        c = Fore.GREEN if t["win"] else Fore.RED
        table.append([
            t["trade_num"],
            t["entry_time"][5:16],
            f"${t['entry_price']:.0f}",
            c + f"${t['pnl']:+.2f}" + Style.RESET_ALL,
            t["exit_reason"],
            t.get("session", "")[:10],
        ])
    
    print(tabulate(table, headers=["#","Entry Time","Price","PnL","Exit","Session"], tablefmt="simple"))
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(Fore.YELLOW + Style.BRIGHT + """
  ╔════════════════════════════════════════════════════════════╗
  ║  BTC SCALPING STRATEGY BACKTESTER                        ║
  ║  Historical Data: 2020-2026 (Binance)                    ║
  ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    df = load_all_historical_data("./historical_data")
    
    if df is None or len(df) < 100:
        print(Fore.RED + "\n❌ Not enough data to backtest!\n")
        return
    
    # Build indicators
    print("🔧 Calculating indicators...")
    df = build_indicators(df)
    print(f"✅ Ready to backtest on {len(df):,} candles")
    
    # Run backtest
    trades, final_equity = run_backtest(df)
    
    # Analyze results
    analyze_results(trades, final_equity)
    
    # Save results
    if trades:
        results_file = "backtest_results.csv"
        pd.DataFrame(trades).to_csv(results_file, index=False)
        print(f"💾 Full results saved to: {results_file}")
        print(f"   Open in Excel/Google Sheets for detailed analysis\n")


if __name__ == "__main__":
    main()