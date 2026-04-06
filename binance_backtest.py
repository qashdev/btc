"""
Binance Historical Data Backtest — BTC A+ Setup Finder
========================================================
Loads all monthly CSV files from ./historical_data/BTCUSDT/
Runs the 5-signal scalping strategy, then exhaustively tests
every filter combination to find what actually achieves 60%+ win rate.

Step 1: py download_binance.py        (run the downloader first)
Step 2: py binance_backtest.py        (run this)

Output:
  backtest_results.png    — charts
  best_filters.json       — the winning filter combo
  all_trades.csv          — every trade with all features attached
"""

import os
import glob
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from dataclasses import dataclass, asdict

# ─────────────────────────────────────────────
# CONFIG — same as your live bot
# ─────────────────────────────────────────────
DATA_DIR         = "./historical_data/spot/monthly/klines/BTCUSDT/5m"
FALLBACK_DIR     = "./historical_data"          # in case folder structure differs

EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.5
MIN_SCORE        = 2

TAKE_PROFIT_PCT  = 0.0025   # 0.25%
STOP_LOSS_PCT    = 0.0008   # 0.08%
MAX_HOLD_BARS    = 24
ACCOUNT_SIZE     = 10_000

# Minimum trades for a filter combo to be considered valid
MIN_TRADES_FOR_VALID_FILTER = 200

# ─────────────────────────────────────────────
# LOAD BINANCE CSVs
# ─────────────────────────────────────────────
def find_csv_files():
    """Search for CSV files in common Binance dump folder structures."""
    patterns = [
        os.path.join(DATA_DIR, "*.csv"),
        os.path.join(DATA_DIR, "**", "*.csv"),
        os.path.join(FALLBACK_DIR, "**", "BTCUSDT*.csv"),
        os.path.join("./historical_data", "**", "*.csv"),
        "./*.csv",  # current directory fallback
    ]
    found = []
    for p in patterns:
        files = glob.glob(p, recursive=True)
        found.extend(files)
    # deduplicate
    found = list(set(found))
    # filter to only BTC 5m files
    found = [f for f in found if "BTCUSDT" in f.upper() or "BTC" in f.upper()]
    return sorted(found)


def load_binance_data(files):
    """
    Binance CSV columns (klines format):
    open_time, open, high, low, close, volume, close_time,
    quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
    """
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None)
            if df.shape[1] >= 6:
                df = df.iloc[:, :6]
                df.columns = ["open_time", "Open", "High", "Low", "Close", "Volume"]
                df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
                df = df.dropna(subset=["open_time"])
                df["Datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df = df.set_index("Datetime")
                df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                dfs.append(df)
                print(f"  Loaded {os.path.basename(f)}: {len(df):,} candles "
                      f"({df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')})")
        except Exception as e:
            print(f"  WARNING: Could not load {f}: {e}")

    if not dfs:
        raise RuntimeError(
            "\nNo CSV files found!\n"
            "Run the downloader first: py download_binance.py\n"
            f"Then check that files exist in: {DATA_DIR}\n"
            "Expected filename pattern: BTCUSDT-5m-YYYY-MM.csv"
        )

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    print(f"\nTotal: {len(combined):,} candles | "
          f"{combined.index[0].strftime('%Y-%m-%d')} → {combined.index[-1].strftime('%Y-%m-%d')}")
    return combined


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()

    # EMAs
    d["ema_fast"]  = d["Close"].ewm(span=EMA_FAST,  adjust=False).mean()
    d["ema_slow"]  = d["Close"].ewm(span=EMA_SLOW,  adjust=False).mean()
    d["ema50"]     = d["Close"].ewm(span=50,         adjust=False).mean()
    d["ema200"]    = d["Close"].ewm(span=200,        adjust=False).mean()
    d["ema_cross"] = np.sign(d["ema_fast"] - d["ema_slow"])

    # RSI
    delta  = d["Close"].diff()
    gain   = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss   = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    d["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Bollinger Bands
    sma          = d["Close"].rolling(BB_PERIOD).mean()
    std          = d["Close"].rolling(BB_PERIOD).std()
    d["bb_upper"] = sma + BB_STD * std
    d["bb_lower"] = sma - BB_STD * std
    d["bb_pct"]   = (d["Close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (sma + 1e-9)

    # VWAP (rolling)
    tp         = (d["High"] + d["Low"] + d["Close"]) / 3
    vol        = d["Volume"].replace(0, 1)
    d["vwap"]  = (tp * vol).rolling(VWAP_PERIOD).sum() / vol.rolling(VWAP_PERIOD).sum()
    d["vwap_dist"] = (d["Close"] - d["vwap"]) / d["vwap"]

    # Volume
    d["vol_avg"]   = d["Volume"].rolling(20).mean()
    d["vol_spike"] = d["Volume"] / (d["vol_avg"] + 1e-9)

    # Momentum
    d["mom3"]  = d["Close"].pct_change(3)
    d["mom10"] = d["Close"].pct_change(10)

    # ATR
    hl  = d["High"] - d["Low"]
    hpc = abs(d["High"] - d["Close"].shift(1))
    lpc = abs(d["Low"]  - d["Close"].shift(1))
    d["atr"]     = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()
    d["atr_pct"] = d["atr"] / d["Close"]

    # Time features
    d["hour"]     = d.index.hour
    d["minute"]   = d.index.minute
    d["weekday"]  = d.index.weekday   # 0=Mon 4=Fri
    d["time_mins"]= d["hour"] * 60 + d["minute"]

    # Pre-computed boolean features used by filters
    d["above_ema50"]     = (d["Close"] > d["ema50"]).astype(int)
    d["above_ema200"]    = (d["Close"] > d["ema200"]).astype(int)
    d["ema50_gt_200"]    = (d["ema50"]  > d["ema200"]).astype(int)
    d["fresh_cross"]     = ((d["ema_cross"].shift(1) <= 0) & (d["ema_cross"] > 0)).astype(int)
    d["below_vwap"]      = (d["vwap_dist"] < 0).astype(int)
    d["in_session_orig"] = (((d["hour"] >= 7) & (d["hour"] < 21)) & (d["weekday"] < 5)).astype(int)

    return d.dropna()


# ─────────────────────────────────────────────
# SIGNAL SCORER
# ─────────────────────────────────────────────
def score_row(row):
    score_l = 0.0
    cross = float(row["ema_cross"])
    prev_cross = float(row["prev_cross"])
    if prev_cross <= 0 and cross > 0:   score_l += 1
    elif cross > 0:                      score_l += 0.5

    rsi = float(row["rsi"])
    if rsi < 35:   score_l += 1

    vd = float(row["vwap_dist"])
    if vd < -0.001: score_l += 1

    bp = float(row["bb_pct"])
    if bp < 0.2:   score_l += 1

    vs = float(row["vol_spike"])
    if vs >= VOLUME_SPIKE_MIN and float(row["mom3"]) > 0:
        score_l += 1

    return score_l


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
@dataclass
class Trade:
    date: str
    entry: float
    exit_price: float
    pnl: float
    win: bool
    bars_held: int
    stop_hit: bool
    tp_hit: bool
    score: float
    # feature snapshot at entry
    rsi: float
    bb_pct: float
    bb_width: float
    vwap_dist: float
    vol_spike: float
    mom3: float
    atr_pct: float
    hour: int
    weekday: int
    above_ema50: int
    above_ema200: int
    ema50_gt_200: int
    fresh_cross: int
    below_vwap: int


def run_backtest(df):
    """Run baseline backtest, return list of Trade objects with all features."""
    print("Running backtest engine...")
    d = df.copy()
    d["prev_cross"] = d["ema_cross"].shift(1).fillna(0)

    trades = []
    skip_until = 0

    for i in range(250, len(d) - MAX_HOLD_BARS):
        if i < skip_until:
            continue

        row = d.iloc[i]

        # original session filter
        if int(row["in_session_orig"]) == 0:
            continue

        score = score_row(row)
        if score < MIN_SCORE:
            continue

        # sanity check
        vals = [row["ema_fast"], row["rsi"], row["bb_pct"], row["vwap_dist"], row["vol_spike"]]
        if not all(np.isfinite(float(v)) for v in vals):
            continue

        entry   = float(row["Close"])
        sl_p    = entry * (1 - STOP_LOSS_PCT)
        tp_p    = entry * (1 + TAKE_PROFIT_PCT)
        future  = d.iloc[i+1 : i+1+MAX_HOLD_BARS]

        hit_sl = hit_tp = False
        bars_held   = MAX_HOLD_BARS
        exit_price  = float(future.iloc[-1]["Close"])

        for b, (_, fr) in enumerate(future.iterrows(), 1):
            p = float(fr["Close"])
            if p <= sl_p:
                hit_sl = True; exit_price = p; bars_held = b; break
            if p >= tp_p:
                hit_tp = True; exit_price = p; bars_held = b; break

        pnl = (exit_price - entry) / entry
        win = pnl > 0
        skip_until = i + bars_held

        trades.append(Trade(
            date       = str(d.index[i])[:16],
            entry      = round(entry, 2),
            exit_price = round(exit_price, 2),
            pnl        = round(pnl * 100, 4),   # store as %
            win        = win,
            bars_held  = bars_held,
            stop_hit   = hit_sl,
            tp_hit     = hit_tp,
            score      = score,
            rsi        = round(float(row["rsi"]), 2),
            bb_pct     = round(float(row["bb_pct"]), 3),
            bb_width   = round(float(row["bb_width"]), 4),
            vwap_dist  = round(float(row["vwap_dist"]), 5),
            vol_spike  = round(float(row["vol_spike"]), 2),
            mom3       = round(float(row["mom3"]), 5),
            atr_pct    = round(float(row["atr_pct"]), 5),
            hour       = int(row["hour"]),
            weekday    = int(row["weekday"]),
            above_ema50  = int(row["above_ema50"]),
            above_ema200 = int(row["above_ema200"]),
            ema50_gt_200 = int(row["ema50_gt_200"]),
            fresh_cross  = int(row["fresh_cross"]),
            below_vwap   = int(row["below_vwap"]),
        ))

    print(f"  {len(trades):,} trades generated")
    return trades


# ─────────────────────────────────────────────
# FILTER OPTIMIZER
# ─────────────────────────────────────────────
def in_session(h, m, windows):
    mins = h * 60 + m
    for sh, sm, eh, em in windows:
        if sh * 60 + sm <= mins < eh * 60 + em:
            return True
    return False


def apply_filter(trades, f):
    """Apply a single filter spec to a list of trades."""
    result = trades

    if f.get("session"):
        windows = f["session"]
        result = [t for t in result if in_session(t.hour, t.minute if hasattr(t,'minute') else 0, windows)]

    if f.get("hour_range"):
        lo, hi = f["hour_range"]
        result = [t for t in result if lo <= t.hour < hi]

    if f.get("weekdays"):
        wds = f["weekdays"]
        result = [t for t in result if t.weekday in wds]

    if f.get("ema_trend") == "bull":
        result = [t for t in result if t.ema50_gt_200 == 1]
    elif f.get("ema_trend") == "bear":
        result = [t for t in result if t.ema50_gt_200 == 0]

    if f.get("above_ema200"):
        result = [t for t in result if t.above_ema200 == 1]

    if f.get("above_ema50"):
        result = [t for t in result if t.above_ema50 == 1]

    if f.get("fresh_cross"):
        result = [t for t in result if t.fresh_cross == 1]

    if f.get("below_vwap"):
        result = [t for t in result if t.below_vwap == 1]

    if f.get("rsi_max") is not None:
        result = [t for t in result if t.rsi < f["rsi_max"]]

    if f.get("rsi_min") is not None:
        result = [t for t in result if t.rsi >= f["rsi_min"]]

    if f.get("bb_pct_max") is not None:
        result = [t for t in result if t.bb_pct < f["bb_pct_max"]]

    if f.get("min_score") is not None:
        result = [t for t in result if t.score >= f["min_score"]]

    if f.get("vol_spike_min") is not None:
        result = [t for t in result if t.vol_spike >= f["vol_spike_min"]]

    if f.get("exclude_1bar"):
        result = [t for t in result if not (not t.tp_hit and not t.stop_hit and t.bars_held < 2)]

    return result


def wr(trades):
    if not trades: return 0
    return sum(1 for t in trades if t.win) / len(trades) * 100


def pf(trades):
    wins   = sum(t.pnl for t in trades if t.win)
    losses = abs(sum(t.pnl for t in trades if not t.win))
    return wins / (losses + 1e-9)


def optimize_filters(trades):
    """Test every meaningful combination, rank by win rate with min sample size."""
    print("\nRunning filter optimizer (this may take ~60 seconds)...")
    base_wr = wr(trades)
    print(f"  Baseline: {len(trades):,} trades, WR={base_wr:.1f}%")

    results = []

    # Define all individual filter options
    hour_ranges    = [None, (7,10), (8,10), (13,17), (14,16), (7,10), (13,17)]
    session_opts   = [
        None,
        [(7,0,10,0),(13,30,16,30)],   # London + NY open
        [(7,0,10,0)],                  # London only
        [(13,0,17,0)],                 # NY only
        [(7,0,12,0)],                  # full morning
    ]
    trend_opts     = [None, "bull", "bear"]
    ema200_opts    = [False, True]
    fresh_opts     = [False, True]
    vwap_opts      = [False, True]
    rsi_max_opts   = [None, 40, 45, 50]
    bb_pct_opts    = [None, 0.2, 0.35, 0.5]
    score_opts     = [None, 2.5, 3.0, 3.5]
    weekday_opts   = [None, [0,1,2], [1,2,3], [0,1,2,3,4]]
    vol_opts       = [None, 1.5, 2.0]
    exclude1_opts  = [False, True]

    combo_count = 0

    # Efficient: test singles first, then pairs, then triples
    # Singles
    single_filters = []
    for sess in session_opts[1:]:
        single_filters.append({"session": sess})
    for hr in [(7,10),(8,10),(13,17),(14,16),(7,12)]:
        single_filters.append({"hour_range": hr})
    for tr in ["bull","bear"]:
        single_filters.append({"ema_trend": tr})
    single_filters.append({"above_ema200": True})
    single_filters.append({"fresh_cross": True})
    single_filters.append({"below_vwap": True})
    for r in [40,45,50]:
        single_filters.append({"rsi_max": r})
    for b in [0.2,0.35]:
        single_filters.append({"bb_pct_max": b})
    for s in [2.5,3.0,3.5]:
        single_filters.append({"min_score": s})
    for wd in [[0,1,2],[1,2,3]]:
        single_filters.append({"weekdays": wd})
    single_filters.append({"exclude_1bar": True})

    # Test all singles
    for f in single_filters:
        sub = apply_filter(trades, f)
        if len(sub) >= MIN_TRADES_FOR_VALID_FILTER:
            results.append({"filters": f, "n": len(sub), "wr": wr(sub), "pf": pf(sub)})
        combo_count += 1

    print(f"  Singles done ({combo_count} combos tested)...")

    # Pairs — combine each single with every other single
    pair_results = []
    for i, f1 in enumerate(single_filters):
        for j, f2 in enumerate(single_filters):
            if j <= i: continue
            # skip conflicting keys
            if set(f1.keys()) & set(f2.keys()): continue
            combined = {**f1, **f2}
            sub = apply_filter(trades, combined)
            if len(sub) >= MIN_TRADES_FOR_VALID_FILTER:
                w = wr(sub)
                pair_results.append({"filters": combined, "n": len(sub), "wr": w, "pf": pf(sub)})
            combo_count += 1

    results.extend(pair_results)
    print(f"  Pairs done ({combo_count} combos tested)...")

    # Triples — take top 20 pairs and add one more filter
    top_pairs = sorted(pair_results, key=lambda x: -x["wr"])[:20]
    for pr in top_pairs:
        for f3 in single_filters:
            if set(pr["filters"].keys()) & set(f3.keys()): continue
            combined = {**pr["filters"], **f3}
            sub = apply_filter(trades, combined)
            if len(sub) >= MIN_TRADES_FOR_VALID_FILTER:
                w = wr(sub)
                results.append({"filters": combined, "n": len(sub), "wr": w, "pf": pf(sub)})
            combo_count += 1

    print(f"  Triples done ({combo_count} combos tested)")

    results_sorted = sorted(results, key=lambda x: -x["wr"])
    return results_sorted


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def compute_stats(trades):
    if not trades: return {}
    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    pnls   = [t.pnl for t in trades]
    # simulate equity curve
    equity = ACCOUNT_SIZE
    peak   = ACCOUNT_SIZE
    max_dd = 0.0
    for t in trades:
        equity += equity * (t.pnl / 100)  # pnl stored as %
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak * 100
        max_dd  = max(max_dd, dd)

    return {
        "total":         len(trades),
        "win_rate":      len(wins) / len(trades) * 100,
        "total_pnl_pct": sum(pnls),
        "avg_win_pct":   float(np.mean([t.pnl for t in wins]))   if wins   else 0,
        "avg_loss_pct":  float(np.mean([t.pnl for t in losses])) if losses else 0,
        "profit_factor": abs(sum(t.pnl for t in wins)) / (abs(sum(t.pnl for t in losses)) + 1e-9),
        "max_dd_pct":    max_dd,
        "tp_rate":       len([t for t in trades if t.tp_hit]) / len(trades) * 100,
        "sl_rate":       len([t for t in trades if t.stop_hit]) / len(trades) * 100,
        "avg_bars":      float(np.mean([t.bars_held for t in trades])),
        "final_equity":  equity,
    }


# ─────────────────────────────────────────────
# PRINT REPORTS
# ─────────────────────────────────────────────
def print_stats(label, s):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades       : {s['total']:,}")
    print(f"  Win rate     : {s['win_rate']:.1f}%")
    print(f"  Profit factor: {s['profit_factor']:.2f}")
    print(f"  Max drawdown : {s['max_dd_pct']:.2f}%")
    print(f"  TP rate      : {s['tp_rate']:.1f}%")
    print(f"  SL rate      : {s['sl_rate']:.1f}%")
    print(f"  Avg hold     : {s['avg_bars']:.1f} bars ({s['avg_bars']*5:.0f} min)")
    print(f"{'='*60}")


def print_top_filters(results, n=15):
    print(f"\n{'='*60}")
    print(f"  TOP {n} FILTER COMBINATIONS (min {MIN_TRADES_FOR_VALID_FILTER} trades)")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'WR%':>6} {'PF':>5} {'Trades':>7}  Filters")
    print(f"  {'-'*55}")
    for rank, r in enumerate(results[:n], 1):
        tag = " ← 60%+ TARGET" if r["wr"] >= 60 else (" ← close!" if r["wr"] >= 55 else "")
        print(f"  #{rank:<4} {r['wr']:>5.1f}% {r['pf']:>5.2f} {r['n']:>7,}  {r['filters']}{tag}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def make_chart(trades_base, trades_best, stats_base, stats_best, top_results):
    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f1a")
    fig.suptitle("BTC Scalping Backtest — Binance Historical Data (Filter Optimizer)",
                 fontsize=14, color="white", fontweight="bold")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)
    tc = "#e0e0e0"; gc = "#2a2a3a"; gold = "#FFD700"; teal = "#00e5cc"
    grn = "#00e676"; red = "#ff1744"

    def style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color=tc, fontsize=9, fontweight="bold")
        ax.tick_params(colors=tc, labelsize=7)
        ax.grid(color=gc, linestyle="--", linewidth=0.4, alpha=0.6)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    # ── Row 0: equity curves ──────────────────
    ax0 = fig.add_subplot(gs[0, :3])
    style(ax0, "Equity Curve — Baseline vs Best Filter")
    def sim_equity(trades):
        eq = [ACCOUNT_SIZE]
        cur = ACCOUNT_SIZE
        for t in trades:
            cur += cur * t.pnl / 100
            eq.append(max(cur, 0))
        return eq
    eq_b = sim_equity(trades_base)
    eq_a = sim_equity(trades_best)
    ax0.plot(range(len(eq_b)), eq_b, color=gold, linewidth=1.0, label="Baseline", alpha=0.6)
    ax0.plot(range(len(eq_a)), eq_a, color=teal, linewidth=1.8, label="Best filter")
    ax0.axhline(ACCOUNT_SIZE, color=tc, linestyle="--", linewidth=0.6, alpha=0.3)
    ax0.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=tc)
    ax0.set_ylabel("Equity ($)", color=tc, fontsize=8)
    ax0.set_xlabel("Trade #", color=tc, fontsize=8)

    # ── Stats box ────────────────────────────
    ax_s = fig.add_subplot(gs[0, 3])
    ax_s.set_facecolor("#1a1a2e"); ax_s.axis("off")
    ax_s.set_title("Comparison", color=tc, fontsize=9, fontweight="bold")
    rows = [
        ("", "Baseline", "Best"),
        ("Trades", f"{stats_base['total']:,}", f"{stats_best['total']:,}"),
        ("Win rate", f"{stats_base['win_rate']:.1f}%", f"{stats_best['win_rate']:.1f}%"),
        ("Prof factor", f"{stats_base['profit_factor']:.2f}", f"{stats_best['profit_factor']:.2f}"),
        ("Max DD", f"{stats_base['max_dd_pct']:.2f}%", f"{stats_best['max_dd_pct']:.2f}%"),
        ("TP rate", f"{stats_base['tp_rate']:.1f}%", f"{stats_best['tp_rate']:.1f}%"),
        ("SL rate", f"{stats_base['sl_rate']:.1f}%", f"{stats_best['sl_rate']:.1f}%"),
        ("Avg hold", f"{stats_base['avg_bars']:.1f}b", f"{stats_best['avg_bars']:.1f}b"),
    ]
    for j, (m, b, a) in enumerate(rows):
        y = 0.96 - j * 0.115
        hdr = j == 0
        ax_s.text(0.01, y, m,  transform=ax_s.transAxes, color=tc if not hdr else gold,
                  fontsize=8, va="top", fontweight="bold" if hdr else "normal")
        ax_s.text(0.45, y, b, transform=ax_s.transAxes, color=gold, fontsize=8, va="top")
        ax_s.text(0.74, y, a, transform=ax_s.transAxes, color=teal, fontsize=8, va="top", fontweight="bold")

    # ── Row 1: hour win rate ──────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    style(ax1, "Win rate by hour (baseline)")
    hours = range(7, 21)
    h_wr  = []
    h_cnt = []
    for h in hours:
        sub = [t for t in trades_base if t.hour == h]
        h_wr.append(wr(sub) if sub else 0)
        h_cnt.append(len(sub))
    colors_h = [grn if w >= 45 else red if w < 35 else gold for w in h_wr]
    ax1.bar([str(h) for h in hours], h_wr, color=colors_h, alpha=0.8)
    ax1.axhline(wr(trades_base), color=tc, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("WR %", color=tc, fontsize=7)

    # ── Row 1: WR by EMA trend ───────────────
    ax2 = fig.add_subplot(gs[1, 1])
    style(ax2, "Win rate: bull vs bear trend")
    bull = [t for t in trades_base if t.ema50_gt_200 == 1]
    bear = [t for t in trades_base if t.ema50_gt_200 == 0]
    ax2.bar(["Bull\n(EMA50>200)", "Bear\n(EMA50<200)"],
            [wr(bull), wr(bear)], color=[grn, red], alpha=0.8)
    ax2.text(0, wr(bull)+0.5, f"n={len(bull):,}", ha="center", va="bottom", color=tc, fontsize=7)
    ax2.text(1, wr(bear)+0.5, f"n={len(bear):,}", ha="center", va="bottom", color=tc, fontsize=7)
    ax2.axhline(50, color=tc, linestyle="--", linewidth=0.6, alpha=0.4)
    ax2.set_ylabel("WR %", color=tc, fontsize=7)

    # ── Row 1: WR by RSI ─────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    style(ax3, "Win rate by RSI zone")
    rsi_bins = [(20,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,70)]
    r_wr = []; r_lab = []
    for lo,hi in rsi_bins:
        sub = [t for t in trades_base if lo <= t.rsi < hi]
        if len(sub) > 20:
            r_wr.append(wr(sub)); r_lab.append(f"{lo}-{hi}\n({len(sub)})")
    colors_r = [grn if w >= 45 else red if w < 35 else gold for w in r_wr]
    ax3.bar(r_lab, r_wr, color=colors_r, alpha=0.8)
    ax3.axhline(50, color=tc, linestyle="--", linewidth=0.6, alpha=0.4)
    ax3.set_ylabel("WR %", color=tc, fontsize=7)

    # ── Row 1: Top filter WRs ─────────────────
    ax4 = fig.add_subplot(gs[1, 3])
    style(ax4, "Top 10 filter combos")
    top10 = top_results[:10]
    labels = [f"#{i+1}" for i in range(len(top10))]
    vals   = [r["wr"] for r in top10]
    colors_t = [grn if v >= 60 else gold if v >= 50 else red for v in vals]
    bars = ax4.barh(labels[::-1], vals[::-1], color=colors_t[::-1], alpha=0.8)
    ax4.axvline(60, color=teal, linestyle="--", linewidth=1.0, alpha=0.7)
    ax4.axvline(50, color=tc,   linestyle="--", linewidth=0.6, alpha=0.4)
    ax4.set_xlabel("Win rate %", color=tc, fontsize=7)

    # ── Row 2: PnL distribution ───────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style(ax5, "PnL distribution — best filter")
    pnls = [t.pnl for t in trades_best]
    if pnls:
        bins = np.linspace(min(pnls), max(pnls), 30)
        ax5.hist([p for p in pnls if p >= 0], bins=bins, color=grn, alpha=0.7, label="Wins")
        ax5.hist([p for p in pnls if p <  0], bins=bins, color=red,  alpha=0.7, label="Losses")
        ax5.legend(fontsize=7, facecolor="#1a1a2e", labelcolor=tc)
        ax5.set_xlabel("PnL %", color=tc, fontsize=7)

    # ── Row 2: WR by bb_pct ──────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    style(ax6, "Win rate by Bollinger %")
    bb_bins = [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,1.0)]
    b_wr = []; b_lab = []
    for lo,hi in bb_bins:
        sub = [t for t in trades_base if lo <= t.bb_pct < hi]
        if len(sub) > 20:
            b_wr.append(wr(sub)); b_lab.append(f"{lo:.1f}-{hi:.1f}")
    colors_b = [grn if w >= 45 else red if w < 35 else gold for w in b_wr]
    ax6.bar(b_lab, b_wr, color=colors_b, alpha=0.8)
    ax6.axhline(50, color=tc, linestyle="--", linewidth=0.6, alpha=0.4)
    ax6.set_ylabel("WR %", color=tc, fontsize=7)
    ax6.tick_params(axis='x', labelsize=6, rotation=30)

    # ── Row 2: Fresh cross analysis ──────────
    ax7 = fig.add_subplot(gs[2, 2])
    style(ax7, "Fresh cross vs continuing")
    fresh = [t for t in trades_base if t.fresh_cross == 1]
    cont  = [t for t in trades_base if t.fresh_cross == 0]
    ax7.bar(["Fresh cross\n(new signal)", "Continuing\ncross"],
            [wr(fresh), wr(cont)], color=[teal, gold], alpha=0.8)
    ax7.text(0, wr(fresh)+0.5, f"n={len(fresh):,}", ha="center", va="bottom", color=tc, fontsize=7)
    ax7.text(1, wr(cont)+0.5,  f"n={len(cont):,}",  ha="center", va="bottom", color=tc, fontsize=7)
    ax7.axhline(50, color=tc, linestyle="--", linewidth=0.6, alpha=0.4)
    ax7.set_ylabel("WR %", color=tc, fontsize=7)

    # ── Row 2: Weekday ────────────────────────
    ax8 = fig.add_subplot(gs[2, 3])
    style(ax8, "Win rate by weekday")
    day_names = ["Mon","Tue","Wed","Thu","Fri"]
    d_wr = []
    for di in range(5):
        sub = [t for t in trades_base if t.weekday == di]
        d_wr.append(wr(sub) if sub else 0)
    colors_d = [grn if w >= 45 else red if w < 35 else gold for w in d_wr]
    ax8.bar(day_names, d_wr, color=colors_d, alpha=0.8)
    ax8.axhline(wr(trades_base), color=tc, linestyle="--", linewidth=0.8, alpha=0.5)
    ax8.set_ylabel("WR %", color=tc, fontsize=7)

    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("\nChart saved: backtest_results.png")


# ─────────────────────────────────────────────
# SAVE TRADES CSV
# ─────────────────────────────────────────────
def save_trades_csv(trades, filename="all_trades.csv"):
    if not trades: return
    rows = [asdict(t) for t in trades]
    df   = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Trades saved: {filename} ({len(trades):,} rows)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BTC Scalping Backtest — Binance Data + Filter Optimizer")
    print("=" * 60)

    # 1. Find and load data
    print("\nSearching for CSV files...")
    files = find_csv_files()
    if not files:
        print(f"\nERROR: No CSV files found.")
        print(f"Expected location: {DATA_DIR}")
        print("Run the downloader first: py download_binance.py")
        return

    print(f"Found {len(files)} CSV files")
    df_raw = load_binance_data(files)

    # 2. Build indicators
    print("\nBuilding indicators...")
    df = build_indicators(df_raw)
    print(f"  Ready: {len(df):,} bars with indicators")

    # 3. Run baseline backtest
    trades = run_backtest(df)
    if not trades:
        print("No trades generated — check data or signal parameters")
        return

    stats_base = compute_stats(trades)
    print_stats("BASELINE (no extra filters)", stats_base)

    # 4. Save all trades to CSV for your own analysis
    save_trades_csv(trades)

    # 5. Run filter optimizer
    top_results = optimize_filters(trades)

    # 6. Print top combos
    print_top_filters(top_results, n=20)

    # 7. Apply best filter and show stats
    best = top_results[0]
    print(f"\nBest filter found:")
    print(f"  {best['filters']}")
    print(f"  Win rate : {best['wr']:.1f}%")
    print(f"  Trades   : {best['n']:,}")
    print(f"  PF       : {best['pf']:.2f}")

    trades_best = apply_filter(trades, best["filters"])
    stats_best  = compute_stats(trades_best)
    print_stats(f"BEST FILTER — {best['filters']}", stats_best)

    # Show all combos that hit 60%+
    hits_60 = [r for r in top_results if r["wr"] >= 60]
    if hits_60:
        print(f"\n{'='*60}")
        print(f"  FILTERS ACHIEVING 60%+ WIN RATE ({len(hits_60)} found)")
        print(f"{'='*60}")
        for r in hits_60:
            print(f"  WR={r['wr']:.1f}%  n={r['n']:,}  PF={r['pf']:.2f}  {r['filters']}")
    else:
        print(f"\nNo single combo hit 60%+ with {MIN_TRADES_FOR_VALID_FILTER}+ trades.")
        print("Best achieved:", top_results[0]["wr"] if top_results else "N/A")
        print(f"Closest results:")
        for r in top_results[:5]:
            print(f"  WR={r['wr']:.1f}%  n={r['n']:,}  {r['filters']}")

    # 8. Save best filters to JSON
    with open("best_filters.json", "w") as f:
        json.dump(top_results[:50], f, indent=2, default=str)
    print("\nTop 50 filter combos saved: best_filters.json")

    # 9. Chart
    print("Generating charts...")
    make_chart(trades, trades_best, stats_base, stats_best, top_results)

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
    print("  backtest_results.png  — visual analysis")
    print("  best_filters.json     — top 50 filter combos")
    print("  all_trades.csv        — every trade with features")
    print("="*60)


if __name__ == "__main__":
    main()
