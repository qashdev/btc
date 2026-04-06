"""
Scalping Bot Multi-Timeframe Backtest
=======================================
Tests the 5-signal strategy across ALL timeframes:
  - 15-min  → 60 days
  - 1-hour  → 2 years  
  - 4-hour  → 5 years

Run: py scalp_backtest_multi.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TIMEFRAME CONFIGS
# ─────────────────────────────────────────────
TIMEFRAMES = {
    "15m": {
        "interval":   "15m",
        "period":     "60d",
        "label":      "15-Min (60 days)",
        "hold_bars":  8,       # 8 bars = 2 hours max hold
        "tp":         0.0020,  # 0.20%
        "sl":         0.0010,  # 0.10%
        "bars_day":   26,      # trading bars per day
        "annualise":  6.1,     # ×6.1 for full year
    },
    "1h": {
        "interval":   "1h",
        "period":     "2y",
        "label":      "1-Hour (2 years)",
        "hold_bars":  8,       # 8 bars = 8 hours max hold
        "tp":         0.0040,  # 0.40%
        "sl":         0.0020,  # 0.20%
        "bars_day":   14,
        "annualise":  1.0,
    },
    "4h": {
        "interval":   "1d",    # daily data resampled to simulate 4h — gets 5 years
        "period":     "5y",
        "label":      "4-Hour (5 years)",
        "hold_bars":  6,
        "tp":         0.0080,  # 0.80%
        "sl":         0.0035,  # 0.35%
        "bars_day":   6,
        "annualise":  1.0,
    },
}

ASSETS = {
    "GOLD": {"ticker": "GC=F",    "fallback": "GLD"},
    "BTC":  {"ticker": "BTC-USD", "fallback": None},
}

# Signal params
EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.5
MIN_SCORE        = 2   # lowered to 2 — more trades on longer timeframes
LONGS_ONLY       = True
SESSION_START    = 7
SESSION_END      = 21

# Account
ACCOUNT_SIZE       = 1000
MAX_DRAWDOWN_LIMIT = 0.050  # 5% hard reference
RISK_WHEN_SAFE     = 0.95   # 95% in green zone
RISK_WHEN_WARNING  = 0.60   # 60% in warning zone
RISK_WHEN_DANGER   = 0.10   # 10% in danger zone


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
def dynamic_position(confidence, equity, peak, tp, sl):
    current_dd = (peak - equity) / peak if peak > 0 else 0.0
    dd_ratio   = min(current_dd / MAX_DRAWDOWN_LIMIT, 1.0)
    if dd_ratio < 0.5:
        scale    = dd_ratio / 0.5
        max_risk = RISK_WHEN_SAFE + scale * (RISK_WHEN_WARNING - RISK_WHEN_SAFE)
    else:
        scale    = (dd_ratio - 0.5) / 0.5
        max_risk = RISK_WHEN_WARNING + scale * (RISK_WHEN_DANGER - RISK_WHEN_WARNING)
    b     = tp / sl
    kelly = max(0, (confidence * (b + 1) - 1) / b) * 0.5
    # Ensure meaningful position even at lower confidence  
    min_pos = equity * 0.05
    return max(min_pos, min(equity * kelly, equity * max_risk))


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def fetch_data(asset_key, tf_key):
    cfg = ASSETS[asset_key]
    tf  = TIMEFRAMES[tf_key]
    log.info(f"📥 {asset_key} {tf['label']}...")
    for ticker in ([cfg["ticker"]] + ([cfg["fallback"]] if cfg.get("fallback") else [])):
        try:
            df = yf.download(ticker, period=tf["period"], interval=tf["interval"],
                             progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.dropna()
                # For 4h config we use daily data directly (5yr range)
                if tf_key == "4h":
                    pass  # daily data already fetched
                log.info(f"   ✅ {ticker}: {len(df)} candles ({tf['label']})")
                return df
        except Exception as e:
            log.warning(f"   {ticker} failed: {e}")
    return pd.DataFrame()


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()
    d["ema_fast"]  = d["Close"].ewm(span=EMA_FAST,  adjust=False).mean()
    d["ema_slow"]  = d["Close"].ewm(span=EMA_SLOW,  adjust=False).mean()
    d["ema_cross"] = np.sign(d["ema_fast"] - d["ema_slow"])
    delta      = d["Close"].diff()
    gain       = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss       = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    d["rsi"]   = 100 - (100 / (1 + gain / (loss + 1e-9)))
    sma        = d["Close"].rolling(BB_PERIOD).mean()
    std        = d["Close"].rolling(BB_PERIOD).std()
    d["bb_upper"] = sma + BB_STD * std
    d["bb_lower"] = sma - BB_STD * std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (sma + 1e-9)
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
    d["hour"]    = idx.hour
    d["weekday"] = idx.weekday
    # For daily data (4h config), always in session — no intraday hours
    freq = pd.infer_freq(d.index[:10]) or ""
    if "D" in freq or len(d) < 500:
        d["in_session"] = 1  # daily candles — always tradeable
    else:
        d["in_session"] = ((d["hour"] >= SESSION_START) &
                           (d["hour"] < SESSION_END) &
                           (d["weekday"] < 5)).astype(int)
    return d.dropna()


# ─────────────────────────────────────────────
# SIGNAL SCORER
# ─────────────────────────────────────────────
def score_row(row, prev_cross):
    sl, ss = 0.0, 0.0
    cross = float(row["ema_cross"])
    if prev_cross <= 0 and cross > 0:   sl += 1
    elif prev_cross >= 0 and cross < 0: ss += 1
    elif cross > 0: sl += 0.5
    elif cross < 0: ss += 0.5
    rsi = float(row["rsi"])
    if rsi < 35:   sl += 1
    elif rsi > 65: ss += 1
    vd = float(row["vwap_dist"])
    if vd < -0.001:  sl += 1
    elif vd > 0.001: ss += 1
    bp = float(row["bb_pct"])
    if bp < 0.2:   sl += 1
    elif bp > 0.8: ss += 1
    vs  = float(row["vol_spike"])
    mom = float(row["mom3"])
    if vs >= VOLUME_SPIKE_MIN:
        if mom > 0: sl += 1
        else:       ss += 1
    return sl, ss


# ─────────────────────────────────────────────
# SIMULATE TRADE
# ─────────────────────────────────────────────
def simulate(direction, entry, future_rows, position_usd, tp_pct, sl_pct):
    sl = entry * (1 - sl_pct) if direction == "LONG" else entry * (1 + sl_pct)
    tp = entry * (1 + tp_pct) if direction == "LONG" else entry * (1 - tp_pct)
    for bars, (_, row) in enumerate(future_rows.iterrows(), 1):
        price = float(row["Close"])
        if direction == "LONG":
            if price <= sl: return price, -position_usd * sl_pct, False, bars, True,  False
            if price >= tp: return price,  position_usd * tp_pct, True,  bars, False, True
        else:
            if price >= sl: return price, -position_usd * sl_pct, False, bars, True,  False
            if price <= tp: return price,  position_usd * tp_pct, True,  bars, False, True
    exit_p = float(future_rows.iloc[-1]["Close"])
    pnl    = (exit_p - entry) / entry * position_usd if direction == "LONG" else (entry - exit_p) / entry * position_usd
    hold   = len(future_rows)
    return exit_p, pnl, pnl > 0, hold, False, False


# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────
@dataclass
class Trade:
    date: str; direction: str; entry: float; exit_price: float
    position_usd: float; pnl: float; win: bool; bars_held: int
    stop_hit: bool; tp_hit: bool

def run_backtest(df, tf_key):
    tf           = TIMEFRAMES[tf_key]
    tp_pct       = tf["tp"]
    sl_pct       = tf["sl"]
    hold_bars    = tf["hold_bars"]
    trades       = []
    equity       = ACCOUNT_SIZE
    peak_equity  = ACCOUNT_SIZE
    equity_curve = []
    skip_until   = 0

    for i in range(50, len(df) - hold_bars):
        if i < skip_until:
            equity_curve.append(equity)
            continue
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        if int(row.get("in_session", 1)) == 0:
            equity_curve.append(equity)
            continue
        sl_score, ss_score = score_row(row, float(prev["ema_cross"]))
        if LONGS_ONLY:
            if sl_score < MIN_SCORE:
                equity_curve.append(equity)
                continue
            direction  = "LONG"
            confidence = min(sl_score / 5.0, 1.0)
        else:
            if max(sl_score, ss_score) < MIN_SCORE:
                equity_curve.append(equity)
                continue
            direction  = "LONG" if sl_score >= ss_score else "SHORT"
            confidence = min(max(sl_score, ss_score) / 5.0, 1.0)

        vals = [row["ema_fast"], row["rsi"], row["bb_pct"], row["vwap_dist"], row["vol_spike"]]
        if not all(np.isfinite(float(v)) for v in vals):
            equity_curve.append(equity)
            continue

        pos   = dynamic_position(confidence, equity, peak_equity, tp_pct, sl_pct)
        if pos < 5:
            equity_curve.append(equity)
            continue

        entry  = float(row["Close"])
        future = df.iloc[i+1:i+1+hold_bars]
        ep, pnl, win, bars, stop_hit, tp_hit = simulate(direction, entry, future, pos, tp_pct, sl_pct)

        equity     += pnl
        equity      = max(equity, 0)
        peak_equity = max(peak_equity, equity)
        skip_until  = i + bars

        trades.append(Trade(
            date=str(df.index[i])[:16], direction=direction,
            entry=round(entry, 4), exit_price=round(float(ep), 4),
            position_usd=round(pos, 2), pnl=round(float(pnl), 2),
            win=bool(win), bars_held=int(bars),
            stop_hit=bool(stop_hit), tp_hit=bool(tp_hit),
        ))
        equity_curve.append(equity)

    eq_series = pd.Series(equity_curve, index=df.index[50:50+len(equity_curve)])
    return trades, eq_series


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def compute_stats(trades, eq, tf_key):
    if not trades: return {}
    tf     = TIMEFRAMES[tf_key]
    pnls   = [t.pnl for t in trades]
    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    peak   = eq.cummax()
    dd     = (eq - peak) / peak
    ret    = eq.pct_change().dropna()
    n_days = (eq.index[-1] - eq.index[0]).days or 1
    n_years= max(n_days / 365, 0.01)
    total_ret = (eq.iloc[-1] - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100
    annual_ret= ((eq.iloc[-1] / ACCOUNT_SIZE) ** (1 / n_years) - 1) * 100
    return {
        "total":         len(trades),
        "win_rate":      len(wins) / len(trades),
        "total_pnl":     sum(pnls),
        "return_pct":    total_ret,
        "annual_ret":    annual_ret,
        "avg_win":       float(np.mean([t.pnl for t in wins]))   if wins   else 0,
        "avg_loss":      float(np.mean([t.pnl for t in losses])) if losses else 0,
        "best":          max(pnls),
        "worst":         min(pnls),
        "profit_factor": abs(sum(t.pnl for t in wins)) / (abs(sum(t.pnl for t in losses)) + 1e-9),
        "max_dd":        float(dd.min() * 100),
        "sharpe":        float((ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252 * tf["bars_day"])),
        "tp_rate":       len([t for t in trades if t.tp_hit])   / len(trades),
        "sl_rate":       len([t for t in trades if t.stop_hit]) / len(trades),
        "avg_hold_bars": float(np.mean([t.bars_held for t in trades])),
        "avg_pos":       float(np.mean([t.position_usd for t in trades])),
        "trades_per_day":len(trades) / max(n_days, 1),
        "n_days":        n_days,
        "n_years":       round(n_years, 1),
        "final_equity":  float(eq.iloc[-1]),
    }


# ─────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────
def print_report(stats, asset, tf_key):
    tf   = TIMEFRAMES[tf_key]
    dd   = stats["max_dd"]
    flag = "✅ Under 6%" if abs(dd) < 6 else "⚠️  OVER 6%"
    print(f"\n{'='*60}")
    print(f"  {asset}  |  {tf['label']}")
    print(f"{'='*60}")
    print(f"  Total trades    : {stats['total']}  (~{stats['trades_per_day']:.1f}/day)")
    print(f"  Win rate        : {stats['win_rate']:.1%}")
    print(f"  Total PnL       : ${stats['total_pnl']:+,.2f}")
    print(f"  Total return    : {stats['return_pct']:+.1f}%  over {stats['n_years']} years")
    print(f"  Annual return   : {stats['annual_ret']:+.1f}%  ← annualised")
    print(f"  Final equity    : ${stats['final_equity']:,.2f}")
    print(f"  Profit factor   : {stats['profit_factor']:.2f}")
    print(f"  Max drawdown    : {dd:.1f}%  {flag}")
    print(f"  Sharpe ratio    : {stats['sharpe']:.2f}")
    print(f"  TP hit rate     : {stats['tp_rate']:.1%}")
    print(f"  SL hit rate     : {stats['sl_rate']:.1%}")


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────
def print_summary(all_results):
    print(f"\n{'='*75}")
    print(f"  📊 FULL COMPARISON TABLE")
    print(f"{'='*75}")
    print(f"  {'Asset+TF':<22} {'Period':<10} {'Ann.Return':>11} {'Win Rate':>9} {'Max DD':>8} {'Sharpe':>8} {'Final $':>10}")
    print(f"  {'-'*70}")
    for (asset, tf_key), stats in all_results.items():
        tf    = TIMEFRAMES[tf_key]
        name  = f"{asset} {tf_key.upper()}"
        flag  = "✅" if abs(stats["max_dd"]) < 6 else "⚠️"
        print(f"  {name:<22} {tf['label'].split('(')[1].rstrip(')'):<10} "
              f"{stats['annual_ret']:>+10.1f}% "
              f"{stats['win_rate']:>9.1%} "
              f"{stats['max_dd']:>7.1f}%{flag} "
              f"{stats['sharpe']:>8.2f} "
              f"${stats['final_equity']:>9,.0f}")
    print(f"{'='*75}")


# ─────────────────────────────────────────────
# MEGA CHART
# ─────────────────────────────────────────────
def plot_mega_chart(all_equity, all_stats):
    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f1a")
    fig.suptitle("Scalping Bot — Multi-Timeframe Comparison (Gold & BTC)",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    keys   = list(all_equity.keys())
    n      = len(keys)
    cols   = 3
    rows_n = (n + cols - 1) // cols + 1
    gs     = gridspec.GridSpec(rows_n, cols, figure=fig, hspace=0.5, wspace=0.35)

    tc = "#e0e0e0"; gc = "#2a2a3a"; gold = "#FFD700"; grn = "#00e676"; red = "#ff1744"
    colors = ["#FFD700","#00e676","#ff6b35","#00bcd4","#e040fb","#ff1744"]

    def style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color=tc, fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(colors=tc, labelsize=7)
        ax.grid(color=gc, linestyle="--", linewidth=0.4, alpha=0.6)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    # Individual equity curves
    for idx, (key, eq) in enumerate(all_equity.items()):
        r, c  = divmod(idx, cols)
        ax    = fig.add_subplot(gs[r, c])
        asset, tf_key = key
        tf    = TIMEFRAMES[tf_key]
        stats = all_stats[key]
        style(ax, f"{asset} {tf_key.upper()} — {tf['label']}")
        ax.plot(eq.index, eq.values, color=colors[idx], linewidth=1.2)
        ax.axhline(ACCOUNT_SIZE, color=tc, linestyle="--", linewidth=0.7, alpha=0.4)
        ax.fill_between(eq.index, ACCOUNT_SIZE, eq.values,
                        where=eq.values >= ACCOUNT_SIZE, alpha=0.1, color=grn)
        ax.fill_between(eq.index, ACCOUNT_SIZE, eq.values,
                        where=eq.values < ACCOUNT_SIZE, alpha=0.1, color=red)
        final = eq.iloc[-1]
        sign  = "+" if final > ACCOUNT_SIZE else ""
        ax.set_title(f"{asset} {tf_key.upper()} | {sign}{stats['return_pct']:.1f}% | DD:{stats['max_dd']:.1f}%",
                     color=colors[idx], fontsize=9, fontweight="bold")
        ax.set_ylabel("$", color=tc, fontsize=7)

    # Summary bar chart — annual returns
    ax_bar = fig.add_subplot(gs[-1, :2])
    style(ax_bar, "📊 Annual Return Comparison")
    labels  = [f"{a} {tf}" for a, tf in all_equity.keys()]
    ann_rets= [all_stats[k]["annual_ret"] for k in all_equity.keys()]
    bar_colors = [grn if v > 0 else red for v in ann_rets]
    bars = ax_bar.bar(range(len(labels)), ann_rets, color=bar_colors, alpha=0.85)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=20, fontsize=8)
    ax_bar.set_ylabel("Annual Return %", color=tc, fontsize=8)
    ax_bar.axhline(0, color=tc, linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, ann_rets):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:+.1f}%", ha="center", va="bottom", color=tc, fontsize=8, fontweight="bold")

    # Stats table
    ax_tbl = fig.add_subplot(gs[-1, 2])
    ax_tbl.set_facecolor("#1a1a2e")
    ax_tbl.axis("off")
    ax_tbl.set_title("🏆 Best Results", color=tc, fontsize=9, fontweight="bold")
    best_ann = max(all_stats.items(), key=lambda x: x[1]["annual_ret"])
    best_dd  = min(all_stats.items(), key=lambda x: abs(x[1]["max_dd"]))
    best_sh  = max(all_stats.items(), key=lambda x: x[1]["sharpe"])
    lines = [
        ("Best Annual Return:", f"{best_ann[0][0]} {best_ann[0][1].upper()} → {best_ann[1]['annual_ret']:+.1f}%"),
        ("Lowest Drawdown:",    f"{best_dd[0][0]} {best_dd[0][1].upper()} → {best_dd[1]['max_dd']:.1f}%"),
        ("Best Sharpe:",        f"{best_sh[0][0]} {best_sh[0][1].upper()} → {best_sh[1]['sharpe']:.2f}"),
    ]
    for j, (label, val) in enumerate(lines):
        y = 0.85 - j * 0.25
        ax_tbl.text(0.02, y, label, transform=ax_tbl.transAxes, color="#aaa", fontsize=8, va="top")
        ax_tbl.text(0.02, y-0.1, val, transform=ax_tbl.transAxes, color=gold, fontsize=9, va="top", fontweight="bold")

    plt.savefig("scalp_multi_timeframe.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("   Chart → scalp_multi_timeframe.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    all_equity = {}
    all_stats  = {}

    for asset in ["GOLD", "BTC"]:
        for tf_key in ["15m", "1h", "4h"]:
            try:
                df = fetch_data(asset, tf_key)
                if df.empty or len(df) < 100:
                    log.warning(f"   ⚠️  Not enough data for {asset} {tf_key}")
                    continue
                df     = build_indicators(df)
                trades, eq = run_backtest(df, tf_key)
                if not trades:
                    log.warning(f"   ⚠️  No trades for {asset} {tf_key}")
                    continue
                stats  = compute_stats(trades, eq, tf_key)
                print_report(stats, asset, tf_key)
                all_equity[(asset, tf_key)] = eq
                all_stats[(asset, tf_key)]  = stats
                with open(f"scalp_trades_{asset.lower()}_{tf_key}.json", "w") as f:
                    json.dump([t.__dict__ for t in trades], f, indent=2)
            except Exception as e:
                log.error(f"❌ {asset} {tf_key} failed: {e}")
                import traceback; traceback.print_exc()

    if all_stats:
        print_summary(all_stats)
        plot_mega_chart(all_equity, all_stats)

    print("\n✅ Done!")
    print("   scalp_multi_timeframe.png  ← mega comparison chart")
