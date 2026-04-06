"""
Scalping Bot Backtest — Gold & Bitcoin
========================================
Tests the 5-signal scalping strategy on real 5-minute historical data.
Same dynamic position sizing as the daily bot (95%/60%/10% zones).

Run: py scalp_backtest.py
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
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ASSET            = "GOLD"       # change to "BTC" for bitcoin
GOLD_TICKER      = "GC=F"
BTC_TICKER       = "BTC-USD"
LOOKBACK         = "60d"        # 60 days of 5-min data (max yfinance allows)
TIMEFRAME        = "5m"

# Signal params
EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.5
MIN_SCORE        = 2            # need 2/5 signals — more trades

# Trade params
TAKE_PROFIT_PCT  = 0.0025       # 0.25% — wider TP for better R:R
STOP_LOSS_PCT    = 0.0008       # 0.08% — keep tight SL → 3:1 R:R — keep tight SL
HOLD_BARS        = 24           # max hold: 24 bars = 120 minutes — let winners run
LONGS_ONLY       = True

# Session filter (UTC) — London + NY only
SESSION_START    = 7
SESSION_END      = 21

# Dynamic position sizing
ACCOUNT_SIZE         = 10_000
MAX_DRAWDOWN_LIMIT   = 0.05
RISK_WHEN_SAFE       = 0.95   # full aggression in green zone   # already max
RISK_WHEN_WARNING    = 0.60
RISK_WHEN_DANGER     = 0.10


# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────
def dynamic_position(confidence, equity, peak):
    current_dd = (peak - equity) / peak if peak > 0 else 0.0
    dd_ratio   = min(current_dd / MAX_DRAWDOWN_LIMIT, 1.0)
    if dd_ratio < 0.5:
        scale    = dd_ratio / 0.5
        max_risk = RISK_WHEN_SAFE + scale * (RISK_WHEN_WARNING - RISK_WHEN_SAFE)
    else:
        scale    = (dd_ratio - 0.5) / 0.5
        max_risk = RISK_WHEN_WARNING + scale * (RISK_WHEN_DANGER - RISK_WHEN_WARNING)
    b     = TAKE_PROFIT_PCT / STOP_LOSS_PCT
    kelly = max(0, (confidence * (b + 1) - 1) / b) * 0.5
    return max(equity * 0.01, min(equity * kelly, equity * max_risk))


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def fetch_data(asset):
    ticker   = GOLD_TICKER if asset == "GOLD" else BTC_TICKER
    fallback = "GLD" if asset == "GOLD" else None
    log.info(f"📥 Fetching {LOOKBACK} of {ticker} 5-min data...")
    for t in ([ticker] + ([fallback] if fallback else [])):
        try:
            df = yf.download(t, period=LOOKBACK, interval=TIMEFRAME,
                             progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.dropna()
                log.info(f"   ✅ {t}: {len(df)} 5-min candles")
                return df
        except Exception as e:
            log.warning(f"   {t} failed: {e}")
    raise RuntimeError("No data")


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def build_indicators(df):
    d = df.copy()

    d["ema_fast"]  = d["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    d["ema_slow"]  = d["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    d["ema_cross"] = np.sign(d["ema_fast"] - d["ema_slow"])

    delta      = d["Close"].diff()
    gain       = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss       = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    d["rsi"]   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    sma          = d["Close"].rolling(BB_PERIOD).mean()
    std          = d["Close"].rolling(BB_PERIOD).std()
    d["bb_upper"]= sma + BB_STD * std
    d["bb_lower"]= sma - BB_STD * std
    d["bb_width"]= (d["bb_upper"] - d["bb_lower"]) / (sma + 1e-9)
    d["bb_pct"]  = (d["Close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)

    tp         = (d["High"] + d["Low"] + d["Close"]) / 3
    vol        = d["Volume"].replace(0, 1)
    d["vwap"]  = (tp * vol).rolling(VWAP_PERIOD).sum() / vol.rolling(VWAP_PERIOD).sum()
    d["vwap_dist"] = (d["Close"] - d["vwap"]) / d["vwap"]

    d["vol_avg"]   = d["Volume"].rolling(20).mean()
    d["vol_spike"] = d["Volume"] / (d["vol_avg"] + 1e-9)

    d["mom3"]  = d["Close"].pct_change(3)

    # Session filter
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
    score_l = 0.0
    score_s = 0.0

    # EMA cross
    cross = float(row["ema_cross"])
    if prev_cross <= 0 and cross > 0:   score_l += 1
    elif prev_cross >= 0 and cross < 0: score_s += 1
    elif cross > 0:  score_l += 0.5
    elif cross < 0:  score_s += 0.5

    # RSI
    rsi = float(row["rsi"])
    if rsi < 35:   score_l += 1
    elif rsi > 65: score_s += 1

    # VWAP
    vd = float(row["vwap_dist"])
    if vd < -0.001:  score_l += 1
    elif vd > 0.001: score_s += 1

    # Bollinger
    bp = float(row["bb_pct"])
    if bp < 0.2:   score_l += 1
    elif bp > 0.8: score_s += 1

    # Volume + momentum
    vs   = float(row["vol_spike"])
    mom  = float(row["mom3"])
    if vs >= VOLUME_SPIKE_MIN:
        if mom > 0:  score_l += 1
        else:        score_s += 1

    return score_l, score_s


# ─────────────────────────────────────────────
# TRADE SIMULATION
# ─────────────────────────────────────────────
@dataclass
class Trade:
    date: str
    direction: str
    entry: float
    exit_price: float
    position_usd: float
    pnl: float
    win: bool
    bars_held: int
    stop_hit: bool
    tp_hit: bool

def simulate(direction, entry, future_rows, position_usd):
    sl = entry * (1 - STOP_LOSS_PCT) if direction == "LONG" else entry * (1 + STOP_LOSS_PCT)
    tp = entry * (1 + TAKE_PROFIT_PCT) if direction == "LONG" else entry * (1 - TAKE_PROFIT_PCT)
    for bars, (_, row) in enumerate(future_rows.iterrows(), 1):
        price = float(row["Close"])
        if direction == "LONG":
            if price <= sl:
                return price, -position_usd * STOP_LOSS_PCT, False, bars, True, False
            if price >= tp:
                return price,  position_usd * TAKE_PROFIT_PCT, True, bars, False, True
        else:
            if price >= sl:
                return price, -position_usd * STOP_LOSS_PCT, False, bars, True, False
            if price <= tp:
                return price,  position_usd * TAKE_PROFIT_PCT, True, bars, False, True
    # Time exit
    exit_p = float(future_rows.iloc[-1]["Close"])
    pnl    = (exit_p - entry) / entry * position_usd if direction == "LONG" else (entry - exit_p) / entry * position_usd
    return exit_p, pnl, pnl > 0, HOLD_BARS, False, False


# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────
def run_backtest(df):
    log.info(f"🔄 Running scalp backtest ({len(df)} bars, LONGS ONLY={'✅' if LONGS_ONLY else '❌'})...")
    trades       = []
    equity       = ACCOUNT_SIZE
    peak_equity  = ACCOUNT_SIZE
    equity_curve = []
    skip_until   = 0   # bar index to skip to after a trade

    rows = list(df.iterrows())
    for i in range(50, len(df) - HOLD_BARS):
        if i < skip_until:
            equity_curve.append(equity)
            continue

        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # Session filter
        if int(row["in_session"]) == 0:
            equity_curve.append(equity)
            continue

        # Score signals
        prev_cross       = float(prev["ema_cross"])
        score_l, score_s = score_row(row, prev_cross)

        if LONGS_ONLY:
            if score_l < MIN_SCORE:
                equity_curve.append(equity)
                continue
            direction  = "LONG"
            confidence = min(score_l / 5.0, 1.0)
        else:
            if max(score_l, score_s) < MIN_SCORE:
                equity_curve.append(equity)
                continue
            if score_l >= score_s:
                direction  = "LONG"
                confidence = min(score_l / 5.0, 1.0)
            else:
                direction  = "SHORT"
                confidence = min(score_s / 5.0, 1.0)

        # Nan check
        vals = [row["ema_fast"], row["rsi"], row["bb_pct"], row["vwap_dist"], row["vol_spike"]]
        if not all(np.isfinite(v) for v in vals):
            equity_curve.append(equity)
            continue

        position_usd = dynamic_position(confidence, equity, peak_equity)
        if position_usd < 5:
            equity_curve.append(equity)
            continue

        entry        = float(row["Close"])
        future       = df.iloc[i+1:i+1+HOLD_BARS]
        exit_p, pnl, win, bars_held, stop_hit, tp_hit = simulate(direction, entry, future, position_usd)

        equity      += pnl
        equity       = max(equity, 0)
        peak_equity  = max(peak_equity, equity)
        skip_until   = i + bars_held  # no overlapping trades

        trades.append(Trade(
            date=str(rows[i][0])[:16],
            direction=direction,
            entry=round(entry, 4),
            exit_price=round(float(exit_p), 4),
            position_usd=round(position_usd, 2),
            pnl=round(float(pnl), 2),
            win=bool(win),
            bars_held=int(bars_held),
            stop_hit=bool(stop_hit),
            tp_hit=bool(tp_hit),
        ))
        equity_curve.append(equity)

    log.info(f"   ✅ {len(trades)} scalp trades")
    equity_series = pd.Series(equity_curve, index=df.index[50:50+len(equity_curve)])
    return trades, equity_series


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def compute_stats(trades, equity_series):
    if not trades: return {}
    pnls   = [t.pnl for t in trades]
    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    peak   = equity_series.cummax()
    dd     = (equity_series - peak) / peak
    ret    = equity_series.pct_change().dropna()
    avg_bars = float(np.mean([t.bars_held for t in trades]))
    return {
        "total":         len(trades),
        "win_rate":      len(wins) / len(trades),
        "total_pnl":     sum(pnls),
        "return_pct":    sum(pnls) / ACCOUNT_SIZE * 100,
        "avg_win":       float(np.mean([t.pnl for t in wins]))   if wins   else 0,
        "avg_loss":      float(np.mean([t.pnl for t in losses])) if losses else 0,
        "best":          max(pnls),
        "worst":         min(pnls),
        "profit_factor": abs(sum(t.pnl for t in wins)) / (abs(sum(t.pnl for t in losses)) + 1e-9),
        "max_dd":        float(dd.min() * 100),
        "sharpe":        float((ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252 * 78)),  # 78 5-min bars/day
        "tp_rate":       len([t for t in trades if t.tp_hit])   / len(trades),
        "sl_rate":       len([t for t in trades if t.stop_hit]) / len(trades),
        "avg_bars":      avg_bars,
        "avg_hold_mins": avg_bars * 5,
        "avg_pos":       float(np.mean([t.position_usd for t in trades])),
        "trades_per_day":len(trades) / 60,
    }


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────
def print_report(stats, asset):
    dd     = stats["max_dd"]
    dd_flag= "✅ Under 6%" if abs(dd) < 6 else "⚠️  OVER 6%"
    print("\n" + "="*57)
    print(f"  SCALPING BOT — {asset} BACKTEST REPORT")
    print("="*57)
    print(f"  Timeframe:        5-minute candles")
    print(f"  Period:           60 days")
    print(f"  Session:          London + NY (07:00–21:00 UTC)")
    print(f"  Starting capital: ${ACCOUNT_SIZE:,}")
    print(f"  Stop Loss:        {STOP_LOSS_PCT*100:.2f}%  |  Take Profit: {TAKE_PROFIT_PCT*100:.2f}%")
    print(f"  R:R Ratio:        1 : {TAKE_PROFIT_PCT/STOP_LOSS_PCT:.1f}  (3.1:1)")
    print(f"  Max hold:         {HOLD_BARS} bars = 60 minutes")
    print("-"*57)
    print(f"  Total trades    : {stats['total']}")
    print(f"  Trades/day      : ~{stats['trades_per_day']:.1f}")
    print(f"  Avg hold time   : {stats['avg_hold_mins']:.0f} minutes")
    print(f"  Win rate        : {stats['win_rate']:.1%}  ← your real number")
    print(f"  Total PnL       : ${stats['total_pnl']:+,.2f}")
    print(f"  Total return    : {stats['return_pct']:+.1f}%  (60 days)")
    print(f"  Annualised      : {stats['return_pct'] * 6:+.1f}%  (×6 for full year)")
    print(f"  Avg win         : ${stats['avg_win']:+.2f}")
    print(f"  Avg loss        : ${stats['avg_loss']:+.2f}")
    print(f"  Best trade      : ${stats['best']:+.2f}")
    print(f"  Worst trade     : ${stats['worst']:+.2f}")
    print(f"  Avg position    : ${stats['avg_pos']:.2f}")
    print(f"  Profit factor   : {stats['profit_factor']:.2f}")
    print(f"  Max drawdown    : {dd:.1f}%  {dd_flag}")
    print(f"  Sharpe ratio    : {stats['sharpe']:.2f}")
    print(f"  Take profit hit : {stats['tp_rate']:.1%}")
    print(f"  Stop loss hit   : {stats['sl_rate']:.1%}")
    print("="*57)


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def plot_results(trades, equity_series, stats, asset):
    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f1a")
    fig.suptitle(f"Scalping Bot — {asset} 60-Day Backtest (5-Min Candles)",
                 fontsize=15, color="white", fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    tc  = "#e0e0e0"; gc = "#2a2a3a"; gold = "#FFD700"; grn = "#00e676"; red = "#ff1744"

    def style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color=tc, fontsize=10, fontweight="bold")
        ax.tick_params(colors=tc, labelsize=7)
        ax.grid(color=gc, linestyle="--", linewidth=0.4, alpha=0.6)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, "💰 Equity Curve")
    ax1.plot(equity_series.index, equity_series.values, color=gold, linewidth=1.5)
    ax1.axhline(ACCOUNT_SIZE, color=tc, linestyle="--", linewidth=0.8, alpha=0.4)
    final = equity_series.iloc[-1]
    ax1.fill_between(equity_series.index, ACCOUNT_SIZE, equity_series.values,
                     where=equity_series.values >= ACCOUNT_SIZE, alpha=0.1, color=grn)
    ax1.fill_between(equity_series.index, ACCOUNT_SIZE, equity_series.values,
                     where=equity_series.values < ACCOUNT_SIZE, alpha=0.1, color=red)
    ax1.set_ylabel("Account ($)", color=tc, fontsize=8)

    # Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, "📉 Drawdown")
    peak = equity_series.cummax()
    dd   = (equity_series - peak) / peak * 100
    ax2.fill_between(dd.index, 0, dd.values, color=red, alpha=0.6)
    ax2.set_ylabel("DD %", color=tc, fontsize=8)

    # PnL histogram
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, "📊 PnL Distribution")
    pnls = [t.pnl for t in trades]
    bins = np.linspace(min(pnls), max(pnls), 30)
    ax3.hist([p for p in pnls if p >= 0], bins=bins, color=grn, alpha=0.7, label="Wins")
    ax3.hist([p for p in pnls if p <  0], bins=bins, color=red,  alpha=0.7, label="Losses")
    ax3.legend(fontsize=7, facecolor="#1a1a2e", labelcolor=tc)

    # Stats panel
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1a1a2e")
    ax4.axis("off")
    ax4.set_title("📈 Key Stats", color=tc, fontsize=10, fontweight="bold")
    lines = [
        ("Trades",        f"{stats['total']}"),
        ("Per Day",       f"~{stats['trades_per_day']:.1f}"),
        ("Avg Hold",      f"{stats['avg_hold_mins']:.0f} min"),
        ("Win Rate",      f"{stats['win_rate']:.1%}"),
        ("60d Return",    f"{stats['return_pct']:+.1f}%"),
        ("Annual Est.",   f"{stats['return_pct']*6:+.1f}%"),
        ("Profit Factor", f"{stats['profit_factor']:.2f}"),
        ("Max DD",        f"{stats['max_dd']:.1f}%"),
        ("Sharpe",        f"{stats['sharpe']:.2f}"),
    ]
    for j, (label, val) in enumerate(lines):
        y = 0.95 - j * 0.105
        ax4.text(0.02, y, label+":", transform=ax4.transAxes, color=tc, fontsize=9, va="top")
        c = gold
        ax4.text(0.6, y, val, transform=ax4.transAxes, color=c, fontsize=9, va="top", fontweight="bold")

    plt.savefig(f"scalp_backtest_{asset.lower()}.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"   Chart → scalp_backtest_{asset.lower()}.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    for asset in ["GOLD", "BTC"]:
        try:
            df     = fetch_data(asset)
            df     = build_indicators(df)
            trades, eq = run_backtest(df)
            if not trades:
                print(f"\n⚠️  No trades for {asset}")
                continue
            stats  = compute_stats(trades, eq)
            print_report(stats, asset)
            plot_results(trades, eq, stats, asset)
            with open(f"scalp_trades_{asset.lower()}.json", "w") as f:
                json.dump([t.__dict__ for t in trades], f, indent=2)
            log.info(f"   Trades → scalp_trades_{asset.lower()}.json")
        except Exception as e:
            log.error(f"❌ {asset} failed: {e}")
            import traceback; traceback.print_exc()

    print("\n✅ Done!")
    print("   scalp_backtest_gold.png")
    print("   scalp_backtest_btc.png")
