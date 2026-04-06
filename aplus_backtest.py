"""
A+ Setup Backtest — BTC & Gold
================================
Tests ONLY the two filters that proved effective on real data:
  Filter A: Session window — 07:00-10:00 UTC (London open) OR 13:30-16:30 UTC (NY open)
  Filter B: Hold 2-7 bars minimum before time-exit (skips instant 1-bar noise trades)

Everything else stays identical to original backtest.
Run: python aplus_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import json, logging, numpy as np, pandas as pd, yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────────
ASSET            = "BTC"
GOLD_TICKER      = "GC=F"
BTC_TICKER       = "BTC-USD"
LOOKBACK         = "60d"
TIMEFRAME        = "5m"

EMA_FAST         = 9
EMA_SLOW         = 21
RSI_PERIOD       = 14
BB_PERIOD        = 20
BB_STD           = 2.0
VWAP_PERIOD      = 20
VOLUME_SPIKE_MIN = 1.5
MIN_SCORE        = 2          # same as original — filters do the selection, not score

TAKE_PROFIT_PCT  = 0.0025
STOP_LOSS_PCT    = 0.0008
HOLD_BARS        = 24
LONGS_ONLY       = True

# ── THE TWO PROVEN FILTERS ──────────────────────────────────────
# Filter A: only these UTC session windows
APLUS_SESSIONS = [
    (7,  0,  10, 0),    # London open
    (13, 30, 16, 30),   # NY open
]
# Filter B: time-exit must have held at least MIN_HOLD_BARS
# (stop-loss / take-profit exits are always counted regardless of hold time)
MIN_HOLD_FOR_TIME_EXIT = 2   # skip time-exits that fired in bar 1

ACCOUNT_SIZE         = 10_000
MAX_DRAWDOWN_LIMIT   = 0.05
RISK_WHEN_SAFE       = 0.95
RISK_WHEN_WARNING    = 0.60
RISK_WHEN_DANGER     = 0.10


# ── HELPERS ─────────────────────────────────────────────────────
def in_aplus_session(hour, minute):
    mins = hour * 60 + minute
    for (sh, sm, eh, em) in APLUS_SESSIONS:
        if sh * 60 + sm <= mins < eh * 60 + em:
            return True
    return False

def dynamic_position(confidence, equity, peak):
    dd_ratio = min((peak - equity) / peak / MAX_DRAWDOWN_LIMIT, 1.0) if peak > 0 else 0.0
    if dd_ratio < 0.5:
        max_risk = RISK_WHEN_SAFE + (dd_ratio / 0.5) * (RISK_WHEN_WARNING - RISK_WHEN_SAFE)
    else:
        max_risk = RISK_WHEN_WARNING + ((dd_ratio - 0.5) / 0.5) * (RISK_WHEN_DANGER - RISK_WHEN_WARNING)
    b     = TAKE_PROFIT_PCT / STOP_LOSS_PCT
    kelly = max(0, (confidence * (b + 1) - 1) / b) * 0.5
    return max(equity * 0.01, min(equity * kelly, equity * max_risk))


# ── DATA ────────────────────────────────────────────────────────
def fetch_data(asset):
    ticker   = GOLD_TICKER if asset == "GOLD" else BTC_TICKER
    fallback = "GLD" if asset == "GOLD" else None
    log.info(f"Fetching {LOOKBACK} of {ticker} 5-min data...")
    for t in ([ticker] + ([fallback] if fallback else [])):
        try:
            df = yf.download(t, period=LOOKBACK, interval=TIMEFRAME,
                             progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.dropna()
                log.info(f"  {t}: {len(df)} candles")
                return df
        except Exception as e:
            log.warning(f"  {t} failed: {e}")
    raise RuntimeError("No data")


# ── INDICATORS ──────────────────────────────────────────────────
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

    try:
        idx = d.index.tz_convert("UTC") if d.index.tzinfo else d.index.tz_localize("UTC")
    except Exception:
        idx = d.index
    d["hour"]       = idx.hour
    d["minute"]     = idx.minute
    d["weekday"]    = idx.weekday
    # original session: london+ny 07-21
    d["in_session"] = ((d["hour"] >= 7) & (d["hour"] < 21) & (d["weekday"] < 5)).astype(int)
    # A+ session filter
    d["in_aplus"]   = d.apply(lambda r: int(in_aplus_session(int(r["hour"]), int(r["minute"]))), axis=1)
    return d.dropna()


# ── SCORER ──────────────────────────────────────────────────────
def score_row(row, prev_cross):
    sl = ss = 0.0
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

    vs = float(row["vol_spike"])
    if vs >= VOLUME_SPIKE_MIN:
        if float(row["mom3"]) > 0: sl += 1
        else:                       ss += 1
    return sl, ss


# ── TRADE SIMULATION ────────────────────────────────────────────
@dataclass
class Trade:
    date: str
    session: str
    entry: float
    exit_price: float
    position_usd: float
    pnl: float
    win: bool
    bars_held: int
    stop_hit: bool
    tp_hit: bool
    time_exit: bool
    aplus: bool

def simulate(entry, future_rows, position_usd):
    sl = entry * (1 - STOP_LOSS_PCT)
    tp = entry * (1 + TAKE_PROFIT_PCT)
    for bars, (_, row) in enumerate(future_rows.iterrows(), 1):
        price = float(row["Close"])
        if price <= sl:
            return price, -position_usd * STOP_LOSS_PCT, False, bars, True,  False, True
        if price >= tp:
            return price,  position_usd * TAKE_PROFIT_PCT, True, bars, False, True,  True
    exit_p = float(future_rows.iloc[-1]["Close"])
    pnl    = (exit_p - entry) / entry * position_usd
    return exit_p, pnl, pnl > 0, HOLD_BARS, False, False, False


# ── BACKTEST ────────────────────────────────────────────────────
def run_backtest(df, use_aplus_filter):
    label = "A+ ONLY" if use_aplus_filter else "BASELINE"
    log.info(f"Running {label} backtest ({len(df)} bars)...")
    trades = []
    equity = ACCOUNT_SIZE
    peak   = ACCOUNT_SIZE
    equity_curve = []
    skip_until = 0

    rows = list(df.iterrows())
    for i in range(50, len(df) - HOLD_BARS):
        if i < skip_until:
            equity_curve.append(equity)
            continue

        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if int(row["in_session"]) == 0:
            equity_curve.append(equity)
            continue

        # ── FILTER A: A+ session window ──────────────
        if use_aplus_filter and int(row["in_aplus"]) == 0:
            equity_curve.append(equity)
            continue

        prev_cross       = float(prev["ema_cross"])
        score_l, score_s = score_row(row, prev_cross)

        if score_l < MIN_SCORE:
            equity_curve.append(equity)
            continue

        vals = [row["ema_fast"], row["rsi"], row["bb_pct"], row["vwap_dist"], row["vol_spike"]]
        if not all(np.isfinite(v) for v in vals):
            equity_curve.append(equity)
            continue

        confidence   = min(score_l / 5.0, 1.0)
        position_usd = dynamic_position(confidence, equity, peak)
        if position_usd < 5:
            equity_curve.append(equity)
            continue

        entry  = float(row["Close"])
        future = df.iloc[i+1:i+1+HOLD_BARS]
        exit_p, pnl, win, bars_held, stop_hit, tp_hit, sl_or_tp = simulate(entry, future, position_usd)

        # ── FILTER B: skip time-exits that held < 2 bars (1-bar noise) ──
        if use_aplus_filter and not sl_or_tp and bars_held < MIN_HOLD_FOR_TIME_EXIT:
            equity_curve.append(equity)
            continue

        equity  += pnl
        equity   = max(equity, 0)
        peak     = max(peak, equity)
        skip_until = i + bars_held

        h = int(row["hour"])
        m = int(row["minute"])
        if 7*60 <= h*60+m < 10*60:
            sess = "London"
        elif 13*60+30 <= h*60+m < 16*60+30:
            sess = "NY"
        else:
            sess = "Other"

        trades.append(Trade(
            date=str(rows[i][0])[:16],
            session=sess,
            entry=round(entry, 2),
            exit_price=round(float(exit_p), 2),
            position_usd=round(position_usd, 2),
            pnl=round(float(pnl), 2),
            win=bool(win),
            bars_held=int(bars_held),
            stop_hit=bool(stop_hit),
            tp_hit=bool(tp_hit),
            time_exit=not sl_or_tp,
            aplus=bool(int(row["in_aplus"])),
        ))
        equity_curve.append(equity)

    log.info(f"  {len(trades)} trades")
    eq_series = pd.Series(equity_curve, index=df.index[50:50+len(equity_curve)])
    return trades, eq_series


# ── STATS ────────────────────────────────────────────────────────
def compute_stats(trades, eq):
    if not trades: return {}
    pnls   = [t.pnl for t in trades]
    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]
    peak   = eq.cummax()
    dd     = (eq - peak) / peak
    ret    = eq.pct_change().dropna()
    return {
        "total":         len(trades),
        "win_rate":      len(wins) / len(trades),
        "total_pnl":     sum(pnls),
        "return_pct":    sum(pnls) / ACCOUNT_SIZE * 100,
        "avg_win":       float(np.mean([t.pnl for t in wins]))   if wins   else 0,
        "avg_loss":      float(np.mean([t.pnl for t in losses])) if losses else 0,
        "profit_factor": abs(sum(t.pnl for t in wins)) / (abs(sum(t.pnl for t in losses)) + 1e-9),
        "max_dd":        float(dd.min() * 100),
        "sharpe":        float((ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252 * 78)),
        "tp_rate":       len([t for t in trades if t.tp_hit])   / len(trades),
        "sl_rate":       len([t for t in trades if t.stop_hit]) / len(trades),
        "avg_bars":      float(np.mean([t.bars_held for t in trades])),
        "trades_per_day":len(trades) / 60,
        "london_wr":     len([t for t in trades if t.session=="London" and t.win]) / max(1, len([t for t in trades if t.session=="London"])),
        "ny_wr":         len([t for t in trades if t.session=="NY" and t.win]) / max(1, len([t for t in trades if t.session=="NY"])),
    }


# ── REPORT ───────────────────────────────────────────────────────
def print_report(stats, label, asset):
    print(f"\n{'='*60}")
    print(f"  {label} — {asset}")
    print(f"{'='*60}")
    print(f"  Trades       : {stats['total']}   ({stats['trades_per_day']:.1f}/day)")
    print(f"  Win rate     : {stats['win_rate']:.1%}")
    print(f"  Total PnL    : ${stats['total_pnl']:+,.2f}")
    print(f"  Return       : {stats['return_pct']:+.1f}% over 60 days")
    print(f"  Annualised   : {stats['return_pct']*6:+.1f}%")
    print(f"  Avg win      : ${stats['avg_win']:+.2f}")
    print(f"  Avg loss     : ${stats['avg_loss']:+.2f}")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Max drawdown : {stats['max_dd']:.1f}%")
    print(f"  Sharpe       : {stats['sharpe']:.2f}")
    print(f"  TP rate      : {stats['tp_rate']:.1%}")
    print(f"  SL rate      : {stats['sl_rate']:.1%}")
    print(f"  London WR    : {stats['london_wr']:.1%}")
    print(f"  NY WR        : {stats['ny_wr']:.1%}")
    print(f"{'='*60}")


# ── CHART ────────────────────────────────────────────────────────
def plot_comparison(base_trades, base_eq, aplus_trades, aplus_eq,
                    base_stats, aplus_stats, asset):
    fig = plt.figure(figsize=(18, 12), facecolor="#0f0f1a")
    fig.suptitle(f"A+ Setup Filter Backtest — {asset} (60 days, 5-min)",
                 fontsize=15, color="white", fontweight="bold")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    tc  = "#e0e0e0"; gc = "#2a2a3a"
    gold = "#FFD700"; teal = "#00e5cc"; grn = "#00e676"; red = "#ff1744"

    def style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color=tc, fontsize=10, fontweight="bold")
        ax.tick_params(colors=tc, labelsize=7)
        ax.grid(color=gc, linestyle="--", linewidth=0.4, alpha=0.6)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    # Row 0: equity curves side by side
    ax0 = fig.add_subplot(gs[0, :2])
    style(ax0, "Equity Curve — Baseline vs A+ Filter")
    ax0.plot(base_eq.index,  base_eq.values,  color=gold, linewidth=1.2, label="Baseline", alpha=0.7)
    ax0.plot(aplus_eq.index, aplus_eq.values, color=teal, linewidth=1.8, label="A+ Only")
    ax0.axhline(ACCOUNT_SIZE, color=tc, linestyle="--", linewidth=0.6, alpha=0.3)
    ax0.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=tc)
    ax0.set_ylabel("Equity ($)", color=tc, fontsize=8)

    # Stats comparison panel
    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.set_facecolor("#1a1a2e"); ax_s.axis("off")
    ax_s.set_title("Side-by-side", color=tc, fontsize=10, fontweight="bold")
    rows = [
        ("Metric",       "Baseline",  "A+ Filter"),
        ("Trades/day",   f"{base_stats['trades_per_day']:.1f}", f"{aplus_stats['trades_per_day']:.1f}"),
        ("Win rate",     f"{base_stats['win_rate']:.1%}", f"{aplus_stats['win_rate']:.1%}"),
        ("Total PnL",    f"${base_stats['total_pnl']:+,.0f}", f"${aplus_stats['total_pnl']:+,.0f}"),
        ("Profit Factor",f"{base_stats['profit_factor']:.2f}", f"{aplus_stats['profit_factor']:.2f}"),
        ("Max DD",       f"{base_stats['max_dd']:.1f}%", f"{aplus_stats['max_dd']:.1f}%"),
        ("Sharpe",       f"{base_stats['sharpe']:.2f}", f"{aplus_stats['sharpe']:.2f}"),
        ("London WR",    f"{base_stats['london_wr']:.1%}", f"{aplus_stats['london_wr']:.1%}"),
        ("NY WR",        f"{base_stats['ny_wr']:.1%}", f"{aplus_stats['ny_wr']:.1%}"),
    ]
    for j, (m, b, a) in enumerate(rows):
        y = 0.95 - j * 0.1
        hdr = j == 0
        ax_s.text(0.01, y, m, transform=ax_s.transAxes, color=tc if not hdr else gold,
                  fontsize=8 if not hdr else 9, va="top", fontweight="bold" if hdr else "normal")
        ax_s.text(0.48, y, b, transform=ax_s.transAxes, color=gold, fontsize=8, va="top")
        ax_s.text(0.75, y, a, transform=ax_s.transAxes, color=teal, fontsize=8, va="top", fontweight="bold")

    # Row 1: drawdowns
    ax1 = fig.add_subplot(gs[1, 0])
    style(ax1, "Drawdown — Baseline")
    peak_b = base_eq.cummax()
    dd_b   = (base_eq - peak_b) / peak_b * 100
    ax1.fill_between(dd_b.index, 0, dd_b.values, color=gold, alpha=0.5)
    ax1.set_ylabel("DD %", color=tc, fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1])
    style(ax2, "Drawdown — A+ Filter")
    peak_a = aplus_eq.cummax()
    dd_a   = (aplus_eq - peak_a) / peak_a * 100
    ax2.fill_between(dd_a.index, 0, dd_a.values, color=teal, alpha=0.5)
    ax2.set_ylabel("DD %", color=tc, fontsize=8)

    # PnL hist comparison
    ax3 = fig.add_subplot(gs[1, 2])
    style(ax3, "PnL distribution — A+ trades")
    pnls = [t.pnl for t in aplus_trades]
    if pnls:
        bins = np.linspace(min(pnls), max(pnls), 25)
        ax3.hist([p for p in pnls if p >= 0], bins=bins, color=grn, alpha=0.7, label="Wins")
        ax3.hist([p for p in pnls if p <  0], bins=bins, color=red,  alpha=0.7, label="Losses")
        ax3.legend(fontsize=7, facecolor="#1a1a2e", labelcolor=tc)

    # Row 2: session breakdown for A+ trades
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, "A+ trades by session")
    london = [t for t in aplus_trades if t.session == "London"]
    ny     = [t for t in aplus_trades if t.session == "NY"]
    cats   = ["London\n07-10", "NY\n13:30-16:30"]
    wrs    = [
        len([t for t in london if t.win]) / max(1, len(london)) * 100,
        len([t for t in ny     if t.win]) / max(1, len(ny))     * 100,
    ]
    counts = [len(london), len(ny)]
    bars   = ax4.bar(cats, wrs, color=[teal, gold], alpha=0.8)
    ax4.axhline(50, color=tc, linestyle="--", linewidth=0.6, alpha=0.4)
    ax4.set_ylabel("Win rate %", color=tc, fontsize=8)
    for bar, c in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{c} trades", ha="center", va="bottom", color=tc, fontsize=7)

    # Hold bars analysis for A+ trades
    ax5 = fig.add_subplot(gs[2, 1])
    style(ax5, "Win rate by bars held — A+ trades")
    bk = [(1,1,"1"), (2,3,"2-3"), (4,7,"4-7"), (8,15,"8-15"), (16,24,"16-24")]
    bwr = []
    blab = []
    for lo, hi, lab in bk:
        grp = [t for t in aplus_trades if lo <= t.bars_held <= hi]
        if grp:
            bwr.append(len([t for t in grp if t.win]) / len(grp) * 100)
            blab.append(f"{lab}\n({len(grp)})")
    if bwr:
        colors_b = [grn if w >= 55 else red if w < 45 else gold for w in bwr]
        ax5.bar(blab, bwr, color=colors_b, alpha=0.8)
        ax5.axhline(50, color=tc, linestyle="--", linewidth=0.6, alpha=0.4)
        ax5.set_ylabel("Win rate %", color=tc, fontsize=8)

    # Trade count comparison
    ax6 = fig.add_subplot(gs[2, 2])
    style(ax6, "Trade count: what gets filtered out")
    total_b  = len(base_trades)
    total_a  = len(aplus_trades)
    filtered = total_b - total_a
    ax6.bar(["Baseline\ntrades", "A+ trades\n(kept)", "Filtered\nout"],
            [total_b, total_a, filtered],
            color=[gold, teal, red], alpha=0.8)
    ax6.set_ylabel("Count", color=tc, fontsize=8)

    plt.savefig(f"aplus_backtest_{asset.lower()}.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Chart saved: aplus_backtest_{asset.lower()}.png")


# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    for asset in ["BTC", "GOLD"]:
        try:
            df = fetch_data(asset)
            df = build_indicators(df)

            base_trades,  base_eq  = run_backtest(df, use_aplus_filter=False)
            aplus_trades, aplus_eq = run_backtest(df, use_aplus_filter=True)

            if not base_trades or not aplus_trades:
                print(f"No trades for {asset}"); continue

            base_stats  = compute_stats(base_trades,  base_eq)
            aplus_stats = compute_stats(aplus_trades, aplus_eq)

            print_report(base_stats,  "BASELINE (no filter)", asset)
            print_report(aplus_stats, "A+ FILTER (session + min hold)", asset)

            improvement = (aplus_stats["win_rate"] - base_stats["win_rate"]) * 100
            print(f"\n  Win rate improvement : +{improvement:.1f} percentage points")
            print(f"  Trades kept          : {len(aplus_trades)}/{len(base_trades)} "
                  f"({len(aplus_trades)/len(base_trades):.0%})")

            plot_comparison(base_trades, base_eq, aplus_trades, aplus_eq,
                            base_stats, aplus_stats, asset)

            with open(f"aplus_trades_{asset.lower()}.json", "w") as f:
                json.dump([t.__dict__ for t in aplus_trades], f, indent=2)
            log.info(f"Trades saved: aplus_trades_{asset.lower()}.json")

        except Exception as e:
            log.error(f"{asset} failed: {e}")
            import traceback; traceback.print_exc()

    print("\nDone!")
    print("  aplus_backtest_btc.png")
    print("  aplus_backtest_gold.png")
    print("  aplus_trades_btc.json")
    print("  aplus_trades_gold.json")
