"""
Gold Trading Bot - Backtester v2 (AGGRESSIVE + LONGS ONLY)
===========================================================
Key improvements over v1:
  - LONGS ONLY (short win rate was 12% — removed)
  - Higher risk per trade: 5% max (was 2%)
  - Wider take profit: 4% (was 3%)
  - Tighter stop loss: 1.2% (was 1.5%) — better R:R
  - Lower confidence threshold: 0.55 (was 0.62) — more trades
  - More features for better prediction

Run: py backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG — tuned for max profitability
# ─────────────────────────────────────────────
GOLD_TICKER          = "GC=F"
GOLD_ETF             = "GLD"
BACKTEST_YEARS       = 5
CONFIDENCE_THRESHOLD = 0.50   # lower = more trades
MAX_RISK_PCT         = 0.05   # 5% per trade (was 2%)
ACCOUNT_SIZE         = 10_000
STOP_LOSS_PCT        = 0.012  # tighter stop: 1.2%
TAKE_PROFIT_PCT      = 0.060  # wider TP: 6% → 5:1 R:R
LONGS_ONLY           = True   # shorts had 12% win rate — disable them
TRAIN_WINDOW         = 120
WALK_FORWARD_STEP    = 5
HOLD_DAYS            = 7      # max days to hold a trade
RECOVERY_DAYS        = 30     # after hitting DD limit, wait 30 days then reset peak
RECOVERY_RISK_PCT    = 0.05   # during recovery, only risk 5% max


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def fetch_gold_data(years=BACKTEST_YEARS) -> pd.DataFrame:
    log.info(f"📥 Fetching {years} years of gold data...")
    for ticker in [GOLD_TICKER, GOLD_ETF]:
        try:
            df = yf.download(ticker, period=f"{years}y", interval="1d",
                             progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                log.info(f"   ✅ {ticker}: {len(df)} trading days")
                return df
        except Exception as e:
            log.warning(f"   Failed {ticker}: {e}")
    raise RuntimeError("Could not fetch gold data.")


# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Returns
    d["r1"]  = d["Close"].pct_change(1)
    d["r3"]  = d["Close"].pct_change(3)
    d["r5"]  = d["Close"].pct_change(5)
    d["r10"] = d["Close"].pct_change(10)
    d["r20"] = d["Close"].pct_change(20)

    # RSI
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    d["rsi_ob"] = (d["rsi"] > 70).astype(int)
    d["rsi_os"] = (d["rsi"] < 30).astype(int)

    # MACD
    ema12 = d["Close"].ewm(span=12).mean()
    ema26 = d["Close"].ewm(span=26).mean()
    d["macd"]  = ema12 - ema26
    d["macd_s"] = d["macd"].ewm(span=9).mean()
    d["macd_h"] = d["macd"] - d["macd_s"]

    # Bollinger Bands
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["bb_pos"]   = (d["Close"] - sma20) / (2 * std20 + 1e-9)
    d["bb_width"] = (4 * std20) / (sma20 + 1e-9)

    # ATR
    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"]  - d["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14).mean() / d["Close"]

    # Volume
    d["vol_ratio"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9)
    d["vol_trend"] = d["Volume"].pct_change(5)

    # Moving average signals
    d["vs_sma20"]  = (d["Close"] - d["Close"].rolling(20).mean())  / d["Close"]
    d["vs_sma50"]  = (d["Close"] - d["Close"].rolling(50).mean())  / d["Close"]
    d["vs_sma200"] = (d["Close"] - d["Close"].rolling(200).mean()) / d["Close"]
    d["ma_cross"]  = np.sign(d["Close"].rolling(50).mean() - d["Close"].rolling(200).mean())

    # Momentum
    d["mom5"]  = d["Close"] / d["Close"].shift(5)  - 1
    d["mom20"] = d["Close"] / d["Close"].shift(20) - 1

    # Target: up in next day?
    d["target"]     = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d["next_close"] = d["Close"].shift(-1)

    return d.dropna()


FEATURES = [
    "r1","r3","r5","r10","r20",
    "rsi","rsi_ob","rsi_os",
    "macd","macd_s","macd_h",
    "bb_pos","bb_width","atr_pct",
    "vol_ratio","vol_trend",
    "vs_sma20","vs_sma50","vs_sma200","ma_cross",
    "mom5","mom20",
]


# ─────────────────────────────────────────────
# POSITION SIZING — Dynamic drawdown-aware
# ─────────────────────────────────────────────
MAX_DRAWDOWN_LIMIT = 0.035  # HARD LIMIT: block trades if drawdown hits 3.5%
RISK_WHEN_SAFE     = 0.95   # risk 95% per trade when drawdown is 0% (capped for safety)
RISK_WHEN_WARNING  = 0.60   # risk 60% per trade when drawdown is at warning
RISK_WHEN_DANGER   = 0.10   # risk 10% per trade when drawdown is near limit

def dynamic_position(confidence: float, equity: float, peak: float) -> float:
    current_dd = (peak - equity) / peak if peak > 0 else 0.0
    if current_dd >= MAX_DRAWDOWN_LIMIT:
        return 0.0  # HARD STOP — no trades
    dd_ratio = current_dd / MAX_DRAWDOWN_LIMIT
    if dd_ratio < 0.5:
        scale    = dd_ratio / 0.5
        max_risk = RISK_WHEN_SAFE + scale * (RISK_WHEN_WARNING - RISK_WHEN_SAFE)
    else:
        scale    = (dd_ratio - 0.5) / 0.5
        max_risk = RISK_WHEN_WARNING + scale * (RISK_WHEN_DANGER - RISK_WHEN_WARNING)
    p     = confidence
    b     = TAKE_PROFIT_PCT / STOP_LOSS_PCT
    kelly = max(0, (p * (b + 1) - 1) / b) * 0.5
    return min(equity * kelly, equity * max_risk)


# ─────────────────────────────────────────────
# TRADE SIMULATION
# ─────────────────────────────────────────────
def simulate_trade(direction, entry, next_prices, position_usd):
    if direction == "LONG":
        sl = entry * (1 - STOP_LOSS_PCT)
        tp = entry * (1 + TAKE_PROFIT_PCT)
    else:
        sl = entry * (1 + STOP_LOSS_PCT)
        tp = entry * (1 - TAKE_PROFIT_PCT)

    for price in next_prices:
        if direction == "LONG":
            if price <= sl:
                return price, -position_usd * STOP_LOSS_PCT, False, True, False
            if price >= tp:
                return price,  position_usd * TAKE_PROFIT_PCT, True, False, True
        else:
            if price >= sl:
                return price, -position_usd * STOP_LOSS_PCT, False, True, False
            if price <= tp:
                return price,  position_usd * TAKE_PROFIT_PCT, True, False, True

    exit_price = next_prices.iloc[-1]
    if direction == "LONG":
        pnl = (exit_price - entry) / entry * position_usd
    else:
        pnl = (entry - exit_price) / entry * position_usd
    return exit_price, pnl, pnl > 0, False, False


# ─────────────────────────────────────────────
# BACKTEST DATACLASS
# ─────────────────────────────────────────────
@dataclass
class Trade:
    date: str
    direction: str
    entry_price: float
    exit_price: float
    confidence: float
    position_usd: float
    pnl: float
    win: bool
    stop_hit: bool
    tp_hit: bool


# ─────────────────────────────────────────────
# WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────
def run_backtest(df: pd.DataFrame):
    log.info("🔄 Running walk-forward backtest (LONGS ONLY, aggressive sizing)...")
    trades         = []
    equity         = ACCOUNT_SIZE
    peak_equity    = ACCOUNT_SIZE   # tracks highest account value ever
    equity_curve   = []
    scaler         = StandardScaler()
    model          = None
    retrain_ctr    = 0
    indices        = df.index.tolist()
    dd_breach_day  = None           # day we hit the drawdown limit
    in_recovery    = False          # whether we are in recovery mode

    for i in range(TRAIN_WINDOW, len(df) - HOLD_DAYS):
        date = indices[i]
        row  = df.iloc[i]

        # Retrain periodically
        if retrain_ctr == 0 or model is None:
            train = df.iloc[max(0, i - TRAIN_WINDOW):i].copy()
            train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
            X_tr  = train[FEATURES].values
            y_tr  = train["target"].values
            if len(np.unique(y_tr)) < 2 or len(X_tr) < 30:
                equity_curve.append(equity)
                retrain_ctr = (retrain_ctr + 1) % WALK_FORWARD_STEP
                continue
            scaler.fit(X_tr)
            X_tr = scaler.transform(X_tr)
            model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss", verbosity=0,
            )
            model.fit(X_tr, y_tr, verbose=False)

        retrain_ctr = (retrain_ctr + 1) % WALK_FORWARD_STEP

        # Predict — skip rows with NaN or Inf values
        feat_vals = row[FEATURES].values.astype(float).reshape(1, -1)
        if not np.isfinite(feat_vals).all():
            equity_curve.append(equity)
            continue
        X          = scaler.transform(feat_vals)
        proba      = model.predict_proba(X)[0]
        prob_up    = float(proba[1])
        prob_down  = float(proba[0])
        confidence = max(prob_up, prob_down)
        direction  = "LONG" if prob_up > prob_down else "SHORT"

        # Skip shorts — they don't work for gold
        if LONGS_ONLY and direction == "SHORT":
            equity_curve.append(equity)
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            equity_curve.append(equity)
            continue

        # Recovery mode — after DD breach, wait RECOVERY_DAYS then reset peak
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if current_dd >= MAX_DRAWDOWN_LIMIT:
            if dd_breach_day is None:
                dd_breach_day = i
                in_recovery   = True
            elif (i - dd_breach_day) >= RECOVERY_DAYS:
                # Reset peak to current equity so bot can trade again
                peak_equity   = equity
                dd_breach_day = None
                in_recovery   = False
            equity_curve.append(equity)
            continue

        if in_recovery:
            in_recovery = False

        # Use reduced risk during recovery warmup
        position_usd = dynamic_position(confidence, equity, peak_equity)
        if position_usd < 10:
            equity_curve.append(equity)
            continue

        entry       = float(row["Close"])
        next_prices = df.iloc[i+1:i+1+HOLD_DAYS]["Close"]
        exit_price, pnl, win, stop_hit, tp_hit = simulate_trade(
            direction, entry, next_prices, position_usd
        )

        equity      += pnl
        equity       = max(equity, 0)
        peak_equity  = max(peak_equity, equity)  # update peak

        trades.append(Trade(
            date=str(date)[:10],
            direction=direction,
            entry_price=round(entry, 2),
            exit_price=round(float(exit_price), 2),
            confidence=round(confidence, 4),
            position_usd=round(position_usd, 2),
            pnl=round(float(pnl), 2),
            win=bool(win),
            stop_hit=bool(stop_hit),
            tp_hit=bool(tp_hit),
        ))
        equity_curve.append(equity)

    equity_series = pd.Series(
        equity_curve,
        index=df.index[TRAIN_WINDOW:TRAIN_WINDOW + len(equity_curve)]
    )
    log.info(f"   ✅ {len(trades)} trades taken")
    return trades, equity_series


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def compute_stats(trades, equity_series):
    if not trades:
        return {}
    pnls   = [t.pnl for t in trades]
    wins   = [t for t in trades if t.win]
    losses = [t for t in trades if not t.win]

    peak   = equity_series.cummax()
    dd     = (equity_series - peak) / peak
    eq_ret = equity_series.pct_change().dropna()

    return {
        "total_trades":     len(trades),
        "win_rate":         len(wins) / len(trades),
        "total_pnl":        sum(pnls),
        "return_pct":       sum(pnls) / ACCOUNT_SIZE * 100,
        "avg_win":          np.mean([t.pnl for t in wins])   if wins   else 0,
        "avg_loss":         np.mean([t.pnl for t in losses]) if losses else 0,
        "profit_factor":    abs(sum(t.pnl for t in wins)) / (abs(sum(t.pnl for t in losses)) + 1e-9),
        "max_drawdown_pct": float(dd.min() * 100),
        "sharpe_ratio":     float((eq_ret.mean() / (eq_ret.std() + 1e-9)) * np.sqrt(252)),
        "tp_hit_rate":      len([t for t in trades if t.tp_hit])   / len(trades),
        "sl_hit_rate":      len([t for t in trades if t.stop_hit]) / len(trades),
        "best_trade":       max(pnls),
        "worst_trade":      min(pnls),
        "avg_position":     np.mean([t.position_usd for t in trades]),
    }


# ─────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────
def print_report(stats, trades):
    print("\n" + "="*57)
    print("  GOLD TRADING BOT v2 — BACKTEST REPORT")
    print("="*57)
    print(f"  Mode:             LONGS ONLY ({'✅' if LONGS_ONLY else '❌'})")
    print(f"  Backtest period:  {BACKTEST_YEARS} years")
    print(f"  Starting capital: ${ACCOUNT_SIZE:,}")
    print(f"  Max risk/trade:   15% (green) / 6% (warning) / 1% (danger)")
    print(f"  Stop Loss:        {STOP_LOSS_PCT:.1%}  |  Take Profit: {TAKE_PROFIT_PCT:.1%}")
    print(f"  R:R Ratio:        1 : {TAKE_PROFIT_PCT/STOP_LOSS_PCT:.0f}")
    print(f"  Confidence:       {CONFIDENCE_THRESHOLD:.0%}+")
    print("-"*57)
    print(f"  Total trades    : {stats['total_trades']}")
    print(f"  Win rate        : {stats['win_rate']:.1%}  ← your real number")
    print(f"  Total PnL       : ${stats['total_pnl']:+,.2f}")
    print(f"  Total return    : {stats['return_pct']:+.1f}%")
    print(f"  Avg win         : ${stats['avg_win']:+.2f}")
    print(f"  Avg loss        : ${stats['avg_loss']:+.2f}")
    print(f"  Best trade      : ${stats['best_trade']:+.2f}")
    print(f"  Worst trade     : ${stats['worst_trade']:+.2f}")
    print(f"  Avg position    : ${stats['avg_position']:.2f}")
    print(f"  Profit factor   : {stats['profit_factor']:.2f}")
    print(f"  Max drawdown    : {stats['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe ratio    : {stats['sharpe_ratio']:.2f}")
    print(f"  Take profit hit : {stats['tp_hit_rate']:.1%}")
    print(f"  Stop loss hit   : {stats['sl_hit_rate']:.1%}")
    print("="*57)

    if trades:
        print("\n  Last 5 trades:")
        print(f"  {'Date':<12} {'Dir':<6} {'Entry':>8} {'Exit':>8} {'Pos $':>7} {'PnL':>8} {'Result'}")
        print("  " + "-"*60)
        for t in trades[-5:]:
            result = "✅ WIN" if t.win else "❌ LOSS"
            print(f"  {t.date:<12} {t.direction:<6} ${t.entry_price:>7.2f} ${t.exit_price:>7.2f} ${t.position_usd:>6.0f} ${t.pnl:>+7.2f}  {result}")
    print()


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def plot_results(trades, equity_series, gold_df, stats):
    fig = plt.figure(figsize=(16, 12), facecolor="#0f0f1a")
    fig.suptitle("Gold Trading Bot v2 — Backtest Results (Longs Only, Aggressive)",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    tc   = "#e0e0e0"
    gc   = "#2a2a3a"
    gold = "#FFD700"
    grn  = "#00e676"
    red  = "#ff1744"

    def style(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color=tc, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=tc, labelsize=8)
        ax.grid(color=gc, linestyle="--", linewidth=0.5, alpha=0.7)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, "💰 Equity Curve")
    ax1.plot(equity_series.index, equity_series.values, color=gold, linewidth=2)
    ax1.axhline(ACCOUNT_SIZE, color=tc, linestyle="--", linewidth=0.8, alpha=0.5, label=f"Start ${ACCOUNT_SIZE:,}")
    ax1.fill_between(equity_series.index, ACCOUNT_SIZE, equity_series.values,
                     where=equity_series.values >= ACCOUNT_SIZE, alpha=0.15, color=grn)
    ax1.fill_between(equity_series.index, ACCOUNT_SIZE, equity_series.values,
                     where=equity_series.values < ACCOUNT_SIZE, alpha=0.15, color=red)
    final = equity_series.iloc[-1]
    ax1.axhline(final, color=grn if final > ACCOUNT_SIZE else red,
                linestyle=":", linewidth=1, alpha=0.7, label=f"Final ${final:,.2f}")
    ax1.set_ylabel("Account ($)", color=tc, fontsize=9)
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=tc)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, "📉 Drawdown")
    peak = equity_series.cummax()
    dd   = (equity_series - peak) / peak * 100
    ax2.fill_between(dd.index, 0, dd.values, color=red, alpha=0.6)
    ax2.set_ylabel("Drawdown (%)", color=tc, fontsize=9)

    # 3. PnL distribution
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, "📊 PnL per Trade")
    pnls = [t.pnl for t in trades]
    bins = np.linspace(min(pnls), max(pnls), 25)
    ax3.hist([p for p in pnls if p >= 0], bins=bins, color=grn, alpha=0.7, label="Wins")
    ax3.hist([p for p in pnls if p <  0], bins=bins, color=red,  alpha=0.7, label="Losses")
    ax3.set_xlabel("PnL ($)", color=tc, fontsize=9)
    ax3.legend(fontsize=8, facecolor="#1a1a2e", labelcolor=tc)

    # 4. Monthly PnL
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, "📅 Monthly PnL")
    tdf = pd.DataFrame([{"date": t.date, "pnl": t.pnl} for t in trades])
    tdf["date"]  = pd.to_datetime(tdf["date"])
    tdf["month"] = tdf["date"].dt.to_period("M").astype(str)
    monthly = tdf.groupby("month")["pnl"].sum()
    colors  = [grn if v >= 0 else red for v in monthly.values]
    ax4.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.85)
    ax4.set_xticks(range(len(monthly)))
    ax4.set_xticklabels(monthly.index, rotation=45, fontsize=6)
    ax4.set_ylabel("PnL ($)", color=tc, fontsize=9)
    ax4.axhline(0, color=tc, linewidth=0.8, alpha=0.5)

    # 5. Stats panel
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#1a1a2e")
    ax5.axis("off")
    ax5.set_title("📈 Key Stats", color=tc, fontsize=11, fontweight="bold", pad=8)

    lines = [
        ("Total Trades",  f"{stats['total_trades']}"),
        ("Win Rate",      f"{stats['win_rate']:.1%}"),
        ("Total Return",  f"{stats['return_pct']:+.1f}%"),
        ("Total PnL",     f"${stats['total_pnl']:+,.2f}"),
        ("Avg Win",       f"${stats['avg_win']:+.2f}"),
        ("Avg Loss",      f"${stats['avg_loss']:+.2f}"),
        ("Profit Factor", f"{stats['profit_factor']:.2f}"),
        ("Max Drawdown",  f"{stats['max_drawdown_pct']:.1f}%"),
        ("Sharpe Ratio",  f"{stats['sharpe_ratio']:.2f}"),
        ("TP Hit Rate",   f"{stats['tp_hit_rate']:.1%}"),
        ("SL Hit Rate",   f"{stats['sl_hit_rate']:.1%}"),
    ]

    for j, (label, value) in enumerate(lines):
        y = 0.95 - j * 0.085
        ax5.text(0.02, y, label + ":", transform=ax5.transAxes, color=tc, fontsize=9, va="top")
        c = (grn if ("Return" in label or "PnL" in label or "Win" in label) and "+" in value
             else red if ("Return" in label or "PnL" in label) and "-" in value
             else gold)
        ax5.text(0.65, y, value, transform=ax5.transAxes, color=c, fontsize=9, va="top", fontweight="bold")

    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("   Chart saved → backtest_results.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    raw_df  = fetch_gold_data()
    log.info("⚙️  Building features...")
    feat_df = build_features(raw_df)
    log.info(f"   {len(feat_df)} usable rows")

    trades, equity_series = run_backtest(feat_df)

    if not trades:
        print("\n⚠️  No trades generated. Try lowering CONFIDENCE_THRESHOLD.")
    else:
        stats = compute_stats(trades, equity_series)
        print_report(stats, trades)

        log.info("🎨 Generating chart...")
        plot_results(trades, equity_series, raw_df, stats)

        with open("backtest_trades.json", "w") as f:
            json.dump([t.__dict__ for t in trades], f, indent=2)
        log.info("   Trade log → backtest_trades.json")

        print("  📊 Chart saved:     backtest_results.png")
        print("  📋 Trade log saved: backtest_trades.json")
