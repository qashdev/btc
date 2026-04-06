"""
Bitcoin Trading Bot - Backtester
==================================
Runs the full strategy against 2 years of historical BTC price data.
Walk-forward testing — no lookahead bias.

Run:
    python btc_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
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
# CONFIG  (BTC-tuned)
# ─────────────────────────────────────────────
BTC_TICKER           = "BTC-USD"
BACKTEST_YEARS       = 5
CONFIDENCE_THRESHOLD = 0.50   # same as gold
MAX_RISK_PCT         = 0.05   # will be replaced by dynamic sizing
ACCOUNT_SIZE         = 10_000
STOP_LOSS_PCT        = 0.020   # 2% — BTC needs more room than gold
TAKE_PROFIT_PCT      = 0.080   # 8% — 4:1 R:R for BTC
TRAIN_WINDOW         = 120
WALK_FORWARD_STEP    = 5
HOLD_DAYS            = 7
LONGS_ONLY           = True

# Dynamic drawdown-aware sizing
MAX_DRAWDOWN_LIMIT = 0.035
RISK_WHEN_SAFE     = 0.95
RISK_WHEN_WARNING  = 0.60
RISK_WHEN_DANGER   = 0.10


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def fetch_btc_data(years=BACKTEST_YEARS) -> pd.DataFrame:
    log.info(f"📥 Fetching {years} years of BTC data...")
    df = yf.download(BTC_TICKER, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("Could not fetch BTC-USD data.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    log.info(f"   ✅ BTC-USD: {len(df)} trading days")
    return df


# ─────────────────────────────────────────────
# FEATURES  (crypto-specific)
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["return_1d"]  = d["Close"].pct_change(1)
    d["return_3d"]  = d["Close"].pct_change(3)
    d["return_7d"]  = d["Close"].pct_change(7)
    d["return_14d"] = d["Close"].pct_change(14)
    d["return_30d"] = d["Close"].pct_change(30)

    # RSI
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"]    = 100 - (100 / (1 + gain / (loss + 1e-9)))
    d["rsi_ob"] = (d["rsi"] > 70).astype(int)
    d["rsi_os"] = (d["rsi"] < 30).astype(int)

    # MACD
    ema12 = d["Close"].ewm(span=12).mean()
    ema26 = d["Close"].ewm(span=26).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # Bollinger
    sma20 = d["Close"].rolling(20).mean()
    std20 = d["Close"].rolling(20).std()
    d["bb_position"] = (d["Close"] - sma20) / (2 * std20 + 1e-9)
    d["bb_width"]    = (4 * std20) / (sma20 + 1e-9)

    # ATR
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift()).abs()
    lc  = (d["Low"]  - d["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14).mean() / d["Close"]

    # Volume
    d["volume_ratio"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9)
    d["volume_trend"] = d["Volume"].pct_change(5)

    # Price vs MAs
    d["price_vs_sma20"]  = (d["Close"] - d["Close"].rolling(20).mean())  / d["Close"]
    d["price_vs_sma50"]  = (d["Close"] - d["Close"].rolling(50).mean())  / d["Close"]
    d["price_vs_sma200"] = (d["Close"] - d["Close"].rolling(200).mean()) / d["Close"]

    # Golden/death cross
    d["ma_cross"] = np.sign(d["Close"].rolling(50).mean() - d["Close"].rolling(200).mean())

    d["sentiment"] = 0.0  # neutral placeholder (live bot uses real sentiment)

    d["target"]    = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d["next_close"] = d["Close"].shift(-1)

    return d.dropna()


FEATURE_COLS = [
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d",
    "rsi", "rsi_ob", "rsi_os",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width", "atr_pct",
    "volume_ratio", "volume_trend",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "ma_cross", "sentiment",
]


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────
@dataclass
class BacktestTrade:
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


def dynamic_position(confidence: float, equity: float, peak: float) -> float:
    # NO hard stop — always trade, scale size by drawdown
    current_dd = (peak - equity) / peak if peak > 0 else 0.0
    dd_ratio   = min(current_dd / MAX_DRAWDOWN_LIMIT, 1.0)
    if dd_ratio < 0.5:
        scale    = dd_ratio / 0.5
        max_risk = RISK_WHEN_SAFE + scale * (RISK_WHEN_WARNING - RISK_WHEN_SAFE)
    else:
        scale    = (dd_ratio - 0.5) / 0.5
        max_risk = RISK_WHEN_WARNING + scale * (RISK_WHEN_DANGER - RISK_WHEN_WARNING)
    p     = confidence
    b     = TAKE_PROFIT_PCT / STOP_LOSS_PCT
    kelly = max(0, (p * (b + 1) - 1) / b) * 0.5
    return max(equity * 0.01, min(equity * kelly, equity * max_risk))


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


def run_backtest(df: pd.DataFrame):
    log.info("🔄 Running BTC walk-forward backtest (LONGS ONLY, aggressive sizing)...")
    trades       = []
    equity       = ACCOUNT_SIZE
    peak_equity  = ACCOUNT_SIZE
    equity_curve = []
    scaler       = StandardScaler()
    model        = None
    retrain_ctr  = 0
    indices      = df.index.tolist()

    for i in range(TRAIN_WINDOW, len(df) - HOLD_DAYS):
        date = indices[i]
        row  = df.iloc[i]

        if retrain_ctr == 0 or model is None:
            train_slice = df.iloc[max(0, i - TRAIN_WINDOW):i]
            X_tr = train_slice[FEATURE_COLS].values
            y_tr = train_slice["target"].values
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

        # NaN/inf check
        feat_vals = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        if not np.isfinite(feat_vals).all():
            equity_curve.append(equity)
            continue

        X          = scaler.transform(feat_vals)
        proba      = model.predict_proba(X)[0]
        prob_up    = float(proba[1])
        prob_down  = float(proba[0])
        confidence = max(prob_up, prob_down)
        direction  = "LONG" if prob_up > prob_down else "SHORT"

        # Longs only — BTC trends up long term
        if LONGS_ONLY and direction == "SHORT":
            equity_curve.append(equity)
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            equity_curve.append(equity)
            continue

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
        peak_equity  = max(peak_equity, equity)

        trades.append(BacktestTrade(
            date=str(date)[:10],
            direction=direction,
            entry_price=round(entry, 2),
            exit_price=round(exit_price, 2),
            confidence=round(confidence, 4),
            position_usd=round(position_usd, 2),
            pnl=round(pnl, 2),
            win=win,
            stop_hit=stop_hit,
            tp_hit=tp_hit,
        ))
        equity_curve.append(equity)

    equity_series = pd.Series(
        equity_curve,
        index=df.index[TRAIN_WINDOW:TRAIN_WINDOW + len(equity_curve)]
    )
    log.info(f"   Backtest complete: {len(trades)} trades")
    return trades, equity_series


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────
def compute_stats(trades, equity_series):
    if not trades:
        return {}
    pnls    = [t.pnl for t in trades]
    wins    = [t for t in trades if t.win]
    losses  = [t for t in trades if not t.win]
    longs   = [t for t in trades if t.direction == "LONG"]
    shorts  = [t for t in trades if t.direction == "SHORT"]

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
        "max_drawdown_pct": dd.min() * 100,
        "sharpe_ratio":     (eq_ret.mean() / (eq_ret.std() + 1e-9)) * np.sqrt(365),
        "long_trades":      len(longs),
        "short_trades":     len(shorts),
        "long_win_rate":    len([t for t in longs  if t.win]) / max(len(longs), 1),
        "short_win_rate":   len([t for t in shorts if t.win]) / max(len(shorts), 1),
        "tp_hit_rate":      len([t for t in trades if t.tp_hit])   / len(trades),
        "sl_hit_rate":      len([t for t in trades if t.stop_hit]) / len(trades),
    }


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────
def plot_results(trades, equity_series, stats, output_path):
    fig = plt.figure(figsize=(16, 12), facecolor="#0d0d1a")
    fig.suptitle("Bitcoin Trading Bot — Backtest Results", fontsize=18,
                 color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    text_color = "#e0e0e0"
    grid_color = "#2a2a3a"
    btc_orange = "#F7931A"   # Bitcoin orange
    green      = "#00e676"
    red        = "#ff1744"

    def style_ax(ax, title):
        ax.set_facecolor("#12122a")
        ax.set_title(title, color=text_color, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.grid(color=grid_color, linestyle="--", linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "💰 Equity Curve (BTC Bot)")
    ax1.plot(equity_series.index, equity_series.values, color=btc_orange, linewidth=2)
    ax1.axhline(ACCOUNT_SIZE, color=text_color, linestyle="--", linewidth=0.8, alpha=0.5, label="Start $10k")
    ax1.fill_between(equity_series.index, ACCOUNT_SIZE, equity_series.values,
                     where=equity_series.values >= ACCOUNT_SIZE, alpha=0.15, color=green)
    ax1.fill_between(equity_series.index, ACCOUNT_SIZE, equity_series.values,
                     where=equity_series.values < ACCOUNT_SIZE, alpha=0.15, color=red)
    ax1.set_ylabel("Account Value ($)", color=text_color, fontsize=9)
    ax1.legend(fontsize=8, facecolor="#12122a", labelcolor=text_color)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, "📉 Drawdown")
    peak = equity_series.cummax()
    dd   = (equity_series - peak) / peak * 100
    ax2.fill_between(dd.index, 0, dd.values, color=red, alpha=0.6)
    ax2.set_ylabel("Drawdown (%)", color=text_color, fontsize=9)

    # 3. PnL distribution
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, "📊 PnL Distribution")
    pnls = [t.pnl for t in trades]
    bins = np.linspace(min(pnls), max(pnls), 30)
    ax3.hist([p for p in pnls if p >= 0], bins=bins, color=green, alpha=0.7, label="Wins")
    ax3.hist([p for p in pnls if p <  0], bins=bins, color=red,   alpha=0.7, label="Losses")
    ax3.set_xlabel("PnL ($)", color=text_color, fontsize=9)
    ax3.legend(fontsize=8, facecolor="#12122a", labelcolor=text_color)

    # 4. Monthly PnL
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, "📅 Monthly PnL")
    tdf = pd.DataFrame([{"date": t.date, "pnl": t.pnl} for t in trades])
    tdf["date"]  = pd.to_datetime(tdf["date"])
    tdf["month"] = tdf["date"].dt.to_period("M").astype(str)
    monthly = tdf.groupby("month")["pnl"].sum()
    colors  = [green if v >= 0 else red for v in monthly.values]
    ax4.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(monthly)))
    ax4.set_xticklabels(monthly.index, rotation=45, fontsize=6)
    ax4.set_ylabel("PnL ($)", color=text_color, fontsize=9)
    ax4.axhline(0, color=text_color, linewidth=0.8, alpha=0.5)

    # 5. Stats panel
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#12122a")
    ax5.axis("off")
    ax5.set_title("₿ Key Statistics", color=text_color, fontsize=11, fontweight="bold", pad=8)

    stat_lines = [
        ("Total Trades",   f"{stats['total_trades']}"),
        ("Win Rate",       f"{stats['win_rate']:.1%}"),
        ("Total PnL",      f"${stats['total_pnl']:+,.2f}"),
        ("Total Return",   f"{stats['return_pct']:+.1f}%"),
        ("Avg Win",        f"${stats['avg_win']:+.2f}"),
        ("Avg Loss",       f"${stats['avg_loss']:+.2f}"),
        ("Profit Factor",  f"{stats['profit_factor']:.2f}"),
        ("Max Drawdown",   f"{stats['max_drawdown_pct']:.1f}%"),
        ("Sharpe Ratio",   f"{stats['sharpe_ratio']:.2f}"),
        ("Long Win Rate",  f"{stats['long_win_rate']:.1%}"),
        ("Short Win Rate", f"{stats['short_win_rate']:.1%}"),
        ("TP Hit Rate",    f"{stats['tp_hit_rate']:.1%}"),
        ("SL Hit Rate",    f"{stats['sl_hit_rate']:.1%}"),
    ]

    for j, (label, value) in enumerate(stat_lines):
        y_pos = 0.95 - j * 0.072
        ax5.text(0.02, y_pos, label + ":", transform=ax5.transAxes,
                 color=text_color, fontsize=9, va="top")
        color = (green  if ("Win" in label or "PnL" in label or "Return" in label) and "+" in value
                 else red if ("PnL" in label or "Return" in label) and "-" in value
                 else btc_orange)
        ax5.text(0.65, y_pos, value, transform=ax5.transAxes,
                 color=color, fontsize=9, va="top", fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"   Chart saved → {output_path}")


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────
def print_report(stats, trades):
    print("\n" + "="*55)
    print("  BITCOIN TRADING BOT — BACKTEST REPORT")
    print("="*55)
    print(f"  Backtest period : {BACKTEST_YEARS} years")
    print(f"  Starting capital: ${ACCOUNT_SIZE:,}")
    print(f"  Stop Loss:        {STOP_LOSS_PCT:.1%}  |  Take Profit: {TAKE_PROFIT_PCT:.1%}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print("-"*55)
    print(f"  Total trades    : {stats['total_trades']}")
    print(f"  Win rate        : {stats['win_rate']:.1%}  ← your real number")
    print(f"  Total PnL       : ${stats['total_pnl']:+,.2f}")
    print(f"  Total return    : {stats['return_pct']:+.1f}%")
    print(f"  Avg win         : ${stats['avg_win']:+.2f}")
    print(f"  Avg loss        : ${stats['avg_loss']:+.2f}")
    print(f"  Profit factor   : {stats['profit_factor']:.2f}")
    dd = stats['max_drawdown_pct']
    dd_flag = "✅ Under 6%" if abs(dd) < 6 else "⚠️  OVER 6% — reduce risk"
    print(f"  Max drawdown    : {dd:.1f}%  {dd_flag}")
    print(f"  Sharpe ratio    : {stats['sharpe_ratio']:.2f}")
    print("-"*55)
    print(f"  Long  trades    : {stats['long_trades']}  (win: {stats['long_win_rate']:.1%})")
    print(f"  Short trades    : {stats['short_trades']}  (win: {stats['short_win_rate']:.1%})")
    print(f"  Take profit hit : {stats['tp_hit_rate']:.1%}")
    print(f"  Stop loss hit   : {stats['sl_hit_rate']:.1%}")
    print("="*55)

    if trades:
        print("\n  Last 5 trades:")
        print(f"  {'Date':<12} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'PnL':>8} {'Result'}")
        print("  " + "-"*58)
        for t in trades[-5:]:
            result = "✅ WIN" if t.win else "❌ LOSS"
            print(f"  {t.date:<12} {t.direction:<6} ${t.entry_price:>9,.0f} ${t.exit_price:>9,.0f} ${t.pnl:>+7.2f}  {result}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    raw_df  = fetch_btc_data(years=BACKTEST_YEARS)
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
        chart_path = "btc_backtest_results.png"
        plot_results(trades, equity_series, stats, output_path=chart_path)

        log_path = "btc_backtest_trades.json"
        with open(log_path, "w") as f:
            json.dump([t.__dict__ for t in trades], f, indent=2)
        log.info(f"   Trade log → {log_path}")

        print(f"btc_backtest_results.png")
        print(f"btc_backtest_trades.json")
