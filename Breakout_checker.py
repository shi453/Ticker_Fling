import json
import os
import time

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Breakout Scanner Pro", layout="wide")

# =====================================================================
# CONFIG  (defaults — most are overridable from the sidebar)
# =====================================================================
CONFIG = {
    # data
    "min_bars": 60,            # refuse to analyse a stock with fewer bars
    "chart_days": 252,         # how many trading days to draw on the chart
    "benchmark": "^NSEI",      # index used for Relative Strength (NIFTY 50)
    # detection
    "box_method": "Pivot High",  # "Pivot High" (default) or "Darvas Box"
    "pivot_window": 3,         # bars on each side for a swing-high pivot
    "res_lookback": 120,       # fallback resistance lookback
    "touch_tolerance": 0.01,   # +/-1% counts as "touching" resistance
    # long-tested base (multi-year horizontal ceiling)
    "lb_lookback": 1000,       # bars to search for a long-tested level (~4y)
    "lb_window": 5,            # pivot window for the long base
    "lb_tol": 0.03,            # +/-3% counts as a touch of the level
    "lb_min_touches": 3,       # min touches to call it a tested level
    "lb_long_bars": 250,       # >= ~1y span = "long base"
    "lb_multiyear_bars": 500,  # >= ~2y span = "multi-year base"
    "lb_recent_bars": 126,     # ~6mo: a level needs a touch within this to stay "active"
    "lb_min_recent": 1,        # min touches inside the recent window
    "lb_cross_bars": 30,       # a breakout must be a FRESH cross (price below within N bars)
    "lb_stop_lowbars": 60,     # recent-swing-low window for the level-based stop
    "lb_stop_atr": 1.5,        # ATR multiple for the level-based stop
    # thresholds (editable in sidebar)
    "near_pct": 3.0,           # within X% of resistance = "near"
    "vol_surge_mult": 1.5,     # breakout volume must beat avg * this
    "rsi_low": 50,             # healthy momentum band
    "rsi_high": 70,            # above this = overbought
    # scoring weights (editable in sidebar) — sum to 100
    "w_trend": 20,
    "w_near": 15,
    "w_compression": 10,
    "w_volume_dry": 10,
    "w_higher_lows": 10,
    "w_atr": 5,
    "w_touches": 10,
    "w_rsi": 5,
    "w_rel_strength": 5,
    "w_obv": 10,
    # backtest
    "bt_threshold": 80,        # score needed to take a historical trade
    "bt_horizon": 20,          # bars to let a trade resolve
    "bt_lookback": 750,        # how far back to backtest (≈ 3 years)
}

# Plain-English explanation for every checklist item.
EXPLANATIONS = {
    "Trend": "Price is above EMA20, and EMA20 > EMA50 > EMA200. This 'stacked' "
             "moving-average order confirms a healthy up-trend across short, "
             "medium and long horizons — breakouts work best with the trend.",
    "Near Resistance": "Price is sitting close to the detected resistance level. "
                       "The closer it is, the smaller the move needed to break out "
                       "and the tighter your stop can be.",
    "Compression": "The recent price range is tighter than the prior range — the "
                   "stock is 'coiling'. Volatility contraction often precedes a "
                   "sharp expansion (the breakout).",
    "Volume Dry-up (in base)": "During the consolidation/base, recent average volume "
                     "is lower than the prior period. Falling volume while the stock "
                     "bases means sellers are exhausted — fuel for a move once buyers "
                     "step in. (Note: the *breakout day itself* should instead show a "
                     "volume SPIKE — see the Volume Confirmation section.)",
    "Higher Lows": "The most recent swing low is higher than the previous one. "
                   "Buyers are stepping in earlier each time — accumulation.",
    "Volume Accumulation (OBV)": "On-Balance Volume is rising — more volume flows in "
                   "on up-days than out on down-days. Rising OBV while price bases "
                   "means big players are quietly accumulating before the breakout.",
    "ATR Compression": "Average True Range (volatility) is shrinking. Like price "
                       "compression, low ATR is the 'quiet before the move'.",
    "Resistance Touches": "How many times price has tested this resistance. More "
                          "touches = a more significant level, so a clean break is "
                          "more meaningful.",
    "RSI Healthy": "RSI(14) is in the momentum band (default 50–70): strong enough "
                   "to show buying interest, but not so high that it's overbought "
                   "and prone to a pullback.",
    "Relative Strength": "The stock is outperforming the benchmark index over the "
                         "lookback window. Leaders break out first and run furthest.",
}

# Precise mechanics — "what exactly we check" for each checklist item.
METHODS = {
    "Trend": "Pass if Close > EMA20 > EMA50 > EMA200 (all exponential moving "
             "averages of the closing price, computed on full history).",
    "Near Resistance": "Pass if |(resistance − price) / resistance × 100| ≤ the "
                       "'near-resistance' threshold set in the sidebar.",
    "Compression": "Pass if the High−Low range of the last 10 bars is smaller than "
                   "the High−Low range of the 20 bars before that (bars −30 to −10).",
    "Volume Dry-up (in base)": "Pass if the mean Volume of the last 10 bars is lower "
                     "than the mean Volume of the prior 20 bars (bars −30 to −10). "
                     "This measures the BASE, not the breakout day.",
    "Higher Lows": "Pass if the lowest Low of the last 10 bars is higher than the "
                   "lowest Low of the prior 10 bars (bars −20 to −10).",
    "Volume Accumulation (OBV)": "OBV adds the day's volume on up-closes and subtracts "
                   "it on down-closes. Pass if the mean OBV of the last 10 bars is "
                   "higher than the mean OBV of the prior 20 bars (i.e. OBV is rising).",
    "ATR Compression": "Pass if the mean ATR(14) of the last 10 bars is lower than "
                       "the mean ATR(14) of the prior 20 bars (bars −30 to −10).",
    "Resistance Touches": "Count, over the last 80 bars, how many Highs fall within "
                          "±1% of the resistance level. Pass if that count ≥ 3.",
    "RSI Healthy": "Compute Wilder's RSI(14) on closing price. Pass if the latest "
                   "value sits inside the healthy band set in the sidebar.",
    "Relative Strength": "Compare the stock's 60-day % return against the benchmark's "
                         "60-day % return. Pass if the stock's return is higher.",
}

# Definitions shown as the ⓘ tooltip next to each sidebar weight.
WEIGHT_HELP = {
    "w_trend": "TREND: The order of the moving averages. When price sits above the "
               "20-day EMA, which sits above the 50-day, which sits above the "
               "200-day, all timeframes agree the stock is rising. Breakouts that go "
               "with the trend succeed far more often than counter-trend ones.",
    "w_near": "NEAR RESISTANCE: Resistance is a price ceiling sellers have defended "
              "before. The closer price is to it, the smaller the push needed to "
              "break out and the tighter (cheaper) your stop can be.",
    "w_compression": "COMPRESSION: A 'coil'. When the recent price range narrows "
                     "versus the prior range, the stock is winding up like a spring — "
                     "tight ranges often precede a sharp expansion (the breakout).",
    "w_volume_dry": "VOLUME DRY-UP (IN BASE): Falling volume while a stock bases means "
                    "sellers are exhausted and few shares are changing hands. That "
                    "leaves room for a sharp move up once buyers return. This is the "
                    "consolidation phase — the breakout DAY itself should instead show "
                    "a volume spike, which is checked separately.",
    "w_obv": "VOLUME ACCUMULATION (OBV): On-Balance Volume is a running total that "
             "adds a day's volume when the stock closes up and subtracts it when it "
             "closes down. A rising OBV during a flat base reveals hidden buying "
             "(accumulation) before it shows up in price — a leading bullish tell.",
    "w_higher_lows": "HIGHER LOWS: Each dip bottoms higher than the last, showing "
                     "buyers are stepping in earlier and more aggressively — classic "
                     "accumulation under a resistance ceiling.",
    "w_atr": "ATR COMPRESSION: ATR (Average True Range) is the average size of a "
             "stock's daily move — i.e. its volatility, in price points. ATR "
             "compression means that daily movement is shrinking (the stock is going "
             "quiet). This 'calm before the storm' frequently comes right before an "
             "explosive breakout, which is why a falling ATR is bullish for a setup.",
    "w_touches": "RESISTANCE TOUCHES: How many times price has tested the same "
                 "ceiling. The more touches, the more significant the level — so a "
                 "clean break above it is a stronger, more meaningful signal.",
    "w_rsi": "RSI HEALTHY: RSI (Relative Strength Index) is a 0–100 momentum gauge. "
             "We want it in a healthy band (default 50–70): above 50 shows real "
             "buying momentum, but below ~70 keeps it from being 'overbought' and "
             "prone to a pullback right as it breaks out.",
    "w_rel_strength": "RELATIVE STRENGTH: Whether the stock is beating the market "
                      "index over the last 60 days. Market leaders break out first and "
                      "run furthest, so out-performance is a strong tailwind.",
}


# =====================================================================
# DATA LOADERS  (cached — avoids re-downloading on every interaction)
# =====================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_history(ticker):
    """Full daily history. Retries on transient failures (Yahoo throttles shared
    cloud IPs harder), and returns an empty DataFrame if it ultimately fails."""
    for attempt in range(3):
        try:
            df = yf.download(
                ticker, period="max", interval="1d",
                auto_adjust=False, progress=False,
            )
            if df is not None and not df.empty:
                # Flatten the MultiIndex yfinance returns for single tickers.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(inplace=True)
                return df
        except Exception:
            pass
        if attempt < 2:
            time.sleep(0.8 * (attempt + 1))     # brief backoff before retrying
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_info(ticker):
    """Fundamental / quote metadata. Returns {} on failure."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def benchmark_return(symbol, days):
    """Percentage return of the benchmark over `days` trading days."""
    hist = load_history(symbol)
    if hist.empty or len(hist) < days:
        return None
    close = hist["Close"]
    return (close.iloc[-1] - close.iloc[-days]) / close.iloc[-days] * 100


@st.cache_data(show_spinner=False, ttl=3600)
def market_regime(symbol):
    """Is the broad market healthy? (index trading above its 200-day EMA)."""
    hist = load_history(symbol)
    if hist.empty or len(hist) < 200:
        return None
    ema200 = hist["Close"].ewm(span=200, adjust=False).mean().iloc[-1]
    price = hist["Close"].iloc[-1]
    pct = (price - ema200) / ema200 * 100
    return {"price": float(price), "ema200": float(ema200),
            "risk_on": bool(price > ema200), "pct": float(pct)}


# =====================================================================
# INDICATORS
# =====================================================================
def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # True Range -> ATR14
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ),
    )
    df["ATR14"] = df["TR"].rolling(14).mean()

    # RSI(14) — Wilder-style smoothing
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # On-Balance Volume: cumulative volume signed by the day's price change.
    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()

    # ADX(14) — trend STRENGTH (Wilder). +DI/-DI give direction.
    period = 14
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_w = df["TR"].ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    return df


# =====================================================================
# SMALL HELPERS
# =====================================================================
def fmt(x, nd=2):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def normalize_ticker(t):
    """Auto-append '.NS' (NSE) for bare symbols. Leaves indices (^...) and
    symbols that already carry an exchange suffix (with a '.') untouched."""
    t = (t or "").strip().upper()
    if not t or t.startswith("^") or "." in t:
        return t
    return t + ".NS"


# =====================================================================
# WATCHLIST  (persisted to a JSON file beside this script)
# =====================================================================
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # __file__ missing in some run contexts
    _HERE = os.getcwd()
WATCHLIST_FILE = os.path.join(_HERE, "watchlist.json")


def load_watchlist():
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted({str(t).strip().upper() for t in data if str(t).strip()})
    except Exception:
        return []


def save_watchlist(tickers):
    try:
        clean = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
        return True
    except Exception:
        return False


def get_watchlist():
    """Session-scoped watchlist. Seeded once from the local file (so a desktop
    user keeps their saved list); on Streamlit Cloud the file is absent, so it
    simply starts empty and lives only for the browser session."""
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = load_watchlist()
    return st.session_state["watchlist"]


def set_watchlist(tickers):
    """Update the session watchlist and best-effort persist to disk (a no-op
    that vanishes on cloud restarts — that's the intended session-only behaviour)."""
    clean = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    st.session_state["watchlist"] = clean
    save_watchlist(clean)
    return clean


# =====================================================================
# RESISTANCE (pivot / swing-high detection)
# =====================================================================
def find_resistance(df, lookback=None, window=None):
    lookback = lookback or CONFIG["res_lookback"]
    window = window or CONFIG["pivot_window"]

    highs = df["High"].values
    levels = []
    for i in range(window, len(highs) - window):
        left = highs[i - window:i]
        right = highs[i + 1:i + window + 1]
        if highs[i] > max(left) and highs[i] > max(right):
            levels.append(highs[i])

    if not levels:
        return float(df["High"].tail(lookback).max())
    return float(max(levels[-5:]))


def base_metrics(df, resistance, cap=None, tol=0.005):
    """
    Describe the consolidation ('base') under resistance.
    Returns (length_in_bars, depth_pct, base_low).
    The base is the most recent stretch of bars whose highs stay under
    resistance (within `tol`), looking back at most `cap` bars.
    """
    cap = cap or CONFIG["res_lookback"]
    highs = df["High"].values
    lows = df["Low"].values
    n = len(highs)
    limit = max(0, n - cap)

    start = n - 1
    for k in range(n - 1, limit - 1, -1):
        if highs[k] > resistance * (1 + tol):
            break          # a prior high above resistance => base starts after here
        start = k

    length = n - start
    base_low = float(lows[start:].min())
    depth = (resistance - base_low) / resistance * 100
    return length, depth, base_low


def darvas_box(df, confirm=3, lookback=None):
    """
    Most recent Darvas box.
      • Ceiling (box top) = the highest 'confirmed' high — a high not exceeded by
        the next `confirm` bars (Darvas's rule for a held top).
      • Floor (box bottom) = the lowest low made after that top.
    Returns {top, bottom, top_date, bottom_date}.
    """
    confirm = confirm or 3
    lookback = lookback or CONFIG["res_lookback"]
    sub = df.tail(lookback)
    highs = sub["High"].values
    lows = sub["Low"].values
    idx = sub.index
    n = len(highs)

    top, top_i, best = None, None, -1.0
    for i in range(max(0, n - confirm)):
        h = highs[i]
        if h > best and all(highs[i + 1 + k] < h for k in range(confirm)):
            top, top_i, best = float(h), i, float(h)
    if top is None:                                   # fallback: plain highest high
        top_i = int(highs.argmax())
        top = float(highs[top_i])

    if top_i + 1 < n:
        region = lows[top_i + 1:]
        bottom = float(region.min())
        bottom_i = top_i + 1 + int(region.argmin())
    else:
        bottom = float(lows[top_i])
        bottom_i = top_i
    return {"top": top, "bottom": bottom,
            "top_date": idx[top_i], "bottom_date": idx[bottom_i]}


def long_base(df, lookback=None, window=None, tol=None, min_touches=None,
              recent_bars=None, min_recent=None):
    """
    Find the most-tested horizontal ceiling that is STILL ACTIVE.

    Scans swing-high pivots in the lookback and clusters them by price (±tol).
    A cluster only qualifies if it has >= min_touches total AND >= min_recent
    touches within the last `recent_bars` — so stale levels the stock left behind
    long ago are ignored. Also returns `bars_since_below` (how fresh any cross is).
    Returns raw facts, or None if no active tested ceiling exists.
    """
    lookback = lookback or CONFIG["lb_lookback"]
    window = window or CONFIG["lb_window"]
    tol = tol or CONFIG["lb_tol"]
    min_touches = min_touches or CONFIG["lb_min_touches"]
    recent_bars = recent_bars or CONFIG["lb_recent_bars"]
    min_recent = CONFIG["lb_min_recent"] if min_recent is None else min_recent

    sub = df.tail(lookback)
    highs = sub["High"].values
    closes = sub["Close"].values
    idx = sub.index
    n = len(highs)
    if n < 60:
        return None

    pivots = []                                   # (price, position)
    for i in range(window, n - window):
        if highs[i] >= highs[i - window:i].max() and highs[i] >= highs[i + 1:i + window + 1].max():
            pivots.append((float(highs[i]), i))
    if len(pivots) < min_touches:
        return None

    recent_cut = n - recent_bars
    # most-touched cluster that is STILL ACTIVE (>= min_recent recent touches); tie -> higher level
    best = None
    for price, _ in pivots:
        level = max(p for (p, q) in pivots if abs(p - price) <= price * tol)
        tp = [q for (p, q) in pivots if abs(p - level) <= level * tol]
        if len(tp) < min_touches:
            continue
        recent = sum(1 for q in tp if q >= recent_cut)
        if recent < min_recent:
            continue
        key = (len(tp), level)
        if best is None or key > best[0]:
            best = (key, level, tp, recent)
    if best is None:
        return None

    _, level, touch_pos, recent_touches = best
    first_pos, last_pos = min(touch_pos), max(touch_pos)
    below = np.where(closes < level)[0]
    bars_since_below = int((n - 1) - below[-1]) if len(below) else None
    last_close = float(closes[-1])
    return {
        "level": float(level),
        "touches": len(touch_pos),
        "recent_touches": int(recent_touches),
        "bars_since_below": bars_since_below,
        "span_bars": int(n - first_pos),
        "first_date": idx[first_pos],
        "last_touch_date": idx[last_pos],
        "dist_pct": (level - last_close) / level * 100,
        "broke": last_close > level,
        "last_vol": float(sub["Volume"].iloc[-1]),
        "avg_vol": float(sub["Volume"].tail(20).mean()),
        "touch_dates": [idx[q] for q in touch_pos],
        "touch_prices": [float(highs[q]) for q in touch_pos],
    }


# =====================================================================
# INDIVIDUAL CHECKS  -> each returns (passed: bool, detail: str)
# =====================================================================
def check_trend(df):
    price = df["Close"].iloc[-1]
    e20, e50, e200 = df["EMA20"].iloc[-1], df["EMA50"].iloc[-1], df["EMA200"].iloc[-1]
    ok = price > e20 > e50 > e200
    detail = f"Price {price:.2f} | EMA20 {e20:.2f} | EMA50 {e50:.2f} | EMA200 {e200:.2f}"
    return ok, detail


def check_near(df, resistance, near_pct):
    price = df["Close"].iloc[-1]
    dist_pct = (resistance - price) / resistance * 100
    dist_pts = resistance - price
    ok = abs(dist_pct) <= near_pct
    detail = f"{dist_pts:+.2f} pts ({dist_pct:+.2f}%) from resistance"
    return ok, detail, dist_pct, dist_pts


def check_compression(df):
    recent = df["High"].tail(10).max() - df["Low"].tail(10).min()
    prev = df["High"].iloc[-30:-10].max() - df["Low"].iloc[-30:-10].min()
    ok = recent < prev
    detail = f"Recent range {recent:.2f} vs prior {prev:.2f}"
    return ok, detail


def check_volume_dry(df):
    recent = df["Volume"].tail(10).mean()
    prev = df["Volume"].iloc[-30:-10].mean()
    ok = recent < prev
    detail = f"Recent avg vol {recent:,.0f} vs prior {prev:,.0f}"
    return ok, detail


def check_higher_lows(df):
    recent_low = df["Low"].tail(10).min()
    prev_low = df["Low"].iloc[-20:-10].min()
    ok = recent_low > prev_low
    detail = f"Recent low {recent_low:.2f} vs prior low {prev_low:.2f}"
    return ok, detail


def check_atr_compression(df):
    recent = df["ATR14"].tail(10).mean()
    prev = df["ATR14"].iloc[-30:-10].mean()
    ok = recent < prev
    detail = f"Recent ATR {recent:.2f} vs prior {prev:.2f}"
    return ok, detail


def check_obv(df):
    recent = df["OBV"].tail(10).mean()
    prev = df["OBV"].iloc[-30:-10].mean()
    ok = recent > prev
    detail = f"Recent OBV {recent:,.0f} vs prior {prev:,.0f} ({'rising' if ok else 'falling'})"
    return ok, detail


def count_resistance_touches(df, resistance, tolerance=None):
    tol = tolerance or CONFIG["touch_tolerance"]
    touches = int(sum(1 for h in df["High"].tail(80)
                      if abs(h - resistance) <= resistance * tol))
    return touches


def check_rsi(df, lo, hi):
    rsi = df["RSI"].iloc[-1]
    ok = lo <= rsi <= hi
    detail = f"RSI(14) = {rsi:.1f} (band {lo}-{hi})"
    return ok, detail, rsi


def check_relative_strength(df, days=60):
    bench = benchmark_return(CONFIG["benchmark"], days)
    if bench is None or len(df) < days:
        return False, "Benchmark data unavailable", None
    stock = (df["Close"].iloc[-1] - df["Close"].iloc[-days]) / df["Close"].iloc[-days] * 100
    ok = stock > bench
    detail = f"Stock {stock:+.1f}% vs {CONFIG['benchmark']} {bench:+.1f}% (60d)"
    return ok, detail, stock - bench


# =====================================================================
# BREAKOUT CONFIRMATION  (has it *already* broken out today?)
# =====================================================================
def breakout_confirmed(df, resistance, vol_mult):
    last_close = df["Close"].iloc[-1]
    last_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].tail(20).mean()
    above = last_close > resistance
    vol_ok = last_vol > avg_vol * vol_mult
    return above and vol_ok, above, vol_ok, last_vol, avg_vol


# =====================================================================
# SCORING ENGINE
# =====================================================================
def calculate_score(df, resistance, cfg):
    """Returns (score, checklist, dist_pct, dist_pts, touches)."""
    checklist = {}      # name -> {passed, detail, points, explanation}

    def add(name, ok, detail, weight):
        checklist[name] = {
            "passed": bool(ok),
            "detail": detail,
            "points": weight if ok else 0,
            "max": weight,
            "explanation": EXPLANATIONS.get(name, ""),
            "method": METHODS.get(name, ""),
        }

    ok, d = check_trend(df);                       add("Trend", ok, d, cfg["w_trend"])
    ok, d, dist_pct, dist_pts = check_near(df, resistance, cfg["near_pct"])
    add("Near Resistance", ok, d, cfg["w_near"])
    ok, d = check_compression(df);                 add("Compression", ok, d, cfg["w_compression"])
    ok, d = check_volume_dry(df);                  add("Volume Dry-up (in base)", ok, d, cfg["w_volume_dry"])
    ok, d = check_obv(df);                          add("Volume Accumulation (OBV)", ok, d, cfg["w_obv"])
    ok, d = check_higher_lows(df);                 add("Higher Lows", ok, d, cfg["w_higher_lows"])
    ok, d = check_atr_compression(df);             add("ATR Compression", ok, d, cfg["w_atr"])

    touches = count_resistance_touches(df, resistance)
    add("Resistance Touches", touches >= 3,
        f"{touches} touch(es) within ±{CONFIG['touch_tolerance']*100:.0f}%", cfg["w_touches"])

    ok, d, _ = check_rsi(df, cfg["rsi_low"], cfg["rsi_high"]); add("RSI Healthy", ok, d, cfg["w_rsi"])
    ok, d, _ = check_relative_strength(df);        add("Relative Strength", ok, d, cfg["w_rel_strength"])

    score = sum(item["points"] for item in checklist.values())
    return score, checklist, dist_pct, dist_pts, touches


# =====================================================================
# TRADE PLANNER
# =====================================================================
def trade_levels(df, resistance, structural_stop=None):
    price = df["Close"].iloc[-1]
    atr = df["ATR14"].iloc[-1]
    entry = resistance * 1.002              # trigger just above resistance
    if structural_stop is not None and structural_stop < entry:
        stop = structural_stop * 0.997     # Darvas: just below the box floor
    else:
        stop = entry - 1.5 * atr           # volatility-based stop
    risk = entry - stop
    target = entry + 2 * risk               # 2R target
    rr = (target - entry) / risk if risk else 0
    return price, entry, stop, target, atr, risk, rr


def get_signal(score):
    if score >= 80:
        return "🟢 STRONG BREAKOUT SETUP"
    elif score >= 65:
        return "🟡 WATCH CLOSELY"
    return "🔴 NO TRADE"


# =====================================================================
# BACKTEST  (does a high score actually win historically?)
# =====================================================================
def backtest(df, cfg):
    """
    Walk history bar-by-bar. Whenever the as-of score >= threshold AND a
    volume-confirmed breakout fires, simulate a trade (entry = resistance*1.002,
    stop = entry - 1.5*ATR, target = entry + 2*risk) and see whether target or
    stop is hit first within `horizon` bars. Non-overlapping trades only.

    NOTE: this reuses the live scoring. The Relative-Strength component uses the
    *current* benchmark return, so it carries minor look-ahead; treat the win-rate
    as indicative, not exact.
    """
    threshold = cfg["bt_threshold"]
    horizon = cfg["bt_horizon"]
    data = df.tail(cfg["bt_lookback"]).reset_index(drop=True)
    n = len(data)
    trades = []

    darvas = cfg.get("box_method") == "Darvas Box"
    i = cfg["min_bars"]
    while i < n - 1:
        sub = data.iloc[:i + 1]
        if darvas:
            bx = darvas_box(sub)
            res, sstop = bx["top"], bx["bottom"]
        else:
            res, sstop = find_resistance(sub), None
        score = calculate_score(sub, res, cfg)[0]
        confirmed = breakout_confirmed(sub, res, cfg["vol_surge_mult"])[0]

        if score >= threshold and confirmed:
            atr = sub["ATR14"].iloc[-1]
            entry = res * 1.002
            if sstop is not None and sstop < entry:
                stop = sstop * 0.997
            else:
                stop = entry - 1.5 * atr
            risk = entry - stop
            target = entry + 2 * risk

            outcome, exit_i = None, None
            if risk > 0:
                for j in range(i + 1, min(i + 1 + horizon, n)):
                    lo, hi = data["Low"].iloc[j], data["High"].iloc[j]
                    if lo <= stop:                 # stop checked first = conservative
                        outcome, exit_i = "loss", j
                        break
                    if hi >= target:
                        outcome, exit_i = "win", j
                        break
            if outcome is None:                    # timed out — mark to last close
                exit_i = min(i + horizon, n - 1)
                last = data["Close"].iloc[exit_i]
                r_mult = (last - entry) / risk if risk else 0
                outcome = "win" if r_mult > 0 else "loss"
                trades.append({"r": r_mult, "outcome": outcome, "bars": exit_i - i})
            else:
                trades.append({"r": 2.0 if outcome == "win" else -1.0,
                               "outcome": outcome, "bars": exit_i - i})
            i = exit_i + 1                          # no overlapping trades
        else:
            i += 1

    total = len(trades)
    if total == 0:
        return {"trades": 0}
    wins = sum(1 for t in trades if t["outcome"] == "win")
    win_rate = wins / total * 100
    avg_r = sum(t["r"] for t in trades) / total
    avg_hold = sum(t["bars"] for t in trades) / total
    return {
        "trades": total, "wins": wins, "losses": total - wins,
        "win_rate": win_rate, "avg_r": avg_r, "avg_hold": avg_hold,
        "expectancy": avg_r,           # R per trade
        "span_bars": n,
    }


# =====================================================================
# CURRENT STATS (price / 52wk / all-time / PE / holdings)
# =====================================================================
def build_stats(df, info):
    price = info.get("currentPrice") or info.get("regularMarketPrice") or df["Close"].iloc[-1]
    day_high = info.get("dayHigh") or info.get("regularMarketDayHigh") or df["High"].iloc[-1]
    day_low = info.get("dayLow") or info.get("regularMarketDayLow") or df["Low"].iloc[-1]
    wk_high = info.get("fiftyTwoWeekHigh") or df["High"].tail(252).max()
    wk_low = info.get("fiftyTwoWeekLow") or df["Low"].tail(252).min()

    # ADX = trend strength; +DI vs -DI = direction
    adx = df["ADX"].iloc[-1] if "ADX" in df else np.nan
    if pd.isna(adx):
        adx_str = "N/A"
    else:
        strength = "strong trend" if adx >= 25 else "building" if adx >= 20 else "weak / choppy"
        bullish = df["PLUS_DI"].iloc[-1] >= df["MINUS_DI"].iloc[-1]
        adx_str = f"{adx:.1f} ({strength}, {'↑ bullish' if bullish else '↓ bearish'})"

    return {
        "Current Price": fmt(price),
        "Day High": fmt(day_high),
        "Day Low": fmt(day_low),
        "52-Week High": fmt(wk_high),
        "52-Week Low": fmt(wk_low),
        "All-Time High": fmt(float(df["High"].max())),
        "All-Time Low": fmt(float(df["Low"].min())),
        "ADX (14)": adx_str,
        "Trailing P/E": fmt(info.get("trailingPE")),
        "Forward P/E": fmt(info.get("forwardPE")),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        # yfinance free API does NOT expose industry-average PE.
        "Industry P/E": "N/A (not provided by data source)",
        "Promoter/Insider Holding": (
            fmt(info.get("heldPercentInsiders", 0) * 100) + "%"
            if info.get("heldPercentInsiders") is not None else "N/A"
        ),
        "Institutional Holding": (
            fmt(info.get("heldPercentInstitutions", 0) * 100) + "%"
            if info.get("heldPercentInstitutions") is not None else "N/A"
        ),
        "Market Cap": fmt(info.get("marketCap"), 0),
    }


# =====================================================================
# FULL SINGLE-TICKER ANALYSIS (shared by single + scan modes)
# =====================================================================
def analyze(ticker, cfg):
    raw = load_history(ticker)
    if raw.empty:
        return {"error": "No data found"}
    if len(raw) < cfg["min_bars"]:
        return {"error": f"Only {len(raw)} bars — need >= {cfg['min_bars']}"}

    df = add_indicators(raw)

    # resistance / stop depend on the chosen box method
    if cfg.get("box_method") == "Darvas Box":
        box = darvas_box(df)
        resistance = box["top"]
        structural_stop = box["bottom"]
    else:
        box = None
        resistance = find_resistance(df)
        structural_stop = None

    score, checklist, dist_pct, dist_pts, touches = calculate_score(df, resistance, cfg)
    price, entry, stop, target, atr, risk, rr = trade_levels(df, resistance, structural_stop)
    confirmed, above, vol_ok, last_vol, avg_vol = breakout_confirmed(
        df, resistance, cfg["vol_surge_mult"])
    base_len, base_depth, base_low = base_metrics(df, resistance)

    # long-tested horizontal ceiling ("multi-year base") — display/badge only
    lb = long_base(df, cfg.get("lb_lookback"), cfg.get("lb_window"),
                   cfg.get("lb_tol"), cfg.get("lb_min_touches"),
                   cfg.get("lb_recent_bars"), cfg.get("lb_min_recent"))
    if lb:
        lb["vol_ratio"] = lb["last_vol"] / lb["avg_vol"] if lb["avg_vol"] else 0.0
        vol_ok_lb = lb["vol_ratio"] >= cfg["vol_surge_mult"]
        span = lb["span_bars"]
        # a breakout must be a FRESH cross: price was below the level within the window
        fresh_cross = lb["bars_since_below"] is not None and \
            lb["bars_since_below"] <= cfg["lb_cross_bars"]
        near = abs(lb["dist_pct"]) <= cfg["near_pct"]
        if lb["broke"] and vol_ok_lb and fresh_cross:
            if span >= cfg["lb_multiyear_bars"]:
                lb["state"] = "multiyear_breakout"
            elif span >= cfg["lb_long_bars"]:
                lb["state"] = "long_breakout"
            else:
                lb["state"] = "breakout"
        elif (not lb["broke"]) and near and span >= cfg["lb_long_bars"]:
            lb["state"] = "testing"
        else:
            lb["state"] = "none"

    return {
        "df": df, "resistance": resistance, "score": score, "checklist": checklist,
        "dist_pct": dist_pct, "dist_pts": dist_pts, "touches": touches,
        "price": price, "entry": entry, "stop": stop, "target": target,
        "atr": atr, "risk": risk, "rr": rr,
        "signal": get_signal(score), "confirmed": confirmed,
        "above": above, "vol_ok": vol_ok, "last_vol": last_vol, "avg_vol": avg_vol,
        "base_len": base_len, "base_depth": base_depth, "base_low": base_low,
        "box": box, "box_method": cfg.get("box_method", "Pivot High"),
        "long_base": lb,
    }


# =====================================================================
# SIDEBAR — tunable settings
# =====================================================================
st.sidebar.header("⚙️ Settings")

if st.sidebar.button("🔄 Clear cache & refresh data", use_container_width=True,
                     help="Discards cached prices/fundamentals so the next analysis "
                          "re-downloads fresh data from Yahoo Finance."):
    st.cache_data.clear()
    st.toast("Cache cleared — data will re-download on the next run.")
    st.rerun()

cfg = dict(CONFIG)

cfg["benchmark"] = st.sidebar.text_input("Benchmark (Relative Strength)", CONFIG["benchmark"])
CONFIG["benchmark"] = cfg["benchmark"]
cfg["box_method"] = st.sidebar.selectbox(
    "Box / resistance method", ["Pivot High", "Darvas Box"],
    help="Pivot High (default) = swing-high resistance. Darvas Box = the box top "
         "(ceiling) is the breakout trigger and the box floor is a structural stop. "
         "Switch and re-run the backtest to compare which works better for a stock.")
cfg["near_pct"] = st.sidebar.slider("Near-resistance threshold (%)", 1.0, 10.0, CONFIG["near_pct"], 0.5)
cfg["vol_surge_mult"] = st.sidebar.slider("Breakout volume surge (×avg)", 1.0, 3.0, CONFIG["vol_surge_mult"], 0.1)
c_lo, c_hi = st.sidebar.slider("Healthy RSI band", 0, 100, (CONFIG["rsi_low"], CONFIG["rsi_high"]))
cfg["rsi_low"], cfg["rsi_high"] = c_lo, c_hi

with st.sidebar.expander("Scoring weights (sum should = 100)"):
    for key, label in [
        ("w_trend", "Trend"), ("w_near", "Near Resistance"),
        ("w_compression", "Compression"), ("w_volume_dry", "Volume Dry-up (in base)"),
        ("w_obv", "Volume Accumulation (OBV)"),
        ("w_higher_lows", "Higher Lows"), ("w_atr", "ATR Compression"),
        ("w_touches", "Resistance Touches"), ("w_rsi", "RSI Healthy"),
        ("w_rel_strength", "Relative Strength"),
    ]:
        cfg[key] = st.number_input(label, 0, 50, CONFIG[key], 5, key=f"wt_{key}",
                                   help=WEIGHT_HELP.get(key, ""))
    total_w = sum(cfg[k] for k in cfg if k.startswith("w_"))
    st.caption(f"Total weight: **{total_w}**")

with st.sidebar.expander("Backtest settings"):
    cfg["bt_threshold"] = st.slider("Score needed to take a trade", 50, 100,
                                    CONFIG["bt_threshold"], 5)
    cfg["bt_horizon"] = st.slider("Bars to resolve a trade", 5, 60,
                                  CONFIG["bt_horizon"], 5)
    cfg["bt_lookback"] = st.slider("History to test (bars)", 250, 2000,
                                   CONFIG["bt_lookback"], 250)

with st.sidebar.expander("Long-base settings"):
    cfg["lb_lookback"] = st.slider("Lookback (bars)", 250, 2500, CONFIG["lb_lookback"], 250,
                                   help="How far back to search for a repeatedly-tested "
                                        "ceiling. ~250 bars ≈ 1 year.")
    cfg["lb_tol"] = st.slider("Touch tolerance (±%)", 1.0, 6.0,
                              CONFIG["lb_tol"] * 100, 0.5) / 100
    cfg["lb_min_touches"] = st.slider("Min touches", 2, 8, CONFIG["lb_min_touches"], 1)
    cfg["lb_recent_bars"] = st.slider("Recent-touch window (bars)", 40, 378,
                                      CONFIG["lb_recent_bars"], 20,
                                      help="A level must have a touch within this many recent "
                                           "bars to count as an ACTIVE ceiling (else it's stale "
                                           "and ignored). ~126 bars ≈ 6 months.")
    cfg["lb_cross_bars"] = st.slider("Fresh-cross window (bars)", 5, 90,
                                     CONFIG["lb_cross_bars"], 5,
                                     help="A breakout only counts if price was BELOW the level "
                                          "within this many bars — a fresh cross, not one from "
                                          "long ago.")
    cfg["lb_stop_lowbars"] = st.slider("Stop: recent-low window (bars)", 20, 120,
                                       CONFIG["lb_stop_lowbars"], 10,
                                       help="Recent swing-low window for the level-based stop.")
    cfg["lb_stop_atr"] = st.slider("Stop: ATR multiple", 1.0, 3.0,
                                   CONFIG["lb_stop_atr"], 0.5,
                                   help="ATR stop = level − N×ATR. The plan uses the tighter of "
                                        "this and the recent swing low.")

_wl = get_watchlist()
with st.sidebar.expander(f"📋 Watchlist ({len(_wl)})"):
    st.write(", ".join(_wl) if _wl else "_Empty — add stocks from the analysis view._")
    st.caption("Session-only: kept for this browser session.")
    new_t = st.text_input("Add ticker", key="wl_add", placeholder="e.g. TCS or TCS.NS")
    if st.button("Add", key="wl_add_btn"):
        t = normalize_ticker(new_t)
        if t and t not in _wl:
            set_watchlist(_wl + [t])
            st.rerun()
    if _wl:
        rem = st.multiselect("Remove", _wl, key="wl_rem")
        if st.button("Remove selected", key="wl_rem_btn") and rem:
            set_watchlist([x for x in _wl if x not in rem])
            st.rerun()


# =====================================================================
# MAIN UI
# =====================================================================
st.title("📈 Siva's Darvas Pivot Breakout")

mode = st.radio("Mode", ["Single ticker (detailed)", "Scan multiple"], horizontal=True)


def render_checklist(checklist):
    st.write("### 📊 Setup Checklist")
    for name, item in checklist.items():
        icon = "✅" if item["passed"] else "❌"
        with st.expander(f"{icon}  {name}  —  {item['points']}/{item['max']} pts", expanded=False):
            st.write(f"**What we check:** {item['method']}")
            st.write(f"**Current reading:** {item['detail']}")
            st.caption(f"💡 {item['explanation']}")


def render_regime(cfg):
    reg = market_regime(cfg["benchmark"])
    if reg is None:
        st.info(f"Market regime: benchmark **{cfg['benchmark']}** data unavailable.")
        return
    if reg["risk_on"]:
        st.success(f"🟢 Market regime: **RISK-ON** — {cfg['benchmark']} is "
                   f"{reg['pct']:+.1f}% above its 200-day EMA. Breakouts have the "
                   f"market tailwind.")
    else:
        st.error(f"🔴 Market regime: **RISK-OFF** — {cfg['benchmark']} is "
                 f"{reg['pct']:+.1f}% vs its 200-day EMA (below it). Most breakouts "
                 f"fail in a weak market — trade smaller or wait.")


def plain_summary(ticker, res, cfg):
    """One-sentence, plain-English read of the setup."""
    trend = res["checklist"].get("Trend", {}).get("passed", False)
    dp = res["dist_pct"]
    where = (f"{abs(dp):.1f}% below resistance" if dp > 0
             else f"{abs(dp):.1f}% above resistance (already broken out)")
    trend_txt = "in a clean uptrend" if trend else "not yet in a clean uptrend"
    vol_need = res["avg_vol"] * cfg["vol_surge_mult"]
    s = (f"**{ticker}** is {trend_txt}, consolidating in a **{res['base_len']}-bar base** "
         f"(~{res['base_depth']:.0f}% deep), currently **{where}** at "
         f"**{res['resistance']:.2f}**. ")
    if res["confirmed"]:
        s += (f"✅ **Breakout confirmed today** on "
              f"{res['last_vol'] / res['avg_vol']:.1f}× average volume. ")
    elif res["above"]:
        s += (f"Price is above resistance but volume hasn't confirmed — it needs "
              f"> **{vol_need:,.0f}** ({cfg['vol_surge_mult']}× avg). ")
    else:
        s += (f"A valid breakout needs a close above **{res['entry']:.2f}** on volume "
              f"> **{vol_need:,.0f}** ({cfg['vol_surge_mult']}× the 20-day average). ")
    s += f"Score **{res['score']}/100** → {res['signal']}."
    return s


def score_gauge(score):
    color = "#2ecc71" if score >= 80 else "#f1c40f" if score >= 65 else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 26}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 65], "color": "rgba(231,76,60,0.20)"},
                {"range": [65, 80], "color": "rgba(241,196,15,0.25)"},
                {"range": [80, 100], "color": "rgba(46,204,113,0.25)"},
            ],
        },
        title={"text": "Setup Score"},
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_trade_plan(res, cfg):
    entry = res["entry"]
    stop_pts = res["stop"] - entry
    tgt_pts = res["target"] - entry
    stop_pct = stop_pts / entry * 100
    tgt_pct = tgt_pts / entry * 100

    entry_help = ("Entry = resistance × 1.002 — i.e. just 0.2% above the breakout "
                  "level. You only enter once price actually clears resistance, so a "
                  "stock sitting below it hasn't triggered yet.")
    stop_help = ("Stop = Entry − 1.5 × ATR(14). ATR(14) is the average daily range "
                 "(volatility) over 14 days, so the stop sits 1.5 average days' worth "
                 "of movement below entry — wide enough to survive normal noise, tight "
                 "enough to cap the loss. Risk per share = Entry − Stop = 1.5 × ATR.")
    target_help = ("Target = Entry + 2 × Risk, where Risk = Entry − Stop. That's a "
                   "'2R' target — you aim to make twice what you're risking (a 2:1 "
                   "reward-to-risk trade).")

    t1, t2, t3 = st.columns(3)
    t1.metric("Entry", fmt(entry), help=entry_help)
    t2.metric("Stop Loss", fmt(res["stop"]),
              delta=f"{stop_pts:+.2f} pts ({stop_pct:+.2f}%)", delta_color="inverse",
              help=stop_help)
    t3.metric("Target (2R)", fmt(res["target"]),
              delta=f"{tgt_pts:+.2f} pts ({tgt_pct:+.2f}%)", help=target_help)
    st.caption(f"Risk per share: **{res['risk']:.2f}** ({abs(stop_pct):.2f}% down)  |  "
               f"Reward: **{tgt_pts:.2f}** ({tgt_pct:.2f}% up)  |  "
               f"Reward:Risk ≈ **{res['rr']:.1f} : 1**  |  "
               f"Entry triggers on a close above **{entry:.2f}**.")

    # recalc from current price (only if price already past entry)
    if res["price"] > entry:
        atr = res["atr"]
        cur = res["price"]
        adj_stop = cur - 1.5 * atr
        adj_risk = cur - adj_stop
        adj_target = cur + 2 * adj_risk
        chase_pct = (cur - entry) / entry * 100
        a_stop_pct = (adj_stop - cur) / cur * 100
        a_tgt_pct = (adj_target - cur) / cur * 100

        st.warning(f"⚠️ Price (**{cur:.2f}**) is already **{chase_pct:.2f}% above** the "
                   f"ideal entry of {entry:.2f} — the breakout has extended and buying "
                   f"now means chasing. Below is the plan recalculated using the "
                   f"**current price as entry**.")
        st.write("#### 🔁 Plan If You Enter at Current Price")
        r1, r2, r3 = st.columns(3)
        r1.metric("Entry (current)", fmt(cur))
        r2.metric("Stop Loss", fmt(adj_stop),
                  delta=f"{adj_stop - cur:+.2f} pts ({a_stop_pct:+.2f}%)", delta_color="inverse",
                  help="Recalculated: current price − 1.5 × ATR(14).")
        r3.metric("Target (2R)", fmt(adj_target),
                  delta=f"{adj_target - cur:+.2f} pts ({a_tgt_pct:+.2f}%)",
                  help="Recalculated: current price + 2 × (new risk).")
        st.caption(f"New risk per share: **{adj_risk:.2f}** ({abs(a_stop_pct):.2f}% down)  |  "
                   f"Reward: **{adj_target - cur:.2f}** ({a_tgt_pct:.2f}% up)  |  "
                   f"Reward:Risk still **2 : 1** by construction.")
        st.caption("ℹ️ The 2:1 ratio holds, but your stop is now **higher** — check it "
                   "still sits below a logical support (ideally the breakout level "
                   f"~{entry:.2f}). Buying extended raises the odds of a near-term "
                   "pullback to that level, so waiting for a retest is often the "
                   "lower-risk choice.")

    # volume confirmation
    vol_threshold = res["avg_vol"] * cfg["vol_surge_mult"]
    vol_ratio = res["last_vol"] / res["avg_vol"] if res["avg_vol"] else 0
    st.write("#### 🔊 Volume Confirmation")
    v1, v2, v3 = st.columns(3)
    v1.metric("20-day Avg Vol", f"{res['avg_vol']:,.0f}")
    v2.metric("Required (breakout)", f"{vol_threshold:,.0f}",
              help=f"{cfg['vol_surge_mult']}× the 20-day average volume")
    v3.metric("Today's Vol", f"{res['last_vol']:,.0f}", delta=f"{vol_ratio:.1f}× avg")
    if res["vol_ok"]:
        st.success(f"✅ Volume confirms: today's {res['last_vol']:,.0f} is "
                   f"{vol_ratio:.1f}× the average — take the breakout if price closes above entry.")
    else:
        st.warning(f"⚠️ Volume NOT confirmed yet: need ≥ {vol_threshold:,.0f} "
                   f"({cfg['vol_surge_mult']}× avg). A price break on weak volume often fails — "
                   f"wait for a high-volume close above {entry:.2f}.")

    if res["score"] < 65:
        st.warning("Score below 65 — setup not yet high quality. Levels shown for planning only.")


def longbase_badge(lb):
    """(kind, message) for the long-base banner, or None if not noteworthy."""
    if not lb or lb.get("state") in (None, "none"):
        return None
    yrs = lb["span_bars"] / 252
    t, r, d = lb["touches"], lb.get("vol_ratio", 0), lb["dist_pct"]
    s = lb["state"]
    if s == "multiyear_breakout":
        return ("success", f"🏆 **Multi-Year Base Breakout** — just cleared a ceiling tested "
                f"**{t}× over ~{yrs:.1f} years** on **{r:.1f}× volume**. The bigger the base, "
                f"the bigger the potential move — high-quality setup.")
    if s == "long_breakout":
        return ("success", f"🏅 **Long-Base Breakout** — cleared a level tested {t}× over "
                f"~{yrs:.1f} years on {r:.1f}× volume.")
    if s == "breakout":
        return ("info", f"✅ **Base breakout** — cleared a level tested {t}× (base ~{yrs:.1f} yrs).")
    if s == "testing":
        return ("info", f"👀 **Testing a long-tested ceiling** — {t}× touches over ~{yrs:.1f} years, "
                f"currently **{abs(d):.1f}% below**. Watch for a decisive high-volume break.")
    return None


def level_trade_plan(df, lb, cfg):
    """
    A trade plan anchored purely to the tested level + the stock's own structure:
      Entry  = level * 1.002 (a decisive close above the ceiling)
      Stop   = tighter of (level - N*ATR) and the recent swing low (kept below the level)
      Target = nearest overhead pivot high(s) = "where sellers showed up before";
               if none exist (all-time highs) -> measured-move projection.
    """
    level = lb["level"]
    atr = float(df["ATR14"].iloc[-1])
    cur = float(df["Close"].iloc[-1])
    lowbars = cfg.get("lb_stop_lowbars", 60)
    atr_mult = cfg.get("lb_stop_atr", 1.5)

    entry = level * 1.002

    # ---- stop: tighter (higher) of ATR-stop and recent swing low, kept below the level ----
    stop, stop_rule = level - atr_mult * atr, f"{atr_mult:g}×ATR"
    recent_low = float(df["Low"].tail(lowbars).min())
    if recent_low < level:
        swing_stop = recent_low * 0.997
        if swing_stop > stop:                       # tighter wins
            stop, stop_rule = swing_stop, f"recent {lowbars}-bar low"
    stop = min(stop, level * 0.995)                 # ensure below the level
    risk = entry - stop

    # ---- targets: nearest overhead pivot highs above the entry/current price ----
    lookback = cfg.get("lb_lookback", CONFIG["lb_lookback"])
    win = cfg.get("lb_window", 5)
    tol = cfg.get("lb_tol", 0.03)
    h = df.tail(lookback)["High"].values
    ref = max(entry, cur)
    piv = sorted({float(h[i]) for i in range(win, len(h) - win)
                  if h[i] >= h[i - win:i].max() and h[i] >= h[i + 1:i + win + 1].max()
                  and h[i] > ref * (1 + tol)})     # meaningfully above (not the same zone)
    zones = []                                       # collapse nearby pivots into one zone
    for p in piv:
        if not zones or p > zones[-1] * (1 + tol):
            zones.append(p)

    fallback = False
    if zones:
        t1 = zones[0]
        t2 = zones[1] if len(zones) > 1 else None
    else:                                            # blue-sky: measured move off recent range
        recent = df.tail(cfg.get("lb_recent_bars", 126))
        rng = float(recent["High"].max() - recent["Low"].min())
        t1 = entry + rng if rng > 0 else entry + 3 * risk
        t2 = None
        fallback = True

    rr = (t1 - entry) / risk if risk > 0 else 0
    rr_now = (t1 - cur) / (cur - stop) if (cur - stop) > 0 else 0
    return {"entry": entry, "stop": stop, "stop_rule": stop_rule, "risk": risk,
            "t1": t1, "t2": t2, "rr": rr, "rr_now": rr_now, "fallback": fallback, "cur": cur}


def render_long_base(res, cfg):
    lb = res.get("long_base")
    st.write("### 🏛 Long-Tested Base / Multi-Year Resistance")
    st.caption("A horizontal ceiling that price has tested repeatedly over a long span. A "
               "decisive, high-volume break of such a level is a higher-quality breakout than "
               "a fresh 52-week-high tick — more overhead supply is cleared, and *the bigger "
               "the base, the bigger the potential move.*")
    if not lb:
        st.info(f"No **active** long-tested ceiling found (need ≥ {cfg.get('lb_min_touches', 3)} "
                f"touches within ±{cfg.get('lb_tol', 0.03) * 100:.0f}%, including at least one in "
                "the recent window). Old levels the stock has already left far behind are "
                "deliberately ignored. The stock may be trending freely, mid-range, or lack a "
                "clear multi-touch level. Adjust tolerance/lookback/recent-window in the sidebar.")
        return

    yrs = lb["span_bars"] / 252
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tested level", fmt(lb["level"]))
    c2.metric("Touches", lb["touches"],
              help="Times price tested this ceiling (swing highs within the tolerance) over the lookback.")
    c3.metric("Base duration", f"{yrs:.1f} yrs",
              help=f"{lb['span_bars']} bars since the first touch on {lb['first_date']:%b %Y}.")
    c4.metric("Distance", f"{lb['dist_pct']:+.2f}%",
              help="How far current price is below (+) or above (−) the tested level.")

    badge = longbase_badge(lb)
    if badge:
        (st.success if badge[0] == "success" else st.info)(badge[1])
    elif lb["broke"]:
        st.info("Price is above the level, but volume hasn't confirmed a decisive break yet.")
    else:
        st.caption(f"Not near a breakout: {abs(lb['dist_pct']):.1f}% from the level, "
                   f"{lb['touches']}× touches over ~{yrs:.1f} yrs.")

    # ----- extended vs at-the-level entry flag -----
    d = lb["dist_pct"]                       # + = price below level, − = price above
    band = cfg.get("near_pct", 3.0)
    level = lb["level"]
    if d <= -band:                           # price is well ABOVE the level → extended
        st.warning(f"⏳ **Extended — don't chase.** Price is **{-d:.1f}% above** the base "
                   f"({level:.2f}). The low-risk entry was *at* the level; consider waiting for "
                   f"a **pullback/retest toward ~{level:.2f}** (old resistance → new support) "
                   f"rather than buying here.")
    elif -band < d < 0:                      # just above the level
        st.success(f"🎯 **At the breakout zone** — price is just {-d:.1f}% above the base "
                   f"({level:.2f}). Near an ideal entry *if it holds above the level*.")
    elif 0 <= d <= band:                     # at / just below the level
        st.info(f"👀 **At the level** — price is {d:.1f}% below {level:.2f}. A decisive, "
                f"high-volume close **above** it would trigger the breakout.")
    else:                                    # far below
        st.caption(f"Price is {d:.1f}% below the tested level — not near a breakout yet.")

    st.caption(f"First tested: **{lb['first_date']:%d %b %Y}** · most recent touch: "
               f"**{lb['last_touch_date']:%d %b %Y}**.")

    # ----- level-based trade plan (structural: level + own history) -----
    plan = level_trade_plan(res["df"], lb, cfg)
    st.write("#### 🎯 Level-Based Trade Plan")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Entry (trigger)", fmt(plan["entry"]),
              help="Tested level × 1.002 — a decisive close just above the ceiling.")
    q2.metric("Stop", fmt(plan["stop"]),
              delta=f"{(plan['stop']/plan['entry'] - 1) * 100:+.1f}%", delta_color="inverse",
              help=f"Tighter of {cfg.get('lb_stop_atr', 1.5):g}×ATR and the recent swing low, "
                   f"kept below the level. Rule used: {plan['stop_rule']}.")
    q3.metric("Target 1", fmt(plan["t1"]),
              delta=f"{(plan['t1']/plan['entry'] - 1) * 100:+.1f}%",
              help="Nearest overhead resistance (prior swing high) — the first place sellers "
                   "showed up. A measured-move projection is used if there's no overhead "
                   "resistance (all-time highs).")
    q4.metric("Reward : Risk", f"{plan['rr']:.1f} : 1")

    bits = [f"Risk/share **{plan['risk']:.2f}**", f"Stop rule: **{plan['stop_rule']}**"]
    if plan["t2"]:
        bits.append(f"T2 **{fmt(plan['t2'])}** ({(plan['t2']/plan['entry'] - 1) * 100:+.1f}%)")
    if plan["fallback"]:
        bits.append("_T1 is a measured-move projection (blue-sky — no overhead resistance)_")
    st.caption(" · ".join(bits))
    if plan["cur"] > plan["entry"]:
        st.caption(f"⏳ From the current price ({fmt(plan['cur'])}) the reward:risk is only "
                   f"**{plan['rr_now']:.1f} : 1** — you'd be chasing. A pullback/retest toward "
                   f"**{fmt(plan['entry'])}** restores the ~{plan['rr']:.1f}:1.")
    elif plan["cur"] < plan["entry"]:
        st.caption(f"ℹ️ Price ({fmt(plan['cur'])}) is still **below** the trigger — this is the "
                   f"plan *if/when* it breaks **{fmt(plan['entry'])}**, not a live entry yet.")

    # ----- chart: price + tested level + touch markers, with a volume panel -----
    df = res["df"].tail(cfg.get("lb_lookback", CONFIG["lb_lookback"]))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.72, 0.28])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#1f77b4", width=1)), row=1, col=1)
    fig.add_hline(y=level, line_dash="dash", line_color="#d62728",
                  annotation_text=f"Tested level {level:.2f}", row=1, col=1)
    fig.add_trace(go.Scatter(x=lb["touch_dates"], y=lb["touch_prices"], mode="markers",
                             name="Touches",
                             marker=dict(color="#d62728", size=10, symbol="circle-open",
                                         line=dict(width=2))), row=1, col=1)
    # level-based trade-plan lines
    for y, color, label in [
        (plan["entry"], "#1f77b4", "Entry"),
        (plan["stop"], "#e74c3c", "Stop"),
        (plan["t1"], "#2ca02c", "Target 1"),
    ]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=label, row=1, col=1)
    if plan["t2"]:
        fig.add_hline(y=plan["t2"], line_dash="dot", line_color="#2ca02c",
                      annotation_text="Target 2", row=1, col=1)
    # volume panel — green up-days / red down-days + 20-day average
    up = df["Close"] >= df["Open"]
    bar_colors = ["#26a69a" if u else "#ef5350" for u in up]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=bar_colors,
                         name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume"].rolling(20).mean(),
                             name="20-day avg vol", line=dict(color="black", width=1.2)),
                  row=2, col=1)
    fig.update_layout(height=560, title="Long-tested ceiling, touches & volume",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def render_single(ticker, res, cfg):
    info = load_info(ticker)
    stats = build_stats(res["df"], info)

    # ----- method indicator -----
    method = res.get("box_method", "Pivot High")
    chip = "🟦 Darvas Box" if method == "Darvas Box" else "🟩 Pivot High"
    st.markdown(f"🧭 **Method:** {chip} — resistance & stop are derived from the "
                f"**{method}** logic (change it in the sidebar).")

    # ----- market regime + plain-language summary -----
    render_regime(cfg)
    st.info(plain_summary(ticker, res, cfg))

    # ----- score gauge + key metrics -----
    resistance_help = (
        "Resistance = the most significant recent swing-high 'ceiling'. We scan for "
        "pivot highs — bars whose High is greater than the 3 bars on each side — across "
        "the lookback window, then take the highest of the last 5 such pivots. If no "
        "pivot is found, we fall back to the highest High of the last 120 bars.")
    gcol, mcol = st.columns([1, 2])
    with gcol:
        st.plotly_chart(score_gauge(res["score"]), use_container_width=True)
    with mcol:
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", fmt(res["price"]))
        m2.metric("Resistance", fmt(res["resistance"]), help=resistance_help)
        m3.metric("ATR(14)", fmt(res["atr"]))
        last_dt = res["df"].index[-1]
        quote_ts = ""
        mkt = info.get("regularMarketTime")
        if mkt:
            try:
                tz = info.get("exchangeTimezoneName") or "UTC"
                qt = pd.to_datetime(mkt, unit="s", utc=True).tz_convert(tz)
                quote_ts = f" · live quote {qt:%d %b %Y %H:%M} {qt.tzname()}"
            except Exception:
                quote_ts = ""
        st.caption(f"🕒 Price & volume as of **{last_dt:%d %b %Y}** (latest daily bar){quote_ts}.")
        st.subheader(res["signal"])

    # ----- retained breakout confirmation banner -----
    if res["confirmed"]:
        st.success(f"🚀 BREAKOUT CONFIRMED TODAY — close above resistance on "
                   f"{res['last_vol']/res['avg_vol']:.1f}× average volume.")
    elif res["above"]:
        st.info("Price is above resistance but volume hasn't confirmed yet.")

    # ----- multi-year base breakout badge (prominent) -----
    _badge = longbase_badge(res.get("long_base"))
    if _badge:
        (st.success if _badge[0] == "success" else st.info)(_badge[1])

    # ----- tabs -----
    tab_o, tab_c, tab_t, tab_ch, tab_lb, tab_b = st.tabs(
        ["📋 Overview", "📊 Checklist", "💰 Trade Plan", "📈 Chart",
         "🏛 Long Base", "🔁 Backtest"])

    with tab_o:
        st.write("#### 🧾 Current Stats")
        s1, s2, s3 = st.columns(3)
        groups = [
            ["Current Price", "Day High", "Day Low", "52-Week High", "52-Week Low",
             "All-Time High", "All-Time Low"],
            ["ADX (14)", "Market Cap", "Trailing P/E", "Forward P/E", "Industry P/E"],
            ["Sector", "Industry", "Promoter/Insider Holding", "Institutional Holding"],
        ]
        for col, keys in zip((s1, s2, s3), groups):
            for k in keys:
                col.write(f"**{k}:** {stats[k]}")
        st.caption("ℹ️ ADX measures trend **strength**, not direction: ≥25 = strong trend, "
                   "20–25 = building, <20 = weak/choppy (breakouts whipsaw more in chop). "
                   "+DI vs −DI gives the bullish/bearish bias.")
        if stats["Industry P/E"].startswith("N/A"):
            st.caption("ℹ️ Industry/sector average P/E is not available from Yahoo Finance's "
                       "free API — compare the stock's Trailing P/E against peers manually.")

        st.write("#### 🎯 Distance to Resistance")
        d1, d2 = st.columns(2)
        d1.metric("Distance (points)", f"{res['dist_pts']:+.2f}")
        d2.metric("Distance (%)", f"{res['dist_pct']:+.2f}%")

        st.write("#### 🧱 The Base")
        b1, b2, b3 = st.columns(3)
        b1.metric("Base length", f"{res['base_len']} bars",
                  help="How many trading days the stock has been consolidating under "
                       "resistance. Longer, tighter bases (often 5+ weeks) tend to launch "
                       "stronger, more durable breakouts than short, loose ones.")
        b2.metric("Base depth", f"{res['base_depth']:.1f}%",
                  help="How far price fell from resistance to the base low, as a %. "
                       "Shallow bases (roughly <15–20%) are healthier — a very deep base "
                       "signals heavy selling and a weaker structure.")
        b3.metric("Base low", fmt(res["base_low"]),
                  help="The lowest price within the base — the floor of the consolidation.")
        st.caption("A *base* is the sideways consolidation under resistance where the "
                   "stock pauses and builds energy. The breakout is the move *out* of "
                   "this base; a tight, shallow, mature base is the launch-pad you want.")

    with tab_c:
        render_checklist(res["checklist"])

    with tab_t:
        render_trade_plan(res, cfg)

    with tab_ch:
        render_chart(ticker, res)

    with tab_lb:
        render_long_base(res, cfg)

    with tab_b:
        render_backtest(ticker, res, cfg)


def render_backtest(ticker, res, cfg):
    st.write("### 🔁 Backtest — does this score actually work?")
    st.caption(f"Replays history using the **{cfg.get('box_method', 'Pivot High')}** method: "
               f"every time the score reached **≥ {cfg['bt_threshold']}** with a "
               f"volume-confirmed breakout, it simulates the same Entry/Stop/Target and "
               f"checks whether Target (+2R) or Stop (−1R) hit first within "
               f"{cfg['bt_horizon']} bars. Switch the method in the sidebar and re-run to compare.")
    if st.button("Run backtest"):
        st.session_state["run_bt"] = True
    if not st.session_state.get("run_bt"):
        return

    with st.spinner("Backtesting…"):
        bt = backtest(res["df"], cfg)

    if bt.get("trades", 0) == 0:
        st.info("No historical signals at this threshold — try lowering the backtest "
                "score threshold in the sidebar, or widening the history window.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trades", bt["trades"])
    m2.metric("Win rate", f"{bt['win_rate']:.0f}%", help=f"{bt['wins']}W / {bt['losses']}L")
    m3.metric("Expectancy", f"{bt['expectancy']:+.2f}R",
              help="Average R-multiple per trade. Positive = the setup made money "
                   "historically (1R = the amount risked).")
    m4.metric("Avg hold", f"{bt['avg_hold']:.0f} bars")

    if bt["expectancy"] > 0:
        st.success(f"✅ Positive edge in this sample: {bt['win_rate']:.0f}% win rate, "
                   f"{bt['expectancy']:+.2f}R per trade over {bt['trades']} trades.")
    else:
        st.warning(f"⚠️ Negative/flat edge in this sample ({bt['expectancy']:+.2f}R). "
                   f"The score didn't reliably predict breakouts for this stock.")
    st.caption("⚠️ Indicative only: single stock, no costs/slippage, stop checked before "
               "target on the same bar, and the Relative-Strength input carries minor "
               "look-ahead. Use it to compare settings, not as a guarantee.")


def render_chart(ticker, res):
    df = res["df"].tail(CONFIG["chart_days"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.74, 0.26])

    # ---- price panel (row 1) ----
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    for col, color in [("EMA20", "blue"), ("EMA50", "orange"), ("EMA200", "red")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                 line=dict(color=color, width=1)), row=1, col=1)
    for y, dash, color, label in [
        (res["resistance"], "dash", "green", "Resistance"),
        (res["entry"], "dot", "blue", "Entry"),
        (res["stop"], "dot", "red", "Stop Loss"),
        (res["target"], "dot", "purple", "Target"),
    ]:
        fig.add_hline(y=y, line_dash=dash, line_color=color,
                      annotation_text=label, row=1, col=1)

    # ---- volume panel (row 2) ----
    up = df["Close"] >= df["Open"]
    bar_colors = ["#26a69a" if u else "#ef5350" for u in up]   # green up / red down
    avg20 = df["Volume"].rolling(20).mean()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=bar_colors,
                         name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=avg20, name="20-day avg vol",
                             line=dict(color="black", width=1.2)), row=2, col=1)

    # ---- Darvas box (ceiling / floor) drawn as a shaded rectangle ----
    box = res.get("box")
    if box:
        x0 = max(pd.Timestamp(box["top_date"]), df.index[0]).isoformat()
        x1 = df.index[-1].isoformat()
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=box["bottom"], y1=box["top"],
                      line=dict(color="rgba(41,98,255,0.7)", width=1.2),
                      fillcolor="rgba(41,98,255,0.08)", row=1, col=1)
        fig.add_annotation(x=x1, y=box["top"], text=f"Ceiling {box['top']:.2f}",
                           showarrow=False, yshift=10, font=dict(color="#2962ff", size=11),
                           row=1, col=1)
        fig.add_annotation(x=x1, y=box["bottom"], text=f"Floor {box['bottom']:.2f}",
                           showarrow=False, yshift=-10, font=dict(color="#2962ff", size=11),
                           row=1, col=1)

    # highlight the breakout bar if confirmed today
    if res["confirmed"]:
        bx = df.index[-1].isoformat()       # ISO string avoids a plotly Timestamp bug
        fig.add_vline(x=bx, line_dash="dot", line_color="green", row=1, col=1)
        fig.add_annotation(x=bx, y=res["resistance"], text="Breakout", showarrow=False,
                           yshift=10, font=dict(color="green", size=11), row=1, col=1)

    title_suffix = "Darvas Box" if box else "Pivot High"
    fig.update_layout(height=760, xaxis_rangeslider_visible=False,
                      title=f"{ticker} Breakout Scanner ({title_suffix})",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ---------- SINGLE MODE ----------
if mode == "Single ticker (detailed)":
    wl = get_watchlist()
    pick = st.selectbox("Pick from watchlist", ["— type below —"] + wl)
    typed = st.text_input("…or enter a Yahoo Finance ticker", value="RELIANCE.NS",
                          help="Bare NSE symbols auto-get '.NS' (e.g. TCS → TCS.NS). "
                               "Use a suffix for other exchanges (e.g. .BO for BSE) "
                               "or ^ for indices (e.g. ^NSEI).")
    ticker = normalize_ticker(typed) if pick == "— type below —" else pick
    if pick == "— type below —" and ticker and ticker != typed.strip().upper():
        st.caption(f"🔎 Resolved to **{ticker}** (NSE).")

    ca, cb = st.columns(2)
    if ca.button("Analyze"):
        st.session_state["analyzed"] = True
        st.session_state["ticker"] = ticker
        st.session_state["run_bt"] = False          # reset backtest on new analysis
    if cb.button("➕ Add to watchlist"):
        t = normalize_ticker(ticker)
        if t and t not in wl:
            set_watchlist(wl + [t])
            st.success(f"Added **{t}** to your watchlist.")
        else:
            st.info(f"**{t}** is already in your watchlist.")

    if st.session_state.get("analyzed"):
        tk = st.session_state.get("ticker", ticker)
        with st.spinner(f"Analysing {tk}…"):
            res = analyze(tk, cfg)
        if "error" in res:
            st.error(res["error"])
        else:
            render_single(tk, res, cfg)

# ---------- SCAN MODE ----------
else:
    wl = get_watchlist()
    use_wl = st.checkbox(f"Scan my watchlist ({len(wl)} stocks)",
                         value=bool(wl), disabled=not wl)
    raw = st.text_area(
        "Enter tickers (comma or newline separated)",
        value="RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS",
        height=100, disabled=use_wl,
    )
    if st.button("Scan"):
        tickers = wl if use_wl else [
            normalize_ticker(t) for t in raw.replace("\n", ",").split(",") if t.strip()]
        if not tickers:
            st.warning("No tickers to scan — add some to the box or your watchlist.")
            st.stop()
        rows, progress = [], st.progress(0.0)
        for i, tk in enumerate(tickers, 1):
            res = analyze(tk, cfg)
            if "error" in res:
                rows.append({"Ticker": tk, "Score": None, "Signal": "⚠️ " + res["error"]})
            else:
                entry = res["entry"]
                stop_pct = (res["stop"] - entry) / entry * 100
                tgt_pct = (res["target"] - entry) / entry * 100
                vol_ratio = res["last_vol"] / res["avg_vol"] if res["avg_vol"] else None

                lb = res.get("long_base")
                if lb:
                    my_dist = round(lb["dist_pct"], 2)
                    flag = {"multiyear_breakout": "🏆", "long_breakout": "🏅",
                            "breakout": "✅", "testing": "👀"}.get(lb.get("state"), "")
                    my_base = f"{flag} {lb['touches']}×" if flag else f"{lb['touches']}×"
                else:
                    my_dist, my_base = None, "—"

                rows.append({
                    "Ticker": tk,
                    "Score": res["score"],
                    "Signal": res["signal"],
                    "Breakout?": "🚀" if res["confirmed"] else "",
                    "MY Base": my_base,
                    "MY Dist %": my_dist,
                    "Vol ×avg": round(vol_ratio, 2) if vol_ratio is not None else None,
                    "Today Vol": round(res["last_vol"]),
                    "20d Avg Vol": round(res["avg_vol"]),
                    "Price": round(res["price"], 2),
                    "Resistance": round(res["resistance"], 2),
                    "Dist %": round(res["dist_pct"], 2),
                    "Dist pts": round(res["dist_pts"], 2),
                    "Entry": round(entry, 2),
                    "Stop": round(res["stop"], 2),
                    "Stop %": round(stop_pct, 2),
                    "Target": round(res["target"], 2),
                    "Target %": round(tgt_pct, 2),
                    "Base len": res["base_len"],
                    "Base depth %": round(res["base_depth"], 1),
                })
            progress.progress(i / len(tickers))
        progress.empty()

        table = pd.DataFrame(rows)
        # enforce a stable column order (Breakout? before Price; %s beside levels)
        col_order = ["Ticker", "Score", "Signal", "Breakout?", "MY Base", "MY Dist %",
                     "Vol ×avg", "Today Vol", "20d Avg Vol", "Price", "Resistance",
                     "Dist %", "Dist pts", "Entry", "Stop", "Stop %", "Target",
                     "Target %", "Base len", "Base depth %"]
        table = table.reindex(columns=[c for c in col_order if c in table.columns])
        if "Score" in table:
            table = table.sort_values(
                "Score", ascending=False, na_position="last").reset_index(drop=True)
        st.session_state["scan_table"] = table
        st.session_state["scan_detail"] = None      # clear previously opened detail
        st.session_state["scan_method"] = cfg.get("box_method", "Pivot High")

    # ----- persisted results + one-click drill-down -----
    if st.session_state.get("scan_table") is not None:
        table = st.session_state["scan_table"]
        render_regime(cfg)
        used = st.session_state.get("scan_method", "Pivot High")
        chip = "🟦 Darvas Box" if used == "Darvas Box" else "🟩 Pivot High"
        st.write("### 🔍 Scan Results (best setups on top)")
        st.markdown(f"🧭 **Method used for these results:** {chip}")
        if cfg.get("box_method") != used:
            st.warning(f"⚠️ You've switched the method to **{cfg.get('box_method')}** — "
                       f"the table below still reflects **{used}**. Click **Scan** again "
                       f"to recompute with the new method.")
        st.caption("👉 **Click a row** to load the full detailed analysis below.")
        st.caption("**MY Base** = multi-year tested ceiling: 🏆 breakout / 🏅 long-base break / "
                   "👀 testing / `N×` = touch count / `—` none. **MY Dist %** = distance to that "
                   "ceiling (smaller = nearer; negative = already above). Sort by **MY Dist %** to "
                   "find stocks closest to a multi-year breakout.")

        # tidy decimals everywhere; red Stop %, green Target %. Selection still works.
        fmt_map = {}
        for c in ["Price", "Resistance", "Dist pts", "Entry", "Stop", "Target"]:
            if c in table:
                fmt_map[c] = "{:.2f}"
        for c, f in {"Dist %": "{:+.2f}%", "Stop %": "{:+.2f}%",
                     "Target %": "{:+.2f}%", "Base depth %": "{:.1f}%",
                     "MY Dist %": "{:+.2f}%", "Vol ×avg": "{:.1f}×"}.items():
            if c in table:
                fmt_map[c] = f
        for c in ["Score", "Base len", "Today Vol", "20d Avg Vol"]:
            if c in table:
                fmt_map[c] = "{:,.0f}"

        thr = cfg.get("vol_surge_mult", 1.5)

        def _vol_color(v):
            try:
                return "color: #2ecc71" if v >= thr else "color: #e0a800"
            except Exception:
                return ""

        styler = table.style
        if "Stop %" in table:
            styler = styler.set_properties(subset=["Stop %"], **{"color": "#e74c3c"})
        if "Target %" in table:
            styler = styler.set_properties(subset=["Target %"], **{"color": "#2ecc71"})
        if "Vol ×avg" in table:
            styler = styler.map(_vol_color, subset=["Vol ×avg"])
        if fmt_map:
            styler = styler.format(fmt_map, na_rep="—")

        event = st.dataframe(styler, use_container_width=True, hide_index=True,
                             on_select="rerun", selection_mode="single-row", key="scan_df")
        sel = event.selection.rows if (event and event.selection) else []
        if sel:
            sel_ticker = str(table.iloc[sel[0]]["Ticker"])
            if st.session_state.get("scan_detail") != sel_ticker:
                st.session_state["scan_detail"] = sel_ticker
                st.session_state["run_bt"] = False    # fresh backtest per stock
            st.divider()
            st.subheader(f"🔎 Detailed analysis — {sel_ticker}")
            with st.spinner(f"Analysing {sel_ticker}…"):
                dres = analyze(sel_ticker, cfg)
            if "error" in dres:
                st.error(dres["error"])
            else:
                render_single(sel_ticker, dres, cfg)
