# crypto_streamlit_trader.py — ✅ FIXED & ENHANCED VERSION
"""
Streamlit Crypto Trader (mobile-friendly)
- Live Binance spot data via ccxt
- Technical indicators (ta): RSI, EMA20/50, MACD, ATR
- Rule-based + OpenAI-enhanced analysis
- Playable voice via gTTS
- Logs all scans & AI calls
"""

import os
import time
import json
import pathlib
from datetime import datetime
from io import BytesIO

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from gtts import gTTS
from dotenv import load_dotenv
from openai import OpenAI  # ✅ new OpenAI SDK

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crypto Streamlit Trader", layout="wide")
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
DEFAULT_PORTFOLIO = 1000.0
DEFAULT_RISK_PCT = 0.01

load_dotenv()

# ---------------- HELPERS ----------------
def now_iso():
    return datetime.utcnow().isoformat()

def safe_write_json(obj, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def log_csv_row(row: dict, fname="trades_log.csv"):
    import csv
    fpath = LOG_DIR / fname
    exists = fpath.exists()
    with open(fpath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ---------------- MARKET DATA ----------------
@st.cache_data(ttl=30)
def get_binance_spot_client(timeout=20000):
    return ccxt.binance({"enableRateLimit": True, "timeout": timeout})

def top_usdt_pairs(exchange, limit=10):
    try:
        tickers = exchange.fetch_tickers()
        rows = []
        for sym, info in tickers.items():
            if not sym.endswith("/USDT"):
                continue
            vol = info.get("quoteVolume") or info.get("quoteVolume24h") or 0
            try:
                vol = float(vol)
            except:
                vol = 0.0
            rows.append((sym, vol))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in rows[:limit]]
    except Exception as e:
        st.warning(f"⚠️ Binance fetch_tickers failed: {e}")
        return []

def fetch_ohlcv(exchange, symbol, timeframe="5m", limit=200):
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts")
        return df
    except Exception as e:
        st.warning(f"⚠️ Fetch failed for {symbol}: {e}")
        return None

def compute_indicators(df):
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

# ---------------- RULE BASED ----------------
def rule_based_recommendation(df):
    latest = df.iloc[-1]
    rsi = latest["rsi"]
    ema20 = latest["ema20"]
    ema50 = latest["ema50"]
    macd = latest["macd"]
    macd_sig = latest["macd_sig"]
    price = latest["close"]

    score = 0
    reasons = []
    if ema20 > ema50:
        score += 1; reasons.append("EMA20>EMA50")
    else:
        score -= 0.3; reasons.append("EMA20<EMA50")
    if macd > macd_sig:
        score += 0.8; reasons.append("MACD>Signal")
    else:
        score -= 0.3; reasons.append("MACD<Signal")
    if rsi < 35:
        score += 1; reasons.append(f"RSI{rsi:.0f} (oversold)")
    elif rsi > 65:
        score -= 0.8; reasons.append(f"RSI{rsi:.0f} (overbought)")
    else:
        reasons.append(f"RSI{rsi:.0f}")

    if score >= 1.5:
        rec = "STRONG BUY"
    elif 0.6 <= score < 1.5:
        rec = "BUY"
    elif -0.5 <= score < 0.6:
        rec = "HOLD"
    else:
        rec = "SELL"

    return {"recommendation": rec, "score": score, "reasons": reasons, "price": price, "rsi": rsi}

# ---------------- OPENAI INTEGRATION ----------------
SYSTEM_PROMPT = (
    "You are a concise crypto trading assistant. Always respond in valid JSON only with keys: "
    '{"symbol","timestamp","recommendation","entry","stop","target","position_size_usd","confidence","rationale"}. '
    "recommendation must be BUY/SELL/HOLD. If unsure, return HOLD and confidence 0.3."
)

def build_openai_prompt(snapshot, strategy_hint):
    return "\n".join([
        f"Snapshot for {snapshot['symbol']} at {snapshot['ts']}:",
        f"Price: {snapshot['price']}",
        f"RSI: {snapshot['rsi']}",
        f"EMA20: {snapshot['ema20']} | EMA50: {snapshot['ema50']}",
        f"MACD: {snapshot['macd']} | MACD_Signal: {snapshot['macd_sig']}",
        f"ATR: {snapshot['atr']}",
        f"Portfolio: {snapshot['portfolio_usd']}$ Risk: {snapshot['risk_pct']*100:.1f}%",
        f"Strategy hint: {strategy_hint}"
    ])

def call_openai(snapshot, strategy_hint, model="gpt-4o-mini"):
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_KEY")
    if not key:
        return None, "OpenAI key not configured"
    try:
        client = OpenAI(api_key=key)
        user_prompt = build_openai_prompt(snapshot, strategy_hint)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        text = resp.choices[0].message.content.strip()
        try:
            return json.loads(text), None
        except:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1:
                return json.loads(text[s:e+1]), None
            return None, "Invalid JSON"
    except Exception as e:
        return None, str(e)

# ---------------- TTS ----------------
def make_tts_mp3(text: str):
    mp3_fp = BytesIO()
    try:
        gTTS(text=text, lang="en").write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None

# ---------------- STREAMLIT UI ----------------
st.title("📱 Crypto Streamlit Trader — Mobile Ready")
st.caption("Educational use only. Not financial advice.")

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("⚙️ Settings")
    portfolio_usd = st.number_input("Portfolio (USD)", value=DEFAULT_PORTFOLIO)
    risk_pct = st.number_input("Risk per trade", value=DEFAULT_RISK_PCT, step=0.01)
    top_n = st.slider("Top N pairs", 3, 20, 8)
    timeframe = st.selectbox("Candle timeframe", ["1m","3m","5m","15m","1h"], index=2)
    strategy_hint = st.text_input("Strategy hint", value="Buy if RSI<35 and EMA20>EMA50.")
    enable_openai = st.checkbox("Enable OpenAI", value=False)
    if enable_openai:
        api_key_input = st.text_input("🔑 OpenAI API key", type="password")
        if st.button("Save API Key"):
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.session_state["OPENAI_KEY"] = api_key_input
            st.success("✅ Key saved for this session.")
    scan_btn = st.button("🔍 Run Market Scan")

with col2:
    st.subheader("📊 Results")
    results_placeholder = st.empty()
    symbol_input = st.text_input("Chart symbol (e.g., BTC/USDT)")
    tts_button = st.checkbox("Enable TTS voice", value=True)

# ---------------- SCAN ----------------
if scan_btn:
    ex = get_binance_spot_client()
    pairs = top_usdt_pairs(ex, limit=top_n)
    if not pairs:
        st.warning("⚠️ No pairs found, try again.")
    else:
        progress = st.progress(0)
        all_results = []
        for i, sym in enumerate(pairs, start=1):
            df = fetch_ohlcv(ex, sym, timeframe=timeframe)
            time.sleep(0.3)
            if df is None or len(df) < 60:
                progress.progress(i/len(pairs))
                continue
            df_ind = compute_indicators(df)
            if df_ind is None:
                continue
            rule = rule_based_recommendation(df_ind)
            snap = {
                "symbol": sym,
                "ts": df_ind.index[-1].isoformat(),
                "price": float(df_ind["close"].iloc[-1]),
                "rsi": float(df_ind["rsi"].iloc[-1]),
                "ema20": float(df_ind["ema20"].iloc[-1]),
                "ema50": float(df_ind["ema50"].iloc[-1]),
                "macd": float(df_ind["macd"].iloc[-1]),
                "macd_sig": float(df_ind["macd_sig"].iloc[-1]),
                "atr": float(df_ind["atr"].iloc[-1]),
                "portfolio_usd": portfolio_usd,
                "risk_pct": risk_pct,
            }
            ai_out, ai_err = (None, None)
            if enable_openai:
                ai_out, ai_err = call_openai(snap, strategy_hint)
                safe_write_json({"snapshot": snap, "ai_out": ai_out, "ai_err": ai_err}, LOG_DIR / f"ai_{sym.replace('/','_')}_{now_iso()}.json")
            all_results.append({"symbol": sym, "rule": rule, "ai_out": ai_out, "ai_err": ai_err})
            progress.progress(i/len(pairs))
        st.session_state["last_results"] = all_results
        st.success("✅ Scan Complete")

# ---------------- ASK ANYTHING ----------------
st.markdown("---")
st.subheader("💬 Ask AI Assistant")
q = st.text_input("Ask anything (e.g. 'Should I buy BTC today?')")
if st.button("Ask"):
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_KEY")
    if not key:
        st.warning("⚠️ OpenAI key missing.")
    else:
        try:
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise trading assistant."},
                    {"role": "user", "content": q},
                ],
                temperature=0.3,
            )
            ans = resp.choices[0].message.content.strip()
            st.write("🤖 **AI Answer:**", ans)
            if tts_button:
                mp3 = make_tts_mp3(ans)
                if mp3:
                    st.audio(mp3.read(), format="audio/mp3")
        except Exception as e:
            st.error(f"OpenAI failed: {e}")

st.markdown("---")
st.caption("Logs stored in /logs. For research only.")

