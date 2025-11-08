"""
crypto_gui_trader.py
GUI desktop crypto advisor (spot) with OpenAI integration.

Features:
- Fetches top USDT pairs from Binance (spot), computes indicators (RSI/EMA/MACD/ATR)
- Sends compact snapshot to OpenAI for structured JSON recommendation
- GUI: Run scan, view top suggestions, ask questions via chat box
- Safe: NO hardcoded API key. Logs prompts/responses for audit.
- Safety: No automatic order execution. Human confirmation required.

Dependencies: PySimpleGUI, ccxt, pandas, ta, openai, python-dotenv, pyttsx3, requests
"""

import os
import time
import json
import pathlib
import threading
from datetime import datetime

import PySimpleGUI as sg
import ccxt
import pandas as pd
import numpy as np
import ta
import openai
from dotenv import load_dotenv

# --- Config ---
TOP_N = 8
TIMEFRAME = "5m"
CANDLE_LIMIT = 200
PORTFOLIO_USD = float(os.getenv("PORTFOLIO_USD", "1000"))
RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))
LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)
VOICE_ENABLED = True  # GUI includes optional voice button (pyttsx3) — will not break if missing

# Load .env if present
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Optional TTS
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 160)
except Exception:
    tts_engine = None

def speak(text: str):
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass

# ---------------- helpers ----------------
def now_iso():
    return datetime.utcnow().isoformat()

def safe_log_json(obj, fname):
    with open(LOG_DIR / fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def log_csv_row(row, fname):
    import csv
    csv_path = LOG_DIR / fname
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# --------------- market / indicators ---------------
def get_binance_spot(timeout=20000):
    return ccxt.binance({"enableRateLimit": True, "timeout": timeout})

def top_usdt_symbols(exchange, limit=TOP_N):
    try:
        tickers = exchange.fetch_tickers()
        rows = []
        for sym, info in tickers.items():
            if not sym.endswith("/USDT"):
                continue
            vol = info.get("quoteVolume") or 0
            try:
                vol = float(vol)
            except Exception:
                vol = 0.0
            rows.append((sym, vol))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in rows[:limit]]
    except Exception as e:
        print("top_usdt_symbols error:", e)
        return []

def fetch_ohlcv(exchange, symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT):
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts")
        return df
    except Exception as e:
        print(f"fetch_ohlcv failed for {symbol}: {e}")
        return None

def compute_indicators(df):
    if len(df) < 60:
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

def build_snapshot(symbol, df):
    latest = df.iloc[-1]
    return {
        "symbol": symbol,
        "ts": latest.name.isoformat(),
        "price": float(latest["close"]),
        "rsi": float(latest["rsi"]) if not np.isnan(latest["rsi"]) else None,
        "ema20": float(latest["ema20"]) if not np.isnan(latest["ema20"]) else None,
        "ema50": float(latest["ema50"]) if not np.isnan(latest["ema50"]) else None,
        "macd": float(latest["macd"]) if not np.isnan(latest["macd"]) else None,
        "macd_sig": float(latest["macd_sig"]) if not np.isnan(latest["macd_sig"]) else None,
        "atr": float(latest["atr"]) if not np.isnan(latest["atr"]) else None
    }

# --------------- OpenAI prompt & call (structured JSON) ---------------
SYSTEM_PROMPT = (
    "You are a concise trading assistant. ALWAYS respond in valid JSON only, exactly keys:\n"
    '{"symbol","timestamp","recommendation","entry","stop","target","position_size_usd","confidence","rationale"}\n'
    "recommendation: BUY/SELL/HOLD. confidence: 0.0-1.0. Entry/stop/target numeric or null.\n"
    "If unsure, return recommendation 'HOLD' and confidence 0.3. Use only provided snapshot."
)

def build_user_prompt(snapshot, hint="Use RSI<35 + EMA20>EMA50 + MACD>signal to buy. Risk 1% of portfolio."):
    s = snapshot
    lines = [
        f"Snapshot {s['symbol']} at {s['ts']}:",
        f"price: {s['price']}",
        f"rsi: {s['rsi']}",
        f"ema20: {s['ema20']}",
        f"ema50: {s['ema50']}",
        f"macd: {s['macd']}",
        f"macd_signal: {s['macd_sig']}",
        f"atr: {s['atr']}",
        f"portfolio_usd: {PORTFOLIO_USD}",
        f"risk_pct: {RISK_PCT}",
        f"Hint: {hint}",
        "Return valid JSON following system instructions."
    ]
    return "\n".join(lines)

def call_openai(system_prompt, user_prompt):
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not set. Please set OPENAI_API_KEY in environment or load .env.")
    try:
        res = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.0,
            max_tokens=400
        )
        text = res["choices"][0]["message"]["content"].strip()
        # try direct parse
        try:
            return json.loads(text)
        except Exception:
            # try extract {...}
            s = text.find("{"); e = text.rfind("}")
            if s != -1 and e != -1:
                try:
                    return json.loads(text[s:e+1])
                except Exception:
                    return None
            return None
    except Exception as e:
        print("OpenAI call error:", e)
        return None

def clamp_and_validate(out, snapshot):
    # ensure fields present
    res = {k: out.get(k) for k in ["symbol","timestamp","recommendation","entry","stop","target","position_size_usd","confidence","rationale"]}
    # defaults
    res["symbol"] = res.get("symbol") or snapshot["symbol"]
    rec = (res.get("recommendation") or "HOLD").upper()
    if rec not in ("BUY","SELL","HOLD"):
        rec = "HOLD"
    res["recommendation"] = rec
    price = snapshot["price"]
    def n(x):
        try:
            return float(x) if x is not None else None
        except Exception:
            return None
    entry = n(res.get("entry")) or price
    stop = n(res.get("stop"))
    target = n(res.get("target"))
    if rec == "BUY":
        if stop is None or stop >= entry:
            stop = entry * (1 - 0.03)
        if target is None or target <= entry:
            target = entry + max((entry - stop)*1.5, entry*0.01)
    if rec == "SELL":
        if stop is None or stop <= entry:
            stop = entry * (1 + 0.03)
        if target is None or target >= entry:
            target = entry - max((stop - entry)*1.5, entry*0.01)
    if rec == "HOLD":
        stop = None; target = None
    pos = n(res.get("position_size_usd")) or 0.0
    max_allowed = PORTFOLIO_USD * 0.2
    pos = max(0.0, min(pos, max_allowed))
    try:
        conf = float(res.get("confidence"))
    except Exception:
        conf = 0.3
    conf = max(0.0, min(conf, 1.0))
    res.update({"entry": entry, "stop": stop, "target": target, "position_size_usd": pos, "confidence": conf})
    res["rationale"] = res.get("rationale") or ""
    return res

# ----------------- GUI layout -----------------
sg.theme("DarkBlue3")

left_col = [
    [sg.Text("OpenAI API Key (or set in environment)", size=(40,1))],
    [sg.InputText(default_text=os.getenv("OPENAI_API_KEY") or "", key="-APIKEY-", password_char="*", size=(50,1)), 
     sg.Button("Save Key", key="-SAVEKEY-")],
    [sg.HorizontalSeparator()],
    [sg.Frame("Scan Controls", [
        [sg.Button("Scan Top Symbols", key="-SCAN-"), sg.Text("Status:", size=(6,1)), sg.Text("", key="-STATUS-", size=(30,1))],
        [sg.Text("Top N:"), sg.Spin([i for i in range(3,21)], initial_value=TOP_N, key="-TOPN-", size=(5,1)),
         sg.Text("Timeframe:"), sg.Combo(["1m","3m","5m","15m","1h"], default_value=TIMEFRAME, key="-TF-")],
    ])],
    [sg.Frame("Latest Results", [[sg.Multiline("", size=(60,15), key="-RESULTS-", autoscroll=True)]])],
    [sg.Frame("Logs", [[sg.Button("Open logs folder", key="-OPENLOG-"), sg.Button("Clear logs", key="-CLEARLOG-")]])]
]

right_col = [
    [sg.Frame("Ask / Chat", [
        [sg.Multiline("", size=(50,6), key="-CHATINPUT-")],
        [sg.Button("Ask", key="-ASK-"), sg.Button("Speak Answer", key="-SPEAK-")],
        [sg.Text("Answer:" )],
        [sg.Multiline("", size=(50,10), key="-CHATOUT-")]
    ])],
    [sg.Frame("Quick Help", [[sg.Text("Examples: 'Should I buy BTC now?', 'Which coin is best today?', 'Exit'")]])]
]

layout = [
    [sg.Column(left_col), sg.VerticalSeparator(), sg.Column(right_col)]
]

window = sg.Window("Crypto GUI Trader (OpenAI-backed) — No hardcoded API key", layout, finalize=True)

# ----------------- Threaded scan worker -----------------
scan_thread = None
scan_stop_flag = False

def worker_scan(topn, timeframe, window):
    global scan_stop_flag
    try:
        api_key_input = window["-APIKEY-"].get().strip()
        if api_key_input:
            openai.api_key = api_key_input
        exchange = get_binance_spot()
        window["-STATUS-"].update("Fetching symbols...")
        symbols = top_usdt_symbols(exchange, limit=topn)
        if not symbols:
            window["-STATUS-"].update("No symbols; fallback")
            # fallback common list
            symbols = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","ADA/USDT","XRP/USDT","MATIC/USDT","DOGE/USDT"][:topn]
        window["-STATUS-"].update(f"Analyzing {len(symbols)}")
        results_text = []
        results = []
        for s in symbols:
            if scan_stop_flag:
                break
            window["-STATUS-"].update(f"Fetching {s}...")
            df = fetch_ohlcv(exchange, s, timeframe=timeframe, limit=CANDLE_LIMIT)
            time.sleep(0.6)
            if df is None or len(df) < 60:
                results_text.append(f"{s}: insufficient data\n")
                continue
            df_ind = compute_indicators(df)
            snap = build_snapshot(s, df_ind)
            user_prompt = build_user_prompt(snap)
            # log prompt
            ts = now_iso()
            safe_log_json({"ts":ts,"symbol":s,"prompt":user_prompt}, f"prompt_{s.replace('/','_')}_{ts}.json")
            # openai call
            ai_out = None
            try:
                ai_out = call_openai(SYSTEM_PROMPT, user_prompt) if openai.api_key else None
            except Exception as e:
                print("OpenAI call exception:", e)
            if not ai_out:
                results_text.append(f"{s}: model unavailable -> HOLD\n")
                continue
            safe_log_json(ai_out, f"raw_{s.replace('/','_')}_{ts}.json")
            final = clamp_and_validate(ai_out, snap)
            results.append(final)
            line = f"{final['symbol']}: {final['recommendation']} | Entry {final['entry']:.6f} | Stop {final['stop']:.6f} | Target {final['target']:.6f} | Pos ${final['position_size_usd']:.2f} | Conf {final['confidence']:.2f}\n"
            results_text.append(line)
            # csv log row
            row = {"ts":ts,"symbol":final["symbol"],"price":snap["price"],"rec":final["recommendation"],
                   "entry":final["entry"],"stop":final["stop"],"target":final["target"],
                   "pos_usd":final["position_size_usd"],"confidence":final["confidence"],"rationale":final["rationale"]}
            log_csv_row(row, "trades_log.csv")
        # display results
        window["-RESULTS-"].update("".join(results_text))
        window["-STATUS-"].update("Scan complete")
        # store in window metadata for QA
        window.metadata = {"results": results, "symbols": symbols}
        speak("Scan complete. Results updated.")
    except Exception as e:
        window["-STATUS-"].update(f"Error: {e}")
        print("worker_scan error:", e)
        speak(f"Error during scan: {e}")

# ----------------- GUI Event Loop -----------------
while True:
    event, values = window.read(timeout=100)
    if event == sg.WIN_CLOSED:
        break

    if event == "-SAVEKEY-":
        k = values["-APIKEY-"].strip()
        if k:
            # save to .env (safe option) and set openai.key now
            with open(".env","w",encoding="utf-8") as f:
                f.write(f"OPENAI_API_KEY={k}\n")
            openai.api_key = k
            window["-STATUS-"].update("API Key saved to .env")
        else:
            window["-STATUS-"].update("No key entered")

    if event == "-SCAN-":
        if scan_thread and scan_thread.is_alive():
            window["-STATUS-"].update("Scan already running")
        else:
            TOPN = int(values["-TOPN-"])
            TF = values["-TF-"]
            scan_stop_flag = False
            window.metadata = {}
            scan_thread = threading.Thread(target=worker_scan, args=(TOPN, TF, window), daemon=True)
            scan_thread.start()
            window["-STATUS-"].update("Scan started...")

    if event == "-OPENLOG-":
        # open logs folder in file explorer
        try:
            os.startfile(str(LOG_DIR.resolve()))
        except Exception:
            window["-STATUS-"].update("Could not open logs folder")

    if event == "-CLEARLOG-":
        for f in LOG_DIR.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        window["-STATUS-"].update("Logs cleared")

    if event == "-ASK-":
        q = values["-CHATINPUT-"].strip()
        if not q:
            window["-CHATOUT-"].update("Type a question first.")
            continue
        # try to answer using cached results first (fast)
        cached = getattr(window, "metadata", {}).get("results", [])
        qlow = q.lower()
        answered = False
        # symbol-specific?
        if cached:
            for r in cached:
                base = r["symbol"].split("/")[0].lower()
                if base in qlow:
                    s = r
                    out = f"{s['symbol']} → {s['recommendation']}. Price approx {s.get('entry'):.6f}. Confidence {s['confidence']:.2f}. Rationale: {s['rationale']}"
                    window["-CHATOUT-"].update(out)
                    if values["-SPEAK-"]:
                        speak(out)
                    answered = True
                    break
        if answered:
            continue
        # no cached answer — perform live small analysis: ask OpenAI for general guidance
        if not openai.api_key:
            window["-CHATOUT-"].update("OpenAI API key not set. Set in top field or .env")
            continue
        # build a small prompt: ask model to answer directly but short
        user_msg = f"User question: {q}\nYou are a concise trading assistant. Answer briefly. If question references a coin symbol like BTC, ETH, use 'I can check live data' instruction. Do not produce JSON now; reply in short text."
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":"You are a helpful trading assistant."},
                          {"role":"user","content":user_msg}],
                temperature=0.0,
                max_tokens=200
            )
            text = resp["choices"][0]["message"]["content"].strip()
            window["-CHATOUT-"].update(text)
            if values["-SPEAK-"]:
                speak(text)
        except Exception as e:
            window["-CHATOUT-"].update(f"Error calling model: {e}")

    if event == "-SPEAK-":
        # toggle speak button (UI doesn't change visually besides reading)
        pass

window.close()

