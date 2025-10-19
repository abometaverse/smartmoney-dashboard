#!/usr/bin/env python3
import os, json, time, math, requests
import pandas as pd
import numpy as np

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

def http():
    s = requests.Session()
    s.headers.update({"User-Agent": "smartmoney-autoscan/1.0"})
    return s
S = http()

def _first_ok(path, params=None, timeout=10):
    last = None
    for base in BINANCE_ENDPOINTS:
        try:
            r = S.get(f"{base}{path}", params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = r.status_code
        except requests.RequestException:
            last = "req"
        time.sleep(0.15)
    raise RuntimeError(f"binance_fail:{last}")

def exchange_info():
    js = _first_ok("/api/v3/exchangeInfo")
    df = pd.DataFrame(js.get("symbols", []))
    if df.empty: return df
    df = df[(df["quoteAsset"]=="USDT") & (df["status"]=="TRADING")]
    mask = ~df["symbol"].str.contains(r"(?:UP|DOWN|BULL|BEAR)", regex=True)
    df = df[mask]
    return df[["symbol","baseAsset"]].copy()

def ticker_24h():
    for base in BINANCE_ENDPOINTS:
        try:
            r = S.get(f"{base}/api/v3/ticker/24hr", timeout=8)
            if r.status_code == 200:
                return pd.DataFrame(r.json())
        except requests.RequestException:
            pass
        time.sleep(0.15)
    return pd.DataFrame()

def top100_ids():
    info = exchange_info()
    t = ticker_24h()
    if info.empty or t.empty:
        # minimaler Fallback
        return ["BTCUSDT","ETHUSDT","SOLUSDT"]
    df = t.merge(info, on="symbol", how="inner")
    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
    df = df.sort_values("quoteVolume", ascending=False).head(int(os.getenv("TOP_N", "100")))
    return df["symbol"].tolist()

def load_history(symbol, days=180):
    kl = None
    for base in BINANCE_ENDPOINTS:
        try:
            r = S.get(f"{base}/api/v3/klines",
                      params={"symbol": symbol, "interval": "1d", "limit": min(1000, int(days)+5)},
                      timeout=8)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data and isinstance(data[0], list):
                    kl = data; break
        except requests.RequestException:
            pass
        time.sleep(0.15)
    if kl is None:
        return pd.DataFrame()

    df = pd.DataFrame(kl, columns=["openTime","open","high","low","close","volume",
                                   "closeTime","qav","numTrades","takerBase","takerQuote","ignore"])
    df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True, errors="coerce")
    df["price"]     = pd.to_numeric(df["close"], errors="coerce")
    df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
    df = df[["timestamp","price","volume"]].dropna().sort_values("timestamp").tail(int(days)+1)
    return df

def ma(series, w):
    return series.rolling(window=w, min_periods=max(2, w//2)).mean()

def compute_entry_flags(df, lookback, vol_surge):
    if df.empty: return pd.Series(dtype=bool)
    d = df.copy()
    d = d.set_index("timestamp").sort_index().resample("1D").last().dropna()
    if d.empty: return pd.Series(dtype=bool)
    d["ma20"] = ma(d["price"], 20)
    d["ma50"] = ma(d["price"], 50)
    d["roll_max_prev"] = d["price"].shift(1).rolling(int(lookback), min_periods=5).max()
    d["vol7"] = d["volume"].rolling(7, min_periods=3).mean()
    d["vol_ratio"] = d["volume"] / d["vol7"]
    entry = (d["price"] > d["ma20"]) & (d["ma20"] > d["ma50"]) & (d["price"] > d["roll_max_prev"]) & (d["vol_ratio"] >= vol_surge)
    return entry.fillna(False)

def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN","")
    chat  = os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat: return False
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=10)
        return True
    except Exception:
        return False

def load_state(path):
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def main():
    # Params
    DAYS   = int(os.getenv("DAYS_HIST", "120"))
    LOOK   = int(os.getenv("LOOKBACK", "20"))
    VSURGE = float(os.getenv("VOL_SURGE", "1.5"))
    COOLDN = int(os.getenv("ALERT_COOLDOWN_MIN", "120")) * 60
    APPURL = os.getenv("APP_URL", "").strip()
    STATEF = os.getenv("STATE_FILE", "autoscan_state.json")

    now = int(time.time())
    state = load_state(STATEF)
    state.setdefault("last_alert", {})  # {symbol: ts}

    ids = top100_ids()
    hits = []
    for sym in ids:
        df = load_history(sym, DAYS)
        if df.empty: continue
        entry = compute_entry_flags(df, LOOK, VSURGE)
        if entry.any():
            # letztes True
            last_idx = entry[entry].index[-1]
            # Cooldown pro Symbol
            last_ts = state["last_alert"].get(sym, 0)
            if now - last_ts >= COOLDN:
                name = sym.replace("USDT","")
                link = (APPURL + (("?coin="+sym) if "?" not in APPURL else ("&coin="+sym))) if APPURL else ""
                msg = f"🚨 Entry-Signal: {name} ({name}) erkannt.\n{link}".strip()
                if send_telegram(msg):
                    hits.append(sym)
                    state["last_alert"][sym] = now

    save_state(STATEF, state)
    # Exit-Code 0, egal ob Treffer; Actions sollen grün bleiben
    print(f"Done. Alerts sent for: {', '.join(hits) if hits else 'none'}")

if __name__ == "__main__":
    main()
