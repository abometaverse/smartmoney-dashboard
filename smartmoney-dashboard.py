# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Geschützt (Streamlit)
# v3.2  (Top-100: Binance-only & robust; Multi-endpoint; Dummy-Fallback,
#       Fortschritt, Table-driven Chart mit Entry-Dots, sauberes Money-Format)
#
# Secrets (Streamlit → Advanced settings → Secrets):
# APP_PASSWORD = "DeinStarkesPasswort"
# TELEGRAM_BOT_TOKEN = "123:abc"   # optional
# TELEGRAM_CHAT_ID   = "123456789" # optional
# -------------------------------------------------------------

import math
import time
from typing import Tuple, Dict, List

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# ----------------- App Config -----------------
st.set_page_config(page_title="Smart Money Dashboard — Geschützt", layout="wide")

# ================= Session Helpers =================
def save_state(keys):
    for k in keys:
        if k in st.session_state:
            st.session_state[f"_saved_{k}"] = st.session_state[k]

def restore_state(keys):
    for k in keys:
        saved_key = f"_saved_{k}"
        if saved_key in st.session_state:
            st.session_state[k] = st.session_state[saved_key]

# ================= Auth Gate =================
def auth_gate() -> None:
    st.title("🧠 Smart Money Dashboard — Geschützt")
    secret_pw = st.secrets.get("APP_PASSWORD", None)
    if not secret_pw:
        st.error("Konfiguration fehlt: Setze `APP_PASSWORD` unter Settings → Secrets.")
        st.stop()

    if st.session_state.get("AUTH_OK", False):
        top = st.container()
        col1, col2 = top.columns([6,1])
        col1.success("Zugriff gewährt.")
        if col2.button("Logout"):
            save_state([
                "selected_ids", "min_mktcap", "min_volume",
                "vol_surge_thresh", "lookback_res", "alerts_enabled",
                "days_hist", "batch_size_slider"
            ])
            st.session_state["AUTH_OK"] = False
            st.success("Einstellungen gespeichert.")
            time.sleep(0.2)
            st.rerun()
        return

    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input("Passwort eingeben", type="password")
        ok = st.form_submit_button("Login")
        if ok:
            if pw == secret_pw:
                st.session_state["AUTH_OK"] = True
                restore_state([
                    "selected_ids", "min_mktcap", "min_volume",
                    "vol_surge_thresh", "lookback_res", "alerts_enabled",
                    "days_hist", "batch_size_slider"
                ])
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    st.stop()

auth_gate()

# ================= Constants & HTTP =================
FIAT = "usd"
CG_BASE = "https://api.coingecko.com/api/v3"  # nur noch als Fallback für Historie/Snapshot

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

@st.cache_resource(show_spinner=False)
def get_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "smartmoney-dashboard/3.2 (+streamlit)"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

def _get_json(url, params=None, timeout=12, retries=2, backoff=1.6) -> Dict:
    session = get_http()
    last_err = ""
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return {"ok": True, "json": r.json(), "status": 200}
            if r.status_code in (429, 451, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (i + 1)); continue
            return {"ok": False, "json": None, "status": r.status_code, "error": r.text[:300]}
        except requests.RequestException as e:
            last_err = str(e)[:200]; time.sleep(backoff * (i + 1)); continue
    return {"ok": False, "json": None, "status": None, "error": last_err or "request failed"}

# ================= Formatting =================
def human_abbr(n: float) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)): return ""
    sign = "-" if n < 0 else ""
    x = abs(float(n))
    if x >= 1_000_000_000_000: return f"{sign}{x/1_000_000_000_000:.2f}T"
    if x >= 1_000_000_000:     return f"{sign}{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:         return f"{sign}{x/1_000_000:.2f}M"
    if x >= 1_000:             return f"{sign}{x:,.0f}"
    return f"{sign}{x:.2f}"

def fmt_money(n: float, decimals: int = 2) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)): return ""
    return f"{n:,.{decimals}f}"

# ================= Basic helpers =================
def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

# ---------- Binance helpers (robust multi-endpoint) ----------
def _binance_first_ok(path: str, params: Dict=None, timeout: int=10):
    sess = get_http()
    last_err = None
    for base in BINANCE_ENDPOINTS:
        url = f"{base}{path}"
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_err = r.status_code
            time.sleep(0.25)
        except requests.RequestException:
            last_err = "req"
            time.sleep(0.25)
    raise RuntimeError(f"binance_fail:{last_err}")

@st.cache_data(ttl=900, show_spinner=False)
def binance_exchange_info() -> pd.DataFrame:
    try:
        js = _binance_first_ok("/api/v3/exchangeInfo")
    except Exception:
        return pd.DataFrame()
    df = pd.DataFrame(js.get("symbols", []))
    if df.empty: return df
    df = df[(df["quoteAsset"]=="USDT") & (df["status"]=="TRADING")]
    # keine gehebelten/Leveraged Tokens
    mask = ~df["symbol"].str.contains(r"(UP|DOWN|BULL|BEAR)", regex=True)
    df = df[mask]
    return df[["symbol","baseAsset","quoteAsset"]].copy()

@st.cache_data(ttl=300, show_spinner=False)
def binance_ticker_24hr() -> pd.DataFrame:
    """Kombiniert 24h-Ticker über mehrere Endpoints, gibt DataFrame zurück."""
    frames = []
    for base in BINANCE_ENDPOINTS:
        try:
            url = f"{base}/api/v3/ticker/24hr"
            r = get_http().get(url, timeout=8)
            if r.status_code == 200:
                frames.append(pd.DataFrame(r.json()))
                break
        except Exception:
            time.sleep(0.2); continue
    if not frames: return pd.DataFrame()
    return frames[0]

@st.cache_data(ttl=300, show_spinner=False)
def binance_top100_by_quote_volume() -> pd.DataFrame:
    """Top-100 nach Quote-Volumen (24h), nur USDT-Spot, robust."""
    info = binance_exchange_info()
    if info.empty: return pd.DataFrame()
    t = binance_ticker_24hr()
    if t.empty: return pd.DataFrame()
    # nur USDT-Spot-Paare
    df = t.merge(info, on="symbol", how="inner")
    # numeric
    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
    df = df.sort_values("quoteVolume", ascending=False)
    df = df.head(100).reset_index(drop=True)
    return df[["symbol","baseAsset","quoteVolume"]]

# ---------- CoinGecko helpers (nur für Snapshot/Detail-Fallback) ----------
@st.cache_data(ttl=3600, show_spinner=False)
def cg_top_coins(limit: int = 500) -> pd.DataFrame:
    rows = []; per_page = 250; pages = int(np.ceil(limit / per_page))
    for page in range(1, pages + 1):
        resp = _get_json(
            f"{CG_BASE}/coins/markets",
            {"vs_currency": FIAT, "order": "market_cap_desc", "per_page": per_page, "page": page, "sparkline": "false"},
        )
        if not resp.get("ok"): break
        part = pd.DataFrame(resp["json"])[["id", "symbol", "name", "market_cap"]]
        rows.append(part)
    if not rows: return pd.DataFrame(columns=["id","symbol","name","market_cap"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["id"]).head(limit)

@st.cache_data(ttl=600, show_spinner=False)
def cg_simple_price(ids: List[str]) -> pd.DataFrame:
    if not ids: return pd.DataFrame()
    resp = _get_json(
        f"{CG_BASE}/coins/markets",
        {"vs_currency": FIAT, "ids": ",".join(ids), "order":"market_cap_desc", "per_page": max(1,len(ids)), "page":1, "sparkline":"false"}
    )
    if not resp.get("ok"): return pd.DataFrame()
    df = pd.DataFrame(resp["json"])
    cols = ["id","symbol","name","current_price","market_cap","total_volume","price_change_percentage_24h"]
    for c in cols: 
        if c not in df.columns: df[c]=np.nan
    return df[cols].rename(columns={"current_price":"price","total_volume":"volume_24h"})

# ================= Historie (Binance-first, CG fallback) =================
@st.cache_data(ttl=1200, show_spinner=False)
def load_history(coin_or_symbol: str, days: int = 180) -> pd.DataFrame:
    """Wenn coin_or_symbol wie 'BTCUSDT' aussieht -> direkt Binance.
       Sonst: versuche Binance-Mapping über baseAsset; wenn das nicht klappt -> CG-Fallback."""
    def _empty(status_text: str, source: str = "") -> pd.DataFrame:
        df = pd.DataFrame(columns=["timestamp","price","volume"])
        df.attrs["status"] = status_text
        if source: df.attrs["source"] = source
        return df

    sym = None
    if coin_or_symbol.upper().endswith("USDT") and coin_or_symbol.isascii():
        sym = coin_or_symbol.upper()
    if not sym:
        try:
            info = binance_exchange_info()
            base = coin_or_symbol.upper()
            row = info[(info["baseAsset"]==base)]
            if not row.empty: sym = row.iloc[0]["symbol"]
        except Exception:
            pass

    if sym:
        kl = None; last_status = None
        for base in BINANCE_ENDPOINTS:
            try:
                r = get_http().get(f"{base}/api/v3/klines",
                                   params={"symbol": sym, "interval":"1d", "limit":min(1000, int(days)+5)}, timeout=8)
                last_status = r.status_code
                if r.status_code != 200: time.sleep(0.2); continue
                data = r.json()
                if isinstance(data, list) and data and isinstance(data[0], (list,tuple)):
                    kl = data; break
            except Exception:
                time.sleep(0.2); continue
        if kl is None:
            pass
        else:
            try:
                df = pd.DataFrame(kl, columns=["openTime","open","high","low","close","volume","closeTime","qav","numTrades","takerBase","takerQuote","ignore"])
                df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True, errors="coerce")
                df["price"]     = pd.to_numeric(df["close"], errors="coerce")
                df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
                df = df[["timestamp","price","volume"]].dropna().sort_values("timestamp").tail(int(days)+1)
                if df.empty: return _empty("empty_binance","binance")
                df.attrs["status"]="ok"; df.attrs["source"]="binance"; return df
            except Exception:
                return _empty("parse_binance","binance")

    # --- CG fallback (nur wenn ID eine CG-ID ist) ---
    resp = _get_json(
        f"{CG_BASE}/coins/{coin_or_symbol}/market_chart",
        {"vs_currency": FIAT, "days": days, "interval": "daily"},
        timeout=10, retries=1, backoff=1.2
    )
    if not resp.get("ok"): return _empty(f"err_cg:{resp.get('status') or 'req'}","cg")
    data = resp["json"]; prices = data.get("prices", []); vols = data.get("total_volumes", [])
    if not prices: return _empty("empty_cg","cg")
    dfp = pd.DataFrame(prices, columns=["ts","price"]); dfv = pd.DataFrame(vols, columns=["ts","volume"])
    df = dfp.merge(dfv, on="ts", how="left")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce"); df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df[["timestamp","price","volume"]].dropna().sort_values("timestamp").tail(int(days)+1)
    if df.empty: return _empty("empty_cg","cg")
    df.attrs["status"]="ok"; df.attrs["source"]="cg"; return df

# ================= Analytics =================
def calc_local_levels(dfd: pd.DataFrame, lookback: int = 20) -> Tuple[float,float]:
    if dfd.empty: return (np.nan, np.nan)
    d = dfd.iloc[:-1].tail(lookback)
    if d.empty: return (np.nan, np.nan)
    return float(d["price"].max()), float(d["price"].min())

def volume_signals(dfd: pd.DataFrame) -> dict:
    out = {"vol_ratio_1d_vs_7d": np.nan, "distribution_risk": False, "price_chg_7d": np.nan}
    if dfd.empty or len(dfd) < 8: return out
    last = dfd.iloc[-1]; avg7 = dfd["volume"].iloc[-8:-1].mean()
    vr = float(last["volume"]/avg7) if (avg7 and avg7 == avg7) else np.nan
    out["vol_ratio_1d_vs_7d"] = vr
    p7 = dfd["price"].iloc[-8]; out["price_chg_7d"] = float((last["price"]/p7)-1.0) if p7 else np.nan
    out["distribution_risk"] = bool((out["price_chg_7d"] > 0) and (vr < 0.8)); return out

def trend_signals(dfd: pd.DataFrame) -> dict:
    out = {"ma20": np.nan, "ma50": np.nan, "breakout_ma": False}
    if dfd.empty: return out
    df = dfd.copy(); df["ma20"] = ma(df["price"], 20); df["ma50"] = ma(df["price"], 50); last = df.iloc[-1]
    out["ma20"], out["ma50"] = float(last["ma20"]), float(last["ma50"])
    out["breakout_ma"] = bool(last["price"] > last["ma20"] > last["ma50"]); return out

def send_telegram(msg: str) -> bool:
    token = st.secrets.get("TELEGRAM_BOT_TOKEN", None)
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", None)
    if not token or not chat_id: return False
    try:
        get_http().post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": msg}, timeout=10)
        return True
    except Exception:
        return False

def trailing_stop(current_high: float, trail_pct: float) -> float:
    return current_high * (1 - trail_pct/100.0)

# ================= Sidebar =================
st.sidebar.header("Settings")

# Defaults in Session
for k, v in {
    "selected_ids": [],
    "min_mktcap": 300_000_000,
    "min_volume": 50_000_000,
    "vol_surge_thresh": 1.5,
    "lookback_res": 20,
    "alerts_enabled": True,
    "days_hist": 90,
    "batch_size_slider": 3,
    "scan_index": 0
}.items():
    st.session_state.setdefault(k, v)

days_hist = st.sidebar.slider("Historie (Tage)", 60, 365, int(st.session_state["days_hist"]), 15, key="days_hist")

# Watchlist (CG für komfortable Suche; wenn CG ausfällt → Fallback IDs)
top_df = cg_top_coins(limit=500)
if top_df.empty:
    st.sidebar.warning("Top-Liste (CG) konnte nicht geladen werden. Fallback.")
    default_ids = ["bitcoin","ethereum","solana","arbitrum","render-token","bittensor"]
    selected_labels = st.sidebar.multiselect(
        "Watchlist (Fallback)",
        options=default_ids,
        default=st.session_state["selected_ids"] or default_ids[:3],
        key="watchlist_fallback"
    )
    selected_ids = selected_labels
else:
    top_df["label"] = top_df.apply(lambda r: f"{r['name']} ({str(r['symbol']).upper()}) — {r['id']}", axis=1)
    default_ids = st.session_state["selected_ids"] or ["bitcoin","ethereum","solana","arbitrum","render-token","bittensor"]
    default_labels = top_df[top_df["id"].isin(default_ids)]["label"].tolist()
    selected_labels = st.sidebar.multiselect(
        "Watchlist auswählen (Top 500, Suche per Tippen)",
        options=top_df["label"].tolist(),
        default=default_labels,
        help="Tippe Name oder Ticker, wähle per Klick.",
        key="watchlist_top"
    )
    label_to_id = dict(zip(top_df["label"], top_df["id"]))
    selected_ids = [label_to_id[l] for l in selected_labels]

manual = st.sidebar.text_input("Zusätzliche ID (optional)", value="", key="manual_id")
if manual.strip(): selected_ids.append(manual.strip())
if not selected_ids: selected_ids = ["bitcoin","ethereum"]

min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=int(st.session_state["min_mktcap"]), step=50_000_000, key="min_mc")
min_volume = st.sidebar.number_input("Min 24h Volume (USD)", min_value=0, value=int(st.session_state["min_volume"]), step=10_000_000, key="min_vol")
vol_surge_thresh = st.sidebar.slider("Vol Surge vs 7d (x)", 1.0, 5.0, float(st.session_state["vol_surge_thresh"]), 0.1, key="vol_surge")
lookback_res = st.sidebar.slider("Lookback für Widerstand/Support (Tage)", 10, 60, int(st.session_state["lookback_res"]), 1, key="lookback")
alerts_enabled = st.sidebar.checkbox("Telegram-Alerts aktivieren (Secrets nötig)", value=bool(st.session_state["alerts_enabled"]), key="alerts_on")

# Scan-Steuerung
scan_now_batch = st.sidebar.button("🔔 Watchlist BATCH scannen", key="scan_btn_batch")
scan_now_full  = st.sidebar.button("🔁 Ganze Watchlist scannen", key="scan_btn_full")

# Batch-Regler
batch_size = st.sidebar.slider("Coins pro Scan (Batchgröße)", 2, 15, int(st.session_state["batch_size_slider"]), 1, key="batch_size_slider")
if st.sidebar.button("🔄 Batch zurücksetzen", key="reset_batch_btn"):
    st.session_state["scan_index"] = 0

# persist basic settings
st.session_state["selected_ids"] = selected_ids
st.session_state["min_mktcap"]   = min_mktcap
st.session_state["min_volume"]   = min_volume
st.session_state["vol_surge_thresh"] = vol_surge_thresh
st.session_state["lookback_res"] = lookback_res
st.session_state["alerts_enabled"] = alerts_enabled

st.caption("🔒 Passwortschutz aktiv — Alerts via Telegram (optional). • Scans: Batch oder komplette Watchlist (mit Fortschritt).")

# ================= Snapshot =================
with st.spinner("Lade Snapshot …"):
    spot = cg_simple_price(selected_ids)

if not spot.empty:
    filt = spot[(spot["market_cap"] >= min_mktcap) & (spot["volume_24h"] >= min_volume)]
    st.subheader("📊 Snapshot (Filter)")
    disp = filt.rename(columns={
        "id":"ID","symbol":"Symbol","name":"Name","price":"Price",
        "market_cap":"MktCap","volume_24h":"Vol 24h","price_change_percentage_24h":"% 24h"
    }).copy()
    if not disp.empty:
        disp["Price"]   = disp["Price"].apply(lambda x: fmt_money(x, 4))
        disp["MktCap"]  = disp["MktCap"].apply(human_abbr)
        disp["Vol 24h"] = disp["Vol 24h"].apply(human_abbr)
        disp["% 24h"]   = pd.to_numeric(disp["% 24h"], errors="coerce").map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ================= Signals (Batch / Full) =================
if "signals_cache" not in st.session_state: st.session_state["signals_cache"] = pd.DataFrame()
if "history_cache" not in st.session_state: st.session_state["history_cache"] = {}

def _name_and_symbol(coin_id: str) -> Tuple[str,str]:
    """Name/Symbol-Anzeige. Für Binance-Symbole (XYZUSDT) -> (XYZ, XYZ)."""
    if coin_id.upper().endswith("USDT"):
        sym = coin_id[:-4].upper()
        return sym, sym
    # aus CG-Top500, falls vorhanden:
    try:
        row = top_df[top_df["id"]==coin_id]
        if not row.empty:
            return row.iloc[0]["name"], str(row.iloc[0]["symbol"]).upper()
    except Exception:
        pass
    return coin_id, coin_id[:4].upper()

def compute_rows_for_ids(id_list: List[str], days_hist: int, vol_thresh: float, lookback: int,
                         progress_label: str = "Scanne …") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows, history_cache = [], {}
    total = len(id_list)
    prog  = st.progress(0, text=progress_label)
    note  = st.empty()
    PAUSE_BETWEEN = 0.55

    for i, cid in enumerate(id_list, start=1):
        note.info(f"{progress_label}: Rang {i}/{total} — {cid}")
        time.sleep(PAUSE_BETWEEN)

        hist = load_history(cid, days=days_hist)
        status_val = hist.attrs.get("status", "") if isinstance(hist, pd.DataFrame) else "no_df"

        name, symbol = _name_and_symbol(cid)

        if (hist is None) or hist.empty or (status_val != "ok"):
            rows.append({
                "id": cid, "name": name, "symbol": symbol,
                "price": np.nan, "MA20": np.nan, "MA50": np.nan,
                "Breakout_MA": False, "Vol_Surge_x": np.nan,
                "Resistance": np.nan, "Support": np.nan,
                "Breakout_Resistance": False, "Distribution_Risk": False,
                "Entry_Signal": False, "status": status_val or "no data",
                "source": hist.attrs.get("source","") if isinstance(hist,pd.DataFrame) else ""
            })
            prog.progress(min(i/total, 1.0), text=f"{progress_label} ({i}/{total})")
            continue

        history_cache[cid] = hist
        dfd = hist.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd["price"]  = pd.to_numeric(dfd["price"], errors="coerce")
        dfd["volume"] = pd.to_numeric(dfd["volume"], errors="coerce")
        dfd = dfd.dropna(subset=["timestamp","price"]).set_index("timestamp").sort_index()
        dfd = dfd.resample("1D").last().dropna(subset=["price"])

        t_sig = trend_signals(dfd)
        v_sig = volume_signals(dfd)
        resistance, support = calc_local_levels(dfd, lookback=lookback)

        last = dfd.iloc[-1]
        price = float(last["price"])
        volsurge = v_sig["vol_ratio_1d_vs_7d"]; is_valid_vol = not np.isnan(volsurge)
        breakout_res = bool(price >= (resistance * 1.0005)) if not math.isnan(resistance) else False
        entry_ok = bool(t_sig["breakout_ma"] and is_valid_vol and (volsurge >= vol_thresh))

        rows.append({
            "id": cid, "name": name, "symbol": symbol,
            "price": price,
            "MA20": t_sig["ma20"], "MA50": t_sig["ma50"],
            "Breakout_MA": t_sig["breakout_ma"], "Vol_Surge_x": volsurge,
            "Resistance": resistance, "Support": support,
            "Breakout_Resistance": breakout_res,
            "Distribution_Risk": v_sig["distribution_risk"],
            "Entry_Signal": entry_ok and breakout_res,
            "status": "ok",
            "source": hist.attrs.get("source","")
        })
        prog.progress(min(i/total, 1.0), text=f"{progress_label} ({i}/{total})")

    prog.progress(1.0, text=f"{progress_label} (fertig)")
    note.empty()
    return pd.DataFrame(rows), history_cache

def run_scan_batch():
    start = st.session_state.get("scan_index", 0)
    end = min(start + int(st.session_state["batch_size_slider"]), len(selected_ids))
    batch = selected_ids[start:end]
    if not batch:
        st.warning("Keine Coins im aktuellen Batch. Batch zurücksetzen.")
        return pd.DataFrame(), {}
    st.info(f"⏳ Scanne Batch {start+1}–{end} von {len(selected_ids)} …")
    df, cache = compute_rows_for_ids(batch, days_hist, vol_surge_thresh, lookback_res, "Batch-Scan")
    st.session_state["scan_index"] = end % max(1, len(selected_ids))
    return df, cache

def run_scan_full_watchlist():
    st.info("🔁 Scanne gesamte Watchlist …")
    return compute_rows_for_ids(selected_ids, days_hist, vol_surge_thresh, lookback_res, "Watchlist-Scan")

# Ausführung Scans
if scan_now_batch:
    sig, hist_cache = run_scan_batch()
    st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(hist_cache)
elif scan_now_full:
    sig, hist_cache = run_scan_full_watchlist()
    st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(hist_cache)

signals_df = st.session_state["signals_cache"].copy()

st.subheader("🔎 Signals & Levels (Watchlist)")
if not signals_df.empty:
    for c in ["price","MA20","MA50","Resistance","Support"]:
        if c in signals_df.columns: signals_df[c] = pd.to_numeric(signals_df[c], errors="coerce")
    display_df = signals_df.copy()
    display_df["price"]      = display_df["price"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["MA20"]       = display_df["MA20"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["MA50"]       = display_df["MA50"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["Resistance"] = display_df["Resistance"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["Support"]    = display_df["Support"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["Vol_Surge_x"]= pd.to_numeric(display_df["Vol_Surge_x"], errors="coerce").map(lambda x: f"{x:.2f}x" if pd.notna(x) else "")

    def _row_style(row):
        if bool(row.get("Entry_Signal", False)): return ['background-color: #e6ffed'] * len(row)
        if bool(row.get("Distribution_Risk", False)): return ['background-color: #ffecec'] * len(row)
        if bool(row.get("Breakout_MA", False)) or bool(row.get("Breakout_Resistance", False)):
            return ['background-color: #fff9e6'] * len(row)
        return [''] * len(row)

    display_df = display_df[["name","symbol","id","price","MA20","MA50","Vol_Surge_x","Resistance","Support","Breakout_MA","Breakout_Resistance","Distribution_Risk","Entry_Signal","status","source"]]
    st.dataframe(display_df.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)

# ================= Detail & Risk — Top-100 (Binance-only) =================
st.markdown("---")
st.subheader("📈 Detail & Risk — Top-100 Universum (Binance)")

st.session_state.setdefault("top100_df", pd.DataFrame())
st.session_state.setdefault("top100_cache_ts", 0.0)

c1, c2, c3 = st.columns([1.5,1.2,1.2])
with c1:
    auto_refresh = st.toggle("Top-100 bei Aufruf aktualisieren", value=False)
with c2:
    refresh_btn = st.button("🔄 Top-100 aktualisieren")
with c3:
    top_filter = st.radio("Filter", ["Alle (Top-100)", "Nur Entry-Signal (Top-100)", "Nur Watchlist"], index=0, horizontal=True)

def load_top100(days_hist: int, vol_surge_thresh: float, lookback_res: int) -> pd.DataFrame:
    top = binance_top100_by_quote_volume()
    if top.empty:
        # Dummy-Fallback, falls Binance ebenfalls blockt
        st.warning("Binance Top-100 nicht verfügbar. Verwende Dummy-Fallback (BTC/ETH/SOL).")
        top = pd.DataFrame({"symbol":["BTCUSDT","ETHUSDT","SOLUSDT"], "baseAsset":["BTC","ETH","SOL"], "quoteVolume":[0,0,0]})

    ids = top["symbol"].tolist()
    names = {r["symbol"]:(str(r["baseAsset"]).upper(), str(r["baseAsset"]).upper()) for _, r in top.iterrows()}

    df100, cache = compute_rows_for_ids(ids, days_hist, vol_surge_thresh, lookback_res, "Top-100-Scan")
    # Name + Symbol sauber setzen
    df100["name"]   = df100["id"].map(lambda x: names.get(x, (x,x))[0])
    df100["symbol"] = df100["id"].map(lambda x: names.get(x, (x,x))[1])

    st.session_state["history_cache"].update(cache)
    return df100

do_load = refresh_btn or (auto_refresh and (time.time() - st.session_state["top100_cache_ts"] > 120)) or st.session_state["top100_df"].empty
if do_load:
    df100 = load_top100(st.session_state["days_hist"], st.session_state["vol_surge_thresh"], st.session_state["lookback_res"])
    if not df100.empty:
        st.session_state["top100_df"] = df100
        st.session_state["top100_cache_ts"] = time.time()

top100_df = st.session_state["top100_df"].copy()

# ---- Table-driven Selection (Checkbox "View") ----
selected_coin_from_table = None

if not top100_df.empty:
    if top_filter == "Nur Entry-Signal (Top-100)":
        top100_view = top100_df[(top100_df["Entry_Signal"] == True) & (top100_df["status"] == "ok")].copy()
    elif top_filter == "Nur Watchlist":
        wl = set(selected_ids); top100_view = top100_df[top100_df["id"].isin(wl)].copy()
    else:
        top100_view = top100_df.copy()

    top100_view = top100_view.sort_values(["Entry_Signal","id"], ascending=[False, True]).reset_index(drop=True)
    if "View" not in top100_view.columns:
        top100_view["View"] = False

    prev = st.session_state.get("selected_coin")
    if prev in set(top100_view["id"]):
        top100_view.loc[top100_view["id"]==prev, "View"] = True
    elif not top100_view.empty:
        top100_view.loc[0, "View"] = True

    show_cols = ["View","name","symbol","id","price","MA20","MA50","Vol_Surge_x","Resistance","Support","Breakout_MA","Breakout_Resistance","Distribution_Risk","Entry_Signal","status","source"]
    for c in ["price","MA20","MA50","Resistance","Support","Vol_Surge_x"]:
        if c in top100_view.columns:
            top100_view[c] = pd.to_numeric(top100_view[c], errors="coerce")

    st.data_editor(
        top100_view[show_cols],
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in show_cols if c!="View"],
        column_config={"View": st.column_config.CheckboxColumn(required=False, help="Anklicken, um den Chart unten zu laden")},
        num_rows="fixed",
        key="top100_editor"
    )

    current_view_df = st.session_state["top100_editor"]["data"] if "data" in st.session_state["top100_editor"] else top100_view.to_dict("records")
    for row in current_view_df:
        if row.get("View", False):
            selected_coin_from_table = row.get("id")
            break
    if not selected_coin_from_table and not top100_view.empty:
        selected_coin_from_table = top100_view.iloc[0]["id"]

    st.session_state["selected_coin"] = selected_coin_from_table
else:
    st.info("Noch keine Top-100 Daten geladen. Klicke auf **„Top-100 aktualisieren“** oder aktiviere Auto-Refresh.")

# ================= Einzel-Chart & Risk-Tools =================
st.markdown("---")
st.subheader("🔍 Einzel-Chart & Risk-Tools")

coin_select = st.session_state.get("selected_coin") or (selected_ids[0] if selected_ids else "bitcoin")

if coin_select:
    with st.spinner(f"Lade Historie für {coin_select} …"):
        d = st.session_state.get("history_cache", {}).get(coin_select)
        if d is None or not isinstance(d, pd.DataFrame) or d.empty:
            d = load_history(coin_select, days=st.session_state["days_hist"])
            if isinstance(d, pd.DataFrame) and not d.empty:
                st.session_state.setdefault("history_cache", {})
                st.session_state["history_cache"][coin_select] = d

    status_val = d.attrs.get("status", "") if isinstance(d, pd.DataFrame) else ""
    src_val    = d.attrs.get("source", "")
    if (d is None) or d.empty or (status_val != "ok"):
        st.warning("Keine Historie verfügbar (API-Limit, 451/403 oder leere Daten).")
    else:
        st.caption(f"Datenquelle: {src_val or 'unbekannt'}")

        dfd = d.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd["price"]  = pd.to_numeric(dfd["price"], errors="coerce")
        dfd["volume"] = pd.to_numeric(dfd["volume"], errors="coerce")
        dfd = dfd.dropna(subset=["timestamp","price"]).set_index("timestamp").sort_index()
        d_daily = dfd.resample("1D").last().dropna(subset=["price"]) if not dfd.empty else dfd
        if d_daily.empty:
            st.warning("Keine Tagesdaten.")
        else:
            r, s  = calc_local_levels(d_daily, lookback_res)
            d_daily["ma20"] = ma(d_daily["price"], 20)
            d_daily["ma50"] = ma(d_daily["price"], 50)
            d_daily["vol7"] = d_daily["volume"].rolling(7, min_periods=3).mean()
            d_daily["vol_ratio"] = d_daily["volume"] / d_daily["vol7"]
            d_daily["roll_max_prev"] = d_daily["price"].shift(1).rolling(lookback_res, min_periods=5).max()
            d_daily["entry_flag"] = (d_daily["price"] > d_daily["ma20"]) & (d_daily["ma20"] > d_daily["ma50"]) & \
                                    (d_daily["price"] > d_daily["roll_max_prev"]) & \
                                    (d_daily["vol_ratio"] >= st.session_state["vol_surge_thresh"])
            entries = d_daily[d_daily["entry_flag"]]

            fig, ax_price = plt.subplots()
            ax_vol = ax_price.twinx()

            ax_price.plot(d_daily.index, d_daily["price"], label="Price", linewidth=1.6)
            ax_price.plot(d_daily.index, d_daily["ma20"],  label="MA20", linewidth=1.0)
            ax_price.plot(d_daily.index, d_daily["ma50"],  label="MA50", linewidth=1.0)
            if not np.isnan(r): ax_price.axhline(r, linestyle="--", label=f"Resistance {r:.3f}")
            if not np.isnan(s): ax_price.axhline(s, linestyle="--", label=f"Support {s:.3f}")

            if not entries.empty:
                ax_price.scatter(entries.index, entries["price"], s=36, zorder=5, color="#16a34a", label="Entry (hist)")

            ax_vol.bar(d_daily.index, d_daily["volume"], alpha=0.28)
            ax_vol.set_ylabel("Volume")

            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            formatter = mdates.ConciseDateFormatter(locator)
            ax_price.xaxis.set_major_locator(locator)
            ax_price.xaxis.set_major_formatter(formatter)

            ax_price.set_title(f"{coin_select} — Price, MAs, Levels & Volume")
            ax_price.set_xlabel("Date"); ax_price.set_ylabel("USD")
            ax_price.legend(loc="upper left")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # === Position & Trailing Stop ===
            st.markdown("### 🧮 Position & Trailing Stop")
            c1, c2, c3, c4 = st.columns(4)
            last_px = float(d_daily['price'].iloc[-1])
            portfolio   = c1.number_input("Portfolio (USD)", min_value=0.0, value=8000.0, step=100.0, format="%.2f", key=f"pos_port_{coin_select}")
            risk_pct    = c2.slider("Risiko/Trade (%)", 0.5, 3.0, 2.0, 0.1, key=f"pos_risk_{coin_select}")
            stop_pct    = c3.slider("Stop-Entfernung (%)", 3.0, 25.0, 8.0, 0.5, key=f"pos_stop_{coin_select}")
            entry_price = c4.number_input("Entry-Preis", min_value=0.0, value=last_px, step=0.001, format="%.6f", key=f"pos_entry_{coin_select}")

            max_loss = portfolio * (risk_pct/100.0)
            size_usd = max_loss / (stop_pct/100.0) if stop_pct>0 else 0.0
            size_qty = size_usd / entry_price if entry_price>0 else 0.0
            st.write(f"**Max. Verlust:** ${max_loss:,.2f} • **Positionsgröße:** ${size_usd:,.2f} (~ {size_qty:,.4f} {str(coin_select).upper()})")

            st.markdown("#### Trailing Stop")
            t1, t2 = st.columns(2)
            trail_pct = t1.slider("Trail (%)", 5.0, 25.0, 10.0, 0.5, key=f"trail_pct_{coin_select}")
            high_since_entry = t2.number_input("Höchster Kurs seit Entry", min_value=0.0, value=last_px, step=0.001, format="%.6f", key=f"trail_high_{coin_select}")
            tstop = trailing_stop(high_since_entry, trail_pct)
            st.write(f"Trailing Stop bei **${tstop:,.3f}** (High {high_since_entry:,.3f}, Trail {trail_pct:.1f}%)")

            v_sig = volume_signals(d_daily)
            if v_sig["distribution_risk"]:
                st.warning("Distribution-Risk: Preis ↑ bei Volumen < 0.8× 7d-Ø.")
            else:
                st.success("Volumen ok (keine Distribution-Anzeichen).")

# Fortschritts-Hinweis bei Batch
if len(selected_ids) > 0:
    start = st.session_state.get("scan_index", 0)
    end = min(start + int(st.session_state["batch_size_slider"]), len(selected_ids))
    if end == len(selected_ids):
        st.success("✅ Batch-Scan: Ende der Liste erreicht. Nächster Klick startet wieder vorn.")
    else:
        nxt_end = min(end + int(st.session_state["batch_size_slider"]), len(selected_ids))
        st.info(f"➡️ Nächster Batch lädt Coins {end+1}–{nxt_end} von {len(selected_ids)}.")
