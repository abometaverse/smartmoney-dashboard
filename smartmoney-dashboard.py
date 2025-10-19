# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Geschützt (Streamlit)
# v5.3 (CMC-Link in Tabelle)
#  - CMC-Link-Spalte ("CMC"): Klick öffnet CoinMarketCap-Suche für den Coin
#  - Ermittelt Ticker aus 'symbol' oder aus 'id' (…USDT -> Basis)
#  - Rest unverändert: eine Tabelle (Watchlist+Top100), Double-Click-Select,
#    Zellfarben (Vol/Breakout/Entry), max. 2 Nachkommastellen, Telegram, Auto-Scan
# -------------------------------------------------------------

import math
import time
from typing import Tuple, Dict, List, Optional

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# ----------------- App Config -----------------
st.set_page_config(page_title="Smart Money Dashboard — Geschützt", layout="wide")

# ================= Session Helpers =================
PERSIST_KEYS = [
    "selected_ids","vol_surge_thresh","lookback_res","alerts_enabled",
    "days_hist","batch_size_slider","scan_index","selected_coin",
    "top100_df","top100_last_sync_ts","top100_cooldown_min",
    "auto_scan_enabled","auto_scan_hours","auto_last_ts","auto_alerted_ids",
    "signals_cache","history_cache"
]

def save_state(keys):
    for k in keys:
        if k in st.session_state:
            st.session_state[f"_saved_{k}"] = st.session_state[k]

def restore_state(keys):
    for k in keys:
        saved_key = f"_saved_{k}"
        if saved_key in st.session_state:
            st.session_state[k] = st.session_state[saved_key]

def ensure_defaults():
    defaults = {
        "selected_ids": [],
        "vol_surge_thresh": 1.5,
        "lookback_res": 20,
        "alerts_enabled": True,
        "days_hist": 90,
        "batch_size_slider": 3,
        "scan_index": 0,
        "selected_coin": None,
        "top100_df": pd.DataFrame(),
        "top100_last_sync_ts": 0.0,
        "top100_cooldown_min": 15,
        "auto_scan_enabled": False,
        "auto_scan_hours": 1.0,
        "auto_last_ts": 0.0,
        "auto_alerted_ids": [],
        "signals_cache": pd.DataFrame(),
        "history_cache": {},
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def set_active_coin(coin_id: str):
    st.session_state["selected_coin"] = str(coin_id)

# ================= Auth Gate =================
def auth_gate() -> None:
    st.title("🧠 Smart Money Dashboard — Geschützt")
    secret_pw = st.secrets.get("APP_PASSWORD", None)
    if not secret_pw:
        st.error("Konfiguration fehlt: Setze `APP_PASSWORD` unter Settings → Secrets.")
        st.stop()

    if st.session_state.get("AUTH_OK", False):
        top = st.container()
        c1, c2 = top.columns([6,1])
        c1.success("Zugriff gewährt.")
        if c2.button("Logout"):
            save_state(PERSIST_KEYS)
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
                restore_state(PERSIST_KEYS)
                ensure_defaults()
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    st.stop()

auth_gate()
ensure_defaults()

# ================= Constants & HTTP =================
FIAT = "usd"
CG_BASE = "https://api.coingecko.com/api/v3"   # nur für Watchlist-Suche
BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

@st.cache_resource(show_spinner=False)
def get_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "smartmoney-dashboard/5.3 (+streamlit)"})
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
def fmt_money(n: float, decimals: int = 2) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)): return ""
    return f"{n:,.{decimals}f}"

# ================= Basic helpers =================
def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

# ---------- Binance helpers (multi-endpoint) ----------
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
            time.sleep(0.2)
        except requests.RequestException:
            last_err = "req"
            time.sleep(0.2)
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
    mask = ~df["symbol"].str.contains(r"(?:UP|DOWN|BULL|BEAR)", regex=True)
    df = df[mask]
    return df[["symbol","baseAsset","quoteAsset"]].copy()

@st.cache_data(ttl=300, show_spinner=False)
def binance_ticker_24hr() -> pd.DataFrame:
    for base in BINANCE_ENDPOINTS:
        try:
            url = f"{base}/api/v3/ticker/24hr"
            r = get_http().get(url, timeout=8)
            if r.status_code == 200:
                return pd.DataFrame(r.json())
        except Exception:
            time.sleep(0.2)
            continue
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def binance_top100_by_quote_volume() -> pd.DataFrame:
    info = binance_exchange_info()
    if info.empty: return pd.DataFrame()
    t = binance_ticker_24hr()
    if t.empty: return pd.DataFrame()
    df = t.merge(info, on="symbol", how="inner")
    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
    df = df.sort_values("quoteVolume", ascending=False).head(100).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["name"] = df["baseAsset"].str.upper()
    df["symbol_txt"] = df["baseAsset"].str.upper()
    df["id"] = df["symbol"]
    return df[["rank","id","symbol","name","symbol_txt","quoteVolume"]].rename(columns={"symbol_txt":"symbol2"})

# ---------- CoinGecko nur für Watchlist-Suche ----------
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

# ---------- CG-ID → Binance-Symbol Mapping ----------
@st.cache_data(ttl=3600, show_spinner=False)
def resolve_to_binance_symbol(coin_id_or_symbol: str) -> Optional[str]:
    if not coin_id_or_symbol: return None
    s = str(coin_id_or_symbol).strip()
    if s.upper().endswith("USDT"):
        return s.upper()
    base = s.upper()
    info = binance_exchange_info()
    if not info.empty and base in set(info["baseAsset"].str.upper()):
        sym = info.loc[info["baseAsset"].str.upper()==base, "symbol"].iloc[0]
        return str(sym)
    top = cg_top_coins(limit=500)
    if not top.empty and s in set(top["id"]):
        cg_sym = top.loc[top["id"]==s, "symbol"].iloc[0]
        base2 = str(cg_sym).upper()
        if base2 in set(info["baseAsset"].str.upper()):
            sym = info.loc[info["baseAsset"].str.upper()==base2, "symbol"].iloc[0]
            return str(sym)
    return None

# ---------- History (Binance-first) ----------
@st.cache_data(ttl=1200, show_spinner=False)
def load_history(coin_or_symbol: str, days: int = 180) -> pd.DataFrame:
    def _empty(status_text: str, source: str = "") -> pd.DataFrame:
        df = pd.DataFrame(columns=["timestamp","price","volume"])
        df.attrs["status"] = status_text
        if source: df.attrs["source"] = source
        return df

    sym = resolve_to_binance_symbol(coin_or_symbol)
    if not sym:
        return _empty("no_route","")

    kl = None
    for base in BINANCE_ENDPOINTS:
        try:
            r = get_http().get(f"{base}/api/v3/klines",
                               params={"symbol": sym, "interval":"1d", "limit":min(1000, int(days)+5)}, timeout=8)
            if r.status_code != 200:
                time.sleep(0.2); continue
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], (list,tuple)):
                kl = data; break
        except Exception:
            time.sleep(0.2); continue

    if kl is None:
        return _empty("err_binance","binance")

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

def trailing_stop(current_high: float, trail_pct: float) -> float:
    return current_high * (1 - trail_pct/100.0)

# ================= Telegram =================
def send_telegram(msg: str) -> bool:
    token = st.secrets.get("TELEGRAM_BOT_TOKEN", None)
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", None)
    if not token or not chat_id:
        return False
    try:
        get_http().post(f"https://api.telegram.org/bot{token}/sendMessage",
                        json={"chat_id": chat_id, "text": msg}, timeout=10)
        return True
    except Exception:
        return False

def telegram_alert_for_entries(df: pd.DataFrame) -> List[str]:
    if df.empty: return []
    app_url = st.secrets.get("APP_URL", "")
    alerted = []
    sent_before: set = set(st.session_state.get("auto_alerted_ids", []))
    for _, r in df[(df["Entry_Signal"]==True) & (df["status"]=="ok")].iterrows():
        cid = str(r["id"])
        if cid in sent_before:
            continue
        name = str(r.get("name", cid))
        sym  = str(r.get("symbol", "")) or cid
        link = f"\n{app_url}?coin={cid}" if app_url else ""
        msg = f"🚨 Entry-Signal: {name} ({sym}) erkannt.{link}"
        ok = send_telegram(msg)
        if ok:
            alerted.append(cid)
            sent_before.add(cid)
            st.session_state["selected_coin"] = cid
    st.session_state["auto_alerted_ids"] = list(sent_before)
    return alerted

# ================= Sidebar =================
st.sidebar.header("Settings")

days_hist = st.sidebar.slider("Historie (Tage)", 60, 365, int(st.session_state["days_hist"]), 15, key="days_hist")
vol_surge_thresh = st.sidebar.slider("Vol Surge vs 7d (x)", 1.0, 5.0, float(st.session_state["vol_surge_thresh"]), 0.1, key="vol_surge")
lookback_res = st.sidebar.slider("Lookback für Widerstand/Support (Tage)", 10, 60, int(st.session_state["lookback_res"]), 1, key="lookback")

st.session_state["top100_cooldown_min"] = st.sidebar.number_input(
    "Top-100 Refresh-Sperre (Minuten)", min_value=1, max_value=120,
    value=int(st.session_state["top100_cooldown_min"]), step=1, key="cooldown_min"
)

c_as1, c_as2 = st.sidebar.columns([1,1])
st.session_state["auto_scan_enabled"] = c_as1.checkbox("Auto-Scan & Telegram", value=bool(st.session_state["auto_scan_enabled"]), key="auto_enabled")
st.session_state["auto_scan_hours"]   = c_as2.number_input("Intervall (Std.)", min_value=0.5, max_value=24.0, step=0.5, value=float(st.session_state["auto_scan_hours"]), key="auto_hours")

# Watchlist (CG nur für Suche/Komfort)
top_df_search = cg_top_coins(limit=500)
if top_df_search.empty:
    st.sidebar.warning("Top-Liste (CG) aktuell nicht verfügbar. Fallback-Auswahl.")
    default_ids = ["bitcoin","ethereum","solana","render-token","bittensor"]
    selected_labels = st.sidebar.multiselect(
        "Watchlist (Fallback, IDs)",
        options=default_ids,
        default=st.session_state["selected_ids"] or default_ids[:3],
        key="watchlist_fallback"
    )
    selected_ids = selected_labels
else:
    top_df_search["label"] = top_df_search.apply(lambda r: f"{r['name']} ({str(r['symbol']).upper()}) — {r['id']}", axis=1)
    default_ids = st.session_state["selected_ids"] or ["bitcoin","ethereum","solana","render-token","bittensor"]
    default_labels = top_df_search[top_df_search["id"].isin(default_ids)]["label"].tolist()
    selected_labels = st.sidebar.multiselect(
        "Watchlist (Top 500, Suche per Tippen)",
        options=top_df_search["label"].tolist(),
        default=default_labels,
        key="watchlist_top"
    )
    label_to_id = dict(zip(top_df_search["label"], top_df_search["id"]))
    selected_ids = [label_to_id.get(l, l) for l in selected_labels]

manual = st.sidebar.text_input("Zusätzliche ID (optional: CG-ID oder BASE/BASEUSDT)", value="", key="manual_id")
if manual.strip(): selected_ids.append(manual.strip())
if not selected_ids: selected_ids = ["bitcoin","ethereum"]
st.session_state["selected_ids"] = selected_ids

# Scan-Steuerung
c_scan1, c_scan2 = st.sidebar.columns(2)
scan_now_full  = c_scan1.button("🔁 Ganze Watchlist", key="scan_btn_full")
refresh_top100 = c_scan2.button("🔄 Top-100 aktualisieren", key="top100_refresh")

batch_size = st.sidebar.slider("Coins pro Scan (Batchgröße)", 2, 15, int(st.session_state["batch_size_slider"]), 1, key="batch_size_slider")
scan_now_batch = st.sidebar.button("🔔 Batch scannen", key="scan_btn_batch")
if st.sidebar.button("🔄 Batch zurücksetzen", key="reset_batch_btn"):
    st.session_state["scan_index"] = 0

st.caption("🔒 Passwortschutz aktiv • Eine Tabelle mit Filter & Doppelklick-Auswahl • Telegram-Alerts optional.")

# ================= Utility: Name/Symbol =================
def _name_and_symbol_any(coin_id_or_symbol: str) -> Tuple[str,str]:
    sym = resolve_to_binance_symbol(coin_id_or_symbol)
    base_guess = None
    if sym and sym.upper().endswith("USDT"):
        base_guess = sym[:-4].upper()
    try:
        if isinstance(top_df_search, pd.DataFrame) and not top_df_search.empty and coin_id_or_symbol in set(top_df_search["id"]):
            row = top_df_search[top_df_search["id"]==coin_id_or_symbol].iloc[0]
            return str(row["name"]), str(row["symbol"]).upper()
    except Exception:
        pass
    s = str(coin_id_or_symbol).upper()
    if base_guess:
        return base_guess, base_guess
    if s.endswith("USDT"): s = s[:-4]
    return s, s

# ================= Signals & Levels — Core Scan =================
def compute_rows_for_ids(id_list: List[str], days_hist: int, vol_thresh: float, lookback: int,
                         progress_label: str = "Scanne …") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows, history_cache = [], {}
    total = len(id_list)
    if total == 0:
        return pd.DataFrame(), {}
    prog  = st.progress(0, text=progress_label)
    note  = st.empty()
    PAUSE_BETWEEN = 0.45

    for i, cid in enumerate(id_list, start=1):
        note.info(f"{progress_label}: {i}/{total} — {cid}")
        time.sleep(PAUSE_BETWEEN)
        hist = load_history(cid, days=days_hist)
        status_val = hist.attrs.get("status", "") if isinstance(hist, pd.DataFrame) else "no_df"
        name, symbol = _name_and_symbol_any(cid)

        if (hist is None) or hist.empty or (status_val != "ok"):
            rows.append({
                "universe": "", "rank": i, "name": name, "symbol": symbol,
                "id": resolve_to_binance_symbol(cid) or cid, "price": np.nan, "MA20": np.nan, "MA50": np.nan,
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
            "universe": "", "rank": i, "name": name, "symbol": symbol,
            "id": resolve_to_binance_symbol(cid) or cid, "price": price,
            "MA20": t_sig["ma20"], "MA50": t_sig["ma50"],
            "Vol_Surge_x": volsurge, "Breakout_MA": t_sig["breakout_ma"],
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
    df = pd.DataFrame(rows)
    df = df.sort_values("rank", kind="stable").reset_index(drop=True)
    return df, history_cache

def run_scan_full_watchlist():
    ids = list(st.session_state.get("selected_ids", []))
    df, cache = compute_rows_for_ids(ids, st.session_state["days_hist"], st.session_state["vol_surge_thresh"], st.session_state["lookback_res"], "Watchlist-Scan")
    return df, cache

def run_scan_batch():
    ids = list(st.session_state.get("selected_ids", []))
    start = st.session_state.get("scan_index", 0)
    end = min(start + int(st.session_state["batch_size_slider"]), len(ids))
    batch = ids[start:end]
    if not batch:
        st.warning("Keine Coins im aktuellen Batch. Batch zurücksetzen.")
        return pd.DataFrame(), {}
    df, cache = compute_rows_for_ids(batch, st.session_state["days_hist"], st.session_state["vol_surge_thresh"], st.session_state["lookback_res"], "Batch-Scan")
    st.session_state["scan_index"] = end % max(1, len(ids))
    return df, cache

# Aktionen (Scans)
if scan_now_full:
    sig, hist_cache = run_scan_full_watchlist()
    if not sig.empty:
        st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(hist_cache)

if scan_now_batch:
    sig, hist_cache = run_scan_batch()
    if not sig.empty:
        st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(hist_cache)

# Top-100 Builder & Refresh
def build_top100(days_hist: int, vol_surge_thresh: float, lookback_res: int) -> pd.DataFrame:
    base = binance_top100_by_quote_volume()
    if base.empty:
        base = pd.DataFrame({
            "rank":[1,2,3],
            "id":["BTCUSDT","ETHUSDT","SOLUSDT"],
            "symbol":["BTCUSDT","ETHUSDT","SOLUSDT"],
            "name":["BTC","ETH","SOL"],
            "symbol2":["BTC","ETH","SOL"],
            "quoteVolume":[0,0,0]
        })
    ids = base["id"].tolist()
    names = {r["id"]:(str(r["name"]), str(r.get("symbol2", r["name"]))) for _, r in base.iterrows()}
    df100, cache = compute_rows_for_ids(ids, days_hist, vol_surge_thresh, lookback_res, "Top-100-Scan")
    if df100.empty: return df100
    df100["universe"] = "Top100"
    df100["rank"] = df100["id"].map(lambda x: int(base[base["id"]==x]["rank"].iloc[0]) if x in set(base["id"]) and not base[base["id"]==x].empty else 999)
    df100["name"]   = df100["id"].map(lambda x: names.get(x, (x,x))[0])
    df100["symbol"] = df100["id"].map(lambda x: names.get(x, (x,x))[1])
    st.session_state["history_cache"].update(cache)
    return df100.sort_values("rank", kind="stable").reset_index(drop=True)

now = time.time()
cooldown = int(st.session_state["top100_cooldown_min"]) * 60
should_refresh = refresh_top100 and (now - st.session_state["top100_last_sync_ts"] >= cooldown)

if refresh_top100 and not should_refresh and st.session_state["top100_df"].empty is False:
    wait_left = int((st.session_state["top100_last_sync_ts"] + cooldown - now) / 60) + 1
    st.warning(f"Top-100 Refresh-Sperre aktiv. Bitte in ~{max(wait_left,1)} Min. erneut.")

if should_refresh or st.session_state["top100_df"].empty:
    df100_new = build_top100(st.session_state["days_hist"], st.session_state["vol_surge_thresh"], st.session_state["lookback_res"])
    if not df100_new.empty:
        st.session_state["top100_df"] = df100_new
        st.session_state["top100_last_sync_ts"] = time.time()
        if st.session_state.get("alerts_enabled", True) and should_refresh:
            telegram_alert_for_entries(df100_new)

# Auto-Scan
if st.session_state["auto_scan_enabled"]:
    last = float(st.session_state.get("auto_last_ts", 0.0) or 0.0)
    interval = float(st.session_state.get("auto_scan_hours", 1.0)) * 3600.0
    if time.time() - last >= interval:
        df100_new = build_top100(st.session_state["days_hist"], st.session_state["vol_surge_thresh"], st.session_state["lookback_res"])
        if not df100_new.empty:
            st.session_state["top100_df"] = df100_new
            st.session_state["top100_last_sync_ts"] = time.time()
            if st.session_state.get("alerts_enabled", True):
                hits = telegram_alert_for_entries(df100_new)
                if hits:
                    st.session_state["selected_coin"] = hits[0]
        st.session_state["auto_last_ts"] = time.time()

# ===================== Eine Tabelle (Union) =====================
st.markdown("---")
hdr = st.container()
cA, cB = hdr.columns([6,1])
cA.subheader("📊 Watchlist + Top-100 (eine Tabelle)")
ts = st.session_state.get("top100_last_sync_ts", 0.0)
if ts:
    cB.caption(f"Letzter Top-100-Sync: {time.strftime('%Y-%m-%d %H:%M', time.gmtime(ts))} UTC")

watch_df = st.session_state.get("signals_cache", pd.DataFrame()).copy()
top_df   = st.session_state.get("top100_df", pd.DataFrame()).copy()

if not watch_df.empty:
    watch_df["universe"] = "Watchlist"
if not top_df.empty:
    top_df["universe"] = "Top100"

union = pd.concat([watch_df.assign(_prio=0), top_df.assign(_prio=1)], ignore_index=True)
if not union.empty:
    union.sort_values(by=["id","_prio"], inplace=True)
    union = union.drop_duplicates(subset=["id"], keep="first").drop(columns=["_prio"])

flt = st.radio("Filter", ["Top-100", "Watchlist", "Entry-Signale"], horizontal=True)
if flt == "Top-100":
    union = union[union["universe"]=="Top100"]
elif flt == "Watchlist":
    union = union[union["universe"]=="Watchlist"]
else:
    union = union[(union["Entry_Signal"]==True) & (union["status"]=="ok")]

if st.session_state.get("selected_coin") is None:
    qp = st.query_params.get("coin")
    if qp:
        st.session_state["selected_coin"] = str(qp)

if union.empty:
    st.info("Keine Daten für den gewählten Filter. Scanne Watchlist oder aktualisiere Top-100.")
else:
    display = union.copy()

    # Datenquelle-Kürzel (Bi/Cg)
    def _src_short(s: str) -> str:
        s = (s or "").lower()
        if "binance" in s: return "Bi"
        if "gecko" in s:   return "Cg"
        return ""
    display["src"] = display.get("source", "").map(_src_short) if "source" in display.columns else ""

    # CMC-URL aus Symbol oder ID ableiten (Fallback: Suche)
    def cmc_url_from_row(row) -> str:
        sym = str(row.get("symbol", "") or "")
        cid = str(row.get("id", "") or "")
        base = ""
        if sym:
            base = sym.upper()
        elif cid.upper().endswith("USDT"):
            base = cid.upper()[:-4]
        else:
            base = cid.upper() or "BTC"
        # Direkte Suche (robust, unabhängig vom CMC-Slug)
        return f"https://coinmarketcap.com/currencies/search/?q={base}"

    display["cmc"] = display.apply(cmc_url_from_row, axis=1)

    if "rank" in display.columns:
        display = display.sort_values("rank", kind="stable").reset_index(drop=True)

    show_cols = ["universe","rank","name","symbol","price","MA20","MA50","Vol_Surge_x",
                 "Resistance","Support","Breakout_MA","Breakout_Resistance","Entry_Signal",
                 "status","src","cmc","id"]
    for c in show_cols:
        if c not in display.columns:
            display[c] = np.nan if c in ["price","MA20","MA50","Vol_Surge_x","Resistance","Support"] else ""

    display = display[show_cols].copy()

    # ---- AG Grid Konfiguration ----
    gb = GridOptionsBuilder.from_dataframe(display)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)

    # Formatter (max 2 Dezimalstellen)
    fmt2 = JsCode("""
        function(params){
            if (params.value == null || isNaN(params.value)) return '';
            return Number(params.value).toLocaleString(undefined,{maximumFractionDigits:2});
        }
    """)
    fmt2x = JsCode("""
        function(params){
            if (params.value == null || isNaN(params.value)) return '';
            return Number(params.value).toLocaleString(undefined,{maximumFractionDigits:2}) + 'x';
        }
    """)

    # Zellfarben: Vol_Surge_x (grün ≥ Schwelle, rot < 0.8)
    vol_thresh_js = float(st.session_state["vol_surge_thresh"])
    cell_style_vol = JsCode(f"""
        function(params) {{
            var v = Number(params.value);
            if (isNaN(v)) return {{}};
            if (v >= {vol_thresh_js}) return {{backgroundColor:'#e6ffed', color:'#065f46', fontWeight:'600'}};
            if (v < 0.80) return {{backgroundColor:'#ffecec', color:'#7f1d1d', fontWeight:'600'}};
            return {{}};
        }}
    """)

    # Zellfarben: Bool (True -> grün, False -> rot)
    cell_style_bool = JsCode("""
        function(params){
            if (params.value === true)  return {backgroundColor:'#e6ffed', color:'#065f46', fontWeight:'600', textAlign:'center'};
            if (params.value === false) return {backgroundColor:'#ffecec', color:'#7f1d1d', fontWeight:'600', textAlign:'center'};
            return {};
        }
    """)

    # CMC-Link Renderer
    cell_renderer_cmc = JsCode("""
        function(params){
            var url = params.value;
            if (!url) return '';
            return "<a href='"+url+"' target='_blank' rel='noopener' style='text-decoration:none;font-weight:600;'>CMC</a>";
        }
    """)

    gb.configure_column("id", hide=True)
    gb.configure_column("universe", headerName="Ursprung", hide=True)
    gb.configure_column("symbol", headerName="Ticker", hide=True)

    gb.configure_column("rank", headerName="Rang", width=90, sort="asc")
    gb.configure_column("name", headerName="Name", width=140)

    gb.configure_column("price", headerName="Price", type=["rightAligned"], valueFormatter=fmt2)
    gb.configure_column("MA20", headerName="MA20", type=["rightAligned"], valueFormatter=fmt2)
    gb.configure_column("MA50", headerName="MA50", type=["rightAligned"], valueFormatter=fmt2)
    gb.configure_column("Resistance", headerName="Resistance", type=["rightAligned"], valueFormatter=fmt2)
    gb.configure_column("Support", headerName="Support", type=["rightAligned"], valueFormatter=fmt2)

    gb.configure_column("Vol_Surge_x", headerName="Vol x7d", type=["rightAligned"], valueFormatter=fmt2x, cellStyle=cell_style_vol)
    gb.configure_column("Breakout_MA", headerName="Breakout MA", cellStyle=cell_style_bool)
    gb.configure_column("Breakout_Resistance", headerName="Breakout Resistance", cellStyle=cell_style_bool)
    gb.configure_column("Entry_Signal", headerName="Entry", cellStyle=cell_style_bool)

    gb.configure_column("status", headerName="Status", width=90)
    gb.configure_column("src", headerName="Daten", width=80)

    gb.configure_column("cmc", headerName="CMC", width=80, cellRenderer=cell_renderer_cmc, sortable=False, filter=False)

    gb.configure_selection(selection_mode="single", use_checkbox=False)
    js_dbl = JsCode("""
        function(e) {
            e.api.forEachNode(function(n){ n.setSelected(false); });
            e.node.setSelected(true);
        }
    """)
    opts = gb.build()
    if "columnDefs" in opts:
        for col in opts["columnDefs"]:
            col["checkboxSelection"] = False
            col["headerCheckboxSelection"] = False
    opts["rowSelection"] = "single"
    opts["suppressRowClickSelection"] = True
    opts["rowMultiSelectWithClick"] = False
    opts["domLayout"] = "autoHeight"
    opts["onRowDoubleClicked"] = js_dbl

    grid = AgGrid(
        display,
        gridOptions=opts,
        theme="balham",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True
    )

    sel = grid.get("selected_rows", [])
    if sel:
        chosen_id = str(sel[0]["id"])
        if st.session_state.get("selected_coin") != chosen_id:
            st.session_state["selected_coin"] = chosen_id
            st.query_params.update({"coin": chosen_id})
            st.rerun()

    if st.button("Aktive Auswahl zurücksetzen"):
        st.session_state["selected_coin"] = None
        if "coin" in st.query_params: del st.query_params["coin"]
        st.rerun()

# ===================== Chart + Tools =====================
st.markdown("---")
active = st.session_state.get("selected_coin")

if not active and not union.empty:
    df_e = union[(union["Entry_Signal"]==True) & (union["status"]=="ok")]
    if not df_e.empty:
        active = str(df_e.iloc[0]["id"])
        st.session_state["selected_coin"] = active

if active:
    name_badge, sym_badge = _name_and_symbol_any(active)
    st.markdown(
        f"<div style='display:inline-block;padding:6px 12px;border-radius:999px;background:#eef6ff;color:#1e3a8a;font-weight:600;'>"
        f"Aktiv: {name_badge} ({sym_badge})</div>",
        unsafe_allow_html=True
    )

if active:
    with st.spinner(f"Lade Historie für {active} …"):
        d = st.session_state.get("history_cache", {}).get(active)
        if d is None or not isinstance(d, pd.DataFrame) or d.empty:
            d = load_history(active, days=st.session_state["days_hist"])
            if isinstance(d, pd.DataFrame) and not d.empty:
                st.session_state.setdefault("history_cache", {})
                st.session_state["history_cache"][active] = d

    status_val = d.attrs.get("status", "") if isinstance(d, pd.DataFrame) else ""
    src_val    = d.attrs.get("source", "")
    if (d is None) or d.empty or (status_val != "ok"):
        st.warning("Keine Historie verfügbar (API-Limit, 451/403 oder leere Daten).")
    else:
        st.caption(f"Datenquelle: {('Bi' if 'binance' in (src_val or '').lower() else 'Cg')}")
        dfd = d.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd["price"]  = pd.to_numeric(dfd["price"], errors="coerce")
        dfd["volume"] = pd.to_numeric(dfd["volume"], errors="coerce")
        dfd = dfd.dropna(subset=["timestamp","price"]).set_index("timestamp").sort_index()
        d_daily = dfd.resample("1D").last().dropna(subset=["price"]) if not dfd.empty else dfd
        if d_daily.empty:
            st.warning("Keine Tagesdaten.")
        else:
            r, s  = calc_local_levels(d_daily, st.session_state["lookback_res"])
            d_daily["ma20"] = ma(d_daily["price"], 20)
            d_daily["ma50"] = ma(d_daily["price"], 50)
            d_daily["vol7"] = d_daily["volume"].rolling(7, min_periods=3).mean()
            d_daily["vol_ratio"] = d_daily["volume"] / d_daily["vol7"]
            d_daily["roll_max_prev"] = d_daily["price"].shift(1).rolling(st.session_state["lookback_res"], min_periods=5).max()

            entry_mask = (
                (d_daily["price"] > d_daily["ma20"]) &
                (d_daily["ma20"] > d_daily["ma50"]) &
                (d_daily["price"] > d_daily["roll_max_prev"]) &
                (d_daily["vol_ratio"] >= st.session_state["vol_surge_thresh"])
            )
            entries = d_daily[entry_mask].dropna(subset=["price"])

            fig, ax_price = plt.subplots()
            ax_vol = ax_price.twinx()

            ax_price.plot(d_daily.index, d_daily["price"], label="Price", linewidth=1.6)
            ax_price.plot(d_daily.index, d_daily["ma20"],  label="MA20", linewidth=1.0)
            ax_price.plot(d_daily.index, d_daily["ma50"],  label="MA50", linewidth=1.0)
            if not np.isnan(r): ax_price.axhline(r, linestyle="--", label=f"Resistance {r:.2f}")
            if not np.isnan(s): ax_price.axhline(s, linestyle="--", label=f"Support {s:.2f}")
            if not entries.empty:
                ax_price.scatter(entries.index, entries["price"].astype(float), s=36, zorder=5, color="#16a34a", label="Entry (hist)")
            ax_vol.bar(d_daily.index, d_daily["volume"], alpha=0.28)
            ax_vol.set_ylabel("Volume")

            locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
            formatter = mdates.ConciseDateFormatter(locator)
            ax_price.xaxis.set_major_locator(locator)
            ax_price.xaxis.set_major_formatter(formatter)

            ax_price.set_title(f"{active} — Price, MAs, Levels & Volume")
            ax_price.set_xlabel("Date"); ax_price.set_ylabel("USD")
            ax_price.legend(loc="upper left")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # === Position & Trailing Stop (2 Dezimalstellen) ===
            st.markdown("### 🧮 Position & Trailing Stop")
            c1, c2, c3, c4 = st.columns(4)
            last_px = float(d_daily['price'].iloc[-1])
            portfolio   = c1.number_input("Portfolio (USD)", min_value=0.0, value=8000.0, step=100.0, format="%.2f", key=f"pos_port_{active}")
            risk_pct    = c2.slider("Risiko/Trade (%)", 0.5, 3.0, 2.0, 0.1, key=f"pos_risk_{active}")
            stop_pct    = c3.slider("Stop-Entfernung (%)", 3.0, 25.0, 8.0, 0.5, key=f"pos_stop_{active}")
            entry_price = c4.number_input("Entry-Preis", min_value=0.0, value=round(last_px,2), step=0.01, format="%.2f", key=f"pos_entry_{active}")

            max_loss = portfolio * (risk_pct/100.0)
            size_usd = max_loss / (stop_pct/100.0) if stop_pct>0 else 0.0
            size_qty = size_usd / entry_price if entry_price>0 else 0.0
            st.write(f"**Max. Verlust:** ${max_loss:,.2f} • **Positionsgröße:** ${size_usd:,.2f} (~ {size_qty:,.2f} {str(active).upper()})")

            st.markdown("#### Trailing Stop")
            t1, t2 = st.columns(2)
            trail_pct = t1.slider("Trail (%)", 5.0, 25.0, 10.0, 0.5, key=f"trail_pct_{active}")
            high_since_entry = t2.number_input("Höchster Kurs seit Entry", min_value=0.0, value=round(last_px,2), step=0.01, format="%.2f", key=f"trail_high_{active}")
            tstop = trailing_stop(high_since_entry, trail_pct)
            st.write(f"Trailing Stop bei **${tstop:,.2f}** (High {high_since_entry:,.2f}, Trail {trail_pct:.1f}%)")
else:
    st.info("Kein aktiver Coin. **Doppelklicke** eine Zeile in der Tabelle, um den Chart zu sehen.")
