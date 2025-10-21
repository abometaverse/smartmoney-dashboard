# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Geschützt (Streamlit)
# v5.9
#  - Persistente Settings & Watchlist via st.query_params
#  - Eine Tabelle (AG Grid) mit Filtern: Top-100, Watchlist, Entry-Signale
#  - Doppelklick aktiviert genau einen Coin (Chart unten)
#  - Binance-First Daten, CG nur für Komfort-Suche/Mapping
#  - Auto-Scan + Telegram-Alerts (optional)
#  - Chart: Price+MA20/MA50+Levels+Volume, grüne Punkte für historische Entry
#  - Dist_Risk in Tabelle: False = grün, True = rot
#  - NEU: "Chg 24h (%)" direkt nach Price, farbig (grün/rot), max. 2 Nachkommastellen
#
# Secrets:
#   APP_PASSWORD       = "...'  (erforderlich)
#   TELEGRAM_BOT_TOKEN = "123:abc"   (optional)
#   TELEGRAM_CHAT_ID   = "123456789" (optional)
#   APP_URL            = "https://deine-app-url.streamlit.app"  (optional)
#
# Requirements (requirements.txt):
#   streamlit
#   st-aggrid
#   matplotlib
#   pandas
#   numpy
#   requests
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
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

# ----------------- App Config -----------------
st.set_page_config(page_title="Smart Money Dashboard — Geschützt", layout="wide")

# ================= Query Params Persistenz =================
def _read_qp() -> dict:
    qp = st.query_params
    out = {}

    def get_bool(k, default=False):
        v = str(qp.get(k, str(default))).lower()
        return v in ("1","true","yes","on")

    def get_float(k, default=0.0):
        try:
            return float(qp.get(k, default))
        except Exception:
            return default

    def get_int(k, default=0):
        try:
            return int(float(qp.get(k, default)))
        except Exception:
            return default

    def get_list(k, default=None):
        v = qp.get(k, "")
        if not v:
            return default or []
        return [x for x in str(v).split(",") if x]

    out["selected_ids"]       = get_list("wl", [])
    out["vol_surge_thresh"]   = get_float("vsth", 1.5)
    out["lookback_res"]       = get_int("lbr", 20)
    out["alerts_enabled"]     = get_bool("alerts", True)
    out["days_hist"]          = get_int("dh", 90)
    out["batch_size_slider"]  = get_int("bs", 3)
    out["scan_index"]         = get_int("scanix", 0)
    out["selected_coin"]      = qp.get("active","") or None
    out["top100_last_sync_ts"]= get_float("t100ts", 0.0)
    out["top100_cooldown_min"]= get_int("t100cd", 15)
    out["auto_scan_enabled"]  = get_bool("auto", False)
    out["auto_scan_hours"]    = get_float("autoh", 1.0)
    return out

def _write_qp():
    # Alles schlank in die URL zurückschreiben (ohne DataFrames)
    qp = {
        "wl": ",".join(st.session_state.get("selected_ids", [])),
        "vsth": st.session_state.get("vol_surge_thresh", 1.5),
        "lbr": st.session_state.get("lookback_res", 20),
        "alerts": str(bool(st.session_state.get("alerts_enabled", True))).lower(),
        "dh": st.session_state.get("days_hist", 90),
        "bs": st.session_state.get("batch_size_slider", 3),
        "scanix": st.session_state.get("scan_index", 0),
        "active": st.session_state.get("selected_coin") or "",
        "t100ts": st.session_state.get("top100_last_sync_ts", 0.0),
        "t100cd": st.session_state.get("top100_cooldown_min", 15),
        "auto": str(bool(st.session_state.get("auto_scan_enabled", False))).lower(),
        "autoh": st.session_state.get("auto_scan_hours", 1.0),
    }
    st.query_params.clear()
    st.query_params.update(**{k:str(v) for k,v in qp.items()})

# ================= Session Defaults =================
PERSIST_KEYS = [
    "selected_ids","vol_surge_thresh","lookback_res","alerts_enabled",
    "days_hist","batch_size_slider","scan_index","selected_coin",
    "top100_last_sync_ts","top100_cooldown_min",
    "auto_scan_enabled","auto_scan_hours","auto_last_ts","auto_alerted_ids",
    "signals_cache","history_cache","top100_df"
]

def ensure_defaults_from_qp():
    # initial mit QueryParams befüllen
    init = _read_qp()
    defaults = {
        "selected_ids": init["selected_ids"],
        "vol_surge_thresh": init["vol_surge_thresh"],
        "lookback_res": init["lookback_res"],
        "alerts_enabled": init["alerts_enabled"],
        "days_hist": init["days_hist"],
        "batch_size_slider": init["batch_size_slider"],
        "scan_index": init["scan_index"],
        "selected_coin": init["selected_coin"],
        "top100_last_sync_ts": init["top100_last_sync_ts"],
        "top100_cooldown_min": init["top100_cooldown_min"],
        "auto_scan_enabled": init["auto_scan_enabled"],
        "auto_scan_hours": init["auto_scan_hours"],
        "auto_last_ts": 0.0,
        "auto_alerted_ids": [],
        "signals_cache": pd.DataFrame(),
        "history_cache": {},
        "top100_df": pd.DataFrame(),
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
        st.error("Konfiguration fehlt: `APP_PASSWORD` in Secrets setzen.")
        st.stop()

    if st.session_state.get("AUTH_OK", False):
        top = st.container()
        c1, c2 = top.columns([6,1])
        c1.success("Zugriff gewährt.")
        if c2.button("Logout"):
            st.session_state["AUTH_OK"] = False
            _write_qp()
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
                ensure_defaults_from_qp()
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    st.stop()

auth_gate()
ensure_defaults_from_qp()

# ================= Constants & HTTP =================
FIAT = "usd"
CG_BASE = "https://api.coingecko.com/api/v3"
BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]

@st.cache_resource(show_spinner=False)
def get_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "smartmoney-dashboard/5.9 (+streamlit)"})
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

# ================= Helpers =================
def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

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
    frames = []
    for base in BINANCE_ENDPOINTS:
        try:
            url = f"{base}/api/v3/ticker/24hr"
            r = get_http().get(url, timeout=8)
            if r.status_code == 200:
                frames.append(pd.DataFrame(r.json())); break
        except Exception:
            time.sleep(0.2)
            continue
    if not frames: return pd.DataFrame()
    return frames[0]

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
    df["id"] = df["symbol"]  # ID = Binance-Symbol
    return df[["rank","id","symbol","name","symbol_txt","quoteVolume"]].rename(columns={"symbol_txt":"symbol2"})

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

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_to_binance_symbol(coin_id_or_symbol: str) -> Optional[str]:
    """CG-ID ('render-token'), BaseAsset ('RNDR') oder BINANCE-Symbol ('RNDRUSDT')"""
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

@st.cache_data(ttl=1200, show_spinner=False)
def load_history(coin_or_symbol: str, days: int = 180) -> pd.DataFrame:
    def _empty(status_text: str, source: str = "") -> pd.DataFrame:
        df = pd.DataFrame(columns=["timestamp","price","volume"])
        df.attrs["status"] = status_text
        if source: df.attrs["source"] = source
        return df

    sym = resolve_to_binance_symbol(coin_or_symbol) or coin_or_symbol
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
    for _, r in df[(df["Entry_Signal"] == True) & (df["status"]=="ok")].iterrows():
        cid = str(r["id"])
        if cid in sent_before:
            continue
        name = str(r.get("name", cid))
        sym  = str(r.get("symbol", "")) or cid
        link = f"\n{app_url}" if app_url else ""
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

days_hist = st.sidebar.slider(
    "Historie (Tage)", 60, 365,
    int(st.session_state.get("days_hist", 90)), 15, key="days_hist"
)
vol_surge_thresh = st.sidebar.slider(
    "Vol Surge vs 7d (x)", 1.0, 5.0,
    float(st.session_state.get("vol_surge_thresh", 1.5)), 0.1, key="vol_surge"
)
lookback_res = st.sidebar.slider(
    "Lookback für Widerstand/Support (Tage)", 10, 60,
    int(st.session_state.get("lookback_res", 20)), 1, key="lookback"
)

st.session_state["top100_cooldown_min"] = st.sidebar.number_input(
    "Top-100 Refresh-Sperre (Minuten)", min_value=1, max_value=120,
    value=int(st.session_state.get("top100_cooldown_min", 15)), step=1, key="cooldown_min"
)

c_as1, c_as2 = st.sidebar.columns([1,1])
st.session_state["auto_scan_enabled"] = c_as1.checkbox(
    "Auto-Scan & Telegram",
    value=bool(st.session_state.get("auto_scan_enabled", False)), key="auto_enabled"
)
st.session_state["auto_scan_hours"]   = c_as2.number_input(
    "Intervall (Std.)", min_value=0.5, max_value=24.0, step=0.5,
    value=float(st.session_state.get("auto_scan_hours", 1.0)), key="auto_hours"
)

# Watchlist aus CG für Komfort-Suche (nur Textliste, Mapping später)
top_df = cg_top_coins(limit=500)
if top_df.empty:
    st.sidebar.info("Top-500 (CG) derzeit nicht verfügbar.")
else:
    top_df["label"] = top_df.apply(lambda r: f"{r['name']} ({str(r['symbol']).upper()}) — {r['id']}", axis=1)
    if st.sidebar.checkbox("Watchlist-Suche (CG) einblenden", value=False, key="wl_search"):
        default_ids = st.session_state.get("selected_ids", [])
        default_labels = top_df[top_df["id"].isin(default_ids)]["label"].tolist()
        selected_labels = st.sidebar.multiselect(
            "Watchlist (Top 500, Suche per Tippen)",
            options=top_df["label"].tolist(),
            default=default_labels,
            key="watchlist_top"
        )
        label_to_id = dict(zip(top_df["label"], top_df["id"]))
        wl_ids = [label_to_id.get(l, l) for l in selected_labels]
        st.session_state["selected_ids"] = wl_ids

# ================= Utility: Name/Symbol =================
def _name_and_symbol_any(coin_id_or_symbol: str) -> Tuple[str,str]:
    sym = resolve_to_binance_symbol(coin_id_or_symbol) or coin_id_or_symbol
    base_guess = None
    if sym and sym.upper().endswith("USDT"):
        base_guess = sym[:-4].upper()
    try:
        if isinstance(top_df, pd.DataFrame) and not top_df.empty and coin_id_or_symbol in set(top_df["id"]):
            row = top_df[top_df["id"]==coin_id_or_symbol].iloc[0]
            return str(row["name"]), str(row["symbol"]).upper()
    except Exception:
        pass
    if base_guess:
        return base_guess, base_guess
    s = str(coin_id_or_symbol).upper()
    if s.endswith("USDT"): s = s[:-4]
    return s, s

# ================= Scan/Compute =================
if "signals_cache" not in st.session_state:
    st.session_state["signals_cache"] = pd.DataFrame()
if "history_cache" not in st.session_state:
    st.session_state["history_cache"] = {}

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
                "universe": "",
                "rank": i, "name": name, "symbol": symbol,
                "id": resolve_to_binance_symbol(cid) or cid,
                "price": np.nan, "Chg_24h_pct": np.nan,
                "MA20": np.nan, "MA50": np.nan,
                "Vol_Surge_x": np.nan, "Breakout_MA": False,
                "Resistance": np.nan, "Support": np.nan,
                "Breakout_Resistance": False, "Dist_Risk": False,
                "Entry_Signal": False, "status": status_val or "no data",
                "src": hist.attrs.get("source","") if isinstance(hist,pd.DataFrame) else ""
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

        # ---- NEU: 24h Change in % (gegenüber Vortag) ----
        chg_24h = np.nan
        if len(dfd) >= 2:
            prev = float(dfd["price"].iloc[-2])
            if prev > 0:
                chg_24h = (price / prev - 1.0) * 100.0

        volsurge = v_sig["vol_ratio_1d_vs_7d"]; is_valid_vol = not np.isnan(volsurge)
        breakout_res = bool(price >= (resistance * 1.0005)) if not math.isnan(resistance) else False
        entry_ok = bool(t_sig["breakout_ma"] and is_valid_vol and (volsurge >= vol_thresh))
        dist_risk = bool(v_sig["distribution_risk"])

        rows.append({
            "universe": "",
            "rank": i, "name": name, "symbol": symbol,
            "id": resolve_to_binance_symbol(cid) or cid,
            "price": price, "Chg_24h_pct": chg_24h,
            "MA20": t_sig["ma20"], "MA50": t_sig["ma50"],
            "Vol_Surge_x": volsurge, "Breakout_MA": t_sig["breakout_ma"],
            "Resistance": resistance, "Support": support,
            "Breakout_Resistance": breakout_res,
            "Dist_Risk": dist_risk,
            "Entry_Signal": entry_ok and breakout_res,
            "status": "ok", "src": hist.attrs.get("source","")
        })
        prog.progress(min(i/total, 1.0), text=f"{progress_label} ({i}/{total})")

    prog.progress(1.0, text=f"{progress_label} (fertig)")
    note.empty()
    df = pd.DataFrame(rows)
    df = df.sort_values("rank", kind="stable").reset_index(drop=True)
    return df, history_cache

def run_scan_ids(ids: List[str], label: str):
    info_ph = st.empty()
    info_ph.info(f"⏳ {label} …")
    df, cache = compute_rows_for_ids(ids, st.session_state["days_hist"], st.session_state["vol_surge_thresh"], st.session_state["lookback_res"], label)
    info_ph.empty()
    return df, cache

# ================= Top-100 Laden & Caches =================
def load_top100_scan() -> pd.DataFrame:
    top = binance_top100_by_quote_volume()
    if top.empty:
        st.warning("Binance Top-100 nicht verfügbar. Fallback BTC/ETH/SOL.")
        top = pd.DataFrame({
            "rank":[1,2,3],
            "id":["BTCUSDT","ETHUSDT","SOLUSDT"],
            "symbol":["BTCUSDT","ETHUSDT","SOLUSDT"],
            "name":["BTC","ETH","SOL"],
            "symbol2":["BTC","ETH","SOL"],
            "quoteVolume":[0,0,0]
        })
    ids = top["id"].tolist()
    names = {r["id"]:(str(r["name"]), str(r.get("symbol2", r["name"]))) for _, r in top.iterrows()}
    df100, cache = compute_rows_for_ids(ids, st.session_state["days_hist"], st.session_state["vol_surge_thresh"], st.session_state["lookback_res"], "Top-100-Scan")
    if df100.empty:
        return df100
    df100["rank"] = df100["id"].map(lambda x: int(top[top["id"]==x]["rank"].iloc[0]) if x in set(top["id"]) and not top[top["id"]==x].empty else 999)
    df100["name"]   = df100["id"].map(lambda x: names.get(x, (x,x))[0])
    df100["symbol"] = df100["id"].map(lambda x: names.get(x, (x,x))[1])
    st.session_state["history_cache"].update(cache)
    return df100.sort_values("rank", kind="stable").reset_index(drop=True)

# ================= UI: Controls =================
c1, c2, c3 = st.columns([1.2,1.2,2])
with c1:
    refresh_top_btn = st.button("🔄 Top-100 aktualisieren", key="top100_refresh")
with c2:
    last_sync_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.get("top100_last_sync_ts", 0.0))) if st.session_state.get("top100_last_sync_ts", 0.0) else "—"
    st.write(f"Letzte Sync: {last_sync_str}")
with c3:
    st.caption("Tabelle filtern: Top-100 • Watchlist • Entry-Signale")

top_filter = st.radio("Filter", ["Top-100", "Watchlist", "Entry-Signale"], index=0, horizontal=True, key="top100_filter")

now = time.time()
cooldown = int(st.session_state.get("top100_cooldown_min", 15)) * 60
should_refresh = refresh_top_btn and (now - st.session_state.get("top100_last_sync_ts", 0.0) >= cooldown)

if refresh_top_btn and not should_refresh:
    wait_left = int((st.session_state["top100_last_sync_ts"] + cooldown - now) / 60) + 1
    st.warning(f"Top-100 Refresh-Sperre aktiv. Bitte in ~{max(wait_left,1)} Min. erneut versuchen.")

if should_refresh or st.session_state.get("top100_df", pd.DataFrame()).empty:
    df100_new = load_top100_scan()
    if not df100_new.empty:
        st.session_state["top100_df"] = df100_new
        st.session_state["top100_last_sync_ts"] = time.time()

# ================= Tabelle (AG Grid) =================
df_all = st.session_state.get("top100_df", pd.DataFrame()).copy()

# Watchlist mapping: IDs können CG-IDs/BASE/BINANCE sein → auf Binance-Symbol abbilden
wl_raw = st.session_state.get("selected_ids", [])
wl_mapped = list(filter(None, [resolve_to_binance_symbol(x) or x for x in wl_raw]))

if top_filter == "Watchlist":
    if not df_all.empty:
        df_view = df_all[df_all["id"].isin(wl_mapped)].copy()
    else:
        # Falls Top100 leer, scanne gezielt Watchlist
        df_view, cache = run_scan_ids(wl_mapped, "Watchlist-Scan")
        st.session_state["history_cache"].update(cache)
else:
    df_view = df_all.copy()

if top_filter == "Entry-Signale" and not df_view.empty:
    df_view = df_view[(df_view["Entry_Signal"] == True) & (df_view["status"]=="ok")].copy()

# Anzeigeformat & Hilfsspalten
if not df_view.empty:
    for c in ["price","MA20","MA50","Resistance","Support","Vol_Surge_x","Chg_24h_pct"]:
        if c in df_view.columns:
            df_view[c] = pd.to_numeric(df_view[c], errors="coerce")

    # kurze Datenquelle (Bi/Cg)
    df_view["dataSrc"] = df_view["src"].map(lambda s: "Bi" if str(s).lower().startswith("bina") or s=="" else "Cg")

    # CoinMarketCap-Link
    def _cmc_url_from_row(r):
        base = str(r.get("id",""))
        if base.upper().endswith("USDT"): base = base[:-4]
        return f"https://coinmarketcap.com/currencies/{base.lower()}/"
    df_view["cmc"] = df_view.apply(_cmc_url_from_row, axis=1)

    # Spaltenreihenfolge (Chg_24h_pct direkt nach price)
    show_cols = [
        "universe","rank","name","symbol",
        "price","Chg_24h_pct",
        "MA20","MA50","Vol_Surge_x","Resistance","Support",
        "Breakout_MA","Breakout_Resistance","Dist_Risk","Entry_Signal",
        "status","dataSrc","cmc","id"
    ]
    for c in show_cols:
        if c not in df_view.columns:
            df_view[c] = np.nan

    gb = GridOptionsBuilder.from_dataframe(df_view[show_cols])

    # Nummern-Format (2 Nachkommastellen), Bool-Farben, Dist_Risk invertiert
    fmt_2 = JsCode("""
        function(params){
            if (params.value == null || isNaN(params.value)) return '';
            return Number(params.value).toLocaleString(undefined,{maximumFractionDigits:2});
        }
    """)
    fmt_pct = JsCode("""
        function(params){
            if (params.value == null || isNaN(params.value)) return '';
            var v = Number(params.value);
            return v.toLocaleString(undefined,{maximumFractionDigits:2}) + '%';
        }
    """)
    cell_style_pct = JsCode("""
        function(params){
            var v = Number(params.value);
            if (isNaN(v)) return {};
            if (v > 0)   return {color:'#065f46', fontWeight:'600'};
            if (v < 0)   return {color:'#7f1d1d', fontWeight:'600'};
            return {};
        }
    """)
    cell_style_num = JsCode("""
        function(params){
            if (params.value == null || isNaN(params.value)) return {};
            return {};
        }
    """)
    cell_style_vol = JsCode("""
        function(params){
            var v = Number(params.value);
            if (isNaN(v)) return {};
            if (v >= %f) return {backgroundColor:'#e6ffed', color:'#065f46', fontWeight:'600'};
            if (v < 0.8) return {backgroundColor:'#ffecec', color:'#7f1d1d', fontWeight:'600'};
            return {};
        }
    """ % float(st.session_state.get("vol_surge_thresh", 1.5)))
    cell_style_bool = JsCode("""
        function(params){
            if (params.value === true)  return {backgroundColor:'#e6ffed', color:'#065f46', fontWeight:'600', textAlign:'center'};
            if (params.value === false) return {backgroundColor:'#ffecec', color:'#7f1d1d', fontWeight:'600', textAlign:'center'};
            return {};
        }
    """)
    cell_style_dist = JsCode("""
        function(params){
            if (params.value === false) return {backgroundColor:'#e6ffed', color:'#065f46', fontWeight:'600', textAlign:'center'};
            if (params.value === true)  return {backgroundColor:'#ffecec', color:'#7f1d1d', fontWeight:'600', textAlign:'center'};
            return {};
        }
    """)
    # CMC Link (klickbar)
    cell_renderer_link = JsCode("""
        class UrlCellRenderer {
          init(params) {
            this.eGui = document.createElement('a');
            this.eGui.innerText = 'CMC';
            this.eGui.setAttribute('href', params.value || '#');
            this.eGui.setAttribute('target', '_blank');
            this.eGui.style.fontWeight = '600';
          }
          getGui() { return this.eGui; }
          refresh() { return false; }
        }
    """)

    # Spalten konfigurieren
    gb.configure_column("universe", headerName="", width=40)
    gb.configure_column("rank", headerName="Rang", width=80, sortable=True)
    gb.configure_column("name", headerName="Name", width=160)
    gb.configure_column("symbol", headerName="Symbol", width=100)
    gb.configure_column("price", headerName="Price", type=["rightAligned"], valueFormatter=fmt_2, cellStyle=cell_style_num, width=120)
    gb.configure_column("Chg_24h_pct", headerName="Chg 24h (%)", type=["rightAligned"], valueFormatter=fmt_pct, cellStyle=cell_style_pct, width=120)
    gb.configure_column("MA20", headerName="MA20", type=["rightAligned"], valueFormatter=fmt_2, cellStyle=cell_style_num, width=110)
    gb.configure_column("MA50", headerName="MA50", type=["rightAligned"], valueFormatter=fmt_2, cellStyle=cell_style_num, width=110)
    gb.configure_column("Vol_Surge_x", headerName="Vol x7d", type=["rightAligned"], valueFormatter=fmt_2, cellStyle=cell_style_vol, width=110)
    gb.configure_column("Resistance", headerName="Resistance", type=["rightAligned"], valueFormatter=fmt_2, cellStyle=cell_style_num, width=130)
    gb.configure_column("Support", headerName="Support", type=["rightAligned"], valueFormatter=fmt_2, cellStyle=cell_style_num, width=130)
    gb.configure_column("Breakout_MA", headerName="BO MA", cellStyle=cell_style_bool, width=100)
    gb.configure_column("Breakout_Resistance", headerName="BO Res", cellStyle=cell_style_bool, width=110)
    gb.configure_column("Dist_Risk", headerName="Dist_Risk", cellStyle=cell_style_dist, width=110)
    gb.configure_column("Entry_Signal", headerName="Entry", cellStyle=cell_style_bool, width=100)
    gb.configure_column("status", headerName="Status", width=110)
    gb.configure_column("dataSrc", headerName="Src", width=70)
    gb.configure_column("cmc", headerName="CMC", cellRenderer=cell_renderer_link, width=70)
    gb.configure_column("id", headerName="ID", hide=True)

    gb.configure_grid_options(
        rowSelection="single",
        suppressRowClickSelection=False,
        animateRows=True,
        ensureDomOrder=True
    )

    grid = AgGrid(
        df_view[show_cols],
        gridOptions=gb.build(),
        height=420,                      # Fixe Höhe -> Header bleibt oben beim Scrollen
        theme="balham",
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True
    )

    # SELECTION: ein Coin aktiv
    sel = grid.get("selected_rows", [])
    if sel:
        sel_id = sel[0].get("id")
        if sel_id:
            set_active_coin(sel_id)

    st.caption("**Legende:** Vol x7d ≥ Schwelle → grün; < 0.8 → rot • BO MA/BO Res/Entry: True grün, False rot • Dist_Risk: False grün, True rot • Src: Bi=Binance, Cg=CoinGecko")

    # Buttons zum Watchlist-Management
    add_col1, add_col2 = st.columns([1,1])
    # Hinzufügen
    if sel and add_col1.button(f"➕ {sel[0].get('name','?')} zur Watchlist", key="btn_add_wl"):
        sid = sel[0].get("id")
        # zurück mappen auf CG-ID/BASE? -> wir speichern Binance-Symbol stabil
        if sid and sid not in st.session_state["selected_ids"]:
            st.session_state["selected_ids"].append(sid)
            _write_qp()
            st.success("Zur Watchlist hinzugefügt.")
            st.rerun()
    # Entfernen
    if sel and add_col2.button(f"🗑 {sel[0].get('name','?')} aus Watchlist", key="btn_del_wl"):
        sid = sel[0].get("id")
        if sid in st.session_state["selected_ids"]:
            st.session_state["selected_ids"] = [x for x in st.session_state["selected_ids"] if x != sid]
            _write_qp()
            st.success("Aus Watchlist entfernt.")
            st.rerun()

else:
    st.info("Noch keine Daten. Klicke auf „Top-100 aktualisieren“ oder aktiviere Watchlist-Scan.")

# ================= Auto-Scan Scheduler (Top-100 + Telegram) =================
auto_info_ph = st.empty()
if st.session_state.get("auto_scan_enabled", False):
    last = float(st.session_state.get("auto_last_ts", 0.0) or 0.0)
    interval = float(st.session_state.get("auto_scan_hours", 1.0)) * 3600.0
    if time.time() - last >= interval:
        auto_info_ph.info("⏳ Auto-Scan läuft …")
        df100_new = load_top100_scan()
        if not df100_new.empty:
            st.session_state["top100_df"] = df100_new
            st.session_state["top100_last_sync_ts"] = time.time()
            if st.session_state.get("alerts_enabled", True):
                alerted_ids = telegram_alert_for_entries(df100_new)
                if alerted_ids:
                    st.success(f"Telegram-Alerts gesendet: {', '.join(alerted_ids)}")
        st.session_state["auto_last_ts"] = time.time()
        auto_info_ph.empty()

# ================= Badge aktiver Coin & Einzel-Chart =================
st.markdown("---")
active = st.session_state.get("selected_coin")

# Fallback-Logik, falls keiner aktiv: erstes Entry aus Top100 oder erster Watchlist-Treffer
if not active:
    df100 = st.session_state.get("top100_df", pd.DataFrame())
    if not df100.empty:
        df_e = df100[(df100["Entry_Signal"]==True) & (df_e["status"]=="ok")] if "df_e" in locals() else df100[(df100["Entry_Signal"]==True) & (df100["status"]=="ok")]
        if not df_e.empty:
            active = str(df_e.iloc[0]["id"])
            st.session_state["selected_coin"] = active
    if not active and st.session_state.get("selected_ids"):
        for x in st.session_state["selected_ids"]:
            m = resolve_to_binance_symbol(x) or x
            if m:
                active = m
                st.session_state["selected_coin"] = active
                break

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
        st.caption(f"Datenquelle: {'binance' if (not src_val) else src_val}")
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
            d_daily["entry_flag"] = entry_mask
            entries = d_daily[entry_mask].dropna(subset=["price"])

            fig, ax_price = plt.subplots()
            ax_vol = ax_price.twinx()

            ax_price.plot(d_daily.index, d_daily["price"], label="Price", linewidth=1.6)
            ax_price.plot(d_daily.index, d_daily["ma20"],  label="MA20", linewidth=1.0)
            ax_price.plot(d_daily.index, d_daily["ma50"],  label="MA50", linewidth=1.0)
            if not np.isnan(r): ax_price.axhline(r, linestyle="--", label=f"Resistance {r:.3f}")
            if not np.isnan(s): ax_price.axhline(s, linestyle="--", label=f"Support {s:.3f}")
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

            # === Position & Trailing Stop ===
            st.markdown("### 🧮 Position & Trailing Stop")
            c1, c2, c3, c4 = st.columns(4)
            last_px = float(d_daily['price'].iloc[-1])
            portfolio   = c1.number_input("Portfolio (USD)", min_value=0.0, value=8000.0, step=100.0, format="%.2f", key=f"pos_port_{active}")
            risk_pct    = c2.slider("Risiko/Trade (%)", 0.5, 3.0, 2.0, 0.1, key=f"pos_risk_{active}")
            stop_pct    = c3.slider("Stop-Entfernung (%)", 3.0, 25.0, 8.0, 0.5, key=f"pos_stop_{active}")
            entry_price = c4.number_input("Entry-Preis", min_value=0.0, value=last_px, step=0.001, format="%.6f", key=f"pos_entry_{active}")

            max_loss = portfolio * (risk_pct/100.0)
            size_usd = max_loss / (stop_pct/100.0) if stop_pct>0 else 0.0
            size_qty = size_usd / entry_price if entry_price>0 else 0.0
            st.write(f"**Max. Verlust:** ${max_loss:,.2f} • **Positionsgröße:** ${size_usd:,.2f} (~ {size_qty:,.4f} {str(active).upper()})")

            st.markdown("#### Trailing Stop")
            t1, t2 = st.columns(2)
            trail_pct = t1.slider("Trail (%)", 5.0, 25.0, 10.0, 0.5, key=f"trail_pct_{active}")
            high_since_entry = t2.number_input("Höchster Kurs seit Entry", min_value=0.0, value=last_px, step=0.001, format="%.6f", key=f"trail_high_{active}")
            tstop = trailing_stop(high_since_entry, trail_pct)
            st.write(f"Trailing Stop bei **${tstop:,.3f}** (High {high_since_entry:,.3f}, Trail {trail_pct:.1f}%)")

# ================= Schluss: QueryParams updaten =================
_write_qp()
