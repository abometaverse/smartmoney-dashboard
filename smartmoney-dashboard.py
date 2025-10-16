# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Geschützt (Streamlit)
# v2.6: Performance + Stabilität + Komfort
#  - Compute-on-Click (Batch) + Full-Scan mit Fortschrittsbalken
#  - Top-100-Scanner (Entry-Setup)
#  - Konsolidierte Data-Source (cg/binance) + robuster Parser
#  - Geldwerte mit Trennzeichen + kompakte Abkürzungen (K/M/B/T)
#  - Keine Nutzung von st.experimental_get_query_params
#  - Detail-Panel robust (Resample-Fallback)
#
# Secrets (Streamlit → Advanced settings → Secrets):
# APP_PASSWORD = "DeinStarkesPasswort"
# TELEGRAM_BOT_TOKEN = "123:abc"   # optional
# TELEGRAM_CHAT_ID   = "123456789" # optional
# -------------------------------------------------------------

import math
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple, Dict, List

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
CG_BASE = "https://api.coingecko.com/api/v3"

@st.cache_resource(show_spinner=False)
def get_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "smartmoney-dashboard/2.6 (+streamlit)"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def _get_json(url, params=None, timeout=12, retries=2, backoff=1.6) -> Dict:
    """HTTP GET mit kurzen Timeouts (fail-fast) + minimalen Retries."""
    session = get_http()
    last_err = ""
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return {"ok": True, "json": r.json(), "status": 200}
            if r.status_code in (429, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (i + 1))
                continue
            return {"ok": False, "json": None, "status": r.status_code, "error": r.text[:300]}
        except requests.RequestException as e:
            last_err = str(e)[:200]
            time.sleep(backoff * (i + 1))
            continue
    return {"ok": False, "json": None, "status": None, "error": last_err or "request failed"}

# ================= Formatting =================
def human_abbr(n: float) -> str:
    """Abkürzen (K/M/B/T) mit Tausendertrennzeichen; robust bei NaN."""
    if n is None or (isinstance(n, float) and math.isnan(n)): return ""
    sign = "-" if n < 0 else ""
    x = abs(float(n))
    if x >= 1_000_000_000_000:
        return f"{sign}{x/1_000_000_000_000:.2f}T"
    if x >= 1_000_000_000:
        return f"{sign}{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{sign}{x:,.0f}"
    return f"{sign}{x:.2f}"

def fmt_money(n: float, decimals: int = 2) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)): return ""
    return f"{n:,.{decimals}f}"

# ================= Analytics Helpers =================
def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

@st.cache_data(ttl=3600, show_spinner=True)
def cg_top_coins(limit: int = 500) -> pd.DataFrame:
    rows = []
    per_page = 250
    pages = int(np.ceil(limit / per_page))
    for page in range(1, pages + 1):
        resp = _get_json(
            f"{CG_BASE}/coins/markets",
            {"vs_currency": FIAT, "order": "market_cap_desc", "per_page": per_page, "page": page, "sparkline": "false"},
        )
        if not resp.get("ok"):
            break
        part = pd.DataFrame(resp["json"])[["id", "symbol", "name", "market_cap"]]
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["id","symbol","name","market_cap"])
    df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["id"]).head(limit)
    return df

@st.cache_data(ttl=1200, show_spinner=False)
def cg_market_chart(coin_id: str, days: int = 180) -> pd.DataFrame:
    """
    Historische Preise/Volumen — CoinGecko (kurz) → Fallback Binance (mehrere Hosts).
    Setzt df.attrs["status"] = "ok" und df.attrs["source"] = "cg"|"binance".
    """
    def _empty(status_text: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=["timestamp", "price", "volume"])
        df.attrs["status"] = status_text
        return df

    # --- 1) CoinGecko (schnell) ---
    resp = _get_json(
        f"{CG_BASE}/coins/{coin_id}/market_chart",
        {"vs_currency": FIAT, "days": days, "interval": "daily"},
        timeout=10, retries=1, backoff=1.2
    )
    if resp.get("ok"):
        data = resp["json"]
        prices = data.get("prices", [])
        vols   = data.get("total_volumes", [])
        if prices:
            dfp = pd.DataFrame(prices, columns=["ts","price"])
            dfv = pd.DataFrame(vols,    columns=["ts","volume"])
            df  = dfp.merge(dfv, on="ts", how="left")
            df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
            df["price"]     = pd.to_numeric(df["price"], errors="coerce")
            df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
            df = df[["timestamp","price","volume"]].dropna()
            if not df.empty:
                df = df.sort_values("timestamp").tail(int(days)+1)
                df.attrs["status"] = "ok"
                df.attrs["source"] = "cg"
                return df
        # sonst Fallback

    # --- 2) Binance Fallback ---
    symbol_map = {
        "bitcoin": "BTCUSDT",
        "ethereum": "ETHUSDT",
        "solana": "SOLUSDT",
        "arbitrum": "ARBUSDT",
        "render-token": "RNDRUSDT",
        "bittensor": "TAOUSDT",
    }
    sym = symbol_map.get(coin_id)
    if not sym:
        try:
            top = cg_top_coins(limit=500)
            sym = top.loc[top["id"] == coin_id, "symbol"].str.upper().iloc[0] + "USDT"
        except Exception:
            return _empty("no_symbol")

    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]
    kl = None; last_status = None
    for base in endpoints:
        try:
            r = get_http().get(
                base,
                params={"symbol": sym, "interval": "1d", "limit": min(1000, int(days)+5)},
                timeout=8
            )
            last_status = r.status_code
            if r.status_code != 200:
                time.sleep(0.2)
                continue
            data = r.json()
            # robust: Liste von Listen erwartet
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
                kl = data
                break
        except Exception:
            time.sleep(0.2)
            continue
    if kl is None:
        return _empty(f"err_binance:{last_status}")

    try:
        df = pd.DataFrame(kl, columns=[
            "openTime","open","high","low","close","volume",
            "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True, errors="coerce")
        df["price"]     = pd.to_numeric(df["close"], errors="coerce")
        df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
        df = df[["timestamp","price","volume"]].dropna()
        if df.empty:
            return _empty("empty_binance")
        df = df.sort_values("timestamp").tail(int(days)+1)
        df.attrs["status"] = "ok"
        df.attrs["source"] = "binance"
        return df
    except Exception:
        return _empty("parse_binance")

@st.cache_data(ttl=600, show_spinner=False)
def cg_simple_price(ids: List[str]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    resp = _get_json(
        f"{CG_BASE}/coins/markets",
        {"vs_currency": FIAT, "ids": ",".join(ids), "order":"market_cap_desc",
         "per_page": max(1,len(ids)), "page":1, "sparkline":"false"}
    )
    if not resp.get("ok"):
        return pd.DataFrame()
    df = pd.DataFrame(resp["json"])
    cols = ["id","symbol","name","current_price","market_cap","total_volume","price_change_percentage_24h"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df[cols].rename(columns={"current_price":"price","total_volume":"volume_24h"})

def calc_local_levels(dfd: pd.DataFrame, lookback: int = 20) -> Tuple[float,float]:
    if dfd.empty:
        return (np.nan, np.nan)
    d = dfd.iloc[:-1].tail(lookback)
    if d.empty:
        return (np.nan, np.nan)
    return float(d["price"].max()), float(d["price"].min())

def volume_signals(dfd: pd.DataFrame) -> dict:
    out = {"vol_ratio_1d_vs_7d": np.nan, "distribution_risk": False, "price_chg_7d": np.nan}
    if dfd.empty or len(dfd) < 8:
        return out
    last = dfd.iloc[-1]
    avg7 = dfd["volume"].iloc[-8:-1].mean()
    vr = float(last["volume"]/avg7) if (avg7 and avg7 == avg7) else np.nan
    out["vol_ratio_1d_vs_7d"] = vr
    p7 = dfd["price"].iloc[-8]
    out["price_chg_7d"] = float((last["price"]/p7)-1.0) if p7 else np.nan
    out["distribution_risk"] = bool((out["price_chg_7d"] > 0) and (vr < 0.8))
    return out

def trend_signals(dfd: pd.DataFrame) -> dict:
    out = {"ma20": np.nan, "ma50": np.nan, "breakout_ma": False}
    if dfd.empty:
        return out
    df = dfd.copy()
    df["ma20"] = ma(df["price"], 20)
    df["ma50"] = ma(df["price"], 50)
    last = df.iloc[-1]
    out["ma20"], out["ma50"] = float(last["ma20"]), float(last["ma50"])
    out["breakout_ma"] = bool(last["price"] > last["ma20"] > last["ma50"])
    return out

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

# Watchlist
top_df = cg_top_coins(limit=500)
if top_df.empty:
    st.sidebar.warning("Top-Liste konnte nicht geladen werden (API-Limit?). Fallback-Auswahl.")
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
if manual.strip():
    selected_ids.append(manual.strip())
if not selected_ids:
    selected_ids = ["bitcoin","ethereum"]

min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=int(st.session_state["min_mktcap"]), step=50_000_000, key="min_mc")
min_volume = st.sidebar.number_input("Min 24h Volume (USD)", min_value=0, value=int(st.session_state["min_volume"]), step=10_000_000, key="min_vol")
vol_surge_thresh = st.sidebar.slider("Vol Surge vs 7d (x)", 1.0, 5.0, float(st.session_state["vol_surge_thresh"]), 0.1, key="vol_surge")
lookback_res = st.sidebar.slider("Lookback für Widerstand/Support (Tage)", 10, 60, int(st.session_state["lookback_res"]), 1, key="lookback")
alerts_enabled = st.sidebar.checkbox("Telegram-Alerts aktivieren (Secrets nötig)", value=bool(st.session_state["alerts_enabled"]), key="alerts_on")

# Scan-Steuerung
scan_now_batch = st.sidebar.button("🔔 Watchlist BATCH scannen", key="scan_btn_batch")
scan_now_full  = st.sidebar.button("🔁 Ganze Watchlist scannen", key="scan_btn_full")
scan_top100    = st.sidebar.button("🔍 Top 100 nach Setup scannen", key="scan_top100")

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

st.caption("🔒 Passwortschutz aktiv — Alerts via Telegram (optional).  •  Scans: Batch oder vollständige Watchlist mit Fortschrittsbalken.")

# ================= Checklist =================
with st.expander("📋 Tägliche Checkliste", expanded=False):
    st.markdown("""
**Morgens:** Scan → Kandidaten (Breakout + Volumen) notieren  
**Mittags:** Entry nur bei Preis > MA20 > MA50 **und** Vol-Surge ≥ Schwelle **und** Breakout über Widerstand  
**Abends:** Volumen-Trend prüfen, Trailing Stop nachziehen, Teilgewinne sichern
""")

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

    # Geldwerte formatieren
    if not disp.empty:
        disp["Price"]   = disp["Price"].apply(lambda x: fmt_money(x, 4))
        disp["MktCap"]  = disp["MktCap"].apply(human_abbr)
        disp["Vol 24h"] = disp["Vol 24h"].apply(human_abbr)
        disp["% 24h"]   = pd.to_numeric(disp["% 24h"], errors="coerce").map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    st.dataframe(disp, use_container_width=True, hide_index=True)

# ================= Signals (Batch / Full) =================
if "signals_cache" not in st.session_state:
    st.session_state["signals_cache"] = pd.DataFrame()
if "history_cache" not in st.session_state:
    st.session_state["history_cache"] = {}

def compute_rows_for_ids(id_list: List[str], days_hist: int, vol_thresh: float, lookback: int,
                         progress_label: str = "Scanne …") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows, history_cache = [], {}
    total = len(id_list)
    prog  = st.progress(0, text=progress_label)
    PAUSE_BETWEEN = 0.55

    for i, cid in enumerate(id_list, start=1):
        time.sleep(PAUSE_BETWEEN)
        hist = cg_market_chart(cid, days=days_hist)
        status_val = hist.attrs.get("status", "") if isinstance(hist, pd.DataFrame) else "no_df"

        if (hist is None) or hist.empty or (status_val != "ok"):
            rows.append({
                "id": cid, "price": np.nan, "MA20": np.nan, "MA50": np.nan,
                "Breakout_MA": False, "Vol_Surge_x": np.nan,
                "Resistance": np.nan, "Support": np.nan,
                "Breakout_Resistance": False, "Distribution_Risk": False,
                "Entry_Signal": False, "status": status_val or "no data",
                "source": status_val
            })
            prog.progress(min(i/total, 1.0), text=f"{progress_label} ({i}/{total})")
            continue

        history_cache[cid] = hist
        dfd = hist.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd["price"]  = pd.to_numeric(dfd["price"], errors="coerce")
        dfd["volume"] = pd.to_numeric(dfd["volume"], errors="coerce")
        dfd = dfd.dropna(subset=["timestamp","price"])
        dfd = (dfd.set_index("timestamp").sort_index().resample("1D").last().dropna(subset=["price"])
               if not dfd.empty else dfd)

        t_sig = trend_signals(dfd)
        v_sig = volume_signals(dfd)
        resistance, support = calc_local_levels(dfd, lookback=lookback)

        last = dfd.iloc[-1]
        price = float(last["price"])
        volsurge = v_sig["vol_ratio_1d_vs_7d"]
        is_valid_vol = not np.isnan(volsurge)
        breakout_res = bool(price >= (resistance * 1.0005)) if not math.isnan(resistance) else False
        entry_ok = bool(t_sig["breakout_ma"] and is_valid_vol and (volsurge >= vol_thresh))

        rows.append({
            "id": cid, "price": price,
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

# Ausführung
if scan_now_batch:
    sig, hist_cache = run_scan_batch()
    st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(hist_cache)

elif scan_now_full:
    sig, hist_cache = run_scan_full_watchlist()
    st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(hist_cache)

signals_df = st.session_state["signals_cache"].copy()

st.subheader("🔎 Signals & Levels")
if not signals_df.empty:
    # Geldwerte formatieren (Preis, Levels)
    for c in ["price","MA20","MA50","Resistance","Support"]:
        if c in signals_df.columns:
            signals_df[c] = pd.to_numeric(signals_df[c], errors="coerce")
    display_df = signals_df.copy()
    display_df["price"]      = display_df["price"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["MA20"]       = display_df["MA20"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["MA50"]       = display_df["MA50"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["Resistance"] = display_df["Resistance"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["Support"]    = display_df["Support"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
    display_df["Vol_Surge_x"]= pd.to_numeric(display_df["Vol_Surge_x"], errors="coerce").map(lambda x: f"{x:.2f}x" if pd.notna(x) else "")

    def _row_style(row):
        if bool(row.get("Entry_Signal", False)): return ['background-color: #e6ffed'] * len(row)   # grün
        if bool(row.get("Distribution_Risk", False)): return ['background-color: #ffecec'] * len(row)  # rot
        if bool(row.get("Breakout_MA", False)) or bool(row.get("Breakout_Resistance", False)):
            return ['background-color: #fff9e6'] * len(row)  # gelb
        return [''] * len(row)

    st.dataframe(display_df.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)

    # Alerts auf Batch- oder Full-Scan
    if (scan_now_batch or scan_now_full) and st.session_state["alerts_enabled"]:
        hits = signals_df[signals_df["Entry_Signal"] == True]
        if hits.empty:
            st.info("Keine Entry-Signale.")
        else:
            ok_any = False
            for _, r in hits.iterrows():
                ok = send_telegram(
                    f"🚨 Entry-Signal: {r['id']} | Preis: ${r['price']:.3f} | "
                    f"Breakout über Widerstand {r['Resistance']:.3f} | Vol-Surge: {r['Vol_Surge_x']:.2f}x"
                )
                ok_any = ok_any or ok
            st.success("Telegram-Alerts gesendet." if ok_any else "Alert-Versand fehlgeschlagen (TELEGRAM_* prüfen).")

# Fortschritts-Hinweis bei Batch
if len(selected_ids) > 0 and not scan_now_full:
    start = st.session_state.get("scan_index", 0)
    end = min(start + int(st.session_state["batch_size_slider"]), len(selected_ids))
    if end == len(selected_ids):
        st.success("✅ Batch-Scan: Ende der Liste erreicht. Nächster Klick startet wieder vorn.")
    else:
        nxt_end = min(end + int(st.session_state["batch_size_slider"]), len(selected_ids))
        st.info(f"➡️ Nächster Batch lädt Coins {end+1}–{nxt_end} von {len(selected_ids)}.")

# ----------------- Detail & Risk-Tools -------------
st.markdown("---")
st.subheader("📈 Detail & Risk-Tools")

coin_select = st.selectbox(
    "Coin",
    options=[r["id"] for _, r in signals_df.iterrows()] if not signals_df.empty else selected_ids,
    key="detail_coin"
)

if coin_select:
    d = st.session_state.get("history_cache", {}).get(coin_select)
    # Falls nicht im Batch geladen: jetzt einmalig nachladen
    if d is None or d.empty:
        d = cg_market_chart(coin_select, days=st.session_state["days_hist"])
        if isinstance(d, pd.DataFrame) and not d.empty:
            # im Session-Cache ablegen, damit bei erneutem Öffnen kein zweiter Call nötig ist
            st.session_state.setdefault("history_cache", {})
            st.session_state["history_cache"][coin_select] = d

    # Status akzeptiert "ok", "ok_cg", "ok_binance"
    status_val = (d.attrs.get("status", "") if isinstance(d, pd.DataFrame) else "")
    if d is None or d.empty or (not str(status_val).startswith("ok")):
        st.warning("Keine Historie verfügbar (API-Limit oder leere Daten).")
        st.stop()

    # optional Quelle anzeigen
    st.caption(f"Datenquelle: {str(status_val).replace('ok_', '')}")

    df = d.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    if df.empty:
        st.warning("Keine validen Datenpunkte für das Chart gefunden.")
        st.stop()

    daily = df.set_index("timestamp").resample("1D").last().dropna(subset=["price"])
    daily["ma20"] = ma(daily["price"], 20)
    daily["ma50"] = ma(daily["price"], 50)
    lookback_val = int(st.session_state.get("lookback_res", lookback_res))
    resistance_lvl, support_lvl = calc_local_levels(daily.reset_index(drop=True), lookback=lookback_val)

    col_chart, col_trail = st.columns([3, 1])

    with col_chart:
        fig, ax_price = plt.subplots(figsize=(10, 5))
        ax_price.plot(daily.index, daily["price"], label="Preis", color="#1f77b4", linewidth=2)
        if daily["ma20"].notna().any():
            ax_price.plot(daily.index, daily["ma20"], label="MA20", color="#ff7f0e", linestyle="--")
        if daily["ma50"].notna().any():
            ax_price.plot(daily.index, daily["ma50"], label="MA50", color="#2ca02c", linestyle=":")
        if not math.isnan(resistance_lvl):
            ax_price.axhline(
                resistance_lvl,
                color="#d62728",
                linestyle="--",
                linewidth=1.2,
                label=f"Resistance ({lookback_val}d)",
            )
        if not math.isnan(support_lvl):
            ax_price.axhline(
                support_lvl,
                color="#17becf",
                linestyle="--",
                linewidth=1.2,
                label=f"Support ({lookback_val}d)",
            )
        ax_price.set_ylabel("Preis (USD)")
        ax_price.grid(True, linestyle=":", alpha=0.4)

        ax_vol = ax_price.twinx()
        ax_vol.bar(daily.index, daily["volume"], label="Volumen", color="#bbbbbb", alpha=0.4)
        ax_vol.set_ylabel("Volumen")

        handles, labels = ax_price.get_legend_handles_labels()
        if handles:
            ax_price.legend(handles, labels, loc="upper left")
        fig.autofmt_xdate()
        st.pyplot(fig, clear_figure=True)

    with col_trail:
        res_text = f"${resistance_lvl:,.2f}" if not math.isnan(resistance_lvl) else "–"
        sup_text = f"${support_lvl:,.2f}" if not math.isnan(support_lvl) else "–"
        st.metric("Resistance", res_text, help=f"Berechnet aus den letzten {lookback_val} Tagen (ohne aktuelle Kerze).")
        st.metric("Support", sup_text, help=f"Berechnet aus den letzten {lookback_val} Tagen (ohne aktuelle Kerze).")
        trail_pct = st.slider("Trailing Stop (%)", min_value=2, max_value=30, value=10, step=1)
        lookback_window = min(lookback_val, len(daily))
        if lookback_window > 0:
            window_slice = daily["price"].iloc[-lookback_window:]
            recent_high = float(window_slice.max())
            stop_level = trailing_stop(recent_high, trail_pct)
            st.metric("Trailing Stop", f"${stop_level:,.2f}")
        else:
            st.metric("Trailing Stop", "–")

    st.caption(
        "Preis (Linie) mit MA20/MA50, Widerstand/Support (Lookback) sowie Volumen (Balken). Rechts: Levels und dynamischer Trailing Stop."
    )

# ================= Top-100 Scanner =================
st.markdown("---")
st.subheader("🏆 Top-100 Setup-Scanner")

if scan_top100:
    with st.spinner("Lade Top-100 Liste …"):
        top100 = cg_top_coins(limit=100)
    ids100 = top100["id"].tolist() if not top100.empty else []
    if not ids100:
        st.warning("Konnte Top-100 nicht laden.")
    else:
        df100, _ = compute_rows_for_ids(ids100, days_hist, vol_surge_thresh, lookback_res, "Top-100-Scan")
        hits = df100[(df100["Entry_Signal"] == True) & (df100["status"] == "ok")].copy()

        # Geldwerte formatieren
        for c in ["price","MA20","MA50","Resistance","Support"]:
            if c in hits.columns:
                hits[c] = pd.to_numeric(hits[c], errors="coerce")
        hits["price"]      = hits["price"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
        hits["MA20"]       = hits["MA20"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
        hits["MA50"]       = hits["MA50"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
        hits["Resistance"] = hits["Resistance"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
        hits["Support"]    = hits["Support"].map(lambda x: fmt_money(x, 4) if pd.notna(x) else "")
        hits["Vol_Surge_x"]= pd.to_numeric(hits["Vol_Surge_x"], errors="coerce").map(lambda x: f"{x:.2f}x" if pd.notna(x) else "")

        st.subheader("✅ Treffer (Top-100, Setup erfüllt)")
        if hits.empty:
            st.info("Kein Top-100 Coin erfüllt aktuell das Setup.")
        else:
            st.dataframe(hits, use_container_width=True, hide_index=True)

