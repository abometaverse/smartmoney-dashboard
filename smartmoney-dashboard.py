# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Binance only
# v4.0: Datenquellen 100% Binance (Top100, Preise, Historie),
#       AgGrid Row-Click, Batch-Scan, MA20/MA50, R/S, Entry-Punkte,
#       tausender-Trennzeichen, kompakte Zahlen, Telegram optional.
#
# Secrets (Streamlit → Advanced settings → Secrets):
# APP_PASSWORD        = "DeinStarkesPasswort"
# TELEGRAM_BOT_TOKEN  = "123:abc"   # optional
# TELEGRAM_CHAT_ID    = "123456789" # optional
# -------------------------------------------------------------

import math
import time
from typing import List, Dict, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# ----------------- App Config -----------------
st.set_page_config(page_title="Smart Money Dashboard — Geschützt", layout="wide")

# ----------------- Helpers: session/persist -----------------
def save_state(keys):
    for k in keys:
        if k in st.session_state:
            st.session_state[f"_saved_{k}"] = st.session_state[k]

def restore_state(keys):
    for k in keys:
        v = st.session_state.get(f"_saved_{k}")
        if v is not None:
            st.session_state[k] = v

# ----------------- Auth Gate -----------------
def auth_gate():
    st.title("🧠 Smart Money Dashboard — Geschützt")
    pw = st.secrets.get("APP_PASSWORD")
    if not pw:
        st.error("Setze `APP_PASSWORD` unter Settings → Secrets.")
        st.stop()

    if st.session_state.get("AUTH_OK"):
        top = st.container()
        c1, c2 = top.columns([6,1])
        c1.success("Zugriff gewährt.")
        if c2.button("Logout"):
            save_state([
                "days_hist","vol_surge_thresh","lookback_res",
                "alerts_enabled","batch_size_slider","scan_index",
                "watchlist_manual"
            ])
            st.session_state["AUTH_OK"] = False
            st.rerun()
        return

    with st.form("login"):
        x = st.text_input("Passwort eingeben", type="password")
        ok = st.form_submit_button("Login")
    if ok:
        if x == pw:
            st.session_state["AUTH_OK"] = True
            restore_state([
                "days_hist","vol_surge_thresh","lookback_res",
                "alerts_enabled","batch_size_slider","scan_index",
                "watchlist_manual"
            ])
            st.rerun()
        else:
            st.error("Falsches Passwort.")
            st.stop()
    else:
        st.stop()

auth_gate()

# ----------------- HTTP -----------------
@st.cache_resource(show_spinner=False)
def http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "smartmoney-dashboard/4.0 (+streamlit)"})
    ad = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0)
    s.mount("https://", ad); s.mount("http://", ad)
    return s

def _get(url, params=None, timeout=10, retries=2, backoff=1.6) -> Dict:
    last = ""
    for i in range(retries+1):
        try:
            r = http().get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return {"ok": True, "json": r.json(), "status": 200}
            # 400/403/451/5xx sauber behandeln
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 451, 403, 502, 503, 504):
                time.sleep(backoff*(i+1)); continue
            return {"ok": False, "json": None, "status": r.status_code, "error": r.text[:300]}
        except requests.RequestException as e:
            last = str(e)[:200]; time.sleep(backoff*(i+1)); continue
    return {"ok": False, "json": None, "status": None, "error": last or "req failed"}

# ----------------- Formatting -----------------
def human_abbr(n: float) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)): return ""
    x = abs(float(n)); s = "-" if n < 0 else ""
    if x >= 1_000_000_000_000: return f"{s}{x/1_000_000_000_000:.2f}T"
    if x >= 1_000_000_000:     return f"{s}{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:         return f"{s}{x/1_000_000:.2f}M"
    if x >= 1_000:             return f"{s}{x:,.0f}"
    return f"{s}{x:.2f}"

def fmt(n: float, d=4) -> str:
    if n is None or (isinstance(n,float) and math.isnan(n)): return ""
    return f"{n:,.{d}f}"

# ----------------- Binance universe/top100 -----------------
BINANCE = "https://api.binance.com"

@st.cache_data(ttl=600, show_spinner=True)
def binance_exchange_info() -> pd.DataFrame:
    r = _get(f"{BINANCE}/api/v3/exchangeInfo")
    if not r.get("ok"): return pd.DataFrame(columns=["symbol","baseAsset","quoteAsset","status"])
    d = pd.DataFrame(r["json"]["symbols"])
    return d[["symbol","baseAsset","quoteAsset","status"]]

@st.cache_data(ttl=300, show_spinner=True)
def binance_top100_usdt() -> pd.DataFrame:
    """
    Top-USDT-Paare nach 24h quoteVolume (Annäherung an 'Top100').
    Filtert Leveraged/DOWN/UP/.*BULL/BEAR usw.
    """
    r = _get(f"{BINANCE}/api/v3/ticker/24hr")
    if not r.get("ok"): return pd.DataFrame()
    t = pd.DataFrame(r["json"])
    # nur USDT-Spot
    t = t[t["symbol"].str.endswith("USDT")].copy()
    # Hebel-/ETF-Filter
    bad = ("UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT","3LUSDT","3SUSDT","5LUSDT","5SUSDT","2LUSDT","2SUSDT")
    t = t[~t["symbol"].str.endswith(bad)]
    # nach quoteVolume sortieren
    t["quoteVolume"] = pd.to_numeric(t["quoteVolume"], errors="coerce")
    t = t.sort_values("quoteVolume", ascending=False)
    t = t.head(100).copy()

    ex = binance_exchange_info()
    if ex.empty: 
        # fallback: base = symbol[:-4]
        t["baseAsset"] = t["symbol"].str.replace("USDT","", regex=False)
        t["name"] = t["baseAsset"]
    else:
        m = ex.set_index("symbol")[["baseAsset","quoteAsset"]]
        t["baseAsset"] = t["symbol"].map(lambda s: m.loc[s,"baseAsset"] if s in m.index else s.replace("USDT",""))
        t["quoteAsset"] = t["symbol"].map(lambda s: m.loc[s,"quoteAsset"] if s in m.index else "USDT")
        t["name"] = t["baseAsset"]

    t["coin_id"] = t["baseAsset"].str.lower()   # interne ID
    t["label"] = t.apply(lambda r: f"{r['name']} ({r['baseAsset']})", axis=1)
    t["lastPrice"] = pd.to_numeric(t["lastPrice"], errors="coerce")
    t["volume"]    = pd.to_numeric(t["volume"], errors="coerce")
    t["quoteVolume"] = pd.to_numeric(t["quoteVolume"], errors="coerce")
    return t[["coin_id","symbol","baseAsset","name","label","lastPrice","volume","quoteVolume"]]

# ----------------- Historie (Binance Klines) -----------------
@st.cache_data(ttl=900, show_spinner=False)
def binance_klines_usdt(base_asset: str, days: int = 180) -> pd.DataFrame:
    """
    Holt 1d-Kerzen für BASEUSDT. Probiert mehrere Endpunkte; 400/403/451/5xx werden
    sauber abgefangen. Rückgabe: df(timestamp, price, volume) mit attrs: status='ok'
    """
    if not base_asset: 
        df = pd.DataFrame(columns=["timestamp","price","volume"]); df.attrs["status"] = "no_symbol"; return df
    symbol = f"{base_asset.upper()}USDT"
    endpoints = [
        f"{BINANCE}/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]
    kl, last_status = None, None
    limit = min(1000, int(days)+5)
    for url in endpoints:
        r = _get(url, params={"symbol": symbol, "interval": "1d", "limit": limit}, timeout=8, retries=1)
        last_status = r.get("status")
        if r.get("ok") and isinstance(r["json"], list) and r["json"]:
            kl = r["json"]; break
        time.sleep(0.2)
    if kl is None:
        df = pd.DataFrame(columns=["timestamp","price","volume"]); df.attrs["status"] = f"err_binance:{last_status}"; return df

    try:
        df = pd.DataFrame(kl, columns=[
            "openTime","open","high","low","close","volume","closeTime","qav","numTrades","takerBase","takerQuote","ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True, errors="coerce")
        df["price"]     = pd.to_numeric(df["close"], errors="coerce")
        df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
        df = df[["timestamp","price","volume"]].dropna().sort_values("timestamp").tail(int(days)+1)
        df.attrs["status"] = "ok"; df.attrs["source"] = "binance"
        return df
    except Exception:
        df = pd.DataFrame(columns=["timestamp","price","volume"]); df.attrs["status"] = "parse_binance"; return df

# ----------------- Analytics -----------------
def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

def calc_local_levels(dfd: pd.DataFrame, lookback: int = 20) -> Tuple[float,float]:
    if dfd.empty: return (np.nan, np.nan)
    d = dfd.iloc[:-1].tail(lookback)
    if d.empty: return (np.nan, np.nan)
    return float(d["price"].max()), float(d["price"].min())

def volume_signals(dfd: pd.DataFrame) -> Dict:
    out = {"vol_ratio_1d_vs_7d": np.nan, "distribution_risk": False, "price_chg_7d": np.nan}
    if dfd.empty or len(dfd) < 8: return out
    last = dfd.iloc[-1]; avg7 = dfd["volume"].iloc[-8:-1].mean()
    vr = float(last["volume"]/avg7) if (avg7 and avg7 == avg7) else np.nan
    out["vol_ratio_1d_vs_7d"] = vr
    p7 = dfd["price"].iloc[-8]; out["price_chg_7d"] = float((last["price"]/p7)-1.0) if p7 else np.nan
    out["distribution_risk"] = bool((out["price_chg_7d"] > 0) and (vr < 0.8))
    return out

def trend_signals(dfd: pd.DataFrame) -> Dict:
    out = {"ma20": np.nan, "ma50": np.nan, "breakout_ma": False}
    if dfd.empty: return out
    df = dfd.copy(); df["ma20"] = ma(df["price"], 20); df["ma50"] = ma(df["price"], 50); last = df.iloc[-1]
    out["ma20"], out["ma50"] = float(last["ma20"]), float(last["ma50"])
    out["breakout_ma"] = bool(last["price"] > last["ma20"] > last["ma50"])
    return out

def send_telegram(msg: str) -> bool:
    t = st.secrets.get("TELEGRAM_BOT_TOKEN"); c = st.secrets.get("TELEGRAM_CHAT_ID")
    if not t or not c: return False
    try:
        http().post(f"https://api.telegram.org/bot{t}/sendMessage", json={"chat_id": c, "text": msg}, timeout=10)
        return True
    except Exception:
        return False

def trailing_stop(current_high: float, trail_pct: float) -> float:
    return current_high * (1 - trail_pct/100.0)

# ----------------- Sidebar -----------------
st.sidebar.header("Settings")
defaults = {
    "days_hist": 90, "vol_surge_thresh": 1.5, "lookback_res": 20,
    "alerts_enabled": True, "batch_size_slider": 5, "scan_index": 0,
    "watchlist_manual": ""
}
for k,v in defaults.items(): st.session_state.setdefault(k,v)

days_hist = st.sidebar.slider("Historie (Tage)", 60, 365, int(st.session_state["days_hist"]), 15, key="days_hist")
vol_surge_thresh = st.sidebar.slider("Vol Surge vs 7d (x)", 1.0, 5.0, float(st.session_state["vol_surge_thresh"]), 0.1, key="vol_surge")
lookback_res = st.sidebar.slider("Lookback Widerstand/Support (Tage)", 10, 60, int(st.session_state["lookback_res"]), 1, key="lookback")
alerts_enabled = st.sidebar.checkbox("Telegram-Alerts aktivieren (optional)", value=bool(st.session_state["alerts_enabled"]), key="alerts_on")

st.sidebar.markdown("**Watchlist (manuell, BASE-Assets, Komma-getrennt)**")
st.session_state["watchlist_manual"] = st.sidebar.text_area("z.B. BTC, ETH, SOL", value=st.session_state["watchlist_manual"], height=80)

scan_batch = st.sidebar.button("🔔 Batch-Scan (Watchlist)", key="scan_batch")
scan_full  = st.sidebar.button("🔁 Voll-Scan (Watchlist)", key="scan_full")
batch_size = st.sidebar.slider("Coins pro Scan (Batchgröße)", 2, 20, int(st.session_state["batch_size_slider"]), 1, key="batch_size_slider")
if st.sidebar.button("🔄 Batch zurücksetzen"):
    st.session_state["scan_index"] = 0

st.caption("🔒 Passwortschutz aktiv • Quelle: **Binance** (Top100, Preise, Historie).")

# ----------------- Universe / Snapshot -----------------
top100 = binance_top100_usdt()
if top100.empty:
    st.error("Konnte Binance Top-100 nicht laden.")
else:
    st.subheader("📊 Snapshot Top-100 (Binance, nach QuoteVolume)")
    disp = top100.copy()
    disp.rename(columns={"label":"Coin","lastPrice":"Price","volume":"Vol","quoteVolume":"QuoteVol"}, inplace=True)
    disp["Price"] = disp["Price"].map(lambda x: fmt(x, 4))
    disp["Vol"]   = disp["Vol"].map(human_abbr)
    disp["QuoteVol"] = disp["QuoteVol"].map(human_abbr)
    st.dataframe(disp[["Coin","Price","Vol","QuoteVol","coin_id","symbol"]], use_container_width=True, hide_index=True)

# ----------------- Watchlist -----------------
watchlist = []
if st.session_state["watchlist_manual"].strip():
    watchlist = [a.strip().upper() for a in st.session_state["watchlist_manual"].split(",") if a.strip()]
else:
    # Default: nimm die Top100 BaseAssets (praktisch zum Start)
    watchlist = top100["baseAsset"].head(10).tolist()

# ----------------- Signals Cache -----------------
if "signals_cache" not in st.session_state: st.session_state["signals_cache"] = pd.DataFrame()
if "history_cache" not in st.session_state: st.session_state["history_cache"] = {}

def compute_for_assets(assets: List[str], label="Scan") -> Tuple[pd.DataFrame, Dict]:
    rows, cache = [], {}
    total = len(assets)
    prog = st.progress(0, text=f"{label} startet …")
    msg  = st.empty()
    SLEEP = 0.45
    for i, base in enumerate(assets, start=1):
        msg.info(f"{label}: {i}/{total} — {base}/USDT")
        time.sleep(SLEEP)
        hist = binance_klines_usdt(base, days=days_hist)
        status = hist.attrs.get("status","")
        if hist.empty or status != "ok":
            rows.append({
                "base": base, "price": np.nan, "MA20": np.nan, "MA50": np.nan,
                "Breakout_MA": False, "Vol_Surge_x": np.nan,
                "Resistance": np.nan, "Support": np.nan,
                "Breakout_Resistance": False, "Distribution_Risk": False,
                "Entry_Signal": False, "status": status
            })
            prog.progress(min(i/total,1.0), text=f"{label} ({i}/{total})"); continue

        cache[base] = hist
        df = hist.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp").sort_index().resample("1D").last().dropna()
        t = trend_signals(df); v = volume_signals(df)
        r, s = calc_local_levels(df, lookback_res)
        last = df.iloc[-1]; price = float(last["price"])
        volx = v["vol_ratio_1d_vs_7d"]; ok_vol = not np.isnan(volx)
        br_res = bool(price >= (r*1.0005)) if not math.isnan(r) else False
        entry = bool(t["breakout_ma"] and ok_vol and (volx >= vol_surge_thresh))

        rows.append({
            "base": base, "price": price, "MA20": t["ma20"], "MA50": t["ma50"],
            "Breakout_MA": t["breakout_ma"], "Vol_Surge_x": volx,
            "Resistance": r, "Support": s, "Breakout_Resistance": br_res,
            "Distribution_Risk": v["distribution_risk"], "Entry_Signal": entry,
            "status": "ok"
        })
        prog.progress(min(i/total,1.0), text=f"{label} ({i}/{total})")
    prog.progress(1.0, text=f"{label} fertig"); msg.empty()
    return pd.DataFrame(rows), cache

def run_batch():
    start = st.session_state.get("scan_index", 0)
    end = min(start + batch_size, len(watchlist))
    batch = watchlist[start:end]
    if not batch:
        st.warning("Keine Assets im Batch. Batch zurückgesetzt.")
        st.session_state["scan_index"] = 0
        return pd.DataFrame(), {}
    st.info(f"⏳ Batch {start+1}–{end} von {len(watchlist)} …")
    df, c = compute_for_assets(batch, "Batch-Scan")
    st.session_state["scan_index"] = end % max(1,len(watchlist))
    return df, c

def run_full():
    st.info("🔁 Voll-Scan (Watchlist) …")
    return compute_for_assets(watchlist, "Voll-Scan")

if scan_batch:
    sig, h = run_batch()
    st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(h)
elif scan_full:
    sig, h = run_full()
    st.session_state["signals_cache"] = sig
    st.session_state["history_cache"].update(h)

signals = st.session_state["signals_cache"].copy()
st.subheader("🔎 Signals & Levels (Watchlist)")

if not signals.empty:
    # join mit Snapshot für Label
    lbl = top100.set_index("baseAsset")["label"].to_dict()
    signals["Coin"] = signals["base"].map(lambda b: lbl.get(b, f"{b} ({b})"))
    # format
    for c in ["price","MA20","MA50","Resistance","Support"]:
        if c in signals.columns:
            signals[c] = pd.to_numeric(signals[c], errors="coerce")
    view = signals.copy()
    view["price"]      = view["price"].map(lambda x: fmt(x,4) if pd.notna(x) else "")
    view["MA20"]       = view["MA20"].map(lambda x: fmt(x,4) if pd.notna(x) else "")
    view["MA50"]       = view["MA50"].map(lambda x: fmt(x,4) if pd.notna(x) else "")
    view["Resistance"] = view["Resistance"].map(lambda x: fmt(x,4) if pd.notna(x) else "")
    view["Support"]    = view["Support"].map(lambda x: fmt(x,4) if pd.notna(x) else "")
    view["Vol_Surge_x"]= pd.to_numeric(view["Vol_Surge_x"], errors="coerce").map(lambda x: f"{x:.2f}x" if pd.notna(x) else "")

    def _row_style(r):
        if r.get("Entry_Signal"): return ['background-color:#e6ffed']*len(r)
        if r.get("Distribution_Risk"): return ['background-color:#ffecec']*len(r)
        if r.get("Breakout_MA") or r.get("Breakout_Resistance"): return ['background-color:#fff9e6']*len(r)
        return ['']*len(r)

    st.dataframe(
        view[["Coin","price","MA20","MA50","Vol_Surge_x","Resistance","Support","Breakout_MA","Breakout_Resistance","Distribution_Risk","Entry_Signal","status","base"]]
        .style.apply(_row_style, axis=1),
        use_container_width=True, hide_index=True
    )

# ----------------- Detail & Risk — Top100 (AgGrid Row-Click) -----------------
st.markdown("---")
st.subheader("📈 Detail & Risk — Top-100 (Binance)")

if not top100.empty:
    gdf = top100.rename(columns={"label":"Coin","lastPrice":"Price","volume":"Vol","quoteVolume":"QuoteVol"}).copy()
    gdf["Price"] = gdf["Price"].map(lambda x: fmt(x,4))
    gdf["Vol"]   = gdf["Vol"].map(human_abbr)
    gdf["QuoteVol"] = gdf["QuoteVol"].map(human_abbr)

    gb = GridOptionsBuilder.from_dataframe(gdf[["Coin","Price","Vol","QuoteVol","baseAsset","symbol"]])
    gb.configure_selection("single", use_checkbox=False)
    gb.configure_grid_options(domLayout='normal')
    grid = AgGrid(gdf, gridOptions=gb.build(), update_mode=GridUpdateMode.SELECTION_CHANGED,
                  theme="balham", height=420, fit_columns_on_grid_load=True)
    sel = grid.get("selected_rows", [])
    if sel:
        st.session_state["selected_base"] = sel[0]["baseAsset"]
    else:
        if "selected_base" not in st.session_state:
            st.session_state["selected_base"] = gdf.iloc[0]["baseAsset"]

# ----------------- Einzel-Chart & Risk-Tools -----------------
st.markdown("---")
st.subheader("🔍 Einzel-Chart & Risk-Tools")

base = st.session_state.get("selected_base", (watchlist[0] if watchlist else "BTC"))
if base:
    with st.spinner(f"Lade Historie für {base}/USDT …"):
        d = st.session_state.get("history_cache", {}).get(base)
        if d is None or d.empty:
            d = binance_klines_usdt(base, days=days_hist)
            if not d.empty:
                st.session_state.setdefault("history_cache", {})
                st.session_state["history_cache"][base] = d

    if d.empty or d.attrs.get("status") != "ok":
        st.warning("Keine Historie verfügbar (451/403/400/Ratelimit?).")
    else:
        dfd = d.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd = dfd.set_index("timestamp").sort_index().resample("1D").last().dropna()
        r, s = calc_local_levels(dfd, lookback_res)
        dfd["ma20"] = ma(dfd["price"], 20); dfd["ma50"] = ma(dfd["price"], 50)
        dfd["vol7"] = dfd["volume"].rolling(7, min_periods=3).mean()
        dfd["vol_ratio"] = dfd["volume"]/dfd["vol7"]
        dfd["roll_max_prev"] = dfd["price"].shift(1).rolling(lookback_res, min_periods=5).max()
        dfd["entry_flag"] = (dfd["price"] > dfd["ma20"]) & (dfd["ma20"] > dfd["ma50"]) & \
                            (dfd["price"] > dfd["roll_max_prev"]) & \
                            (dfd["vol_ratio"] >= vol_surge_thresh)
        entries = dfd[dfd["entry_flag"]]

        fig, ax_p = plt.subplots()
        ax_v = ax_p.twinx()
        ax_p.plot(dfd.index, dfd["price"], label="Price", linewidth=1.6)
        ax_p.plot(dfd.index, dfd["ma20"], label="MA20", linewidth=1.0)
        ax_p.plot(dfd.index, dfd["ma50"], label="MA50", linewidth=1.0)
        if not np.isnan(r): ax_p.axhline(r, linestyle="--", label=f"Resistance {r:.3f}")
        if not np.isnan(s): ax_p.axhline(s, linestyle="--", label=f"Support {s:.3f}")
        if not entries.empty:
            ax_p.scatter(entries.index, entries["price"], s=36, color="#16a34a", zorder=5, label="Entry (hist)")

        ax_v.bar(dfd.index, dfd["volume"], alpha=0.28)
        ax_v.set_ylabel("Volume")

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax_p.xaxis.set_major_locator(locator)
        ax_p.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax_p.set_title(f"{base}/USDT — Price, MAs, Levels & Volume")
        ax_p.set_xlabel("Date"); ax_p.set_ylabel("USD"); ax_p.legend(loc="upper left")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # ---- Position & Trailing ----
        st.markdown("### 🧮 Position & Trailing Stop")
        c1,c2,c3,c4 = st.columns(4)
        last_px = float(dfd["price"].iloc[-1])
        portfolio = c1.number_input("Portfolio (USD)", min_value=0.0, value=8000.0, step=100.0, format="%.2f")
        risk_pct  = c2.slider("Risiko/Trade (%)", 0.5, 3.0, 2.0, 0.1)
        stop_pct  = c3.slider("Stop-Entfernung (%)", 3.0, 25.0, 8.0, 0.5)
        entry_p   = c4.number_input("Entry-Preis", min_value=0.0, value=last_px, step=0.001, format="%.6f")
        max_loss  = portfolio*(risk_pct/100.0)
        size_usd  = max_loss/(stop_pct/100.0) if stop_pct>0 else 0.0
        size_qty  = size_usd/entry_p if entry_p>0 else 0.0
        st.write(f"**Max. Verlust:** ${max_loss:,.2f} • **Positionsgröße:** ${size_usd:,.2f} (~ {size_qty:,.4f} {base})")

        st.markdown("#### Trailing Stop")
        t1,t2 = st.columns(2)
        trail = t1.slider("Trail (%)", 5.0, 25.0, 10.0, 0.5)
        high  = t2.number_input("Höchster Kurs seit Entry", min_value=0.0, value=last_px, step=0.001, format="%.6f")
        tstop = trailing_stop(high, trail)
        st.write(f"Trailing Stop bei **${tstop:,.3f}** (High {high:,.3f}, Trail {trail:.1f}%)")

        v_sig = volume_signals(dfd)
        if v_sig["distribution_risk"]:
            st.warning("Distribution-Risk: Preis ↑ bei Volumen < 0.8× 7d-Ø.")
        else:
            st.success("Volumen ok (kein Distribution-Hinweis).")

# Fortschritt Batch
if watchlist:
    s = st.session_state.get("scan_index", 0)
    e = min(s + batch_size, len(watchlist))
    if e == len(watchlist):
        st.success("✅ Batch-Scan: Ende der Liste erreicht. Nächster Klick startet wieder vorn.")
    else:
        st.info(f"➡️ Nächster Batch lädt {e+1}–{min(e+batch_size,len(watchlist))} von {len(watchlist)}.")
