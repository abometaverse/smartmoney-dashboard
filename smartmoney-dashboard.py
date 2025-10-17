# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Geschützt (Streamlit)
# v4.1  Eine Tabelle (Watchlist + Top100) mit st-aggrid:
#       - Doppelklick -> aktiver Coin (kein Checkbox-Select, immer genau 1 aktiv)
#       - active_from_table init None -> kein Chart bis Doppelklick
#       - Filter: Alle / Nur Entry-Signal / Nur Watchlist
#       - Tausenderformat, Sortieren/Filtern im Grid
#       - Telegram-Alerts (manuell + Auto-Scan alle X Stunden)
#       - Link in Alerts inkl. aktivem Coin via st.query_params
#
# Secrets (optional/empfohlen):
# APP_PASSWORD="..."
# TELEGRAM_BOT_TOKEN="123:abc"
# TELEGRAM_CHAT_ID="123456789"
# APP_URL="https://dein-name.streamlit.app"
# -------------------------------------------------------------

import math, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(page_title="Smart Money Dashboard — Geschützt", layout="wide")

# ================== Session Defaults ==================
PERSIST_KEYS = [
    "selected_ids","vol_surge_thresh","lookback_res","alerts_enabled",
    "days_hist","batch_size_slider","scan_index",
    "top100_df","top100_last_sync_ts","top100_cooldown_min",
    "auto_scan_enabled","auto_scan_hours","auto_last_ts","auto_alerted_ids",
    "signals_cache","history_cache","active_from_table"
]

def ensure_defaults():
    defaults = {
        "selected_ids": [],
        "vol_surge_thresh": 1.5,
        "lookback_res": 20,
        "alerts_enabled": True,
        "days_hist": 90,
        "batch_size_slider": 3,
        "scan_index": 0,
        "top100_df": pd.DataFrame(),
        "top100_last_sync_ts": 0.0,
        "top100_cooldown_min": 15,
        "auto_scan_enabled": False,
        "auto_scan_hours": 1.0,
        "auto_last_ts": 0.0,
        "auto_alerted_ids": [],
        "signals_cache": pd.DataFrame(),
        "history_cache": {},
        "active_from_table": None,  # << initial leer => kein Chart bis Double-Click
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def save_state(keys): 
    for k in keys:
        if k in st.session_state: st.session_state[f"_saved_{k}"]=st.session_state[k]

def restore_state(keys):
    for k in keys:
        s=f"_saved_{k}"
        if s in st.session_state: st.session_state[k]=st.session_state[s]

# ================== Auth ==================
def auth_gate():
    st.title("🧠 Smart Money Dashboard — Geschützt")
    pw = st.secrets.get("APP_PASSWORD")
    if not pw:
        st.error("Setze `APP_PASSWORD` in den Secrets."); st.stop()
    if st.session_state.get("AUTH_OK", False):
        c1,c2=st.columns([6,1]); c1.success("Zugriff gewährt.")
        if c2.button("Logout"):
            save_state(PERSIST_KEYS)
            st.session_state["AUTH_OK"]=False
            st.success("Einstellungen gespeichert.")
            time.sleep(0.2); st.rerun()
        return
    with st.form("login"):
        p = st.text_input("Passwort", type="password")
        ok= st.form_submit_button("Login")
        if ok:
            if p==pw:
                st.session_state["AUTH_OK"]=True
                restore_state(PERSIST_KEYS)
                ensure_defaults()
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    st.stop()

auth_gate()
ensure_defaults()

# ================== HTTP helpers & Sources ==================
BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]
FIAT="usd"

@st.cache_resource
def http()->requests.Session:
    s=requests.Session()
    s.headers.update({"User-Agent":"smartmoney-dashboard/4.1 (+streamlit)"})
    ad=requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    s.mount("https://",ad); s.mount("http://",ad)
    return s

def _binance_first_ok(path:str, params:Dict=None, timeout:int=10):
    sess=http(); last=None
    for base in BINANCE_ENDPOINTS:
        try:
            r=sess.get(f"{base}{path}", params=params, timeout=timeout)
            if r.status_code==200: return r.json()
            last=r.status_code; time.sleep(0.2)
        except requests.RequestException: last="req"; time.sleep(0.2)
    raise RuntimeError(f"binance_fail:{last}")

@st.cache_data(ttl=900)
def binance_exchange_info()->pd.DataFrame:
    try:
        js=_binance_first_ok("/api/v3/exchangeInfo")
    except Exception:
        return pd.DataFrame()
    df=pd.DataFrame(js.get("symbols",[]))
    if df.empty: return df
    df=df[(df["quoteAsset"]=="USDT")&(df["status"]=="TRADING")]
    df=df[~df["symbol"].str.contains(r"(?:UP|DOWN|BULL|BEAR)", regex=True)]
    return df[["symbol","baseAsset","quoteAsset"]].copy()

@st.cache_data(ttl=300)
def binance_ticker_24hr()->pd.DataFrame:
    for base in BINANCE_ENDPOINTS:
        try:
            r=http().get(f"{base}/api/v3/ticker/24hr", timeout=8)
            if r.status_code==200: return pd.DataFrame(r.json())
        except Exception: time.sleep(0.2)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def binance_top100_by_quote_volume()->pd.DataFrame:
    info=binance_exchange_info()
    t=binance_ticker_24hr()
    if info.empty or t.empty: return pd.DataFrame()
    df=t.merge(info, on="symbol", how="inner")
    df["quoteVolume"]=pd.to_numeric(df["quoteVolume"], errors="coerce")
    df=df.sort_values("quoteVolume", ascending=False).head(100).reset_index(drop=True)
    df["rank"]=df.index+1
    df["id"]=df["symbol"]
    df["name"]=df["baseAsset"].str.upper()
    df["symbol_txt"]=df["baseAsset"].str.upper()
    return df[["rank","id","symbol","name","symbol_txt","quoteVolume"]]

@st.cache_data(ttl=1200)
def load_history(binance_symbol:str, days:int=180)->pd.DataFrame:
    def _empty(msg): 
        df=pd.DataFrame(columns=["timestamp","price","volume"]); df.attrs["status"]=msg; return df
    sym=str(binance_symbol).upper()
    if not sym.endswith("USDT"): sym=f"{sym}USDT"
    kl=None
    for base in BINANCE_ENDPOINTS:
        try:
            r=http().get(f"{base}/api/v3/klines",
                        params={"symbol":sym,"interval":"1d","limit":min(1000,int(days)+5)},
                        timeout=8)
            if r.status_code==200 and isinstance(r.json(), list):
                kl=r.json(); break
        except Exception: time.sleep(0.2)
    if kl is None: return _empty("err_binance")
    try:
        df=pd.DataFrame(kl, columns=["openTime","open","high","low","close","volume",
                                     "closeTime","qav","numTrades","takerBase","takerQuote","ignore"])
        df["timestamp"]=pd.to_datetime(df["openTime"], unit="ms", utc=True, errors="coerce")
        df["price"]=pd.to_numeric(df["close"], errors="coerce")
        df["volume"]=pd.to_numeric(df["volume"], errors="coerce")
        df=df[["timestamp","price","volume"]].dropna().sort_values("timestamp").tail(int(days)+1)
        df.attrs["status"]="ok"; return df
    except Exception:
        return _empty("parse_binance")

# ================== Indicators ==================
def ma(s:pd.Series, win:int)->pd.Series:
    return s.rolling(win, min_periods=max(2,win//2)).mean()

def levels(df:pd.DataFrame, lb:int)->Tuple[float,float]:
    if df.empty: return (np.nan,np.nan)
    d=df.iloc[:-1].tail(lb)
    if d.empty: return (np.nan,np.nan)
    return float(d["price"].max()), float(d["price"].min())

def vol_sig(df:pd.DataFrame)->Dict:
    out={"vr":np.nan,"dist":False}
    if df.empty or len(df)<8: return out
    last=df.iloc[-1]; avg7=df["volume"].iloc[-8:-1].mean()
    vr=float(last["volume"]/avg7) if (avg7 and avg7==avg7) else np.nan
    out["vr"]=vr; out["dist"]=False; return out

def trend(df:pd.DataFrame)->Dict:
    out={"ma20":np.nan,"ma50":np.nan,"bo":False}
    if df.empty: return out
    d=df.copy(); d["ma20"]=ma(d["price"],20); d["ma50"]=ma(d["price"],50)
    last=d.iloc[-1]
    out["ma20"]=float(last["ma20"]); out["ma50"]=float(last["ma50"])
    out["bo"]=bool(last["price"]>last["ma20"]>last["ma50"]); return out

def fmt_money(x,dec=4):
    if x is None or (isinstance(x,float) and math.isnan(x)): return ""
    return f"{x:,.{dec}f}"

# ================== Telegram ==================
def send_telegram(text:str)->bool:
    token = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat  = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat: return False
    try:
        http().post(f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat, "text": text}, timeout=10)
        return True
    except Exception:
        return False

def alert_entries(df:pd.DataFrame, universe:str):
    """
    Schickt Alerts für neue Entry-Signale; verhindert Doppler via session_state['auto_alerted_ids'].
    """
    if df.empty: return
    app_url = st.secrets.get("APP_URL","")
    alerted = set(st.session_state.get("auto_alerted_ids", []))
    newly = []
    for _,r in df.iterrows():
        if not bool(r.get("Entry_Signal", False)): continue
        cid = str(r["id"])
        if cid in alerted: continue
        name = str(r.get("name", cid))
        sym  = str(r.get("symbol", ""))
        link = app_url
        if app_url:
            # Link inkl. aktivem Coin
            if "?" in app_url:
                link = f"{app_url}&coin={cid}"
            else:
                link = f"{app_url}?coin={cid}"
        txt = f"🚨 Entry-Signal ({universe}): {name} ({sym})\nPreis: {fmt_money(r.get('price'),4)}\nVol-Surge: {r.get('Vol_Surge_x'):.2f}x\n→ {link}" if isinstance(r.get("Vol_Surge_x"), (float,int)) else f"🚨 Entry-Signal ({universe}): {name} ({sym})\n→ {link}"
        ok = send_telegram(txt)
        if ok: newly.append(cid)
    if newly:
        st.session_state["auto_alerted_ids"] = list(alerted.union(newly))

# ================== Sidebar ==================
st.sidebar.header("Settings")
st.session_state["days_hist"] = st.sidebar.slider("Historie (Tage)",60,365,int(st.session_state["days_hist"]),15)
st.session_state["vol_surge_thresh"] = st.sidebar.slider("Vol Surge vs 7d (x)",1.0,5.0,float(st.session_state["vol_surge_thresh"]),0.1)
st.session_state["lookback_res"] = st.sidebar.slider("Lookback Support/Widerstand (Tage)",10,60,int(st.session_state["lookback_res"]),1)
st.session_state["top100_cooldown_min"] = st.sidebar.number_input("Top-100 Cooldown (Min.)",1,120,int(st.session_state["top100_cooldown_min"]),1)

c1,c2=st.sidebar.columns(2)
scan_watchlist = c1.button("🔁 Watchlist scan")
refresh_top100 = c2.button("🔄 Top-100 aktualisieren")

# Auto-Scan / Alerts
st.sidebar.subheader("Auto-Scan & Alerts")
st.session_state["auto_scan_enabled"] = st.sidebar.checkbox("Auto-Scan Top-100 aktiv", value=bool(st.session_state["auto_scan_enabled"]))
st.session_state["auto_scan_hours"]   = st.sidebar.slider("Intervall (Stunden)", 0.5, 24.0, float(st.session_state["auto_scan_hours"]), 0.5)
st.session_state["alerts_enabled"]    = st.sidebar.checkbox("Telegram-Alerts senden", value=bool(st.session_state["alerts_enabled"]))

# Watchlist Eingabe (IDs/BASENAME/BINANCE)
wl_raw = st.sidebar.text_area("Watchlist (je Zeile: BASE oder BASEUSDT)", value="\n".join(st.session_state["selected_ids"]) or "BTC\nETH\nSOL")
if wl_raw.strip():
    st.session_state["selected_ids"] = [x.strip() for x in wl_raw.splitlines() if x.strip()]

# ================== Compute Watchlist & Top100 ==================
def compute_for_symbols(symbols:List[str], label:str)->Tuple[pd.DataFrame, Dict[str,pd.DataFrame]]:
    rows=[]; cache={}
    total=len(symbols); 
    if total==0: return pd.DataFrame(),{}
    bar=st.progress(0, text=f"{label} 0/{total}")
    for i,sym in enumerate(symbols, start=1):
        df=load_history(sym, st.session_state["days_hist"])
        status=df.attrs.get("status","") if isinstance(df,pd.DataFrame) else "no"
        base = sym[:-4] if sym.upper().endswith("USDT") else sym.upper()
        if status!="ok":
            rows.append({"universe":label,"rank":i,"id":f"{base}USDT","name":base,"symbol":base,
                         "price":np.nan,"MA20":np.nan,"MA50":np.nan,"Vol_Surge_x":np.nan,
                         "Resistance":np.nan,"Support":np.nan,"Breakout_MA":False,
                         "Breakout_Resistance":False,"Distribution_Risk":False,
                         "Entry_Signal":False,"status":status})
            bar.progress(i/total, text=f"{label} {i}/{total}"); continue
        cache[f"{base}USDT"]=df
        d=df.copy()
        d["timestamp"]=pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
        d["price"]=pd.to_numeric(d["price"], errors="coerce")
        d["volume"]=pd.to_numeric(d["volume"], errors="coerce")
        d=d.dropna(subset=["timestamp","price"]).set_index("timestamp").sort_index()
        d=d.resample("1D").last().dropna(subset=["price"])
        t=trend(d); v=vol_sig(d); r,s=levels(d, st.session_state["lookback_res"])
        last=float(d.iloc[-1]["price"])
        vr=v["vr"]; is_ok=not np.isnan(vr)
        bo_res = bool(last >= (r*1.0005)) if not math.isnan(r) else False
        entry = bool(t["bo"] and is_ok and (vr>=st.session_state["vol_surge_thresh"]) and bo_res)
        rows.append({"universe":label,"rank":i,"id":f"{base}USDT","name":base,"symbol":base,
                     "price":last,"MA20":t["ma20"],"MA50":t["ma50"],"Vol_Surge_x":vr,
                     "Resistance":r,"Support":s,"Breakout_MA":t["bo"],
                     "Breakout_Resistance":bo_res,"Distribution_Risk":False,
                     "Entry_Signal":entry,"status":"ok"})
        bar.progress(i/total, text=f"{label} {i}/{total}")
    bar.empty()
    df=pd.DataFrame(rows)
    return df, cache

# Watchlist (auf Klick)
if scan_watchlist:
    wl_syms=[]
    for it in st.session_state["selected_ids"]:
        s=it.upper()
        wl_syms.append(s if s.endswith("USDT") else f"{s}USDT")
    with st.spinner("Watchlist-Scan …"):
        sig, cache = compute_for_symbols(wl_syms, "Watchlist")
        st.session_state["signals_cache"]=sig
        st.session_state["history_cache"].update(cache)
        if st.session_state["alerts_enabled"]:
            alert_entries(sig[sig["Entry_Signal"]==True], "Watchlist")

# Top-100 (mit Cooldown)
now=time.time(); cooldown=st.session_state["top100_cooldown_min"]*60
if (refresh_top100 and (now-st.session_state["top100_last_sync_ts"]>=cooldown)) or st.session_state["top100_df"].empty:
    base = binance_top100_by_quote_volume()
    if not base.empty:
        syms=base["id"].tolist()
        with st.spinner("Top-100 Scan …"):
            tdf, tcache = compute_for_symbols(syms, "Top100")
        # Rang aus Basis übernehmen (stabil)
        if not tdf.empty:
            rank_map = dict(zip(base["id"], base["rank"]))
            tdf["rank"] = tdf["id"].map(rank_map).fillna(tdf["rank"])
        st.session_state["top100_df"]=tdf
        st.session_state["history_cache"].update(tcache)
        st.session_state["top100_last_sync_ts"]=time.time()
        if st.session_state["alerts_enabled"]:
            alert_entries(tdf[tdf["Entry_Signal"]==True], "Top-100")
elif refresh_top100 and (now-st.session_state["top100_last_sync_ts"]<cooldown):
    left=int((st.session_state["top100_last_sync_ts"]+cooldown-now)/60)+1
    st.warning(f"Top-100 Cooldown aktiv. Bitte in ~{max(left,1)} Minuten erneut.")

# Auto-Scan (wenn App läuft)
if st.session_state["auto_scan_enabled"]:
    interval = float(st.session_state["auto_scan_hours"])*3600.0
    if now - float(st.session_state["auto_last_ts"]) >= interval:
        base = binance_top100_by_quote_volume()
        if not base.empty:
            syms=base["id"].tolist()
            with st.spinner("⏱️ Auto-Scan …"):
                tdf, tcache = compute_for_symbols(syms, "Top100")
            # Rang übernehmen
            if not tdf.empty:
                rank_map = dict(zip(base["id"], base["rank"]))
                tdf["rank"] = tdf["id"].map(rank_map).fillna(tdf["rank"])
            st.session_state["top100_df"]=tdf
            st.session_state["history_cache"].update(tcache)
            st.session_state["top100_last_sync_ts"]=time.time()
            st.session_state["auto_last_ts"]=time.time()
            if st.session_state["alerts_enabled"]:
                # Alerts + aktiven Coin setzen (den ersten Treffer)
                hits = tdf[tdf["Entry_Signal"]==True].copy()
                alert_entries(hits, "Top-100")
                if not hits.empty:
                    first_id = str(hits.iloc[0]["id"])
                    st.session_state["active_from_table"] = first_id

# ================== Eine Tabelle (Union) ==================
st.markdown("---")
lb = st.container()
cA,cB = lb.columns([6,1])
cA.subheader("📊 Watchlist + Top-100 (eine Tabelle)")
# Letzter Sync
ts=st.session_state.get("top100_last_sync_ts",0.0)
if ts:
    cB.caption(f"Letzter Top-100-Sync: {time.strftime('%Y-%m-%d %H:%M', time.gmtime(ts))} UTC")

watch_df = st.session_state.get("signals_cache", pd.DataFrame())
top_df    = st.session_state.get("top100_df", pd.DataFrame())

# Union mit Priorität: Watchlist überschreibt Top100 bei gleicher id
union = pd.concat([watch_df.assign(_prio=0), top_df.assign(_prio=1)], ignore_index=True)
if not union.empty:
    union.sort_values(by=["id","_prio"], inplace=True)
    union = union.drop_duplicates(subset=["id"], keep="first").drop(columns=["_prio"])

# Filter
f = st.radio("Filter", ["Alle", "Nur Entry-Signal", "Nur Watchlist"], horizontal=True)
if f=="Nur Entry-Signal":
    union = union[(union["Entry_Signal"]==True) & (union["status"]=="ok")]
elif f=="Nur Watchlist":
    union = union[union["universe"]=="Watchlist"]

# Query-Param für aktiven Coin übernehmen (ersetzt deprecated experimental_get_query_params)
if not st.session_state.get("active_from_table"):
    q = st.query_params.get("coin")
    if q:
        st.session_state["active_from_table"] = str(q)

if union.empty:
    st.info("Noch keine Daten. Scanne Watchlist oder aktualisiere Top-100.")
else:
    # Anzeigeformat
    display = union.copy()
    for c in ["price","MA20","MA50","Resistance","Support"]:
        if c in display.columns:
            display[c] = pd.to_numeric(display[c], errors="coerce").map(lambda x: fmt_money(x,4) if pd.notna(x) else "")
    if "Vol_Surge_x" in display.columns:
        display["Vol_Surge_x"] = pd.to_numeric(display["Vol_Surge_x"], errors="coerce").map(lambda x: f"{x:.2f}x" if pd.notna(x) else "")

    # --- st-aggrid Konfiguration ---
    gb = GridOptionsBuilder.from_dataframe(
        display[["universe","rank","name","symbol","price","MA20","MA50","Vol_Surge_x","Resistance","Support",
                 "Breakout_MA","Breakout_Resistance","Entry_Signal","status","id"]]
    )
    gb.configure_column("id", hide=True)
    gb.configure_column("universe", headerName="Ursprung", width=110)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_selection(
        selection_mode="single",
        use_checkbox=False,
        row_multi_select=False,
        suppressRowClickSelection=True
    )
    # Doppelklick -> Row selecten (und damit nach Python zurückgeben)
    cb = JsCode("""
        function(e) {
            e.api.forEachNode(function(node){ node.setSelected(false); });
            e.node.setSelected(true);
        }
    """)
    gb.configure_grid_options(onRowDoubleClicked=cb, domLayout='autoHeight')

    grid = AgGrid(
        display,
        gridOptions=gb.build(),
        theme="balham",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True
    )

    # Auswertung Doppelklick (selected_rows) – genau 1 aktiv
    sel = grid.get("selected_rows", [])
    if sel:
        chosen_id = str(sel[0]["id"])
        if st.session_state.get("active_from_table") != chosen_id:
            st.session_state["active_from_table"] = chosen_id
            st.query_params.update({"coin": chosen_id})  # Permalink auf aktiven Coin
            st.experimental_rerun()

    # Reset-Button für aktive Auswahl
    if st.button("Aktive Auswahl zurücksetzen"):
        st.session_state["active_from_table"] = None
        if "coin" in st.query_params: del st.query_params["coin"]
        st.experimental_rerun()

# ================== Chart nur wenn Doppelklick passiert ==================
st.markdown("---")
active = st.session_state.get("active_from_table")
if not active:
    st.info("Kein aktiver Coin. **Doppelklicke** eine Zeile in der Tabelle, um den Chart zu sehen.")
else:
    with st.spinner(f"Historie laden für {active} …"):
        d = st.session_state.get("history_cache", {}).get(active)
        if d is None or d.empty:
            d = load_history(active, st.session_state["days_hist"])
            if isinstance(d, pd.DataFrame) and not d.empty:
                st.session_state["history_cache"][active] = d

    if d is None or d.empty or d.attrs.get("status")!="ok":
        st.warning("Keine Historie verfügbar.")
    else:
        df = d.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["price"]  = pd.to_numeric(df["price"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["timestamp","price"]).set_index("timestamp").sort_index()
        df = df.resample("1D").last().dropna(subset=["price"])

        r,s = levels(df, st.session_state["lookback_res"])
        df["ma20"] = ma(df["price"],20); df["ma50"]=ma(df["price"],50)
        df["vol7"] = df["volume"].rolling(7, min_periods=3).mean()
        df["vol_ratio"] = df["volume"]/df["vol7"]
        df["roll_max_prev"] = df["price"].shift(1).rolling(st.session_state["lookback_res"], min_periods=5).max()
        entry_mask = (
            (df["price"]>df["ma20"]) & (df["ma20"]>df["ma50"]) &
            (df["price"]>df["roll_max_prev"]) &
            (df["vol_ratio"]>=st.session_state["vol_surge_thresh"])
        )
        entries = df[entry_mask].dropna(subset=["price"])

        fig, ax_price = plt.subplots()
        ax_vol = ax_price.twinx()
        ax_price.plot(df.index, df["price"], label="Price", linewidth=1.6)
        ax_price.plot(df.index, df["ma20"], label="MA20", linewidth=1.0)
        ax_price.plot(df.index, df["ma50"], label="MA50", linewidth=1.0)
        if not np.isnan(r): ax_price.axhline(r, linestyle="--", label=f"Resistance {r:.3f}")
        if not np.isnan(s): ax_price.axhline(s, linestyle="--", label=f"Support {s:.3f}")
        if not entries.empty:
            ax_price.scatter(entries.index, entries["price"].astype(float), s=36, zorder=5, color="#16a34a", label="Entry (hist)")
        ax_vol.bar(df.index, df["volume"], alpha=0.28)
        ax_vol.set_ylabel("Volume")

        locator=mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax_price.xaxis.set_major_locator(locator)
        ax_price.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax_price.set_title(f"{active} — Price, MAs, Levels & Volume")
        ax_price.set_xlabel("Date"); ax_price.set_ylabel("USD")
        ax_price.legend(loc="upper left"); plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
