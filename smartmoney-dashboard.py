# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Geschützt (Streamlit)
# v2.4 Performance: Compute-on-Click, Batch-Scan, HTTP Session reuse,
#                   stabile Caches, weniger Reruns/Calls
#
# Secrets (Streamlit → Advanced settings → Secrets):
# APP_PASSWORD = "DeinStarkesPasswort"
# TELEGRAM_BOT_TOKEN = "123:abc"   # optional
# TELEGRAM_CHAT_ID   = "123456789" # optional
# -------------------------------------------------------------
"""Smart Money Dashboard.

Dieses Skript bündelt alle aktuellen Anpassungen (Support/Resistance,
Watchlist-Persistenz, Easyfi-Auswahl, Telegram-Alerts) in einer einzigen Datei,
sodass der komplette Code bei Bedarf einfach kopiert und in Streamlit
eingesetzt werden kann.
"""

import base64
import json
import math
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----------------- App Config -----------------
st.set_page_config(page_title="Smart Money Dashboard — Geschützt", layout="wide")

# ----------------- Session Helper ------------------
def save_state(keys):
    for k in keys:
        if k in st.session_state:
            st.session_state[f"_saved_{k}"] = st.session_state[k]

def restore_state(keys):
    for k in keys:
        saved_key = f"_saved_{k}"
        if saved_key in st.session_state:
            st.session_state[k] = st.session_state[saved_key]

# ----------------- Auth Gate ------------------
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
            time.sleep(0.3)
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

# ----------------- Constants ------------------
FIAT = "usd"
CG_BASE = "https://api.coingecko.com/api/v3"

# ----------------- HTTP Session (Keep-Alive) ------------
@st.cache_resource(show_spinner=False)
def get_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "smartmoney-dashboard/1.0 (+streamlit)"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# ----------------- HTTP helper (robust + Diagnose) -------
def _get_json(url, params=None, timeout=12, retries=2, backoff=1.6):
    """
    Schneller, fails fast:
    - kurzer Timeout (12s)
    - wenige Retries
    - 429/50x werden kurz gebackofft, alles andere bricht sofort ab
    """
    session = get_http()
    last_err = ""
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return {"ok": True, "json": r.json(), "status": 200}
            # gezielt nur bei Rate-Limit/Serverfehlern kurz warten
            if r.status_code in (429, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(backoff * (i + 1))
                continue
            # alle anderen Fehler sofort zurückgeben
            return {"ok": False, "json": None, "status": r.status_code, "error": r.text[:300]}
        except requests.RequestException as e:
            last_err = str(e)[:200]
            time.sleep(backoff * (i + 1))
            continue
    return {"ok": False, "json": None, "status": None, "error": last_err or "request failed"}


# ----------------- Helpers --------------------
def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

@st.cache_data(ttl=3600, show_spinner=True)
def cg_top_coins(limit: int = 100) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["id","symbol","name","market_cap","rank"])
    df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["id"])
    df["market_cap"] = pd.to_numeric(df.get("market_cap"), errors="coerce")
    df = (
        df.sort_values("market_cap", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    return df


@st.cache_data(ttl=300, show_spinner=False)
def cg_search_coins(query: str) -> pd.DataFrame:
    q = (query or "").strip()
    if len(q) < 2:
        return pd.DataFrame(columns=["id", "symbol", "name", "market_cap_rank"])

    resp = _get_json(f"{CG_BASE}/search", {"query": q}, timeout=8, retries=1, backoff=1.2)
    if not resp.get("ok"):
        return pd.DataFrame(columns=["id", "symbol", "name", "market_cap_rank"])

    coins = resp.get("json", {}).get("coins", [])
    if not coins:
        return pd.DataFrame(columns=["id", "symbol", "name", "market_cap_rank"])

    df = pd.DataFrame(coins)
    if df.empty or "id" not in df.columns:
        return pd.DataFrame(columns=["id", "symbol", "name", "market_cap_rank"])

    keep_cols = {"id": "id", "name": "name", "symbol": "symbol", "market_cap_rank": "market_cap_rank"}
    df = df[[c for c in keep_cols if c in df.columns]].rename(columns=keep_cols)
    df = df.dropna(subset=["id"]).drop_duplicates(subset=["id"])
    df["market_cap_rank"] = pd.to_numeric(df.get("market_cap_rank"), errors="coerce")
    df = df.sort_values("market_cap_rank", na_position="last").head(20)
    return df.reset_index(drop=True)

@st.cache_data(ttl=1200, show_spinner=False)
def cg_market_chart(coin_id: str, days: int = 180) -> pd.DataFrame:
    """Historische Preise/Volumen – zuerst CoinGecko (schnell), sonst Fallback Binance."""
    def _empty(status_text: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=["timestamp", "price", "volume"])
        df.attrs["status"] = status_text
        return df

    # ---------- 1) Schneller Versuch CoinGecko ----------
    resp = _get_json(
        f"{CG_BASE}/coins/{coin_id}/market_chart",
        {"vs_currency": FIAT, "days": days, "interval": "daily"},
        timeout=10, retries=1, backoff=1.2  # kurz und knapp
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
            df = df[["timestamp","price","volume"]].dropna()
            if not df.empty:
                df = df.sort_values("timestamp").tail(int(days)+1)
                df.attrs["status"] = "ok_cg"
                return df
        # wenn leer -> gleich Fallback

    # ---------- 2) Binance-Fallback (USDT-Paare) ----------
    symbol_map = {
        "bitcoin": "BTCUSDT",
        "ethereum": "ETHUSDT",
        "solana": "SOLUSDT",
        "arbitrum": "ARBUSDT",
        "render-token": "RNDRUSDT",
        "bittensor": "TAOUSDT",
        # raydium kommt i. d. R. generisch über Top-500 Mapping -> "RAYUSDT"
    }
    sym = symbol_map.get(coin_id)
    if not sym:
        try:
            top = cg_top_coins(limit=500)
            sym = top.loc[top["id"] == coin_id, "symbol"].str.upper().iloc[0] + "USDT"
        except Exception:
            return _empty("no_symbol")

    # versuche mehrere Binance-Hosts, falls 451/403
    BINANCE_KLINES_ENDPOINTS = [
        "https://api.binance.com/api/v3/klines",
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]

    last_status = None
    klines_json = None
    for base in BINANCE_KLINES_ENDPOINTS:
        r = get_http().get(
            base,
            params={"symbol": sym, "interval": "1d", "limit": min(1000, int(days)+5)},
            timeout=8
        )
        last_status = r.status_code
        if r.status_code == 200:
            try:
                klines_json = r.json()
            except Exception:
                klines_json = None
            if klines_json:
                break
        # bei 451/403/5xx einfach nächsten Host probieren
        time.sleep(0.3)

    if not klines_json:
        return _empty(f"err_binance:{last_status}")

    try:
        df = pd.DataFrame(klines_json, columns=[
            "openTime","open","high","low","close","volume",
            "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True, errors="coerce")
        df["price"]     = pd.to_numeric(df["close"], errors="coerce")
        df["volume"]    = pd.to_numeric(df["volume"], errors="coerce")
        df = df[["timestamp","price","volume"]].dropna()
        if df.empty():
            return _empty("empty_binance")
        df = df.sort_values("timestamp").tail(int(days)+1)
        df.attrs["status"] = "ok_binance"
        return df
    except Exception:
        return _empty("parse_binance")

@st.cache_data(ttl=600, show_spinner=False)
def cg_simple_price(ids) -> pd.DataFrame:
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

def calc_local_levels(dfd: pd.DataFrame, lookback: int = 20):
    if dfd.empty:
        return (np.nan, np.nan)
    d = dfd.iloc[:-1].tail(lookback)  # ohne letzte Kerze
    if d.empty:
        return (np.nan, np.nan)
    return float(d["price"].max()), float(d["price"].min())

def volume_signals(dfd: pd.DataFrame) -> dict:
    out = {"vol_ratio_1d_vs_7d": np.nan, "distribution_risk": False, "price_chg_7d": np.nan}
    if dfd.empty or len(dfd) < 8:  # mind. 1 Woche
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

# ----------------- Persistenz (Mobile/iPhone) --------------------
PERSIST_KEYS = [
    "selected_ids",
    "min_mktcap",
    "min_volume",
    "vol_surge_thresh",
    "lookback_res",
    "alerts_enabled",
    "days_hist",
    "batch_size_slider",
]


def _decode_state(param_value: str) -> dict:
    try:
        raw = base64.urlsafe_b64decode(param_value.encode("utf-8"))
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _encode_state(state: dict) -> str:
    raw = json.dumps(state, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")


def load_persisted_state() -> None:
    params = st.experimental_get_query_params()
    encoded = params.get("state", [None])[0]
    if not encoded:
        return
    data = _decode_state(encoded)
    if not data:
        return
    for k in PERSIST_KEYS:
        if k in data:
            st.session_state.setdefault(k, data[k])
    st.session_state["_persisted_state"] = params.get("state", [None])[0]


def persist_state() -> None:
    state = {k: st.session_state.get(k) for k in PERSIST_KEYS}
    encoded = _encode_state(state)
    if st.session_state.get("_persisted_state") == encoded:
        return
    st.experimental_set_query_params(state=encoded)
    st.session_state["_persisted_state"] = encoded


load_persisted_state()

# ----------------- Sidebar --------------------
st.sidebar.header("Settings")

# Defaults in Session (Persistenz)
for k, v in {
    "selected_ids": [],
    "min_mktcap": 300_000_000,
    "min_volume": 50_000_000,
    "vol_surge_thresh": 1.5,
    "lookback_res": 20,
    "alerts_enabled": True,
    "days_hist": 60,            # etwas niedriger als vorher
    "batch_size_slider": 3,
    "scan_index": 0
}.items():
    st.session_state.setdefault(k, v)

days_hist = st.sidebar.slider("Historie (Tage)", 60, 365, int(st.session_state["days_hist"]), 15, key="days_hist")
# NICHT erneut in session_state schreiben – der Slider pflegt key="days_hist" selbst

# Watchlist
ensure_ids = {"easy": "easyfi", "easyfi": "easyfi"}
top_df = cg_top_coins(limit=100)

search_ids: list[str] = []
search_query = st.sidebar.text_input(
    "Weitere Coins suchen (Echtzeit)",
    key="search_query",
    placeholder="Mindestens 2 Zeichen — z.B. pepe oder arbitrum",
)

if search_query and len(search_query.strip()) >= 2:
    search_df = cg_search_coins(search_query.strip())
    if search_df.empty:
        st.sidebar.info("Keine Treffer für diese Suche gefunden.")
    else:
        def _format_search_label(row: pd.Series) -> str:
            rank_val = row.get("market_cap_rank")
            prefix = f"Rang {int(rank_val):03d} · " if pd.notna(rank_val) else ""
            return f"{prefix}{row['name']} ({str(row['symbol']).upper()}) — {row['id']}"

        search_df["label"] = search_df.apply(_format_search_label, axis=1)
        default_search_ids = [
            cid for cid in st.session_state.get("selected_ids", [])
            if cid in search_df["id"].tolist()
        ]
        default_search_labels = search_df[search_df["id"].isin(default_search_ids)]["label"].tolist()
        selected_search_labels = st.sidebar.multiselect(
            "Suchtreffer hinzufügen",
            options=search_df["label"].tolist(),
            default=default_search_labels,
            help="Treffer anklicken, um sie deiner Watchlist hinzuzufügen.",
            key="search_results_select",
        )
        label_to_id_search = dict(zip(search_df["label"], search_df["id"]))
        search_ids = [
            ensure_ids.get(label_to_id_search[label], label_to_id_search[label])
            for label in selected_search_labels
        ]
else:
    st.sidebar.caption("Mindestens 2 Zeichen eingeben, um zusätzliche Coins in Echtzeit zu finden.")

forced_source = list(st.session_state.get("selected_ids", [])) + search_ids
forced_ids = list({ensure_ids.get(cid, cid) for cid in forced_source if cid})
forced_ids.append("easyfi")
forced_ids = sorted(set(filter(None, forced_ids)))

if forced_ids:
    extra = cg_simple_price(forced_ids)
    if not extra.empty:
        extra = (
            extra[["id", "symbol", "name", "market_cap"]]
            .drop_duplicates(subset=["id"])
        )
        extra["market_cap"] = pd.to_numeric(extra.get("market_cap"), errors="coerce")
        extra["rank"] = np.nan
        if top_df.empty:
            top_df = extra
        else:
            top_df = (
                pd.concat([top_df, extra], ignore_index=True)
                .drop_duplicates(subset=["id"], keep="first")
                .sort_values(["rank", "market_cap"], ascending=[True, False], na_position="last")
                .reset_index(drop=True)
            )

if top_df.empty:
    st.sidebar.warning("Top-Liste konnte nicht geladen werden (API-Limit?). Fallback-Auswahl.")
    default_ids = ["bitcoin","ethereum","solana","arbitrum","render-token","bittensor"]
    fallback_defaults_source = list(
        dict.fromkeys((st.session_state["selected_ids"] or default_ids[:3]) + search_ids)
    )
    fallback_defaults = [cid for cid in fallback_defaults_source if cid in default_ids]
    selected_labels = st.sidebar.multiselect(
        "Watchlist (Fallback)",
        options=default_ids,
        default=list(dict.fromkeys((st.session_state["selected_ids"] or default_ids[:3]) + search_ids)),
        key="watchlist_fallback"
    )
    selected_ids = selected_labels
else:
    top_df = top_df.sort_values(["rank", "market_cap"], ascending=[True, False], na_position="last").reset_index(drop=True)

    def _format_top_label(row: pd.Series) -> str:
        rank_val = row.get("rank")
        prefix = f"Rang {int(rank_val):03d} · " if pd.notna(rank_val) else ""
        return f"{prefix}{row['name']} ({str(row['symbol']).upper()}) — {row['id']}"

    top_df["label"] = top_df.apply(_format_top_label, axis=1)
    default_seed = st.session_state["selected_ids"] or [
        "bitcoin",
        "ethereum",
        "solana",
        "arbitrum",
        "render-token",
        "bittensor",
        "easyfi",
    ]
    default_ids = list(dict.fromkeys(default_seed + search_ids))
    default_labels = top_df[top_df["id"].isin(default_ids)]["label"].tolist()
    selected_labels = st.sidebar.multiselect(
        "Watchlist auswählen (Top 100)",
        options=top_df["label"].tolist(),
        default=default_labels,
        help="Tippe Name oder Ticker, wähle per Klick.",
        key="watchlist_top"
    )
    label_to_id = dict(zip(top_df["label"], top_df["id"]))
    selected_ids = [label_to_id[l] for l in selected_labels]
    selected_ids = [ensure_ids.get(cid, cid) for cid in selected_ids]

selected_ids.extend(search_ids)

manual = st.sidebar.text_input("Zusätzliche ID (optional)", value="", key="manual_id")
if manual.strip():
    selected_ids.append(ensure_ids.get(manual.strip(), manual.strip()))
selected_ids = list(dict.fromkeys(selected_ids))
if not selected_ids:
    selected_ids = ["bitcoin","ethereum"]

min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=int(st.session_state["min_mktcap"]), step=50_000_000, key="min_mc")
min_volume = st.sidebar.number_input("Min 24h Volume (USD)", min_value=0, value=int(st.session_state["min_volume"]), step=10_000_000, key="min_vol")
vol_surge_thresh = st.sidebar.slider("Vol Surge vs 7d (x)", 1.0, 5.0, float(st.session_state["vol_surge_thresh"]), 0.1, key="vol_surge")
lookback_res = st.sidebar.slider("Lookback für Widerstand/Support (Tage)", 10, 60, int(st.session_state["lookback_res"]), 1, key="lookback")
alerts_enabled = st.sidebar.checkbox("Telegram-Alerts aktivieren (Secrets nötig)", value=bool(st.session_state["alerts_enabled"]), key="alerts_on")

# Scan-Steuerung
scan_now = st.sidebar.button("🔔 Watchlist jetzt scannen", key="scan_btn")

# Batch-Regler (einmalig, mit Key)
batch_size = st.sidebar.slider("Coins pro Scan (Batchgröße)", 2, 10, int(st.session_state["batch_size_slider"]), 1, key="batch_size_slider")
if st.sidebar.button("🔁 Batch zurücksetzen", key="reset_batch_btn"):
    st.session_state["scan_index"] = 0

# persist basic settings
st.session_state["selected_ids"] = selected_ids
st.session_state["min_mktcap"]   = min_mktcap
st.session_state["min_volume"]   = min_volume
st.session_state["vol_surge_thresh"] = vol_surge_thresh
st.session_state["lookback_res"] = lookback_res
st.session_state["alerts_enabled"] = alerts_enabled

persist_state()

st.caption("🔒 Passwortschutz aktiv — setze `APP_PASSWORD` in Secrets.  •  Alerts via Telegram (optional).  •  Scans laufen nur auf Klick (Compute-on-Click).")

# ----------------- Checklist ------------------
with st.expander("📋 Tägliche Checkliste", expanded=False):
    st.markdown("""\
**Morgens:** Scan → Kandidaten (Breakout + Vol-Surge) notieren, Funding/TVL querprüfen
**Mittags:** Entry nur bei Preis > MA20 > MA50 **und** Vol-Surge ≥ Schwelle **und** Breakout über Widerstand
**Abends:** Volumen-Trend prüfen, Trailing Stop nachziehen, Teilgewinne sichern
""")

# ----------------- Snapshot (preiswert) -------------------
# Nur einmal pro Rerun, gecacht (600s)
spot = cg_simple_price(selected_ids)
if not spot.empty:
    filt = spot[(spot["market_cap"] >= min_mktcap) & (spot["volume_24h"] >= min_volume)]
    st.subheader("📊 Snapshot (Filter)")
    st.dataframe(
        filt.rename(columns={
            "id":"ID","symbol":"Symbol","name":"Name","price":"Price",
            "market_cap":"MktCap","volume_24h":"Vol 24h","price_change_percentage_24h":"% 24h"}),
        use_container_width=True, hide_index=True
    )

# ----------------- Signals: Compute-on-Click ----------------
# Wir scannen NUR bei Button-Klick. Sonst zeigen wir das letzte Resultat (falls vorhanden).
if "signals_cache" not in st.session_state:
    st.session_state["signals_cache"] = pd.DataFrame()
if "history_cache" not in st.session_state:
    st.session_state["history_cache"] = {}

def run_scan(selected_ids, days_hist, batch_size, vol_surge_thresh, lookback_res):
    rows, history_cache = [], {}

    start = st.session_state.get("scan_index", 0)
    end = min(start + batch_size, len(selected_ids))
    batch = selected_ids[start:end]

    if not batch:
        st.warning("Keine Coins im aktuellen Batch. Starte Scan erneut.")
        return pd.DataFrame(), {}, start, end

    st.info(f"⏳ Scanne Coins {start+1}–{end} von {len(selected_ids)} ...")
    PAUSE_BETWEEN = 0.6  # sanfte Rate-Limitierung

    for cid in batch:
        time.sleep(PAUSE_BETWEEN)
        hist = cg_market_chart(cid, days=days_hist)
        status_val = hist.attrs.get("status", "ok") if hist is not None else "no_df"

        # --- Fehlerfall: leere/fehlerhafte Daten -> nur EIN Row, dann weiter ---
        if (hist is None) or hist.empty or (status_val != "ok_cg" and status_val != "ok_binance" and status_val != "ok"):
            rows.append({
                "id": cid, "price": np.nan, "MA20": np.nan, "MA50": np.nan,
                "Breakout_MA": False, "Vol_Surge_x": np.nan,
                "Resistance": np.nan, "Support": np.nan,
                "Breakout_Resistance": False, "Distribution_Risk": False,
                "Entry_Signal": False,
                "status": status_val or "no data",
                "source": status_val  # zeigt err/empty/err_binance/... an
            })
            continue  # <<< WICHTIG

        # --- Erfolgsfall ---
        history_cache[cid] = hist

        dfd = hist.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd = dfd.set_index("timestamp").sort_index().resample("1D").last().dropna()

        t_sig = trend_signals(dfd)
        v_sig = volume_signals(dfd)
        resistance, support = calc_local_levels(dfd, lookback=lookback_res)

        last = dfd.iloc[-1]
        price = float(last["price"])
        volsurge = v_sig["vol_ratio_1d_vs_7d"]
        is_valid_vol = not np.isnan(volsurge)
        breakout_res = bool(price >= (resistance * 1.0005)) if not math.isnan(resistance) else False
        entry_ok = bool(t_sig["breakout_ma"] and is_valid_vol and (volsurge >= vol_surge_thresh))

        src = hist.attrs.get("status", "ok").replace("ok_", "")  # cg oder binance

        rows.append({
            "id": cid, "price": price,
            "MA20": t_sig["ma20"], "MA50": t_sig["ma50"],
            "Breakout_MA": t_sig["breakout_ma"], "Vol_Surge_x": volsurge,
            "Resistance": resistance, "Support": support,
            "Breakout_Resistance": breakout_res,
            "Distribution_Risk": v_sig["distribution_risk"],
            "Entry_Signal": entry_ok and breakout_res,
            "status": "ok",
            "source": src
        })

    signals_df = pd.DataFrame(rows)
    # Fortschritt für nächsten Scan merken
    st.session_state["scan_index"] = end % max(1, len(selected_ids))
    return signals_df, history_cache, start, end

# Scan ausführen nur bei Klick (oder beim allerersten Start, wenn noch kein Cache existiert)
do_scan = scan_now or st.session_state["signals_cache"].empty

if do_scan:
    sig, hist_cache, start, end = run_scan(selected_ids, days_hist, batch_size, vol_surge_thresh, lookback_res)
    # Cache aktualisieren (append/merge – wir halten nur den letzten Batch sichtbar)
    st.session_state["signals_cache"] = sig
    st.session_state["history_cache"] = hist_cache
else:
    sig = st.session_state["signals_cache"]
    start = st.session_state.get("scan_index", 0)
    end = min(start + batch_size, len(selected_ids))

signals_df = sig.copy()
st.subheader("🔎 Signals & Levels (Batch)")

def _row_style(row):
    if bool(row.get("Entry_Signal", False)): return ['background-color: #e6ffed'] * len(row)   # grün
    if bool(row.get("Distribution_Risk", False)): return ['background-color: #ffecec'] * len(row)  # rot
    if bool(row.get("Breakout_MA", False)) or bool(row.get("Breakout_Resistance", False)):
        return ['background-color: #fff9e6'] * len(row)  # gelb
    return [''] * len(row)

if not signals_df.empty:
    num_cols = ["price","MA20","MA50","Vol_Surge_x","Resistance","Support"]
    for c in num_cols:
        if c in signals_df.columns:
            signals_df[c] = pd.to_numeric(signals_df[c], errors="coerce")
    styled = signals_df.style.apply(_row_style, axis=1).format({
        "price": "{:.4f}", "MA20": "{:.4f}", "MA50": "{:.4f}",
        "Vol_Surge_x": "{:.2f}", "Resistance": "{:.4f}", "Support": "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Alerts nur auf Klick und nur für den aktuellen Batch
    if scan_now and st.session_state["alerts_enabled"]:
        sent = []
        for _, r in signals_df.iterrows():
            if r.get("Entry_Signal", False):
                ok = send_telegram(
                    f"🚨 Entry-Signal: {r['id']} | Preis: ${r['price']:.3f} | "
                    f"Breakout über Widerstand {r['Resistance']:.3f} | Vol-Surge: {r['Vol_Surge_x']:.2f}x"
                )
                sent.append((r["id"], ok))
        if not sent:
            st.info("Keine Entry-Signale im aktuellen Batch.")
        elif any(ok for _, ok in sent):
            st.success("Telegram-Alerts gesendet (Batch).")
        else:
            st.warning("Alert-Versand fehlgeschlagen (TELEGRAM_* Secrets prüfen).")

# Fortschritts-Hinweis
if len(selected_ids) > 0:
    if end == len(selected_ids):
        st.success("✅ Batch-Scan: Ende der Liste erreicht. Nächster Klick startet wieder vorn.")
    else:
        nxt_end = min(end + batch_size, len(selected_ids))
        st.info(f"➡️ Nächster Scan lädt Coins {end+1}–{nxt_end} von {len(selected_ids)}.")

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
