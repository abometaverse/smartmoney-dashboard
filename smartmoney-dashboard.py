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
def _get_json(url, params=None, timeout=40, retries=6, backoff=2.0):
    """HTTP GET mit Retry/Backoff. Rückgabe: dict mit ok/json/status/error."""
    session = get_http()
    last_err = ""
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return {"ok": True, "json": r.json(), "status": 200}
            last_err = f"HTTP {r.status_code}"
            if r.status_code in (429, 502, 503):
                time.sleep(backoff * (i + 1))
                continue
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
    """Historische Preise/Volumen. Bei API-Fehler -> leerer DF mit df.attrs['status'] != 'ok'."""
    resp = _get_json(
        f"{CG_BASE}/coins/{coin_id}/market_chart",
        {"vs_currency": FIAT, "days": days, "interval": "daily"}
    )

    def _empty(status_text: str) -> pd.DataFrame:
        df = pd.DataFrame(columns=["timestamp", "price", "volume"])
        df.attrs["status"] = status_text
        return df

    if not resp.get("ok"):
        return _empty(f"err:{resp.get('status')}")
    data = resp["json"]
    prices = data.get("prices", [])
    vols   = data.get("total_volumes", [])
    if not prices:
        return _empty("empty")

    dfp = pd.DataFrame(prices, columns=["ts","price"])
    dfv = pd.DataFrame(vols,    columns=["ts","volume"])
    df = dfp.merge(dfv, on="ts", how="left")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    df = df[["timestamp","price","volume"]].dropna()
    df.attrs["status"] = "ok"
    return df

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
    "days_hist": 120,            # etwas niedriger als vorher
    "batch_size_slider": 3,
    "scan_index": 0
}.items():
    st.session_state.setdefault(k, v)

days_hist = st.sidebar.slider("Historie (Tage)", 60, 365, int(st.session_state["days_hist"]), 15, key="days_hist")
st.session_state["days_hist"] = days_hist

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

st.caption("🔒 Passwortschutz aktiv — setze `APP_PASSWORD` in Secrets.  •  Alerts via Telegram (optional).  •  Scans laufen nur auf Klick (Compute-on-Click).")

# ----------------- Checklist ------------------
with st.expander("📋 Tägliche Checkliste", expanded=False):
    st.markdown("""
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
    PAUSE_BETWEEN = 0.5  # sanfte Rate-Limitierung

    for cid in batch:
        time.sleep(PAUSE_BETWEEN)
        hist = cg_market_chart(cid, days=days_hist)
        status_val = hist.attrs.get("status", "ok") if hist is not None else "no_df"

        if hist is None or hist.empty or (status_val != "ok"):
            rows.append({"id": cid, "price": np.nan, "MA20": np.nan, "MA50": np.nan,
                         "Breakout_MA": False, "Vol_Surge_x": np.nan,
                         "Resistance": np.nan, "Support": np.nan,
                         "Breakout_Resistance": False, "Distribution_Risk": False,
                         "Entry_Signal": False, "status": status_val or "no data"})
            continue

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

        rows.append({
            "id": cid, "price": price,
            "MA20": t_sig["ma20"], "MA50": t_sig["ma50"],
            "Breakout_MA": t_sig["breakout_ma"], "Vol_Surge_x": volsurge,
            "Resistance": resistance, "Support": support,
            "Breakout_Resistance": breakout_res,
            "Distribution_Risk": v_sig["distribution_risk"],
            "Entry_Signal": entry_ok and breakout_res, "status": "ok"
        })

    signals_df = pd.DataFrame(rows)
    # Fortschritt für nächsten Scan merken
    st.session_state["scan_index"] = (end) % max(1, len(selected_ids))
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
    if d is None or d.empty:
        # Nur nachladen, wenn ausdrücklich Coin gewechselt wurde (kein Massenscan)
        d = cg_market_chart(coin_select, days=st.session_state["days_hist"])
    if d is None or d.empty or (d.attrs.get("status","ok") != "ok"):
        st.warning("Keine Historie verfügbar (API-Limit oder leere Daten).")
    else:
        dfd = d.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd = dfd.set_index("timestamp").sort_index().resample("1D").last().dropna()

        r, s = calc_local_levels(dfd, lookback=lookback_res)
        v_sig = volume_signals(dfd)

        fig, ax = plt.subplots()
        ax.plot(dfd.index, dfd["price"], label="Price")
        ax.plot(dfd.index, ma(dfd["price"],20), label="MA20")
        ax.plot(dfd.index, ma(dfd["price"],50), label="MA50")
        if not np.isnan(r): ax.axhline(r, linestyle="--", label=f"Resistance {r:.3f}")
        if not np.isnan(s): ax.axhline(s, linestyle="--", label=f"Support {s:.3f}")
        ax.set_title(f"{coin_select} — Price & Levels"); ax.set_xlabel("Date"); ax.set_ylabel("USD"); ax.legend()
        st.pyplot(fig, use_container_width=True)

        fig2, ax2 = plt.subplots()
        ax2.bar(dfd.index, dfd["volume"])
        ax2.set_title(f"{coin_select} — Daily Volume"); ax2.set_xlabel("Date"); ax2.set_ylabel("USD")
        st.pyplot(fig2, use_container_width=True)

        if v_sig["distribution_risk"]:
            st.warning("Distribution-Risk: Preis ↑ bei Volumen < 0.8× 7d-Ø.")
        else:
            st.success("Volumen ok (keine Distribution-Anzeichen).")