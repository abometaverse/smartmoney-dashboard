# smartmoney-dashboard.py
# -------------------------------------------------------------
# Smart Money Dashboard — Geschützt (Streamlit)
# v2.3: PW-Login, Top-500 Auswahl, robuste API (Retry + Diagnose),
#       farbige "Signals & Levels", Persistenz nach Logout,
#       Telegram-Alerts (optional), Risk-Tools
#
# In Streamlit (Advanced settings → Secrets) als TOML setzen:
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
        st.error("Konfiguration fehlt: Setze `APP_PASSWORD` unter Settings → Secrets (Streamlit Cloud).")
        st.stop()

    if st.session_state.get("AUTH_OK", False):
        top = st.container()
        col1, col2 = top.columns([6,1])
        col1.success("Zugriff gewährt.")
        if col2.button("Logout"):
            save_state([
                "selected_ids", "min_mktcap", "min_volume",
                "vol_surge_thresh", "lookback_res", "alerts_enabled", "days_hist"
            ])
            st.session_state["AUTH_OK"] = False
            st.success("Einstellungen gespeichert.")
            time.sleep(0.5)
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
                    "vol_surge_thresh", "lookback_res", "alerts_enabled", "days_hist"
                ])
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    st.stop()

auth_gate()

# ----------------- Constants ------------------
FIAT = "usd"
CG_BASE = "https://api.coingecko.com/api/v3"

# ----------------- HTTP helper (robust + Diagnose) -------
def _get_json(url, params=None, timeout=40, retries=5, backoff=1.8):
    """HTTP GET mit Retry/Backoff. Rückgabe: dict mit ok/json/status/error."""
    headers = {"User-Agent": "smartmoney-dashboard/1.0 (+streamlit)"}
    last_err = ""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return {"ok": True, "json": r.json(), "status": 200}
            last_err = f"HTTP {r.status_code}"
            if r.status_code in (429, 502, 503):
                time.sleep(backoff * (i+1))
                continue
            return {"ok": False, "json": None, "status": r.status_code, "error": r.text[:300]}
        except requests.RequestException as e:
            last_err = str(e)[:200]
            time.sleep(backoff * (i+1))
            continue
    return {"ok": False, "json": None, "status": None, "error": last_err or "request failed"}

# ----------------- Helpers --------------------
def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

@st.cache_data(ttl=3600, show_spinner=True)
def cg_top_coins(limit: int = 500) -> pd.DataFrame:
    """Top-500 Coins (id, symbol, name, market_cap) – für Auswahl/Suche."""
    rows = []
    per_page = 250
    pages = int(np.ceil(limit / per_page))
    for page in range(1, pages + 1):
        resp = _get_json(
            f"{CG_BASE}/coins/markets",
            {
                "vs_currency": FIAT, "order": "market_cap_desc",
                "per_page": per_page, "page": page, "sparkline": "false",
            },
        )
        if not resp.get("ok"):
            break
        part = pd.DataFrame(resp["json"])[["id", "symbol", "name", "market_cap"]]
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["id","symbol","name","market_cap"])
    df = pd.concat(rows, ignore_index=True)
    df = df.drop_duplicates(subset=["id"]).head(limit)
    return df

@st.cache_data(ttl=900, show_spinner=False)
def cg_market_chart(coin_id: str, days: int = 180) -> pd.DataFrame:
    """
    Historische Preise/Volumen. Bei API-Fehler -> leerer DF mit df.attrs['status'] != 'ok'.
    """
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

@st.cache_data(ttl=300, show_spinner=False)
def cg_simple_price(ids) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    resp = _get_json(
        f"{CG_BASE}/coins/markets",
        {
            "vs_currency": FIAT, "ids": ",".join(ids),
            "order":"market_cap_desc","per_page": max(1,len(ids)),"page":1,"sparkline":"false"
        }
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
    d = dfd.copy().reset_index(drop=True)
    hist = d.iloc[:-1].tail(lookback)
    if hist.empty:
        return (np.nan, np.nan)
    resistance = float(hist["price"].max())
    support    = float(hist["price"].min())
    return resistance, support

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
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": msg}, timeout=10)
        return True
    except Exception:
        return False

def trailing_stop(current_high: float, trail_pct: float) -> float:
    return current_high * (1 - trail_pct/100.0)

# ----------------- Sidebar --------------------
st.sidebar.header("Settings")

# Defaults in Session (für Persistenz)
for k, v in {
    "selected_ids": [],
    "min_mktcap": 300_000_000,
    "min_volume": 50_000_000,
    "vol_surge_thresh": 1.5,
    "lookback_res": 20,
    "alerts_enabled": True,
    "days_hist": 180
}.items():
    st.session_state.setdefault(k, v)

days_hist = st.sidebar.slider("Historie (Tage)", 60, 365, int(st.session_state["days_hist"]), 15)
st.session_state["days_hist"] = days_hist

# Top-500 Auswahl mit Suche
top_df = cg_top_coins(limit=500)
if top_df.empty:
    st.sidebar.warning("Top-Liste konnte nicht geladen werden (API-Limit?). Fallback-Auswahl.")
    default_ids = ["bitcoin","ethereum","solana","arbitrum","render-token","bittensor"]
    selected_labels = st.sidebar.multiselect(
        "Watchlist (Fallback)",
        options=default_ids,
        default=st.session_state["selected_ids"] or default_ids[:3],
    )
    selected_ids = selected_labels
else:
    top_df["label"] = top_df.apply(lambda r: f"{r['name']} ({str(r['symbol']).upper()}) — {r['id']}", axis=1)
    if st.session_state["selected_ids"]:
        default_labels = top_df[top_df["id"].isin(st.session_state["selected_ids"])]["label"].tolist()
    else:
        default_ids = ["bitcoin","ethereum","solana","arbitrum","render-token","bittensor"]
        default_labels = top_df[top_df["id"].isin(default_ids)]["label"].tolist()

    selected_labels = st.sidebar.multiselect(
        "Watchlist auswählen (Top 500, Suche per Tippen)",
        options=top_df["label"].tolist(),
        default=default_labels,
        help="Tippe Name oder Ticker, wähle per Klick."
    )
    label_to_id = dict(zip(top_df["label"], top_df["id"]))
    selected_ids = [label_to_id[l] for l in selected_labels]

manual = st.sidebar.text_input("Zusätzliche ID (optional)", value="", help="Nur wenn Coin nicht in Top 500.")
if manual.strip():
    selected_ids.append(manual.strip())
if not selected_ids:
    selected_ids = ["bitcoin","ethereum"]

min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=int(st.session_state["min_mktcap"]), step=50_000_000)
min_volume = st.sidebar.number_input("Min 24h Volume (USD)", min_value=0, value=int(st.session_state["min_volume"]), step=10_000_000)
vol_surge_thresh = st.sidebar.slider("Vol Surge vs 7d (x)", 1.0, 5.0, float(st.session_state["vol_surge_thresh"]), 0.1)
lookback_res = st.sidebar.slider("Lookback für Widerstand/Support (Tage)", 10, 60, int(st.session_state["lookback_res"]), 1)
alerts_enabled = st.sidebar.checkbox("Telegram-Alerts aktivieren (Secrets nötig)", value=bool(st.session_state["alerts_enabled"]))
scan_now = st.sidebar.button("🔔 Watchlist jetzt scannen")

# in Session ablegen (für Persistenz)
st.session_state["selected_ids"] = selected_ids
st.session_state["min_mktcap"]   = min_mktcap
st.session_state["min_volume"]   = min_volume
st.session_state["vol_surge_thresh"] = vol_surge_thresh
st.session_state["lookback_res"] = lookback_res
st.session_state["alerts_enabled"] = alerts_enabled

st.caption("🔒 Passwortschutz aktiv — setze `APP_PASSWORD` in Streamlit Secrets.  •  Alerts via Telegram (optional).")

# ----------------- Checklist ------------------
with st.expander("📋 Tägliche Checkliste", expanded=False):
    st.markdown("""
**Morgens:** Scan → Kandidaten (Breakout + Vol-Surge) notieren, Funding/TVL querprüfen  
**Mittags:** Entry nur bei Preis > MA20 > MA50 **und** Vol-Surge ≥ Schwelle **und** Breakout über Widerstand  
**Abends:** Volumen-Trend prüfen, Trailing Stop nachziehen, Teilgewinne sichern
""")

# ----------------- Snapshot -------------------
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

# ----------------- Signals table --------------
rows, history_cache = [], {}

for cid in selected_ids:
    time.sleep(0.25)  # Drossel reduziert Rate-Limit-Treffer
    hist = cg_market_chart(cid, days=days_hist)

    # Status aus Attribut lesen
    status_val = hist.attrs.get("status", "ok") if hist is not None else "no_df"

    if hist is None or hist.empty or (status_val != "ok"):
        rows.append({
            "id": cid, "price": np.nan, "MA20": np.nan, "MA50": np.nan,
            "Breakout_MA": False, "Vol_Surge_x": np.nan,
            "Resistance": np.nan, "Support": np.nan,
            "Breakout_Resistance": False, "Distribution_Risk": False,
            "Entry_Signal": False, "status": status_val or "no data"
        })
        continue

    # Daten vorbereiten
    history_cache[cid] = hist
    dfd = hist.copy()
    dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
    dfd = dfd.set_index("timestamp").sort_index().resample("1D").last().dropna()

    # Signale berechnen
    t_sig = trend_signals(dfd)
    v_sig = volume_signals(dfd)
    resistance, support = calc_local_levels(dfd, lookback=lookback_res)
    last  = dfd.iloc[-1]
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
        "Entry_Signal": entry_ok and breakout_res,
        "status": "ok"
    })

signals_df = pd.DataFrame(rows)
st.subheader("🔎 Signals & Levels")

def _row_style(row):
    # Entry grün, Distribution rot, Breakout gelb
    if bool(row.get("Entry_Signal", False)):
        return ['background-color: #e6ffed'] * len(row)
    if bool(row.get("Distribution_Risk", False)):
        return ['background-color: #ffecec'] * len(row)
    if bool(row.get("Breakout_MA", False)) or bool(row.get("Breakout_Resistance", False)):
        return ['background-color: #fff9e6'] * len(row)
    return [''] * len(row)

if not signals_df.empty:
    display_df = signals_df.copy()
    for c in ["price","MA20","MA50","Vol_Surge_x","Resistance","Support"]:
        if c in display_df.columns:
            display_df[c] = pd.to_numeric(display_df[c], errors="coerce")
    styled = display_df.style.apply(_row_style, axis=1).format({
        "price": "{:.4f}", "MA20": "{:.4f}", "MA50": "{:.4f}",
        "Vol_Surge_x": "{:.2f}", "Resistance": "{:.4f}", "Support": "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Optional Alerts
    if scan_now and alerts_enabled:
        sent = []
        for _, r in signals_df.iterrows():
            if r["Entry_Signal"]:
                ok = send_telegram(
                    f"🚨 Entry-Signal: {r['id']} | Preis: ${r['price']:.3f} | "
                    f"Breakout über Widerstand {r['Resistance']:.3f} | Vol-Surge: {r['Vol_Surge_x']:.2f}x"
                )
                sent.append((r["id"], ok))
        if not sent:
            st.info("Keine Entry-Signale.")
        elif any(ok for _, ok in sent):
            st.success("Telegram-Alerts gesendet.")
        else:
            st.warning("Alert-Versand fehlgeschlagen (prüfe TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")

# ----------------- Detail & Tools -------------
st.markdown("---")
st.subheader("📈 Detail & Risk-Tools")

coin_select = st.selectbox(
    "Coin",
    options=[r["id"] for _, r in signals_df.iterrows()] if not signals_df.empty else selected_ids
)

if coin_select:
    d = history_cache.get(coin_select)
    if d is None or d.empty:
        d = cg_market_chart(coin_select, days=st.session_state["days_hist"])
    if d is None or d.empty or (d.attrs.get("status","ok") != "ok"):
        st.warning("Keine Historie verfügbar (API-Limit oder leere Daten). Probiere weniger Coins oder kürzere Historie.")
    else:
        dfd = d.copy()
        dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True, errors="coerce")
        dfd = dfd.set_index("timestamp").sort_index().resample("1D").last().dropna()

        r, s = calc_local_levels(dfd, lookback=lookback_res)
        v_sig = volume_signals(dfd)

        # Price + MAs + Levels
        fig, ax = plt.subplots()
        ax.plot(dfd.index, dfd["price"], label="Price")
        ax.plot(dfd.index, ma(dfd["price"],20), label="MA20")
        ax.plot(dfd.index, ma(dfd["price"],50), label="MA50")
        if not np.isnan(r): ax.axhline(r, linestyle="--", label=f"Resistance {r:.3f}")
        if not np.isnan(s): ax.axhline(s, linestyle="--", label=f"Support {s:.3f}")
        ax.set_title(f"{coin_select} — Price & Levels"); ax.set_xlabel("Date"); ax.set_ylabel("USD"); ax.legend()
        st.pyplot(fig, use_container_width=True)

        # Volume chart
        fig2, ax2 = plt.subplots()
        ax2.bar(dfd.index, dfd["volume"])
        ax2.set_title(f"{coin_select} — Daily Volume"); ax2.set_xlabel("Date"); ax2.set_ylabel("USD")
        st.pyplot(fig2, use_container_width=True)

        # Distribution hint
        if v_sig["distribution_risk"]:
            st.warning("Distribution-Risk: Preis ↑ bei Volumen < 0.8× 7d-Ø.")
        else:
            st.success("Volumen ok (keine Distribution-Anzeichen).")

        # Position sizing & trailing stop
        st.markdown("### 🧮 Position & Trailing Stop")
        c1, c2, c3, c4 = st.columns(4)
        portfolio   = c1.number_input("Portfolio (USD)", min_value=0.0, value=8000.0, step=100.0)
        risk_pct    = c2.slider("Risiko/Trade (%)", 0.5, 3.0, 2.0, 0.1)
        stop_pct    = c3.slider("Stop-Entfernung (%)", 3.0, 25.0, 8.0, 0.5)
        entry_price = c4.number_input("Entry-Preis", min_value=0.0, value=float(dfd['price'].iloc[-1]), step=0.001, format="%.6f")
        max_loss = portfolio * (risk_pct/100.0)
        size     = max_loss / (stop_pct/100.0) if stop_pct>0 else 0.0
        st.write(f"**Max. Verlust:** ${max_loss:,.2f} • **Positionsgröße (≈):** ${size:,.2f}")

        st.markdown("#### Trailing Stop")
        t1, t2 = st.columns(2)
        trail_pct = t1.slider("Trail (%)", 5.0, 25.0, 10.0, 0.5)
        high_since_entry = t2.number_input("Höchster Kurs seit Entry", min_value=0.0, value=float(dfd['price'].iloc[-1]), step=0.001, format="%.6f")
        tstop = trailing_stop(high_since_entry, trail_pct)
        st.write(f"Trailing Stop bei **${tstop:,.3f}** (High {high_since_entry:,.3f}, Trail {trail_pct:.1f}%)")
