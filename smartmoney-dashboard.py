# smartmoney-dashboard.py
# See previous cell for full comments. This is the protected Streamlit app.

import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Smart Money Dashboard — Geschützt", layout="wide")

def auth_gate() -> None:
    st.title("🧠 Smart Money Dashboard — Geschützt")
    secret_pw = st.secrets.get("APP_PASSWORD", None)
    if not secret_pw:
        st.error("Konfiguration fehlt: Bitte `APP_PASSWORD` in Secrets setzen.")
        st.stop()
    if st.session_state.get("AUTH_OK", False):
        top = st.container()
        col1, col2 = top.columns([6,1])
        col1.success("Zugriff gewährt.")
        if col2.button("Logout"):
            st.session_state["AUTH_OK"] = False
            st.rerun()
        return
    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input("Passwort eingeben", type="password")
        ok = st.form_submit_button("Login")
        if ok:
            if pw == secret_pw:
                st.session_state["AUTH_OK"] = True
                st.rerun()
            else:
                st.error("Falsches Passwort.")
    st.stop()

auth_gate()

FIAT = "usd"
CG_BASE = "https://api.coingecko.com/api/v3"

def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

@st.cache_data(ttl=900, show_spinner=False)
def cg_market_chart(coin_id: str, days: int = 180) -> pd.DataFrame:
    url = f"{CG_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": FIAT, "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    prices = data.get("prices", [])
    vols = data.get("total_volumes", [])
    if not prices:
        return pd.DataFrame()
    dfp = pd.DataFrame(prices, columns=["ts","price"])
    dfv = pd.DataFrame(vols, columns=["ts","volume"])
    df = dfp.merge(dfv, on="ts", how="left")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[["timestamp","price","volume"]]
    return df

@st.cache_data(ttl=300, show_spinner=False)
def cg_simple_price(ids: list[str]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    url = f"{CG_BASE}/coins/markets"
    params = {
        "vs_currency": FIAT, "ids": ",".join(ids),
        "order":"market_cap_desc","per_page": len(ids),"page":1,"sparkline":"false"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.DataFrame(r.json())
    cols = ["id","symbol","name","current_price","market_cap","total_volume","price_change_percentage_24h"]
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df[cols].rename(columns={"current_price":"price","total_volume":"volume_24h"})

def calc_local_levels(dfd: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    if dfd.empty:
        return (np.nan, np.nan)
    d = dfd.copy().reset_index(drop=True)
    hist = d.iloc[:-1].tail(lookback)
    if hist.empty:
        return (np.nan, np.nan)
    resistance = float(hist["price"].max())
    support = float(hist["price"].min())
    return resistance, support

def volume_signals(dfd: pd.DataFrame) -> dict:
    out = {"vol_ratio_1d_vs_7d": np.nan, "distribution_risk": False, "price_chg_7d": np.nan}
    if dfd.empty or len(dfd) < 8:
        return out
    last = dfd.iloc[-1]
    avg7 = dfd["volume"].iloc[-8:-1].mean()
    vr = float(last["volume"]/avg7) if avg7 else np.nan
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

# Sidebar
st.sidebar.header("Settings")
default_watchlist = "bitcoin, ethereum, solana, arbitrum, render-token, bittensor"
watchlist = [c.strip() for c in st.sidebar.text_area("Watchlist (CoinGecko IDs)", value=default_watchlist).split(",") if c.strip()]

min_mktcap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=300_000_000, step=50_000_000)
min_volume = st.sidebar.number_input("Min 24h Volume (USD)", min_value=0, value=50_000_000, step=10_000_000)
vol_surge_thresh = st.sidebar.slider("Vol Surge vs 7d (x)", 1.0, 5.0, 1.5, 0.1)
lookback_res = st.sidebar.slider("Lookback für Widerstand/Support (Tage)", 10, 60, 20, 1)
alerts_enabled = st.sidebar.checkbox("Telegram-Alerts aktivieren (Secrets nötig)", value=True)  # default ON
scan_now = st.sidebar.button("🔔 Watchlist jetzt scannen")

st.caption("🔒 Passwortschutz aktiv — setze `APP_PASSWORD` in Streamlit Secrets.  •  Alerts via Telegram (optional Secrets).")

# Snapshot
spot = cg_simple_price(watchlist)
if not spot.empty:
    filt = spot[(spot["market_cap"] >= min_mktcap) & (spot["volume_24h"] >= min_volume)]
    st.subheader("📊 Snapshot (Filter)")
    st.dataframe(filt.rename(columns={"id":"ID","symbol":"Symbol","name":"Name","price":"Price","market_cap":"MktCap","volume_24h":"Vol 24h","price_change_percentage_24h":"% 24h"}),
                 use_container_width=True, hide_index=True)

# Signals
rows, history_cache = [], {}
for cid in watchlist:
    hist = cg_market_chart(cid, days=180)
    if hist.empty:
        continue
    history_cache[cid] = hist
    dfd = hist.set_index("timestamp").resample("1D").last().dropna()
    t_sig = trend_signals(dfd)
    v_sig = volume_signals(dfd)
    resistance, support = calc_local_levels(dfd, lookback=lookback_res)
    last = dfd.iloc[-1]
    price = float(last["price"])
    volsurge = v_sig["vol_ratio_1d_vs_7d"]
    breakout_res = bool(price > resistance) if not math.isnan(resistance) else False
    entry_ok = bool(t_sig["breakout_ma"] and (volsurge == volsurge and volsurge >= vol_surge_thresh))
    rows.append({
        "id": cid, "price": price,
        "MA20": t_sig["ma20"], "MA50": t_sig["ma50"],
        "Breakout_MA": t_sig["breakout_ma"], "Vol_Surge_x": volsurge,
        "Resistance": resistance, "Support": support,
        "Breakout_Resistance": breakout_res,
        "Distribution_Risk": v_sig["distribution_risk"],
        "Entry_Signal": entry_ok and breakout_res
    })

signals_df = pd.DataFrame(rows)
st.subheader("🔎 Signals & Levels")
if not signals_df.empty:
    st.dataframe(signals_df, use_container_width=True, hide_index=True)

    # Optional Alerts
    if scan_now and alerts_enabled:
        sent = []
        for _, r in signals_df.iterrows():
            if r["Entry_Signal"]:
                ok = send_telegram(f"🚨 Entry-Signal: {r['id']} | Preis: ${r['price']:.3f} | Breakout über Widerstand {r['Resistance']:.3f} | Vol-Surge: {r['Vol_Surge_x']:.2f}x")
                sent.append((r["id"], ok))
        if any(ok for _, ok in sent):
            st.success("Telegram-Alerts gesendet.")
        else:
            st.info("Keine Entry-Signale oder Secrets fehlen.")

# Detail
st.markdown("---")
st.subheader("📈 Detail & Risk-Tools")
coin_select = st.selectbox("Coin", options=[r["id"] for _, r in signals_df.iterrows()] if not signals_df.empty else watchlist)
if coin_select:
    d = history_cache.get(coin_select) or cg_market_chart(coin_select, days=180)
    if not d.empty:
        dfd = d.set_index("timestamp").resample("1D").last().dropna()
        r, s = calc_local_levels(dfd)
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

        st.markdown("### 🧮 Position & Trailing Stop")
        c1, c2, c3, c4 = st.columns(4)
        portfolio = c1.number_input("Portfolio (USD)", min_value=0.0, value=8000.0, step=100.0)
        risk_pct = c2.slider("Risiko/Trade (%)", 0.5, 3.0, 2.0, 0.1)
        stop_pct = c3.slider("Stop-Entfernung (%)", 3.0, 25.0, 8.0, 0.5)
        entry_price = c4.number_input("Entry-Preis", min_value=0.0, value=float(dfd['price'].iloc[-1]), step=0.001, format="%.6f")
        max_loss = portfolio * (risk_pct/100.0)
        size = max_loss / (stop_pct/100.0) if stop_pct>0 else 0.0
        st.write(f"**Max. Verlust:** ${max_loss:,.2f} • **Positionsgröße (≈):** ${size:,.2f}")

        st.markdown("#### Trailing Stop")
        t1, t2 = st.columns(2)
        trail_pct = t1.slider("Trail (%)", 5.0, 25.0, 10.0, 0.5)
        high_since_entry = t2.number_input("Höchster Kurs seit Entry", min_value=0.0, value=float(dfd['price'].iloc[-1]), step=0.001, format="%.6f")
        tstop = high_since_entry * (1 - trail_pct/100.0)
        st.write(f"Trailing Stop bei **${tstop:,.3f}** (High {high_since_entry:,.3f}, Trail {trail_pct:.1f}%)")
