from pathlib import Path
import sys

# app.py is in the project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))         # so "from env.portfolio_env import PortfolioEnv" works

DATA  = ROOT / "data" / "portfolio_features_india.csv"
MODEL = ROOT / "best_model" / "best_model.zip"   # <- SB3 EvalCallback saves as best_model.zip by default

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from stable_baselines3 import PPO
from env.portfolio_env import PortfolioEnv

# Optional (benchmark downloads)
try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

# ---------- helpers ----------
def perf(pv, freq: int = 252):
    # accept Series or 1-col DataFrame
    if isinstance(pv, pd.DataFrame):
        pv = pv.iloc[:, 0] if pv.shape[1] == 1 else pv.mean(axis=1)

    pv = pv.dropna()
    r = pv.pct_change().dropna()
    if r.empty:
        return np.nan, np.nan, np.nan, np.nan

    std = float(r.std())
    mean = float(r.mean())
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (freq / max(len(r), 1)) - 1
    vol = std * np.sqrt(freq)
    sharpe = (mean / std * np.sqrt(freq)) if std > 0 else np.nan
    mdd = ((pv / pv.cummax()) - 1).min()
    return cagr, vol, sharpe, mdd

def pretty_names(cols):
    # Turn "RELIANCE_NS_Close" -> "RELIANCE"
    return [c.replace("_NS", "").replace("_Close", "").replace(".NS", "") for c in cols]

# ---------- app ----------
st.set_page_config("Risk-Aware RL Portfolio — India", layout="wide")

DATA = ROOT / "data" / "portfolio_features_india.csv"
MODEL = ROOT / "best_model" / "ppo_india.zip"

if not DATA.exists():
    st.error(f"Data not found at: {DATA}")
    st.stop()
if not MODEL.exists():
    st.error(f"Model not found at: {MODEL}\nTrain first: `python src/train.py`")
    st.stop()

df = pd.read_csv(DATA, index_col=0, parse_dates=True)

# ---- sidebar controls ----
window = st.sidebar.slider("Observation window (days)", 20, 120, 60, 5)
risk   = st.sidebar.slider("Risk aversion", 0.0, 0.5, 0.15, 0.01)
costbp = st.sidebar.slider("Transaction cost (bps per unit turnover)", 0.0, 10.0, 1.0, 0.1)
init_invest = st.sidebar.number_input("Initial investment (₹)", 1_00_000, 1_00_00_000, 10_00_000, step=50_000)

bench = st.sidebar.selectbox("Benchmark", ["NIFTYBEES.NS", "JUNIORBEES.NS", "Equal-Weight"])
date_range = st.sidebar.date_input(
    "Backtest range",
    [df.index[0].date(), df.index[-1].date()],
    min_value=df.index[0].date(),
    max_value=df.index[-1].date(),
)

# ---- build env and run policy deterministically ----
df_slice = df.loc[str(date_range[0]):str(date_range[1])].copy()
env = PortfolioEnv(
    df_slice,
    window_size=window,
    transaction_cost=costbp / 10000.0,
    risk_aversion=risk,
)

model = PPO.load(str(MODEL), env=None)

obs, info = env.reset()
dates, pv = [], []
for _ in range(len(env.df) - env.window_size - 1):
    dnext = env.df.index[env.current_step]
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    dates.append(dnext)
    pv.append(info["portfolio_value"])
    if done or trunc:
        break

pv_rl = pd.Series(pv, index=pd.Index(dates, name="Date"), name="RL")
pv_rl = pv_rl / pv_rl.iloc[0] * init_invest

# ---- benchmark curve ----
if bench == "Equal-Weight":
    prices = env.df[env.close_cols].astype(float)
    rets = prices.pct_change().fillna(0.0)
    w = np.ones(len(env.close_cols)) / len(env.close_cols)
    pv_bm = (1.0 + rets.dot(w)).cumprod().reindex(pv_rl.index).fillna(method="pad")
else:
    if not HAVE_YF:
        st.warning("`yfinance` not installed; falling back to Equal-Weight benchmark.")
        prices = env.df[env.close_cols].astype(float)
        rets = prices.pct_change().fillna(0.0)
        w = np.ones(len(env.close_cols)) / len(env.close_cols)
        pv_bm = (1.0 + rets.dot(w)).cumprod().reindex(pv_rl.index).fillna(method="pad")
    else:
        px_bm = yf.download(
            bench,
            start=env.df.index[0],
            end=env.df.index[-1],
            auto_adjust=True,
            progress=False,
        )["Close"]
        pv_bm = (px_bm.pct_change().fillna(0).add(1).cumprod()).reindex(pv_rl.index).fillna(method="pad")

pv_bm = pv_bm / pv_bm.iloc[0] * init_invest
pv_bm.name = bench

# ---- KPI tiles ----
k1, k2, k3, k4 = perf(pv_rl)
b1, b2, b3, b4 = perf(pv_bm)
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR (RL)", f"{k1:.2%}", delta=f"{(k1 - b1):+.2%}")
c2.metric("Volatility (RL)", f"{k2:.2%}")
c3.metric("Sharpe (RL)", f"{k3:.2f}", delta=f"{(k3 - b3):+.2f}")
c4.metric("Max Drawdown (RL)", f"{k4:.2%}", delta=f"{(k4 - b4):+.2%}")

# ---- charts ----
equity = pd.concat([pv_rl, pv_bm], axis=1)

tab1, tab2, tab3 = st.tabs(["Equity Curve", "Drawdown", "Rolling Sharpe"])
with tab1:
    st.plotly_chart(px.line(equity, labels={"value": "Value (₹)", "index": "Date"}), use_container_width=True)

with tab2:
    dd = (equity / equity.cummax() - 1)
    st.plotly_chart(px.area(dd, labels={"value": "Drawdown", "index": "Date"}), use_container_width=True)

with tab3:
    roll = (
        equity.pct_change()
        .rolling(252)
        .apply(lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else np.nan)
    )
    st.plotly_chart(px.line(roll, labels={"value": "Rolling Sharpe", "index": "Date"}), use_container_width=True)

# ---- latest weights table ----
w_last = pd.Series(env.portfolio_weights, index=pretty_names(env.close_cols), name="weight")
st.subheader("Latest Portfolio Weights")
st.dataframe((w_last * 100).round(2).sort_values(ascending=False).to_frame("%"))

# ---- downloads ----
st.download_button("Download Equity Curve (CSV)", equity.to_csv().encode(), "equity_curve.csv", "text/csv")