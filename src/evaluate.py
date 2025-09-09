# src/evaluate.py
from pathlib import Path
import sys

# Make project root importable no matter the CWD
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.portfolio_env import PortfolioEnv

DATA_PATH  = ROOT / "data" / "portfolio_features_india.csv"
MODEL_PATH = ROOT / "best_model" / "ppo_india.zip"
OUT_DIR    = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def perf(pv, freq=252):
    r = pv.pct_change().dropna()
    cagr = (pv.iloc[-1] / pv.iloc[0]) ** (freq / max(len(r), 1)) - 1
    vol  = r.std() * np.sqrt(freq)
    sharpe = (r.mean() / r.std() * np.sqrt(freq)) if r.std() > 0 else np.nan
    mdd = ((pv / pv.cummax()) - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": mdd}

def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    env = PortfolioEnv(df.copy())
    model = PPO.load(str(MODEL_PATH), env=None)

    obs, info = env.reset()
    dates, pv = [], []
    for _ in range(len(df) - env.window_size - 1):
        date_next = env.df.index[env.current_step]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        dates.append(date_next)
        pv.append(info["portfolio_value"])
        if terminated or truncated:
            break

    pv_rl = pd.Series(pv, index=pd.Index(dates, name="Date"), name="RL")

    # Equal-weight baseline using the same tradable close columns the env detected
    prices = df[env.close_cols].astype(float)
    rets = prices.pct_change().fillna(0.0)
    ew = np.ones(len(env.close_cols)) / len(env.close_cols)
    pv_ew = (1.0 + rets.dot(ew)).loc[pv_rl.index].cumprod()
    pv_ew = pv_ew / pv_ew.iloc[0] * env.initial_balance
    pv_ew.name = "Equal-Weight"

    # Metrics
    metrics_rl = perf(pv_rl)
    metrics_ew = perf(pv_ew)
    print("RL:", metrics_rl)
    print("EW:", metrics_ew)

    # Save metrics
    pd.DataFrame([metrics_rl, metrics_ew], index=["RL", "EW"]).to_csv(OUT_DIR / "metrics.csv")

    # Plot & save
    plt.figure(figsize=(10, 5))
    plt.plot(pv_rl.index, pv_rl.values, label="RL Portfolio")
    plt.plot(pv_ew.index, pv_ew.values, label="Equal-Weight")
    plt.title("Portfolio Value (₹)")
    plt.xlabel("Date"); plt.ylabel("Value")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT_DIR / "equity_curve.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()