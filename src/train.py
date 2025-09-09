# src/train.py
from pathlib import Path
import sys

# Make project root importable no matter the CWD
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env.portfolio_env import PortfolioEnv  # now resolvable

DATA_PATH = ROOT / "data" / "portfolio_features_india.csv"
TB_LOG    = ROOT / "tb_india"
BEST_DIR  = ROOT / "best_model"
LOG_DIR   = ROOT / "logs"

BEST_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def make_env(df):
    return Monitor(PortfolioEnv(df, window_size=60, transaction_cost=0.001, risk_aversion=0.15))

def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    env_vec  = DummyVecEnv([lambda: make_env(df.copy())])
    eval_env = DummyVecEnv([lambda: make_env(df.copy())])

    model = PPO(
        "MlpPolicy", env_vec, verbose=1, tensorboard_log=str(TB_LOG),
        n_steps=512, batch_size=256, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.0, vf_coef=0.5
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(LOG_DIR),
        eval_freq=5000,
        deterministic=True
    )

    model.learn(total_timesteps=400_000, callback=eval_cb)
    model.save(str(BEST_DIR / "ppo_india"))

if __name__ == "__main__":
    main()