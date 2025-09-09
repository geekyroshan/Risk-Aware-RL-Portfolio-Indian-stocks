# env/portfolio_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Dict, List, Any

class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 60,
        initial_balance: float = 1_000_000,
        transaction_cost: float = 0.001,   # ~10 bps per unit turnover (tune for India)
        risk_aversion: float = 0.15,       # penalize volatility
    ):
        super().__init__()
        self.df = df.copy()
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.risk_aversion = risk_aversion

        self.close_cols = self._infer_close_cols(self.df.columns)
        if len(self.close_cols) == 0:
            raise ValueError("Could not find Close columns in dataset.")
        self.n_assets = len(self.close_cols)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.df.shape[1]),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.reset(seed=42)

    @staticmethod
    def _infer_close_cols(cols) -> List[Any]:
        if isinstance(cols, pd.MultiIndex):
            lv1 = cols.get_level_values(-1)
            if any(str(x).lower() == "close" for x in lv1):
                return [c for c in cols if str(c[-1]).lower() == "close"]
            return [c for c in cols if str(c[0]).lower().endswith("_close") or str(c[0]).lower() == "close"]
        return [c for c in cols if str(c).lower().endswith("_close") or str(c).lower() == "close"]

    def _get_observation(self) -> np.ndarray:
        win = self.df.iloc[self.current_step - self.window_size : self.current_step]
        return win.values.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = float(self.initial_balance)
        self.portfolio_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, 0, 1).astype(np.float32)
        s = float(action.sum())
        action = (action / s) if s > 0 else self.portfolio_weights

        prices_now = self.df.loc[self.df.index[self.current_step - 1], self.close_cols].to_numpy(dtype=np.float32)
        prices_next = self.df.loc[self.df.index[self.current_step], self.close_cols].to_numpy(dtype=np.float32)
        returns = (prices_next - prices_now) / np.where(prices_now == 0, 1e-8, prices_now)

        port_ret = float(np.dot(returns, self.portfolio_weights))
        vol_pen = float(self.risk_aversion * np.std(returns))
        txn_cost = float(self.transaction_cost * np.sum(np.abs(action - self.portfolio_weights)))
        reward = (port_ret - txn_cost) - vol_pen

        self.portfolio_value *= (1.0 + port_ret)
        self.portfolio_weights = action
        self.current_step += 1

        terminated = self.current_step >= (len(self.df) - 1)
        truncated = False
        obs = self._get_observation()
        info = {"portfolio_value": float(self.portfolio_value)}
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(f"Step: {self.current_step} | Portfolio Value: ₹{self.portfolio_value:,.0f}")