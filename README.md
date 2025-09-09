# Risk-Aware RL Portfolio — Indian Stocks

An end-to-end project that learns **daily portfolio weights** for a basket of NSE stocks using **Proximal Policy Optimization (PPO)**.  
The agent balances return vs. risk and accounts for transaction costs, then serves an **interactive Streamlit app** for backtesting and exploration.

> ⚠️ Educational demo only. This is not investment advice.

---

## TL;DR (for reviewers)

- **Problem:** Allocate capital across Indian equities while controlling volatility and trading costs.
- **Approach:** RL policy (PPO) selects continuous portfolio weights; reward = daily return − transaction_cost − risk_penalty.
- **Why it matters:** Demonstrates using RL as an *allocation policy* (not price prediction), with transparent evaluation and a usable UI.

**Live demo:** https://risk-aware-rl-portfolio-indian-stocks-geekyroshan.streamlit.app/  
**Key notebook/app entry point:** `app.py`

---

## Results snapshot

- Backtest window: 2016–2025 (configurable)
- Universe: liquid NSE tickers (e.g., RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS, etc.)
- Typical outcomes on my run (will vary with universe/params):
  - CAGR (RL) > Equal-Weight baseline
  - Lower or comparable max drawdown (depending on risk_aversion & window)
  - Transparent metrics: **CAGR, Volatility, Sharpe, Max Drawdown**



![Equity curve](newplot.png.png)

---

## What the app does

- Lets you select:
  - **Observation window** (state length in days)
  - **Risk aversion** (volatility penalty)
  - **Transaction cost** (basis-points per unit turnover)
  - **Benchmark** (NIFTYBEES / JUNIORBEES / Equal-Weight)
  - **Backtest range** (date span to replay the learned policy)
- Renders:
  - **Equity curve** (RL vs. benchmark)
  - **Drawdown chart**
  - **Rolling Sharpe**
  - **Latest portfolio weights**
  - **CSV download** for equity curve

---

## Method (short)

- **State:** Last *W* days of engineered features (prices, returns, EMA, RSI, volatility) for each stock.
- **Action:** Continuous vector of non-negative weights that sum to 1 (simplex).
- **Reward:**  
  `reward_t = portfolio_return_t − (risk_aversion × volatility_t) − transaction_cost_t`
- **Algorithm:** PPO (stable-baselines3)
- **Environment:** Custom `gymnasium` env in `env/portfolio_env.py`.

---

## Repo structure
portfolio_indian_stock/
├─ app.py                     # Streamlit app
├─ env/
│  └─ portfolio_env.py        # Custom Gymnasium environment
├─ src/
│  ├─ build_dataset_india.py  # Downloads data & builds features
│  ├─ train.py                # Trains PPO; saves best_model/best_model.zip
│  └─ evaluate.py             # Backtest plots + metrics
├─ data/
│  └─ portfolio_features_india.csv
├─ best_model/
│  └─ best_model.zip          # Trained policy (SB3)
├─ requirements.txt
├─ runtime.txt
└─ README.md


---

## Quick start (local)

```bash
# 1) Create env (Python 3.10 recommended)
conda create -n rl_indian_stocks python=3.10 -y
conda activate rl_indian_stocks

# 2) Install
pip install -r requirements.txt

# 3) Build dataset (edit tickers inside if needed)
python src/build_dataset_india.py

# 4) Train PPO (saves best_model/best_model.zip)
python src/train.py

# 5) Launch app
streamlit run app.py

# 6) Reproducing the charts & metrics
python src/evaluate.py