import pandas as pd, numpy as np, yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS",
    "BHARTIARTL.NS","LT.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","HINDUNILVR.NS",
    "ASIANPAINT.NS","BAJFINANCE.NS","SUNPHARMA.NS","MARUTI.NS","ULTRACEMCO.NS",
    "TATASTEEL.NS","WIPRO.NS","TECHM.NS"
]

START = "2016-01-01"
END   = None  # to latest

def _close_1d(df: pd.DataFrame) -> pd.Series:
    """Return a 1-D Close price Series regardless of yfinance's shape."""
    if "Close" not in df.columns:
        raise KeyError("Close column not found")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # Sometimes yfinance returns a 1-column DataFrame; squeeze to Series
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    close.name = "Close"
    return close

def make_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = _close_1d(df)

    out[f"{prefix}_Close"]   = close
    ret = close.pct_change()
    out[f"{prefix}_ret"]     = ret
    out[f"{prefix}_ema_20"]  = EMAIndicator(close=close, window=20).ema_indicator()
    out[f"{prefix}_rsi_14"]  = RSIIndicator(close=close, window=14).rsi()
    out[f"{prefix}_vol_20"]  = ret.rolling(20).std() * np.sqrt(252)
    return out

def main():
    all_feats = []
    for t in TICKERS:
        print("Downloading", t)
        df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
        if df is None or df.empty:
            print(f"  -> skipped {t}: no data")
            continue
        df = df.dropna(how="all")
        try:
            feats = make_features(df, prefix=t.replace(".", "_"))
        except Exception as e:
            print(f"  -> skipped {t}: {e}")
            continue
        all_feats.append(feats)

    if not all_feats:
        raise RuntimeError("No data collected. Check tickers or internet connection.")

    data = pd.concat(all_feats, axis=1).sort_index()
    data = data.ffill().dropna()
    data.to_csv("data/portfolio_features_india.csv")
    print("Saved -> data/portfolio_features_india.csv with shape", data.shape)

if __name__ == "__main__":
    main()