import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_and_preprocess_data(csv_path: str):
    """
    Loads EURUSD data from CSV and preprocesses it by adding RELATIVE technical features.

    The returned DataFrame still contains OHLCV for env internals,
    but `feature_cols` lists only the RELATIVE columns to feed the agent.
    """
    df = pd.read_csv(csv_path)

    # Strip any trailing spaces in headers (e.g. 'Volume ')
    df.columns = df.columns.str.strip()

    # Detect time column
    time_col = None
    for col in ["Time (EET)", "Gmt time", "time", "Date"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
        df = df.set_index(time_col)
    df.sort_index(inplace=True)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Technicals ----
    # RSI and ATR (already scale-invariant-ish)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Moving averages
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)

    # Slopes of the MAs
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    # Distance of price from each MA (relative level)
    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    # MA divergence: MA20 vs MA50
    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # ---- Task 2.1: Time Features ----
    hour = df.index.hour.astype(float)
    dow  = df.index.dayofweek.astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

    # Session flags (binary)
    # London: 06:00–16:00 UTC, New York: 12:00–21:00 UTC
    df["session_london"]   = ((hour >= 6)  & (hour < 16)).astype(float)
    df["session_ny"]       = ((hour >= 12) & (hour < 21)).astype(float)
    df["session_overlap"]  = ((hour >= 12) & (hour < 16)).astype(float)
    df["session_asian"]    = ((hour >= 21) | (hour < 6)).astype(float)

    # ---- Task 2.2: Candle Structure Features ----
    bar_range = df["High"] - df["Low"] + 1e-9
    df["body_ratio"] = (df["Close"] - df["Open"]) / bar_range
    df["upper_wick_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / bar_range
    df["lower_wick_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / bar_range
    df["close_position"] = (df["Close"] - df["Low"]) / bar_range

    # ---- Task 2.3: Multi-lookback Returns ----
    for n in [1, 3, 6, 12, 24]:
        df[f"log_ret_{n}"] = np.log(df["Close"] / df["Close"].shift(n))

    # ---- Task 2.4: Volatility and Momentum Indicators ----
    # Bollinger Bands
    bbands = ta.bbands(df["Close"], length=20, std=2)
    # pandas_ta might use different column names depending on version, let's be robust
    bbl_col = [c for c in bbands.columns if c.startswith("BBL")][0]
    bbu_col = [c for c in bbands.columns if c.startswith("BBU")][0]
    bbm_col = [c for c in bbands.columns if c.startswith("BBM")][0]

    df["bb_pct"] = (df["Close"] - bbands[bbl_col]) / (bbands[bbu_col] - bbands[bbl_col] + 1e-9)
    df["bb_width"] = (bbands[bbu_col] - bbands[bbl_col]) / bbands[bbm_col]

    # MACD histogram
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # Stochastic %K and %D
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # ATR ratio
    df["atr_ratio"] = df["atr_14"] / df["atr_14"].rolling(50).mean()

    # ---- Task 2.5: Volume Features ----
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)
    df["vol_roc"] = df["Volume"].pct_change(3)

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (no raw price levels / raw MAs)
    feature_cols = [
        # Original 8
        "rsi_14", "atr_14", "ma_20_slope", "ma_50_slope",
        "close_ma20_diff", "close_ma50_diff", "ma_spread", "ma_spread_slope",
        # Time 8
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "session_london", "session_ny", "session_overlap", "session_asian",
        # Candle 4
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "close_position",
        # Returns 5
        "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12", "log_ret_24",
        # Volatility 6
        "bb_pct", "bb_width", "macd_hist", "stoch_k", "stoch_d", "atr_ratio",
        # Volume 2
        "vol_ratio", "vol_roc"
    ]

    return df, feature_cols


def fit_scaler(df: pd.DataFrame, feature_cols: list):
    """Fit a StandardScaler on the provided feature columns. Returns scaler."""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    return scaler

def apply_scaler(df: pd.DataFrame, feature_cols: list, scaler) -> pd.DataFrame:
    """Apply a pre-fitted scaler to a dataframe. Returns a copy."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df
