import pandas as pd
import pandas_ta as ta


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

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (no raw price levels / raw MAs)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20_slope",
        "ma_50_slope",
        "close_ma20_diff",
        "close_ma50_diff",
        "ma_spread",
        "ma_spread_slope",
    ]

    return df, feature_cols


from sklearn.preprocessing import StandardScaler
import numpy as np

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
