import pandas as pd

REQUIRED_COLS = {"timestamp", "open", "high", "low", "close"}


def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Loads OHLCV from CSV or Parquet.
    - Accepts timestamp as ISO string OR epoch milliseconds.
    - Returns a DataFrame indexed by UTC timestamp with columns: open, high, low, close (+volume if present).
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    df.columns = [str(c).lower() for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Timestamp can be epoch ms (int) or ISO string
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        cols.append("volume")

    return df[cols]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    rule examples: '5min', '15min', '1H', '1D'
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample(rule).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out
