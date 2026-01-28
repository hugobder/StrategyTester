import pandas as pd


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n).mean()


def donchian_atr_position(
        df: pd.DataFrame,
        *,
        breakout_n: int = 20,
        atr_n: int = 14,
        atr_mult: float = 2.0,
        trend_filter_sma: int | None = 200,
) -> tuple[pd.Series, pd.Series]:
    """
    Donchian Breakout strategy:
    - Long when close breaks above N-high
    - Short when close breaks below N-low
    - Optional trend filter: long only if close > SMA(trend_filter_sma), short only if close < SMA(...)
    - Stop distance is ATR(atr_n) * atr_mult (dynamic, volatility aware)

    Returns:
      - position series in {-1,0,+1} shifted by 1 (no lookahead)
      - stop_distance series (price units) shifted by 1 (so it's known at decision time)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    upper = high.rolling(breakout_n).max()
    lower = low.rolling(breakout_n).min()

    raw_long = close > upper.shift(1)   # breakout uses prior channel
    raw_short = close < lower.shift(1)

    pos = pd.Series(0, index=df.index, dtype=int)

    if trend_filter_sma:
        trend = close.rolling(trend_filter_sma).mean()
        long_ok = close > trend
        short_ok = close < trend
    else:
        long_ok = pd.Series(True, index=df.index)
        short_ok = pd.Series(True, index=df.index)

    pos[raw_long & long_ok] = 1
    pos[raw_short & short_ok] = -1

    # Hold last signal (stay in position until opposite signal)
    pos = pos.replace(0, pd.NA).ffill().fillna(0).astype(int)

    # Dynamic stop distance (ATR * multiplier)
    sd = atr(df, atr_n) * atr_mult

    # No lookahead: decision at bar i, fill at bar i+1 open
    return pos.shift(1).fillna(0).astype(int), sd.shift(1).fillna(method="ffill")
