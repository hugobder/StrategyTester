import pandas as pd


def sma_cross_position(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """
    Returns a position series in {-1, 0, +1}.
    No lookahead: we shift by 1 bar so fills occur on next bar open.
    """
    if fast < 1 or slow < 2:
        raise ValueError("fast/slow must be positive integers")
    if fast >= slow:
        raise ValueError("fast must be < slow")

    close = df["close"]
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()

    raw = (fast_sma > slow_sma).astype(int) - (fast_sma < slow_sma).astype(int)
    pos = raw.shift(1).fillna(0).astype(int)
    return pos
