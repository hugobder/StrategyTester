import numpy as np
import pandas as pd


def _log_returns(close: pd.Series) -> np.ndarray:
    r = np.diff(np.log(close.astype(float).to_numpy()))
    return np.concatenate([[0.0], r])  # align length


def volatility_activity_surface(
    df: pd.DataFrame,
    *,
    max_lag: int = 60,
    window: int = 200,
    stride: int = 5,
    mode: str = "abs",  # "abs" or "sq"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds a quant-style 3D surface:
      X = lag
      Y = tick (time index downsampled)
      Z = activity at lag (volatility clustering measure)

    Activity measure:
      For each tick t and lag L:
        activity[t, L] = mean( v[t-window+1:t] * v[t-L-window+1:t-L] )
      where v = |returns| or returns^2

    Returns: (lags, ticks, Z) with shapes:
      lags:  (max_lag,)
      ticks: (n_ticks,)
      Z:     (max_lag, n_ticks)
    """
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    if window < 10:
        raise ValueError("window too small")
    if len(df) < window + max_lag + 5:
        raise ValueError("Not enough data for chosen window/max_lag")

    close = df["close"]
    r = _log_returns(close)

    if mode == "abs":
        v = np.abs(r)
    elif mode == "sq":
        v = r * r
    else:
        raise ValueError("mode must be 'abs' or 'sq'")

    # ticks we evaluate (downsample for speed)
    t_start = window + max_lag
    ticks = np.arange(t_start, len(v), max(1, int(stride)))

    lags = np.arange(1, max_lag + 1)
    Z = np.full((len(lags), len(ticks)), np.nan, dtype=float)

    for j, t in enumerate(ticks):
        x0 = t - window + 1
        x1 = t + 1
        base = v[x0:x1]

        # Normalize base to avoid pure scale dominating visuals
        b_std = np.std(base)
        if b_std > 0:
            base = (base - np.mean(base)) / b_std

        for i, L in enumerate(lags):
            y0 = x0 - L
            y1 = x1 - L
            lagged = v[y0:y1]

            l_std = np.std(lagged)
            if l_std > 0:
                lagged = (lagged - np.mean(lagged)) / l_std

            # "activity": average product (similar to correlation-ish energy)
            Z[i, j] = float(np.mean(base * lagged))

    return lags, ticks, Z
