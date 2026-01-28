import pandas as pd


def simulate_next_open_risk_based(
        df: pd.DataFrame,
        position: pd.Series,
        *,
        initial_capital: float,
        risk_per_trade: float,
        stop_distance,  # float OR pd.Series aligned to df.index
        spread: float = 0.0,
        slippage: float = 0.0,
) -> pd.DataFrame:
    """
    Risk-based execution model:
    - position in {-1,0,+1}
    - entry/exit at NEXT bar open
    - stop_distance can be:
        - float (fixed)
        - pd.Series (dynamic per bar, e.g. ATR-based)
    - position size derived from risk % and stop distance
    - stops trading if equity <= 0
    """
    if not df.index.equals(position.index):
        position = position.reindex(df.index).fillna(0).astype(int)

    if isinstance(stop_distance, pd.Series):
        if not df.index.equals(stop_distance.index):
            stop_distance = stop_distance.reindex(df.index).ffill()
    else:
        stop_distance = pd.Series(float(stop_distance), index=df.index)

    open_ = df["open"]
    times = df.index

    equity = initial_capital
    trades = []

    side = 0
    entry_time = None
    entry_price = None
    qty = 0.0
    entry_sd = None

    def apply_cost(price: float, action_side: int) -> float:
        cost = (spread / 2.0) + slippage
        return price + (cost if action_side == +1 else -cost)

    for i in range(len(times) - 1):
        if equity <= 0:
            break

        desired = int(position.iloc[i])
        if desired == side:
            continue

        fill_time = times[i + 1]
        fill_price = float(open_.iloc[i + 1])

        # close existing
        if side != 0:
            exit_price = apply_cost(fill_price, -side)
            pnl = (exit_price - entry_price) * qty * side
            r_mult = pnl / (entry_sd * qty) if entry_sd and entry_sd > 0 else 0.0

            equity += pnl

            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": fill_time,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": qty,
                    "stop_distance": entry_sd,
                    "pnl": pnl,
                    "r_multiple": r_mult,
                    "equity_after": equity,
                }
            )

            side = 0
            entry_time = None
            entry_price = None
            qty = 0.0
            entry_sd = None

        # open new
        if desired != 0:
            sd = float(stop_distance.iloc[i])
            if sd <= 0 or pd.isna(sd):
                continue

            risk_amount = equity * risk_per_trade
            qty = risk_amount / sd

            entry_price = apply_cost(fill_price, desired)
            entry_time = fill_time
            entry_sd = sd
            side = desired

    return pd.DataFrame(trades)
