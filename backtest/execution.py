import pandas as pd


def simulate_next_open(
        df: pd.DataFrame,
        position: pd.Series,
        *,
        qty: float = 1.0,
        spread: float = 0.0,
        slippage: float = 0.0,
) -> pd.DataFrame:
    """
    Minimal execution model:
    - position is {-1,0,+1}
    - when position changes at bar i, we fill at OPEN of bar i+1
    - costs: spread + slippage applied as a simple price offset
      (works for testing; refine later per asset class)
    Returns trades dataframe.
    """
    if not df.index.equals(position.index):
        position = position.reindex(df.index).fillna(0).astype(int)

    open_ = df["open"]
    times = df.index

    trades = []
    side = 0
    entry_time = None
    entry_price = None

    def apply_cost(price: float, action_side: int) -> float:
        # action_side: +1 buy, -1 sell
        cost = (spread / 2.0) + slippage
        return price + (cost if action_side == +1 else -cost)

    for i in range(len(times) - 1):
        desired = int(position.iloc[i])
        if desired == side:
            continue

        fill_time = times[i + 1]
        fill_price = float(open_.iloc[i + 1])

        # Close existing
        if side != 0:
            exit_action = -side
            exit_price = apply_cost(fill_price, exit_action)
            pnl = (exit_price - entry_price) * qty * side
            ret = pnl / (abs(entry_price) * qty) if entry_price != 0 else 0.0

            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": fill_time,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "return_pct": ret,
                }
            )
            side = 0
            entry_time = None
            entry_price = None

        # Open new
        if desired != 0:
            entry_action = desired
            entry_price = apply_cost(fill_price, entry_action)
            entry_time = fill_time
            side = desired

    return pd.DataFrame(trades)
