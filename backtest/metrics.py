import numpy as np
import pandas as pd


def compute_report(trades: pd.DataFrame, initial_capital: float) -> dict:
    if trades is None or trades.empty:
        return {
            "net_pnl": 0.0,
            "return_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "max_dd": 0.0,
            "equity": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
        }

    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t.sort_values("exit_time")

    equity = pd.Series(
        [initial_capital] + t["pnl"].cumsum().add(initial_capital).tolist(),
        index=[t["exit_time"].iloc[0]] + t["exit_time"].tolist(),
        )

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    wins = t[t["pnl"] > 0]
    losses = t[t["pnl"] < 0]

    win_rate = float((t["pnl"] > 0).mean())
    expectancy_r = float(t["r_multiple"].mean())

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return {
        "net_pnl": float(t["pnl"].sum()),
        "return_pct": float((equity.iloc[-1] - initial_capital) / initial_capital),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "expectancy_r": expectancy_r,
        "max_dd": max_dd,
        "equity": equity,
        "drawdown": dd,
        "trades": t.reset_index(drop=True),
    }
