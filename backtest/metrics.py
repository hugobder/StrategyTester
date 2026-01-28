import numpy as np
import pandas as pd


def compute_report(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "max_dd": 0.0,
            "equity": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
        }

    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t.sort_values("exit_time")

    equity = t.set_index("exit_time")["pnl"].cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if len(dd) else 0.0

    wins = t[t["pnl"] > 0]["pnl"]
    losses = t[t["pnl"] < 0]["pnl"]

    win_rate = float((t["pnl"] > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative
    loss_rate = 1.0 - win_rate
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        "net_pnl": float(t["pnl"].sum()),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "max_dd": float(max_dd),
        "equity": equity,
        "drawdown": dd.fillna(0.0),
        "trades": t.reset_index(drop=True),
    }
