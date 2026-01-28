from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Trade:
    entry_time: object
    exit_time: object
    side: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    return_pct: float
    reason: Optional[str] = None
