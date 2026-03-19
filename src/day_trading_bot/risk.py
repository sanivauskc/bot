from __future__ import annotations

from dataclasses import dataclass

from day_trading_bot.config import StrategyConfig
from day_trading_bot.strategy import Side, TradeSignal


@dataclass(slots=True)
class RiskState:
    daily_realized_pnl: float = 0.0
    trades_today: int = 0


@dataclass(slots=True)
class PositionPlan:
    side: Side
    quantity: int
    stop_distance: float
    take_profit_distance: float
    allowed: bool
    reason: str


def build_position_plan(
    signal: TradeSignal,
    current_price: float,
    atr: float,
    account_equity: float,
    config: StrategyConfig,
    risk_state: RiskState,
) -> PositionPlan:
    if signal.side == Side.FLAT:
        return PositionPlan(
            side=Side.FLAT,
            quantity=0,
            stop_distance=0.0,
            take_profit_distance=0.0,
            allowed=False,
            reason=signal.reason,
        )

    if risk_state.daily_realized_pnl <= -abs(config.max_daily_loss):
        return PositionPlan(
            side=Side.FLAT,
            quantity=0,
            stop_distance=0.0,
            take_profit_distance=0.0,
            allowed=False,
            reason="Daily loss limit reached",
        )

    if risk_state.trades_today >= config.max_trades_per_day:
        return PositionPlan(
            side=Side.FLAT,
            quantity=0,
            stop_distance=0.0,
            take_profit_distance=0.0,
            allowed=False,
            reason="Max trades per day reached",
        )

    atr = max(float(atr), 0.01)
    stop_distance = atr * config.stop_atr_multiple
    take_profit_distance = atr * config.take_profit_atr_multiple

    risk_dollars = max(account_equity * config.risk_per_trade, 50.0)
    risk_per_share = max(stop_distance, 0.01)
    qty_risk = int(risk_dollars // risk_per_share)
    qty_cap = int(config.max_position_dollars // max(current_price, 0.01))
    quantity = max(0, min(qty_risk, qty_cap))

    if quantity <= 0:
        return PositionPlan(
            side=Side.FLAT,
            quantity=0,
            stop_distance=0.0,
            take_profit_distance=0.0,
            allowed=False,
            reason="Position size resolved to zero",
        )

    return PositionPlan(
        side=signal.side,
        quantity=quantity,
        stop_distance=stop_distance,
        take_profit_distance=take_profit_distance,
        allowed=True,
        reason="Risk checks passed",
    )
