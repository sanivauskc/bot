from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from day_trading_bot.config import IBKRConfig
from day_trading_bot.strategy import Side


@dataclass(slots=True)
class ExecutionResult:
    submitted: bool
    order_ids: list[int]
    details: str


class IBKRExecutor:
    """Place SPY orders with IBKR (TWS / IB Gateway) using ib_insync."""

    def __init__(self, config: IBKRConfig, dry_run: bool = True, ib: Any | None = None) -> None:
        self.config = config
        self.dry_run = dry_run
        self.ib = ib

    def connect(self) -> None:
        if self.dry_run:
            return
        if self.ib is not None:
            return
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise ImportError("ib_insync is required. Install with `pip install ib-insync`.") from exc
        self.ib = IB()
        self.ib.connect(self.config.host, self.config.port, clientId=self.config.client_id, readonly=False)

    def disconnect(self) -> None:
        if self.ib is not None:
            self.ib.disconnect()
            self.ib = None

    def get_account_equity(self, fallback: float = 100_000.0) -> float:
        if self.dry_run:
            return fallback
        self.connect()
        assert self.ib is not None
        try:
            values = self.ib.accountSummary(self.config.account)
            for item in values:
                if item.tag == "NetLiquidation":
                    return float(item.value)
        except Exception:
            return fallback
        return fallback

    def place_spy_bracket_order(
        self,
        side: Side,
        quantity: int,
        entry_price: float,
        stop_distance: float,
        take_profit_distance: float,
    ) -> ExecutionResult:
        if side not in {Side.LONG, Side.SHORT} or quantity <= 0:
            return ExecutionResult(submitted=False, order_ids=[], details="No executable side/quantity")

        action = "BUY" if side == Side.LONG else "SELL"
        take_action = "SELL" if action == "BUY" else "BUY"
        stop_action = take_action
        tp_price = entry_price + take_profit_distance if side == Side.LONG else entry_price - take_profit_distance
        stop_price = entry_price - stop_distance if side == Side.LONG else entry_price + stop_distance

        if self.dry_run:
            return ExecutionResult(
                submitted=True,
                order_ids=[],
                details=(
                    f"DRY RUN {datetime.utcnow().isoformat()} "
                    f"{action} {quantity} SPY @MKT TP={tp_price:.2f} SL={stop_price:.2f}"
                ),
            )

        self.connect()
        assert self.ib is not None
        try:
            from ib_insync import MarketOrder, Order, Stock
        except ImportError as exc:
            raise ImportError("ib_insync import failed during order placement.") from exc

        contract = Stock("SPY", self.config.exchange, self.config.currency)
        self.ib.qualifyContracts(contract)

        parent = MarketOrder(action=action, totalQuantity=quantity, transmit=False, account=self.config.account)
        trade_parent = self.ib.placeOrder(contract, parent)
        parent_id = trade_parent.order.orderId

        take = Order(
            action=take_action,
            orderType="LMT",
            lmtPrice=round(tp_price, 2),
            totalQuantity=quantity,
            parentId=parent_id,
            transmit=False,
            account=self.config.account,
        )
        stop = Order(
            action=stop_action,
            orderType="STP",
            auxPrice=round(stop_price, 2),
            totalQuantity=quantity,
            parentId=parent_id,
            transmit=True,
            account=self.config.account,
        )
        trade_take = self.ib.placeOrder(contract, take)
        trade_stop = self.ib.placeOrder(contract, stop)
        return ExecutionResult(
            submitted=True,
            order_ids=[parent_id, trade_take.order.orderId, trade_stop.order.orderId],
            details="Bracket order submitted",
        )
