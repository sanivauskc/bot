from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from day_trading_bot.config import StrategyConfig
from day_trading_bot.model import ModelOutputs


class Side(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


@dataclass(slots=True)
class TradeSignal:
    side: Side
    confidence: float
    reason: str


def decide_signal(
    outputs: ModelOutputs,
    latest_vix: float,
    config: StrategyConfig,
) -> TradeSignal:
    if latest_vix >= config.vix_circuit_breaker:
        return TradeSignal(
            side=Side.FLAT,
            confidence=1.0,
            reason=f"VIX circuit breaker active ({latest_vix:.2f})",
        )

    if outputs.proba_up >= config.long_threshold:
        confidence = outputs.proba_up - 0.5
        return TradeSignal(side=Side.LONG, confidence=float(confidence), reason="Long probability threshold passed")

    if outputs.proba_up <= config.short_threshold:
        confidence = 0.5 - outputs.proba_up
        return TradeSignal(side=Side.SHORT, confidence=float(confidence), reason="Short probability threshold passed")

    band_center = 0.5
    if abs(outputs.proba_up - band_center) <= config.neutral_band:
        return TradeSignal(side=Side.FLAT, confidence=1.0, reason="Inside neutral probability band")

    bias_side = Side.LONG if outputs.proba_up > 0.5 else Side.SHORT
    return TradeSignal(side=bias_side, confidence=abs(outputs.edge) / 2.0, reason="Weak directional edge")
