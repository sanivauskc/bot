from __future__ import annotations

from day_trading_bot.config import StrategyConfig
from day_trading_bot.model import ModelOutputs
from day_trading_bot.risk import RiskState, build_position_plan
from day_trading_bot.strategy import Side, decide_signal


def test_decide_signal_long_and_short() -> None:
    cfg = StrategyConfig()
    long_sig = decide_signal(ModelOutputs(proba_up=0.7, proba_down=0.3, edge=0.4), latest_vix=16.0, config=cfg)
    short_sig = decide_signal(ModelOutputs(proba_up=0.3, proba_down=0.7, edge=-0.4), latest_vix=16.0, config=cfg)
    assert long_sig.side == Side.LONG
    assert short_sig.side == Side.SHORT


def test_risk_plan_respects_limits() -> None:
    cfg = StrategyConfig(max_trades_per_day=1)
    state = RiskState(daily_realized_pnl=0.0, trades_today=1)
    signal = decide_signal(ModelOutputs(proba_up=0.75, proba_down=0.25, edge=0.5), latest_vix=14.0, config=cfg)
    plan = build_position_plan(
        signal=signal,
        current_price=600.0,
        atr=2.0,
        account_equity=100_000,
        config=cfg,
        risk_state=state,
    )
    assert not plan.allowed
    assert plan.side == Side.FLAT
