from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from day_trading_bot.config import BotConfig
from day_trading_bot.data.schwab_collector import SchwabDataCollector
from day_trading_bot.execution.ibkr_executor import IBKRExecutor
from day_trading_bot.features import build_feature_frame, build_supervised_dataset, summarize_option_chain
from day_trading_bot.model import IntradayEnsembleModel
from day_trading_bot.risk import RiskState, build_position_plan
from day_trading_bot.strategy import Side, decide_signal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CycleResult:
    timestamp: datetime
    proba_up: float
    side: Side
    quantity: int
    executed: bool
    note: str


class TradingBot:
    def __init__(
        self,
        config: BotConfig,
        collector: SchwabDataCollector | None = None,
        executor: IBKRExecutor | None = None,
        model: IntradayEnsembleModel | None = None,
    ) -> None:
        self.config = config
        self.collector = collector or SchwabDataCollector(config.schwab)
        self.executor = executor or IBKRExecutor(config.ibkr, dry_run=config.dry_run)
        self.model = model or IntradayEnsembleModel()
        self.option_factor_history = pd.DataFrame()
        self.risk_state = RiskState()
        self.cycle_idx = 0
        self.current_day = date.today()
        self._feature_columns: list[str] = []

    def run_cycle(self) -> CycleResult:
        self._roll_daily_state_if_needed()
        strat = self.config.strategy
        spy, vix, options = self.collector.load_market_snapshot(
            lookback_days=max(1, strat.lookback_bars // 78),
            frequency_minutes=strat.bar_minutes,
        )
        latest_ts = pd.Timestamp(spy["timestamp"].iloc[-1])
        option_row = summarize_option_chain(options.frame, latest_ts)
        self.option_factor_history = pd.concat(
            [self.option_factor_history, option_row],
            ignore_index=True,
        ).drop_duplicates(subset=["timestamp"], keep="last")

        features = build_feature_frame(spy, vix, self.option_factor_history)
        dataset = build_supervised_dataset(features, horizon_bars=strat.target_horizon_bars)
        if len(dataset.frame) < strat.min_train_rows:
            note = f"Not enough training rows: {len(dataset.frame)} < {strat.min_train_rows}"
            logger.warning(note)
            return CycleResult(
                timestamp=datetime.utcnow(),
                proba_up=0.5,
                side=Side.FLAT,
                quantity=0,
                executed=False,
                note=note,
            )

        if self.cycle_idx % strat.retrain_every_n_cycles == 0 or not self._feature_columns:
            self._train(dataset.frame, dataset.feature_columns, dataset.target_column)

        latest_row = dataset.frame.iloc[-1]
        outputs = self.model.predict_one(latest_row[self._feature_columns])
        signal = decide_signal(outputs, latest_vix=float(latest_row["vix_close"]), config=strat)

        account_equity = self.executor.get_account_equity()
        plan = build_position_plan(
            signal=signal,
            current_price=float(latest_row["close"]),
            atr=float(latest_row["spy_atr_14"]),
            account_equity=account_equity,
            config=strat,
            risk_state=self.risk_state,
        )

        executed = False
        note = signal.reason
        if plan.allowed and plan.side in {Side.LONG, Side.SHORT} and plan.quantity > 0:
            result = self.executor.place_spy_bracket_order(
                side=plan.side,
                quantity=plan.quantity,
                entry_price=float(latest_row["close"]),
                stop_distance=plan.stop_distance,
                take_profit_distance=plan.take_profit_distance,
            )
            executed = result.submitted
            note = result.details
            if executed:
                self.risk_state.trades_today += 1

        self.cycle_idx += 1
        return CycleResult(
            timestamp=datetime.utcnow(),
            proba_up=outputs.proba_up,
            side=plan.side if plan.allowed else Side.FLAT,
            quantity=plan.quantity if plan.allowed else 0,
            executed=executed,
            note=note,
        )

    def run_forever(self) -> None:
        logger.info("Starting bot loop. dry_run=%s", self.config.dry_run)
        while True:
            try:
                result = self.run_cycle()
                logger.info(
                    "cycle ts=%s p_up=%.3f side=%s qty=%s executed=%s note=%s",
                    result.timestamp.isoformat(),
                    result.proba_up,
                    result.side.name,
                    result.quantity,
                    result.executed,
                    result.note,
                )
            except Exception as exc:  # noqa: BLE001 - keep bot alive across transient failures.
                logger.exception("Cycle failed: %s", exc)
            time.sleep(self.config.polling_seconds)

    def _train(self, frame: pd.DataFrame, feature_columns: list[str], target_column: str) -> None:
        self._feature_columns = feature_columns
        X = frame[feature_columns]
        y = frame[target_column]
        self.model.fit(X, y)
        logger.info("Model trained on %s rows", len(frame))

    def _roll_daily_state_if_needed(self) -> None:
        today = date.today()
        if today != self.current_day:
            self.current_day = today
            self.risk_state = RiskState()
            logger.info("Reset daily risk state")
