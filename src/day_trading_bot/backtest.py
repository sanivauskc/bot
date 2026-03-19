from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from day_trading_bot.config import StrategyConfig
from day_trading_bot.features import build_supervised_dataset
from day_trading_bot.model import IntradayEnsembleModel
from day_trading_bot.strategy import Side, decide_signal


@dataclass(slots=True)
class BacktestResult:
    total_return: float
    annualized_sharpe: float
    max_drawdown: float
    trades: int
    equity_curve: pd.Series


def walk_forward_backtest(
    feature_frame: pd.DataFrame,
    config: StrategyConfig,
    training_window: int = 180,
) -> BacktestResult:
    data = build_supervised_dataset(feature_frame, horizon_bars=config.target_horizon_bars)
    frame = data.frame.reset_index(drop=True)
    features = data.feature_columns
    target = data.target_column

    model = IntradayEnsembleModel(random_state=11)
    returns = []
    trade_count = 0

    for idx in range(training_window, len(frame)):
        train_slice = frame.iloc[idx - training_window : idx]
        model.fit(train_slice[features], train_slice[target])

        row = frame.iloc[idx]
        outputs = model.predict_one(row[features])
        signal = decide_signal(outputs, latest_vix=float(row["vix_close"]), config=config)
        if signal.side in {Side.LONG, Side.SHORT}:
            trade_count += 1

        forward_return = float(row["future_return"])
        signed_return = forward_return * int(signal.side)
        returns.append(signed_return)

    if not returns:
        equity = pd.Series([1.0])
        return BacktestResult(0.0, 0.0, 0.0, 0, equity)

    rets = pd.Series(returns, dtype=float)
    equity = (1 + rets.fillna(0)).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    sharpe = _annualized_sharpe(rets, periods_per_year=252 * 390 / config.target_horizon_bars)
    max_dd = _max_drawdown(equity)
    return BacktestResult(
        total_return=total_return,
        annualized_sharpe=sharpe,
        max_drawdown=max_dd,
        trades=trade_count,
        equity_curve=equity,
    )


def _annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    std = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def _max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return float(dd.min())
