from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from day_trading_bot.backtest import walk_forward_backtest
from day_trading_bot.config import BotConfig
from day_trading_bot.features import build_feature_frame
from day_trading_bot.pipeline import TradingBot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPY/VIX/options intelligent day trading bot")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--once", action="store_true", help="Run a single live cycle")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest from CSV bars")
    parser.add_argument("--spy-csv", type=Path, help="SPY bars CSV path for backtest")
    parser.add_argument("--vix-csv", type=Path, help="VIX bars CSV path for backtest")
    parser.add_argument("--option-factors-csv", type=Path, help="Option factors CSV path for backtest")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = BotConfig.from_env(args.env_file)

    if args.backtest:
        if not args.spy_csv or not args.vix_csv:
            raise ValueError("--spy-csv and --vix-csv are required for --backtest")
        spy = _load_bar_csv(args.spy_csv)
        vix = _load_bar_csv(args.vix_csv)
        factors = pd.read_csv(args.option_factors_csv, parse_dates=["timestamp"]) if args.option_factors_csv else None
        feature_frame = build_feature_frame(spy, vix, option_factor_history=factors)
        result = walk_forward_backtest(feature_frame, config.strategy)
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe: {result.annualized_sharpe:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Trades: {result.trades}")
        return

    bot = TradingBot(config=config)
    if args.once:
        result = bot.run_cycle()
        print(result)
    else:
        bot.run_forever()


def _load_bar_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    main()
