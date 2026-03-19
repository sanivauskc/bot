from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class SchwabConfig:
    app_key: str
    app_secret: str
    callback_url: str
    token_path: Path = Path("secrets/schwab_token.json")
    account_id: str | None = None
    base_vix_symbol: str = "$VIX"
    spy_symbol: str = "SPY"


@dataclass(slots=True)
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 19
    account: str | None = None
    exchange: str = "SMART"
    currency: str = "USD"


@dataclass(slots=True)
class StrategyConfig:
    bar_minutes: int = 1
    lookback_bars: int = 390
    retrain_every_n_cycles: int = 30
    min_train_rows: int = 120
    target_horizon_bars: int = 5
    long_threshold: float = 0.58
    short_threshold: float = 0.42
    neutral_band: float = 0.03
    max_position_dollars: float = 25_000.0
    risk_per_trade: float = 0.003
    max_daily_loss: float = 1_500.0
    max_trades_per_day: int = 10
    stop_atr_multiple: float = 1.8
    take_profit_atr_multiple: float = 2.4
    vix_circuit_breaker: float = 35.0


@dataclass(slots=True)
class BotConfig:
    schwab: SchwabConfig
    ibkr: IBKRConfig
    strategy: StrategyConfig
    dry_run: bool = True
    timezone: str = "America/New_York"
    polling_seconds: int = 45

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "BotConfig":
        load_dotenv(env_file)
        schwab = SchwabConfig(
            app_key=_require_env("SCHWAB_APP_KEY"),
            app_secret=_require_env("SCHWAB_APP_SECRET"),
            callback_url=_require_env("SCHWAB_CALLBACK_URL"),
            token_path=Path(os.getenv("SCHWAB_TOKEN_PATH", "secrets/schwab_token.json")),
            account_id=os.getenv("SCHWAB_ACCOUNT_ID"),
            base_vix_symbol=os.getenv("VIX_SYMBOL", "$VIX"),
            spy_symbol=os.getenv("SPY_SYMBOL", "SPY"),
        )

        ibkr = IBKRConfig(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "7497")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "19")),
            account=os.getenv("IBKR_ACCOUNT"),
            exchange=os.getenv("IBKR_EXCHANGE", "SMART"),
            currency=os.getenv("IBKR_CURRENCY", "USD"),
        )

        strategy = StrategyConfig(
            bar_minutes=int(os.getenv("BAR_MINUTES", "1")),
            lookback_bars=int(os.getenv("LOOKBACK_BARS", "390")),
            retrain_every_n_cycles=int(os.getenv("RETRAIN_EVERY_N_CYCLES", "30")),
            min_train_rows=int(os.getenv("MIN_TRAIN_ROWS", "120")),
            target_horizon_bars=int(os.getenv("TARGET_HORIZON_BARS", "5")),
            long_threshold=float(os.getenv("LONG_THRESHOLD", "0.58")),
            short_threshold=float(os.getenv("SHORT_THRESHOLD", "0.42")),
            neutral_band=float(os.getenv("NEUTRAL_BAND", "0.03")),
            max_position_dollars=float(os.getenv("MAX_POSITION_DOLLARS", "25000")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.003")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "1500")),
            max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "10")),
            stop_atr_multiple=float(os.getenv("STOP_ATR_MULTIPLE", "1.8")),
            take_profit_atr_multiple=float(os.getenv("TAKE_PROFIT_ATR_MULTIPLE", "2.4")),
            vix_circuit_breaker=float(os.getenv("VIX_CIRCUIT_BREAKER", "35")),
        )

        return cls(
            schwab=schwab,
            ibkr=ibkr,
            strategy=strategy,
            dry_run=_bool_env("DRY_RUN", True),
            timezone=os.getenv("BOT_TIMEZONE", "America/New_York"),
            polling_seconds=int(os.getenv("POLLING_SECONDS", "45")),
        )


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
