from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from day_trading_bot.config import SchwabConfig


@dataclass(slots=True)
class OptionSnapshot:
    timestamp: datetime
    underlying: str
    underlying_price: float
    frame: pd.DataFrame


class SchwabDataCollector:
    """Collects SPY, VIX, and SPY options chain from Schwab API."""

    def __init__(self, config: SchwabConfig, client: Any | None = None) -> None:
        self.config = config
        self.client = client

    def ensure_client(self) -> Any:
        if self.client is not None:
            return self.client

        try:
            from schwab.auth import client_from_token_file, easy_client
        except ImportError as exc:
            raise ImportError(
                "schwab-py is required. Install with `pip install schwab-py`."
            ) from exc

        token_path = str(self.config.token_path)
        try:
            self.client = client_from_token_file(
                token_path,
                self.config.app_key,
                self.config.app_secret,
            )
        except Exception:
            self.client = easy_client(
                api_key=self.config.app_key,
                app_secret=self.config.app_secret,
                callback_url=self.config.callback_url,
                token_path=token_path,
            )

        return self.client

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    def fetch_intraday_bars(
        self,
        symbol: str,
        lookback_days: int = 5,
        frequency_minutes: int = 1,
    ) -> pd.DataFrame:
        client = self.ensure_client()

        # Prefer generic price-history endpoint and fall back to convenience wrappers.
        response = self._try_methods(
            client,
            [
                ("get_price_history", [], {
                    "symbol": symbol,
                    "period_type": "day",
                    "period": lookback_days,
                    "frequency_type": "minute",
                    "frequency": frequency_minutes,
                    "need_extended_hours_data": False,
                    "need_previous_close": True,
                }),
                ("get_price_history_every_minute", [symbol], {}),
                ("price_history", [], {
                    "symbol": symbol,
                    "periodType": "day",
                    "period": lookback_days,
                    "frequencyType": "minute",
                    "frequency": frequency_minutes,
                    "needExtendedHoursData": False,
                    "needPreviousClose": True,
                }),
            ],
        )

        payload = _as_json(response)
        candles = payload.get("candles", [])
        if not candles:
            raise RuntimeError(f"No candles returned for {symbol}")

        frame = pd.DataFrame(candles)
        frame = frame.rename(
            columns={
                "datetime": "timestamp_ms",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
        frame["symbol"] = symbol
        numeric_cols = ["open", "high", "low", "close", "volume"]
        frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        return frame[
            ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        ].reset_index(drop=True)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    def fetch_quote(self, symbol: str) -> dict[str, Any]:
        client = self.ensure_client()
        response = self._try_methods(
            client,
            [
                ("get_quote", [symbol], {}),
                ("get_quotes", [[symbol]], {}),
                ("quote", [symbol], {}),
            ],
        )
        payload = _as_json(response)
        if symbol in payload:
            return payload[symbol]
        if isinstance(payload, list) and payload:
            return payload[0]
        return payload

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    def fetch_option_chain(self, symbol: str, strike_count: int = 40) -> OptionSnapshot:
        client = self.ensure_client()
        response = self._try_methods(
            client,
            [
                ("get_option_chain", [], {
                    "symbol": symbol,
                    "contract_type": "ALL",
                    "strike_count": strike_count,
                    "include_quotes": True,
                    "strategy": "SINGLE",
                }),
                ("get_option_chains", [], {
                    "symbol": symbol,
                    "contract_type": "ALL",
                    "strike_count": strike_count,
                    "include_quotes": True,
                    "strategy": "SINGLE",
                }),
                ("option_chains", [], {
                    "symbol": symbol,
                    "contractType": "ALL",
                    "strikeCount": strike_count,
                    "includeQuotes": "TRUE",
                    "strategy": "SINGLE",
                }),
            ],
        )

        payload = _as_json(response)
        underlying_price = float(payload.get("underlyingPrice", 0.0) or 0.0)
        now = datetime.utcnow()

        rows: list[dict[str, Any]] = []
        for side, map_key in (("CALL", "callExpDateMap"), ("PUT", "putExpDateMap")):
            exp_map = payload.get(map_key, {})
            for exp_key, strike_map in exp_map.items():
                expiration = exp_key.split(":")[0]
                for strike, contracts in strike_map.items():
                    for contract in contracts:
                        rows.append(
                            {
                                "timestamp": now,
                                "type": side,
                                "symbol": contract.get("symbol"),
                                "expiration": expiration,
                                "strike": float(contract.get("strikePrice", strike)),
                                "bid": float(contract.get("bid", 0.0) or 0.0),
                                "ask": float(contract.get("ask", 0.0) or 0.0),
                                "last": float(contract.get("last", 0.0) or 0.0),
                                "mark": float(contract.get("mark", 0.0) or 0.0),
                                "delta": _to_float(contract.get("delta")),
                                "gamma": _to_float(contract.get("gamma")),
                                "vega": _to_float(contract.get("vega")),
                                "theta": _to_float(contract.get("theta")),
                                "iv": _to_float(contract.get("volatility")),
                                "open_interest": _to_float(contract.get("openInterest")),
                                "volume": _to_float(contract.get("totalVolume")),
                            }
                        )

        if not rows:
            raise RuntimeError(f"No option rows returned for {symbol}")

        frame = pd.DataFrame(rows)
        frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
        frame["underlying_price"] = underlying_price

        return OptionSnapshot(
            timestamp=now,
            underlying=symbol,
            underlying_price=underlying_price,
            frame=frame,
        )

    def load_market_snapshot(
        self,
        lookback_days: int,
        frequency_minutes: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, OptionSnapshot]:
        spy = self.fetch_intraday_bars(
            symbol=self.config.spy_symbol,
            lookback_days=lookback_days,
            frequency_minutes=frequency_minutes,
        )
        vix = self.fetch_intraday_bars(
            symbol=self.config.base_vix_symbol,
            lookback_days=lookback_days,
            frequency_minutes=frequency_minutes,
        )
        options = self.fetch_option_chain(symbol=self.config.spy_symbol)
        return spy, vix, options

    @staticmethod
    def _try_methods(
        client: Any,
        method_specs: list[tuple[str, list[Any], dict[str, Any]]],
    ) -> Any:
        errors: list[str] = []
        for method_name, args, kwargs in method_specs:
            method = getattr(client, method_name, None)
            if method is None:
                errors.append(f"{method_name}: missing")
                continue
            try:
                return method(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - preserve all API wrapper errors.
                errors.append(f"{method_name}: {exc}")
        raise RuntimeError("No compatible Schwab method succeeded. " + " | ".join(errors))


def _as_json(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "json"):
        return response.json()
    raise TypeError(f"Unsupported response object type: {type(response)}")


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
