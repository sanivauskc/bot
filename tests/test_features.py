from __future__ import annotations

import numpy as np
import pandas as pd

from day_trading_bot.features import build_feature_frame, build_supervised_dataset, summarize_option_chain


def _sample_bars(symbol: str, start_price: float, periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-01-02 14:30:00+00:00", periods=periods, freq="1min")
    trend = np.linspace(0, 1.2, periods)
    close = start_price + trend
    return pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": symbol,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(1000, 2000, periods),
        }
    )


def test_option_summary_outputs_core_fields() -> None:
    ts = pd.Timestamp("2026-01-02 15:30:00+00:00")
    option_frame = pd.DataFrame(
        [
            {"type": "CALL", "strike": 600, "delta": 0.25, "iv": 0.18, "mid": 3.1, "open_interest": 1000, "volume": 500, "gamma": 0.02, "underlying_price": 600},
            {"type": "PUT", "strike": 600, "delta": -0.25, "iv": 0.21, "mid": 2.9, "open_interest": 1200, "volume": 650, "gamma": 0.025, "underlying_price": 600},
        ]
    )
    summary = summarize_option_chain(option_frame, ts)
    assert "opt_put_call_oi_ratio" in summary.columns
    assert "opt_iv_skew_25d" in summary.columns
    assert summary["opt_atm_straddle_pct"].iloc[0] > 0


def test_build_feature_frame_and_supervised_dataset() -> None:
    spy = _sample_bars("SPY", start_price=599.0, periods=160)
    vix = _sample_bars("$VIX", start_price=18.0, periods=160)
    feature_frame = build_feature_frame(spy, vix)
    assert "spy_ret_1" in feature_frame.columns
    assert "vix_close" in feature_frame.columns
    dataset = build_supervised_dataset(feature_frame, horizon_bars=5)
    assert dataset.target_column == "target_up"
    assert len(dataset.frame) > 100
