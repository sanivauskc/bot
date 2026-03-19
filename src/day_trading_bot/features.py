from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureSet:
    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str = "target_up"


def summarize_option_chain(option_chain: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Convert raw option chain rows into a single timestamped factor row."""
    frame = option_chain.copy()
    if frame.empty:
        return pd.DataFrame([{"timestamp": timestamp}])

    calls = frame.loc[frame["type"] == "CALL"].copy()
    puts = frame.loc[frame["type"] == "PUT"].copy()

    call_oi = calls["open_interest"].sum(skipna=True)
    put_oi = puts["open_interest"].sum(skipna=True)
    call_vol = calls["volume"].sum(skipna=True)
    put_vol = puts["volume"].sum(skipna=True)
    put_call_oi_ratio = _safe_div(put_oi, call_oi)
    put_call_vol_ratio = _safe_div(put_vol, call_vol)

    iv_skew = np.nan
    if "delta" in frame and "iv" in frame:
        put_25 = _iv_near_delta(puts, target_delta=-0.25)
        call_25 = _iv_near_delta(calls, target_delta=0.25)
        iv_skew = put_25 - call_25

    atm_straddle_pct = np.nan
    underlying_price = float(frame["underlying_price"].dropna().iloc[0]) if not frame["underlying_price"].dropna().empty else np.nan
    if np.isfinite(underlying_price):
        atm_call_mid = _atm_mid(calls, underlying_price)
        atm_put_mid = _atm_mid(puts, underlying_price)
        if np.isfinite(atm_call_mid) and np.isfinite(atm_put_mid):
            atm_straddle_pct = _safe_div(atm_call_mid + atm_put_mid, underlying_price)

    call_gamma = (calls["gamma"] * calls["open_interest"]).sum(skipna=True)
    put_gamma = (puts["gamma"] * puts["open_interest"]).sum(skipna=True)
    gamma_imbalance = _safe_div(call_gamma - put_gamma, abs(call_gamma) + abs(put_gamma))

    row = {
        "timestamp": timestamp,
        "opt_put_call_oi_ratio": put_call_oi_ratio,
        "opt_put_call_vol_ratio": put_call_vol_ratio,
        "opt_iv_skew_25d": iv_skew,
        "opt_atm_straddle_pct": atm_straddle_pct,
        "opt_gamma_imbalance": gamma_imbalance,
    }
    return pd.DataFrame([row])


def build_feature_frame(
    spy_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
    option_factor_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    spy = spy_bars.sort_values("timestamp").copy()
    vix = vix_bars.sort_values("timestamp").copy()

    merged = pd.merge_asof(
        spy[["timestamp", "open", "high", "low", "close", "volume"]],
        vix[["timestamp", "close"]].rename(columns={"close": "vix_close"}),
        on="timestamp",
        direction="backward",
    )

    merged["spy_ret_1"] = merged["close"].pct_change(1)
    merged["spy_ret_5"] = merged["close"].pct_change(5)
    merged["spy_ret_15"] = merged["close"].pct_change(15)
    merged["spy_mom_20"] = merged["close"] / merged["close"].rolling(20).mean() - 1
    merged["spy_rv_20"] = merged["spy_ret_1"].rolling(20).std() * np.sqrt(390)
    merged["spy_volume_z"] = _rolling_zscore(merged["volume"], window=30)
    merged["spy_atr_14"] = _atr(merged, 14)

    merged["vix_ret_1"] = merged["vix_close"].pct_change(1)
    merged["vix_ret_5"] = merged["vix_close"].pct_change(5)
    merged["vix_ma_ratio"] = merged["vix_close"] / merged["vix_close"].rolling(20).mean()
    merged["vix_spike_flag"] = (merged["vix_ret_1"] > 0.05).astype(float)

    merged["cross_risk_pressure"] = merged["vix_ret_1"] - merged["spy_ret_1"]
    merged["cross_vol_momentum"] = merged["spy_rv_20"] * merged["vix_ma_ratio"]

    if option_factor_history is not None and not option_factor_history.empty:
        factors = option_factor_history.sort_values("timestamp").copy()
        merged = pd.merge_asof(merged, factors, on="timestamp", direction="backward")
    else:
        merged["opt_put_call_oi_ratio"] = np.nan
        merged["opt_put_call_vol_ratio"] = np.nan
        merged["opt_iv_skew_25d"] = np.nan
        merged["opt_atm_straddle_pct"] = np.nan
        merged["opt_gamma_imbalance"] = np.nan

    feature_columns = list(_default_feature_columns())
    merged[feature_columns] = merged[feature_columns].replace([np.inf, -np.inf], np.nan)
    merged[feature_columns] = merged[feature_columns].fillna(method="ffill").fillna(0.0)
    return merged


def build_supervised_dataset(
    feature_frame: pd.DataFrame,
    horizon_bars: int = 5,
) -> FeatureSet:
    frame = feature_frame.copy()
    frame["future_return"] = frame["close"].shift(-horizon_bars) / frame["close"] - 1
    frame["target_up"] = (frame["future_return"] > 0).astype(int)
    frame = frame.dropna(subset=["future_return"]).reset_index(drop=True)
    return FeatureSet(
        frame=frame,
        feature_columns=list(_default_feature_columns()),
        target_column="target_up",
    )


def _default_feature_columns() -> Iterable[str]:
    return (
        "spy_ret_1",
        "spy_ret_5",
        "spy_ret_15",
        "spy_mom_20",
        "spy_rv_20",
        "spy_volume_z",
        "spy_atr_14",
        "vix_close",
        "vix_ret_1",
        "vix_ret_5",
        "vix_ma_ratio",
        "vix_spike_flag",
        "cross_risk_pressure",
        "cross_vol_momentum",
        "opt_put_call_oi_ratio",
        "opt_put_call_vol_ratio",
        "opt_iv_skew_25d",
        "opt_atm_straddle_pct",
        "opt_gamma_imbalance",
    )


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _atr(frame: pd.DataFrame, window: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _safe_div(x: float, y: float) -> float:
    if y == 0 or np.isnan(y):
        return np.nan
    return float(x / y)


def _iv_near_delta(option_df: pd.DataFrame, target_delta: float) -> float:
    if option_df.empty:
        return np.nan
    valid = option_df.dropna(subset=["delta", "iv"]).copy()
    if valid.empty:
        return np.nan
    idx = (valid["delta"] - target_delta).abs().idxmin()
    return float(valid.loc[idx, "iv"])


def _atm_mid(option_df: pd.DataFrame, underlying_price: float) -> float:
    if option_df.empty:
        return np.nan
    valid = option_df.dropna(subset=["strike", "mid"]).copy()
    if valid.empty:
        return np.nan
    idx = (valid["strike"] - underlying_price).abs().idxmin()
    return float(valid.loc[idx, "mid"])
