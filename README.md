# Intelligent Day Trading Bot (SPY + VIX + SPY Options)

Research-driven intraday bot that:

- collects **SPY, VIX, and SPY options chain data from Schwab API**
- builds engineered cross-asset features from price + volatility + options flow
- trains an adaptive ML ensemble for short-horizon direction prediction
- sends live bracket orders through **IBKR (TWS / Gateway)** with strict risk controls

> Important: no trading system can guarantee profit. This project is designed to be **profit-seeking**, testable, and risk-aware, but real-world performance depends on execution quality, market regime changes, slippage, and transaction costs.

## What is implemented

- **Schwab collector** (`src/day_trading_bot/data/schwab_collector.py`)
  - Intraday bars for SPY and VIX
  - SPY option chain snapshot
  - Normalized pandas outputs
- **Feature engine** (`src/day_trading_bot/features.py`)
  - SPY momentum, returns, ATR, realized volatility
  - VIX trend/spike regime features
  - SPY options factors:
    - put/call OI ratio
    - put/call volume ratio
    - 25-delta IV skew
    - ATM straddle-implied move
    - gamma imbalance proxy
- **Model layer** (`src/day_trading_bot/model.py`)
  - Ensemble: Logistic Regression + Random Forest
  - Blended probability output for next-horizon SPY direction
- **Strategy + risk** (`src/day_trading_bot/strategy.py`, `src/day_trading_bot/risk.py`)
  - Probability thresholds and neutral zone
  - VIX circuit breaker
  - position sizing by account risk and ATR stop distance
  - daily loss cap and max trades/day
- **Execution layer** (`src/day_trading_bot/execution/ibkr_executor.py`)
  - IBKR bracket order placement (entry + TP + SL)
  - dry-run mode for safe testing
- **Bot orchestration** (`src/day_trading_bot/pipeline.py`)
  - periodic data pull, retraining, signal generation, order submission
- **Walk-forward backtest** (`src/day_trading_bot/backtest.py`)

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Copy `.env.example` to `.env` and fill credentials.

### 3) One-cycle dry run

```bash
python -m day_trading_bot.main --once --env-file .env
```

### 4) Continuous live loop (still dry-run unless DRY_RUN=false)

```bash
python -m day_trading_bot.main --env-file .env
```

### 5) Backtest with CSV inputs

```bash
python -m day_trading_bot.main --backtest --spy-csv data/spy.csv --vix-csv data/vix.csv --option-factors-csv data/option_factors.csv
```

## Project layout

```
src/day_trading_bot/
  config.py
  data/schwab_collector.py
  execution/ibkr_executor.py
  features.py
  model.py
  strategy.py
  risk.py
  backtest.py
  pipeline.py
  main.py
tests/
research/RESEARCH_NOTES.md
```

## Operational checklist

1. Start IBKR TWS or IB Gateway and enable API connections.
2. Validate Schwab OAuth token generation and refresh flow.
3. Keep `DRY_RUN=true` until:
   - signal quality is validated in backtests + paper trading
   - slippage and fees are modeled realistically
   - kill switches are tested
4. Track live metrics:
   - fill quality
   - model probability calibration
   - daily drawdown and trade count

## Testing

```bash
pytest -q
```
