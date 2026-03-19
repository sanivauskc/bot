# Research notes: SPY/VIX/options intraday model design

## Objective

Build a profit-seeking day-trading system that uses:

1. SPY intraday bars
2. VIX intraday regime context
3. SPY options surface/flow signals

while using:

- Schwab API for data collection
- IBKR API for execution

## API research summary

### Schwab

- Schwab Trader API supports intraday price history and option chains via OAuth2.
- Option chain payload includes call/put maps, strike-level quotes, and greeks (when available).
- Minute-level history depth is limited, so frequent snapshots and local persistence are important.

Primary references:

- https://developer.schwab.com/products/trader-api--individual/details/documentation/Retail%20Trader%20API%20Production
- https://schwab-py.readthedocs.io/en/latest/client.html

### IBKR

- `ib_insync` wraps TWS/Gateway API and supports non-blocking order management.
- Bracket orders require parent + child transmit coordination.
- API can supply account net liquidation for risk-aware position sizing.

Primary references:

- https://ib-insync.readthedocs.io/api.html
- https://interactivebrokers.github.io/tws-api/historical_bars.html
- https://interactivebrokers.github.io/tws-api/realtime_bars.html

## Signal research summary

The strategy uses a hybrid of:

- directional momentum and mean-reversion proxies (SPY returns, momentum, volume z-score)
- volatility regime variables (VIX level, return, moving-average ratio, spike flag)
- options-based sentiment/positioning proxies:
  - put/call OI and volume ratios
  - 25-delta IV skew (put IV minus call IV)
  - ATM straddle cost as implied short-term move
  - gamma imbalance (call gamma OI minus put gamma OI, normalized)

Practical rationale:

- VIX captures market-wide stress and often affects intraday trend persistence.
- SPY options skew/flow often prices downside fear before spot moves.
- Combining nonlinear tree model + linear model helps reduce single-model failure modes.

Example public references used for factor selection context:

- https://orats.com/blog/backtesting-basics-trading-spy-options-based-on-contango-skew-and-vix-levels
- https://0dteoption.com/vix1d-true-range-quant-guide/

## Risk architecture

- VIX circuit breaker to avoid execution during dislocated regimes.
- Position sizing from account risk budget and ATR stop distance.
- Daily loss cutoff and max trades/day.
- Bracket orders for immediate protective exits.

## Why profitability is not guaranteed

Even statistically sound features can degrade due to:

- regime shifts
- liquidity shocks
- news-event jumps
- changing microstructure and options dealer positioning
- fee/slippage underestimation

Therefore, this repository includes backtesting and dry-run deployment defaults before any live capital usage.
