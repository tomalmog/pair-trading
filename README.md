# Statistical Arbitrage Pairs Trading System

A comprehensive statistical arbitrage system that identifies mean-reverting pairs of stocks, generates trading signals when spreads diverge from equilibrium, and backtests the strategy with realistic transaction costs.

## Features

- **Universe Selection**: S&P 500 constituents with liquidity filtering
- **Cointegration Testing**: Engle-Granger two-step method with ADF test
- **Spread Construction**: Normalized spreads with z-score calculation
- **Signal Generation**: Mean-reversion based entry/exit rules
- **Backtesting Engine**: Walk-forward analysis with realistic transaction costs
- **Performance Analysis**: Comprehensive metrics (Sharpe, Sortino, Max Drawdown, etc.)
- **Interactive Dashboard**: Streamlit-based visualization of pairs and trades

## Project Structure

```
pair-trading/
├── src/
│   ├── data/               # Data collection and management
│   ├── analysis/           # Cointegration testing and pair analysis
│   ├── backtesting/        # Backtesting engine and strategy logic
│   └── visualization/      # Dashboard and plotting utilities
├── tests/                  # Unit tests
├── data/                   # Cached data and results
├── notebooks/              # Jupyter notebooks for analysis
└── main.py                 # Main execution script
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download and analyze pairs
```python
from src.data.data_loader import DataLoader
from src.analysis.cointegration import CointegrationAnalyzer

# Load data
loader = DataLoader()
data = loader.get_sp500_data(years=3)

# Find cointegrated pairs
analyzer = CointegrationAnalyzer()
pairs = analyzer.find_pairs(data, sector_focus=['Technology', 'Financials'])
```

### 2. Run backtest
```python
from src.backtesting.engine import BacktestEngine

engine = BacktestEngine(
    capital=100000,
    commission=1.0,
    slippage=0.0005,
    short_cost=0.02
)

results = engine.run(pairs, data)
```

### 3. Launch dashboard
```bash
streamlit run src/visualization/dashboard.py
```

## Strategy Overview

### Cointegration Testing
- Uses Engle-Granger two-step method
- ADF test with p-value < 0.05 threshold
- Calculates hedge ratio via OLS regression
- Computes half-life of mean reversion

### Trading Signals
- **Entry**: Z-score < -2.0 (long pair) or > +2.0 (short pair)
- **Exit**: Z-score crosses 0 (mean reversion)
- **Stop Loss**: Z-score reaches ±3.0
- **Time Exit**: 20 trading days if no reversion

### Position Sizing
- Risk 1% of portfolio per trade
- Maximum 5 concurrent pairs
- Dollar-neutral positions (equal long/short value)

### Transaction Costs
- Commission: $1 per trade
- Slippage: 0.05% per side (0.1% round-trip)
- Short borrow cost: 2% annual rate (pro-rated)
- Margin requirement: 50% initial margin

## Performance Metrics

- Cumulative & Annualized Returns
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown & Recovery Time
- Win Rate & Profit Factor
- Average Holding Period
- Regime Analysis (by VIX level)

## Advanced Features

- Dynamic hedge ratio estimation with Kalman filter
- Volatility-adjusted position sizing
- Sector rotation filters
- Half-life based pair selection
- Multi-timeframe analysis

## License

MIT License
