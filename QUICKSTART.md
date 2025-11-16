# Quick Start Guide

Get started with the Statistical Arbitrage Pairs Trading System in just a few steps!

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd pair-trading
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Run

### Option 1: Run the Example Script (Recommended for First Time)

```bash
python examples/basic_usage.py
```

This will:
- Load 2 years of financial sector data
- Find cointegrated pairs
- Calculate current z-scores
- Identify trading opportunities
- Run a backtest
- Display performance metrics

**Expected runtime**: 2-5 minutes

### Option 2: Run the Full Analysis

```bash
python main.py --sectors Financials Technology --years 2
```

This runs a comprehensive analysis and saves results to the `results/` directory.

**Available options**:
```bash
python main.py --help
```

Example with custom parameters:
```bash
python main.py \
  --sectors Financials Technology Energy \
  --years 3 \
  --min-volume 20 \
  --max-pairs 30 \
  --entry-threshold 2.5 \
  --capital 200000
```

### Option 3: Launch the Interactive Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

This opens a web browser with an interactive dashboard where you can:
- Configure all parameters via UI
- View cointegrated pairs
- Analyze backtest results
- See current trading opportunities
- Examine individual pair spreads
- Export data

**Dashboard features**:
- Real-time parameter adjustments
- Interactive charts (zoom, pan, export)
- Downloadable results
- Multiple visualization tabs

## Understanding the Output

### Pair Analysis
```
Stock_A  Stock_B  Hedge_Ratio  P_Value  Half_Life  Correlation
JPM      BAC      1.2345       0.0012   15.3       0.89
```

- **Hedge Ratio**: How many shares of Stock_B to short per share of Stock_A
- **P_Value**: Lower is better (< 0.05 means cointegrated)
- **Half_Life**: Days for spread to revert halfway to mean
- **Correlation**: Should be high (> 0.7)

### Trading Signals

**Entry Signals**:
- `ENTRY_LONG`: Buy Stock_A, Short Stock_B (z-score < -2)
- `ENTRY_SHORT`: Short Stock_A, Buy Stock_B (z-score > +2)

**Exit Signals**:
- `EXIT`: Close position (z-score crossed 0)
- `STOP_LOSS`: Stop loss hit (z-score beyond ±3)
- `TIME_EXIT`: Max holding period reached (20 days)

### Performance Metrics

**Key Metrics**:
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return (>1 is good, >2 is excellent)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss (>1.5 is good)

## Next Steps

### 1. Customize for Your Needs

Edit parameters in `main.py` or use command-line arguments:

```python
# In main.py or custom script
results = main(
    sectors=['Financials', 'Technology', 'Energy'],
    years=3,
    min_volume=20_000_000,
    significance=0.05,
    entry_threshold=2.5,
    initial_capital=200000
)
```

### 2. Analyze Specific Pairs

```python
from src.data.data_loader import DataLoader
from src.analysis.cointegration import CointegrationAnalyzer

loader = DataLoader()
price_data, _ = loader.get_sector_data(sectors=['Financials'], years=2)

analyzer = CointegrationAnalyzer()

# Test a specific pair
from src.analysis.spread import SpreadCalculator
spread_calc = SpreadCalculator()

# Analyze JPM-BAC
pair_result = analyzer.analyze_pair(
    'JPM', 'BAC',
    price_data['JPM'], price_data['BAC']
)

if pair_result:
    spread = spread_calc.calculate_spread(
        price_data['JPM'],
        price_data['BAC'],
        pair_result.hedge_ratio
    )
    zscore = spread_calc.calculate_zscore(spread)
    print(f"Current z-score: {zscore.iloc[-1]:.2f}")
```

### 3. Run Walk-Forward Analysis

The system automatically uses walk-forward optimization:
- **Training**: 252 days (1 year) to find pairs and calibrate
- **Testing**: 63 days (3 months) to trade
- **Rolling**: Continuously rebalances and recalibrates

Adjust in `main.py`:
```python
results = engine.run_walk_forward_backtest(
    price_data,
    pairs,
    train_period=252,  # Increase for more stable pairs
    test_period=63     # Increase for longer hold periods
)
```

### 4. Add Custom Filters

Filter pairs by additional criteria:

```python
# Filter by sector
financial_pairs = [p for p in pairs if p.sector_a == 'Financials']

# Filter by half-life
fast_pairs = [p for p in pairs if p.half_life < 15]

# Filter by p-value
strong_pairs = [p for p in pairs if p.adf_pvalue < 0.01]
```

### 5. Export Results

All results are automatically saved to `results/`:
- `pairs_summary.csv`: All cointegrated pairs
- `equity_curve.csv`: Daily portfolio values
- `trades.csv`: Complete trade log
- `performance_metrics.txt`: Summary statistics

Load saved results:
```python
import pandas as pd

pairs = pd.read_csv('results/pairs_summary.csv')
equity = pd.read_csv('results/equity_curve.csv', index_col=0, parse_dates=True)
trades = pd.read_csv('results/trades.csv', parse_dates=['entry_date', 'exit_date'])
```

## Common Issues

### 1. No Pairs Found
**Causes**: Parameters too strict, insufficient data, or no cointegration in selected stocks

**Solutions**:
- Increase `significance` from 0.05 to 0.10
- Decrease `min_correlation` from 0.7 to 0.6
- Increase `max_half_life` from 30 to 60
- Try different sectors or more stocks
- Use longer historical data (3+ years)

### 2. Slow Data Download
**Causes**: Downloading many stocks from Yahoo Finance

**Solutions**:
- Data is automatically cached in `data/cache/`
- Reduce number of stocks or sectors
- Subsequent runs will be much faster
- Use `min_volume` filter to reduce universe

### 3. Poor Backtest Performance
**Causes**: Overfitting, changing market conditions, high transaction costs

**Solutions**:
- Use out-of-sample validation
- Increase entry threshold (2.0 → 2.5)
- Check that half-life < 20 days
- Ensure sufficient liquidity
- Consider transaction costs realistic

## Advanced Features

### Kalman Filter for Dynamic Hedge Ratio
Coming soon - use Kalman filter to update hedge ratio in real-time

### Multiple Timeframes
Test strategies on different timeframes:
- Daily (current implementation)
- Hourly (requires intraday data)
- Weekly (for longer-term mean reversion)

### Risk Management Enhancements
- Kelly Criterion position sizing
- Volatility-adjusted positions
- Portfolio-level stop losses
- Correlation-based pair selection

## Resources

- **Documentation**: See README.md for detailed architecture
- **Examples**: Check `examples/` directory
- **Tests**: Run `pytest tests/` (coming soon)
- **Issues**: Report bugs or request features on GitHub

## Support

For questions or issues:
1. Check the examples in `examples/`
2. Review the code documentation
3. Open an issue on GitHub
4. Consult the academic papers on pairs trading

Happy trading! 📈
