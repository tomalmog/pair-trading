"""
Basic usage example for the pairs trading system
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread import MultiPairSpreadCalculator
from src.backtesting.engine import BacktestEngine
from src.backtesting.performance import PerformanceAnalyzer

def main():
    """Run a basic pairs trading analysis"""

    print("="*80)
    print("BASIC PAIRS TRADING EXAMPLE")
    print("="*80)

    # 1. Load data for financial stocks
    print("\n1. Loading data...")
    loader = DataLoader()
    price_data, sector_mapping = loader.get_sector_data(
        sectors=['Financials'],
        years=2,
        min_dollar_volume=10_000_000
    )

    print(f"Loaded {len(price_data.columns)} stocks")
    print(f"Stocks: {', '.join(price_data.columns.tolist())}")

    # 2. Find cointegrated pairs
    print("\n2. Finding cointegrated pairs...")
    analyzer = CointegrationAnalyzer(
        significance_level=0.05,
        min_correlation=0.7,
        max_half_life=30,
        same_sector_only=True
    )

    pairs = analyzer.find_pairs(price_data, sector_mapping, max_pairs=10)

    if not pairs:
        print("No cointegrated pairs found!")
        return

    print(f"\nFound {len(pairs)} pairs:")
    for i, pair in enumerate(pairs[:5], 1):
        print(f"{i}. {pair.ticker_a}-{pair.ticker_b}: "
              f"p-value={pair.adf_pvalue:.4f}, "
              f"half-life={pair.half_life:.1f} days")

    # 3. Calculate spreads and z-scores
    print("\n3. Calculating spreads...")
    spread_calc = MultiPairSpreadCalculator(lookback_window=60)
    spreads = spread_calc.calculate_all_spreads(price_data, pairs)
    zscores = spread_calc.calculate_all_zscores(spreads)

    # Show current z-scores
    current_zscores = spread_calc.get_current_zscores(zscores)
    print("\nCurrent z-scores:")
    for pair, zscore in current_zscores.head(5).items():
        print(f"  {pair}: {zscore:.2f}")

    # 4. Find trading opportunities
    print("\n4. Finding trading opportunities...")
    opportunities = spread_calc.identify_trading_opportunities(
        zscores,
        entry_threshold=2.0,
        exit_threshold=0.5
    )

    if not opportunities.empty:
        print(f"\nFound {len(opportunities)} opportunities:")
        print(opportunities.to_string(index=False))
    else:
        print("No trading opportunities at current thresholds")

    # 5. Run backtest
    print("\n5. Running backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission_per_trade=1.0,
        slippage_pct=0.0005,
        short_borrow_rate=0.02
    )

    results = engine.run_walk_forward_backtest(
        price_data,
        pairs,
        train_period=252,
        test_period=63
    )

    # 6. Analyze performance
    print("\n6. Performance metrics:")
    perf_analyzer = PerformanceAnalyzer()
    metrics = perf_analyzer.calculate_all_metrics(
        results.equity_curve,
        results.daily_returns,
        results.trades
    )

    print(f"\n  Total Return: {metrics.total_return*100:.2f}%")
    print(f"  Annualized Return: {metrics.annualized_return*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown*100:.2f}%")
    print(f"  Win Rate: {metrics.win_rate*100:.2f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Number of Trades: {metrics.num_trades}")

    print("\n" + "="*80)
    print("Example complete!")
    print("\nNext steps:")
    print("- Try different sectors or parameters")
    print("- Run the Streamlit dashboard: streamlit run src/visualization/dashboard.py")
    print("- Explore the full analysis: python main.py")
    print("="*80)


if __name__ == "__main__":
    main()
