"""
Main script for running the statistical arbitrage pairs trading system
"""
import argparse
import logging
from pathlib import Path
import pandas as pd

from src.data.data_loader import DataLoader
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread import MultiPairSpreadCalculator
from src.backtesting.signals import SignalGenerator
from src.backtesting.engine import BacktestEngine
from src.backtesting.performance import PerformanceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(
    sectors=None,
    years=2,
    min_volume=10_000_000,
    significance=0.05,
    min_correlation=0.7,
    max_half_life=30,
    same_sector_only=True,
    max_pairs=20,
    entry_threshold=2.0,
    exit_threshold=0.0,
    stop_threshold=3.0,
    initial_capital=100000,
    save_results=True
):
    """
    Run the complete pairs trading analysis

    Args:
        sectors: List of sectors to analyze
        years: Years of historical data
        min_volume: Minimum daily dollar volume
        significance: Significance level for cointegration
        min_correlation: Minimum correlation threshold
        max_half_life: Maximum half-life in days
        same_sector_only: Only test pairs within same sector
        max_pairs: Maximum number of pairs to analyze
        entry_threshold: Z-score entry threshold
        exit_threshold: Z-score exit threshold
        stop_threshold: Z-score stop loss threshold
        initial_capital: Initial capital for backtesting
        save_results: Whether to save results to file
    """
    if sectors is None:
        sectors = ['Financials', 'Technology']

    logger.info("="*80)
    logger.info("STATISTICAL ARBITRAGE PAIRS TRADING SYSTEM")
    logger.info("="*80)

    # Step 1: Load Data
    logger.info("\n[1/6] Loading stock price data...")
    loader = DataLoader()
    price_data, sector_mapping = loader.get_sector_data(
        sectors=sectors,
        years=years,
        min_dollar_volume=min_volume
    )

    logger.info(f"Loaded {len(price_data.columns)} stocks")
    logger.info(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")

    # Step 2: Find Cointegrated Pairs
    logger.info("\n[2/6] Finding cointegrated pairs...")
    analyzer = CointegrationAnalyzer(
        significance_level=significance,
        min_correlation=min_correlation,
        max_half_life=max_half_life,
        same_sector_only=same_sector_only
    )

    pairs = analyzer.find_pairs(price_data, sector_mapping, max_pairs=max_pairs)

    if not pairs:
        logger.warning("No cointegrated pairs found. Try adjusting parameters.")
        return

    logger.info(f"Found {len(pairs)} cointegrated pairs")

    # Display top pairs
    summary_df = analyzer.get_pair_summary(pairs)
    logger.info("\nTop 5 Pairs:")
    print(summary_df.head().to_string(index=False))

    # Step 3: Calculate Spreads and Z-Scores
    logger.info("\n[3/6] Calculating spreads and z-scores...")
    spread_calc = MultiPairSpreadCalculator(lookback_window=60)
    spreads = spread_calc.calculate_all_spreads(price_data, pairs)
    zscores = spread_calc.calculate_all_zscores(spreads)

    # Show current z-scores
    current_zscores = spread_calc.get_current_zscores(zscores)
    logger.info("\nCurrent Z-Scores (Top 10 by absolute value):")
    print(current_zscores.head(10).to_string())

    # Step 4: Identify Trading Opportunities
    logger.info("\n[4/6] Identifying trading opportunities...")
    opportunities = spread_calc.identify_trading_opportunities(
        zscores,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold
    )

    if not opportunities.empty:
        logger.info(f"\nFound {len(opportunities)} trading opportunities:")
        print(opportunities.to_string(index=False))
    else:
        logger.info("No trading opportunities at current thresholds")

    # Step 5: Run Backtest
    logger.info("\n[5/6] Running backtest...")
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_trade=1.0,
        slippage_pct=0.0005,
        short_borrow_rate=0.02
    )

    signal_gen = SignalGenerator(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_threshold=stop_threshold,
        max_holding_period=20
    )

    results = engine.run_walk_forward_backtest(
        price_data,
        pairs,
        train_period=252,
        test_period=63,
        signal_generator=signal_gen
    )

    # Step 6: Analyze Performance
    logger.info("\n[6/6] Analyzing performance...")
    perf_analyzer = PerformanceAnalyzer(risk_free_rate=0.0)
    metrics = perf_analyzer.calculate_all_metrics(
        results.equity_curve,
        results.daily_returns,
        results.trades
    )

    # Print performance report
    perf_analyzer.print_performance_report(metrics)

    # Analyze by pair
    logger.info("\nPerformance by Pair:")
    pair_performance = perf_analyzer.analyze_by_pair(results.trades)
    print(pair_performance.to_string(index=False))

    # Save results
    if save_results:
        logger.info("\nSaving results...")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Save pair summary
        summary_df.to_csv(results_dir / "pairs_summary.csv", index=False)

        # Save equity curve
        results.equity_curve.to_csv(results_dir / "equity_curve.csv")

        # Save trades
        if results.trades:
            trades_df = pd.DataFrame([
                {
                    'pair': t.pair_name,
                    'entry_date': t.entry_date,
                    'exit_date': t.exit_date,
                    'entry_zscore': t.entry_zscore,
                    'exit_zscore': t.exit_zscore,
                    'direction': 'LONG' if t.direction == 1 else 'SHORT',
                    'holding_days': t.holding_days,
                    'pnl': t.pnl,
                    'net_pnl': t.net_pnl,
                    'exit_reason': t.exit_reason
                }
                for t in results.trades
            ])
            trades_df.to_csv(results_dir / "trades.csv", index=False)

        # Save performance metrics
        with open(results_dir / "performance_metrics.txt", 'w') as f:
            f.write("PERFORMANCE METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Return: {metrics.total_return*100:.2f}%\n")
            f.write(f"Annualized Return: {metrics.annualized_return*100:.2f}%\n")
            f.write(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n")
            f.write(f"Sortino Ratio: {metrics.sortino_ratio:.2f}\n")
            f.write(f"Calmar Ratio: {metrics.calmar_ratio:.2f}\n")
            f.write(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%\n")
            f.write(f"Win Rate: {metrics.win_rate*100:.2f}%\n")
            f.write(f"Profit Factor: {metrics.profit_factor:.2f}\n")
            f.write(f"Number of Trades: {metrics.num_trades}\n")
            f.write(f"Average Holding Period: {metrics.avg_holding_period:.1f} days\n")

        logger.info(f"Results saved to {results_dir}/")

    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info("="*80)

    return {
        'price_data': price_data,
        'pairs': pairs,
        'spreads': spreads,
        'zscores': zscores,
        'results': results,
        'metrics': metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Statistical Arbitrage Pairs Trading System')

    parser.add_argument('--sectors', nargs='+', default=['Financials', 'Technology'],
                       help='Sectors to analyze')
    parser.add_argument('--years', type=int, default=2,
                       help='Years of historical data')
    parser.add_argument('--min-volume', type=float, default=10,
                       help='Minimum daily volume in millions')
    parser.add_argument('--significance', type=float, default=0.05,
                       help='Significance level for cointegration')
    parser.add_argument('--min-correlation', type=float, default=0.7,
                       help='Minimum correlation threshold')
    parser.add_argument('--max-half-life', type=float, default=30,
                       help='Maximum half-life in days')
    parser.add_argument('--same-sector', action='store_true', default=True,
                       help='Only test pairs within same sector')
    parser.add_argument('--max-pairs', type=int, default=20,
                       help='Maximum number of pairs to analyze')
    parser.add_argument('--entry-threshold', type=float, default=2.0,
                       help='Z-score entry threshold')
    parser.add_argument('--exit-threshold', type=float, default=0.0,
                       help='Z-score exit threshold')
    parser.add_argument('--stop-threshold', type=float, default=3.0,
                       help='Z-score stop loss threshold')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')

    args = parser.parse_args()

    main(
        sectors=args.sectors,
        years=args.years,
        min_volume=args.min_volume * 1_000_000,
        significance=args.significance,
        min_correlation=args.min_correlation,
        max_half_life=args.max_half_life,
        same_sector_only=args.same_sector,
        max_pairs=args.max_pairs,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_threshold=args.stop_threshold,
        initial_capital=args.capital,
        save_results=not args.no_save
    )
