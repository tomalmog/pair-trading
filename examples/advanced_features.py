"""
Advanced features demonstration:
- Kalman filter for dynamic hedge ratios
- Kelly criterion position sizing
- Volatility-adjusted positions
- Sector rotation
- Regime analysis
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.data_loader import DataLoader
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread import SpreadCalculator
from src.analysis.advanced import (
    KalmanFilter,
    KellyCriterion,
    VolatilityTargeting,
    RegimeAnalyzer,
    SectorRotation
)


def main():
    """Demonstrate advanced features"""

    print("="*80)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("="*80)

    # 1. Load data
    print("\n[1/6] Loading data...")
    loader = DataLoader()
    price_data, sector_mapping = loader.get_sector_data(
        sectors=['Financials', 'Technology'],
        years=2,
        min_dollar_volume=10_000_000
    )

    print(f"Loaded {len(price_data.columns)} stocks")

    # 2. Find pairs
    print("\n[2/6] Finding cointegrated pairs...")
    analyzer = CointegrationAnalyzer()
    pairs = analyzer.find_pairs(price_data, sector_mapping, max_pairs=5)

    if not pairs:
        print("No pairs found!")
        return

    print(f"Found {len(pairs)} pairs")

    # Select first pair for demonstration
    pair = pairs[0]
    print(f"\nAnalyzing: {pair.ticker_a} - {pair.ticker_b}")

    price_a = price_data[pair.ticker_a]
    price_b = price_data[pair.ticker_b]

    # 3. Kalman Filter for Dynamic Hedge Ratio
    print("\n[3/6] Applying Kalman filter for dynamic hedge ratio...")
    kalman = KalmanFilter(delta=1e-4)
    spread_dynamic, hedge_ratio_dynamic = kalman.calculate_dynamic_spread(price_a, price_b)

    # Compare with static hedge ratio
    spread_calc = SpreadCalculator()
    spread_static = spread_calc.calculate_spread(price_a, price_b, pair.hedge_ratio)

    print(f"Static hedge ratio: {pair.hedge_ratio:.4f}")
    print(f"Dynamic hedge ratio (latest): {hedge_ratio_dynamic.iloc[-1]:.4f}")
    print(f"Dynamic hedge ratio (mean): {hedge_ratio_dynamic.mean():.4f}")
    print(f"Dynamic hedge ratio (std): {hedge_ratio_dynamic.std():.4f}")

    # Plot hedge ratio evolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(hedge_ratio_dynamic.index, hedge_ratio_dynamic.values, label='Dynamic (Kalman)', linewidth=2)
    ax1.axhline(y=pair.hedge_ratio, color='red', linestyle='--', label=f'Static ({pair.hedge_ratio:.4f})')
    ax1.set_title(f'{pair.ticker_a}-{pair.ticker_b} Hedge Ratio Evolution')
    ax1.set_ylabel('Hedge Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot spreads comparison
    ax2.plot(spread_static.index, spread_static.values, label='Static Hedge', alpha=0.7)
    ax2.plot(spread_dynamic.index, spread_dynamic.values, label='Dynamic Hedge (Kalman)', alpha=0.7)
    ax2.set_title('Spread Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/kalman_filter_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: results/kalman_filter_analysis.png")
    plt.close()

    # 4. Kelly Criterion Position Sizing
    print("\n[4/6] Calculating Kelly criterion position sizes...")

    # Simulate some historical trades
    win_rate = 0.60
    avg_win = 500
    avg_loss = 300

    kelly = KellyCriterion(fraction=0.5)  # Use half-Kelly for safety
    kelly_fraction = kelly.calculate_kelly_size(win_rate, avg_win, avg_loss)

    print(f"Win rate: {win_rate*100:.1f}%")
    print(f"Average win: ${avg_win:.2f}")
    print(f"Average loss: ${avg_loss:.2f}")
    print(f"Kelly fraction: {kelly_fraction*100:.2f}%")

    capital = 100000
    price = price_a.iloc[-1]
    shares = kelly.calculate_position_size(capital, win_rate, avg_win, avg_loss, price)
    print(f"\nWith ${capital:,.0f} capital and price ${price:.2f}:")
    print(f"Optimal position: {shares:,} shares (${shares*price:,.2f})")

    # 5. Volatility-Adjusted Position Sizing
    print("\n[5/6] Applying volatility targeting...")

    vol_targeting = VolatilityTargeting(target_volatility=0.10)

    # Calculate spread volatility over time
    spread = spread_calc.calculate_spread(price_a, price_b, pair.hedge_ratio)
    base_position = 1000  # shares

    scalar = vol_targeting.calculate_volatility_scalar(spread, window=60)
    adjusted_position = vol_targeting.adjust_position_size(base_position, spread)

    print(f"Base position: {base_position:,} shares")
    print(f"Volatility scalar: {scalar:.2f}x")
    print(f"Adjusted position: {adjusted_position:,.0f} shares")

    current_vol = spread.tail(60).std() * np.sqrt(252)
    print(f"Current spread volatility: {current_vol*100:.2f}%")
    print(f"Target volatility: {vol_targeting.target_volatility*100:.2f}%")

    # 6. Sector Rotation
    print("\n[6/6] Analyzing sector momentum...")

    sector_rotation = SectorRotation(lookback_window=60)
    sector_momentum = sector_rotation.calculate_sector_momentum(price_data, sector_mapping)

    print("\nSector Momentum Rankings:")
    for sector, momentum in sector_momentum.items():
        print(f"  {sector}: {momentum*100:.2f}%")

    # Filter pairs by sector momentum
    filtered_pairs = sector_rotation.filter_pairs_by_sector_momentum(
        pairs,
        sector_momentum,
        min_momentum=0.0
    )

    print(f"\nPairs after sector filter: {len(filtered_pairs)}/{len(pairs)}")

    # 7. Regime Analysis (if VIX data available)
    print("\n[BONUS] Regime Analysis...")
    try:
        regime_analyzer = RegimeAnalyzer()

        # Get date range from price data
        start_date = price_data.index[0].strftime('%Y-%m-%d')
        end_date = price_data.index[-1].strftime('%Y-%m-%d')

        vix_data = regime_analyzer.get_vix_data(start_date, end_date)

        if not vix_data.empty:
            current_vix = vix_data.iloc[-1]
            current_regime = regime_analyzer.identify_regime(current_vix)

            print(f"Current VIX: {current_vix:.2f}")
            print(f"Current regime: {current_regime}")

            # Regime distribution
            regimes = vix_data.apply(regime_analyzer.identify_regime)
            regime_counts = regimes.value_counts()

            print("\nRegime Distribution:")
            for regime, count in regime_counts.items():
                pct = count / len(regimes) * 100
                print(f"  {regime}: {count} days ({pct:.1f}%)")
        else:
            print("VIX data not available")

    except Exception as e:
        print(f"Could not fetch VIX data: {e}")

    # Summary
    print("\n" + "="*80)
    print("ADVANCED FEATURES SUMMARY")
    print("="*80)
    print("\n✅ Kalman Filter: Adapts hedge ratio dynamically")
    print("✅ Kelly Criterion: Optimizes position sizes based on win rate")
    print("✅ Volatility Targeting: Scales positions to maintain constant risk")
    print("✅ Sector Rotation: Filters pairs by sector momentum")
    print("✅ Regime Analysis: Adjusts strategy based on market conditions")
    print("\nThese advanced features can significantly improve:")
    print("- Hedge ratio accuracy (Kalman)")
    print("- Risk-adjusted returns (Kelly + Vol targeting)")
    print("- Trade selection (Sector rotation + Regime analysis)")
    print("="*80)


if __name__ == "__main__":
    # Create results directory
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)

    main()
