"""
Spread construction and z-score calculation for pairs trading
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpreadCalculator:
    """
    Calculate and analyze spreads for cointegrated pairs
    """

    def __init__(self, lookback_window: int = 60):
        """
        Initialize spread calculator

        Args:
            lookback_window: Rolling window for mean/std calculation (default 60 days)
        """
        self.lookback_window = lookback_window

    def calculate_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        hedge_ratio: float,
        use_log: bool = True
    ) -> pd.Series:
        """
        Calculate the spread between two stocks

        Args:
            price_a: Price series for stock A
            price_b: Price series for stock B
            hedge_ratio: Hedge ratio from cointegration test
            use_log: Whether to use log prices (default True)

        Returns:
            Spread series
        """
        if use_log:
            spread = np.log(price_a) - hedge_ratio * np.log(price_b)
        else:
            spread = price_a - hedge_ratio * price_b

        return spread

    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of the spread

        Args:
            spread: Spread series
            window: Rolling window size (uses self.lookback_window if None)

        Returns:
            Z-score series
        """
        if window is None:
            window = self.lookback_window

        # Calculate rolling mean and std
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        # Calculate z-score
        zscore = (spread - rolling_mean) / rolling_std

        return zscore

    def calculate_spread_metrics(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive spread metrics

        Args:
            spread: Spread series
            window: Rolling window size

        Returns:
            DataFrame with spread, mean, std, and z-score
        """
        if window is None:
            window = self.lookback_window

        # Calculate metrics
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std

        # Combine into DataFrame
        metrics = pd.DataFrame({
            'spread': spread,
            'mean': rolling_mean,
            'std': rolling_std,
            'zscore': zscore,
            'upper_2std': rolling_mean + 2 * rolling_std,
            'lower_2std': rolling_mean - 2 * rolling_std,
            'upper_3std': rolling_mean + 3 * rolling_std,
            'lower_3std': rolling_mean - 3 * rolling_std
        })

        return metrics

    def get_spread_statistics(self, spread: pd.Series) -> dict:
        """
        Calculate summary statistics for a spread

        Args:
            spread: Spread series

        Returns:
            Dictionary of statistics
        """
        return {
            'mean': spread.mean(),
            'std': spread.std(),
            'min': spread.min(),
            'max': spread.max(),
            'current': spread.iloc[-1],
            'current_zscore': (spread.iloc[-1] - spread.mean()) / spread.std(),
            'volatility': spread.std() / abs(spread.mean()) if spread.mean() != 0 else np.inf
        }

    def detect_spread_breakout(
        self,
        spread: pd.Series,
        threshold: float = 3.0,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Detect when spread breaks out beyond threshold

        Args:
            spread: Spread series
            threshold: Z-score threshold (default 3.0)
            window: Rolling window size

        Returns:
            Boolean series indicating breakout periods
        """
        zscore = self.calculate_zscore(spread, window)
        breakout = (zscore.abs() > threshold)

        return breakout

    def calculate_spread_velocity(
        self,
        spread: pd.Series,
        periods: int = 5
    ) -> pd.Series:
        """
        Calculate the velocity (rate of change) of the spread

        Args:
            spread: Spread series
            periods: Number of periods for change calculation

        Returns:
            Spread velocity series
        """
        velocity = spread.diff(periods) / periods
        return velocity

    def calculate_bollinger_bands(
        self,
        spread: pd.Series,
        window: Optional[int] = None,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the spread

        Args:
            spread: Spread series
            window: Rolling window size
            num_std: Number of standard deviations for bands

        Returns:
            DataFrame with middle, upper, and lower bands
        """
        if window is None:
            window = self.lookback_window

        middle_band = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        bands = pd.DataFrame({
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band,
            'spread': spread
        })

        return bands

    def identify_mean_crossing(self, spread: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Identify when spread crosses its mean

        Args:
            spread: Spread series
            window: Rolling window size

        Returns:
            Series with values: 1 (cross above), -1 (cross below), 0 (no cross)
        """
        if window is None:
            window = self.lookback_window

        mean = spread.rolling(window=window).mean()

        # Check if spread crosses mean
        above = spread > mean
        crosses = above.astype(int).diff()

        return crosses

    def calculate_spread_percentile(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate the percentile rank of current spread value

        Args:
            spread: Spread series
            window: Rolling window size

        Returns:
            Percentile rank series (0-100)
        """
        if window is None:
            window = self.lookback_window

        def percentile_rank(x):
            if len(x) < 2:
                return 50.0
            return (x < x.iloc[-1]).sum() / len(x) * 100

        percentile = spread.rolling(window=window).apply(percentile_rank, raw=False)

        return percentile


class MultiPairSpreadCalculator:
    """
    Calculate spreads for multiple pairs efficiently
    """

    def __init__(self, lookback_window: int = 60):
        """
        Initialize multi-pair spread calculator

        Args:
            lookback_window: Rolling window for calculations
        """
        self.calculator = SpreadCalculator(lookback_window)
        self.lookback_window = lookback_window

    def calculate_all_spreads(
        self,
        price_data: pd.DataFrame,
        pairs: list,
        use_log: bool = True
    ) -> dict:
        """
        Calculate spreads for all pairs

        Args:
            price_data: DataFrame with stock prices
            pairs: List of PairMetrics objects
            use_log: Whether to use log prices

        Returns:
            Dictionary mapping pair names to spread series
        """
        spreads = {}

        for pair in pairs:
            ticker_a = pair.ticker_a
            ticker_b = pair.ticker_b
            hedge_ratio = pair.hedge_ratio

            pair_name = f"{ticker_a}_{ticker_b}"

            spread = self.calculator.calculate_spread(
                price_data[ticker_a],
                price_data[ticker_b],
                hedge_ratio,
                use_log
            )

            spreads[pair_name] = spread

        return spreads

    def calculate_all_zscores(
        self,
        spreads: dict,
        window: Optional[int] = None
    ) -> dict:
        """
        Calculate z-scores for all spreads

        Args:
            spreads: Dictionary of spread series
            window: Rolling window size

        Returns:
            Dictionary mapping pair names to z-score series
        """
        zscores = {}

        for pair_name, spread in spreads.items():
            zscore = self.calculator.calculate_zscore(spread, window)
            zscores[pair_name] = zscore

        return zscores

    def get_current_zscores(self, zscores: dict) -> pd.Series:
        """
        Get current z-scores for all pairs

        Args:
            zscores: Dictionary of z-score series

        Returns:
            Series with current z-scores
        """
        current = {}
        for pair_name, zscore in zscores.items():
            if len(zscore) > 0 and not pd.isna(zscore.iloc[-1]):
                current[pair_name] = zscore.iloc[-1]

        return pd.Series(current).sort_values()

    def identify_trading_opportunities(
        self,
        zscores: dict,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Identify current trading opportunities across all pairs

        Args:
            zscores: Dictionary of z-score series
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit

        Returns:
            DataFrame with trading opportunities
        """
        opportunities = []

        for pair_name, zscore in zscores.items():
            if len(zscore) == 0:
                continue

            current_z = zscore.iloc[-1]

            if pd.isna(current_z):
                continue

            # Check for entry signals
            if abs(current_z) >= entry_threshold:
                direction = 'SHORT' if current_z > 0 else 'LONG'
                opportunities.append({
                    'pair': pair_name,
                    'zscore': current_z,
                    'signal': 'ENTRY',
                    'direction': direction,
                    'strength': abs(current_z)
                })

            # Check for exit signals (for existing positions)
            elif abs(current_z) <= exit_threshold:
                opportunities.append({
                    'pair': pair_name,
                    'zscore': current_z,
                    'signal': 'EXIT',
                    'direction': 'NEUTRAL',
                    'strength': abs(current_z)
                })

        df = pd.DataFrame(opportunities)
        if not df.empty:
            df = df.sort_values('strength', ascending=False)

        return df


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import DataLoader
    from src.analysis.cointegration import CointegrationAnalyzer

    # Load data
    loader = DataLoader()
    price_data, sector_mapping = loader.get_sector_data(
        sectors=['Financials'],
        years=2
    )

    # Find pairs
    analyzer = CointegrationAnalyzer()
    pairs = analyzer.find_pairs(price_data, sector_mapping, max_pairs=5)

    if pairs:
        # Calculate spreads
        spread_calc = MultiPairSpreadCalculator(lookback_window=60)
        spreads = spread_calc.calculate_all_spreads(price_data, pairs)
        zscores = spread_calc.calculate_all_zscores(spreads)

        # Show current z-scores
        print("\nCurrent Z-Scores:")
        print(spread_calc.get_current_zscores(zscores))

        # Identify opportunities
        opportunities = spread_calc.identify_trading_opportunities(zscores)
        print("\nTrading Opportunities:")
        print(opportunities)
