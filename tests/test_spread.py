"""
Unit tests for spread calculation
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis.spread import SpreadCalculator, MultiPairSpreadCalculator


class TestSpreadCalculator:
    """Test spread calculator"""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        price_a = pd.Series(100 + np.cumsum(np.random.randn(200)), index=dates)
        price_b = pd.Series(50 + np.cumsum(np.random.randn(200)), index=dates)

        return price_a, price_b

    @pytest.fixture
    def calculator(self):
        """Create calculator instance"""
        return SpreadCalculator(lookback_window=60)

    def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert calculator.lookback_window == 60

    def test_calculate_spread(self, calculator, sample_prices):
        """Test spread calculation"""
        price_a, price_b = sample_prices
        hedge_ratio = 1.5

        spread = calculator.calculate_spread(price_a, price_b, hedge_ratio)

        assert isinstance(spread, pd.Series)
        assert len(spread) == len(price_a)
        assert not spread.isnull().all()

    def test_calculate_zscore(self, calculator, sample_prices):
        """Test z-score calculation"""
        price_a, price_b = sample_prices
        spread = calculator.calculate_spread(price_a, price_b, 1.0)

        zscore = calculator.calculate_zscore(spread)

        assert isinstance(zscore, pd.Series)
        assert len(zscore) == len(spread)

        # Z-score should have mean ~0 and std ~1 (after lookback period)
        valid_zscore = zscore.dropna()
        if len(valid_zscore) > 0:
            assert abs(valid_zscore.mean()) < 2  # Roughly centered
            assert 0.5 < valid_zscore.std() < 2  # Roughly normalized

    def test_calculate_spread_metrics(self, calculator, sample_prices):
        """Test spread metrics calculation"""
        price_a, price_b = sample_prices
        spread = calculator.calculate_spread(price_a, price_b, 1.0)

        metrics = calculator.calculate_spread_metrics(spread)

        assert isinstance(metrics, pd.DataFrame)
        assert 'spread' in metrics.columns
        assert 'mean' in metrics.columns
        assert 'std' in metrics.columns
        assert 'zscore' in metrics.columns

    def test_get_spread_statistics(self, calculator, sample_prices):
        """Test spread statistics"""
        price_a, price_b = sample_prices
        spread = calculator.calculate_spread(price_a, price_b, 1.0)

        stats = calculator.get_spread_statistics(spread)

        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'current' in stats
        assert 'current_zscore' in stats

    def test_detect_spread_breakout(self, calculator, sample_prices):
        """Test spread breakout detection"""
        price_a, price_b = sample_prices
        spread = calculator.calculate_spread(price_a, price_b, 1.0)

        breakout = calculator.detect_spread_breakout(spread, threshold=2.0)

        assert isinstance(breakout, pd.Series)
        assert breakout.dtype == bool

    def test_calculate_bollinger_bands(self, calculator, sample_prices):
        """Test Bollinger bands calculation"""
        price_a, price_b = sample_prices
        spread = calculator.calculate_spread(price_a, price_b, 1.0)

        bands = calculator.calculate_bollinger_bands(spread)

        assert isinstance(bands, pd.DataFrame)
        assert 'middle' in bands.columns
        assert 'upper' in bands.columns
        assert 'lower' in bands.columns

        # Upper should be above lower
        valid_bands = bands.dropna()
        if len(valid_bands) > 0:
            assert (valid_bands['upper'] >= valid_bands['lower']).all()


class TestMultiPairSpreadCalculator:
    """Test multi-pair spread calculator"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        return pd.DataFrame({
            'A': 100 + np.cumsum(np.random.randn(200)),
            'B': 50 + np.cumsum(np.random.randn(200)),
            'C': 75 + np.cumsum(np.random.randn(200))
        }, index=dates)

    @pytest.fixture
    def calculator(self):
        """Create calculator instance"""
        return MultiPairSpreadCalculator(lookback_window=60)

    def test_initialization(self, calculator):
        """Test calculator initialization"""
        assert calculator.lookback_window == 60
        assert isinstance(calculator.calculator, SpreadCalculator)

    def test_get_current_zscores(self, calculator):
        """Test getting current z-scores"""
        zscores = {
            'A_B': pd.Series([0.5, 1.0, 1.5, 2.0]),
            'B_C': pd.Series([0.0, -1.0, -2.0, -2.5])
        }

        current = calculator.get_current_zscores(zscores)

        assert isinstance(current, pd.Series)
        assert len(current) == 2
        assert current['A_B'] == 2.0
        assert current['B_C'] == -2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
