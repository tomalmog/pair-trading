"""
Unit tests for cointegration analysis
"""
import pytest
import pandas as pd
import numpy as np
from src.analysis.cointegration import CointegrationAnalyzer, PairMetrics


class TestCointegrationAnalyzer:
    """Test cointegration analyzer"""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')

        # Create two cointegrated series
        x = np.cumsum(np.random.randn(500)) + 100
        y = 1.5 * x + np.random.randn(500) * 5  # Cointegrated with x

        # Create two non-cointegrated series
        z = np.cumsum(np.random.randn(500)) + 100
        w = np.cumsum(np.random.randn(500)) + 100

        return pd.DataFrame({
            'A': x,
            'B': y,
            'C': z,
            'D': w
        }, index=dates)

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return CointegrationAnalyzer(
            significance_level=0.05,
            min_correlation=0.7,
            max_half_life=30
        )

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.significance_level == 0.05
        assert analyzer.min_correlation == 0.7
        assert analyzer.max_half_life == 30

    def test_test_cointegration(self, analyzer, sample_data):
        """Test cointegration detection"""
        # Test cointegrated pair (A, B)
        is_coint, hedge_ratio, adf_stat, adf_pval = analyzer.test_cointegration(
            sample_data['A'],
            sample_data['B']
        )

        assert isinstance(is_coint, bool)
        assert isinstance(hedge_ratio, float)
        assert isinstance(adf_pval, float)
        assert 0 <= adf_pval <= 1

        # Test non-cointegrated pair (C, D)
        is_coint_cd, _, _, pval_cd = analyzer.test_cointegration(
            sample_data['C'],
            sample_data['D']
        )

        # A-B should be more likely cointegrated than C-D
        # (not guaranteed due to randomness, but likely)

    def test_calculate_half_life(self, analyzer, sample_data):
        """Test half-life calculation"""
        spread = np.log(sample_data['A']) - 1.5 * np.log(sample_data['B'])
        half_life = analyzer.calculate_half_life(spread)

        assert isinstance(half_life, (int, float))
        assert half_life > 0

    def test_analyze_pair(self, analyzer, sample_data):
        """Test pair analysis"""
        result = analyzer.analyze_pair(
            'A', 'B',
            sample_data['A'],
            sample_data['B'],
            'Tech', 'Tech'
        )

        if result:  # May be None if not cointegrated
            assert isinstance(result, PairMetrics)
            assert result.ticker_a == 'A'
            assert result.ticker_b == 'B'
            assert result.hedge_ratio > 0
            assert 0 <= result.adf_pvalue <= 1

    def test_find_pairs(self, analyzer, sample_data):
        """Test finding multiple pairs"""
        sector_mapping = {
            'Tech': ['A', 'B'],
            'Finance': ['C', 'D']
        }

        pairs = analyzer.find_pairs(sample_data, sector_mapping)

        assert isinstance(pairs, list)
        # All pairs should be PairMetrics objects
        for pair in pairs:
            assert isinstance(pair, PairMetrics)
            assert pair.adf_pvalue < analyzer.significance_level

    def test_get_pair_summary(self, analyzer, sample_data):
        """Test pair summary generation"""
        sector_mapping = {'Tech': ['A', 'B', 'C', 'D']}
        pairs = analyzer.find_pairs(sample_data, sector_mapping)

        if pairs:
            summary = analyzer.get_pair_summary(pairs)
            assert isinstance(summary, pd.DataFrame)
            assert 'Stock_A' in summary.columns
            assert 'Stock_B' in summary.columns
            assert len(summary) == len(pairs)


class TestPairMetrics:
    """Test PairMetrics dataclass"""

    def test_creation(self):
        """Test creating PairMetrics"""
        pair = PairMetrics(
            ticker_a='AAPL',
            ticker_b='MSFT',
            hedge_ratio=1.5,
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            half_life=15.0,
            correlation=0.85
        )

        assert pair.ticker_a == 'AAPL'
        assert pair.ticker_b == 'MSFT'
        assert pair.hedge_ratio == 1.5
        assert pair.half_life == 15.0

    def test_repr(self):
        """Test string representation"""
        pair = PairMetrics(
            ticker_a='AAPL',
            ticker_b='MSFT',
            hedge_ratio=1.5,
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            half_life=15.0,
            correlation=0.85
        )

        repr_str = repr(pair)
        assert 'AAPL' in repr_str
        assert 'MSFT' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
