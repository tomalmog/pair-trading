"""
Unit tests for performance analysis
"""
import pytest
import pandas as pd
import numpy as np
from src.backtesting.performance import PerformanceAnalyzer
from src.backtesting.engine import Trade


class TestPerformanceAnalyzer:
    """Test performance analyzer"""

    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve"""
        dates = pd.date_range('2020-01-01', periods=252, freq='D')

        # Create equity curve with some volatility and drawdown
        returns = np.random.randn(252) * 0.01 + 0.0005  # Slightly positive drift
        equity = 100000 * (1 + pd.Series(returns, index=dates)).cumprod()

        return equity

    @pytest.fixture
    def sample_returns(self, sample_equity_curve):
        """Calculate returns from equity curve"""
        return sample_equity_curve.pct_change().fillna(0)

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades"""
        trades = []
        for i in range(10):
            trade = Trade(
                pair_name=f"PAIR_{i}",
                entry_date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i*10),
                exit_date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=i*10+5),
                entry_zscore=2.0 if i % 2 == 0 else -2.0,
                exit_zscore=0.0,
                net_pnl=100 if i % 3 != 0 else -50,  # Some wins, some losses
                holding_days=5
            )
            trades.append(trade)

        return trades

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return PerformanceAnalyzer(risk_free_rate=0.0)

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.risk_free_rate == 0.0

    def test_calculate_returns_metrics(self, analyzer, sample_equity_curve, sample_returns):
        """Test return metrics calculation"""
        metrics = analyzer.calculate_returns_metrics(sample_equity_curve, sample_returns)

        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'cumulative_return' in metrics

    def test_calculate_risk_metrics(self, analyzer, sample_equity_curve, sample_returns):
        """Test risk metrics calculation"""
        metrics = analyzer.calculate_risk_metrics(sample_equity_curve, sample_returns)

        assert isinstance(metrics, dict)
        assert 'volatility' in metrics
        assert 'downside_volatility' in metrics
        assert 'max_drawdown' in metrics
        assert 'max_drawdown_duration' in metrics
        assert 'avg_drawdown' in metrics

        # Volatility should be positive
        assert metrics['volatility'] >= 0

    def test_calculate_risk_adjusted_metrics(self, analyzer, sample_returns):
        """Test risk-adjusted metrics"""
        volatility = sample_returns.std() * np.sqrt(252)
        downside_returns = sample_returns[sample_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        metrics = analyzer.calculate_risk_adjusted_metrics(
            sample_returns,
            volatility,
            downside_vol,
            -0.1
        )

        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics

    def test_calculate_trade_metrics(self, analyzer, sample_trades):
        """Test trade metrics calculation"""
        metrics = analyzer.calculate_trade_metrics(sample_trades)

        assert isinstance(metrics, dict)
        assert 'num_trades' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        assert 'expectancy' in metrics

        # Num trades should match
        assert metrics['num_trades'] == len(sample_trades)

        # Win rate should be between 0 and 1
        assert 0 <= metrics['win_rate'] <= 1

    def test_calculate_all_metrics(self, analyzer, sample_equity_curve, sample_returns, sample_trades):
        """Test comprehensive metrics calculation"""
        metrics = analyzer.calculate_all_metrics(
            sample_equity_curve,
            sample_returns,
            sample_trades
        )

        # Check all major metrics are present
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'num_trades')

        # Verify reasonable values
        assert metrics.num_trades == len(sample_trades)
        assert 0 <= metrics.win_rate <= 1

    def test_analyze_by_pair(self, analyzer, sample_trades):
        """Test per-pair analysis"""
        pair_df = analyzer.analyze_by_pair(sample_trades)

        assert isinstance(pair_df, pd.DataFrame)
        assert 'Pair' in pair_df.columns
        assert 'Num_Trades' in pair_df.columns
        assert 'Total_PnL' in pair_df.columns
        assert 'Win_Rate' in pair_df.columns

        # Should have one row per unique pair
        unique_pairs = set(t.pair_name for t in sample_trades)
        assert len(pair_df) == len(unique_pairs)

    def test_empty_trades(self, analyzer, sample_equity_curve, sample_returns):
        """Test handling of empty trade list"""
        metrics = analyzer.calculate_trade_metrics([])

        assert metrics['num_trades'] == 0
        assert metrics['win_rate'] == 0
        assert metrics['profit_factor'] == 0

    def test_drawdown_calculation(self, analyzer):
        """Test drawdown calculation"""
        # Create simple equity curve with known drawdown
        equity = pd.Series([100, 110, 105, 95, 100, 110])
        returns = equity.pct_change().fillna(0)

        metrics = analyzer.calculate_risk_metrics(equity, returns)

        # Should detect the drawdown from 110 to 95
        assert metrics['max_drawdown'] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
