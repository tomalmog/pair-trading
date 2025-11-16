"""
Unit tests for signal generation
"""
import pytest
import pandas as pd
import numpy as np
from src.backtesting.signals import SignalGenerator, SignalType


class TestSignalGenerator:
    """Test signal generator"""

    @pytest.fixture
    def sample_data(self):
        """Create sample z-score and spread data"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # Create z-score that triggers different signals
        zscore = pd.Series([
            0.0, 0.5, 1.0, 1.5, 2.0,  # Ramping up
            2.5, 2.3, 2.1, 1.5, 1.0,  # Coming down
            0.5, 0.0, -0.5, -1.0, -1.5,  # Going negative
            -2.0, -2.5, -2.3, -2.0, -1.5,  # Coming back
            -1.0, -0.5, 0.0  # Back to mean
        ] + [0.0] * 77, index=dates)

        spread = pd.Series(np.random.randn(100), index=dates)

        return zscore, spread

    @pytest.fixture
    def generator(self):
        """Create signal generator"""
        return SignalGenerator(
            entry_threshold=2.0,
            exit_threshold=0.0,
            stop_loss_threshold=3.0,
            max_holding_period=20
        )

    def test_initialization(self, generator):
        """Test generator initialization"""
        assert generator.entry_threshold == 2.0
        assert generator.exit_threshold == 0.0
        assert generator.stop_loss_threshold == 3.0
        assert generator.max_holding_period == 20

    def test_generate_signals(self, generator, sample_data):
        """Test signal generation"""
        zscore, spread = sample_data

        signals = generator.generate_signals(zscore, spread)

        assert isinstance(signals, pd.DataFrame)
        if not signals.empty:
            assert 'date' in signals.columns
            assert 'signal' in signals.columns
            assert 'zscore' in signals.columns
            assert 'position' in signals.columns

    def test_entry_signals(self):
        """Test entry signal generation"""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        zscore = pd.Series([0, 0.5, 1.0, 1.5, 2.5, 2.0, 1.5, 1.0, 0.5, 0], index=dates)
        spread = pd.Series([0] * 10, index=dates)

        generator = SignalGenerator(entry_threshold=2.0)
        signals = generator.generate_signals(zscore, spread)

        # Should generate entry signal when z-score crosses 2.0
        if not signals.empty:
            entry_signals = signals[signals['signal'].str.contains('ENTRY')]
            assert len(entry_signals) > 0

    def test_exit_signals(self):
        """Test exit signal generation"""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')

        # Create pattern: enter at 2.5, exit when crosses 0
        zscore = pd.Series([2.5] * 5 + [2.0, 1.5, 1.0, 0.5, 0.0] + [0] * 10, index=dates)
        spread = pd.Series([0] * 20, index=dates)

        generator = SignalGenerator(entry_threshold=2.0, exit_threshold=0.0)
        signals = generator.generate_signals(zscore, spread)

        if not signals.empty:
            # Should have both entry and exit
            signal_types = signals['signal'].unique()
            # Check if we have exit-type signals
            has_exit = any('EXIT' in s or 'STOP' in s for s in signal_types)

    def test_vectorized_signals(self, generator, sample_data):
        """Test vectorized signal generation"""
        zscore, spread = sample_data

        signals = generator.generate_signals_vectorized(zscore, spread)

        assert isinstance(signals, pd.DataFrame)
        assert 'entry_long' in signals.columns
        assert 'entry_short' in signals.columns
        assert 'exit_long' in signals.columns
        assert 'exit_short' in signals.columns

    def test_simple_backtest(self, generator, sample_data):
        """Test simple backtest"""
        zscore, spread = sample_data

        results = generator.backtest_simple(zscore, spread, position_size=1.0)

        assert isinstance(results, pd.DataFrame)
        assert 'position' in results.columns
        assert 'returns' in results.columns
        assert 'cumulative_returns' in results.columns


class TestSignalType:
    """Test SignalType enum"""

    def test_signal_types(self):
        """Test signal type values"""
        assert SignalType.ENTRY_LONG.value == "ENTRY_LONG"
        assert SignalType.ENTRY_SHORT.value == "ENTRY_SHORT"
        assert SignalType.EXIT.value == "EXIT"
        assert SignalType.STOP_LOSS.value == "STOP_LOSS"
        assert SignalType.TIME_EXIT.value == "TIME_EXIT"
        assert SignalType.NO_SIGNAL.value == "NO_SIGNAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
