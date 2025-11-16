"""
Trading signal generation for pairs trading
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types"""
    ENTRY_LONG = "ENTRY_LONG"      # Long the pair (buy A, short B)
    ENTRY_SHORT = "ENTRY_SHORT"    # Short the pair (short A, buy B)
    EXIT = "EXIT"                   # Exit position (mean reversion)
    STOP_LOSS = "STOP_LOSS"        # Stop loss (diverging further)
    TIME_EXIT = "TIME_EXIT"        # Time-based exit
    NO_SIGNAL = "NO_SIGNAL"        # No action


@dataclass
class Signal:
    """Trading signal"""
    date: pd.Timestamp
    pair_name: str
    signal_type: SignalType
    zscore: float
    spread: float
    reason: str


class SignalGenerator:
    """
    Generate trading signals based on z-score thresholds
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_threshold: float = 3.0,
        max_holding_period: int = 20,
        min_lookback: int = 60
    ):
        """
        Initialize signal generator

        Args:
            entry_threshold: Z-score threshold for entry (default 2.0)
            exit_threshold: Z-score threshold for exit (default 0.0)
            stop_loss_threshold: Z-score threshold for stop loss (default 3.0)
            max_holding_period: Maximum days to hold position (default 20)
            min_lookback: Minimum periods needed before trading (default 60)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_holding_period = max_holding_period
        self.min_lookback = min_lookback

    def generate_signals(
        self,
        zscore: pd.Series,
        spread: pd.Series
    ) -> pd.DataFrame:
        """
        Generate trading signals for a single pair

        Args:
            zscore: Z-score time series
            spread: Spread time series

        Returns:
            DataFrame with signals
        """
        signals = []
        position = 0  # 0 = no position, 1 = long pair, -1 = short pair
        entry_date = None
        entry_zscore = None

        for date in zscore.index:
            # Skip if not enough data
            if pd.isna(zscore.loc[date]):
                continue

            current_z = zscore.loc[date]
            current_spread = spread.loc[date]

            signal_type = SignalType.NO_SIGNAL
            reason = ""

            # Check if we have a position
            if position != 0:
                # Check for exits
                holding_period = (date - entry_date).days if entry_date else 0

                # Stop loss check
                if position == 1 and current_z > self.stop_loss_threshold:
                    signal_type = SignalType.STOP_LOSS
                    reason = f"Long stop loss: z={current_z:.2f} > {self.stop_loss_threshold}"
                    position = 0

                elif position == -1 and current_z < -self.stop_loss_threshold:
                    signal_type = SignalType.STOP_LOSS
                    reason = f"Short stop loss: z={current_z:.2f} < -{self.stop_loss_threshold}"
                    position = 0

                # Mean reversion exit
                elif position == 1 and current_z <= self.exit_threshold:
                    signal_type = SignalType.EXIT
                    reason = f"Long exit: z={current_z:.2f} <= {self.exit_threshold}"
                    position = 0

                elif position == -1 and current_z >= -self.exit_threshold:
                    signal_type = SignalType.EXIT
                    reason = f"Short exit: z={current_z:.2f} >= -{self.exit_threshold}"
                    position = 0

                # Time-based exit
                elif holding_period >= self.max_holding_period:
                    signal_type = SignalType.TIME_EXIT
                    reason = f"Time exit: {holding_period} days >= {self.max_holding_period}"
                    position = 0

            else:
                # No position - check for entries
                # Entry long: spread is too low (z-score < -threshold)
                if current_z < -self.entry_threshold:
                    signal_type = SignalType.ENTRY_LONG
                    reason = f"Entry long: z={current_z:.2f} < -{self.entry_threshold}"
                    position = 1
                    entry_date = date
                    entry_zscore = current_z

                # Entry short: spread is too high (z-score > threshold)
                elif current_z > self.entry_threshold:
                    signal_type = SignalType.ENTRY_SHORT
                    reason = f"Entry short: z={current_z:.2f} > {self.entry_threshold}"
                    position = -1
                    entry_date = date
                    entry_zscore = current_z

            # Record signal if not NO_SIGNAL
            if signal_type != SignalType.NO_SIGNAL:
                signals.append({
                    'date': date,
                    'signal': signal_type.value,
                    'zscore': current_z,
                    'spread': current_spread,
                    'position': position,
                    'reason': reason
                })

        return pd.DataFrame(signals)

    def generate_signals_vectorized(
        self,
        zscore: pd.Series,
        spread: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals using vectorized operations (faster)

        Args:
            zscore: Z-score time series
            spread: Spread time series

        Returns:
            DataFrame with entry and exit signals
        """
        df = pd.DataFrame({
            'zscore': zscore,
            'spread': spread
        })

        # Entry signals
        df['entry_long'] = (df['zscore'] < -self.entry_threshold).astype(int)
        df['entry_short'] = (df['zscore'] > self.entry_threshold).astype(int)

        # Exit signals (mean reversion)
        df['exit_long'] = (df['zscore'] >= self.exit_threshold).astype(int)
        df['exit_short'] = (df['zscore'] <= -self.exit_threshold).astype(int)

        # Stop loss signals
        df['stop_long'] = (df['zscore'] > self.stop_loss_threshold).astype(int)
        df['stop_short'] = (df['zscore'] < -self.stop_loss_threshold).astype(int)

        return df

    def backtest_simple(
        self,
        zscore: pd.Series,
        spread: pd.Series,
        position_size: float = 1.0
    ) -> pd.DataFrame:
        """
        Simple backtest to calculate returns

        Args:
            zscore: Z-score time series
            spread: Spread time series
            position_size: Position size multiplier

        Returns:
            DataFrame with positions and returns
        """
        signals = self.generate_signals_vectorized(zscore, spread)

        # Initialize position tracking
        position = 0
        positions = []

        for idx, row in signals.iterrows():
            # Entry logic
            if position == 0:
                if row['entry_long']:
                    position = position_size
                elif row['entry_short']:
                    position = -position_size

            # Exit logic
            elif position > 0:  # Long position
                if row['exit_long'] or row['stop_long']:
                    position = 0

            elif position < 0:  # Short position
                if row['exit_short'] or row['stop_short']:
                    position = 0

            positions.append(position)

        signals['position'] = positions

        # Calculate returns: position * change in spread
        signals['spread_change'] = spread.diff()
        signals['returns'] = signals['position'].shift(1) * signals['spread_change']
        signals['cumulative_returns'] = signals['returns'].cumsum()

        return signals


class MultiPairSignalGenerator:
    """
    Generate signals for multiple pairs
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_loss_threshold: float = 3.0,
        max_holding_period: int = 20,
        max_concurrent_pairs: int = 5
    ):
        """
        Initialize multi-pair signal generator

        Args:
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_threshold: Z-score threshold for stop loss
            max_holding_period: Maximum holding period in days
            max_concurrent_pairs: Maximum number of concurrent positions
        """
        self.generator = SignalGenerator(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_loss_threshold=stop_loss_threshold,
            max_holding_period=max_holding_period
        )
        self.max_concurrent_pairs = max_concurrent_pairs

    def generate_all_signals(
        self,
        zscores: dict,
        spreads: dict
    ) -> dict:
        """
        Generate signals for all pairs

        Args:
            zscores: Dictionary of z-score series
            spreads: Dictionary of spread series

        Returns:
            Dictionary mapping pair names to signal DataFrames
        """
        all_signals = {}

        for pair_name in zscores.keys():
            if pair_name not in spreads:
                continue

            signals = self.generator.generate_signals(
                zscores[pair_name],
                spreads[pair_name]
            )

            all_signals[pair_name] = signals

        return all_signals

    def get_active_signals(
        self,
        all_signals: dict,
        current_date: pd.Timestamp
    ) -> List[Signal]:
        """
        Get active signals for a specific date

        Args:
            all_signals: Dictionary of signal DataFrames
            current_date: Date to check for signals

        Returns:
            List of Signal objects
        """
        active_signals = []

        for pair_name, signals_df in all_signals.items():
            # Filter to current date
            date_signals = signals_df[signals_df['date'] == current_date]

            for _, row in date_signals.iterrows():
                signal = Signal(
                    date=row['date'],
                    pair_name=pair_name,
                    signal_type=SignalType(row['signal']),
                    zscore=row['zscore'],
                    spread=row['spread'],
                    reason=row['reason']
                )
                active_signals.append(signal)

        return active_signals

    def prioritize_signals(
        self,
        signals: List[Signal],
        current_positions: int
    ) -> List[Signal]:
        """
        Prioritize signals based on z-score magnitude and position limits

        Args:
            signals: List of signals
            current_positions: Number of current open positions

        Returns:
            Prioritized list of signals to execute
        """
        # Separate entry and exit signals
        entry_signals = [s for s in signals if 'ENTRY' in s.signal_type.value]
        exit_signals = [s for s in signals if s.signal_type != SignalType.NO_SIGNAL
                       and 'ENTRY' not in s.signal_type.value]

        # Always execute exit signals
        prioritized = exit_signals.copy()

        # Only add entry signals if under position limit
        available_slots = self.max_concurrent_pairs - current_positions
        if available_slots > 0:
            # Sort entry signals by absolute z-score (highest first)
            entry_signals.sort(key=lambda x: abs(x.zscore), reverse=True)
            prioritized.extend(entry_signals[:available_slots])

        return prioritized


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import DataLoader
    from src.analysis.cointegration import CointegrationAnalyzer
    from src.analysis.spread import MultiPairSpreadCalculator

    # Load data
    loader = DataLoader()
    price_data, sector_mapping = loader.get_sector_data(
        sectors=['Financials'],
        years=2
    )

    # Find pairs
    analyzer = CointegrationAnalyzer()
    pairs = analyzer.find_pairs(price_data, sector_mapping, max_pairs=3)

    if pairs:
        # Calculate spreads and z-scores
        spread_calc = MultiPairSpreadCalculator()
        spreads = spread_calc.calculate_all_spreads(price_data, pairs)
        zscores = spread_calc.calculate_all_zscores(spreads)

        # Generate signals
        signal_gen = MultiPairSignalGenerator(
            entry_threshold=2.0,
            exit_threshold=0.0,
            stop_loss_threshold=3.0,
            max_concurrent_pairs=5
        )

        all_signals = signal_gen.generate_all_signals(zscores, spreads)

        # Show signals for first pair
        first_pair = list(all_signals.keys())[0]
        print(f"\nSignals for {first_pair}:")
        print(all_signals[first_pair])
