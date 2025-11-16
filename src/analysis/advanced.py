"""
Advanced analysis modules including Kalman filter and regime analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Kalman filter for dynamic hedge ratio estimation

    The Kalman filter updates the hedge ratio in real-time as new price data arrives,
    providing better estimates than static OLS regression.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        initial_state_mean: float = 0.0,
        initial_state_cov: float = 1.0
    ):
        """
        Initialize Kalman filter

        Args:
            delta: Transition covariance (controls adaptation speed)
            initial_state_mean: Initial hedge ratio estimate
            initial_state_cov: Initial estimate uncertainty
        """
        self.delta = delta
        self.initial_state_mean = initial_state_mean
        self.initial_state_cov = initial_state_cov

    def estimate_hedge_ratio(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Estimate dynamic hedge ratio using Kalman filter

        Args:
            price_a: Price series for stock A
            price_b: Price series for stock B

        Returns:
            Tuple of (hedge_ratio_series, hedge_ratio_variance_series)
        """
        # Initialize
        state_mean = self.initial_state_mean
        state_cov = self.initial_state_cov

        hedge_ratios = []
        hedge_ratio_vars = []

        # Align indices
        aligned_data = pd.DataFrame({
            'a': price_a,
            'b': price_b
        }).dropna()

        for idx, row in aligned_data.iterrows():
            y = row['a']  # Observation (price of A)
            x = row['b']  # Input (price of B)

            # Prediction step
            state_mean_pred = state_mean
            state_cov_pred = state_cov + self.delta

            # Update step
            residual = y - x * state_mean_pred
            residual_cov = x * state_cov_pred * x + 1.0  # Observation noise = 1

            # Kalman gain
            kalman_gain = state_cov_pred * x / residual_cov

            # State update
            state_mean = state_mean_pred + kalman_gain * residual
            state_cov = state_cov_pred - kalman_gain * x * state_cov_pred

            hedge_ratios.append(state_mean)
            hedge_ratio_vars.append(state_cov)

        return (
            pd.Series(hedge_ratios, index=aligned_data.index),
            pd.Series(hedge_ratio_vars, index=aligned_data.index)
        )

    def calculate_dynamic_spread(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate spread using dynamic hedge ratio

        Args:
            price_a: Price series for stock A
            price_b: Price series for stock B

        Returns:
            Tuple of (spread, hedge_ratio)
        """
        hedge_ratio, _ = self.estimate_hedge_ratio(price_a, price_b)

        # Calculate spread with dynamic hedge ratio
        aligned_data = pd.DataFrame({
            'a': price_a,
            'b': price_b,
            'hr': hedge_ratio
        }).dropna()

        spread = np.log(aligned_data['a']) - aligned_data['hr'] * np.log(aligned_data['b'])

        return spread, hedge_ratio


class KellyCriterion:
    """
    Kelly criterion for optimal position sizing

    Kelly formula: f* = (p*b - q) / b
    where:
    - p = win probability
    - q = loss probability (1-p)
    - b = win/loss ratio
    """

    def __init__(self, fraction: float = 0.5):
        """
        Initialize Kelly criterion calculator

        Args:
            fraction: Fraction of Kelly to use (0.5 = half-Kelly, conservative)
        """
        self.fraction = fraction

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly: float = 0.25
    ) -> float:
        """
        Calculate Kelly fraction for position sizing

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
            max_kelly: Maximum Kelly fraction (cap for safety)

        Returns:
            Kelly fraction (0-1)
        """
        if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0 or avg_win <= 0:
            return 0.0

        # Win/loss ratio
        b = avg_win / avg_loss

        # Kelly formula
        kelly = (win_rate * b - (1 - win_rate)) / b

        # Apply fraction and cap
        kelly = kelly * self.fraction
        kelly = max(0.0, min(kelly, max_kelly))

        return kelly

    def calculate_position_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        price: float
    ) -> int:
        """
        Calculate number of shares to trade

        Args:
            capital: Available capital
            win_rate: Historical win rate
            avg_win: Average win
            avg_loss: Average loss
            price: Current stock price

        Returns:
            Number of shares
        """
        kelly_fraction = self.calculate_kelly_size(win_rate, avg_win, avg_loss)
        position_value = capital * kelly_fraction
        shares = int(position_value / price)

        return shares


class VolatilityTargeting:
    """
    Volatility-adjusted position sizing

    Scales position sizes inversely with volatility to maintain constant risk.
    """

    def __init__(self, target_volatility: float = 0.10):
        """
        Initialize volatility targeting

        Args:
            target_volatility: Target portfolio volatility (e.g., 0.10 = 10% annual)
        """
        self.target_volatility = target_volatility

    def calculate_volatility_scalar(
        self,
        spread: pd.Series,
        window: int = 60
    ) -> float:
        """
        Calculate position size scalar based on current volatility

        Args:
            spread: Spread series
            window: Lookback window for volatility calculation

        Returns:
            Position size scalar (multiply base position by this)
        """
        # Calculate recent volatility
        recent_spread = spread.tail(window)
        current_vol = recent_spread.std() * np.sqrt(252)

        if current_vol <= 0:
            return 0.0

        # Scale inversely with volatility
        scalar = self.target_volatility / current_vol

        # Cap the scalar to prevent extreme positions
        scalar = max(0.1, min(scalar, 3.0))

        return scalar

    def adjust_position_size(
        self,
        base_position: float,
        spread: pd.Series,
        window: int = 60
    ) -> float:
        """
        Adjust position size based on current volatility

        Args:
            base_position: Base position size
            spread: Spread series
            window: Lookback window

        Returns:
            Adjusted position size
        """
        scalar = self.calculate_volatility_scalar(spread, window)
        return base_position * scalar


class RegimeAnalyzer:
    """
    Analyze market regimes based on VIX and other indicators

    Different market regimes may require different trading parameters.
    """

    def __init__(self):
        """Initialize regime analyzer"""
        self.vix_cache = None

    def get_vix_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch VIX data

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            VIX series
        """
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=start_date, end=end_date)
            return vix_data['Close']
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return pd.Series()

    def identify_regime(self, vix_level: float) -> str:
        """
        Identify market regime based on VIX level

        Args:
            vix_level: Current VIX level

        Returns:
            Regime name: 'low_vol', 'normal', 'high_vol', 'crisis'
        """
        if vix_level < 15:
            return 'low_vol'
        elif vix_level < 20:
            return 'normal'
        elif vix_level < 30:
            return 'high_vol'
        else:
            return 'crisis'

    def analyze_performance_by_regime(
        self,
        returns: pd.Series,
        vix_data: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze strategy performance across different regimes

        Args:
            returns: Strategy returns
            vix_data: VIX data aligned with returns

        Returns:
            DataFrame with performance by regime
        """
        # Align data
        aligned = pd.DataFrame({
            'returns': returns,
            'vix': vix_data
        }).dropna()

        # Classify regimes
        aligned['regime'] = aligned['vix'].apply(self.identify_regime)

        # Calculate metrics by regime
        regime_stats = []

        for regime in ['low_vol', 'normal', 'high_vol', 'crisis']:
            regime_data = aligned[aligned['regime'] == regime]

            if len(regime_data) == 0:
                continue

            regime_returns = regime_data['returns']

            stats = {
                'regime': regime,
                'days': len(regime_data),
                'avg_return': regime_returns.mean() * 252,  # Annualized
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                          if regime_returns.std() > 0 else 0),
                'max_drawdown': self._calculate_max_drawdown(regime_returns),
                'avg_vix': regime_data['vix'].mean()
            }

            regime_stats.append(stats)

        return pd.DataFrame(regime_stats)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def should_trade_in_regime(
        self,
        regime: str,
        min_sharpe: float = 0.5
    ) -> bool:
        """
        Determine if strategy should trade in current regime

        Args:
            regime: Current regime
            min_sharpe: Minimum Sharpe ratio to continue trading

        Returns:
            True if should trade
        """
        # In crisis regime, might want to reduce exposure or stop trading
        if regime == 'crisis':
            return False

        return True


class SectorRotation:
    """
    Sector rotation analysis and filtering

    Only trade pairs in sectors with positive momentum.
    """

    def __init__(self, lookback_window: int = 60):
        """
        Initialize sector rotation analyzer

        Args:
            lookback_window: Window for momentum calculation
        """
        self.lookback_window = lookback_window

    def calculate_sector_momentum(
        self,
        price_data: pd.DataFrame,
        sector_mapping: dict
    ) -> pd.Series:
        """
        Calculate momentum for each sector

        Args:
            price_data: Price DataFrame
            sector_mapping: Dict mapping sectors to tickers

        Returns:
            Series with sector momentum scores
        """
        sector_momentum = {}

        for sector, tickers in sector_mapping.items():
            # Get tickers that exist in price data
            available_tickers = [t for t in tickers if t in price_data.columns]

            if not available_tickers:
                continue

            # Calculate sector average momentum
            sector_returns = price_data[available_tickers].pct_change()
            momentum = sector_returns.tail(self.lookback_window).mean().mean()

            sector_momentum[sector] = momentum

        return pd.Series(sector_momentum).sort_values(ascending=False)

    def filter_pairs_by_sector_momentum(
        self,
        pairs: list,
        sector_momentum: pd.Series,
        min_momentum: float = 0.0
    ) -> list:
        """
        Filter pairs to only include those in sectors with positive momentum

        Args:
            pairs: List of PairMetrics
            sector_momentum: Sector momentum scores
            min_momentum: Minimum momentum threshold

        Returns:
            Filtered list of pairs
        """
        filtered_pairs = []

        for pair in pairs:
            sector_a = pair.sector_a
            sector_b = pair.sector_b

            # Get momentum scores
            momentum_a = sector_momentum.get(sector_a, 0)
            momentum_b = sector_momentum.get(sector_b, 0)

            # Both sectors should have positive momentum
            if momentum_a >= min_momentum and momentum_b >= min_momentum:
                filtered_pairs.append(pair)

        logger.info(f"Filtered {len(pairs)} pairs to {len(filtered_pairs)} based on sector momentum")

        return filtered_pairs


if __name__ == "__main__":
    # Example usage
    print("Advanced analysis modules loaded successfully")
    print("\nAvailable classes:")
    print("- KalmanFilter: Dynamic hedge ratio estimation")
    print("- KellyCriterion: Optimal position sizing")
    print("- VolatilityTargeting: Risk-adjusted position sizing")
    print("- RegimeAnalyzer: VIX-based regime analysis")
    print("- SectorRotation: Momentum-based sector filtering")
