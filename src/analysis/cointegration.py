"""
Cointegration testing and pair identification
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from itertools import combinations
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PairMetrics:
    """Store metrics for a cointegrated pair"""
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    adf_statistic: float
    adf_pvalue: float
    half_life: float
    correlation: float
    sector_a: Optional[str] = None
    sector_b: Optional[str] = None

    def __repr__(self):
        return (f"PairMetrics({self.ticker_a}-{self.ticker_b}, "
                f"p={self.adf_pvalue:.4f}, half_life={self.half_life:.1f})")


class CointegrationAnalyzer:
    """
    Analyze stock pairs for cointegration using Engle-Granger two-step method
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_correlation: float = 0.7,
        max_half_life: float = 30.0,
        same_sector_only: bool = False
    ):
        """
        Initialize the cointegration analyzer

        Args:
            significance_level: P-value threshold for ADF test (default 0.05)
            min_correlation: Minimum correlation coefficient (default 0.7)
            max_half_life: Maximum half-life in days (default 30)
            same_sector_only: Only test pairs within same sector
        """
        self.significance_level = significance_level
        self.min_correlation = min_correlation
        self.max_half_life = max_half_life
        self.same_sector_only = same_sector_only

    def test_cointegration(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[bool, float, float, float]:
        """
        Test if two price series are cointegrated using Engle-Granger method

        Args:
            price_a: Price series for stock A
            price_b: Price series for stock B

        Returns:
            Tuple of (is_cointegrated, hedge_ratio, adf_statistic, adf_pvalue)
        """
        # Use log prices for better stationarity
        log_a = np.log(price_a)
        log_b = np.log(price_b)

        # Step 1: Run OLS regression to get hedge ratio
        model = OLS(log_a, log_b).fit()
        hedge_ratio = model.params[0]

        # Calculate residuals (spread)
        residuals = log_a - hedge_ratio * log_b

        # Step 2: Test residuals for stationarity using ADF test
        adf_result = adfuller(residuals, maxlag=1, regression='c', autolag=None)
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]

        # Check if cointegrated
        is_cointegrated = adf_pvalue < self.significance_level

        return is_cointegrated, hedge_ratio, adf_statistic, adf_pvalue

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion for a spread

        Uses AR(1) model: spread(t) = lambda * spread(t-1) + epsilon
        Half-life = -log(2) / log(lambda)

        Args:
            spread: Spread time series

        Returns:
            Half-life in days
        """
        # Lag the spread
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align the series
        spread_lag = spread_lag.iloc[1:]
        spread_diff = spread_diff.iloc[1:]

        # Run regression: spread_diff = alpha + beta * spread_lag + error
        model = OLS(spread_diff, spread_lag).fit()
        beta = model.params[0]

        # Calculate half-life
        if beta >= 0:
            # Non-mean-reverting
            return np.inf

        half_life = -np.log(2) / np.log(1 + beta)

        return half_life

    def analyze_pair(
        self,
        ticker_a: str,
        ticker_b: str,
        price_a: pd.Series,
        price_b: pd.Series,
        sector_a: Optional[str] = None,
        sector_b: Optional[str] = None
    ) -> Optional[PairMetrics]:
        """
        Analyze a single pair for cointegration

        Args:
            ticker_a: Ticker symbol for stock A
            ticker_b: Ticker symbol for stock B
            price_a: Price series for stock A
            price_b: Price series for stock B
            sector_a: Sector for stock A
            sector_b: Sector for stock B

        Returns:
            PairMetrics if pair is cointegrated, None otherwise
        """
        # Check if same sector requirement is met
        if self.same_sector_only and sector_a != sector_b:
            return None

        # Check correlation first (cheaper computation)
        correlation = price_a.corr(price_b)
        if correlation < self.min_correlation:
            return None

        # Test for cointegration
        is_coint, hedge_ratio, adf_stat, adf_pval = self.test_cointegration(price_a, price_b)

        if not is_coint:
            return None

        # Calculate spread
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        spread = log_a - hedge_ratio * log_b

        # Calculate half-life
        half_life = self.calculate_half_life(spread)

        # Filter by half-life
        if half_life > self.max_half_life:
            return None

        return PairMetrics(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            hedge_ratio=hedge_ratio,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pval,
            half_life=half_life,
            correlation=correlation,
            sector_a=sector_a,
            sector_b=sector_b
        )

    def find_pairs(
        self,
        price_data: pd.DataFrame,
        sector_mapping: Optional[Dict[str, List[str]]] = None,
        max_pairs: Optional[int] = None
    ) -> List[PairMetrics]:
        """
        Find all cointegrated pairs in the dataset

        Args:
            price_data: DataFrame with stock prices
            sector_mapping: Dictionary mapping sector names to ticker lists
            max_pairs: Maximum number of pairs to return (best ones by p-value)

        Returns:
            List of PairMetrics for cointegrated pairs
        """
        tickers = price_data.columns.tolist()
        n_tickers = len(tickers)
        n_combinations = n_tickers * (n_tickers - 1) // 2

        logger.info(f"Testing {n_combinations} pair combinations from {n_tickers} stocks")

        # Create sector lookup
        ticker_to_sector = {}
        if sector_mapping:
            for sector, ticker_list in sector_mapping.items():
                for ticker in ticker_list:
                    ticker_to_sector[ticker] = sector

        # Test all pairs
        cointegrated_pairs = []

        for ticker_a, ticker_b in tqdm(combinations(tickers, 2), total=n_combinations, desc="Testing pairs"):
            # Get sector information
            sector_a = ticker_to_sector.get(ticker_a)
            sector_b = ticker_to_sector.get(ticker_b)

            # Analyze pair
            pair_metrics = self.analyze_pair(
                ticker_a, ticker_b,
                price_data[ticker_a], price_data[ticker_b],
                sector_a, sector_b
            )

            if pair_metrics:
                cointegrated_pairs.append(pair_metrics)

        # Sort by p-value (lower is better)
        cointegrated_pairs.sort(key=lambda x: x.adf_pvalue)

        # Limit to top pairs if specified
        if max_pairs and len(cointegrated_pairs) > max_pairs:
            cointegrated_pairs = cointegrated_pairs[:max_pairs]

        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")

        return cointegrated_pairs

    def get_pair_summary(self, pairs: List[PairMetrics]) -> pd.DataFrame:
        """
        Create a summary DataFrame of pair metrics

        Args:
            pairs: List of PairMetrics

        Returns:
            DataFrame with pair summary
        """
        data = []
        for pair in pairs:
            data.append({
                'Stock_A': pair.ticker_a,
                'Stock_B': pair.ticker_b,
                'Hedge_Ratio': pair.hedge_ratio,
                'ADF_Statistic': pair.adf_statistic,
                'P_Value': pair.adf_pvalue,
                'Half_Life': pair.half_life,
                'Correlation': pair.correlation,
                'Sector_A': pair.sector_a,
                'Sector_B': pair.sector_b,
                'Same_Sector': pair.sector_a == pair.sector_b
            })

        return pd.DataFrame(data)

    def validate_pairs_out_of_sample(
        self,
        pairs: List[PairMetrics],
        price_data: pd.DataFrame,
        train_end_idx: int
    ) -> List[PairMetrics]:
        """
        Validate pairs on out-of-sample data

        Args:
            pairs: List of pairs found in training period
            price_data: Full price DataFrame
            train_end_idx: Index where training period ends

        Returns:
            List of pairs that remain cointegrated out-of-sample
        """
        test_data = price_data.iloc[train_end_idx:]

        validated_pairs = []

        for pair in tqdm(pairs, desc="Validating pairs"):
            ticker_a = pair.ticker_a
            ticker_b = pair.ticker_b

            # Re-test on out-of-sample data
            is_coint, hedge_ratio, adf_stat, adf_pval = self.test_cointegration(
                test_data[ticker_a],
                test_data[ticker_b]
            )

            if is_coint:
                validated_pairs.append(pair)

        logger.info(f"{len(validated_pairs)}/{len(pairs)} pairs validated out-of-sample")

        return validated_pairs


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    price_data, sector_mapping = loader.get_sector_data(
        sectors=['Financials', 'Technology'],
        years=3
    )

    # Find cointegrated pairs
    analyzer = CointegrationAnalyzer(
        significance_level=0.05,
        min_correlation=0.7,
        max_half_life=30,
        same_sector_only=True
    )

    pairs = analyzer.find_pairs(price_data, sector_mapping, max_pairs=20)

    # Print summary
    summary = analyzer.get_pair_summary(pairs)
    print("\nTop Cointegrated Pairs:")
    print(summary.to_string(index=False))

    # Validate out-of-sample
    train_end = int(len(price_data) * 0.7)
    validated_pairs = analyzer.validate_pairs_out_of_sample(pairs, price_data, train_end)

    print(f"\n{len(validated_pairs)} pairs validated out-of-sample")
