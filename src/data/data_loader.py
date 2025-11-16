"""
Data loader for fetching and managing stock price data
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data download and caching for stock prices"""

    # S&P 500 stocks by sector (sample - you can expand this)
    SECTOR_STOCKS = {
        'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'USB', 'PNC'],
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'ORCL'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES'],
        'Consumer_Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG'],
        'Healthcare': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],
        'Industrials': ['UPS', 'HON', 'UNP', 'RTX', 'BA', 'CAT', 'DE', 'GE', 'LMT', 'MMM'],
        'Consumer_Staples': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ED'],
        'Real_Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB'],
        'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'NUE', 'DOW', 'DD', 'VMC'],
        'Communication_Services': ['GOOG', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA']
    }

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the data loader

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_sp500_tickers(self, sectors: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Get S&P 500 tickers by sector

        Args:
            sectors: List of sectors to include. If None, returns all sectors.

        Returns:
            Dictionary mapping sector names to lists of tickers
        """
        if sectors is None:
            return self.SECTOR_STOCKS

        return {sector: tickers for sector, tickers in self.SECTOR_STOCKS.items()
                if sector in sectors}

    def download_stock_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        years: int = 3
    ) -> pd.DataFrame:
        """
        Download historical price data for given tickers

        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD). If None, calculated from years param
            end_date: End date (YYYY-MM-DD). If None, uses today
            years: Number of years of data (used if start_date is None)

        Returns:
            DataFrame with adjusted close prices
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            start = datetime.now() - timedelta(days=years*365)
            start_date = start.strftime('%Y-%m-%d')

        logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")

        # Check cache
        cache_file = self.cache_dir / f"prices_{start_date}_{end_date}.pkl"
        if cache_file.exists():
            logger.info(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # Check if all tickers are in cache
            missing_tickers = set(tickers) - set(cached_data.columns)
            if not missing_tickers:
                return cached_data[tickers]
            logger.info(f"Missing tickers in cache: {missing_tickers}")

        # Download data
        data_frames = []
        failed_tickers = []

        for ticker in tqdm(tickers, desc="Downloading stock data"):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)

                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Use adjusted close
                data_frames.append(df[['Close']].rename(columns={'Close': ticker}))

            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
                failed_tickers.append(ticker)

        if not data_frames:
            raise ValueError("No data downloaded successfully")

        # Combine all data
        price_data = pd.concat(data_frames, axis=1)
        price_data = price_data.ffill().dropna()  # Forward fill and drop remaining NaNs

        if failed_tickers:
            logger.warning(f"Failed to download: {failed_tickers}")

        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump(price_data, f)

        logger.info(f"Downloaded {len(price_data.columns)} stocks, {len(price_data)} days")

        return price_data

    def get_volume_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        years: int = 3
    ) -> pd.DataFrame:
        """
        Download volume data for liquidity filtering

        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            years: Number of years

        Returns:
            DataFrame with volume data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            start = datetime.now() - timedelta(days=years*365)
            start_date = start.strftime('%Y-%m-%d')

        volume_data = []

        for ticker in tqdm(tickers, desc="Downloading volume data"):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)

                if df.empty:
                    continue

                volume_data.append(df[['Volume']].rename(columns={'Volume': ticker}))

            except Exception as e:
                logger.error(f"Error downloading volume for {ticker}: {e}")

        if not volume_data:
            return pd.DataFrame()

        return pd.concat(volume_data, axis=1)

    def filter_by_liquidity(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        min_dollar_volume: float = 10_000_000
    ) -> pd.DataFrame:
        """
        Filter stocks by minimum average daily dollar volume

        Args:
            price_data: Price DataFrame
            volume_data: Volume DataFrame
            min_dollar_volume: Minimum average daily dollar volume

        Returns:
            Filtered price DataFrame
        """
        # Calculate dollar volume (price * volume)
        dollar_volume = price_data * volume_data

        # Calculate average daily dollar volume
        avg_dollar_volume = dollar_volume.mean()

        # Filter
        liquid_stocks = avg_dollar_volume[avg_dollar_volume >= min_dollar_volume].index.tolist()

        logger.info(f"Filtered to {len(liquid_stocks)} liquid stocks (min ${min_dollar_volume:,.0f} avg daily volume)")

        return price_data[liquid_stocks]

    def get_sector_data(
        self,
        sectors: List[str],
        years: int = 3,
        min_dollar_volume: float = 10_000_000,
        filter_liquidity: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Get price data for specific sectors with optional liquidity filtering

        Args:
            sectors: List of sector names
            years: Years of historical data
            min_dollar_volume: Minimum daily dollar volume
            filter_liquidity: Whether to apply liquidity filter

        Returns:
            Tuple of (price_data, sector_mapping)
        """
        sector_tickers = self.get_sp500_tickers(sectors)
        all_tickers = [ticker for tickers in sector_tickers.values() for ticker in tickers]

        # Remove duplicates
        all_tickers = list(set(all_tickers))

        # Download price data
        price_data = self.download_stock_data(all_tickers, years=years)

        if filter_liquidity:
            # Download volume data
            volume_data = self.get_volume_data(all_tickers, years=years)

            # Filter by liquidity
            price_data = self.filter_by_liquidity(price_data, volume_data, min_dollar_volume)

        # Update sector mapping to only include stocks that passed filters
        filtered_sector_tickers = {
            sector: [t for t in tickers if t in price_data.columns]
            for sector, tickers in sector_tickers.items()
        }

        return price_data, filtered_sector_tickers

    def get_all_data(
        self,
        years: int = 3,
        min_dollar_volume: float = 10_000_000
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Get data for all available sectors

        Args:
            years: Years of historical data
            min_dollar_volume: Minimum daily dollar volume

        Returns:
            Tuple of (price_data, sector_mapping)
        """
        return self.get_sector_data(
            list(self.SECTOR_STOCKS.keys()),
            years=years,
            min_dollar_volume=min_dollar_volume
        )


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()

    # Get data for financial and tech sectors
    price_data, sector_mapping = loader.get_sector_data(
        sectors=['Financials', 'Technology'],
        years=3,
        min_dollar_volume=10_000_000
    )

    print(f"\nLoaded {len(price_data.columns)} stocks")
    print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
    print(f"\nSector breakdown:")
    for sector, tickers in sector_mapping.items():
        print(f"  {sector}: {len(tickers)} stocks")

    print(f"\nSample data:")
    print(price_data.head())
