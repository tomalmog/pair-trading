"""
Backtesting engine for pairs trading strategy
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from src.backtesting.signals import SignalGenerator, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade"""
    pair_name: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_zscore: float = 0.0
    exit_zscore: float = 0.0
    entry_spread: float = 0.0
    exit_spread: float = 0.0
    direction: int = 0  # 1 = long pair, -1 = short pair
    position_size: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    borrow_cost: float = 0.0
    total_cost: float = 0.0
    net_pnl: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""
    ticker_a: str = ""
    ticker_b: str = ""
    hedge_ratio: float = 0.0
    shares_a: float = 0.0
    shares_b: float = 0.0


@dataclass
class BacktestResults:
    """Results from backtesting"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series())
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series())
    total_pnl: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_holding_days: float = 0.0


class BacktestEngine:
    """
    Backtesting engine with realistic transaction costs
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_per_trade: float = 1.0,
        slippage_pct: float = 0.0005,  # 0.05% per side
        short_borrow_rate: float = 0.02,  # 2% annual
        position_size_pct: float = 0.2,  # 20% of capital per pair
        max_positions: int = 5,
        risk_per_trade_pct: float = 0.01  # 1% risk per trade
    ):
        """
        Initialize backtesting engine

        Args:
            initial_capital: Starting capital
            commission_per_trade: Commission per trade leg ($)
            slippage_pct: Slippage percentage per side
            short_borrow_rate: Annual short borrow cost rate
            position_size_pct: Position size as % of capital
            max_positions: Maximum concurrent positions
            risk_per_trade_pct: Risk per trade as % of capital
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        self.short_borrow_rate = short_borrow_rate
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct

    def calculate_position_size(
        self,
        capital: float,
        price_a: float,
        price_b: float,
        hedge_ratio: float,
        volatility: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for both stocks in a pair

        Args:
            capital: Available capital
            price_a: Current price of stock A
            price_b: Current price of stock B
            hedge_ratio: Hedge ratio from cointegration
            volatility: Spread volatility for risk adjustment

        Returns:
            Tuple of (shares_a, shares_b)
        """
        # Calculate position value
        position_value = capital * self.position_size_pct

        # For dollar-neutral pair, we want equal dollar amounts long and short
        # Position in A = position_value / 2
        # Position in B = position_value / 2

        shares_a = (position_value / 2) / price_a
        shares_b = (position_value / 2) / price_b

        # Adjust shares_b based on hedge ratio to maintain cointegration relationship
        # This is a simplified approach - in practice you might use the hedge ratio differently
        shares_b = shares_b * hedge_ratio

        return shares_a, shares_b

    def calculate_transaction_costs(
        self,
        shares_a: float,
        shares_b: float,
        price_a: float,
        price_b: float,
        holding_days: int = 0,
        is_entry: bool = True
    ) -> Tuple[float, float, float]:
        """
        Calculate transaction costs

        Args:
            shares_a: Number of shares of stock A
            shares_b: Number of shares of stock B
            price_a: Price of stock A
            price_b: Price of stock B
            holding_days: Days held (for borrow cost)
            is_entry: Whether this is entry (True) or exit (False)

        Returns:
            Tuple of (commission, slippage, borrow_cost)
        """
        # Commission: 4 legs (buy A, short B on entry; close both on exit)
        commission = 4 * self.commission_per_trade

        # Slippage: 0.05% per side on the total value traded
        value_a = shares_a * price_a
        value_b = shares_b * price_b
        total_value = value_a + value_b

        slippage = total_value * self.slippage_pct * 2  # Both entry and exit

        # Borrow cost: Only on the short leg (stock B when long pair, stock A when short pair)
        # Pro-rated by holding days
        if holding_days > 0:
            borrow_cost = value_b * self.short_borrow_rate * (holding_days / 365)
        else:
            borrow_cost = 0.0

        return commission, slippage, borrow_cost

    def run_backtest_single_pair(
        self,
        pair_name: str,
        ticker_a: str,
        ticker_b: str,
        price_a: pd.Series,
        price_b: pd.Series,
        zscore: pd.Series,
        spread: pd.Series,
        hedge_ratio: float,
        signal_generator: SignalGenerator
    ) -> List[Trade]:
        """
        Run backtest for a single pair

        Args:
            pair_name: Name of the pair
            ticker_a: Ticker for stock A
            ticker_b: Ticker for stock B
            price_a: Price series for stock A
            price_b: Price series for stock B
            zscore: Z-score series
            spread: Spread series
            hedge_ratio: Hedge ratio
            signal_generator: Signal generator instance

        Returns:
            List of Trade objects
        """
        # Generate signals
        signals = signal_generator.generate_signals(zscore, spread)

        if signals.empty:
            return []

        trades = []
        current_trade = None
        current_capital = self.initial_capital

        for _, signal_row in signals.iterrows():
            date = signal_row['date']
            signal = signal_row['signal']
            current_z = signal_row['zscore']
            current_spread = signal_row['spread']

            # Get current prices
            if date not in price_a.index or date not in price_b.index:
                continue

            curr_price_a = price_a.loc[date]
            curr_price_b = price_b.loc[date]

            # Entry signals
            if signal in ['ENTRY_LONG', 'ENTRY_SHORT'] and current_trade is None:
                # Calculate position size
                shares_a, shares_b = self.calculate_position_size(
                    current_capital,
                    curr_price_a,
                    curr_price_b,
                    hedge_ratio
                )

                # Create trade
                current_trade = Trade(
                    pair_name=pair_name,
                    entry_date=date,
                    entry_zscore=current_z,
                    entry_spread=current_spread,
                    direction=1 if signal == 'ENTRY_LONG' else -1,
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    hedge_ratio=hedge_ratio,
                    shares_a=shares_a,
                    shares_b=shares_b
                )

            # Exit signals
            elif signal in ['EXIT', 'STOP_LOSS', 'TIME_EXIT'] and current_trade is not None:
                current_trade.exit_date = date
                current_trade.exit_zscore = current_z
                current_trade.exit_spread = current_spread
                current_trade.exit_reason = signal
                current_trade.holding_days = (date - current_trade.entry_date).days

                # Calculate P&L
                # For long pair: profit = (spread_exit - spread_entry) * position_size
                # For short pair: profit = (spread_entry - spread_exit) * position_size
                spread_change = current_trade.exit_spread - current_trade.entry_spread
                current_trade.pnl = current_trade.direction * spread_change * current_trade.shares_a * curr_price_a

                # Calculate costs
                commission, slippage, borrow_cost = self.calculate_transaction_costs(
                    current_trade.shares_a,
                    current_trade.shares_b,
                    curr_price_a,
                    curr_price_b,
                    current_trade.holding_days,
                    is_entry=False
                )

                current_trade.commission = commission
                current_trade.slippage = slippage
                current_trade.borrow_cost = borrow_cost
                current_trade.total_cost = commission + slippage + borrow_cost
                current_trade.net_pnl = current_trade.pnl - current_trade.total_cost

                # Update capital
                current_capital += current_trade.net_pnl

                trades.append(current_trade)
                current_trade = None

        return trades

    def run_walk_forward_backtest(
        self,
        price_data: pd.DataFrame,
        pairs: list,
        train_period: int = 252,  # 1 year
        test_period: int = 63,    # 3 months
        signal_generator: Optional[SignalGenerator] = None
    ) -> BacktestResults:
        """
        Run walk-forward backtest

        Args:
            price_data: DataFrame with stock prices
            pairs: List of PairMetrics objects
            train_period: Training period in days
            test_period: Testing period in days
            signal_generator: Custom signal generator (uses default if None)

        Returns:
            BacktestResults object
        """
        if signal_generator is None:
            signal_generator = SignalGenerator()

        from src.analysis.spread import SpreadCalculator

        spread_calc = SpreadCalculator()

        all_trades = []
        equity_curve = []

        # Walk forward
        start_idx = train_period
        end_idx = len(price_data)

        while start_idx < end_idx:
            test_end_idx = min(start_idx + test_period, end_idx)

            # Get test period data
            test_data = price_data.iloc[start_idx:test_end_idx]

            # Run backtest for each pair
            for pair in pairs:
                ticker_a = pair.ticker_a
                ticker_b = pair.ticker_b
                hedge_ratio = pair.hedge_ratio

                if ticker_a not in test_data.columns or ticker_b not in test_data.columns:
                    continue

                # Calculate spread and z-score
                spread = spread_calc.calculate_spread(
                    test_data[ticker_a],
                    test_data[ticker_b],
                    hedge_ratio
                )

                zscore = spread_calc.calculate_zscore(spread)

                # Run backtest
                trades = self.run_backtest_single_pair(
                    f"{ticker_a}_{ticker_b}",
                    ticker_a,
                    ticker_b,
                    test_data[ticker_a],
                    test_data[ticker_b],
                    zscore,
                    spread,
                    hedge_ratio,
                    signal_generator
                )

                all_trades.extend(trades)

            # Move to next period
            start_idx = test_end_idx

        # Calculate results
        results = self._calculate_results(all_trades, price_data.index)

        return results

    def _calculate_results(
        self,
        trades: List[Trade],
        date_index: pd.DatetimeIndex
    ) -> BacktestResults:
        """
        Calculate backtest results from trades

        Args:
            trades: List of Trade objects
            date_index: DatetimeIndex for equity curve

        Returns:
            BacktestResults object
        """
        if not trades:
            return BacktestResults()

        # Calculate equity curve
        capital = self.initial_capital
        equity_curve = pd.Series(index=date_index, data=capital)

        for trade in trades:
            if trade.exit_date and trade.exit_date in equity_curve.index:
                capital += trade.net_pnl
                equity_curve.loc[trade.exit_date:] = capital

        # Calculate returns
        daily_returns = equity_curve.pct_change().fillna(0)

        # Calculate metrics
        total_pnl = sum(t.net_pnl for t in trades)
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        years = len(date_index) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                       if daily_returns.std() > 0 else 0)

        # Maximum drawdown
        cumulative = equity_curve
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = [t for t in trades if t.net_pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Average holding days
        avg_holding_days = np.mean([t.holding_days for t in trades if t.holding_days > 0])

        return BacktestResults(
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            total_pnl=total_pnl,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            avg_holding_days=avg_holding_days
        )


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
        # Run backtest
        engine = BacktestEngine(
            initial_capital=100000,
            commission_per_trade=1.0,
            slippage_pct=0.0005,
            short_borrow_rate=0.02
        )

        results = engine.run_walk_forward_backtest(
            price_data,
            pairs,
            train_period=252,
            test_period=63
        )

        print(f"\nBacktest Results:")
        print(f"Total Return: {results.total_return*100:.2f}%")
        print(f"Annualized Return: {results.annualized_return*100:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown*100:.2f}%")
        print(f"Win Rate: {results.win_rate*100:.2f}%")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Number of Trades: {results.num_trades}")
        print(f"Avg Holding Days: {results.avg_holding_days:.1f}")
