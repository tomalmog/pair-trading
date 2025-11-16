"""
Performance metrics and analysis for backtesting results
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float

    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_period: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Additional metrics
    expectancy: float
    recovery_factor: float
    ulcer_index: float


class PerformanceAnalyzer:
    """
    Analyze backtest performance and calculate metrics
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance analyzer

        Args:
            risk_free_rate: Annual risk-free rate (default 0.0)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """
        Calculate return-based metrics

        Args:
            equity_curve: Equity curve
            returns: Daily returns

        Returns:
            Dictionary of metrics
        """
        initial_capital = equity_curve.iloc[0]
        final_capital = equity_curve.iloc[-1]

        # Total return
        total_return = (final_capital - initial_capital) / initial_capital

        # Annualized return
        years = len(equity_curve) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Cumulative return
        cumulative_return = (1 + returns).cumprod() - 1

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return.iloc[-1] if len(cumulative_return) > 0 else 0
        }

    def calculate_risk_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """
        Calculate risk metrics

        Args:
            equity_curve: Equity curve
            returns: Daily returns

        Returns:
            Dictionary of risk metrics
        """
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Downside volatility (for Sortino)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Drawdown analysis
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        max_drawdown = drawdown.min()

        # Drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(equity_curve)

        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Ulcer Index (measure of downside volatility)
        ulcer_index = np.sqrt((drawdown ** 2).mean())

        return {
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': drawdown_duration,
            'avg_drawdown': avg_drawdown,
            'ulcer_index': ulcer_index
        }

    def _calculate_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        running_max = equity_curve.expanding().max()
        drawdown = equity_curve < running_max

        max_duration = 0
        current_duration = 0

        for is_drawdown in drawdown:
            if is_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def calculate_risk_adjusted_metrics(
        self,
        returns: pd.Series,
        volatility: float,
        downside_volatility: float,
        max_drawdown: float
    ) -> Dict:
        """
        Calculate risk-adjusted performance metrics

        Args:
            returns: Daily returns
            volatility: Annualized volatility
            downside_volatility: Annualized downside volatility
            max_drawdown: Maximum drawdown

        Returns:
            Dictionary of risk-adjusted metrics
        """
        # Annualized return
        annualized_return = returns.mean() * 252

        # Sharpe ratio
        sharpe_ratio = ((annualized_return - self.risk_free_rate) / volatility
                       if volatility > 0 else 0)

        # Sortino ratio
        sortino_ratio = ((annualized_return - self.risk_free_rate) / downside_volatility
                        if downside_volatility > 0 else 0)

        # Calmar ratio
        calmar_ratio = (annualized_return / abs(max_drawdown)
                       if max_drawdown < 0 else 0)

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }

    def calculate_trade_metrics(self, trades: List) -> Dict:
        """
        Calculate trade-based metrics

        Args:
            trades: List of Trade objects

        Returns:
            Dictionary of trade metrics
        """
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_holding_period': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'expectancy': 0
            }

        # Number of trades
        num_trades = len(trades)

        # Winning and losing trades
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]

        # Win rate
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

        # Average win/loss
        avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.net_pnl) for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t.net_pnl for t in winning_trades)
        gross_loss = abs(sum(t.net_pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Average holding period
        avg_holding_period = np.mean([t.holding_days for t in trades if t.holding_days > 0])

        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks(trades)

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_period': avg_holding_period,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'expectancy': expectancy
        }

    def _calculate_consecutive_streaks(self, trades: List) -> tuple:
        """Calculate max consecutive wins and losses"""
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.net_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        trades: List
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics

        Args:
            equity_curve: Equity curve
            returns: Daily returns
            trades: List of trades

        Returns:
            PerformanceMetrics object
        """
        # Calculate each category
        returns_metrics = self.calculate_returns_metrics(equity_curve, returns)
        risk_metrics = self.calculate_risk_metrics(equity_curve, returns)
        risk_adjusted = self.calculate_risk_adjusted_metrics(
            returns,
            risk_metrics['volatility'],
            risk_metrics['downside_volatility'],
            risk_metrics['max_drawdown']
        )
        trade_metrics = self.calculate_trade_metrics(trades)

        # Recovery factor
        total_pnl = sum(t.net_pnl for t in trades)
        recovery_factor = (total_pnl / abs(risk_metrics['max_drawdown'])
                          if risk_metrics['max_drawdown'] < 0 else 0)

        return PerformanceMetrics(
            total_return=returns_metrics['total_return'],
            annualized_return=returns_metrics['annualized_return'],
            cumulative_return=returns_metrics['cumulative_return'],
            volatility=risk_metrics['volatility'],
            downside_volatility=risk_metrics['downside_volatility'],
            max_drawdown=risk_metrics['max_drawdown'],
            max_drawdown_duration=risk_metrics['max_drawdown_duration'],
            avg_drawdown=risk_metrics['avg_drawdown'],
            sharpe_ratio=risk_adjusted['sharpe_ratio'],
            sortino_ratio=risk_adjusted['sortino_ratio'],
            calmar_ratio=risk_adjusted['calmar_ratio'],
            num_trades=trade_metrics['num_trades'],
            win_rate=trade_metrics['win_rate'],
            profit_factor=trade_metrics['profit_factor'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            avg_holding_period=trade_metrics['avg_holding_period'],
            max_consecutive_wins=trade_metrics['max_consecutive_wins'],
            max_consecutive_losses=trade_metrics['max_consecutive_losses'],
            expectancy=trade_metrics['expectancy'],
            recovery_factor=recovery_factor,
            ulcer_index=risk_metrics['ulcer_index']
        )

    def analyze_by_pair(self, trades: List) -> pd.DataFrame:
        """
        Analyze performance by pair

        Args:
            trades: List of trades

        Returns:
            DataFrame with per-pair metrics
        """
        pair_stats = {}

        for trade in trades:
            pair = trade.pair_name

            if pair not in pair_stats:
                pair_stats[pair] = {
                    'num_trades': 0,
                    'total_pnl': 0,
                    'wins': 0,
                    'losses': 0,
                    'avg_holding_days': []
                }

            pair_stats[pair]['num_trades'] += 1
            pair_stats[pair]['total_pnl'] += trade.net_pnl

            if trade.net_pnl > 0:
                pair_stats[pair]['wins'] += 1
            else:
                pair_stats[pair]['losses'] += 1

            pair_stats[pair]['avg_holding_days'].append(trade.holding_days)

        # Convert to DataFrame
        data = []
        for pair, stats in pair_stats.items():
            data.append({
                'Pair': pair,
                'Num_Trades': stats['num_trades'],
                'Total_PnL': stats['total_pnl'],
                'Win_Rate': stats['wins'] / stats['num_trades'] if stats['num_trades'] > 0 else 0,
                'Avg_Holding_Days': np.mean(stats['avg_holding_days'])
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Total_PnL', ascending=False)

        return df

    def print_performance_report(self, metrics: PerformanceMetrics):
        """
        Print a formatted performance report

        Args:
            metrics: PerformanceMetrics object
        """
        print("\n" + "="*60)
        print(" PERFORMANCE REPORT")
        print("="*60)

        print("\nRETURNS:")
        print(f"  Total Return:       {metrics.total_return*100:>10.2f}%")
        print(f"  Annualized Return:  {metrics.annualized_return*100:>10.2f}%")
        print(f"  Cumulative Return:  {metrics.cumulative_return*100:>10.2f}%")

        print("\nRISK METRICS:")
        print(f"  Volatility:         {metrics.volatility*100:>10.2f}%")
        print(f"  Downside Vol:       {metrics.downside_volatility*100:>10.2f}%")
        print(f"  Max Drawdown:       {metrics.max_drawdown*100:>10.2f}%")
        print(f"  Avg Drawdown:       {metrics.avg_drawdown*100:>10.2f}%")
        print(f"  Max DD Duration:    {metrics.max_drawdown_duration:>10} days")
        print(f"  Ulcer Index:        {metrics.ulcer_index:>10.4f}")

        print("\nRISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:      {metrics.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:       {metrics.calmar_ratio:>10.2f}")

        print("\nTRADE STATISTICS:")
        print(f"  Number of Trades:   {metrics.num_trades:>10}")
        print(f"  Win Rate:           {metrics.win_rate*100:>10.2f}%")
        print(f"  Profit Factor:      {metrics.profit_factor:>10.2f}")
        print(f"  Avg Win:            ${metrics.avg_win:>10.2f}")
        print(f"  Avg Loss:           ${metrics.avg_loss:>10.2f}")
        print(f"  Expectancy:         ${metrics.expectancy:>10.2f}")
        print(f"  Avg Holding Period: {metrics.avg_holding_period:>10.1f} days")
        print(f"  Max Consec. Wins:   {metrics.max_consecutive_wins:>10}")
        print(f"  Max Consec. Losses: {metrics.max_consecutive_losses:>10}")

        print("\nOTHER METRICS:")
        print(f"  Recovery Factor:    {metrics.recovery_factor:>10.2f}")

        print("\n" + "="*60)


if __name__ == "__main__":
    # This would typically be used with BacktestResults
    pass
