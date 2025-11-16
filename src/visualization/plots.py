"""
Plotting utilities for pairs trading analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_spread_with_zscore(
    spread: pd.Series,
    zscore: pd.Series,
    pair_name: str,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    stop_threshold: float = 3.0,
    trades: Optional[List] = None
) -> go.Figure:
    """
    Plot spread and z-score with entry/exit bands

    Args:
        spread: Spread series
        zscore: Z-score series
        pair_name: Name of the pair
        entry_threshold: Entry z-score threshold
        exit_threshold: Exit z-score threshold
        stop_threshold: Stop loss z-score threshold
        trades: Optional list of trades to mark on chart

    Returns:
        Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{pair_name} - Spread', f'{pair_name} - Z-Score'),
        row_heights=[0.5, 0.5]
    )

    # Plot spread
    fig.add_trace(
        go.Scatter(x=spread.index, y=spread.values, name='Spread', line=dict(color='blue')),
        row=1, col=1
    )

    # Plot z-score
    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore.values, name='Z-Score', line=dict(color='purple')),
        row=2, col=1
    )

    # Add threshold lines to z-score plot
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red",
                 annotation_text=f"Entry Long ({entry_threshold})", row=2, col=1)
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red",
                 annotation_text=f"Entry Short ({-entry_threshold})", row=2, col=1)
    fig.add_hline(y=exit_threshold, line_dash="solid", line_color="green",
                 annotation_text=f"Exit ({exit_threshold})", row=2, col=1)
    fig.add_hline(y=stop_threshold, line_dash="dot", line_color="orange",
                 annotation_text=f"Stop Loss ({stop_threshold})", row=2, col=1)
    fig.add_hline(y=-stop_threshold, line_dash="dot", line_color="orange",
                 annotation_text=f"Stop Loss ({-stop_threshold})", row=2, col=1)

    # Add trade markers if provided
    if trades:
        entry_dates = [t.entry_date for t in trades]
        entry_spreads = [spread.loc[t.entry_date] if t.entry_date in spread.index else None
                        for t in trades]
        exit_dates = [t.exit_date for t in trades if t.exit_date]
        exit_spreads = [spread.loc[t.exit_date] if t.exit_date in spread.index else None
                       for t in trades if t.exit_date]

        # Entry markers
        fig.add_trace(
            go.Scatter(x=entry_dates, y=entry_spreads, mode='markers',
                      name='Entry', marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )

        # Exit markers
        fig.add_trace(
            go.Scatter(x=exit_dates, y=exit_spreads, mode='markers',
                      name='Exit', marker=dict(color='red', size=10, symbol='triangle-down')),
            row=1, col=1
        )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)

    return fig


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None
) -> go.Figure:
    """
    Plot equity curve with optional benchmark

    Args:
        equity_curve: Equity curve series
        benchmark: Optional benchmark series (e.g., SPY)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Plot strategy equity
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values,
                  name='Strategy', line=dict(color='blue', width=2))
    )

    # Plot benchmark if provided
    if benchmark is not None:
        # Normalize benchmark to start at same value as strategy
        benchmark_normalized = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
        fig.add_trace(
            go.Scatter(x=benchmark_normalized.index, y=benchmark_normalized.values,
                      name='Benchmark', line=dict(color='gray', width=1, dash='dash'))
        )

    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500
    )

    return fig


def plot_drawdown(equity_curve: pd.Series) -> go.Figure:
    """
    Plot drawdown chart

    Args:
        equity_curve: Equity curve series

    Returns:
        Plotly figure
    """
    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values,
                  fill='tozeroy', name='Drawdown',
                  line=dict(color='red'))
    )

    fig.update_layout(
        title='Drawdown Chart',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=400
    )

    return fig


def plot_returns_distribution(returns: pd.Series) -> go.Figure:
    """
    Plot distribution of returns

    Args:
        returns: Daily returns series

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(x=returns.values, nbinsx=50,
                    name='Returns', marker_color='blue')
    )

    fig.update_layout(
        title='Distribution of Daily Returns',
        xaxis_title='Return',
        yaxis_title='Frequency',
        height=400
    )

    return fig


def plot_trade_pnl_distribution(trades: List) -> go.Figure:
    """
    Plot distribution of trade P&L

    Args:
        trades: List of Trade objects

    Returns:
        Plotly figure
    """
    pnls = [t.net_pnl for t in trades]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(x=pnls, nbinsx=30,
                    name='Trade P&L',
                    marker=dict(
                        color=pnls,
                        colorscale='RdYlGn',
                        showscale=True
                    ))
    )

    fig.update_layout(
        title='Distribution of Trade P&L',
        xaxis_title='P&L ($)',
        yaxis_title='Frequency',
        height=400
    )

    return fig


def plot_pair_correlation_heatmap(price_data: pd.DataFrame, sector_mapping: dict) -> go.Figure:
    """
    Plot correlation heatmap of stocks by sector

    Args:
        price_data: DataFrame with stock prices
        sector_mapping: Dictionary mapping sectors to tickers

    Returns:
        Plotly figure
    """
    # Calculate correlation matrix
    corr_matrix = price_data.corr()

    # Create sector labels
    ticker_to_sector = {}
    for sector, tickers in sector_mapping.items():
        for ticker in tickers:
            ticker_to_sector[ticker] = sector

    # Sort by sector
    sorted_tickers = sorted(corr_matrix.columns,
                           key=lambda x: (ticker_to_sector.get(x, 'Unknown'), x))

    corr_matrix = corr_matrix.loc[sorted_tickers, sorted_tickers]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8}
    ))

    fig.update_layout(
        title='Stock Correlation Heatmap',
        height=800,
        width=800
    )

    return fig


def plot_monthly_returns(returns: pd.Series) -> go.Figure:
    """
    Plot monthly returns heatmap

    Args:
        returns: Daily returns series

    Returns:
        Plotly figure
    """
    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table for heatmap
    monthly_returns_df = monthly_returns.to_frame('returns')
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month

    pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='returns')

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values * 100,
        x=month_names,
        y=pivot_table.index,
        colorscale='RdYlGn',
        zmid=0,
        text=pivot_table.values * 100,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title='Monthly Returns Heatmap (%)',
        xaxis_title='Month',
        yaxis_title='Year',
        height=400
    )

    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int = 63) -> go.Figure:
    """
    Plot rolling Sharpe ratio

    Args:
        returns: Daily returns series
        window: Rolling window in days (default 63 = ~3 months)

    Returns:
        Plotly figure
    """
    rolling_sharpe = (returns.rolling(window).mean() /
                     returns.rolling(window).std() * np.sqrt(252))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                  name=f'{window}-day Rolling Sharpe',
                  line=dict(color='green'))
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=1, line_dash="dot", line_color="blue", annotation_text="Sharpe = 1")

    fig.update_layout(
        title=f'Rolling Sharpe Ratio ({window} days)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        height=400
    )

    return fig


def plot_pair_summary_table(pairs_df: pd.DataFrame) -> go.Figure:
    """
    Create a formatted table of top pairs

    Args:
        pairs_df: DataFrame with pair summary

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(pairs_df.columns),
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[pairs_df[col] for col in pairs_df.columns],
                  fill_color='lavender',
                  align='left'))
    ])

    fig.update_layout(
        title='Top Cointegrated Pairs',
        height=400
    )

    return fig


if __name__ == "__main__":
    # Example usage would go here
    pass
