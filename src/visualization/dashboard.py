"""
Streamlit dashboard for pairs trading analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader
from src.analysis.cointegration import CointegrationAnalyzer
from src.analysis.spread import MultiPairSpreadCalculator
from src.backtesting.signals import SignalGenerator
from src.backtesting.engine import BacktestEngine
from src.backtesting.performance import PerformanceAnalyzer
from src.visualization.plots import (
    plot_spread_with_zscore,
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_trade_pnl_distribution,
    plot_monthly_returns,
    plot_rolling_sharpe
)

# Page config
st.set_page_config(
    page_title="Pairs Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📈 Statistical Arbitrage Pairs Trading Dashboard")
st.markdown("---")


@st.cache_data
def load_data(sectors, years, min_volume):
    """Load and cache price data"""
    loader = DataLoader()
    price_data, sector_mapping = loader.get_sector_data(
        sectors=sectors,
        years=years,
        min_dollar_volume=min_volume
    )
    return price_data, sector_mapping


@st.cache_data
def find_cointegrated_pairs(price_data, sector_mapping, significance, min_corr, max_half_life, same_sector, max_pairs):
    """Find and cache cointegrated pairs"""
    analyzer = CointegrationAnalyzer(
        significance_level=significance,
        min_correlation=min_corr,
        max_half_life=max_half_life,
        same_sector_only=same_sector
    )
    pairs = analyzer.find_pairs(price_data, sector_mapping, max_pairs=max_pairs)
    return pairs, analyzer


@st.cache_data
def calculate_spreads_and_zscores(price_data, pairs, lookback):
    """Calculate spreads and z-scores"""
    spread_calc = MultiPairSpreadCalculator(lookback_window=lookback)
    spreads = spread_calc.calculate_all_spreads(price_data, pairs)
    zscores = spread_calc.calculate_all_zscores(spreads)
    return spreads, zscores, spread_calc


@st.cache_data
def run_backtest(price_data, pairs, initial_capital, commission, slippage, short_rate, entry_threshold, exit_threshold, stop_threshold):
    """Run backtest"""
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_per_trade=commission,
        slippage_pct=slippage,
        short_borrow_rate=short_rate
    )

    signal_gen = SignalGenerator(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_threshold=stop_threshold
    )

    results = engine.run_walk_forward_backtest(
        price_data,
        pairs,
        train_period=252,
        test_period=63,
        signal_generator=signal_gen
    )

    return results


# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Data parameters
st.sidebar.subheader("Data Parameters")
available_sectors = ['Financials', 'Technology', 'Energy', 'Consumer_Discretionary',
                    'Healthcare', 'Industrials', 'Consumer_Staples', 'Utilities',
                    'Real_Estate', 'Materials', 'Communication_Services']

selected_sectors = st.sidebar.multiselect(
    "Select Sectors",
    available_sectors,
    default=['Financials', 'Technology']
)

years = st.sidebar.slider("Years of Historical Data", 1, 5, 2)
min_volume = st.sidebar.number_input("Min Daily Volume ($M)", 1, 100, 10) * 1_000_000

# Cointegration parameters
st.sidebar.subheader("Cointegration Parameters")
significance = st.sidebar.slider("Significance Level", 0.01, 0.10, 0.05, 0.01)
min_correlation = st.sidebar.slider("Min Correlation", 0.0, 1.0, 0.7, 0.05)
max_half_life = st.sidebar.slider("Max Half-Life (days)", 5, 60, 30, 5)
same_sector_only = st.sidebar.checkbox("Same Sector Only", value=True)
max_pairs = st.sidebar.slider("Max Pairs to Display", 5, 50, 20, 5)

# Trading parameters
st.sidebar.subheader("Trading Parameters")
entry_threshold = st.sidebar.slider("Entry Z-Score Threshold", 1.0, 3.0, 2.0, 0.1)
exit_threshold = st.sidebar.slider("Exit Z-Score Threshold", -1.0, 1.0, 0.0, 0.1)
stop_threshold = st.sidebar.slider("Stop Loss Z-Score", 2.0, 4.0, 3.0, 0.1)
lookback_window = st.sidebar.slider("Lookback Window (days)", 20, 120, 60, 10)

# Backtest parameters
st.sidebar.subheader("Backtest Parameters")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)
commission = st.sidebar.number_input("Commission per Trade ($)", 0.0, 10.0, 1.0, 0.5)
slippage = st.sidebar.number_input("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100
short_rate = st.sidebar.number_input("Short Borrow Rate (%)", 0.0, 10.0, 2.0, 0.5) / 100

# Load data button
if st.sidebar.button("🔄 Load Data & Run Analysis", type="primary"):
    st.session_state.run_analysis = True

# Main content
if 'run_analysis' not in st.session_state:
    st.info("👈 Configure parameters in the sidebar and click 'Load Data & Run Analysis' to begin")
    st.stop()

# Load data
with st.spinner("Loading data..."):
    try:
        price_data, sector_mapping = load_data(selected_sectors, years, min_volume)
        st.success(f"✅ Loaded {len(price_data.columns)} stocks from {len(selected_sectors)} sectors")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Find pairs
with st.spinner("Finding cointegrated pairs..."):
    try:
        pairs, analyzer = find_cointegrated_pairs(
            price_data, sector_mapping, significance, min_correlation,
            max_half_life, same_sector_only, max_pairs
        )
        st.success(f"✅ Found {len(pairs)} cointegrated pairs")
    except Exception as e:
        st.error(f"Error finding pairs: {e}")
        st.stop()

if not pairs:
    st.warning("No cointegrated pairs found. Try adjusting the parameters.")
    st.stop()

# Calculate spreads
with st.spinner("Calculating spreads and z-scores..."):
    try:
        spreads, zscores, spread_calc = calculate_spreads_and_zscores(
            price_data, pairs, lookback_window
        )
    except Exception as e:
        st.error(f"Error calculating spreads: {e}")
        st.stop()

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pair Analysis",
    "📈 Backtest Results",
    "💹 Current Opportunities",
    "📉 Individual Pairs",
    "ℹ️ About"
])

# Tab 1: Pair Analysis
with tab1:
    st.header("Cointegrated Pairs Summary")

    # Create summary DataFrame
    summary_df = analyzer.get_pair_summary(pairs)

    # Format the DataFrame
    summary_df['P_Value'] = summary_df['P_Value'].apply(lambda x: f"{x:.4f}")
    summary_df['Hedge_Ratio'] = summary_df['Hedge_Ratio'].apply(lambda x: f"{x:.4f}")
    summary_df['Half_Life'] = summary_df['Half_Life'].apply(lambda x: f"{x:.1f}")
    summary_df['Correlation'] = summary_df['Correlation'].apply(lambda x: f"{x:.3f}")

    st.dataframe(summary_df, use_container_width=True, height=400)

    # Download button
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Pairs Data",
        data=csv,
        file_name="cointegrated_pairs.csv",
        mime="text/csv"
    )

    # Current z-scores
    st.subheader("Current Z-Scores")
    current_zscores = spread_calc.get_current_zscores(zscores)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Pairs Above Entry Threshold", len(current_zscores[abs(current_zscores) > entry_threshold]))

    with col2:
        st.metric("Pairs Near Mean", len(current_zscores[abs(current_zscores) < exit_threshold]))

    # Plot current z-scores
    fig_zscores = pd.DataFrame({
        'Pair': current_zscores.index,
        'Z-Score': current_zscores.values
    })
    st.bar_chart(fig_zscores.set_index('Pair'))

# Tab 2: Backtest Results
with tab2:
    st.header("Backtesting Results")

    with st.spinner("Running backtest..."):
        try:
            results = run_backtest(
                price_data, pairs, initial_capital, commission,
                slippage, short_rate, entry_threshold,
                exit_threshold, stop_threshold
            )
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            st.stop()

    # Performance metrics
    analyzer_perf = PerformanceAnalyzer()
    metrics = analyzer_perf.calculate_all_metrics(
        results.equity_curve,
        results.daily_returns,
        results.trades
    )

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{metrics.total_return*100:.2f}%")
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")

    with col2:
        st.metric("Annualized Return", f"{metrics.annualized_return*100:.2f}%")
        st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")

    with col3:
        st.metric("Max Drawdown", f"{metrics.max_drawdown*100:.2f}%")
        st.metric("Calmar Ratio", f"{metrics.calmar_ratio:.2f}")

    with col4:
        st.metric("Win Rate", f"{metrics.win_rate*100:.2f}%")
        st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")

    # Equity curve
    st.subheader("Equity Curve")
    fig_equity = plot_equity_curve(results.equity_curve)
    st.plotly_chart(fig_equity, use_container_width=True)

    # Drawdown
    st.subheader("Drawdown")
    fig_dd = plot_drawdown(results.equity_curve)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Returns distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Returns Distribution")
        fig_returns = plot_returns_distribution(results.daily_returns)
        st.plotly_chart(fig_returns, use_container_width=True)

    with col2:
        st.subheader("Trade P&L Distribution")
        fig_pnl = plot_trade_pnl_distribution(results.trades)
        st.plotly_chart(fig_pnl, use_container_width=True)

    # Monthly returns
    st.subheader("Monthly Returns")
    fig_monthly = plot_monthly_returns(results.daily_returns)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Rolling Sharpe
    st.subheader("Rolling Sharpe Ratio")
    fig_sharpe = plot_rolling_sharpe(results.daily_returns)
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # Trade details
    st.subheader("Recent Trades")
    if results.trades:
        trades_df = pd.DataFrame([
            {
                'Entry Date': t.entry_date.strftime('%Y-%m-%d'),
                'Exit Date': t.exit_date.strftime('%Y-%m-%d') if t.exit_date else 'Open',
                'Pair': t.pair_name,
                'Direction': 'LONG' if t.direction == 1 else 'SHORT',
                'Entry Z': f"{t.entry_zscore:.2f}",
                'Exit Z': f"{t.exit_zscore:.2f}",
                'Holding Days': t.holding_days,
                'Net P&L': f"${t.net_pnl:.2f}",
                'Exit Reason': t.exit_reason
            }
            for t in results.trades[-20:]  # Last 20 trades
        ])
        st.dataframe(trades_df, use_container_width=True, height=400)

# Tab 3: Current Opportunities
with tab3:
    st.header("Current Trading Opportunities")

    opportunities = spread_calc.identify_trading_opportunities(
        zscores,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold
    )

    if not opportunities.empty:
        # Format the DataFrame
        opportunities['zscore'] = opportunities['zscore'].apply(lambda x: f"{x:.2f}")
        opportunities['strength'] = opportunities['strength'].apply(lambda x: f"{x:.2f}")

        st.dataframe(opportunities, use_container_width=True)

        st.info(f"Found {len(opportunities)} trading opportunities")
    else:
        st.info("No trading opportunities at current thresholds")

    # Show pairs closest to entry
    st.subheader("Pairs Approaching Entry Threshold")
    current_zscores = spread_calc.get_current_zscores(zscores)
    close_to_entry = current_zscores[
        (abs(current_zscores) > entry_threshold * 0.8) &
        (abs(current_zscores) < entry_threshold)
    ].sort_values(key=abs, ascending=False)

    if len(close_to_entry) > 0:
        st.write(close_to_entry)
    else:
        st.info("No pairs close to entry threshold")

# Tab 4: Individual Pairs
with tab4:
    st.header("Individual Pair Analysis")

    # Select pair
    pair_names = [f"{p.ticker_a}_{p.ticker_b}" for p in pairs]
    selected_pair = st.selectbox("Select Pair", pair_names)

    if selected_pair:
        # Get pair data
        pair_obj = next((p for p in pairs if f"{p.ticker_a}_{p.ticker_b}" == selected_pair), None)

        if pair_obj:
            # Display pair info
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Hedge Ratio", f"{pair_obj.hedge_ratio:.4f}")
                st.metric("ADF P-Value", f"{pair_obj.adf_pvalue:.4f}")

            with col2:
                st.metric("Half-Life", f"{pair_obj.half_life:.1f} days")
                st.metric("Correlation", f"{pair_obj.correlation:.3f}")

            with col3:
                current_z = zscores[selected_pair].iloc[-1] if selected_pair in zscores else np.nan
                st.metric("Current Z-Score", f"{current_z:.2f}" if not np.isnan(current_z) else "N/A")

            # Get trades for this pair
            pair_trades = [t for t in results.trades if t.pair_name == selected_pair] if 'results' in locals() else []

            # Plot spread and z-score
            st.subheader("Spread and Z-Score Analysis")
            fig_spread = plot_spread_with_zscore(
                spreads[selected_pair],
                zscores[selected_pair],
                selected_pair,
                entry_threshold,
                exit_threshold,
                stop_threshold,
                pair_trades
            )
            st.plotly_chart(fig_spread, use_container_width=True)

            # Price chart
            st.subheader("Price Chart")
            price_chart_data = pd.DataFrame({
                pair_obj.ticker_a: price_data[pair_obj.ticker_a],
                pair_obj.ticker_b: price_data[pair_obj.ticker_b]
            })
            st.line_chart(price_chart_data)

# Tab 5: About
with tab5:
    st.header("About This Dashboard")

    st.markdown("""
    ### Statistical Arbitrage Pairs Trading System

    This dashboard implements a comprehensive pairs trading strategy based on statistical arbitrage principles.

    #### Strategy Overview

    **Cointegration Testing**
    - Uses the Engle-Granger two-step method to identify mean-reverting pairs
    - Tests stationarity of spreads using the Augmented Dickey-Fuller (ADF) test
    - Calculates hedge ratios via OLS regression
    - Estimates half-life of mean reversion

    **Trading Signals**
    - **Entry**: Z-score exceeds ±2.0 (spread diverges from mean)
    - **Exit**: Z-score crosses 0 (mean reversion)
    - **Stop Loss**: Z-score reaches ±3.0 (diverging further)
    - **Time Exit**: After 20 trading days if no reversion

    **Position Sizing**
    - Dollar-neutral positions (equal long and short values)
    - Position size: 20% of capital per pair
    - Maximum 5 concurrent pairs

    **Transaction Costs**
    - Commission: $1 per trade leg
    - Slippage: 0.05% per side (0.1% round-trip)
    - Short borrow cost: 2% annual rate (pro-rated)

    #### How to Use

    1. **Configure Parameters**: Use the sidebar to set data selection, cointegration criteria, and trading rules
    2. **Load Data**: Click the "Load Data & Run Analysis" button
    3. **Review Pairs**: Check the "Pair Analysis" tab to see cointegrated pairs
    4. **Analyze Performance**: View backtest results in the "Backtest Results" tab
    5. **Find Opportunities**: Check current trading signals in "Current Opportunities"
    6. **Deep Dive**: Analyze individual pairs in the "Individual Pairs" tab

    #### Key Metrics

    - **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
    - **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
    - **Calmar Ratio**: Return divided by max drawdown
    - **Win Rate**: Percentage of profitable trades
    - **Profit Factor**: Gross profit / Gross loss

    #### Disclaimer

    This is for educational and research purposes only. Past performance does not guarantee future results.
    Always conduct thorough due diligence and risk management before live trading.
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit 🎈")
