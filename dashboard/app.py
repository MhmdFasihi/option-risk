"""Main Streamlit dashboard for options portfolio risk management."""

import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data import load_cached_sample_portfolio
from risk import PortfolioRisk
from surfaces import (
    create_sample_volatility_surface, 
    create_sample_greek_surface,
    create_returns_surface
)
from utils import format_currency, format_percentage, format_greek
import config


st.set_page_config(
    page_title="Options Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_portfolio():
    """Load the portfolio (cached)."""
    if config.USE_SAMPLE_DATA:
        # Use cached portfolio to avoid yfinance rate limits
        return load_cached_sample_portfolio()
    else:
        # TODO: Implement API integration
        st.warning("API integration not yet implemented. Using sample data.")
        return load_cached_sample_portfolio()


def main():
    """Main dashboard application."""
    
    st.title("📊 Options Portfolio Risk Management")
    st.markdown("---")
    
    # Load portfolio
    with st.spinner("Loading portfolio..."):
        portfolio = load_portfolio()
        risk_analyzer = PortfolioRisk(portfolio, risk_free_rate=config.RISK_FREE_RATE)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    data_source = st.sidebar.radio(
        "Data Source",
        ["Sample Portfolio", "API (Coming Soon)"],
        disabled=True
    )
    
    confidence_level = st.sidebar.select_slider(
        "VaR Confidence Level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{x*100:.0f}%"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Portfolio Value:** {format_currency(portfolio.total_value)}")
    st.sidebar.info(f"**Positions:** {len(portfolio.positions)}")
    
    # Main content - tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Portfolio Overview",
        "⚠️ Risk Metrics",
        "🏔️ 3D Surfaces",
        "💰 Returns Surface",
        "📊 Position Details"
    ])
    
    # Tab 1: Portfolio Overview
    with tab1:
        st.header("Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", format_currency(portfolio.total_value))
        
        with col2:
            greeks = risk_analyzer.get_greeks()
            st.metric("Portfolio Delta", f"{greeks['delta']:.2f}")
        
        with col3:
            st.metric("Portfolio Gamma", f"{greeks['gamma']:.4f}")
        
        with col4:
            gamma_exp = risk_analyzer.get_gamma_exposure()
            st.metric("Gamma Exposure", format_currency(gamma_exp))
        
        st.markdown("---")
        
        # Position breakdown
        st.subheader("Position Breakdown")
        df = portfolio.to_dataframe()
        
        # Format for display
        display_df = df.copy()
        display_df['market_value'] = display_df['market_value'].apply(format_currency)
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        
        if 'iv' in display_df.columns:
            display_df['iv'] = display_df['iv'].apply(
                lambda x: format_percentage(x) if pd.notna(x) else 'N/A'
            )
        
        st.dataframe(display_df, width='stretch')
    
    # Tab 2: Risk Metrics
    with tab2:
        st.header("Risk Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Historical VaR & CVaR")
            
            with st.spinner("Calculating historical metrics..."):
                try:
                    hist_var, hist_cvar = risk_analyzer.calculate_var_cvar(
                        method='historical',
                        confidence_level=confidence_level
                    )
                    
                    # VaR should be displayed as positive loss amount
                    var_display = abs(hist_var)
                    cvar_display = abs(hist_cvar)
                    
                    st.metric(
                        f"Historical VaR ({confidence_level*100:.0f}%)",
                        format_currency(var_display),
                        delta=None,
                        help=f"Maximum expected loss at {confidence_level*100:.0f}% confidence level"
                    )
                    
                    st.metric(
                        f"Historical CVaR ({confidence_level*100:.0f}%)",
                        format_currency(cvar_display),
                        delta=None,
                        help="Expected loss given VaR is exceeded"
                    )
                    
                    # Show percentage of portfolio
                    var_pct = (var_display / portfolio.total_value) * 100
                    st.caption(f"VaR as % of portfolio: {var_pct:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error calculating historical metrics: {str(e)}")
        
        with col2:
            st.subheader("🎲 Monte Carlo VaR & CVaR")
            
            with st.spinner("Running Monte Carlo simulation..."):
                try:
                    mc_var, mc_cvar = risk_analyzer.calculate_var_cvar(
                        method='monte_carlo',
                        confidence_level=confidence_level,
                        num_simulations=config.MC_SIMULATIONS
                    )
                    
                    # Display as positive loss amounts
                    mc_var_display = abs(mc_var)
                    mc_cvar_display = abs(mc_cvar)
                    
                    st.metric(
                        f"Monte Carlo VaR ({confidence_level*100:.0f}%)",
                        format_currency(mc_var_display),
                        delta=None,
                        help=f"VaR from {config.MC_SIMULATIONS:,} simulations"
                    )
                    
                    st.metric(
                        f"Monte Carlo CVaR ({confidence_level*100:.0f}%)",
                        format_currency(mc_cvar_display),
                        delta=None,
                        help="Expected shortfall from simulations"
                    )
                    
                    # Show percentage of portfolio
                    mc_var_pct = (mc_var_display / portfolio.total_value) * 100
                    st.caption(f"VaR as % of portfolio: {mc_var_pct:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error calculating Monte Carlo metrics: {str(e)}")
        
        st.markdown("---")
        
        # Explanation of VaR metrics
        with st.expander("ℹ️ About VaR and CVaR Calculations"):
            st.markdown("""
            **Value at Risk (VaR)**: Maximum expected loss at a given confidence level over 1 day.
            - 95% VaR of $5,000 means: "There's a 95% chance losses won't exceed $5,000 tomorrow"
            - Or equivalently: "There's a 5% chance of losing more than $5,000"
            
            **Conditional VaR (CVaR)**: Expected loss when VaR is exceeded (worst 5% of scenarios).
            - Always higher than VaR
            - Shows average loss in the "tail" scenarios
            
            **Historical Method**: 
            - Uses actual past price movements (252 trading days)
            - Assumes future will behave like the past
            
            **Monte Carlo Method**:
            - Simulates 10,000 possible future scenarios using statistical models
            - Assumes prices follow Geometric Brownian Motion (GBM)
            - Options valued using Delta approximation for speed
            
            **Why do they differ?**
            - Historical reflects actual market behavior
            - Monte Carlo uses theoretical assumptions
            - Both are useful for different perspectives on risk
            """)
        
        st.markdown("---")
        
        # Greeks breakdown
        st.subheader("📊 Portfolio Greeks")
        
        greeks = risk_analyzer.get_greeks()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Delta", format_greek('delta', greeks['delta']))
        
        with col2:
            st.metric("Gamma", format_greek('gamma', greeks['gamma']))
        
        with col3:
            st.metric("Vega", format_greek('vega', greeks['vega']))
        
        with col4:
            st.metric("Theta", format_greek('theta', greeks['theta']))
        
        with col5:
            st.metric("Rho", format_greek('rho', greeks['rho']))
    
    # Tab 3: 3D Surfaces
    with tab3:
        st.header("3D Surfaces")
        
        surface_type = st.radio(
            "Select Surface Type",
            ["Volatility Surface", "Greek Surface"],
            horizontal=True
        )
        
        if surface_type == "Volatility Surface":
            st.subheader("📈 Implied Volatility Surface")
            
            with st.spinner("Generating volatility surface..."):
                try:
                    # Use first stock position as underlying
                    stock_positions = [p for p in portfolio.positions if p.position_type == 'stock']
                    if stock_positions:
                        underlying_price = stock_positions[0].current_price
                    else:
                        underlying_price = 100
                    
                    fig = create_sample_volatility_surface(underlying_price=underlying_price)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 The volatility surface shows how implied volatility varies with strike price and time to expiration.")
                except Exception as e:
                    st.error(f"Error generating volatility surface: {str(e)}")
        
        else:  # Greek Surface
            st.subheader("📊 Greek Surface")
            
            greek_choice = st.selectbox(
                "Select Greek",
                ["Delta", "Gamma", "Vega", "Theta", "Rho"]
            )
            
            with st.spinner(f"Generating {greek_choice} surface..."):
                try:
                    # Use first stock position as underlying
                    stock_positions = [p for p in portfolio.positions if p.position_type == 'stock']
                    if stock_positions:
                        underlying_price = stock_positions[0].current_price
                    else:
                        underlying_price = 100
                    
                    fig = create_sample_greek_surface(
                        greek_name=greek_choice.lower(),
                        underlying_price=underlying_price
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Greek explanations
                    explanations = {
                        'Delta': "Delta measures the rate of change of option price with respect to underlying price.",
                        'Gamma': "Gamma measures the rate of change of delta. High gamma means delta changes quickly.",
                        'Vega': "Vega measures sensitivity to volatility changes.",
                        'Theta': "Theta measures time decay. Usually negative for long options.",
                        'Rho': "Rho measures sensitivity to interest rate changes."
                    }
                    st.info(f"💡 {explanations[greek_choice]}")
                except Exception as e:
                    st.error(f"Error generating Greek surface: {str(e)}")
    
    # Tab 4: Returns Surface
    with tab4:
        st.header("💰 Option Returns Surface")
        st.markdown("""
        This surface shows how option returns vary with:
        - **X-axis**: Underlying asset price
        - **Y-axis**: Implied volatility
        - **Z-axis**: Return percentage
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get option positions
            option_positions = [p for p in portfolio.positions if p.position_type == 'option']
            
            if option_positions:
                option_choices = [
                    f"{p.symbol} {p.strike} {p.option_type.upper()} ({p.quantity} contracts)"
                    for p in option_positions
                ]
                selected_idx = st.selectbox(
                    "Select Option Position",
                    range(len(option_choices)),
                    format_func=lambda i: option_choices[i]
                )
                selected_position = option_positions[selected_idx]
            else:
                st.warning("No option positions in portfolio")
                selected_position = None
        
        with col2:
            # Days to expiry slider
            max_days = 180
            days_to_expiry = st.slider(
                "Days to Expiration",
                min_value=1,
                max_value=max_days,
                value=30,
                help="Adjust to see how returns change with time"
            )
        
        if selected_position:
            with st.spinner("Generating returns surface..."):
                try:
                    # Get underlying price
                    stock_positions = [p for p in portfolio.positions 
                                     if p.symbol == selected_position.symbol and p.position_type == 'stock']
                    if stock_positions:
                        base_price = stock_positions[0].current_price
                    else:
                        base_price = selected_position.strike
                    
                    # Generate price range (±30% from current)
                    price_range = np.linspace(base_price * 0.7, base_price * 1.3, 25)
                    
                    # Generate volatility range (15% to 50%)
                    vol_range = np.linspace(0.15, 0.50, 20)
                    
                    fig = create_returns_surface(
                        underlying_prices=price_range,
                        implied_vols=vol_range,
                        strike=selected_position.strike,
                        days_to_expiry=days_to_expiry,
                        risk_free_rate=config.RISK_FREE_RATE,
                        option_type=selected_position.option_type,
                        initial_option_price=selected_position.current_price
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Strike Price", f"${selected_position.strike:.2f}")
                    with col2:
                        st.metric("Current Price", f"${selected_position.current_price:.2f}")
                    with col3:
                        st.metric("Contracts", selected_position.quantity)
                    
                    st.info("""
                    💡 **How to read this chart:**
                    - Green areas show profitable scenarios
                    - Red areas show losses
                    - The surface shows all possible returns based on price and volatility changes
                    - Use the slider to see how time decay affects returns
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating returns surface: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Tab 5: Position Details
    with tab5:
        st.header("Position Details")
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            position_type = st.multiselect(
                "Position Type",
                ["stock", "option"],
                default=["stock", "option"]
            )
        
        with col2:
            if "option" in position_type:
                option_type = st.multiselect(
                    "Option Type",
                    ["call", "put"],
                    default=["call", "put"]
                )
            else:
                option_type = []
        
        # Filter positions
        df = portfolio.to_dataframe()
        
        if position_type:
            df = df[df['type'].isin(position_type)]
        
        if option_type and 'option' in position_type:
            df = df[(df['type'] == 'stock') | (df['option_type'].isin(option_type))]
        
        # Display detailed table
        st.dataframe(df, width='stretch')
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_stocks = len(df[df['type'] == 'stock'])
            st.metric("Stock Positions", total_stocks)
        
        with col2:
            total_options = len(df[df['type'] == 'option'])
            st.metric("Option Positions", total_options)
        
        with col3:
            total_value = df['market_value'].sum()
            st.metric("Filtered Value", format_currency(total_value))


if __name__ == '__main__':
    main()
