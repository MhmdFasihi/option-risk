"""Risk metrics calculations: VaR, CVaR, and Greeks aggregation."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from data.sample_portfolio import SamplePortfolio, Position
from models.black_scholes import calculate_greeks, time_to_maturity


def calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """Calculate Value at Risk (VaR).
    
    Args:
        returns: Array of portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: 'historical' or 'monte_carlo'
        
    Returns:
        VaR value (negative indicates loss)
    """
    if method == 'historical':
        return np.percentile(returns, (1 - confidence_level) * 100)
    elif method == 'monte_carlo':
        # For MC, returns should already be simulated
        return np.percentile(returns, (1 - confidence_level) * 100)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_cvar(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    
    CVaR is the expected loss given that VaR has been exceeded.
    
    Args:
        returns: Array of portfolio returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR value (negative indicates loss)
    """
    var = calculate_var(returns, confidence_level, method='historical')
    # CVaR is the mean of all returns below VaR
    return returns[returns <= var].mean()


def historical_var_cvar(
    portfolio: SamplePortfolio,
    historical_data: Dict[str, pd.DataFrame],
    confidence_level: float = 0.95,
    window: int = 252
) -> Tuple[float, float]:
    """Calculate historical VaR and CVaR.
    
    Args:
        portfolio: Portfolio object
        historical_data: Dict mapping symbols to historical price DataFrames
        confidence_level: Confidence level (default 95%)
        window: Rolling window in days (default 252 = 1 year)
        
    Returns:
        Tuple of (VaR, CVaR) - both negative values representing potential losses
    """
    # Get all unique symbols
    symbols = set(p.symbol for p in portfolio.positions)
    
    # Build historical portfolio values
    # Find common date range across all symbols
    common_dates = None
    for symbol in symbols:
        if symbol in historical_data and len(historical_data[symbol]) >= window:
            hist = historical_data[symbol]['Close'].tail(window)
            if common_dates is None:
                common_dates = hist.index
            else:
                common_dates = common_dates.intersection(hist.index)
    
    if common_dates is None or len(common_dates) < 2:
        return 0.0, 0.0
    
    # Calculate portfolio value at each date
    portfolio_values = []
    
    for date in common_dates:
        daily_value = 0
        
        for pos in portfolio.positions:
            if pos.position_type == 'stock' and pos.symbol in historical_data:
                hist = historical_data[pos.symbol]
                if date in hist.index:
                    price = hist.loc[date, 'Close']
                    daily_value += price * pos.quantity
        
        portfolio_values.append(daily_value)
    
    if len(portfolio_values) < 2:
        return 0.0, 0.0
    
    # Calculate returns
    portfolio_series = pd.Series(portfolio_values, index=common_dates)
    returns = portfolio_series.pct_change().dropna()
    
    if len(returns) == 0:
        return 0.0, 0.0
    
    # Calculate VaR and CVaR (these are percentages)
    var_pct = calculate_var(returns.values, confidence_level, method='historical')
    cvar_pct = calculate_cvar(returns.values, confidence_level)
    
    # Convert to dollar amounts (negative = loss)
    current_value = portfolio.total_value
    var_dollar = var_pct * current_value
    cvar_dollar = cvar_pct * current_value
    
    return var_dollar, cvar_dollar


def monte_carlo_var_cvar(
    portfolio: SamplePortfolio,
    num_simulations: int = 10000,
    time_horizon: int = 1,
    confidence_level: float = 0.95
) -> Tuple[float, float, np.ndarray]:
    """Calculate Monte Carlo VaR and CVaR.
    
    Uses Geometric Brownian Motion (GBM) for simulation.
    
    Args:
        portfolio: Portfolio object
        num_simulations: Number of Monte Carlo simulations
        time_horizon: Time horizon in days
        confidence_level: Confidence level (default 95%)
        
    Returns:
        Tuple of (VaR, CVaR, simulated_returns)
    """
    current_value = portfolio.total_value
    simulated_values = []
    
    # Group positions by underlying
    underlyings = {}
    for pos in portfolio.positions:
        if pos.symbol not in underlyings:
            underlyings[pos.symbol] = []
        underlyings[pos.symbol].append(pos)
    
    # Simulate for each underlying
    for symbol, positions in underlyings.items():
        # Get stock position for current price
        stock_pos = [p for p in positions if p.position_type == 'stock']
        if stock_pos:
            S0 = stock_pos[0].current_price
        else:
            # Get from option position's underlying price
            # For simplicity, use first option's implied underlying
            S0 = positions[0].current_price  # This should be fetched properly
            
        # Simulate price paths
        # Assume 20% volatility and 5% drift for simplicity
        # In production, estimate from historical data
        mu = 0.05 / 252  # Daily drift
        sigma = 0.20 / np.sqrt(252)  # Daily volatility
        
        dt = time_horizon
        Z = np.random.standard_normal(num_simulations)
        ST = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Calculate position values at simulated prices
        for pos in positions:
            if pos.position_type == 'stock':
                simulated_values.append(ST * pos.quantity)
            # For options, would need to reprice with new underlying
            # Simplified here - just use delta approximation
            elif pos.position_type == 'option' and pos.delta is not None:
                price_change = ST - S0
                option_value_change = pos.delta * price_change * pos.quantity * 100
                simulated_values.append(option_value_change)
    
    # Sum all simulated values
    if not simulated_values:
        return 0.0, 0.0, np.array([])
    
    total_simulated = np.sum(simulated_values, axis=0)
    simulated_returns = (total_simulated - current_value) / current_value
    
    var = calculate_var(simulated_returns, confidence_level, method='monte_carlo')
    cvar = calculate_cvar(simulated_returns, confidence_level)
    
    # Convert to dollar amounts
    var_dollar = var * current_value
    cvar_dollar = cvar * current_value
    
    return var_dollar, cvar_dollar, simulated_returns


def calculate_portfolio_greeks(
    portfolio: SamplePortfolio,
    risk_free_rate: float = 0.05
) -> Dict[str, float]:
    """Calculate portfolio-level Greeks by aggregating position Greeks.
    
    Args:
        portfolio: Portfolio object
        risk_free_rate: Risk-free rate (default 5%)
        
    Returns:
        Dictionary with portfolio Greeks
    """
    total_greeks = {
        'delta': 0.0,
        'gamma': 0.0,
        'vega': 0.0,
        'theta': 0.0,
        'rho': 0.0
    }
    
    for pos in portfolio.positions:
        if pos.position_type == 'stock':
            # Stock delta = 1 per share
            total_greeks['delta'] += pos.quantity
            # Other Greeks are 0 for stock
            
        elif pos.position_type == 'option':
            # Calculate Greeks if not already set
            if pos.delta is None:
                T = time_to_maturity(pos.expiry)
                S = pos.current_price  # Should get underlying price
                
                # Get underlying price from portfolio
                underlying_pos = [p for p in portfolio.positions 
                                 if p.symbol == pos.symbol and p.position_type == 'stock']
                if underlying_pos:
                    S = underlying_pos[0].current_price
                else:
                    # Estimate from strike
                    S = pos.strike
                
                greeks = calculate_greeks(
                    S=S,
                    K=pos.strike,
                    T=T,
                    r=risk_free_rate,
                    sigma=pos.implied_volatility or 0.3,
                    option_type=pos.option_type
                )
                
                # Update position Greeks
                pos.delta = greeks['delta']
                pos.gamma = greeks['gamma']
                pos.vega = greeks['vega']
                pos.theta = greeks['theta']
                pos.rho = greeks['rho']
            
            # Aggregate (multiply by quantity and contract multiplier)
            multiplier = pos.quantity * 100  # Options are per 100 shares
            total_greeks['delta'] += pos.delta * multiplier
            total_greeks['gamma'] += pos.gamma * multiplier
            total_greeks['vega'] += pos.vega * multiplier
            total_greeks['theta'] += pos.theta * multiplier
            total_greeks['rho'] += pos.rho * multiplier
    
    return total_greeks


def calculate_gamma_exposure(
    portfolio: SamplePortfolio,
    risk_free_rate: float = 0.05
) -> float:
    """Calculate dollar gamma exposure.
    
    Dollar gamma = Gamma × (Underlying Price)² × 0.01²
    
    Args:
        portfolio: Portfolio object
        risk_free_rate: Risk-free rate
        
    Returns:
        Dollar gamma exposure
    """
    greeks = calculate_portfolio_greeks(portfolio, risk_free_rate)
    
    # Calculate weighted average underlying price
    total_value = 0
    total_gamma_exposure = 0
    
    for pos in portfolio.positions:
        if pos.position_type == 'option' and pos.gamma:
            # Get underlying price
            underlying_pos = [p for p in portfolio.positions 
                            if p.symbol == pos.symbol and p.position_type == 'stock']
            if underlying_pos:
                S = underlying_pos[0].current_price
            else:
                S = pos.strike
            
            # Dollar gamma for this position
            dollar_gamma = pos.gamma * (S ** 2) * 0.0001 * pos.quantity * 100
            total_gamma_exposure += dollar_gamma
    
    return total_gamma_exposure
