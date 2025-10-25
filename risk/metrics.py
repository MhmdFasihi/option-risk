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
    
    # Calculate Greeks for all option positions first
    from risk.metrics import calculate_portfolio_greeks
    calculate_portfolio_greeks(portfolio)
    
    # Group positions by underlying
    underlyings = {}
    for pos in portfolio.positions:
        if pos.symbol not in underlyings:
            underlyings[pos.symbol] = {
                'positions': [],
                'base_price': None
            }
        underlyings[pos.symbol]['positions'].append(pos)
    
    # Get base prices for each underlying
    for symbol, data in underlyings.items():
        stock_pos = [p for p in data['positions'] if p.position_type == 'stock']
        if stock_pos:
            data['base_price'] = stock_pos[0].current_price
        else:
            # Use strike price of first option as approximation
            option_pos = [p for p in data['positions'] if p.position_type == 'option']
            if option_pos:
                data['base_price'] = option_pos[0].strike
            else:
                data['base_price'] = 100  # Default fallback
    
    # Simulate portfolio values
    simulated_portfolio_values = np.zeros(num_simulations)
    
    for symbol, data in underlyings.items():
        S0 = data['base_price']
        
        # Simulate price paths using GBM
        # Use realistic parameters - could be estimated from historical data
        mu = 0.10 / 252  # 10% annual drift, daily
        sigma = 0.25 / np.sqrt(252)  # 25% annual volatility, daily
        
        dt = time_horizon
        Z = np.random.standard_normal(num_simulations)
        ST = S0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Calculate position values at simulated prices
        for pos in data['positions']:
            if pos.position_type == 'stock':
                # Stock value = price * quantity
                position_values = ST * pos.quantity
                simulated_portfolio_values += position_values
                
            elif pos.position_type == 'option':
                # Use delta approximation for speed
                # (Full repricing would be more accurate but slower)
                if pos.delta is not None:
                    price_change = ST - S0
                    # Delta approximation: change in option value ≈ delta * price change
                    option_value_change = pos.delta * price_change * pos.quantity * 100
                    # Add current option value plus change
                    current_option_value = pos.current_price * pos.quantity * 100
                    position_values = current_option_value + option_value_change
                    simulated_portfolio_values += position_values
                else:
                    # If no delta, just use current value (no change)
                    current_option_value = pos.current_price * pos.quantity * 100
                    simulated_portfolio_values += current_option_value
    
    # Calculate returns
    simulated_returns = (simulated_portfolio_values - current_value) / current_value
    
    # Calculate VaR and CVaR
    var_pct = calculate_var(simulated_returns, confidence_level, method='monte_carlo')
    cvar_pct = calculate_cvar(simulated_returns, confidence_level)
    
    # Convert to dollar amounts (negative = loss)
    var_dollar = var_pct * current_value
    cvar_dollar = cvar_pct * current_value
    
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
