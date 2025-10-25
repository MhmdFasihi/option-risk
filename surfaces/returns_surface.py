"""Option returns surface generation."""

import numpy as np
import plotly.graph_objects as go
from typing import List
from models.black_scholes import BlackScholes


def create_returns_surface(
    underlying_prices: np.ndarray,
    implied_vols: np.ndarray,
    strike: float,
    days_to_expiry: int,
    risk_free_rate: float = 0.05,
    option_type: str = 'call',
    initial_option_price: float = None
) -> go.Figure:
    """Create 3D returns surface for options.
    
    Shows how option returns vary with underlying price and implied volatility.
    
    Args:
        underlying_prices: Array of underlying prices (X-axis)
        implied_vols: Array of implied volatilities (Y-axis)
        strike: Strike price
        days_to_expiry: Days to expiration
        risk_free_rate: Risk-free rate
        option_type: 'call' or 'put'
        initial_option_price: Initial option price (if None, uses ATM price)
        
    Returns:
        Plotly Figure object
    """
    bs = BlackScholes()
    T = days_to_expiry / 365.25
    
    # Create meshgrid
    X, Y = np.meshgrid(underlying_prices, implied_vols)
    Z = np.zeros_like(X)
    
    # Calculate initial option price if not provided
    if initial_option_price is None:
        mid_price = underlying_prices[len(underlying_prices) // 2]
        mid_vol = implied_vols[len(implied_vols) // 2]
        
        if option_type.lower() == 'call':
            initial_option_price = bs.call_price(mid_price, strike, T, risk_free_rate, mid_vol)
        else:
            initial_option_price = bs.put_price(mid_price, strike, T, risk_free_rate, mid_vol)
    
    # Calculate returns for each combination
    for i, vol in enumerate(implied_vols):
        for j, price in enumerate(underlying_prices):
            if option_type.lower() == 'call':
                option_price = bs.call_price(price, strike, T, risk_free_rate, vol)
            else:
                option_price = bs.put_price(price, strike, T, risk_free_rate, vol)
            
            # Calculate return (percentage)
            if initial_option_price > 0:
                returns = ((option_price - initial_option_price) / initial_option_price) * 100
            else:
                returns = 0
            
            Z[i, j] = returns
    
    # Create surface plot
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y * 100,  # Convert to percentage
        z=Z,
        colorscale='RdYlGn',  # Red for losses, green for gains
        colorbar=dict(title="Return (%)"),
        name='Returns'
    )])
    
    fig.update_layout(
        title=f'{option_type.capitalize()} Option Returns Surface (Strike: ${strike:.0f}, {days_to_expiry} days)',
        scene=dict(
            xaxis_title='Underlying Price ($)',
            yaxis_title='Implied Volatility (%)',
            zaxis_title='Return (%)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=900,
        height=700
    )
    
    return fig


def create_portfolio_returns_surface(
    underlying_prices: np.ndarray,
    implied_vols: np.ndarray,
    days_to_expiry: int,
    portfolio_positions: List,
    risk_free_rate: float = 0.05
) -> go.Figure:
    """Create 3D returns surface for entire portfolio.
    
    Args:
        underlying_prices: Array of underlying prices
        implied_vols: Array of implied volatilities
        days_to_expiry: Days to expiration
        portfolio_positions: List of Position objects
        risk_free_rate: Risk-free rate
        
    Returns:
        Plotly Figure object
    """
    bs = BlackScholes()
    T = days_to_expiry / 365.25
    
    # Create meshgrid
    X, Y = np.meshgrid(underlying_prices, implied_vols)
    Z = np.zeros_like(X)
    
    # Calculate initial portfolio value
    initial_value = sum(pos.market_value for pos in portfolio_positions)
    
    # Calculate portfolio returns for each combination
    for i, vol in enumerate(implied_vols):
        for j, base_price in enumerate(underlying_prices):
            portfolio_value = 0
            
            for pos in portfolio_positions:
                if pos.position_type == 'stock':
                    # Stock value changes linearly with price
                    price_ratio = base_price / pos.current_price
                    portfolio_value += pos.market_value * price_ratio
                    
                elif pos.position_type == 'option':
                    # Reprice option
                    underlying_ratio = base_price / pos.current_price
                    new_underlying = pos.strike * underlying_ratio  # Approximate
                    
                    if pos.option_type == 'call':
                        option_price = bs.call_price(
                            new_underlying, pos.strike, T, risk_free_rate, vol
                        )
                    else:
                        option_price = bs.put_price(
                            new_underlying, pos.strike, T, risk_free_rate, vol
                        )
                    
                    portfolio_value += option_price * pos.quantity * 100
            
            # Calculate return
            if initial_value > 0:
                returns = ((portfolio_value - initial_value) / initial_value) * 100
            else:
                returns = 0
            
            Z[i, j] = returns
    
    # Create surface plot
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y * 100,
        z=Z,
        colorscale='RdYlGn',
        colorbar=dict(title="Return (%)"),
        name='Portfolio Returns'
    )])
    
    fig.update_layout(
        title=f'Portfolio Returns Surface ({days_to_expiry} days)',
        scene=dict(
            xaxis_title='Underlying Price ($)',
            yaxis_title='Implied Volatility (%)',
            zaxis_title='Portfolio Return (%)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=900,
        height=700
    )
    
    return fig


if __name__ == '__main__':
    # Test with sample data
    prices = np.linspace(80, 120, 20)
    vols = np.linspace(0.15, 0.45, 15)
    
    fig = create_returns_surface(
        underlying_prices=prices,
        implied_vols=vols,
        strike=100,
        days_to_expiry=30,
        option_type='call'
    )
    fig.show()
