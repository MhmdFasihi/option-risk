"""Greek surface generation."""

import numpy as np
import plotly.graph_objects as go
from typing import List
from datetime import datetime, timedelta
from models.black_scholes import BlackScholes


def create_greek_surface(
    greek_name: str,
    strikes: List[float],
    expiries: List[datetime],
    underlying_price: float,
    volatility: float = 0.25,
    risk_free_rate: float = 0.05,
    option_type: str = 'call'
) -> go.Figure:
    """Create 3D Greek surface visualization.
    
    Args:
        greek_name: Name of the Greek ('delta', 'gamma', 'vega', 'theta', 'rho')
        strikes: List of strike prices
        expiries: List of expiration dates
        underlying_price: Current underlying price
        volatility: Implied volatility (default 25%)
        risk_free_rate: Risk-free rate (default 5%)
        option_type: 'call' or 'put'
        
    Returns:
        Plotly Figure object
    """
    bs = BlackScholes()
    now = datetime.now()
    
    # Convert expiries to days
    days_to_expiry = [(exp - now).days for exp in expiries]
    
    # Create meshgrid
    X, Y = np.meshgrid(strikes, days_to_expiry)
    
    # Calculate Greeks
    greek_func = getattr(bs, greek_name.lower())
    
    Z = np.zeros_like(X)
    for i, strike in enumerate(strikes):
        for j, days in enumerate(days_to_expiry):
            T = max(days / 365.25, 0.001)  # Time in years
            
            if greek_name.lower() in ['delta', 'theta', 'rho']:
                Z[j, i] = greek_func(underlying_price, strike, T, risk_free_rate, volatility, option_type)
            else:  # gamma, vega (same for calls and puts)
                Z[j, i] = greek_func(underlying_price, strike, T, risk_free_rate, volatility)
    
    # Color scale based on Greek
    if greek_name.lower() == 'theta':
        colorscale = 'Reds'  # Theta is usually negative
    elif greek_name.lower() == 'delta':
        colorscale = 'RdYlGn'  # Red to green
    else:
        colorscale = 'Viridis'
    
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        name=greek_name.capitalize()
    )])
    
    fig.update_layout(
        title=f'{greek_name.capitalize()} Surface ({option_type.capitalize()})',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiry',
            zaxis_title=greek_name.capitalize(),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=800,
        height=600
    )
    
    return fig


def create_sample_greek_surface(
    greek_name: str = 'gamma',
    underlying_price: float = 100
) -> go.Figure:
    """Create a sample Greek surface with default parameters.
    
    Args:
        greek_name: Name of the Greek to visualize
        underlying_price: Current underlying price
        
    Returns:
        Plotly Figure object
    """
    # Generate strikes around the underlying
    strikes = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 15)
    
    # Generate expiries from 1 week to 6 months
    now = datetime.now()
    expiries = [now + timedelta(days=d) for d in [7, 14, 30, 60, 90, 120, 180]]
    
    return create_greek_surface(
        greek_name=greek_name,
        strikes=strikes.tolist(),
        expiries=expiries,
        underlying_price=underlying_price,
        volatility=0.25,
        risk_free_rate=0.05,
        option_type='call'
    )


if __name__ == '__main__':
    # Test with sample data
    for greek in ['delta', 'gamma', 'vega', 'theta']:
        fig = create_sample_greek_surface(greek)
        fig.show()
