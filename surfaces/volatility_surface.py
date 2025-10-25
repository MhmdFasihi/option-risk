"""Volatility surface generation."""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from datetime import datetime, timedelta


def create_volatility_surface(
    strikes: List[float],
    expiries: List[datetime],
    implied_vols: np.ndarray,
    underlying_price: float
) -> go.Figure:
    """Create 3D volatility surface visualization.
    
    Args:
        strikes: List of strike prices
        expiries: List of expiration dates
        implied_vols: 2D array of implied volatilities (strikes x expiries)
        underlying_price: Current underlying price
        
    Returns:
        Plotly Figure object
    """
    # Convert expiries to days to expiration
    now = datetime.now()
    days_to_expiry = [(exp - now).days for exp in expiries]
    
    # Create meshgrid
    X, Y = np.meshgrid(strikes, days_to_expiry)
    
    # Ensure implied_vols is 2D
    if implied_vols.ndim == 1:
        implied_vols = implied_vols.reshape(-1, 1)
    
    # Transpose to match meshgrid shape
    Z = implied_vols.T
    
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        name='Implied Volatility'
    )])
    
    fig.update_layout(
        title='Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiry',
            zaxis_title='Implied Volatility',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=800,
        height=600
    )
    
    # Add a marker for ATM
    fig.add_trace(go.Scatter3d(
        x=[underlying_price],
        y=[days_to_expiry[len(days_to_expiry)//2]],
        z=[implied_vols[len(strikes)//2, len(expiries)//2] if implied_vols.shape[1] > 0 else 0.3],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='ATM'
    ))
    
    return fig


def create_sample_volatility_surface(
    underlying_price: float = 100,
    strike_range: Tuple[float, float] = (80, 120),
    num_strikes: int = 10,
    expiry_range_days: Tuple[int, int] = (30, 180),
    num_expiries: int = 6
) -> go.Figure:
    """Create a sample volatility surface with synthetic data.
    
    Args:
        underlying_price: Current underlying price
        strike_range: (min_strike, max_strike)
        num_strikes: Number of strikes to generate
        expiry_range_days: (min_days, max_days) to expiry
        num_expiries: Number of expiration dates
        
    Returns:
        Plotly Figure object
    """
    # Generate strikes
    strikes = np.linspace(strike_range[0], strike_range[1], num_strikes)
    
    # Generate expiries
    now = datetime.now()
    days = np.linspace(expiry_range_days[0], expiry_range_days[1], num_expiries)
    expiries = [now + timedelta(days=int(d)) for d in days]
    
    # Generate synthetic implied volatilities with smile
    implied_vols = np.zeros((num_strikes, num_expiries))
    
    for i, strike in enumerate(strikes):
        for j, days_to_exp in enumerate(days):
            # Volatility smile: higher IV for OTM options
            moneyness = strike / underlying_price
            
            # Base volatility
            base_vol = 0.25
            
            # Smile effect (U-shaped)
            smile = 0.1 * (moneyness - 1) ** 2
            
            # Term structure (longer term = slightly higher vol)
            term = 0.05 * (days_to_exp / 365)
            
            implied_vols[i, j] = base_vol + smile + term
    
    return create_volatility_surface(strikes, expiries, implied_vols, underlying_price)


if __name__ == '__main__':
    # Test with sample data
    fig = create_sample_volatility_surface()
    fig.show()
