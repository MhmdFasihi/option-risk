"""Utility functions."""

from datetime import datetime
import pandas as pd
from typing import Dict, Any


def format_currency(value: float) -> str:
    """Format a number as currency.
    
    Args:
        value: Numeric value
        
    Returns:
        Formatted string with $ and comma separators
    """
    return f"${value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage.
    
    Args:
        value: Numeric value (e.g., 0.25 for 25%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def days_to_expiry(expiry: datetime) -> int:
    """Calculate days to expiration.
    
    Args:
        expiry: Expiration datetime
        
    Returns:
        Number of days to expiry
    """
    now = datetime.now()
    if expiry <= now:
        return 0
    return (expiry - now).days


def format_greek(greek_name: str, value: float) -> str:
    """Format a Greek value for display.
    
    Args:
        greek_name: Name of the Greek
        value: Greek value
        
    Returns:
        Formatted string
    """
    if greek_name.lower() == 'delta':
        return f"{value:.3f}"
    elif greek_name.lower() == 'gamma':
        return f"{value:.4f}"
    elif greek_name.lower() == 'vega':
        return f"{value:.2f}"
    elif greek_name.lower() == 'theta':
        return f"{value:.2f}"
    elif greek_name.lower() == 'rho':
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"


def create_summary_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary table from dictionary data.
    
    Args:
        data: Dictionary of metric names to values
        
    Returns:
        DataFrame formatted for display
    """
    df = pd.DataFrame([
        {'Metric': k, 'Value': v}
        for k, v in data.items()
    ])
    return df


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
