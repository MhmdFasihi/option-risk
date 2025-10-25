"""Configuration settings for the application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Data source configuration
USE_SAMPLE_DATA = os.getenv('USE_SAMPLE_DATA', 'True').lower() == 'true'
API_ENDPOINT = os.getenv('API_ENDPOINT', '')
API_KEY = os.getenv('API_KEY', '')

# Risk calculation parameters
RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.05'))  # 5% default
CONFIDENCE_LEVEL_95 = 0.95
CONFIDENCE_LEVEL_99 = 0.99

# VaR parameters
VAR_WINDOW = int(os.getenv('VAR_WINDOW', '252'))  # 1 year of trading days
MC_SIMULATIONS = int(os.getenv('MC_SIMULATIONS', '10000'))
TIME_HORIZON_DAYS = int(os.getenv('TIME_HORIZON_DAYS', '1'))

# Visualization settings
CHART_WIDTH = 800
CHART_HEIGHT = 600
COLOR_PROFIT = 'green'
COLOR_LOSS = 'red'
COLOR_NEUTRAL = 'blue'

# Portfolio settings
DEFAULT_SAMPLE_PORTFOLIO = 'sample'  # Can be changed to load different samples
