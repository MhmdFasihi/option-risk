# Copilot Instructions - Options Risk Management System

## Project Overview
Portfolio risk management application for analyzing options and underlying assets. Displays risk metrics (VaR, CVaR, Greeks) and volatility surfaces via Streamlit dashboard. Supports both sample portfolios and live API integration.

## Architecture

### Data Pipeline
- **Sample Mode**: Start with hardcoded/CSV sample portfolios for development
- **API Mode**: Connect to portfolio API for real-time position data
- **Risk Engine**: Calculate VaR (Value at Risk), CVaR (Conditional VaR), gamma exposure, and option Greeks (delta, gamma, vega, theta, rho)
- **Surface Generation**: Build 3D volatility surfaces and Greek surfaces for visualization

### Key Components
- `dashboard/`: Streamlit UI with interactive charts and 3D surfaces
- `risk/`: Risk metric calculations (VaR, CVaR, Greeks)
- `data/`: Portfolio data loaders (sample and API clients)
- `surfaces/`: 3D surface generation (volatility, Greeks)
- `models/`: Options pricing models (Black-Scholes, implied volatility)

## Environment Management

### Conda Setup
This project uses **conda** for environment management. Always use conda commands:

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate option-risk

# Update dependencies
conda env update -f environment.yml --prune
```

**Key packages**: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`, `streamlit`, `yfinance`

## Development Workflow

### Running the Dashboard
```bash
streamlit run dashboard/app.py
```

### Typical Development Flow
1. Test with sample portfolio first (`data/sample_portfolio.py` or `.csv`)
2. Implement risk calculations in isolation (unit tests in `tests/`)
3. Integrate into dashboard with clear error handling for missing data
4. Switch to API mode once sample mode works

### Testing Risk Calculations
- Greeks should sum to portfolio-level exposures
- VaR confidence intervals: typically 95% or 99%
- CVaR (ES) must be >= VaR
- Validate against known option pricing results

## Coding Conventions

### Options Data Structure
Portfolio positions should include:
- Underlying symbol, price, quantity
- Option type (call/put), strike, expiry, IV (implied volatility)
- Greeks calculated at position level, then aggregated

### Risk Metric Patterns
- **VaR**: Implement **both** historical simulation and Monte Carlo methods; specify confidence level (95%, 99%) and time horizon (1-day, 10-day)
- **CVaR**: Conditional expectation beyond VaR threshold (must be >= VaR)
- **Gamma Exposure**: Net gamma × (underlying price)² × 0.01² for dollar gamma
- **Greeks**: Calculate per-position using Black-Scholes, then aggregate weighted by position size

### Visualization Standards
- Use `plotly` for 3D surfaces (volatility, gamma surface)
- Color schemes: Red for losses/negative exposure, green for gains/positive
- Volatility surface: X=strike, Y=expiry, Z=implied volatility
- Include hover data: strike, expiry, IV, Greek values

### API Integration Pattern
```python
# Support both modes via config or environment variable
if USE_SAMPLE_DATA:
    portfolio = load_sample_portfolio()  # Uses yfinance for market data
else:
    portfolio = api_client.fetch_portfolio()  # Custom internal API
```

**Market Data Sources**:
- **Development/Testing**: `yfinance` for real-time prices, historical data, and option chains
- **Production**: Custom internal portfolio API (integration details TBD)

## File Organization
- `environment.yml`: Conda dependencies (not `requirements.txt`)
- `config.py` or `.env`: API keys, data source toggle
- `utils/`: Shared helpers (date handling, formatting, Greeks formulas)
- `tests/`: Unit tests for risk calculations

## Common Pitfalls
- Don't mix pip and conda for primary dependencies
- Always handle missing/stale market data gracefully in dashboard
- Options near expiry have extreme Greeks; clamp or filter outliers in visualizations
- Ensure date/time awareness (market hours, expiry dates in correct timezone)

## External Dependencies
- **yfinance**: Primary data source for development - fetch underlying prices, historical data, option chains
- **Custom Portfolio API**: Internal endpoint for production position data (integration pending)
- **Risk-free rate**: Treasury yields for Black-Scholes Greeks calculations

## VaR Implementation Notes
- Implement both methodologies side-by-side for comparison:
  - **Historical VaR**: Rolling window (e.g., 252 trading days), empirical quantile
  - **Monte Carlo VaR**: GBM or custom process, 10,000+ simulations
- Dashboard should display both methods with toggle or side-by-side comparison
- Log discrepancies between methods for model validation
