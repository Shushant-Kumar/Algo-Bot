# Production-Ready Algorithmic Trading System

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

A high-performance, enterprise-grade algorithmic trading system optimized for fast intraday trading with Zerodha Kite API. Features advanced strategies, real-time monitoring, comprehensive risk management, and production-ready security.

## 🚀 Key Features

### Trading & Strategy Management
- **Fast Strategy Execution**: Optimized base classes with O(1) mathematical operations
- **Multiple Strategies**: Bollinger Bands, RSI, MACD, Moving Average, Stochastic, VWAP
- **Real-time Data Processing**: High-frequency data ingestion and analysis
- **Advanced Order Management**: Smart order routing with circuit breakers
- **Risk Management**: Position sizing, daily loss limits, exposure controls

### Security & Authentication
- **Secure Token Management**: AES-256 encrypted token storage
- **Automatic Token Refresh**: Intelligent token caching and validation
- **Environment Protection**: Secure credential management with .env files
- **Production Security**: File permissions, checksums, encryption keys

### Performance & Monitoring
- **Multi-threading Support**: Concurrent strategy execution
- **Real-time Monitoring**: Live performance metrics and health checks
- **Comprehensive Logging**: Structured logging with multiple levels
- **Database Integration**: PostgreSQL, MongoDB, Redis support
- **API Rate Limiting**: Intelligent request throttling

### Production Features
- **Market Hours Validation**: Trading session awareness
- **Circuit Breakers**: Automatic trading halts on anomalies
- **Position Tracking**: Real-time P&L monitoring
- **Deployment Ready**: Docker, Gunicorn, Supervisor integration
- **Error Recovery**: Automatic retry mechanisms and fallback systems

## Table of Contents

- [Quick Start](#quick-start)
- [Trading Strategies](#trading-strategies)
- [Production Setup](#production-setup)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### 1. One-Command Setup
```bash
# Clone and setup
git clone https://github.com/Shushant-Kumar/Algo-Bot.git
cd Algo-Bot
pip install -r requirements.txt

# Quick authentication and testing
python quick_start.py
```

### 2. Secure Authentication
```bash
# First-time setup with secure authentication
python zerodha_integration.py
```

### 3. Production Trading
```bash
# Start the production trading system
python main.py
```

## 🔐 Secure Authentication Setup

Our production-ready authentication system provides enterprise-grade security:

### Features
- **AES-256 Encryption**: All tokens encrypted at rest
- **Automatic Validation**: Token integrity and expiration checks  
- **Secure Storage**: Restricted file permissions and secure cache
- **Retry Logic**: Robust authentication with automatic retries

### Setup Process
1. **Copy Environment Template**:
   ```bash
   cp .env.template .env
   ```

2. **Add Your Credentials** (edit `.env`):
   ```env
   KITE_API_KEY=your_api_key_here
   KITE_API_SECRET=your_api_secret_here
   ```

3. **Run Authentication**:
   ```bash
   python zerodha_integration.py
   ```

4. **Follow Browser Instructions**:
   - Visit the generated login URL
   - Authorize the application
   - Copy the request token
   - Paste when prompted

### Integration Example
```python
from zerodha_integration import get_authenticated_kite
from manager import AdvancedKiteManager

# Get authenticated connection
kite = get_authenticated_kite()
if kite:
    manager = AdvancedKiteManager(kite)
    # Ready for trading!
```

- Integration with Zerodha trading platform
- Multiple technical indicator-based strategies
- Risk management with ATR-based stop loss and take profit levels
- Dynamic capital allocation based on signals and risk parameters
- Order execution with slippage tolerance
- Daily profit/loss tracking and risk limits
- Comprehensive logging system
- Multi-timeframe analysis
- Simulation mode for paper trading

## Trading Strategies

### Technical Indicators
- RSI Strategy with divergence detection and trend filters
- Moving Average Strategy with EMA crossover and ATR-based stops
- Bollinger Bands Strategy with RSI confirmation
- MACD Strategy with zero-line, histogram, and RSI filters
- Stochastic Oscillator Strategy with support/resistance and divergence
- VWAP Strategy with multi-timeframe analysis and volume filters

### Risk Management
- Per-trade risk percentage of capital
- ATR-based position sizing
- Daily loss limits
- Maximum capital allocation per stock
- Strategy-specific exposure limits
- Multiple timeframe confirmation

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Zerodha Kite API credentials
- Internet connection for live data

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shushant-Kumar/Algo-Bot.git
   cd algo-bot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   Create a `.env` file in the root directory with the following:
   ```
   KITE_API_KEY=your_api_key_here
   KITE_API_SECRET=your_api_secret_here
   ZERODHA_ACCESS_TOKEN=your_access_token_here
   ```

6. **Create data directory:**
   ```bash
   mkdir -p data logs
   ```

## Usage

### Step-by-Step Configuration

1. **Edit the config.py file:**
   - Set your capital amount (`TOTAL_CAPITAL`)
   - Configure risk parameters (`RISK_PER_TRADE_PERCENT`, `MAX_RISK_PER_SYMBOL`)
   - Add stocks to watchlist (`WATCHLIST`)
   - Set simulation mode (`SIMULATION_MODE = True` for paper trading)

2. **Prepare test data for simulation mode:**
   - Download historical data for your watchlist symbols
   - Save as CSV files in the `data` directory

3. **Run the bot:**
   ```bash
   python main.py
   ```

4. **Authentication flow:**
   - If no access token is provided, follow the login URL displayed
   - Enter the request token after successful login
   - The bot will automatically store and use the access token

### Trading Process

1. **Initialization:**
   - The bot authenticates with Zerodha
   - It loads watchlist symbols and configuration
   - Initializes the order execution system

2. **Market Data Retrieval:**
   - Fetches real-time quotes for watchlist stocks
   - Retrieves historical data for technical analysis

3. **Strategy Execution:**
   - Applies all technical strategies to the data
   - Aggregates signals from multiple strategies
   - Filters signals based on confidence thresholds

4. **Risk Management:**
   - Calculates position sizes based on ATR and risk parameters
   - Ensures position limits are not exceeded
   - Sets stop-loss and take-profit levels

5. **Order Execution:**
   - In simulation mode, simulates order execution
   - In live mode, places actual orders via Zerodha
   - Handles slippage and order management

6. **Monitoring and Reporting:**
   - Tracks open positions and P&L
   - Monitors stop-loss and take-profit levels
   - Logs all activities and trade performance

### Example Code

```python
# Running a single strategy on historical data
from strategies.rsi_strategy import rsi_strategy_with_filters
import pandas as pd

# Load data
df = pd.read_csv('data/RELIANCE.csv')

# Apply strategy
result = rsi_strategy_with_filters(df)
print(f"Signal: {result['signal']}, Strength: {result['strength']}%")
```

## Project Structure

```
algo-bot/
├── strategies/             # Trading strategies
│   ├── rsi_strategy.py     # RSI-based strategies
│   ├── macd_strategy.py    # MACD-based strategies
│   ├── bollinger_bands_strategy.py
│   ├── moving_average_strategy.py
│   ├── stochastic_oscillator_strategy.py
│   └── vwap_strategy.py
├── utils/                  # Utility functions
│   ├── indicators.py       # Technical indicators calculations
│   ├── logger.py           # Enhanced logging system
│   ├── risk_manager.py     # Risk management functions
│   └── strategy_manager.py # Strategy orchestration
├── data/                   # Stock data storage
├── logs/                   # Log files
├── cache/                  # Token cache and session data
├── main.py                 # Main entry point
├── manager.py              # Zerodha API manager
├── order_execution.py      # Order handling system
├── config.py               # Configuration parameters
├── zerodha_integration.py  # Authentication utility for Zerodha API
└── requirements.txt        # Dependencies
```

### Authentication Utility

The project includes a dedicated authentication utility script:

```bash
python zerodha_integration.py
```

This script helps you generate and manage Zerodha access tokens by:
- Reading API credentials from your .env file or prompting for input
- Generating a login URL and guiding you through authentication
- Saving the access token to both .env file and cache directory
- Making the token immediately available to the bot

Run this script before starting the bot if you need to generate a fresh access token.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the code style guidelines.

## Roadmap

- [ ] Add backtesting framework for strategy optimization
- [ ] Implement machine learning models for pattern recognition
- [ ] Develop Web UI for real-time monitoring and control
- [ ] Add options trading strategies
- [ ] Implement portfolio optimization with risk parity
- [ ] Add multi-broker support (NSE, BSE, Interactive Brokers)
- [ ] Develop alert system (SMS, Email, Telegram)
- [ ] Add sentiment analysis from news and social media

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on algorithmic trading principles and financial market analysis
- Inspired by real-world trading challenges
- Thanks to the trading research community for their publications and insights
- Appreciation to all contributors who help improve and maintain this project
