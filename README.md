# Algo-Bot

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Algo-Bot is a comprehensive algorithmic trading system designed to analyze market data, implement various trading strategies, and execute trades automatically. It connects to the Zerodha trading platform to fetch real-time market data and place orders based on technical indicators and risk management rules.

## Table of Contents

- [Features](#features)
- [Trading Strategies](#trading-strategies)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

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
