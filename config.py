# config.py

# ATR Multiplier for Stop Loss / Take Profit
ATR_STOP_LOSS_MULTIPLIER = 1.5  # SL = 1.5 * ATR
ATR_TAKE_PROFIT_MULTIPLIER = 3  # TP = 3 * ATR

# Risk per trade % of total capital
RISK_PER_TRADE_PERCENT = 1  # 1% of total capital per trade

# Total capital for trading
TOTAL_CAPITAL = 100000  # Example: ₹1,00,000

# Allocation per stock
PER_STOCK_ALLOCATION = {
    'RELIANCE': 10000,
    'INFY': 5000,
    'TCS': 8000
}

# Maximum capital allocation per stock (e.g., 10,000 units)
PER_STOCK_CAPITAL_LIMIT = 10000

# Slippage tolerance as a fraction (e.g., 0.01 for 1%)
SLIPPAGE_TOLERANCE = 0.01

# Trading mode
SIMULATION_MODE = True  # Set to False for real trading with actual orders

# Validate configuration values
if TOTAL_CAPITAL <= 0:
    raise ValueError("TOTAL_CAPITAL must be greater than 0.")
if RISK_PER_TRADE_PERCENT <= 0 or RISK_PER_TRADE_PERCENT > 100:
    raise ValueError("RISK_PER_TRADE_PERCENT must be between 0 and 100.")
if ATR_STOP_LOSS_MULTIPLIER <= 0 or ATR_TAKE_PROFIT_MULTIPLIER <= 0:
    raise ValueError("ATR multipliers must be greater than 0.")
if SLIPPAGE_TOLERANCE < 0 or SLIPPAGE_TOLERANCE > 1:
    raise ValueError("SLIPPAGE_TOLERANCE must be between 0 and 1.")

import requests
import logging

# API configuration
API_KEY = 'your_api_key_here'  # Replace with your stock market API key
BASE_URL = 'https://www.alphavantage.co/query'  # Example: Alpha Vantage API

def fetch_stock_price(symbol):
    """
    Fetch the real-time stock price for a given symbol.
    """
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': '1min',
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    try:
        latest_time = list(data['Time Series (1min)'].keys())[0]
        return float(data['Time Series (1min)'][latest_time]['1. open'])
    except KeyError:
        logging.error(f"Error fetching data for {symbol}: {data.get('Note', 'Unknown error')}")
        raise ValueError(f"Error fetching data for {symbol}: {data.get('Note', 'Unknown error')}")

def calculate_real_time_allocation():
    """
    Calculate the real-time allocation of capital based on stock prices.
    """
    real_time_allocation = {}
    for stock, allocation in PER_STOCK_ALLOCATION.items():
        try:
            price = fetch_stock_price(stock)
            real_time_allocation[stock] = allocation / price
        except ValueError as e:
            print(e)
    return real_time_allocation

# Strategy-specific parameters
# RSI Strategy parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Moving Average Strategy parameters
MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 20
MA_TREND_WINDOW = 50

# Bollinger Bands Strategy parameters
BB_PERIOD = 20
BB_STD_DEV = 2
BB_RSI_PERIOD = 14

# MACD Strategy parameters
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Stochastic Oscillator parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20

# VWAP Strategy parameters
VWAP_VOLUME_FACTOR = 1.5
VWAP_DISTANCE_LIMIT = 1.5

# Advanced risk management parameters
MAX_STRATEGIES_PER_SYMBOL = 2  # Maximum strategies allowed to trade the same symbol
MAX_RISK_PER_STRATEGY = 10     # Maximum % of capital allocated to a single strategy
MAX_RISK_PER_SYMBOL = 15       # Maximum % of capital allocated to a single symbol
MIN_CONFIDENCE_THRESHOLD = 60  # Minimum confidence score (0-100) to place a trade
MIN_SIGNAL_STRENGTH = 50       # Minimum signal strength for capital allocation

# Portfolio diversification parameters
MIN_STOCKS = 3                 # Minimum number of stocks in portfolio
MAX_STOCKS = 10                # Maximum number of stocks in portfolio
SECTOR_ALLOCATION_LIMITS = {   # Maximum allocation per market sector
    'TECHNOLOGY': 30,          # e.g., 30% maximum in technology stocks
    'FINANCE': 25,
    'HEALTHCARE': 20,
    'ENERGY': 15,
    'CONSUMER': 25,
    'OTHER': 10
}

# Performance tracking parameters
PERFORMANCE_HISTORY_DAYS = 90  # Number of days to keep performance history
WIN_LOSS_RATIO_THRESHOLD = 1.5 # Minimum win/loss ratio to maintain for a strategy

# Logging configuration
LOG_LEVEL = "INFO"             # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_JSON_FORMAT = False

# Backtesting parameters
BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = "2023-01-01"
COMMISSION_RATE = 0.0005       # 0.05% commission per trade

# Order execution parameters
USE_MARKET_ORDERS = True       # If False, use limit orders
LIMIT_ORDER_TIMEOUT = 300      # Seconds to wait for limit order execution
MAX_RETRIES_ON_ERROR = 3       # Number of times to retry on API error
RETRY_DELAY_SECONDS = 5        # Seconds to wait between retries

# Watchlist - stocks to monitor
WATCHLIST = [
    'RELIANCE',
    'INFY',
    'TCS',
    'HDFCBANK',
    'ICICIBANK',
    'SBIN',
    'TATAMOTORS',
    'WIPRO',
    'BHARTIARTL',
    'KOTAKBANK'
]

# Validate advanced parameters
if MAX_RISK_PER_STRATEGY <= 0 or MAX_RISK_PER_STRATEGY > 100:
    raise ValueError("MAX_RISK_PER_STRATEGY must be between 0 and 100.")
if MAX_RISK_PER_SYMBOL <= 0 or MAX_RISK_PER_SYMBOL > 100:
    raise ValueError("MAX_RISK_PER_SYMBOL must be between 0 and 100.")
if MIN_CONFIDENCE_THRESHOLD < 0 or MIN_CONFIDENCE_THRESHOLD > 100:
    raise ValueError("MIN_CONFIDENCE_THRESHOLD must be between 0 and 100.")
if MIN_SIGNAL_STRENGTH < 0 or MIN_SIGNAL_STRENGTH > 100:
    raise ValueError("MIN_SIGNAL_STRENGTH must be between 0 and 100.")