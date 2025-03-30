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

# Example usage:
# real_time_allocation = calculate_real_time_allocation()
# print(real_time_allocation)
