from kiteconnect import KiteConnect
from strategies.rsi_strategy import rsi_strategy_with_filters
from strategies.moving_average_strategy import moving_average_strategy_with_filters
from strategies.bollinger_bands_strategy import bollinger_bands_strategy_with_rsi
from strategies.macd_strategy import macd_strategy_with_filters
from strategies.stochastic_oscillator_strategy import stochastic_oscillator_strategy_with_filters
from strategies.vwap_strategy import vwap_strategy_with_filters
import pandas as pd
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    filename="strategy_manager.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Kite Connect
kite = KiteConnect(api_key="your_api_key")

def get_instrument_token(stock_symbol):
    """
    Fetch the instrument token for a given stock symbol.
    """
    retries = 3
    for attempt in range(retries):
        try:
            instruments = kite.instruments("NSE")
            for instrument in instruments:
                if instrument["tradingsymbol"] == stock_symbol:
                    return instrument["instrument_token"]
            logging.error(f"Instrument token not found for {stock_symbol}")
            return None
        except Exception as e:
            logging.error(f"Error fetching instrument token for {stock_symbol} (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None

def fetch_real_time_data(stock_symbols, interval="5minute", days=1):
    """
    Fetch real-time stock data for multiple symbols using Zerodha API.
    """
    retries = 3
    stock_data = {}

    if not stock_symbols:
        logging.error("No stock symbols provided for fetching data.")
        return None  # Return None if no symbols are provided

    # Prepare batch request for LTP
    ltp_request = [f"NSE:{symbol}" for symbol in stock_symbols]

    for attempt in range(retries):
        try:
            # Fetch LTP for all stocks in a single request
            ltp_data = kite.ltp(ltp_request)
            for stock_symbol in stock_symbols:
                ltp_key = f"NSE:{stock_symbol}"
                if ltp_key in ltp_data:
                    ltp = ltp_data[ltp_key]["last_price"]
                    stock_data[stock_symbol] = {"ltp": ltp}
                else:
                    logging.warning(f"LTP not found for {stock_symbol}")
                    stock_data[stock_symbol] = {"ltp": None}

            # Fetch historical data only for stocks with valid LTP
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            for stock_symbol, data in stock_data.items():
                if data["ltp"] is None:
                    continue

                instrument_token = get_instrument_token(stock_symbol)
                if not instrument_token:
                    logging.warning(f"Instrument token not found for {stock_symbol}")
                    stock_data[stock_symbol]["historical_data"] = None
                    continue

                try:
                    historical_data = kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=start_date,
                        to_date=end_date,
                        interval=interval
                    )

                    # Validate and store historical data
                    if not historical_data or len(historical_data) == 0:
                        logging.warning(f"No historical data fetched for {stock_symbol}")
                        stock_data[stock_symbol]["historical_data"] = None
                    else:
                        df = pd.DataFrame(historical_data)
                        required_columns = {"close", "high", "low", "volume"}
                        if required_columns.issubset(df.columns):
                            stock_data[stock_symbol]["historical_data"] = df
                        else:
                            logging.warning(f"Missing required columns in historical data for {stock_symbol}")
                            stock_data[stock_symbol]["historical_data"] = None
                except Exception as e:
                    logging.error(f"Error fetching historical data for {stock_symbol}: {e}")
                    stock_data[stock_symbol]["historical_data"] = None

            logging.debug(f"Fetched data for stocks: {stock_data}")
            return stock_data
        except Exception as e:
            logging.error(f"Error fetching data (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None

    logging.error("Failed to fetch real-time data after retries.")
    return None  # Return None if all retries fail

def process_stock(stock_symbol, stock_data, remaining_capital):
    """
    Process strategies for a single stock symbol using pre-fetched data.
    """
    data = stock_data.get(stock_symbol, {})
    df = data.get("historical_data")
    ltp = data.get("ltp")

    if df is None or df.empty or ltp is None:
        logging.warning(f"Skipping {stock_symbol} due to data issues. DataFrame: {df}, LTP: {ltp}")
        return stock_symbol, "HOLD", remaining_capital

    try:
        # Validate remaining capital before computing quantity
        if remaining_capital < ltp:
            logging.warning(f"Insufficient capital for {stock_symbol}. Remaining capital: {remaining_capital}, LTP: {ltp}")
            return stock_symbol, "HOLD", remaining_capital

        # Compute quantity based on remaining capital
        quantity = int(remaining_capital // ltp)
        if quantity == 0:
            logging.warning(f"Quantity is zero for {stock_symbol}. Remaining capital: {remaining_capital}, LTP: {ltp}")
            return stock_symbol, "HOLD", remaining_capital

        # Run enhanced strategies
        result = {
            "RSI": rsi_strategy_with_filters(df),
            "MovingAverage": moving_average_strategy_with_filters(df),
            "BollingerBands": bollinger_bands_strategy_with_rsi(df),
            "MACD": macd_strategy_with_filters(df),
            "StochasticOscillator": stochastic_oscillator_strategy_with_filters(
                df, support_level=ltp * 0.95, resistance_level=ltp * 1.05
            ),
            "VWAP": vwap_strategy_with_filters(df),
        }

        # Deduct capital after placing an order
        remaining_capital -= quantity * ltp
        logging.info(f"Processed strategies for {stock_symbol}: {result}, Remaining Capital: {remaining_capital}")
        return stock_symbol, result, remaining_capital
    except Exception as e:
        logging.error(f"Error processing strategies for {stock_symbol}: {e}")
        return stock_symbol, "HOLD", remaining_capital

def run_all_strategies_for_stocks(stock_symbols, initial_capital):
    """
    Run all strategies for multiple stocks using Zerodha API in parallel.
    """
    stock_data = fetch_real_time_data(stock_symbols)
    if not stock_data:
        logging.error("Failed to fetch stock data.")
        return {}

    signals = {}
    remaining_capital = initial_capital
    with ThreadPoolExecutor(max_workers=min(10, len(stock_symbols))) as executor:  # Dynamically adjust workers
        futures = {executor.submit(process_stock, stock, stock_data, remaining_capital): stock for stock in stock_symbols}
        for future in as_completed(futures):
            try:
                stock_symbol, result, updated_capital = future.result()
                signals[stock_symbol] = result
                remaining_capital = updated_capital  # Update remaining capital after each stock processing
            except Exception as e:
                logging.error(f"Error in thread execution: {e}")

    logging.info(f"[Multi-Stock Signals]: {signals}")
    return signals
