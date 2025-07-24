from manager import kite_manager
from strategies.rsi_strategy import FastRSIStrategy
from strategies.moving_average_strategy import FastMovingAverageStrategy
from strategies.bollinger_bands_strategy import FastBollingerBandsStrategy
from strategies.macd_strategy import FastMACDStrategy
from strategies.stochastic_oscillator_strategy import FastStochasticStrategy
from strategies.vwap_strategy import FastVWAPStrategy
import pandas as pd
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import enhanced modules
from utils.logger import Logger
from utils.risk_manager import (
    place_order, 
    check_stop_losses_and_take_profits,
    get_strategy_performance,
    get_risk_exposure
)

# Initialize enhanced logger
logger = Logger(console_output=True, file_output=True)

# Initialize strategy instances (global for performance)
strategy_instances = {
    'RSI': FastRSIStrategy(),
    'MovingAverage': FastMovingAverageStrategy(),
    'BollingerBands': FastBollingerBandsStrategy(),
    'MACD': FastMACDStrategy(),
    'StochasticOscillator': FastStochasticStrategy(),
    'VWAP': FastVWAPStrategy()
}

# Initialize Kite Connect
# kite = KiteConnect(api_key="your_api_key")

def get_instrument_token(stock_symbol):
    """
    Fetch the instrument token for a given stock symbol.
    """
    retries = 3
    for attempt in range(retries):
        try:
            instruments = kite_manager.get_instruments("NSE")
            for instrument in instruments:
                if instrument["tradingsymbol"] == stock_symbol:
                    return instrument["instrument_token"]
            logger.error(f"Instrument token not found for {stock_symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching instrument token for {stock_symbol} (Attempt {attempt + 1}/{retries}): {e}")
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
        logger.error("No stock symbols provided for fetching data.")
        return None  # Return None if no symbols are provided

    # Prepare batch request for LTP
    ltp_request = [f"NSE:{symbol}" for symbol in stock_symbols]

    for attempt in range(retries):
        try:
            # Fetch LTP for all stocks in a single request using kite_manager
            ltp_data = kite_manager.get_ltp(ltp_request)
            for stock_symbol in stock_symbols:
                ltp_key = f"NSE:{stock_symbol}"
                if isinstance(ltp_data, dict) and ltp_key in ltp_data:
                    ltp = ltp_data[ltp_key].get("last_price", 0)
                    stock_data[stock_symbol] = {"ltp": ltp}
                else:
                    logger.warning(f"LTP not found for {stock_symbol}")
                    stock_data[stock_symbol] = {"ltp": None}

            # Fetch historical data only for stocks with valid LTP
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            for stock_symbol, data in stock_data.items():
                if data["ltp"] is None:
                    continue

                instrument_token = get_instrument_token(stock_symbol)
                if not instrument_token:
                    logger.warning(f"Instrument token not found for {stock_symbol}")
                    stock_data[stock_symbol]["historical_data"] = None
                    continue

                try:
                    # Use kite_manager for historical data
                    historical_data = kite_manager.get_historical_data(
                        instrument_token=instrument_token,
                        from_date=start_date,
                        to_date=end_date,
                        interval=interval
                    )

                    # Validate and store historical data
                    if historical_data is None or len(historical_data) == 0:
                        logger.warning(f"No historical data fetched for {stock_symbol}")
                        stock_data[stock_symbol]["historical_data"] = None
                    else:
                        df = pd.DataFrame(historical_data)
                        required_columns = {"close", "high", "low", "volume"}
                        if required_columns.issubset(df.columns):
                            stock_data[stock_symbol]["historical_data"] = df
                        else:
                            logger.warning(f"Missing required columns in historical data for {stock_symbol}")
                            stock_data[stock_symbol]["historical_data"] = None
                except Exception as e:
                    logger.error(f"Error fetching historical data for {stock_symbol}: {e}")
                    stock_data[stock_symbol]["historical_data"] = None

            logger.debug(f"Fetched data for stocks: {stock_data}")
            return stock_data
        except Exception as e:
            logger.error(f"Error fetching data (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None

    logger.error("Failed to fetch real-time data after retries.")
    return None  # Return None if all retries fail

def normalize_signal(signal):
    """
    Normalize different signal formats (string or dict) to a standard format
    """
    if isinstance(signal, str):
        return {
            "signal": signal,
            "strength": 100 if signal in ["BUY", "SELL"] else 0
        }
    elif isinstance(signal, dict):
        # If it's already a dict, ensure it has required fields
        if "signal" not in signal:
            if "signal_type" in signal:
                signal["signal"] = signal["signal_type"]
            else:
                # Try to extract signal from common keys
                for key in signal:
                    if isinstance(signal[key], str) and signal[key] in ["BUY", "SELL", "HOLD"]:
                        signal["signal"] = signal[key]
                        break
        
        # Ensure strength/confidence is present
        if "strength" not in signal and "confidence" in signal:
            signal["strength"] = signal["confidence"]
        elif "strength" not in signal:
            signal["strength"] = 100 if signal.get("signal") in ["BUY", "SELL"] else 0
            
        return signal
    else:
        # Default for unknown format
        return {"signal": "HOLD", "strength": 0}

def aggregate_signals(signals_dict):
    """
    Aggregate signals from multiple strategies into a single decision
    
    Parameters:
    - signals_dict: Dictionary of strategy results
    
    Returns:
    - dict: Aggregated signal with strength and contributing strategies
    """
    # Initialize counters
    buy_count = 0
    sell_count = 0
    hold_count = 0
    buy_strength = 0
    sell_strength = 0
    
    # Count signals by type and aggregate strengths
    contributing_strategies = {"BUY": [], "SELL": [], "HOLD": []}
    
    for strategy_name, signal in signals_dict.items():
        norm_signal = normalize_signal(signal)
        signal_type = norm_signal.get("signal", "HOLD")
        strength = norm_signal.get("strength", 0)
        
        if signal_type == "BUY":
            buy_count += 1
            buy_strength += strength
            contributing_strategies["BUY"].append({
                "strategy": strategy_name,
                "strength": strength,
                "details": norm_signal
            })
        elif signal_type == "SELL":
            sell_count += 1
            sell_strength += strength
            contributing_strategies["SELL"].append({
                "strategy": strategy_name,
                "strength": strength,
                "details": norm_signal
            })
        else:
            hold_count += 1
            contributing_strategies["HOLD"].append({
                "strategy": strategy_name,
                "details": norm_signal
            })
    
    # Calculate average strengths
    avg_buy_strength = buy_strength / buy_count if buy_count > 0 else 0
    avg_sell_strength = sell_strength / sell_count if sell_count > 0 else 0
    
    # Decision logic based on counts and strengths
    total_strategies = buy_count + sell_count + hold_count
    
    # Calculate agreement percentages
    buy_agreement = (buy_count / total_strategies * 100) if total_strategies > 0 else 0
    sell_agreement = (sell_count / total_strategies * 100) if total_strategies > 0 else 0
    
    # Make final decision based on voting and strength
    if buy_count > sell_count and buy_agreement >= 40 and avg_buy_strength >= 60:
        final_signal = "BUY"
        final_strength = avg_buy_strength
        contributors = contributing_strategies["BUY"]
    elif sell_count > buy_count and sell_agreement >= 40 and avg_sell_strength >= 60:
        final_signal = "SELL"
        final_strength = avg_sell_strength
        contributors = contributing_strategies["SELL"]
    else:
        final_signal = "HOLD"
        final_strength = 0
        contributors = contributing_strategies["HOLD"]
    
    return {
        "signal": final_signal,
        "strength": final_strength,
        "agreement": buy_agreement if final_signal == "BUY" else (sell_agreement if final_signal == "SELL" else 0),
        "contributing_strategies": contributors,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count
    }

def process_stock(stock_symbol, stock_data):
    """
    Process strategies for a single stock symbol using pre-fetched data.
    """
    data = stock_data.get(stock_symbol, {})
    df = data.get("historical_data")
    ltp = data.get("ltp")

    if df is None or df.empty or ltp is None:
        logger.warning(f"Skipping {stock_symbol} due to data issues. DataFrame available: {df is not None}, LTP: {ltp}")
        return stock_symbol, {"signal": "HOLD", "strength": 0, "reason": "DATA_ISSUE"}

    try:
        # Process each tick through the fast strategies
        strategy_signals = {}
        
        # Reset strategy instances for new stock
        for strategy in strategy_instances.values():
            strategy.reset_daily_counters()
        
        # Process historical data through strategies to build up state
        for _, row in df.iterrows():
            price = row['close']
            volume = row.get('volume', 0)
            high = row.get('high', price)
            low = row.get('low', price)
            
            for name, strategy in strategy_instances.items():
                try:
                    if name in ['MovingAverage', 'VWAP', 'StochasticOscillator']:
                        # These strategies need high/low data
                        result = strategy.add_tick(price, volume, high=high, low=low)
                    else:
                        # Other strategies only need price and volume
                        result = strategy.add_tick(price, volume)
                except Exception as e:
                    logger.error(f"Error processing tick for {name} strategy on {stock_symbol}: {e}")
        
        # Get final signals from each strategy
        for name, strategy in strategy_instances.items():
            try:
                # Get the last result as the final signal
                final_price = df['close'].iloc[-1]
                final_volume = df.get('volume', pd.Series([0])).iloc[-1]
                
                if name in ['MovingAverage', 'VWAP', 'StochasticOscillator']:
                    final_high = df.get('high', pd.Series([final_price])).iloc[-1]
                    final_low = df.get('low', pd.Series([final_price])).iloc[-1]
                    result = strategy.add_tick(final_price, final_volume, high=final_high, low=final_low)
                else:
                    result = strategy.add_tick(final_price, final_volume)
                
                strategy_signals[name] = result
                
            except Exception as e:
                logger.error(f"Error getting final signal from {name} strategy for {stock_symbol}: {e}")
                strategy_signals[name] = {"signal": "ERROR", "strength": 0}
            
        # Aggregate signals from all strategies
        aggregated_signal = aggregate_signals(strategy_signals)
        
        # Log the combined signal
        logger.info(
            f"Aggregated signal for {stock_symbol}: {aggregated_signal['signal']} with {aggregated_signal['strength']}% strength", 
            extra={"symbol": stock_symbol, "aggregated_signal": aggregated_signal}
        )
        
        # Place order if signal is actionable
        if aggregated_signal['signal'] in ['BUY', 'SELL'] and aggregated_signal['strength'] >= 60:
            trade = place_order(
                aggregated_signal, 
                stock_symbol, 
                "Consensus", 
                df
            )
            if trade:
                logger.info(f"Trade placed for {stock_symbol}: {trade}")
        
        return stock_symbol, {
            "aggregated_signal": aggregated_signal,
            "individual_signals": strategy_signals
        }
    except Exception as e:
        logger.error(f"Error processing strategies for {stock_symbol}: {e}")
        return stock_symbol, {"signal": "ERROR", "strength": 0, "reason": str(e)}

def run_all_strategies_for_stocks(stock_symbols, check_existing_positions=True):
    """
    Run all strategies for multiple stocks using Zerodha API in parallel.
    """
    # First check existing positions for stop/take profit hits
    if check_existing_positions:
        logger.info("Checking existing positions for stop loss/take profit triggers")
        closed_trades = check_stop_losses_and_take_profits()
        if closed_trades:
            logger.info(f"Closed {len(closed_trades)} trades due to SL/TP triggers")
            
    # Fetch data for all stocks
    stock_data = fetch_real_time_data(stock_symbols)
    if not stock_data:
        logger.error("Failed to fetch stock data.")
        return {}

    # Process signals for each stock in parallel
    signals = {}
    with ThreadPoolExecutor(max_workers=min(10, len(stock_symbols))) as executor:
        futures = {executor.submit(process_stock, stock, stock_data): stock for stock in stock_symbols}
        for future in as_completed(futures):
            try:
                stock_symbol, result = future.result()
                signals[stock_symbol] = result
            except Exception as e:
                logger.error(f"Error in thread execution: {e}")

    # Log portfolio statistics after processing
    risk_metrics = get_risk_exposure()
    logger.info(f"Portfolio risk metrics: {risk_metrics}")
    
    # Log performance statistics
    performance = get_strategy_performance("Consensus")
    logger.info(f"Overall strategy performance: {performance}")
    
    return signals

def run_scheduled(stock_symbols, interval_minutes=5):
    """
    Run strategies on a schedule and handle automatic trading.
    
    Parameters:
    - stock_symbols: List of stock symbols to monitor
    - interval_minutes: How often to run the strategies (in minutes)
    """
    logger.info(f"Starting scheduled strategy execution every {interval_minutes} minutes")
    
    while True:
        start_time = datetime.now()
        logger.info(f"Running strategies at {start_time}")
        
        try:
            # Execute strategies
            results = run_all_strategies_for_stocks(stock_symbols)
            logger.info(f"Completed strategy execution for {len(results)} symbols")
            
            # Calculate time to next run
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_seconds = max(0, interval_minutes * 60 - elapsed)
            
            if sleep_seconds > 0:
                logger.info(f"Sleeping for {sleep_seconds:.1f} seconds until next run")
                time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("Strategy execution stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduled execution: {e}")
            # Sleep for a shorter time on error to retry sooner
            time.sleep(60)
