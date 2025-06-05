import os
import pandas as pd
from strategies.rsi_strategy import rsi_strategy
from utils.risk_manager import place_order, remaining_capital
import schedule
import time
import datetime
from manager import kite_manager  # Import the singleton instance
from utils.strategy_manager import run_all_strategies_for_stocks
from config import PER_STOCK_ALLOCATION
from ta.volatility import AverageTrueRange  
import logging
from config import calculate_real_time_allocation, TOTAL_CAPITAL, SLIPPAGE_TOLERANCE
from order_execution import OrderExecution

# Configure logging
logging.basicConfig(
    filename='algo_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DAILY_PNL = 0  # Tracks the daily profit and loss
DAILY_LOSS_LIMIT_PERCENTAGE = 5  # Stop trading if daily loss exceeds 5% of total capital

# Initialize Kite Connect with environment variables
# kite = KiteConnect(api_key=os.getenv("ZERODHA_API_KEY"))
# kite.set_access_token(os.getenv("ZERODHA_ACCESS_TOKEN"))

def get_account_balance():
    """
    Fetch account balance from Zerodha API.
    """
    try:
        funds = kite_manager.get_margins("equity")
        balance = funds["available"]["cash"]
        if balance is None:
            raise ValueError("Account balance could not be fetched.")
        return balance
    except Exception as e:
        print(f"Error fetching account balance: {e}")
        return 0  # Return 0 if balance cannot be fetched

def calculate_atr(df):
    """
    Calculate the Average True Range (ATR) for the given DataFrame.
    """
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    return atr_indicator.average_true_range().iloc[-1]

def calculate_position_size(account_balance, atr, risk_percentage=1):
    """
    Calculate position size based on account balance, ATR, and risk percentage.
    :param account_balance: Total account balance.
    :param atr: Average True Range of the stock.
    :param risk_percentage: Percentage of account balance to risk per trade (default: 1%).
    :return: Position size (number of shares).
    """
    risk_amount = (risk_percentage / 100) * account_balance  # Risk amount per trade
    position_size = risk_amount / atr  # Position size based on ATR
    return max(1, int(position_size))  # Ensure at least 1 share is traded

def place_order(stock_symbol, transaction_type, quantity, df):
    """
    Place an order using Zerodha API with ATR-based limit price and track PnL.
    """
    global DAILY_PNL
    try:
        # Calculate ATR
        atr = calculate_atr(df)
        current_price = df['close'].iloc[-1]

        # Determine limit price based on transaction type
        if transaction_type == "BUY":
            limit_price = current_price + (0.5 * atr)  # Add buffer for buy orders
        elif transaction_type == "SELL":
            limit_price = current_price - (0.5 * atr)  # Subtract buffer for sell orders
        else:
            raise ValueError("Invalid transaction type")

        # Simulate order execution (replace this with actual order execution logic)
        executed_price = limit_price  # Assume the order is executed at the limit price

        # Update PnL (example: assume a fixed profit/loss for demonstration)
        if transaction_type == "SELL":
            DAILY_PNL += (executed_price - current_price) * quantity
        elif transaction_type == "BUY":
            DAILY_PNL -= (current_price - executed_price) * quantity

        # Place limit order using kite_manager
        order_id = kite_manager.place_order(
            symbol=stock_symbol,
            exchange="NSE",
            transaction_type=transaction_type,
            quantity=quantity,
            order_type="LIMIT",  # Use limit order
            price=round(limit_price, 2),  # Round to 2 decimal places
            product="MIS"  # Intraday product type
        )
        print(f"Order placed successfully. Order ID: {order_id}")
    except Exception as e:
        print(f"Error placing order for {stock_symbol}: {e}")

def get_stock_files():
    # List all CSV files in /data
    data_folder = 'data/'
    files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    return files

def get_data(stock_symbol):
    """
    Fetch real-time market data for the given stock symbol using Zerodha's kite.quote().
    """
    try:
        # Fetch live market data using kite_manager
        quote = kite_manager.get_quote(stock_symbol)
        data = {
            'high': [quote[f'NSE:{stock_symbol}']['ohlc']['high']],
            'low': [quote[f'NSE:{stock_symbol}']['ohlc']['low']],
            'close': [quote[f'NSE:{stock_symbol}']['last_price']]
        }
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

def run_bot():
    global DAILY_PNL
    print("\n🔄 Running Algo Bot Cycle...\n")
    stock_symbols = list(PER_STOCK_ALLOCATION.keys())  # Use stock symbols from config
    account_balance = get_account_balance()  # Fetch the current account balance

    # Check if daily loss limit is breached
    total_capital = remaining_capital  # Assuming remaining_capital is the total capital
    daily_loss_limit = (DAILY_LOSS_LIMIT_PERCENTAGE / 100) * total_capital

    if DAILY_PNL <= -daily_loss_limit:
        logging.warning(f"❌ Daily loss limit reached: ₹{DAILY_PNL:.2f}. Stopping trading for the day.")
        return  # Stop trading for the day

    for symbol in stock_symbols:
        df = get_data(symbol)  # Fetch live data for the stock

        if df.empty:
            logging.warning(f"Skipping {symbol} due to data fetch error.")
            continue

        signal = rsi_strategy(df)
        current_price = df['close'].iloc[-1]

        logging.info(f"{symbol} | Price: ₹{current_price} | Signal: {signal}")

        # Calculate ATR and position size
        atr = calculate_atr(df)
        if atr <= 0:
            logging.warning(f"{symbol}: Invalid ATR value (ATR: {atr}), skipping...")
            continue

        quantity = calculate_position_size(account_balance, atr, risk_percentage=1)  # Risk 1% of balance

        # Pass signal, price, and symbol to risk manager
        if signal in ["BUY", "SELL"]:
            place_order(symbol, signal, quantity=quantity, df=df)

    logging.info(f"\n✅ Cycle Complete. Remaining Capital: ₹{remaining_capital:.2f}")
    logging.info(f"📊 Daily PnL: ₹{DAILY_PNL:.2f}")

def main():
    """
    Main function to execute the algo-bot with enhanced components and Zerodha integration.
    """
    # Initialize our enhanced logger
    from utils.logger import Logger
    logger = Logger()
    logger.info("Starting algo-bot with enhanced strategies...")
    
    try:
        # Import enhanced components
        from utils.risk_manager import get_risk_exposure, check_stop_losses_and_take_profits
        from utils.strategy_manager import run_all_strategies_for_stocks
        from order_execution import OrderExecution, OrderType
        from config import (
            TOTAL_CAPITAL,
            SLIPPAGE_TOLERANCE,
            WATCHLIST,
            LOG_LEVEL,
            MAX_RETRIES_ON_ERROR,
            RETRY_DELAY_SECONDS,
            USE_MARKET_ORDERS
        )
        
        # Initialize Kite Connect with better error handling
        # kite = KiteConnect(api_key=os.getenv("ZERODHA_API_KEY"))
        
        # Check if access token is available
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        if not access_token:
            # Generate login URL if no access token
            login_url = kite_manager.get_login_url()
            logger.warning(f"No access token found. Please login using: {login_url}")
            request_token = input("Enter request token after login: ")
            
            # Generate access token
            try:
                session = kite_manager.generate_session(request_token)
                access_token = session["access_token"]
                logger.info(f"Access token generated successfully. Please set as environment variable.")
                print(f"Access Token: {access_token}")
            except Exception as e:
                logger.error(f"Failed to generate access token: {e}")
                return
        else:
            # Set access token
            try:
                kite_manager.set_access_token(access_token)
                logger.info("Successfully authenticated with Zerodha")
            except Exception as e:
                logger.error(f"Failed to set access token: {e}")
                return
        
        # Test authentication
        if not kite_manager.set_access_token(access_token):
            logger.critical("Could not authenticate with Zerodha. Please check your API key and access token.")
            return
        
        # Import SIMULATION_MODE from config or use default value
        try:
            from config import SIMULATION_MODE
        except ImportError:
            logger.warning("SIMULATION_MODE not found in config, defaulting to True (paper trading)")
            SIMULATION_MODE = True
        
        # Initialize order execution with parameters from config
        order_executor = OrderExecution(
            slippage_tolerance=SLIPPAGE_TOLERANCE,
            max_retries=MAX_RETRIES_ON_ERROR,
            retry_delay=RETRY_DELAY_SECONDS,
            use_market_orders=USE_MARKET_ORDERS,
            simulation_mode=SIMULATION_MODE
        )
        
        # Log simulation mode status
        if SIMULATION_MODE:
            logger.info("Running in SIMULATION MODE (paper trading)")
        else:
            logger.info("Running in LIVE TRADING MODE - Real orders will be placed!")
            
        # Schedule trading functions
        def trading_cycle():
            """Run a full trading cycle with enhanced strategies"""
            logger.info("Starting trading cycle")
            
            try:
                # First check existing positions for stop losses/take profits
                closed_positions = check_stop_losses_and_take_profits()
                if closed_positions:
                    logger.info(f"Closed {len(closed_positions)} positions based on SL/TP")
                
                # Get current risk exposure
                risk_metrics = get_risk_exposure()
                logger.info(f"Current risk metrics: {risk_metrics}")
                
                # Run all strategies for watchlist stocks
                results = run_all_strategies_for_stocks(WATCHLIST)
                
                # Log summary of results
                buy_signals = sum(1 for r in results.values() 
                                if r.get('aggregated_signal', {}).get('signal') == 'BUY')
                sell_signals = sum(1 for r in results.values() 
                                 if r.get('aggregated_signal', {}).get('signal') == 'SELL')
                
                logger.info(f"Trading cycle complete. Signals: {buy_signals} buys, {sell_signals} sells")
                
                # Log updated risk exposure
                updated_metrics = get_risk_exposure()
                logger.info(f"Updated risk metrics: {updated_metrics}")
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
        
        # Schedule trading cycle at market open or immediately if during market hours
        current_time = datetime.datetime.now().time()
        market_open = datetime.time(9, 15)  # 9:15 AM
        market_close = datetime.time(15, 30)  # 3:30 PM
        
        if market_open <= current_time <= market_close:
            logger.info("Market is open. Running initial trading cycle...")
            trading_cycle()
        else:
            logger.info("Market is closed. Scheduling for next market open.")
        
        # Schedule regular cycles during market hours
        schedule.every(5).minutes.do(trading_cycle).tag('trading')
        
        # Schedule daily reset for new trading day
        def daily_reset():
            """Reset daily metrics and prepare for new trading day"""
            logger.info("Performing daily reset for new trading day")
            global DAILY_PNL
            DAILY_PNL = 0
            
            # Reset any daily metrics in other modules
            # Other reset operations as needed
            
        schedule.every().day.at("09:00").do(daily_reset)
        
        # Setup market close routine
        def market_close_routine():
            """Execute end-of-day routine at market close"""
            logger.info("Market closing. Running end-of-day routine.")
            
            # Cancel any pending orders
            active_orders = order_executor.get_active_orders()
            for order_id in active_orders:
                if active_orders[order_id]['status'] == 'pending':
                    order_executor.cancel_order(order_id)
            
            # Log final performance for the day
            from utils.risk_manager import get_strategy_performance
            performance = get_strategy_performance()
            logger.info(f"End of day performance: {performance}")
            
            # Clear trading schedule for after-hours
            schedule.clear('trading')
            
        schedule.every().day.at("15:30").do(market_close_routine)
        
        # Main loop
        logger.info("Entering main loop. Press Ctrl+C to exit.")
        print("🚀 Algo Bot Started... Waiting for next run...")
        
        last_reset_day = datetime.date.today()
        
        while True:
            # Check if it's a new day
            today = datetime.date.today()
            if last_reset_day != today:
                logger.info("New day detected. Resetting metrics.")
                daily_reset()
                last_reset_day = today
            
            # Run scheduled tasks
            schedule.run_pending()
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        print("Bot stopped by user.")
    except Exception as e:
        logger.critical(f"Critical error in main loop: {e}")
        print(f"Critical error: {e}")

if __name__ == "__main__":
    main()