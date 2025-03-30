import os
import pandas as pd
from strategies.rsi_strategy import rsi_strategy
from utils.risk_manager import place_order, remaining_capital
import schedule
import time
import datetime
from kiteconnect import KiteConnect
from utils.strategy_manager import run_all_strategies_for_stocks
from config import PER_STOCK_ALLOCATION
from ta.volatility import AverageTrueRange  # Add this import for ATR calculation
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
kite = KiteConnect(api_key=os.getenv("ZERODHA_API_KEY"))
kite.set_access_token(os.getenv("ZERODHA_ACCESS_TOKEN"))

def get_account_balance():
    """
    Fetch account balance from Zerodha API.
    """
    try:
        funds = kite.margins("equity")
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

        # Place limit order
        order_id = kite.place_order(
            tradingsymbol=stock_symbol,
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
        # Fetch live market data
        quote = kite.quote(f"NSE:{stock_symbol}")
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

# Run every 5 mins
schedule.every(5).minutes.do(run_bot)

print("🚀 Algo Bot Started... Waiting for next run...")

# Daily PnL Reset (Optional - we'll improve this later)
last_reset_day = None

while True:
    today = datetime.date.today()
    if last_reset_day != today:
        print("\n🔁 New Day! Resetting Capital and PnL\n")
        from config import TOTAL_CAPITAL
        from utils import risk_manager
        risk_manager.remaining_capital = TOTAL_CAPITAL
        last_reset_day = today
        DAILY_PNL = 0  # Reset daily PnL

    schedule.run_pending()
    time.sleep(1)

def main():
    """
    Main function to execute the algo-bot.
    """
    logging.info("Starting algo-bot...")
    try:
        # Initialize order execution with slippage tolerance
        order_executor = OrderExecution(slippage_tolerance=SLIPPAGE_TOLERANCE)

        # Example usage
        ltp = 100  # Last traded price
        market_depth = [(100, 50), (101, 30), (102, 20)]  # Price levels with quantities
        order_executor.execute_order('buy', 60, ltp, market_depth)

        real_time_allocation = calculate_real_time_allocation()
        logging.info(f"Real-time allocation: {real_time_allocation}")
        print(f"Total Capital: {TOTAL_CAPITAL}")
        print(f"Real-time Allocation: {real_time_allocation}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()