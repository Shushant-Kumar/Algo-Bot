# utils/risk_manager.py
import pandas as pd
import logging

from config import (
    TOTAL_CAPITAL,
    RISK_PER_TRADE_PERCENT,
    ATR_STOP_LOSS_MULTIPLIER,
    ATR_TAKE_PROFIT_MULTIPLIER
)
from utils.indicators import calculate_atr

remaining_capital = TOTAL_CAPITAL

def place_order(signal, current_price, symbol):
    global remaining_capital

    # Simulate fetching data for the stock
    df = pd.read_csv(f"data/{symbol}.csv")
    df = calculate_atr(df)

    if signal == 'HOLD':
        logging.info(f"{symbol}: HOLD signal, no trade.")
        return

    # Get latest price & ATR
    current_price = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1]  # Store ATR value in a variable

    if pd.isna(atr) or atr <= 0:
        logging.warning(f"{symbol}: ATR not ready or invalid (ATR: {atr}), skipping...")
        return

    # ATR-based Stop Loss distance
    stop_loss_distance = ATR_STOP_LOSS_MULTIPLIER * atr
    take_profit_distance = ATR_TAKE_PROFIT_MULTIPLIER * atr

    # Risk per trade (₹)
    risk_per_trade = (RISK_PER_TRADE_PERCENT / 100) * TOTAL_CAPITAL

    # Qty = risk / stop_loss_distance
    quantity = int(risk_per_trade / stop_loss_distance)

    if quantity == 0:
        logging.warning(f"{symbol}: Quantity zero, risk too low or ATR too high.")
        return

    # Stop Loss & Take Profit Prices
    if signal == 'BUY':
        stop_loss_price = current_price - stop_loss_distance
        take_profit_price = current_price + take_profit_distance
    elif signal == 'SELL':
        stop_loss_price = current_price + stop_loss_distance
        take_profit_price = current_price - take_profit_distance

    # Check if we have enough capital
    order_value = quantity * current_price
    if order_value > remaining_capital:
        logging.warning(f"{symbol}: Not enough capital to place order. Order Value: {order_value}, Remaining Capital: {remaining_capital}")
        return

    # Simulate placing order
    logging.info(f"{symbol}: {signal} Order - Qty: {quantity} @ ₹{current_price:.2f}")
    logging.info(f"{symbol}: SL: ₹{stop_loss_price:.2f} | TP: ₹{take_profit_price:.2f}")
    remaining_capital -= order_value
    logging.info(f"Remaining Capital: ₹{remaining_capital:.2f}")
