# utils/risk_manager.py
"""
Production-Ready Risk Management Module

This module provides comprehensive risk management for algorithmic trading,
including position sizing, stop-loss management, and portfolio risk controls.

Features:
- Dynamic position sizing based on volatility
- Real-time stop-loss and take-profit monitoring
- Portfolio-level risk exposure limits
- Strategy-specific risk controls
- Performance tracking and analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import time
import threading
from collections import defaultdict

# Update to use our enhanced logger
from utils.logger import Logger

try:
    from config import (
        TOTAL_CAPITAL,
        RISK_PER_TRADE_PERCENT,
        ATR_STOP_LOSS_MULTIPLIER,
        ATR_TAKE_PROFIT_MULTIPLIER,
        MAX_STRATEGIES_PER_SYMBOL,
        MAX_RISK_PER_STRATEGY,
        MAX_RISK_PER_SYMBOL,
        MIN_CONFIDENCE_THRESHOLD
    )
except ImportError:
    # Default configuration if config.py is not available
    TOTAL_CAPITAL = 100000
    RISK_PER_TRADE_PERCENT = 2.0
    ATR_STOP_LOSS_MULTIPLIER = 2.0
    ATR_TAKE_PROFIT_MULTIPLIER = 3.0
    MAX_STRATEGIES_PER_SYMBOL = 3
    MAX_RISK_PER_STRATEGY = 20.0
    MAX_RISK_PER_SYMBOL = 10.0
    MIN_CONFIDENCE_THRESHOLD = 60.0

from utils.indicators import calculate_atr

# Initialize logger
logger = Logger(console_output=True, file_output=True)

# Global risk management state
class RiskManager:
    """Centralized risk management system for production trading."""
    
    def __init__(self):
        self.remaining_capital = float(TOTAL_CAPITAL)
        self.active_positions = {}
        self.strategy_exposure = defaultdict(float)
        self.symbol_exposure = defaultdict(float)
        self.trades_history = []
        self.daily_pnl = 0.0
        self.max_daily_loss = float(TOTAL_CAPITAL * 0.05)  # 5% max daily loss
        self.position_lock = threading.Lock()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
    
    def reset_daily_counters(self):
        """Reset daily performance counters."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.start_time = datetime.now()

# Global risk manager instance
risk_manager = RiskManager()

# Backward compatibility functions
def get_active_positions():
    """Get all active positions."""
    return risk_manager.active_positions

def get_strategy_performance(strategy_name=None):
    """
    Get performance metrics for a specific strategy or all strategies.
    
    Parameters:
    - strategy_name: Optional name of strategy to filter for
    
    Returns:
    - dict: Performance metrics
    """
    filtered_trades = risk_manager.trades_history
    if strategy_name:
        filtered_trades = [t for t in risk_manager.trades_history if t['strategy'] == strategy_name]
    
    closed_trades = [t for t in filtered_trades if t['status'] == 'CLOSED']
    
    # If no closed trades, return empty metrics
    if not closed_trades:
        return {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'profit_loss': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'expectancy': 0
        }
    
    # Calculate metrics
    total_trades = len(closed_trades)
    winning_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
    losing_trades = [t for t in closed_trades if t.get('profit_loss', 0) <= 0]
    
    wins = len(winning_trades)
    losses = len(losing_trades)
    
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
    total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
    
    avg_profit = total_profit / wins if wins > 0 else 0
    avg_loss = total_loss / losses if losses > 0 else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    expectancy = ((win_rate / 100) * avg_profit) - ((1 - (win_rate / 100)) * avg_loss)
    
    return {
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_loss': total_profit - total_loss,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy
    }

def get_risk_exposure():
    """
    Get current risk exposure metrics.
    
    Returns:
    - dict: Risk exposure metrics
    """
    return {
        'remaining_capital': risk_manager.remaining_capital,
        'total_exposure': TOTAL_CAPITAL - risk_manager.remaining_capital,
        'exposure_percent': ((TOTAL_CAPITAL - risk_manager.remaining_capital) / TOTAL_CAPITAL) * 100,
        'strategy_exposure': dict(risk_manager.strategy_exposure),
        'symbol_exposure': dict(risk_manager.symbol_exposure),
        'positions_count': len(risk_manager.active_positions),
        'symbols_count': len(risk_manager.symbol_exposure),
        'daily_pnl': risk_manager.daily_pnl
    }

def calculate_stop_loss_take_profit(df, signal_type, current_price, signal_data=None):
    """
    Calculate stop loss and take profit levels using either:
    1. Strategy-provided levels from signal_data
    2. ATR-based calculations if not provided by strategy
    
    Parameters:
    - df: DataFrame with price data
    - signal_type: 'BUY' or 'SELL'
    - current_price: Current price of the asset
    - signal_data: Optional dictionary with stop_loss and take_profit keys
    
    Returns:
    - tuple: (stop_loss_price, take_profit_price)
    """
    # If strategy provides stop loss and take profit, use them
    if isinstance(signal_data, dict):
        if 'stop_loss' in signal_data and signal_data['stop_loss'] is not None:
            strategy_stop_loss = signal_data['stop_loss']
            strategy_take_profit = signal_data.get('take_profit') or signal_data.get('take_profit_1')
            
            # If both are provided, return them
            if strategy_take_profit:
                return strategy_stop_loss, strategy_take_profit
    
    # Calculate ATR if not already in the dataframe
    if 'ATR' not in df.columns:
        df = calculate_atr(df)
    
    # Get latest ATR value
    atr = df['ATR'].iloc[-1]
    
    # Use ATR for stop loss and take profit calculations
    if pd.isna(atr) or atr <= 0:
        # Fallback to percentage-based if ATR is invalid
        stop_loss_distance = current_price * 0.02  # 2% of price
        take_profit_distance = current_price * 0.04  # 4% of price
    else:
        stop_loss_distance = ATR_STOP_LOSS_MULTIPLIER * atr
        take_profit_distance = ATR_TAKE_PROFIT_MULTIPLIER * atr
    
    # Calculate stop loss and take profit prices
    if signal_type == 'BUY':
        stop_loss_price = current_price - stop_loss_distance
        take_profit_price = current_price + take_profit_distance
    else:  # SELL
        stop_loss_price = current_price + stop_loss_distance
        take_profit_price = current_price - take_profit_distance
    
    return stop_loss_price, take_profit_price

def calculate_position_size(signal_type, current_price, stop_loss_price, confidence=None):
    """
    Calculate position size based on risk per trade and confidence.
    
    Parameters:
    - signal_type: 'BUY' or 'SELL'
    - current_price: Current price of the asset
    - stop_loss_price: Stop loss price
    - confidence: Optional confidence score (0-100) from the strategy
    
    Returns:
    - int: Quantity to trade
    """
    global risk_manager
    
    # Adjust risk based on confidence
    if confidence is not None:
        # Scale risk between 50% and 100% of max risk based on confidence
        confidence_factor = 0.5 + (0.5 * confidence / 100)
    else:
        confidence_factor = 1.0
    
    # Calculate risk per trade
    risk_per_trade = (RISK_PER_TRADE_PERCENT / 100) * TOTAL_CAPITAL * confidence_factor
    
    # Calculate stop loss distance
    if signal_type == 'BUY':
        stop_loss_distance = current_price - stop_loss_price
    else:  # SELL
        stop_loss_distance = stop_loss_price - current_price
    
    # Ensure stop loss distance is positive
    stop_loss_distance = abs(stop_loss_distance)
    
    # Calculate quantity based on risk
    if stop_loss_distance > 0:
        quantity = int(risk_per_trade / stop_loss_distance)
    else:
        quantity = 0
    
    # Ensure minimum quantity
    quantity = max(1, quantity)
    
    # Ensure position size doesn't exceed remaining capital
    order_value = quantity * current_price
    if order_value > risk_manager.remaining_capital:
        quantity = int(risk_manager.remaining_capital / current_price)
    
    return quantity

def normalize_signal_data(signal):
    """
    Normalize different signal formats to a standard format.
    
    Parameters:
    - signal: Can be string ('BUY', 'SELL', 'HOLD') or dict with signal info
    
    Returns:
    - dict: Normalized signal data with standard keys
    """
    if isinstance(signal, str):
        return {
            'signal': signal,
            'strength': 100 if signal in ['BUY', 'SELL'] else 0,
            'stop_loss': None,
            'take_profit': None
        }
    
    elif isinstance(signal, dict):
        # Extract signal type
        signal_type = signal.get('signal')
        
        # If no explicit signal key, try to extract from dictionary
        if not signal_type:
            for key in signal:
                if isinstance(signal[key], str) and signal[key] in ['BUY', 'SELL', 'HOLD']:
                    signal_type = signal[key]
                    break
        
        # Extract confidence/strength
        strength = signal.get('strength', signal.get('confidence', 100 if signal_type in ['BUY', 'SELL'] else 0))
        
        return {
            'signal': signal_type,
            'strength': strength,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit') or signal.get('take_profit_1'),
            'divergence': signal.get('divergence'),
            'vwap_distance': signal.get('vwap_distance'),
            'multi_timeframe': signal.get('multi_timeframe')
        }
    
    else:
        return {
            'signal': 'UNKNOWN',
            'strength': 0,
            'stop_loss': None,
            'take_profit': None
        }

def check_portfolio_risk(strategy, symbol, position_size, current_price):
    """
    Check if adding this position would violate portfolio risk limits.
    
    Parameters:
    - strategy: Strategy name
    - symbol: Symbol/ticker
    - position_size: Size of position to add
    - current_price: Current price of asset
    
    Returns:
    - tuple: (can_add, reason_if_not)
    """
    global risk_manager
    
    # Calculate order value
    order_value = position_size * current_price
    
    # Check if adding this position would exceed strategy risk limit
    strategy_exposure_after = risk_manager.strategy_exposure[strategy] + order_value
    if strategy_exposure_after > TOTAL_CAPITAL * (MAX_RISK_PER_STRATEGY / 100):
        return False, f"Would exceed max risk per strategy ({MAX_RISK_PER_STRATEGY}%)"
    
    # Check if adding this position would exceed symbol risk limit
    symbol_exposure_after = risk_manager.symbol_exposure[symbol] + order_value
    if symbol_exposure_after > TOTAL_CAPITAL * (MAX_RISK_PER_SYMBOL / 100):
        return False, f"Would exceed max risk per symbol ({MAX_RISK_PER_SYMBOL}%)"
    
    # Check if we already have max number of strategies per symbol
    symbol_strategies = set()
    for pos_id in risk_manager.active_positions:
        if risk_manager.active_positions[pos_id]['symbol'] == symbol:
            symbol_strategies.add(risk_manager.active_positions[pos_id]['strategy'])
    
    if strategy not in symbol_strategies and len(symbol_strategies) >= MAX_STRATEGIES_PER_SYMBOL:
        return False, f"Already have {MAX_STRATEGIES_PER_SYMBOL} strategies trading this symbol"
    
    # Check daily loss limit
    if risk_manager.daily_pnl < -risk_manager.max_daily_loss:
        return False, f"Daily loss limit exceeded ({risk_manager.max_daily_loss})"
    
    return True, "OK"

def place_order(signal, symbol, strategy_name=None, df=None):
    """
    Enhanced order placement with support for all strategy types.
    
    Parameters:
    - signal: Signal data (string or dictionary)
    - symbol: Symbol/ticker
    - strategy_name: Name of the strategy generating the signal
    - df: Optional DataFrame with price data (will be loaded if not provided)
    
    Returns:
    - dict: Order details if placed, None otherwise
    """
    global risk_manager
    
    with risk_manager.position_lock:
        # Normalize signal data
        signal_data = normalize_signal_data(signal)
        signal_type = signal_data['signal']
        confidence = signal_data['strength']
        
        # Skip if signal is HOLD or confidence below threshold
        if signal_type == 'HOLD' or signal_type == 'INSUFFICIENT_DATA' or confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.info(f"{symbol}: {signal_type} signal with confidence {confidence}, no trade.", strategy_name)
            return None
        
        # Fetch price data if not provided
        if df is None:
            try:
                df = pd.read_csv(f"data/{symbol}.csv")
            except FileNotFoundError:
                logger.error(f"Price data for {symbol} not found.", strategy_name)
                return None
        
        # Get current price
        current_price = float(df['close'].iloc[-1])
        
        # Calculate stop loss and take profit
        stop_loss_price, take_profit_price = calculate_stop_loss_take_profit(
            df, signal_type, current_price, signal_data
        )
        
        # Calculate position size
        quantity = calculate_position_size(signal_type, current_price, stop_loss_price, confidence)
        
        if quantity <= 0:
            logger.warning(f"{symbol}: Quantity zero, risk too low or stop loss too far.", strategy_name)
            return None
        
        # Check portfolio risk limits
        can_place, reason = check_portfolio_risk(strategy_name, symbol, quantity, current_price)
        if not can_place:
            logger.warning(f"{symbol}: Order rejected - {reason}", strategy_name)
            return None
        
        # Calculate order value
        order_value = quantity * current_price
        
        # Ensure we have enough capital
        if order_value > risk_manager.remaining_capital:
            logger.warning(
                f"{symbol}: Not enough capital to place order. Order Value: {order_value}, Remaining Capital: {risk_manager.remaining_capital}", 
                strategy_name
            )
            return None
        
        # Generate unique trade ID
        trade_id = str(uuid.uuid4())[:8]
        
        # Create trade object
        trade = {
            'id': trade_id,
            'timestamp': datetime.now(),
            'strategy': strategy_name,
            'symbol': symbol,
            'type': signal_type,
            'price': current_price,
            'quantity': quantity,
            'value': order_value,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'confidence': confidence,
            'status': 'OPEN'
        }
        
        # Update exposures
        risk_manager.remaining_capital -= order_value
        if strategy_name:
            risk_manager.strategy_exposure[strategy_name] += order_value
        risk_manager.symbol_exposure[symbol] += order_value
        
        # Add to active positions
        risk_manager.active_positions[trade_id] = trade
        
        # Add to trade history
        risk_manager.trades_history.append(trade)
        
        # Update daily counters
        risk_manager.trades_today += 1
        
        # Log the trade
        logger.log_trade(
            strategy_name, 
            symbol, 
            signal_type, 
            current_price, 
            quantity, 
            stop_loss_price, 
            take_profit_price, 
            trade_id
        )
        
        # Log risk metrics
        risk_metrics = {
            'remaining_capital': risk_manager.remaining_capital,
            'total_exposure': TOTAL_CAPITAL - risk_manager.remaining_capital,
            'exposure_percent': ((TOTAL_CAPITAL - risk_manager.remaining_capital) / TOTAL_CAPITAL) * 100,
            'positions_count': len(risk_manager.active_positions),
            'daily_pnl': risk_manager.daily_pnl
        }
        logger.info(f"Risk metrics after trade: {risk_metrics}", strategy_name, risk_metrics)
        
        return trade

def close_position(trade_id, exit_price=None, exit_reason='MANUAL'):
    """
    Close an open position.
    
    Parameters:
    - trade_id: ID of the trade to close
    - exit_price: Price at which to exit (if None, uses current market price)
    - exit_reason: Reason for closing the position (STOP_LOSS, TAKE_PROFIT, MANUAL)
    
    Returns:
    - dict: Trade result details
    """
    global risk_manager
    
    with risk_manager.position_lock:
        if trade_id not in risk_manager.active_positions:
            logger.warning(f"Trade ID {trade_id} not found in active positions")
            return None
        
        trade = risk_manager.active_positions[trade_id]
        symbol = trade['symbol']
        strategy_name = trade['strategy']
        
        # Fetch current price if exit price not provided
        if exit_price is None:
            try:
                df = pd.read_csv(f"data/{symbol}.csv")
                exit_price = float(df['close'].iloc[-1])
            except:
                logger.error(f"Failed to fetch current price for {symbol}, using last known price")
                exit_price = trade['price']
        
        # Calculate P/L
        if trade['type'] == 'BUY':
            profit_loss = (exit_price - trade['price']) * trade['quantity']
            profit_loss_pct = ((exit_price / trade['price']) - 1) * 100
        else:  # SELL
            profit_loss = (trade['price'] - exit_price) * trade['quantity']
            profit_loss_pct = ((trade['price'] / exit_price) - 1) * 100
        
        # Update trade object
        trade['exit_price'] = exit_price
        trade['exit_time'] = datetime.now()
        trade['exit_reason'] = exit_reason
        trade['profit_loss'] = profit_loss
        trade['profit_loss_pct'] = profit_loss_pct
        trade['status'] = 'CLOSED'
        
        # Update capital and exposures
        risk_manager.remaining_capital += (trade['quantity'] * exit_price)
        if strategy_name:
            risk_manager.strategy_exposure[strategy_name] = max(0, risk_manager.strategy_exposure[strategy_name] - trade['value'])
        risk_manager.symbol_exposure[symbol] = max(0, risk_manager.symbol_exposure[symbol] - trade['value'])
        
        # Update daily P/L and counters
        risk_manager.daily_pnl += profit_loss
        if profit_loss > 0:
            risk_manager.wins_today += 1
        else:
            risk_manager.losses_today += 1
        
        # Remove from active positions
        del risk_manager.active_positions[trade_id]
        
        # Log the trade result
        result_type = 'WIN' if profit_loss > 0 else 'LOSS'
        logger.log_trade_result(
            trade_id, 
            result_type, 
            trade['price'], 
            exit_price, 
            profit_loss, 
            profit_loss_pct
        )
        
        return trade

def check_stop_losses_and_take_profits(current_prices=None):
    """
    Check all active positions for stop loss or take profit triggers.
    
    Parameters:
    - current_prices: Optional dict of {symbol: price} to avoid repeated data fetches
    
    Returns:
    - list: List of trades that were closed
    """
    global risk_manager
    closed_trades = []
    
    # Create a copy of keys to avoid dictionary changed during iteration
    position_ids = list(risk_manager.active_positions.keys())
    
    for trade_id in position_ids:
        if trade_id not in risk_manager.active_positions:
            continue  # Position may have been closed already
            
        trade = risk_manager.active_positions[trade_id]
        symbol = trade['symbol']
        
        # Get current price
        if current_prices and symbol in current_prices:
            current_price = current_prices[symbol]
        else:
            try:
                df = pd.read_csv(f"data/{symbol}.csv")
                current_price = float(df['close'].iloc[-1])
                if current_prices is not None:
                    current_prices[symbol] = current_price
            except:
                logger.error(f"Failed to fetch current price for {symbol}, skipping SL/TP check")
                continue
        
        # Check stop loss and take profit conditions
        should_close = False
        reason = None
        
        if trade['type'] == 'BUY':
            if current_price <= trade['stop_loss']:
                should_close = True
                reason = 'STOP_LOSS'
            elif current_price >= trade['take_profit']:
                should_close = True
                reason = 'TAKE_PROFIT'
        elif trade['type'] == 'SELL':
            if current_price >= trade['stop_loss']:
                should_close = True
                reason = 'STOP_LOSS'
            elif current_price <= trade['take_profit']:
                should_close = True
                reason = 'TAKE_PROFIT'
        
        if should_close and reason:
            logger.info(f"{reason} triggered for {trade_id} ({symbol}) at {current_price}", trade['strategy'])
            closed_trade = close_position(trade_id, current_price, reason)
            if closed_trade:
                closed_trades.append(closed_trade)
    
    return closed_trades
