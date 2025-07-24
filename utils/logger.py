"""
Production-Ready Logging Module

This module provides comprehensive logging capabilities for algorithmic trading systems,
including trade logging, performance tracking, and error monitoring.

Features:
- Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- JSON and text format support
- Trade-specific logging with performance metrics
- Thread-safe operations
- Automatic log rotation and archiving
- Real-time performance monitoring
"""

import os
import json
from datetime import datetime, timedelta
from enum import Enum
import traceback
import threading
from collections import deque
import time

class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class Logger:
    """Enhanced logging system for production trading applications."""
    
    def __init__(self, log_dir='logs', console_output=True, file_output=True, 
                 json_format=False, log_level=LogLevel.INFO, max_log_size_mb=50):
        """
        Initialize logger with enhanced configuration options.
        
        Parameters:
        - log_dir: Directory to store log files
        - console_output: Whether to print logs to console
        - file_output: Whether to write logs to file
        - json_format: Whether to format logs as JSON
        - log_level: Minimum log level to record
        - max_log_size_mb: Maximum log file size before rotation
        """
        self.log_dir = log_dir
        self.console_output = console_output
        self.file_output = file_output
        self.json_format = json_format
        self.log_level = log_level
        self.max_log_size = max_log_size_mb * 1024 * 1024  # Convert to bytes
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.recent_logs = deque(maxlen=1000)  # Keep last 1000 log entries
        self.start_time = time.time()
        
        # Create log directory if it doesn't exist
        if self.file_output:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create a trades directory for trade logs
        self.trades_dir = os.path.join(log_dir, 'trades')
        if self.file_output:
            os.makedirs(self.trades_dir, exist_ok=True)
        
        # Store performance metrics
        self.performance = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0,
            'start_time': datetime.now(),
            'last_trade_time': None
        }
    
    def _rotate_log_if_needed(self, log_file):
        """Rotate log file if it exceeds maximum size."""
        try:
            if os.path.exists(log_file) and os.path.getsize(log_file) > self.max_log_size:
                # Create backup filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = f"{log_file}.{timestamp}.bak"
                os.rename(log_file, backup_file)
                
                # Keep only last 5 backup files
                self._cleanup_old_backups(log_file)
        except Exception as e:
            print(f"Error rotating log file: {e}")
    
    def _cleanup_old_backups(self, log_file, keep_count=5):
        """Remove old backup files, keeping only the most recent ones."""
        try:
            log_dir = os.path.dirname(log_file)
            base_name = os.path.basename(log_file)
            
            # Find all backup files
            backup_files = []
            for filename in os.listdir(log_dir):
                if filename.startswith(base_name) and filename.endswith('.bak'):
                    backup_path = os.path.join(log_dir, filename)
                    backup_files.append((backup_path, os.path.getmtime(backup_path)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            for backup_path, _ in backup_files[keep_count:]:
                os.remove(backup_path)
                
        except Exception as e:
            print(f"Error cleaning up backup files: {e}")
    
    def _get_log_file(self, strategy=None):
        """Get the appropriate log file path based on date and strategy."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        if strategy:
            return os.path.join(self.log_dir, f"{strategy}_log_{date_str}.txt")
        return os.path.join(self.log_dir, f"log_{date_str}.txt")
    
    def _format_message(self, level, message, strategy=None, extra=None):
        """Format log message based on configuration."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if self.json_format:
            log_data = {
                'timestamp': timestamp,
                'level': level.name,
                'message': message,
            }
            if strategy:
                log_data['strategy'] = strategy
            if extra:
                log_data.update(extra)
            return json.dumps(log_data)
        else:
            strategy_prefix = f"[{strategy}] " if strategy else ""
            return f"[{timestamp}] [{level.name}] {strategy_prefix}{message}"
    
    def _write_log(self, level, message, strategy=None, extra=None):
        """Write log to file and/or console based on configuration."""
        if level.value < self.log_level.value:
            return
        
        with self.lock:
            formatted_message = self._format_message(level, message, strategy, extra)
            
            # Add to recent logs for monitoring
            log_entry = {
                'timestamp': time.time(),
                'level': level.name,
                'message': message,
                'strategy': strategy
            }
            self.recent_logs.append(log_entry)
            
            # Print to console if enabled
            if self.console_output:
                print(formatted_message)
            
            # Write to file if enabled
            if self.file_output:
                try:
                    log_file = self._get_log_file(strategy)
                    self._rotate_log_if_needed(log_file)
                    
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{formatted_message}\n")
                except Exception as e:
                    print(f"Error writing to log file: {e}")
    
    def debug(self, message, strategy=None, extra=None):
        """Log a debug message."""
        self._write_log(LogLevel.DEBUG, message, strategy, extra)
    
    def info(self, message, strategy=None, extra=None):
        """Log an info message."""
        self._write_log(LogLevel.INFO, message, strategy, extra)
    
    def warning(self, message, strategy=None, extra=None):
        """Log a warning message."""
        self._write_log(LogLevel.WARNING, message, strategy, extra)
    
    def error(self, message, strategy=None, extra=None):
        """Log an error message."""
        self._write_log(LogLevel.ERROR, message, strategy, extra)
        # Add stack trace for errors
        if self.log_level.value <= LogLevel.DEBUG.value:
            self._write_log(LogLevel.DEBUG, f"Stack trace:\n{traceback.format_exc()}", strategy)
    
    def critical(self, message, strategy=None, extra=None):
        """Log a critical message."""
        self._write_log(LogLevel.CRITICAL, message, strategy, extra)
        # Always add stack trace for critical errors
        self._write_log(LogLevel.CRITICAL, f"Stack trace:\n{traceback.format_exc()}", strategy)
    
    def log_signal(self, strategy, symbol, signal_type, confidence=None, 
                   price=None, indicators=None):
        """
        Log a trading signal generated by a strategy.
        
        Parameters:
        - strategy: Name of the strategy generating the signal
        - symbol: Trading symbol/ticker
        - signal_type: Type of signal (BUY, SELL, HOLD)
        - confidence: Signal confidence score (0-100)
        - price: Current price when signal was generated
        - indicators: Dict of indicator values that triggered the signal
        """
        extra = {
            'symbol': symbol,
            'signal_type': signal_type,
            'price': price,
        }
        
        if confidence is not None:
            extra['confidence'] = confidence
        
        if indicators:
            extra['indicators'] = indicators
        
        signal_msg = f"Signal: {signal_type} {symbol}"
        if confidence is not None:
            signal_msg += f" (Confidence: {confidence}%)"
        
        self.info(signal_msg, strategy, extra)
    
    def log_trade(self, strategy, symbol, trade_type, price, quantity, 
                  stop_loss=None, take_profit=None, trade_id=None):
        """
        Log a trade execution.
        
        Parameters:
        - strategy: Name of the strategy executing the trade
        - symbol: Trading symbol/ticker
        - trade_type: Type of trade (BUY, SELL)
        - price: Execution price
        - quantity: Trade quantity
        - stop_loss: Stop loss price level
        - take_profit: Take profit price level
        - trade_id: Unique identifier for the trade
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_id = trade_id or f"{symbol}_{trade_type}_{timestamp}"
        
        trade_data = {
            'timestamp': timestamp,
            'strategy': strategy,
            'symbol': symbol,
            'trade_type': trade_type,
            'price': price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trade_id': trade_id
        }
        
        # Update trade counter
        self.performance['trades'] += 1
        
        # Log to the main log
        self.info(
            f"Trade: {trade_type} {quantity} {symbol} @ {price}",
            strategy,
            trade_data
        )
        
        # Write to trades log file
        if self.file_output:
            trades_file = os.path.join(self.trades_dir, f"{strategy}_trades.json")
            
            # Read existing trades
            existing_trades = []
            if os.path.exists(trades_file):
                try:
                    with open(trades_file, 'r') as f:
                        existing_trades = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupt, start fresh
                    existing_trades = []
            
            # Append new trade
            existing_trades.append(trade_data)
            
            # Write back to file
            with open(trades_file, 'w') as f:
                json.dump(existing_trades, f, indent=2)
    
    def log_trade_result(self, trade_id, result_type, entry_price, exit_price, 
                         profit_loss, profit_loss_pct):
        """
        Log the result of a closed trade.
        
        Parameters:
        - trade_id: ID of the original trade
        - result_type: WIN or LOSS
        - entry_price: Original entry price
        - exit_price: Exit price
        - profit_loss: Absolute profit/loss amount
        - profit_loss_pct: Percentage profit/loss
        """
        trade_result = {
            'trade_id': trade_id,
            'result': result_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Update performance metrics
        if result_type == 'WIN':
            self.performance['wins'] += 1
        else:
            self.performance['losses'] += 1
        
        self.performance['profit_loss'] += profit_loss
        
        # Log the trade result
        result_msg = f"Trade Result: {result_type} on {trade_id}, P/L: {profit_loss} ({profit_loss_pct}%)"
        self.info(result_msg, extra=trade_result)
    
    def log_performance(self, strategy=None, reset=False):
        """
        Log current performance metrics.
        
        Parameters:
        - strategy: Strategy name to include in the log
        - reset: Whether to reset metrics after logging
        """
        win_rate = 0
        if self.performance['trades'] > 0:
            win_rate = (self.performance['wins'] / self.performance['trades']) * 100
        
        performance_msg = (
            f"Performance Summary - "
            f"Trades: {self.performance['trades']}, "
            f"Wins: {self.performance['wins']}, "
            f"Losses: {self.performance['losses']}, "
            f"Win Rate: {win_rate:.2f}%, "
            f"Net P/L: {self.performance['profit_loss']}"
        )
        
        self.info(performance_msg, strategy, self.performance)
        
        if reset:
            self.performance = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit_loss': 0.0,
            }

# For backward compatibility
def log(message, level=LogLevel.INFO, strategy=None):
    logger = Logger()
    if level == LogLevel.DEBUG:
        logger.debug(message, strategy)
    elif level == LogLevel.INFO:
        logger.info(message, strategy)
    elif level == LogLevel.WARNING:
        logger.warning(message, strategy)
    elif level == LogLevel.ERROR:
        logger.error(message, strategy)
    elif level == LogLevel.CRITICAL:
        logger.critical(message, strategy)
