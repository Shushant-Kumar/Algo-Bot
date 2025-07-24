"""
Production-Ready Bollinger Bands Strategy for High-Frequency Intraday Trading

This module provides an optimized, low-latency implementation of the Bollinger Bands
strategy designed for very fast intraday trading with real-time data processing.
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Optional, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

class FastBollingerBandsStrategy:
    """
    High-performance Bollinger Bands strategy optimized for intraday trading.
    
    Features:
    - Rolling window calculations with deque for O(1) operations
    - Cached indicators to avoid recalculation
    - Microsecond precision timestamps
    - Real-time risk management
    - Optimized memory usage
    """
    
    def __init__(self, 
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 sma_period: int = 50,
                 volume_factor: float = 1.5,
                 confirmation_ticks: int = 3,
                 max_position_size: float = 10000.0,
                 max_daily_trades: int = 100,
                 risk_per_trade: float = 0.02):
        """
        Initialize the fast Bollinger Bands strategy.
        
        Parameters:
        - bb_period: Bollinger Bands period
        - bb_std: Standard deviation multiplier
        - rsi_period: RSI calculation period
        - rsi_overbought/oversold: RSI thresholds
        - sma_period: SMA trend filter period
        - volume_factor: Volume surge detection factor
        - confirmation_ticks: Number of ticks to confirm signal
        - max_position_size: Maximum position size
        - max_daily_trades: Daily trade limit
        - risk_per_trade: Risk per trade as fraction of capital
        """
        
        # Strategy parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.sma_period = sma_period
        self.volume_factor = volume_factor
        self.confirmation_ticks = confirmation_ticks
        
        # Risk management
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.risk_per_trade = risk_per_trade
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset_date = pd.Timestamp.now().date()
        
        # Rolling data storage (using deque for O(1) operations)
        max_window = max(bb_period, rsi_period, sma_period) + 50  # Buffer
        self.prices = deque(maxlen=max_window)
        self.volumes = deque(maxlen=max_window)
        self.timestamps = deque(maxlen=max_window)
        
        # Cached indicators
        self.sma_values = deque(maxlen=max_window)
        self.std_values = deque(maxlen=max_window)
        self.upper_band_values = deque(maxlen=max_window)
        self.lower_band_values = deque(maxlen=max_window)
        self.rsi_values = deque(maxlen=max_window)
        self.percent_b_values = deque(maxlen=max_window)
        
        # RSI calculation helpers
        self.gains = deque(maxlen=rsi_period)
        self.losses = deque(maxlen=rsi_period)
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        
        # Signal tracking
        self.last_signal = "HOLD"
        self.signal_strength = 0.0
        self.consecutive_signals = 0
        self.last_signal_time = None
        
        # Performance tracking
        self.calculation_times = deque(maxlen=1000)
        
    def reset_daily_counters(self):
        """Reset daily trading counters."""
        current_date = pd.Timestamp.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def add_tick(self, price: float, volume: float = 0.0, timestamp: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        """
        Add a new price tick and update all indicators efficiently.
        
        Parameters:
        - price: Current price
        - volume: Current volume (optional)
        - timestamp: Tick timestamp (optional, uses current time if None)
        
        Returns:
        - Dictionary with signal, indicators, and metadata
        """
        start_time = time.perf_counter()
        
        # Reset daily counters if needed
        self.reset_daily_counters()
        
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Validate input
        if not isinstance(price, (int, float)) or price <= 0:
            return self._create_response("HOLD", "INVALID_PRICE", 0.0)
        
        # Add new data
        self.prices.append(float(price))
        self.volumes.append(float(volume))
        self.timestamps.append(timestamp)
        
        # Calculate indicators if we have enough data
        if len(self.prices) < self.bb_period:
            return self._create_response("HOLD", "INSUFFICIENT_DATA", 0.0)
        
        # Update all indicators
        self._update_bollinger_bands()
        self._update_rsi()
        self._update_trend_filter()
        
        # Generate trading signal
        signal_result = self._generate_signal()
        
        # Track performance
        calc_time = time.perf_counter() - start_time
        self.calculation_times.append(calc_time)
        
        # Add performance metadata
        signal_result.update({
            'calculation_time_ms': calc_time * 1000,
            'avg_calc_time_ms': np.mean(self.calculation_times) * 1000,
            'data_points': len(self.prices),
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl
        })
        
        return signal_result
    
    def _update_bollinger_bands(self):
        """Update Bollinger Bands indicators efficiently."""
        if len(self.prices) < self.bb_period:
            return
        
        # Calculate SMA using last N prices
        recent_prices = list(self.prices)[-self.bb_period:]
        sma = np.mean(recent_prices)
        self.sma_values.append(sma)
        
        # Calculate standard deviation
        std = np.std(recent_prices, ddof=1)
        self.std_values.append(std)
        
        # Calculate bands
        upper_band = sma + (self.bb_std * std)
        lower_band = sma - (self.bb_std * std)
        
        self.upper_band_values.append(upper_band)
        self.lower_band_values.append(lower_band)
        
        # Calculate %B (position within bands)
        current_price = self.prices[-1]
        band_width = upper_band - lower_band
        if band_width > 0:
            percent_b = (current_price - lower_band) / band_width
        else:
            percent_b = 0.5  # Default to middle if no width
        
        self.percent_b_values.append(percent_b)
    
    def _update_rsi(self):
        """Update RSI using Wilder's smoothing method for efficiency."""
        if len(self.prices) < 2:
            return
        
        # Calculate price change
        price_change = self.prices[-1] - self.prices[-2]
        
        # Separate gains and losses
        gain = max(price_change, 0)
        loss = max(-price_change, 0)
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        if len(self.gains) < self.rsi_period:
            # Initial calculation
            if len(self.gains) == self.rsi_period:
                self.avg_gain = np.mean(self.gains)
                self.avg_loss = np.mean(self.losses)
        else:
            # Wilder's smoothing (more efficient than recalculating)
            self.avg_gain = ((self.avg_gain * (self.rsi_period - 1)) + gain) / self.rsi_period
            self.avg_loss = ((self.avg_loss * (self.rsi_period - 1)) + loss) / self.rsi_period
        
        # Calculate RSI
        if self.avg_loss == 0:
            rsi = 100.0
        elif self.avg_gain == 0:
            rsi = 0.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        self.rsi_values.append(rsi)
    
    def _update_trend_filter(self):
        """Update trend filter (SMA) for trend direction."""
        if len(self.prices) < self.sma_period:
            return
        
        # Calculate trend SMA
        recent_prices = list(self.prices)[-self.sma_period:]
        trend_sma = np.mean(recent_prices)
        return trend_sma
    
    def _generate_signal(self) -> Dict[str, Any]:
        """Generate trading signal based on current indicators."""
        
        # Check if we have enough data
        if (len(self.prices) < max(self.bb_period, self.rsi_period, self.sma_period) or
            len(self.rsi_values) == 0 or len(self.percent_b_values) == 0):
            return self._create_response("HOLD", "INSUFFICIENT_DATA", 0.0)
        
        # Get current values
        current_price = self.prices[-1]
        current_rsi = self.rsi_values[-1]
        current_percent_b = self.percent_b_values[-1]
        current_upper = self.upper_band_values[-1]
        current_lower = self.lower_band_values[-1]
        
        # Get trend direction
        trend_sma = self._update_trend_filter()
        in_uptrend = current_price > trend_sma if trend_sma else True
        
        # Check volume surge
        volume_surge = self._check_volume_surge()
        
        # Risk management checks
        if not self._risk_checks_pass():
            return self._create_response("HOLD", "RISK_LIMIT", 0.0)
        
        # Signal generation logic
        signal = "HOLD"
        reason = "NO_SIGNAL"
        strength = 0.0
        
        # SELL signal conditions
        if (current_percent_b > 1.0 and  # Price above upper band
            current_rsi > self.rsi_overbought and  # RSI overbought
            not in_uptrend and  # Not in strong uptrend
            volume_surge):  # Volume confirmation
            
            # Check for confirmation over multiple ticks
            if self._check_signal_confirmation("SELL"):
                signal = "SELL"
                reason = "BB_RSI_SELL"
                strength = min(100, (current_percent_b - 1) * 50 + (current_rsi - 70) * 2)
        
        # BUY signal conditions
        elif (current_percent_b < 0.0 and  # Price below lower band
              current_rsi < self.rsi_oversold and  # RSI oversold
              in_uptrend and  # In uptrend
              volume_surge):  # Volume confirmation
            
            # Check for confirmation over multiple ticks
            if self._check_signal_confirmation("BUY"):
                signal = "BUY"
                reason = "BB_RSI_BUY"
                strength = min(100, abs(current_percent_b) * 50 + (30 - current_rsi) * 2)
        
        # Update signal tracking
        self._update_signal_tracking(signal)
        
        return self._create_response(signal, reason, strength)
    
    def _check_volume_surge(self) -> bool:
        """Check if current volume indicates a surge."""
        if len(self.volumes) < 20:
            return True  # Default to True if insufficient volume data
        
        recent_volumes = list(self.volumes)[-20:]
        avg_volume = np.mean(recent_volumes)
        current_volume = self.volumes[-1]
        
        return current_volume > (avg_volume * self.volume_factor)
    
    def _check_signal_confirmation(self, signal_type: str) -> bool:
        """Check if signal is confirmed over multiple ticks."""
        if self.last_signal == signal_type:
            self.consecutive_signals += 1
        else:
            self.consecutive_signals = 1
            self.last_signal = signal_type
        
        return self.consecutive_signals >= self.confirmation_ticks
    
    def _risk_checks_pass(self) -> bool:
        """Perform real-time risk management checks."""
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        # Check if too soon since last signal
        if (self.last_signal_time and 
            (pd.Timestamp.now() - self.last_signal_time).total_seconds() < 1.0):
            return False
        
        # Additional risk checks can be added here
        return True
    
    def _update_signal_tracking(self, signal: str):
        """Update signal tracking variables."""
        if signal in ["BUY", "SELL"]:
            self.last_signal_time = pd.Timestamp.now()
            if signal != "HOLD":
                self.daily_trades += 1
    
    def _create_response(self, signal: str, reason: str, strength: float) -> Dict[str, Any]:
        """Create standardized response dictionary."""
        
        # Get current indicator values
        current_values = {}
        if self.prices:
            current_values['price'] = self.prices[-1]
        if self.rsi_values:
            current_values['rsi'] = self.rsi_values[-1]
        if self.percent_b_values:
            current_values['percent_b'] = self.percent_b_values[-1]
        if self.upper_band_values and self.lower_band_values:
            current_values['upper_band'] = self.upper_band_values[-1]
            current_values['lower_band'] = self.lower_band_values[-1]
        if self.sma_values:
            current_values['sma'] = self.sma_values[-1]
        
        return {
            'signal': signal,
            'reason': reason,
            'strength': strength,
            'timestamp': pd.Timestamp.now(),
            'indicators': current_values,
            'consecutive_signals': self.consecutive_signals,
            'can_trade': self._risk_checks_pass()
        }
    
    def get_position_size(self, price: float, account_balance: float, atr = None) -> float:
        """
        Calculate optimal position size based on risk management.
        
        Parameters:
        - price: Current stock price
        - account_balance: Available account balance
        - atr: Average True Range for volatility-based sizing
        
        Returns:
        - Recommended position size
        """
        
        # Calculate base position size
        risk_amount = account_balance * self.risk_per_trade
        
        # Use ATR for stop loss if available
        if atr:
            stop_distance = atr * 2  # 2x ATR stop loss
            shares = risk_amount / stop_distance
        else:
            # Use 2% of price as stop loss if no ATR
            stop_distance = price * 0.02
            shares = risk_amount / stop_distance
        
        # Apply maximum position size limit
        max_shares_by_limit = self.max_position_size / price
        
        # Return the smaller of calculated size and maximum limit
        return min(shares, max_shares_by_limit)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'avg_calculation_time_ms': np.mean(self.calculation_times) * 1000 if self.calculation_times else 0,
            'max_calculation_time_ms': np.max(self.calculation_times) * 1000 if self.calculation_times else 0,
            'data_points_stored': len(self.prices),
            'last_signal': self.last_signal,
            'consecutive_signals': self.consecutive_signals
        }

# Convenience function for backward compatibility
def fast_bollinger_bands_strategy(price: float, volume: float = 0.0, 
                                 strategy_instance: Optional[FastBollingerBandsStrategy] = None,
                                 **kwargs) -> str:
    """
    Fast Bollinger Bands strategy function for compatibility with existing code.
    
    Parameters:
    - price: Current stock price
    - volume: Current volume
    - strategy_instance: Existing strategy instance (recommended for performance)
    - **kwargs: Strategy parameters
    
    Returns:
    - Trading signal: "BUY", "SELL", or "HOLD"
    """
    
    if strategy_instance is None:
        strategy_instance = FastBollingerBandsStrategy(**kwargs)
    
    result = strategy_instance.add_tick(price, volume)
    return result['signal']
