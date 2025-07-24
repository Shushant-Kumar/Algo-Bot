"""
Production-Ready RSI Strategy for High-Frequency Intraday Trading

This module provides an optimized, low-latency implementation of the RSI
strategy with divergence detection for very fast intraday trading.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any
import time
from strategies.fast_strategy_base import FastStrategyBase

class FastRSIStrategy(FastStrategyBase):
    """
    High-performance RSI strategy optimized for intraday trading.
    
    Features:
    - Rolling RSI calculation with O(1) operations
    - Real-time divergence detection
    - MACD confirmation filter
    - Volume surge detection
    - Microsecond precision timestamps
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 divergence_lookback: int = 20,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 volume_factor: float = 1.5,
                 **kwargs):
        """
        Initialize the fast RSI strategy.
        
        Parameters:
        - rsi_period: RSI calculation period
        - rsi_overbought/oversold: RSI thresholds
        - divergence_lookback: Bars to look back for divergence
        - macd_fast/slow/signal: MACD parameters
        - volume_factor: Volume surge detection factor
        """
        super().__init__(**kwargs)
        
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.divergence_lookback = divergence_lookback
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.volume_factor = volume_factor
        
        # Rolling data for calculations
        self.price_changes = deque(maxlen=rsi_period * 2)
        self.gains = deque(maxlen=rsi_period)
        self.losses = deque(maxlen=rsi_period)
        self.rsi_values = deque(maxlen=divergence_lookback * 2)
        
        # MACD components
        self.ema_fast = None
        self.ema_slow = None
        self.macd_line = deque(maxlen=100)
        self.signal_line = deque(maxlen=100)
        
        # Divergence tracking
        self.price_pivots = []
        self.rsi_pivots = []
        
        # Volume analysis
        self.volume_sma = deque(maxlen=20)
        
    def add_tick(self, price: float, volume: int, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a new market tick and generate trading signals.
        
        Parameters:
        - price: Current price
        - volume: Current volume
        - timestamp: Optional timestamp
        
        Returns:
        - Dictionary with signal and metadata
        """
        start_time = time.perf_counter()
        
        # Update base data
        self._update_base_data(price, volume, timestamp)
        
        # Initialize result
        result = {
            'signal': 'HOLD',
            'strength': 0.0,
            'rsi': 50.0,
            'macd': 0.0,
            'divergence': False,
            'volume_surge': False,
            'price': price,
            'volume': volume,
            'calculation_time_ms': 0.0
        }
        
        # Need minimum data for calculations
        if len(self.prices) < max(self.rsi_period, self.macd_slow) + 1:
            result['calculation_time_ms'] = (time.perf_counter() - start_time) * 1000
            return result
        
        # Calculate RSI
        rsi = self._calculate_rsi_optimized()
        result['rsi'] = rsi
        self.rsi_values.append(rsi)
        
        # Calculate MACD
        macd_value = self._calculate_macd_optimized()
        result['macd'] = macd_value
        
        # Check volume surge
        volume_surge = self._check_volume_surge(volume)
        result['volume_surge'] = volume_surge
        
        # Detect divergence
        divergence = self._detect_divergence_fast()
        result['divergence'] = divergence
        
        # Generate signal
        signal, strength = self._generate_signal(rsi, macd_value, divergence, volume_surge)
        
        # Confirm signal
        if self._confirm_signal(signal):
            result['signal'] = signal
            result['strength'] = strength
        
        # Performance tracking
        calc_time = (time.perf_counter() - start_time) * 1000
        if hasattr(self, 'calculation_times') and self.calculation_times is not None:
            self.calculation_times.append(calc_time)
        result['calculation_time_ms'] = calc_time
        
        return result
    
    def _calculate_rsi_optimized(self) -> float:
        """Calculate RSI using optimized rolling calculations."""
        if len(self.prices) < 2:
            return 50.0
        
        # Calculate price change
        current_change = self.prices[-1] - self.prices[-2]
        self.price_changes.append(current_change)
        
        # Update gains and losses
        if current_change > 0:
            self.gains.append(current_change)
            self.losses.append(0.0)
        else:
            self.gains.append(0.0)
            self.losses.append(abs(current_change))
        
        # Need enough data for RSI
        if len(self.gains) < self.rsi_period:
            return 50.0
        
        # Calculate average gain and loss
        avg_gain = np.mean(list(self.gains)[-self.rsi_period:])
        avg_loss = np.mean(list(self.losses)[-self.rsi_period:])
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return float(rsi)
    
    def _calculate_macd_optimized(self) -> float:
        """Calculate MACD using optimized EMA calculations."""
        current_price = self.prices[-1]
        
        # Initialize EMAs on first calculation
        if self.ema_fast is None:
            if len(self.prices) >= self.macd_fast:
                self.ema_fast = np.mean(list(self.prices)[-self.macd_fast:])
            else:
                self.ema_fast = current_price
        
        if self.ema_slow is None:
            if len(self.prices) >= self.macd_slow:
                self.ema_slow = np.mean(list(self.prices)[-self.macd_slow:])
            else:
                self.ema_slow = current_price
        
        # Update EMAs
        alpha_fast = 2.0 / (self.macd_fast + 1)
        alpha_slow = 2.0 / (self.macd_slow + 1)
        
        self.ema_fast = alpha_fast * current_price + (1 - alpha_fast) * self.ema_fast
        self.ema_slow = alpha_slow * current_price + (1 - alpha_slow) * self.ema_slow
        
        # Calculate MACD line
        macd = self.ema_fast - self.ema_slow
        self.macd_line.append(macd)
        
        # Calculate signal line (EMA of MACD)
        if len(self.macd_line) >= self.macd_signal:
            if not self.signal_line:
                signal = np.mean(list(self.macd_line)[-self.macd_signal:])
            else:
                alpha_signal = 2.0 / (self.macd_signal + 1)
                signal = alpha_signal * macd + (1 - alpha_signal) * self.signal_line[-1]
            
            self.signal_line.append(signal)
            return macd - signal  # MACD histogram
        
        return 0.0
    
    def _check_volume_surge(self, current_volume: int) -> bool:
        """Check for volume surge."""
        self.volume_sma.append(current_volume)
        
        if len(self.volume_sma) < 10:
            return False
        
        avg_volume = np.mean(list(self.volume_sma)[:-1])  # Exclude current volume
        return bool(current_volume > (avg_volume * self.volume_factor))
    
    def _detect_divergence_fast(self) -> bool:
        """Fast divergence detection using recent data."""
        if len(self.prices) < self.divergence_lookback or len(self.rsi_values) < self.divergence_lookback:
            return False
        
        # Get recent data
        recent_prices = list(self.prices)[-self.divergence_lookback:]
        recent_rsi = list(self.rsi_values)[-self.divergence_lookback:]
        
        # Simple divergence check: compare first half vs second half
        mid_point = len(recent_prices) // 2
        
        price_trend = recent_prices[-1] - recent_prices[mid_point]
        rsi_trend = recent_rsi[-1] - recent_rsi[mid_point]
        
        # Bullish divergence: price down, RSI up
        # Bearish divergence: price up, RSI down
        divergence_threshold = 2.0
        
        bullish_div = price_trend < 0 and rsi_trend > divergence_threshold
        bearish_div = price_trend > 0 and rsi_trend < -divergence_threshold
        
        return bullish_div or bearish_div
    
    def _generate_signal(self, rsi: float, macd: float, divergence: bool, volume_surge: bool) -> Tuple[str, float]:
        """
        Generate trading signal based on all indicators.
        
        Returns:
        - Tuple of (signal, strength)
        """
        signal = 'HOLD'
        strength = 0.0
        
        # Base RSI signals
        if rsi < self.rsi_oversold:
            signal = 'BUY'
            strength = (self.rsi_oversold - rsi) / self.rsi_oversold * 100
        elif rsi > self.rsi_overbought:
            signal = 'SELL'
            strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought) * 100
        
        # MACD confirmation
        if signal == 'BUY' and macd < 0:
            strength *= 0.7  # Reduce strength if MACD doesn't confirm
        elif signal == 'SELL' and macd > 0:
            strength *= 0.7
        elif signal == 'BUY' and macd > 0:
            strength *= 1.3  # Increase strength if MACD confirms
        elif signal == 'SELL' and macd < 0:
            strength *= 1.3
        
        # Divergence boost
        if divergence:
            if signal == 'BUY':
                strength *= 1.5
            elif signal == 'SELL':
                strength *= 1.5
        
        # Volume confirmation
        if volume_surge and signal != 'HOLD':
            strength *= 1.2
        
        # Cap strength at 100%
        strength = min(strength, 100.0)
        
        return signal, strength
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy state information."""
        return {
            'strategy_type': 'FastRSI',
            'rsi_period': self.rsi_period,
            'current_rsi': self.rsi_values[-1] if self.rsi_values else 50.0,
            'current_macd': self.macd_line[-1] if self.macd_line else 0.0,
            'data_points': len(self.prices),
            'ready_for_signals': len(self.prices) >= max(self.rsi_period, self.macd_slow) + 1
        }
