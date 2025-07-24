"""
Production-Ready MACD Strategy for High-Frequency Intraday Trading

This module provides an optimized, low-latency implementation of the MACD
strategy with signal line crossovers for very fast intraday trading.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any
import time
from strategies.fast_strategy_base import FastStrategyBase

class FastMACDStrategy(FastStrategyBase):
    """
    High-performance MACD strategy optimized for intraday trading.
    
    Features:
    - Rolling MACD calculation with O(1) operations
    - Signal line crossover detection
    - RSI filter for trade confirmation
    - Volume surge detection
    - Real-time histogram analysis
    - Microsecond precision timestamps
    """
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 volume_factor: float = 1.5,
                 histogram_threshold: float = 0.001,
                 **kwargs):
        """
        Initialize the fast MACD strategy.
        
        Parameters:
        - fast_period: Fast EMA period
        - slow_period: Slow EMA period
        - signal_period: Signal line EMA period
        - rsi_period: RSI filter period
        - rsi_overbought/oversold: RSI thresholds
        - volume_factor: Volume surge detection factor
        - histogram_threshold: Minimum histogram value for signals
        """
        super().__init__(**kwargs)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_factor = volume_factor
        self.histogram_threshold = histogram_threshold
        
        # MACD components
        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None
        self.macd_line = deque(maxlen=100)
        self.signal_line = deque(maxlen=100)
        self.histogram = deque(maxlen=100)
        
        # Crossover tracking
        self.signal_crossovers = deque(maxlen=10)
        self.zero_crossovers = deque(maxlen=10)
        
        # Volume analysis
        self.volume_sma = deque(maxlen=20)
        
        # Divergence tracking
        self.price_pivots = deque(maxlen=20)
        self.macd_pivots = deque(maxlen=20)
        
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
            'macd': 0.0,
            'signal_line': 0.0,
            'histogram': 0.0,
            'rsi': 50.0,
            'volume_surge': False,
            'signal_crossover': False,
            'zero_crossover': False,
            'divergence': False,
            'price': price,
            'volume': volume,
            'calculation_time_ms': 0.0
        }
        
        # Need minimum data for calculations
        if len(self.prices) < self.slow_period:
            result['calculation_time_ms'] = (time.perf_counter() - start_time) * 1000
            return result
        
        # Calculate MACD components
        macd_value = self._calculate_macd(price)
        signal_value = self._calculate_signal_line(macd_value)
        histogram_value = macd_value - signal_value
        
        result['macd'] = macd_value
        result['signal_line'] = signal_value
        result['histogram'] = histogram_value
        
        # Store values
        self.macd_line.append(macd_value)
        self.signal_line.append(signal_value)
        self.histogram.append(histogram_value)
        
        # Calculate RSI for filter
        rsi = self._calculate_rsi_fast()
        result['rsi'] = rsi
        
        # Check volume surge
        volume_surge = self._check_volume_surge(volume)
        result['volume_surge'] = volume_surge
        
        # Detect crossovers
        signal_crossover = self._detect_signal_crossover(macd_value, signal_value)
        zero_crossover = self._detect_zero_crossover(macd_value)
        result['signal_crossover'] = signal_crossover
        result['zero_crossover'] = zero_crossover
        
        # Detect divergence
        divergence = self._detect_macd_divergence()
        result['divergence'] = divergence
        
        # Generate signal
        signal, strength = self._generate_signal(
            macd_value, signal_value, histogram_value, rsi, 
            signal_crossover, zero_crossover, divergence, volume_surge
        )
        
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
    
    def _calculate_macd(self, price: float) -> float:
        """Calculate MACD line using optimized EMA calculations."""
        # Initialize fast EMA
        if self.fast_ema is None:
            if len(self.prices) >= self.fast_period:
                self.fast_ema = float(np.mean(list(self.prices)[-self.fast_period:]))
            else:
                self.fast_ema = price
        
        # Initialize slow EMA
        if self.slow_ema is None:
            if len(self.prices) >= self.slow_period:
                self.slow_ema = float(np.mean(list(self.prices)[-self.slow_period:]))
            else:
                self.slow_ema = price
        
        # Update EMAs
        alpha_fast = 2.0 / (self.fast_period + 1)
        alpha_slow = 2.0 / (self.slow_period + 1)
        
        self.fast_ema = alpha_fast * price + (1 - alpha_fast) * self.fast_ema
        self.slow_ema = alpha_slow * price + (1 - alpha_slow) * self.slow_ema
        
        # Calculate MACD line
        return self.fast_ema - self.slow_ema
    
    def _calculate_signal_line(self, macd_value: float) -> float:
        """Calculate signal line (EMA of MACD)."""
        if self.signal_ema is None:
            if len(self.macd_line) >= self.signal_period:
                self.signal_ema = float(np.mean(list(self.macd_line)[-self.signal_period:]))
            else:
                self.signal_ema = macd_value
            return self.signal_ema
        
        alpha_signal = 2.0 / (self.signal_period + 1)
        self.signal_ema = alpha_signal * macd_value + (1 - alpha_signal) * self.signal_ema
        return self.signal_ema
    
    def _check_volume_surge(self, current_volume: int) -> bool:
        """Check for volume surge."""
        self.volume_sma.append(current_volume)
        
        if len(self.volume_sma) < 10:
            return False
        
        avg_volume = float(np.mean(list(self.volume_sma)[:-1]))
        return current_volume > (avg_volume * self.volume_factor)
    
    def _detect_signal_crossover(self, macd_value: float, signal_value: float) -> bool:
        """Detect MACD signal line crossover."""
        if len(self.signal_crossovers) == 0:
            self.signal_crossovers.append(macd_value > signal_value)
            return False
        
        current_position = macd_value > signal_value
        previous_position = self.signal_crossovers[-1]
        
        self.signal_crossovers.append(current_position)
        
        return current_position != previous_position
    
    def _detect_zero_crossover(self, macd_value: float) -> bool:
        """Detect MACD zero line crossover."""
        if len(self.zero_crossovers) == 0:
            self.zero_crossovers.append(macd_value > 0)
            return False
        
        current_position = macd_value > 0
        previous_position = self.zero_crossovers[-1]
        
        self.zero_crossovers.append(current_position)
        
        return current_position != previous_position
    
    def _detect_macd_divergence(self) -> bool:
        """Detect divergence between price and MACD."""
        if len(self.prices) < 20 or len(self.macd_line) < 20:
            return False
        
        # Simple divergence check using recent trends
        price_trend = self.prices[-1] - self.prices[-10]
        macd_trend = self.macd_line[-1] - self.macd_line[-10]
        
        # Divergence threshold
        threshold = 0.001
        
        # Bullish divergence: price down, MACD up
        # Bearish divergence: price up, MACD down
        bullish_div = price_trend < 0 and macd_trend > threshold
        bearish_div = price_trend > 0 and macd_trend < -threshold
        
        return bullish_div or bearish_div
    
    def _generate_signal(self, macd_value: float, signal_value: float, histogram_value: float,
                        rsi: float, signal_crossover: bool, zero_crossover: bool,
                        divergence: bool, volume_surge: bool) -> Tuple[str, float]:
        """
        Generate trading signal based on all indicators.
        
        Returns:
        - Tuple of (signal, strength)
        """
        signal = 'HOLD'
        strength = 0.0
        
        # Base signals from MACD signal line crossover
        if signal_crossover and macd_value > signal_value and abs(histogram_value) > self.histogram_threshold:
            signal = 'BUY'
            strength = 60.0
        elif signal_crossover and macd_value < signal_value and abs(histogram_value) > self.histogram_threshold:
            signal = 'SELL'
            strength = 60.0
        
        # Zero line crossover confirmation
        if signal == 'BUY' and macd_value > 0:
            strength *= 1.3  # Above zero line
        elif signal == 'SELL' and macd_value < 0:
            strength *= 1.3  # Below zero line
        elif signal == 'BUY' and macd_value < 0:
            strength *= 0.7  # Below zero line
        elif signal == 'SELL' and macd_value > 0:
            strength *= 0.7  # Above zero line
        
        # Histogram momentum
        if len(self.histogram) >= 2:
            histogram_momentum = self.histogram[-1] - self.histogram[-2]
            if signal == 'BUY' and histogram_momentum > 0:
                strength *= 1.2  # Increasing histogram
            elif signal == 'SELL' and histogram_momentum < 0:
                strength *= 1.2  # Decreasing histogram
        
        # RSI filter
        if signal == 'BUY' and rsi > self.rsi_overbought:
            strength *= 0.6  # Overbought condition
        elif signal == 'SELL' and rsi < self.rsi_oversold:
            strength *= 0.6  # Oversold condition
        elif signal == 'BUY' and rsi < self.rsi_overbought:
            strength *= 1.1
        elif signal == 'SELL' and rsi > self.rsi_oversold:
            strength *= 1.1
        
        # Divergence boost
        if divergence and signal != 'HOLD':
            strength *= 1.4
        
        # Zero crossover boost
        if zero_crossover and signal != 'HOLD':
            strength *= 1.3
        
        # Volume confirmation
        if volume_surge and signal != 'HOLD':
            strength *= 1.2
        
        # Cap strength at 100%
        strength = min(strength, 100.0)
        
        return signal, strength
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy state information."""
        return {
            'strategy_type': 'FastMACD',
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'current_macd': self.macd_line[-1] if self.macd_line else 0.0,
            'current_signal': self.signal_line[-1] if self.signal_line else 0.0,
            'current_histogram': self.histogram[-1] if self.histogram else 0.0,
            'fast_ema': self.fast_ema or 0.0,
            'slow_ema': self.slow_ema or 0.0,
            'data_points': len(self.prices),
            'ready_for_signals': len(self.prices) >= self.slow_period
        }
    
    def _calculate_rsi_fast(self) -> float:
        """Calculate RSI using optimized algorithm for real-time processing."""
        if len(self.prices) < self.rsi_period + 1:
            return 50.0  # Neutral RSI
        
        # Get price changes for RSI period
        price_changes = [self.prices[i] - self.prices[i-1] for i in range(1, min(len(self.prices), self.rsi_period + 1))]
        
        # Calculate gains and losses
        gains = [change if change > 0 else 0 for change in price_changes]
        losses = [-change if change < 0 else 0 for change in price_changes]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Calculate RSI
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
