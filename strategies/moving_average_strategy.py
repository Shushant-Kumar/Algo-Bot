"""
Production-Ready Moving Average Strategy for High-Frequency Intraday Trading

This module provides an optimized, low-latency implementation of the Moving Average
strategy with multiple filters for very fast intraday trading.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any
import time
from strategies.fast_strategy_base import FastStrategyBase

class FastMovingAverageStrategy(FastStrategyBase):
    """
    High-performance Moving Average strategy optimized for intraday trading.
    
    Features:
    - Rolling EMA calculations with O(1) operations
    - ATR-based position sizing and stop losses
    - Volume filter and RSI confirmation
    - Real-time trend analysis
    - Microsecond precision timestamps
    """
    
    def __init__(self, 
                 short_period: int = 5,
                 long_period: int = 20,
                 trend_period: int = 50,
                 rsi_period: int = 14,
                 atr_period: int = 14,
                 atr_multiplier: float = 1.5,
                 volume_factor: float = 1.5,
                 **kwargs):
        """
        Initialize the fast Moving Average strategy.
        
        Parameters:
        - short_period: Short EMA period
        - long_period: Long EMA period
        - trend_period: Trend filter SMA period
        - rsi_period: RSI filter period
        - atr_period: ATR calculation period
        - atr_multiplier: ATR multiplier for stops
        - volume_factor: Volume surge detection factor
        """
        super().__init__(**kwargs)
        
        self.short_period = short_period
        self.long_period = long_period
        self.trend_period = trend_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_factor = volume_factor
        
        # EMA components
        self.short_ema = None
        self.long_ema = None
        self.trend_sma = deque(maxlen=trend_period)
        
        # ATR components
        self.high_prices = deque(maxlen=atr_period)
        self.low_prices = deque(maxlen=atr_period)
        self.close_prices = deque(maxlen=atr_period)
        self.true_ranges = deque(maxlen=atr_period)
        
        # Volume analysis
        self.volume_sma = deque(maxlen=20)
        
        # Signal tracking
        self.ema_crossovers = deque(maxlen=10)
        self.trend_alignment = deque(maxlen=5)
        
    def add_tick(self, price: float, volume: int, timestamp: Optional[float] = None,
                 high: Optional[float] = None, low: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a new market tick and generate trading signals.
        
        Parameters:
        - price: Current price (close)
        - volume: Current volume
        - timestamp: Optional timestamp
        - high: Optional high price (uses price if None)
        - low: Optional low price (uses price if None)
        
        Returns:
        - Dictionary with signal and metadata
        """
        start_time = time.perf_counter()
        
        # Use price as high/low if not provided (for tick data)
        if high is None:
            high = price
        if low is None:
            low = price
        
        # Update base data
        self._update_base_data(price, volume, timestamp)
        
        # Update OHLC data
        self.high_prices.append(high)
        self.low_prices.append(low)
        self.close_prices.append(price)
        
        # Initialize result
        result = {
            'signal': 'HOLD',
            'strength': 0.0,
            'short_ema': 0.0,
            'long_ema': 0.0,
            'trend_sma': 0.0,
            'atr': 0.0,
            'rsi': 50.0,
            'volume_surge': False,
            'crossover': False,
            'trend_aligned': False,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'price': price,
            'volume': volume,
            'calculation_time_ms': 0.0
        }
        
        # Need minimum data for calculations
        if len(self.prices) < max(self.long_period, self.trend_period):
            result['calculation_time_ms'] = (time.perf_counter() - start_time) * 1000
            return result
        
        # Calculate EMAs
        short_ema = self._calculate_short_ema(price)
        long_ema = self._calculate_long_ema(price)
        result['short_ema'] = short_ema
        result['long_ema'] = long_ema
        
        # Calculate trend SMA
        trend_sma = self._calculate_trend_sma()
        result['trend_sma'] = trend_sma
        
        # Calculate ATR
        atr = self._calculate_atr()
        result['atr'] = atr
        
        # Calculate RSI for filter
        rsi = self._calculate_rsi_fast()
        result['rsi'] = rsi
        
        # Check volume surge
        volume_surge = self._check_volume_surge(volume)
        result['volume_surge'] = volume_surge
        
        # Detect crossover
        crossover = self._detect_crossover(short_ema, long_ema)
        result['crossover'] = crossover
        
        # Check trend alignment
        trend_aligned = self._check_trend_alignment(price, trend_sma)
        result['trend_aligned'] = trend_aligned
        
        # Generate signal
        signal, strength = self._generate_signal(
            short_ema, long_ema, trend_sma, rsi, crossover, trend_aligned, volume_surge
        )
        
        # Calculate stop loss and take profit
        if signal != 'HOLD':
            stop_loss, take_profit = self._calculate_stops(price, atr, signal)
            result['stop_loss'] = stop_loss
            result['take_profit'] = take_profit
        
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
    
    def _calculate_short_ema(self, price: float) -> float:
        """Calculate short EMA using optimized method."""
        if self.short_ema is None:
            if len(self.prices) >= self.short_period:
                self.short_ema = float(np.mean(list(self.prices)[-self.short_period:]))
            else:
                self.short_ema = price
            return self.short_ema
        
        alpha = 2.0 / (self.short_period + 1)
        self.short_ema = alpha * price + (1 - alpha) * self.short_ema
        return self.short_ema
    
    def _calculate_long_ema(self, price: float) -> float:
        """Calculate long EMA using optimized method."""
        if self.long_ema is None:
            if len(self.prices) >= self.long_period:
                self.long_ema = float(np.mean(list(self.prices)[-self.long_period:]))
            else:
                self.long_ema = price
            return self.long_ema
        
        alpha = 2.0 / (self.long_period + 1)
        self.long_ema = alpha * price + (1 - alpha) * self.long_ema
        return self.long_ema
    
    def _calculate_trend_sma(self) -> float:
        """Calculate trend SMA."""
        self.trend_sma.append(self.prices[-1])
        
        if len(self.trend_sma) < self.trend_period:
            return float(np.mean(self.trend_sma))
        
        return float(np.mean(self.trend_sma))
    
    def _calculate_atr(self) -> float:
        """Calculate Average True Range."""
        if len(self.close_prices) < 2:
            return 0.0
        
        # Calculate True Range
        high = self.high_prices[-1]
        low = self.low_prices[-1]
        prev_close = self.close_prices[-2]
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = max(tr1, tr2, tr3)
        self.true_ranges.append(true_range)
        
        # Calculate ATR as SMA of True Range
        if len(self.true_ranges) < self.atr_period:
            return float(np.mean(self.true_ranges))
        
        return float(np.mean(list(self.true_ranges)[-self.atr_period:]))
    
    def _check_volume_surge(self, current_volume: int) -> bool:
        """Check for volume surge."""
        self.volume_sma.append(current_volume)
        
        if len(self.volume_sma) < 10:
            return False
        
        avg_volume = float(np.mean(list(self.volume_sma)[:-1]))
        return current_volume > (avg_volume * self.volume_factor)
    
    def _detect_crossover(self, short_ema: float, long_ema: float) -> bool:
        """Detect EMA crossover."""
        if len(self.ema_crossovers) == 0:
            self.ema_crossovers.append(short_ema > long_ema)
            return False
        
        current_position = short_ema > long_ema
        previous_position = self.ema_crossovers[-1]
        
        self.ema_crossovers.append(current_position)
        
        # Return True if crossover occurred
        return current_position != previous_position
    
    def _check_trend_alignment(self, price: float, trend_sma: float) -> bool:
        """Check if price is aligned with trend."""
        trend_up = price > trend_sma
        self.trend_alignment.append(trend_up)
        return trend_up
    
    def _generate_signal(self, short_ema: float, long_ema: float, trend_sma: float,
                        rsi: float, crossover: bool, trend_aligned: bool,
                        volume_surge: bool) -> Tuple[str, float]:
        """
        Generate trading signal based on all indicators.
        
        Returns:
        - Tuple of (signal, strength)
        """
        signal = 'HOLD'
        strength = 0.0
        
        price = self.prices[-1]
        
        # Base signals from EMA crossover
        if crossover and short_ema > long_ema:
            signal = 'BUY'
            strength = 60.0
        elif crossover and short_ema < long_ema:
            signal = 'SELL'
            strength = 60.0
        
        # Trend filter
        if signal == 'BUY' and not trend_aligned:
            strength *= 0.5  # Reduce strength if against trend
        elif signal == 'SELL' and trend_aligned:
            strength *= 0.5
        elif signal == 'BUY' and trend_aligned:
            strength *= 1.3  # Increase strength if with trend
        elif signal == 'SELL' and not trend_aligned:
            strength *= 1.3
        
        # RSI filter
        if signal == 'BUY' and rsi > 70:
            strength *= 0.6  # Overbought condition
        elif signal == 'SELL' and rsi < 30:
            strength *= 0.6  # Oversold condition
        elif signal == 'BUY' and rsi < 70:
            strength *= 1.2
        elif signal == 'SELL' and rsi > 30:
            strength *= 1.2
        
        # Volume confirmation
        if volume_surge and signal != 'HOLD':
            strength *= 1.3
        
        # EMA separation (momentum)
        ema_separation = abs(short_ema - long_ema) / price * 100
        if ema_separation > 0.5:  # Strong momentum
            strength *= 1.2
        
        # Cap strength at 100%
        strength = min(strength, 100.0)
        
        return signal, strength
    
    def _calculate_stops(self, price: float, atr: float, signal: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        if atr == 0:
            atr = price * 0.01  # Fallback to 1% of price
        
        stop_distance = atr * self.atr_multiplier
        
        if signal == 'BUY':
            stop_loss = price - stop_distance
            take_profit = price + (stop_distance * 2)  # 2:1 risk-reward
        elif signal == 'SELL':
            stop_loss = price + stop_distance
            take_profit = price - (stop_distance * 2)
        else:
            stop_loss = price
            take_profit = price
        
        return stop_loss, take_profit
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy state information."""
        return {
            'strategy_type': 'FastMovingAverage',
            'short_period': self.short_period,
            'long_period': self.long_period,
            'current_short_ema': self.short_ema or 0.0,
            'current_long_ema': self.long_ema or 0.0,
            'current_trend_sma': float(np.mean(self.trend_sma)) if self.trend_sma else 0.0,
            'current_atr': float(np.mean(self.true_ranges)) if self.true_ranges else 0.0,
            'data_points': len(self.prices),
            'ready_for_signals': len(self.prices) >= max(self.long_period, self.trend_period)
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
