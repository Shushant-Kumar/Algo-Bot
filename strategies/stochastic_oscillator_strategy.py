"""
Production-Ready Stochastic Oscillator Strategy for High-Frequency Intraday Trading

This module provides an optimized, low-latency implementation of the Stochastic
Oscillator strategy with divergence detection for very fast intraday trading.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any
import time
from strategies.fast_strategy_base import FastStrategyBase

class FastStochasticStrategy(FastStrategyBase):
    """
    High-performance Stochastic Oscillator strategy optimized for intraday trading.
    
    Features:
    - Rolling stochastic calculation with O(1) operations
    - Fast %K and %D calculation
    - Divergence detection with price
    - Overbought/oversold signals
    - Volume confirmation
    - Microsecond precision timestamps
    """
    
    def __init__(self, 
                 k_period: int = 14,
                 d_period: int = 3,
                 smooth_k: int = 3,
                 overbought: float = 80.0,
                 oversold: float = 20.0,
                 divergence_lookback: int = 20,
                 volume_factor: float = 1.5,
                 **kwargs):
        """
        Initialize the fast Stochastic strategy.
        
        Parameters:
        - k_period: Period for %K calculation
        - d_period: Period for %D smoothing
        - smooth_k: Period for %K smoothing
        - overbought/oversold: Stochastic thresholds
        - divergence_lookback: Bars to look back for divergence
        - volume_factor: Volume surge detection factor
        """
        super().__init__(**kwargs)
        
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.overbought = overbought
        self.oversold = oversold
        self.divergence_lookback = divergence_lookback
        self.volume_factor = volume_factor
        
        # OHLC data storage
        self.high_prices = deque(maxlen=k_period)
        self.low_prices = deque(maxlen=k_period)
        self.close_prices = deque(maxlen=k_period)
        
        # Stochastic components
        self.raw_k = deque(maxlen=100)
        self.smooth_k_values = deque(maxlen=100)
        self.d_values = deque(maxlen=100)
        
        # Crossover tracking
        self.k_d_crossovers = deque(maxlen=10)
        self.threshold_crossovers = deque(maxlen=10)
        
        # Volume analysis
        self.volume_sma = deque(maxlen=20)
        
        # Divergence tracking
        self.stoch_pivots = deque(maxlen=divergence_lookback)
        self.price_pivots = deque(maxlen=divergence_lookback)
        
    def add_tick(self, price: float, volume: int, timestamp: Optional[float] = None,
                 high: Optional[float] = None, low: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a new market tick and generate trading signals.
        
        Parameters:
        - price: Current price (close)
        - volume: Current volume
        - timestamp: Optional timestamp
        - high: Optional high price
        - low: Optional low price
        
        Returns:
        - Dictionary with signal and metadata
        """
        start_time = time.perf_counter()
        
        # Use price as high/low if not provided
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
            'k_percent': 50.0,
            'd_percent': 50.0,
            'raw_k': 50.0,
            'volume_surge': False,
            'k_d_crossover': False,
            'threshold_cross': False,
            'divergence': False,
            'overbought': False,
            'oversold': False,
            'price': price,
            'volume': volume,
            'calculation_time_ms': 0.0
        }
        
        # Need minimum data for calculations
        if len(self.high_prices) < self.k_period:
            result['calculation_time_ms'] = (time.perf_counter() - start_time) * 1000
            return result
        
        # Calculate stochastic components
        raw_k = self._calculate_raw_k()
        smooth_k = self._calculate_smooth_k(raw_k)
        d_value = self._calculate_d(smooth_k)
        
        result['raw_k'] = raw_k
        result['k_percent'] = smooth_k
        result['d_percent'] = d_value
        
        # Store values
        self.raw_k.append(raw_k)
        self.smooth_k_values.append(smooth_k)
        self.d_values.append(d_value)
        
        # Check volume surge
        volume_surge = self._check_volume_surge(volume)
        result['volume_surge'] = volume_surge
        
        # Detect crossovers
        k_d_crossover = self._detect_k_d_crossover(smooth_k, d_value)
        threshold_cross = self._detect_threshold_crossover(smooth_k)
        result['k_d_crossover'] = k_d_crossover
        result['threshold_cross'] = threshold_cross
        
        # Check overbought/oversold
        overbought = smooth_k > self.overbought
        oversold = smooth_k < self.oversold
        result['overbought'] = overbought
        result['oversold'] = oversold
        
        # Detect divergence
        divergence = self._detect_stochastic_divergence()
        result['divergence'] = divergence
        
        # Generate signal
        signal, strength = self._generate_signal(
            smooth_k, d_value, overbought, oversold, k_d_crossover,
            threshold_cross, divergence, volume_surge
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
    
    def _calculate_raw_k(self) -> float:
        """Calculate raw %K value."""
        if len(self.high_prices) < self.k_period:
            return 50.0
        
        # Get high and low over the period
        period_high = max(self.high_prices)
        period_low = min(self.low_prices)
        current_close = self.close_prices[-1]
        
        # Avoid division by zero
        if period_high == period_low:
            return 50.0
        
        raw_k = ((current_close - period_low) / (period_high - period_low)) * 100
        return max(0.0, min(100.0, raw_k))
    
    def _calculate_smooth_k(self, raw_k: float) -> float:
        """Calculate smoothed %K value."""
        if len(self.raw_k) < self.smooth_k:
            return raw_k
        
        recent_k = list(self.raw_k)[-(self.smooth_k-1):] + [raw_k]
        return float(np.mean(recent_k))
    
    def _calculate_d(self, smooth_k: float) -> float:
        """Calculate %D value (SMA of %K)."""
        if len(self.smooth_k_values) < self.d_period:
            return smooth_k
        
        recent_k = list(self.smooth_k_values)[-(self.d_period-1):] + [smooth_k]
        return float(np.mean(recent_k))
    
    def _check_volume_surge(self, current_volume: int) -> bool:
        """Check for volume surge."""
        self.volume_sma.append(current_volume)
        
        if len(self.volume_sma) < 10:
            return False
        
        avg_volume = float(np.mean(list(self.volume_sma)[:-1]))
        return current_volume > (avg_volume * self.volume_factor)
    
    def _detect_k_d_crossover(self, k_value: float, d_value: float) -> bool:
        """Detect %K and %D crossover."""
        if len(self.k_d_crossovers) == 0:
            self.k_d_crossovers.append(k_value > d_value)
            return False
        
        current_position = k_value > d_value
        previous_position = self.k_d_crossovers[-1]
        
        self.k_d_crossovers.append(current_position)
        
        return current_position != previous_position
    
    def _detect_threshold_crossover(self, k_value: float) -> bool:
        """Detect crossover of overbought/oversold thresholds."""
        if len(self.threshold_crossovers) == 0:
            # Determine current zone
            if k_value > self.overbought:
                zone = 'overbought'
            elif k_value < self.oversold:
                zone = 'oversold'
            else:
                zone = 'neutral'
            
            self.threshold_crossovers.append(zone)
            return False
        
        # Determine current zone
        if k_value > self.overbought:
            current_zone = 'overbought'
        elif k_value < self.oversold:
            current_zone = 'oversold'
        else:
            current_zone = 'neutral'
        
        previous_zone = self.threshold_crossovers[-1]
        self.threshold_crossovers.append(current_zone)
        
        # Return True if crossing from extreme to neutral
        crossover = (previous_zone in ['overbought', 'oversold'] and 
                    current_zone == 'neutral')
        
        return crossover
    
    def _detect_stochastic_divergence(self) -> bool:
        """Detect divergence between price and stochastic."""
        if (len(self.prices) < self.divergence_lookback or 
            len(self.smooth_k_values) < self.divergence_lookback):
            return False
        
        # Get recent data
        recent_prices = list(self.prices)[-self.divergence_lookback:]
        recent_stoch = list(self.smooth_k_values)[-self.divergence_lookback:]
        
        # Simple divergence check: compare first half vs second half
        mid_point = len(recent_prices) // 2
        
        price_trend = recent_prices[-1] - recent_prices[mid_point]
        stoch_trend = recent_stoch[-1] - recent_stoch[mid_point]
        
        # Divergence thresholds
        price_threshold = recent_prices[mid_point] * 0.005  # 0.5% price change
        stoch_threshold = 5.0  # 5 point stochastic change
        
        # Bullish divergence: price down significantly, stochastic up
        # Bearish divergence: price up significantly, stochastic down
        bullish_div = (price_trend < -price_threshold and stoch_trend > stoch_threshold)
        bearish_div = (price_trend > price_threshold and stoch_trend < -stoch_threshold)
        
        return bullish_div or bearish_div
    
    def _generate_signal(self, k_value: float, d_value: float, overbought: bool,
                        oversold: bool, k_d_crossover: bool, threshold_cross: bool,
                        divergence: bool, volume_surge: bool) -> Tuple[str, float]:
        """
        Generate trading signal based on stochastic analysis.
        
        Returns:
        - Tuple of (signal, strength)
        """
        signal = 'HOLD'
        strength = 0.0
        
        # Base signals from threshold crossovers
        if threshold_cross and len(self.threshold_crossovers) >= 2:
            prev_zone = self.threshold_crossovers[-2]
            if prev_zone == 'oversold':
                signal = 'BUY'
                strength = 60.0
            elif prev_zone == 'overbought':
                signal = 'SELL'
                strength = 60.0
        
        # %K %D crossover signals
        if k_d_crossover and k_value > d_value and k_value < self.overbought:
            signal = 'BUY'
            strength = max(strength, 50.0)
        elif k_d_crossover and k_value < d_value and k_value > self.oversold:
            signal = 'SELL'
            strength = max(strength, 50.0)
        
        # Extreme level signals (mean reversion)
        if overbought and k_value > 90:  # Very overbought
            signal = 'SELL'
            strength = max(strength, 70.0)
        elif oversold and k_value < 10:  # Very oversold
            signal = 'BUY'
            strength = max(strength, 70.0)
        
        # %K and %D alignment
        if signal == 'BUY' and k_value > d_value:
            strength *= 1.2  # Both moving up
        elif signal == 'SELL' and k_value < d_value:
            strength *= 1.2  # Both moving down
        
        # Divergence boost
        if divergence and signal != 'HOLD':
            strength *= 1.4
        
        # Volume confirmation
        if volume_surge and signal != 'HOLD':
            strength *= 1.2
        
        # Momentum check
        if len(self.smooth_k_values) >= 3:
            k_momentum = self.smooth_k_values[-1] - self.smooth_k_values[-3]
            if signal == 'BUY' and k_momentum > 0:
                strength *= 1.1  # Positive momentum
            elif signal == 'SELL' and k_momentum < 0:
                strength *= 1.1  # Negative momentum
        
        # Reduce strength if in middle zone without crossover
        if not k_d_crossover and not threshold_cross and 30 < k_value < 70:
            strength *= 0.7
        
        # Cap strength at 100%
        strength = min(strength, 100.0)
        
        return signal, strength
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy state information."""
        return {
            'strategy_type': 'FastStochastic',
            'k_period': self.k_period,
            'd_period': self.d_period,
            'current_k': self.smooth_k_values[-1] if self.smooth_k_values else 50.0,
            'current_d': self.d_values[-1] if self.d_values else 50.0,
            'current_raw_k': self.raw_k[-1] if self.raw_k else 50.0,
            'overbought_level': self.overbought,
            'oversold_level': self.oversold,
            'data_points': len(self.prices),
            'ready_for_signals': len(self.high_prices) >= self.k_period
        }
