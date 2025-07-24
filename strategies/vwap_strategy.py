"""
Production-Ready VWAP Strategy for High-Frequency Intraday Trading

This module provides an optimized, low-latency implementation of the VWAP
strategy with multi-timeframe analysis for very fast intraday trading.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Any
import time
from datetime import datetime, time as dt_time
from strategies.fast_strategy_base import FastStrategyBase

class FastVWAPStrategy(FastStrategyBase):
    """
    High-performance VWAP strategy optimized for intraday trading.
    
    Features:
    - Rolling VWAP calculation with O(1) operations
    - Multiple VWAP timeframes (session, hourly)
    - VWAP bands for support/resistance
    - Volume profile analysis
    - Real-time deviation tracking
    - Microsecond precision timestamps
    """
    
    def __init__(self, 
                 vwap_bands_std: float = 1.0,
                 volume_factor: float = 1.5,
                 deviation_threshold: float = 0.5,
                 session_start: str = "09:30",
                 session_end: str = "15:30",
                 **kwargs):
        """
        Initialize the fast VWAP strategy.
        
        Parameters:
        - vwap_bands_std: Standard deviation for VWAP bands
        - volume_factor: Volume surge detection factor
        - deviation_threshold: Price deviation from VWAP threshold (%)
        - session_start: Trading session start time (HH:MM)
        - session_end: Trading session end time (HH:MM)
        """
        super().__init__(**kwargs)
        
        self.vwap_bands_std = vwap_bands_std
        self.volume_factor = volume_factor
        self.deviation_threshold = deviation_threshold
        
        # Parse session times
        start_parts = session_start.split(":")
        end_parts = session_end.split(":")
        self.session_start = dt_time(int(start_parts[0]), int(start_parts[1]))
        self.session_end = dt_time(int(end_parts[0]), int(end_parts[1]))
        
        # VWAP components
        self.cumulative_volume = 0.0
        self.cumulative_pv = 0.0  # Price * Volume
        self.cumulative_pv2 = 0.0  # Price^2 * Volume
        self.session_vwap = 0.0
        
        # Hourly VWAP components
        self.hourly_volume = 0.0
        self.hourly_pv = 0.0
        self.hourly_vwap = 0.0
        self.current_hour = None
        
        # VWAP bands
        self.vwap_upper = 0.0
        self.vwap_lower = 0.0
        
        # Rolling data for calculations
        self.typical_prices = deque(maxlen=1000)
        self.price_deviations = deque(maxlen=100)
        
        # Volume analysis
        self.volume_sma = deque(maxlen=20)
        self.volume_profile = {}  # Price level -> Volume
        
        # Session tracking
        self.session_high = 0.0
        self.session_low = float('inf')
        self.session_reset_needed = False
        
    def add_tick(self, price: float, volume: int, timestamp: Optional[float] = None,
                 high: Optional[float] = None, low: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a new market tick and generate trading signals.
        
        Parameters:
        - price: Current price
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
        
        # Get current time
        current_time = datetime.fromtimestamp(timestamp / 1_000_000 if timestamp else time.time()).time()
        
        # Check if session reset is needed
        if self._should_reset_session(current_time):
            self._reset_session()
        
        # Initialize result
        result = {
            'signal': 'HOLD',
            'strength': 0.0,
            'session_vwap': 0.0,
            'hourly_vwap': 0.0,
            'vwap_upper': 0.0,
            'vwap_lower': 0.0,
            'price_deviation': 0.0,
            'volume_surge': False,
            'above_vwap': False,
            'at_band': False,
            'volume_profile_strength': 0.0,
            'price': price,
            'volume': volume,
            'calculation_time_ms': 0.0
        }
        
        # Calculate typical price
        typical_price = (high + low + price) / 3.0
        self.typical_prices.append(typical_price)
        
        # Update session data
        self._update_session_data(typical_price, volume, current_time)
        
        # Calculate VWAPs
        self._calculate_session_vwap()
        self._calculate_hourly_vwap(current_time)
        
        # Calculate VWAP bands
        self._calculate_vwap_bands()
        
        # Update results
        result['session_vwap'] = self.session_vwap
        result['hourly_vwap'] = self.hourly_vwap
        result['vwap_upper'] = self.vwap_upper
        result['vwap_lower'] = self.vwap_lower
        
        # Calculate price deviation
        deviation = self._calculate_price_deviation(price)
        result['price_deviation'] = deviation
        
        # Check volume surge
        volume_surge = self._check_volume_surge(volume)
        result['volume_surge'] = volume_surge
        
        # Position relative to VWAP
        above_vwap = price > self.session_vwap
        result['above_vwap'] = above_vwap
        
        # Check if at bands
        at_band = self._check_at_bands(price)
        result['at_band'] = at_band
        
        # Volume profile analysis
        volume_strength = self._analyze_volume_profile(price, volume)
        result['volume_profile_strength'] = volume_strength
        
        # Generate signal
        signal, strength = self._generate_signal(
            price, deviation, above_vwap, at_band, volume_surge, volume_strength
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
    
    def _should_reset_session(self, current_time: dt_time) -> bool:
        """Check if session should be reset."""
        # Reset at session start or if crossing midnight
        if current_time >= self.session_start and self.session_reset_needed:
            return True
        
        # Mark for reset after session end
        if current_time >= self.session_end:
            self.session_reset_needed = True
        
        return False
    
    def _reset_session(self):
        """Reset session-level data."""
        self.cumulative_volume = 0.0
        self.cumulative_pv = 0.0
        self.cumulative_pv2 = 0.0
        self.session_vwap = 0.0
        self.session_high = 0.0
        self.session_low = float('inf')
        self.session_reset_needed = False
        self.volume_profile = {}
    
    def _update_session_data(self, typical_price: float, volume: int, current_time: dt_time):
        """Update session-level data."""
        # Update session high/low
        self.session_high = max(self.session_high, typical_price)
        self.session_low = min(self.session_low, typical_price)
        
        # Update cumulative values
        pv = typical_price * volume
        self.cumulative_volume += volume
        self.cumulative_pv += pv
        self.cumulative_pv2 += typical_price * pv
        
        # Update volume profile
        price_level = round(typical_price, 2)
        if price_level in self.volume_profile:
            self.volume_profile[price_level] += volume
        else:
            self.volume_profile[price_level] = volume
    
    def _calculate_session_vwap(self):
        """Calculate session VWAP."""
        if self.cumulative_volume > 0:
            self.session_vwap = self.cumulative_pv / self.cumulative_volume
        else:
            self.session_vwap = self.prices[-1] if self.prices else 0.0
    
    def _calculate_hourly_vwap(self, current_time: dt_time):
        """Calculate hourly VWAP."""
        current_hour = current_time.hour
        
        # Reset hourly data if new hour
        if self.current_hour != current_hour:
            self.hourly_volume = 0.0
            self.hourly_pv = 0.0
            self.current_hour = current_hour
        
        # Update hourly data
        typical_price = self.typical_prices[-1]
        volume = self.volumes[-1]
        
        self.hourly_volume += volume
        self.hourly_pv += typical_price * volume
        
        if self.hourly_volume > 0:
            self.hourly_vwap = self.hourly_pv / self.hourly_volume
        else:
            self.hourly_vwap = typical_price
    
    def _calculate_vwap_bands(self):
        """Calculate VWAP bands using standard deviation."""
        if self.cumulative_volume == 0 or len(self.typical_prices) < 20:
            self.vwap_upper = self.session_vwap
            self.vwap_lower = self.session_vwap
            return
        
        # Calculate variance
        variance = (self.cumulative_pv2 / self.cumulative_volume) - (self.session_vwap ** 2)
        std_dev = np.sqrt(max(variance, 0))
        
        # Calculate bands
        self.vwap_upper = self.session_vwap + (std_dev * self.vwap_bands_std)
        self.vwap_lower = self.session_vwap - (std_dev * self.vwap_bands_std)
    
    def _calculate_price_deviation(self, price: float) -> float:
        """Calculate price deviation from VWAP as percentage."""
        if self.session_vwap == 0:
            return 0.0
        
        deviation = ((price - self.session_vwap) / self.session_vwap) * 100
        self.price_deviations.append(abs(deviation))
        return deviation
    
    def _check_volume_surge(self, current_volume: int) -> bool:
        """Check for volume surge."""
        self.volume_sma.append(current_volume)
        
        if len(self.volume_sma) < 10:
            return False
        
        avg_volume = float(np.mean(list(self.volume_sma)[:-1]))
        return current_volume > (avg_volume * self.volume_factor)
    
    def _check_at_bands(self, price: float) -> bool:
        """Check if price is at VWAP bands."""
        tolerance = 0.001  # 0.1% tolerance
        
        upper_touch = abs(price - self.vwap_upper) / price < tolerance
        lower_touch = abs(price - self.vwap_lower) / price < tolerance
        
        return upper_touch or lower_touch
    
    def _analyze_volume_profile(self, price: float, volume: int) -> float:
        """Analyze volume profile strength at current price level."""
        if not self.volume_profile:
            return 0.0
        
        price_level = round(price, 2)
        current_volume = self.volume_profile.get(price_level, 0)
        max_volume = max(self.volume_profile.values())
        
        if max_volume == 0:
            return 0.0
        
        # Return strength as percentage of max volume
        return (current_volume / max_volume) * 100
    
    def _generate_signal(self, price: float, deviation: float, above_vwap: bool,
                        at_band: bool, volume_surge: bool, volume_strength: float) -> Tuple[str, float]:
        """
        Generate trading signal based on VWAP analysis.
        
        Returns:
        - Tuple of (signal, strength)
        """
        signal = 'HOLD'
        strength = 0.0
        
        # Base signals from VWAP deviation
        abs_deviation = abs(deviation)
        
        if deviation < -self.deviation_threshold and not above_vwap:
            signal = 'BUY'  # Price below VWAP threshold
            strength = min(abs_deviation * 20, 60.0)  # Scale with deviation
        elif deviation > self.deviation_threshold and above_vwap:
            signal = 'SELL'  # Price above VWAP threshold
            strength = min(abs_deviation * 20, 60.0)
        
        # Band bounce signals
        if at_band:
            if price <= self.vwap_lower:
                signal = 'BUY'
                strength = max(strength, 70.0)
            elif price >= self.vwap_upper:
                signal = 'SELL'
                strength = max(strength, 70.0)
        
        # Mean reversion logic
        if abs_deviation > 2.0:  # Extreme deviation
            if signal == 'BUY':
                strength *= 1.3  # Stronger mean reversion signal
            elif signal == 'SELL':
                strength *= 1.3
        
        # Volume profile confirmation
        if volume_strength > 70:  # High volume area
            if signal == 'BUY':
                strength *= 1.2  # Support at high volume
            elif signal == 'SELL':
                strength *= 1.2  # Resistance at high volume
        
        # Volume surge confirmation
        if volume_surge and signal != 'HOLD':
            strength *= 1.3
        
        # Hourly VWAP confirmation
        hourly_deviation = ((price - self.hourly_vwap) / self.hourly_vwap * 100) if self.hourly_vwap else 0
        if signal == 'BUY' and hourly_deviation < -0.2:
            strength *= 1.1  # Below both VWAPs
        elif signal == 'SELL' and hourly_deviation > 0.2:
            strength *= 1.1  # Above both VWAPs
        
        # Cap strength at 100%
        strength = min(strength, 100.0)
        
        return signal, strength
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy state information."""
        avg_deviation = float(np.mean(self.price_deviations)) if self.price_deviations else 0.0
        
        return {
            'strategy_type': 'FastVWAP',
            'session_vwap': self.session_vwap,
            'hourly_vwap': self.hourly_vwap,
            'vwap_upper': self.vwap_upper,
            'vwap_lower': self.vwap_lower,
            'cumulative_volume': self.cumulative_volume,
            'avg_deviation_pct': round(avg_deviation, 3),
            'volume_profile_levels': len(self.volume_profile),
            'session_high': self.session_high,
            'session_low': self.session_low,
            'data_points': len(self.prices),
            'ready_for_signals': len(self.prices) >= 20
        }
