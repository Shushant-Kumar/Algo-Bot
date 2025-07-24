"""
Production-Ready Technical Indicators Module

This module provides optimized technical indicator calculations for high-frequency
trading applications. All functions are designed for speed and memory efficiency.

Features:
- Optimized pandas operations
- Robust error handling
- Memory-efficient calculations
- Support for real-time data processing
"""

import pandas as pd
import numpy as np
from collections import deque
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class FastIndicators:
    """
    Fast indicator calculations using deque for O(1) operations.
    This class provides memory-efficient indicators for real-time processing.
    """
    
    def __init__(self, max_history=500):
        """
        Initialize fast indicators with fixed-size history.
        
        Parameters:
        - max_history: Maximum number of data points to keep in memory
        """
        self.max_history = max_history
        self.prices = deque(maxlen=max_history)
        self.volumes = deque(maxlen=max_history)
        self.highs = deque(maxlen=max_history)
        self.lows = deque(maxlen=max_history)
        
        # Cached calculations
        self._rsi_cache = {}
        self._ema_cache = {}
        self._sma_cache = {}
    
    def add_tick(self, price, volume=0, high=None, low=None):
        """Add a new tick to the indicator calculations."""
        self.prices.append(float(price))
        self.volumes.append(int(volume))
        self.highs.append(float(high) if high is not None else float(price))
        self.lows.append(float(low) if low is not None else float(price))
    
    def calculate_ema(self, period=14, price_data=None):
        """Calculate EMA using efficient algorithm."""
        if price_data is None:
            price_data = list(self.prices)
        
        if len(price_data) < period:
            return None
        
        # Use cached value if available
        cache_key = f"ema_{period}_{len(price_data)}"
        if cache_key in self._ema_cache:
            return self._ema_cache[cache_key]
        
        alpha = 2.0 / (period + 1)
        ema = price_data[0]
        
        for price in price_data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        self._ema_cache[cache_key] = ema
        return ema
    
    def calculate_rsi(self, period=14):
        """Calculate RSI using efficient algorithm."""
        if len(self.prices) < period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        price_data = list(self.prices)
        changes = [price_data[i] - price_data[i-1] for i in range(1, len(price_data))]
        
        if len(changes) < period:
            return 50.0
        
        gains = [max(0, change) for change in changes[-period:]]
        losses = [abs(min(0, change)) for change in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)

# Keep the original functions for backward compatibility
def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) with enhanced error handling.
    """
    try:
        # Ensure required columns exist
        if not {'high', 'low', 'close'}.issubset(df.columns):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
        
        # Create a copy to avoid modifying original
        df_work = df.copy()
        
        # Calculate True Range (TR) components
        df_work['H-L'] = df_work['high'] - df_work['low']
        df_work['H-PC'] = abs(df_work['high'] - df_work['close'].shift(1))
        df_work['L-PC'] = abs(df_work['low'] - df_work['close'].shift(1))
        
        # Calculate TR and ATR
        df_work['TR'] = df_work[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        # Use exponential moving average for smoother ATR
        df_work['ATR'] = df_work['TR'].ewm(span=period, adjust=False).mean()
        
        # Add ATR to original dataframe
        df['ATR'] = df_work['ATR']
        
        return df
        
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        # Return original dataframe with NaN ATR column
        df['ATR'] = np.nan
        return df

def calculate_rsi(df, period=14, price_col='close', ema=True):
    """
    Calculate Relative Strength Index (RSI) with optimized performance.
    
    Parameters:
    - df: DataFrame with price data
    - period: Period for RSI calculation
    - price_col: Column name for price data
    - ema: If True, use EMA for smoothing (recommended for real-time)
    
    Returns:
    - DataFrame with added 'RSI' column
    """
    try:
        # Ensure required columns exist
        if price_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{price_col}' column.")
        
        # Create a copy to avoid modifying original
        df_work = df.copy()
        
        # Calculate price change
        delta = df_work[price_col].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        if ema:
            # Use exponential moving average for better responsiveness
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
        else:
            # Use simple moving average
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
        
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 1e-9)
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add RSI to original dataframe
        df['RSI'] = rsi
        
        return df
        
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        df['RSI'] = 50.0  # Neutral RSI on error
        return df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
    """
    Calculate Moving Average Convergence Divergence (MACD) with optimized performance.
    
    Parameters:
    - df: DataFrame with price data
    - fast_period: Period for fast EMA
    - slow_period: Period for slow EMA
    - signal_period: Period for signal line EMA
    - price_col: Column name for price data
    
    Returns:
    - DataFrame with added MACD columns
    """
    try:
        # Ensure required columns exist
        if price_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{price_col}' column.")
        
        # Calculate MACD components using efficient EMA
        fast_ema = df[price_col].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['MACD'] = fast_ema - slow_ema
        
        # Calculate Signal line
        df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD Histogram
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Store EMA values for reference
        df['Fast_EMA'] = fast_ema
        df['Slow_EMA'] = slow_ema
        
        return df
        
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        # Add default columns on error
        df['MACD'] = 0.0
        df['Signal_Line'] = 0.0
        df['MACD_Histogram'] = 0.0
        df['Fast_EMA'] = df[price_col] if price_col in df.columns else 0.0
        df['Slow_EMA'] = df[price_col] if price_col in df.columns else 0.0
        return df
    df['Slow_EMA'] = df[price_col].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['Fast_EMA'] - df['Slow_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2, price_col='close'):
    """
    Calculate Bollinger Bands with enhanced error handling.
    
    Parameters:
    - df: DataFrame with price data
    - period: Period for moving average
    - std_dev: Number of standard deviations for bands
    - price_col: Column name for price data
    
    Returns:
    - DataFrame with added Bollinger Bands columns
    """
    try:
        # Ensure required columns exist
        if price_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{price_col}' column.")
        
        # Calculate Bollinger Bands components
        sma = df[price_col].rolling(window=period).mean()
        std = df[price_col].rolling(window=period).std()
        
        df['BB_SMA'] = sma
        df['BB_STD'] = std
        df['Upper_Band'] = sma + (std * std_dev)
        df['Lower_Band'] = sma - (std * std_dev)
        
        # Calculate additional metrics
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['BB_SMA']
        
        # %B calculation with division by zero protection
        band_diff = df['Upper_Band'] - df['Lower_Band']
        band_diff = band_diff.replace(0, 1e-9)
        df['%B'] = (df[price_col] - df['Lower_Band']) / band_diff
        
        return df
        
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        # Add default columns on error
        df['BB_SMA'] = df[price_col] if price_col in df.columns else 0.0
        df['BB_STD'] = 0.0
        df['Upper_Band'] = df[price_col] if price_col in df.columns else 0.0
        df['Lower_Band'] = df[price_col] if price_col in df.columns else 0.0
        df['BB_Width'] = 0.0
        df['%B'] = 0.5
        return df

def calculate_stochastic(df, k_period=14, d_period=3, high_col='high', low_col='low', close_col='close'):
    """
    Calculate Stochastic Oscillator.
    
    Parameters:
    - df: DataFrame with price data
    - k_period: Period for %K
    - d_period: Period for %D moving average
    - high_col, low_col, close_col: Column names for price data
    
    Returns:
    - DataFrame with added Stochastic columns
    """
    # Ensure required columns exist
    if not {high_col, low_col, close_col}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain '{high_col}', '{low_col}', and '{close_col}' columns.")
    
    # Calculate Stochastic components
    df[f'{k_period}_High'] = df[high_col].rolling(window=k_period).max()
    df[f'{k_period}_Low'] = df[low_col].rolling(window=k_period).min()
    
    # Protect against division by zero
    denominator = df[f'{k_period}_High'] - df[f'{k_period}_Low']
    denominator = denominator.replace(0, 1e-9)
    
    # Calculate %K and %D
    df['%K'] = (df[close_col] - df[f'{k_period}_Low']) / denominator * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    
    return df

def calculate_vwap(df, reset_period='D'):
    """
    Calculate Volume Weighted Average Price (VWAP) with proper period reset.
    
    Parameters:
    - df: DataFrame with OHLCV data and datetime index
    - reset_period: Frequency to reset VWAP calculation ('D' for daily, 'W' for weekly)
    
    Returns:
    - DataFrame with added VWAP column
    """
    # Ensure required columns exist
    if not {'high', 'low', 'close', 'volume'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'high', 'low', 'close', and 'volume' columns.")
    
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index for VWAP calculation")
    
    # Create a copy of the dataframe to avoid modifying the original
    result = df.copy()
    
    # Create a reset marker based on the reset period
    result['period_marker'] = result.index.to_period(reset_period)
    
    # Calculate typical price
    result['tp'] = (result['high'] + result['low'] + result['close']) / 3
    
    # Calculate VWAP components
    result['tp_x_vol'] = result['tp'] * result['volume']
    
    # Group by reset period and calculate cumulative values
    groups = result.groupby('period_marker')
    result['cum_vol'] = groups['volume'].cumsum()
    result['cum_tp_x_vol'] = groups['tp_x_vol'].cumsum()
    
    # Calculate VWAP
    result['VWAP'] = result['cum_tp_x_vol'] / result['cum_vol']
    
    # Clean up intermediate columns
    result.drop(['period_marker', 'tp', 'tp_x_vol', 'cum_vol', 'cum_tp_x_vol'], axis=1, inplace=True)
    
    return result

def detect_support_resistance(df, lookback=100, window=5, threshold=0.03):
    """
    Detect support and resistance levels from price data.
    
    Parameters:
    - df: DataFrame with price data
    - lookback: Number of bars to look back
    - window: Window size for pivot detection
    - threshold: Price percentage to cluster levels
    
    Returns:
    - Dictionary with detected support and resistance levels
    """
    # Ensure required columns exist
    if not {'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
    
    # Check if we have enough data
    if len(df) < lookback:
        return {"support": None, "resistance": None, "levels": []}
    
    # Get recent data
    recent_df = df.iloc[-lookback:].copy()
    
    # Find pivot points
    pivot_highs = []
    pivot_lows = []
    
    for i in range(window, len(recent_df) - window):
        # Pivot high
        if all(recent_df['high'].iloc[i] > recent_df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(recent_df['high'].iloc[i] > recent_df['high'].iloc[i+j] for j in range(1, window+1)):
            pivot_highs.append(recent_df['high'].iloc[i])
        
        # Pivot low
        if all(recent_df['low'].iloc[i] < recent_df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(recent_df['low'].iloc[i] < recent_df['low'].iloc[i+j] for j in range(1, window+1)):
            pivot_lows.append(recent_df['low'].iloc[i])
    
    # Cluster nearby levels
    def cluster_levels(levels, threshold_pct):
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            # If this level is within threshold% of the average of the current cluster
            if (level - np.mean(current_cluster))/np.mean(current_cluster) <= threshold_pct:
                current_cluster.append(level)
            else:
                # Start a new cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
            
        return clusters
    
    support_clusters = cluster_levels(pivot_lows, threshold)
    resistance_clusters = cluster_levels(pivot_highs, threshold)
    
    # Get most recent price
    current_price = df['close'].iloc[-1]
    
    # Find closest support and resistance
    supports_below = [s for s in support_clusters if s < current_price]
    resistances_above = [r for r in resistance_clusters if r > current_price]
    
    closest_support = max(supports_below) if supports_below else None
    closest_resistance = min(resistances_above) if resistances_above else None
    
    # Combine all levels for visualization
    all_levels = sorted(support_clusters + resistance_clusters)
    
    return {
        "support": closest_support,
        "resistance": closest_resistance,
        "support_levels": support_clusters,
        "resistance_levels": resistance_clusters,
        "all_levels": all_levels
    }

def detect_divergence(df, price_col='close', indicator_col='RSI', lookback=20, window=3):
    """
    Detect regular and hidden divergences between price and an indicator.
    
    Parameters:
    - df: DataFrame with price and indicator data
    - price_col: Column name for price data
    - indicator_col: Column name for indicator (e.g., 'RSI', '%K')
    - lookback: Number of bars to look back
    - window: Window size for pivot detection
    
    Returns:
    - Dictionary with detected divergences
    """
    # Ensure required columns exist
    if not {price_col, indicator_col}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain '{price_col}' and '{indicator_col}' columns.")
    
    # Check if we have enough data
    if len(df) < lookback + window * 2:
        return {"bullish": False, "bearish": False, "hidden_bullish": False, "hidden_bearish": False}
    
    # Get recent data
    recent_df = df.iloc[-lookback:].copy()
    
    # Find price pivot points
    price_highs = []
    price_lows = []
    
    for i in range(window, len(recent_df) - window):
        # Pivot high
        if all(recent_df[price_col].iloc[i] > recent_df[price_col].iloc[i-j] for j in range(1, window+1)) and \
           all(recent_df[price_col].iloc[i] > recent_df[price_col].iloc[i+j] for j in range(1, window+1)):
            price_highs.append(i)
        
        # Pivot low
        if all(recent_df[price_col].iloc[i] < recent_df[price_col].iloc[i-j] for j in range(1, window+1)) and \
           all(recent_df[price_col].iloc[i] < recent_df[price_col].iloc[i+j] for j in range(1, window+1)):
            price_lows.append(i)
    
    # Find indicator pivot points
    indicator_highs = []
    indicator_lows = []
    
    for i in range(window, len(recent_df) - window):
        # Pivot high
        if all(recent_df[indicator_col].iloc[i] > recent_df[indicator_col].iloc[i-j] for j in range(1, window+1)) and \
           all(recent_df[indicator_col].iloc[i] > recent_df[indicator_col].iloc[i+j] for j in range(1, window+1)):
            indicator_highs.append(i)
        
        # Pivot low
        if all(recent_df[indicator_col].iloc[i] < recent_df[indicator_col].iloc[i-j] for j in range(1, window+1)) and \
           all(recent_df[indicator_col].iloc[i] < recent_df[indicator_col].iloc[i+j] for j in range(1, window+1)):
            indicator_lows.append(i)
    
    # Check for regular divergences
    bullish_div = False  # Price lower low, indicator higher low
    bearish_div = False  # Price higher high, indicator lower high
    hidden_bullish_div = False  # Price higher low, indicator lower low
    hidden_bearish_div = False  # Price lower high, indicator higher high
    
    # Check for bullish divergence (regular)
    if len(price_lows) >= 2 and len(indicator_lows) >= 2:
        p1, p2 = price_lows[-1], price_lows[-2]
        i1, i2 = indicator_lows[-1], indicator_lows[-2]
        
        # Price made lower low but indicator made higher low
        if (recent_df[price_col].iloc[p1] < recent_df[price_col].iloc[p2] and 
            recent_df[indicator_col].iloc[i1] > recent_df[indicator_col].iloc[i2]):
            bullish_div = True
    
    # Check for bearish divergence (regular)
    if len(price_highs) >= 2 and len(indicator_highs) >= 2:
        p1, p2 = price_highs[-1], price_highs[-2]
        i1, i2 = indicator_highs[-1], indicator_highs[-2]
        
        # Price made higher high but indicator made lower high
        if (recent_df[price_col].iloc[p1] > recent_df[price_col].iloc[p2] and 
            recent_df[indicator_col].iloc[i1] < recent_df[indicator_col].iloc[i2]):
            bearish_div = True
    
    # Check for hidden bullish divergence
    if len(price_lows) >= 2 and len(indicator_lows) >= 2:
        p1, p2 = price_lows[-1], price_lows[-2]
        i1, i2 = indicator_lows[-1], indicator_lows[-2]
        
        # Price made higher low but indicator made lower low
        if (recent_df[price_col].iloc[p1] > recent_df[price_col].iloc[p2] and 
            recent_df[indicator_col].iloc[i1] < recent_df[indicator_col].iloc[i2]):
            hidden_bullish_div = True
    
    # Check for hidden bearish divergence
    if len(price_highs) >= 2 and len(indicator_highs) >= 2:
        p1, p2 = price_highs[-1], price_highs[-2]
        i1, i2 = indicator_highs[-1], indicator_highs[-2]
        
        # Price made lower high but indicator made higher high
        if (recent_df[price_col].iloc[p1] < recent_df[price_col].iloc[p2] and 
            recent_df[indicator_col].iloc[i1] > recent_df[indicator_col].iloc[i2]):
            hidden_bearish_div = True
    
    return {
        "bullish": bullish_div,
        "bearish": bearish_div,
        "hidden_bullish": hidden_bullish_div,
        "hidden_bearish": hidden_bearish_div
    }

def calculate_moving_averages(df, periods=[9, 20, 50, 200], price_col='close', ma_type='sma'):
    """
    Calculate multiple moving averages for a given price series.
    
    Parameters:
    - df: DataFrame with price data
    - periods: List of periods for moving average calculation
    - price_col: Column name for price data
    - ma_type: Type of moving average ('sma' or 'ema')
    
    Returns:
    - DataFrame with added moving average columns
    """
    # Ensure required columns exist
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")
    
    # Calculate moving averages
    for period in periods:
        if ma_type.lower() == 'sma':
            df[f'SMA_{period}'] = df[price_col].rolling(window=period).mean()
        elif ma_type.lower() == 'ema':
            df[f'EMA_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
        else:
            raise ValueError("ma_type must be either 'sma' or 'ema'")
    
    return df

def detect_ma_crossover(df, fast_ma, slow_ma):
    """
    Detect moving average crossovers.
    
    Parameters:
    - df: DataFrame with calculated moving averages
    - fast_ma: Column name for fast moving average
    - slow_ma: Column name for slow moving average
    
    Returns:
    - DataFrame with added crossover signals
    """
    # Ensure required columns exist
    if not {fast_ma, slow_ma}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain '{fast_ma}' and '{slow_ma}' columns.")
    
    # Detect crossovers
    df['MA_Above'] = df[fast_ma] > df[slow_ma]
    df['MA_Cross_Up'] = (df['MA_Above'] & ~df['MA_Above'].shift(1)).astype(int)
    df['MA_Cross_Down'] = (~df['MA_Above'] & df['MA_Above'].shift(1)).astype(int)
    
    return df