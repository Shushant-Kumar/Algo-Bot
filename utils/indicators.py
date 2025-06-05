import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    # Ensure required columns exist
    if not {'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
    
    # Calculate True Range (TR) components
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    
    # Calculate TR and ATR
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    # Drop intermediate columns to keep the DataFrame clean
    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True)
    
    return df

def calculate_rsi(df, period=14, price_col='close', ema=False):
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    - df: DataFrame with price data
    - period: Period for RSI calculation
    - price_col: Column name for price data
    - ema: If True, use EMA for smoothing instead of SMA
    
    Returns:
    - DataFrame with added 'RSI' column
    """
    # Ensure required columns exist
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")
    
    # Calculate price change
    delta = df[price_col].diff()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss
    if ema:
        # First values are SMA
        avg_gain = gain[:period].mean()
        avg_loss = loss[:period].mean()
        
        # Initialize RSI columns
        df.loc[:period, 'avg_gain'] = np.nan
        df.loc[:period, 'avg_loss'] = np.nan
        df.loc[:period, 'RSI'] = np.nan
        
        # Calculate subsequent values with EMA smoothing
        for i in range(period, len(delta)):
            avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
            
            rs = avg_gain / max(avg_loss, 1e-9)  # Avoid division by zero
            df.loc[delta.index[i], 'avg_gain'] = avg_gain
            df.loc[delta.index[i], 'avg_loss'] = avg_loss
            df.loc[delta.index[i], 'RSI'] = 100 - (100 / (1 + rs))
    else:
        # Simple rolling average
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean().replace(0, 1e-9)  # Avoid division by zero
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Parameters:
    - df: DataFrame with price data
    - fast_period: Period for fast EMA
    - slow_period: Period for slow EMA
    - signal_period: Period for signal line EMA
    - price_col: Column name for price data
    
    Returns:
    - DataFrame with added MACD columns
    """
    # Ensure required columns exist
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")
    
    # Calculate MACD components
    df['Fast_EMA'] = df[price_col].ewm(span=fast_period, adjust=False).mean()
    df['Slow_EMA'] = df[price_col].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['Fast_EMA'] - df['Slow_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2, price_col='close'):
    """
    Calculate Bollinger Bands.
    
    Parameters:
    - df: DataFrame with price data
    - period: Period for moving average
    - std_dev: Number of standard deviations for bands
    - price_col: Column name for price data
    
    Returns:
    - DataFrame with added Bollinger Bands columns
    """
    # Ensure required columns exist
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column.")
    
    # Calculate Bollinger Bands components
    df['BB_SMA'] = df[price_col].rolling(window=period).mean()
    df['BB_STD'] = df[price_col].rolling(window=period).std()
    df['Upper_Band'] = df['BB_SMA'] + (df['BB_STD'] * std_dev)
    df['Lower_Band'] = df['BB_SMA'] - (df['BB_STD'] * std_dev)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['BB_SMA']
    df['%B'] = (df[price_col] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'] + 1e-9)
    
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