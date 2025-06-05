import numpy as np
import pandas as pd

def find_pivots(series, window=5):
    """
    Identify pivot points in a series.
    A pivot high is a local maximum, and a pivot low is a local minimum.
    """
    pivot_highs = []
    pivot_lows = []
    
    # Need at least 2*window+1 points to find pivots
    if len(series) < 2*window+1:
        return pivot_highs, pivot_lows
    
    for i in range(window, len(series) - window):
        if all(series.iloc[i] > series.iloc[i-j] for j in range(1, window+1)) and \
           all(series.iloc[i] > series.iloc[i+j] for j in range(1, window+1)):
            pivot_highs.append(i)
        if all(series.iloc[i] < series.iloc[i-j] for j in range(1, window+1)) and \
           all(series.iloc[i] < series.iloc[i+j] for j in range(1, window+1)):
            pivot_lows.append(i)
            
    return pivot_highs, pivot_lows

def detect_stochastic_divergence(df, lookback=20, window=3):
    """
    Detect divergence between price and stochastic oscillator using proper pivot points.
    
    Parameters:
    - df: DataFrame with price and stochastic data
    - lookback: Number of bars to look back for divergences
    - window: Window size for pivot detection
    
    Returns:
    - dict: Containing bullish and bearish divergence information
    """
    # Ensure we have enough data
    if len(df) < lookback + window * 2:
        return {'bullish': False, 'bearish': False, 'strength': 0}
    
    # Get the recent part of the dataframe
    recent_df = df.iloc[-lookback:].copy()
    
    # Find price pivots
    price_highs, price_lows = find_pivots(recent_df['close'], window)
    
    # Find stochastic pivots
    stoch_highs, stoch_lows = find_pivots(recent_df['%K'], window)
    
    # Check for bullish divergence
    bullish_div = False
    bullish_strength = 0
    
    if len(price_lows) >= 2 and len(stoch_lows) >= 2:
        # Get the two most recent price lows and stochastic lows
        if len(price_lows) >= 2 and len(stoch_lows) >= 2:
            # Compare the most recent pivots
            p1, p2 = price_lows[-1], price_lows[-2]
            s1, s2 = stoch_lows[-1], stoch_lows[-2]
            
            # Bullish divergence: price makes lower low but stochastic makes higher low
            if (recent_df['close'].iloc[p1] < recent_df['close'].iloc[p2] and 
                recent_df['%K'].iloc[s1] > recent_df['%K'].iloc[s2]):
                bullish_div = True
                # Calculate strength based on the divergence magnitude
                price_change = (recent_df['close'].iloc[p1] / recent_df['close'].iloc[p2] - 1) * 100
                stoch_change = (recent_df['%K'].iloc[s1] / recent_df['%K'].iloc[s2] - 1) * 100
                bullish_strength = abs(stoch_change - price_change)
    
    # Check for bearish divergence
    bearish_div = False
    bearish_strength = 0
    
    if len(price_highs) >= 2 and len(stoch_highs) >= 2:
        # Get the two most recent price highs and stochastic highs
        p1, p2 = price_highs[-1], price_highs[-2]
        s1, s2 = stoch_highs[-1], stoch_highs[-2]
        
        # Bearish divergence: price makes higher high but stochastic makes lower high
        if (recent_df['close'].iloc[p1] > recent_df['close'].iloc[p2] and 
            recent_df['%K'].iloc[s1] < recent_df['%K'].iloc[s2]):
            bearish_div = True
            # Calculate strength based on the divergence magnitude
            price_change = (recent_df['close'].iloc[p1] / recent_df['close'].iloc[p2] - 1) * 100
            stoch_change = (recent_df['%K'].iloc[s1] / recent_df['%K'].iloc[s2] - 1) * 100
            bearish_strength = abs(stoch_change - price_change)
    
    return {
        'bullish': bullish_div,
        'bearish': bearish_div,
        'strength': max(bullish_strength, bearish_strength)
    }

def detect_support_resistance(df, lookback=100, window=5, threshold=0.03):
    """
    Dynamically detect support and resistance levels from recent data.
    
    Parameters:
    - df: DataFrame with price data
    - lookback: Number of bars to look back
    - window: Window size for pivot detection
    - threshold: Price percentage to cluster levels
    
    Returns:
    - tuple: (support levels, resistance levels)
    """
    if len(df) < lookback:
        return None, None
    
    # Get recent data
    recent_df = df.iloc[-lookback:].copy()
    
    # Find pivot points
    highs, lows = find_pivots(recent_df['close'], window)
    
    # Extract prices at pivot points
    resistance_levels = [recent_df['high'].iloc[i] for i in highs]
    support_levels = [recent_df['low'].iloc[i] for i in lows]
    
    # Cluster nearby levels (within threshold%)
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
    
    support_clusters = cluster_levels(support_levels, threshold)
    resistance_clusters = cluster_levels(resistance_levels, threshold)
    
    # Return most recent support and resistance
    support = min(support_clusters) if support_clusters else None
    resistance = max(resistance_clusters) if resistance_clusters else None
    
    return support, resistance

def stochastic_oscillator_strategy_with_filters(df, support_level=None, resistance_level=None, 
                                              k_period=14, d_period=3, trend_period=50,
                                              overbought=80, oversold=20, near_level_pct=2,
                                              volume_factor=1.5):
    """
    Enhanced Stochastic Oscillator Strategy with dynamic support/resistance levels, 
    trend filter, proper divergence detection, and volume confirmation.
    
    Parameters:
    - df: DataFrame with OHLCV data
    - support_level: Optional support level (detected if None)
    - resistance_level: Optional resistance level (detected if None)
    - k_period: Period for %K calculation
    - d_period: Period for %D calculation
    - trend_period: Period for trend SMA
    - overbought: Threshold for overbought condition
    - oversold: Threshold for oversold condition
    - near_level_pct: Percentage to consider "near" support/resistance
    - volume_factor: Factor for volume confirmation
    
    Returns:
    - dict: Signal details including type, strength, and stop levels
    """
    # Check for sufficient data
    required_length = max(k_period, trend_period) + 20  # Need extra for divergence detection
    if len(df) < required_length:
        return {"signal": "INSUFFICIENT_DATA", "strength": 0}
    
    # Calculate Stochastic Oscillator with division by zero protection
    df[f'{k_period}_High'] = df['high'].rolling(window=k_period).max()
    df[f'{k_period}_Low'] = df['low'].rolling(window=k_period).min()
    
    # Protect against division by zero
    denominator = df[f'{k_period}_High'] - df[f'{k_period}_Low']
    denominator = denominator.replace(0, 0.0001)
    
    df['%K'] = (df['close'] - df[f'{k_period}_Low']) / denominator * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    
    # Calculate trend filter
    df['trend_SMA'] = df['close'].rolling(window=trend_period).mean()
    
    # If support/resistance not provided, detect them
    if support_level is None or resistance_level is None:
        dynamic_support, dynamic_resistance = detect_support_resistance(df)
        support_level = dynamic_support if support_level is None else support_level
        resistance_level = dynamic_resistance if resistance_level is None else resistance_level
        
        # If still None, use recent lows/highs
        if support_level is None:
            support_level = df['low'].iloc[-20:].min()
        if resistance_level is None:
            resistance_level = df['high'].iloc[-20:].max()
    
    # Detect divergence using proper pivot point analysis
    divergence = detect_stochastic_divergence(df, lookback=20, window=3)
    
    # Add volume analysis if volume column exists
    has_volume = 'volume' in df.columns
    if has_volume:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        volume_surge = df['volume_ratio'].iloc[-1] > volume_factor
    else:
        volume_surge = True  # Default to True if no volume data
    
    # Check if price is near support or resistance
    near_support = df['close'].iloc[-1] <= support_level * (1 + near_level_pct/100)
    near_resistance = df['close'].iloc[-1] >= resistance_level * (1 - near_level_pct/100)
    
    # Current values
    latest_k = df['%K'].iloc[-1]
    latest_d = df['%D'].iloc[-1]
    prev_k = df['%K'].iloc[-2]
    prev_d = df['%D'].iloc[-2]
    latest_price = df['close'].iloc[-1]
    latest_trend = df['trend_SMA'].iloc[-1]
    
    # Consistency check (look for multiple aligned signals)
    k_above_d = (df['%K'] > df['%D']).iloc[-3:].sum() >= 2
    k_below_d = (df['%K'] < df['%D']).iloc[-3:].sum() >= 2
    oversold_count = (df['%K'] < oversold).iloc[-5:].sum()
    overbought_count = (df['%K'] > overbought).iloc[-5:].sum()
    
    # Calculate ATR for stop loss
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    latest_atr = df['atr'].iloc[-1]
    
    # Calculate signal strength (0-100)
    def calculate_strength(signal_type):
        base = 50  # Start at neutral
        
        # Stochastic position (0-20)
        if signal_type == "BUY":
            stoch_score = max(0, min(20, ((oversold + 10) - latest_k) / 2))
        else:
            stoch_score = max(0, min(20, (latest_k - (overbought - 10)) / 2))
            
        # Crossover strength (0-10)
        crossover_score = min(10, abs(latest_k - latest_d) * 2)
        
        # Divergence (0-20)
        div_score = 0
        if (signal_type == "BUY" and divergence['bullish']) or \
           (signal_type == "SELL" and divergence['bearish']):
            div_score = min(20, divergence['strength'] * 2)
            
        # Support/Resistance proximity (0-15)
        sr_score = 0
        if signal_type == "BUY" and near_support:
            sr_score = 15
        elif signal_type == "SELL" and near_resistance:
            sr_score = 15
            
        # Volume confirmation (0-10)
        vol_score = 10 if volume_surge else 0
        
        # Total score
        return min(100, base + stoch_score + crossover_score + div_score + sr_score + vol_score)
        
    # Enhanced signal generation with multiple filters
    if (
        latest_k > latest_d and  # %K crosses above %D
        prev_k <= prev_d and
        (latest_k < oversold + 10) and  # Coming from oversold area
        oversold_count >= 2 and  # Was consistently oversold recently
        (latest_price > latest_trend or near_support) and  # Above trend or at support
        (divergence['bullish'] or near_support) and  # Either divergence or strong support
        (not has_volume or volume_surge)  # Volume confirmation if available
    ):
        # Calculate signal strength
        strength = calculate_strength("BUY")
        
        # Calculate stop loss and take profit
        stop_loss = max(latest_price - latest_atr * 2, support_level * 0.98)
        take_profit = latest_price + latest_atr * 3
        
        return {
            "signal": "BUY",
            "strength": strength,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "divergence": divergence['bullish'],
            "entry_zone": (support_level, support_level * (1 + near_level_pct/100))
        }
        
    elif (
        latest_k < latest_d and  # %K crosses below %D
        prev_k >= prev_d and
        (latest_k > overbought - 10) and  # Coming from overbought area
        overbought_count >= 2 and  # Was consistently overbought recently
        (latest_price < latest_trend or near_resistance) and  # Below trend or at resistance
        (divergence['bearish'] or near_resistance) and  # Either divergence or strong resistance
        (not has_volume or volume_surge)  # Volume confirmation if available
    ):
        # Calculate signal strength
        strength = calculate_strength("SELL")
        
        # Calculate stop loss and take profit
        stop_loss = min(latest_price + latest_atr * 2, resistance_level * 1.02)
        take_profit = latest_price - latest_atr * 3
        
        return {
            "signal": "SELL",
            "strength": strength,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "divergence": divergence['bearish'],
            "entry_zone": (resistance_level * (1 - near_level_pct/100), resistance_level)
        }
        
    else:
        # Check for potential signals developing
        potential = "NONE"
        potential_strength = 0
        
        if latest_k < oversold + 5:
            potential = "BUY"
            potential_strength = 30 - min(30, (latest_k - oversold) * 6)
        elif latest_k > overbought - 5:
            potential = "SELL"
            potential_strength = 30 - min(30, (overbought - latest_k) * 6)
            
        return {
            "signal": "HOLD", 
            "potential": potential, 
            "strength": potential_strength
        }