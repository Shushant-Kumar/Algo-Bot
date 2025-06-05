# strategies/rsi_strategy.py

import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    """
    Calculate RSI with protection against division by zero.
    """
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Protect against division by zero
    loss = loss.replace(0, 0.00001)
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_macd(data):
    data['12_EMA'] = data['close'].ewm(span=12, adjust=False).mean()
    data['26_EMA'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['12_EMA'] - data['26_EMA']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal_Line']
    return data

def find_pivots(series, window=3):
    """
    Identify pivot points in a series.
    A pivot high is a local maximum, and a pivot low is a local minimum.
    """
    pivot_highs = (series.shift(window) < series) & (series.shift(-window) < series)
    pivot_lows = (series.shift(window) > series) & (series.shift(-window) > series)
    return pivot_highs, pivot_lows

def detect_divergence(df, window=3, lookback=20):
    """
    Detect divergence using pivot points over a specified window.
    Improved to detect recent divergences within a lookback period.
    
    Parameters:
    - df: DataFrame with price and RSI
    - window: Window size for pivot detection
    - lookback: Number of bars to look back for divergence
    
    Returns:
    - tuple: (bullish divergence flag, bearish divergence flag, divergence strength)
    """
    # Ensure we have enough data
    if len(df) < lookback + window * 2:
        return False, False, 0
    
    # Get recent data for divergence detection
    recent_df = df.iloc[-lookback:].copy()
    
    price_highs, price_lows = find_pivots(recent_df['close'], window)
    rsi_highs, rsi_lows = find_pivots(recent_df['rsi'], window)

    # Find higher lows in price but lower lows in RSI (bullish)
    bullish_div_points = []
    for i in range(1, len(recent_df)):
        if not price_lows.iloc[i] or not rsi_lows.iloc[i]:
            continue
            
        # Look for previous pivot points
        for j in range(max(0, i-10), i):
            if not price_lows.iloc[j] or not rsi_lows.iloc[j]:
                continue
                
            # Check if price made higher low but RSI made lower low
            if (recent_df['close'].iloc[i] > recent_df['close'].iloc[j] and 
                recent_df['rsi'].iloc[i] < recent_df['rsi'].iloc[j]):
                strength = (recent_df['close'].iloc[i] / recent_df['close'].iloc[j] - 1) * 100
                bullish_div_points.append((i, j, strength))
    
    # Find lower highs in price but higher highs in RSI (bearish)
    bearish_div_points = []
    for i in range(1, len(recent_df)):
        if not price_highs.iloc[i] or not rsi_highs.iloc[i]:
            continue
            
        # Look for previous pivot points
        for j in range(max(0, i-10), i):
            if not price_highs.iloc[j] or not rsi_highs.iloc[j]:
                continue
                
            # Check if price made lower high but RSI made higher high
            if (recent_df['close'].iloc[i] < recent_df['close'].iloc[j] and 
                recent_df['rsi'].iloc[i] > recent_df['rsi'].iloc[j]):
                strength = (1 - recent_df['close'].iloc[i] / recent_df['close'].iloc[j]) * 100
                bearish_div_points.append((i, j, strength))
    
    # Get the most recent and strongest divergence
    bullish_strength = max([s for _, _, s in bullish_div_points]) if bullish_div_points else 0
    bearish_strength = max([s for _, _, s in bearish_div_points]) if bearish_div_points else 0
    
    # Only consider recent divergences in the last 5 bars
    recent_bullish = any(i >= len(recent_df) - 5 for i, _, _ in bullish_div_points)
    recent_bearish = any(i >= len(recent_df) - 5 for i, _, _ in bearish_div_points)
    
    return recent_bullish, recent_bearish, max(bullish_strength, bearish_strength)

def analyze_multiple_timeframes(df, periods=[14, 21, 50]):
    """
    Analyze RSI across multiple timeframes for confirmation.
    
    Returns:
    - dict: RSI values and trend alignment across timeframes
    """
    results = {}
    
    for period in periods:
        # Calculate RSI for this timeframe
        rsi_value = calculate_rsi(df, period).iloc[-1]
        
        # Determine trend direction for this timeframe
        rsi_trend = "bullish" if rsi_value < 40 else ("bearish" if rsi_value > 60 else "neutral")
        
        results[f'rsi_{period}'] = {
            'value': rsi_value,
            'trend': rsi_trend
        }
    
    # Check alignment across timeframes
    trends = [results[f'rsi_{p}']['trend'] for p in periods]
    
    results['aligned_bullish'] = all(t == "bullish" for t in trends)
    results['aligned_bearish'] = all(t == "bearish" for t in trends)
    results['partially_aligned_bullish'] = trends.count("bullish") >= len(trends) // 2
    results['partially_aligned_bearish'] = trends.count("bearish") >= len(trends) // 2
    
    return results

def rsi_strategy_with_filters(df, 
                             rsi_period=14, 
                             rsi_overbought=70, 
                             rsi_oversold=30,
                             trend_period=50,
                             confirmation_days=2,
                             volume_factor=1.5):
    """
    Enhanced RSI Strategy with divergence, MACD confirmation, trend filter, and more precise signals.
    
    Parameters:
    - df: DataFrame with OHLCV data
    - rsi_period: Period for RSI calculation
    - rsi_overbought: Threshold for overbought condition
    - rsi_oversold: Threshold for oversold condition
    - trend_period: Period for the trend filter SMA
    - confirmation_days: Days required to confirm a signal
    - volume_factor: Factor to identify significant volume
    
    Returns:
    - dict: Signal details including type, strength, and stop levels
    """
    # Check for sufficient data
    required_length = max(rsi_period, trend_period) + 30  # Need extra for divergence detection
    if len(df) < required_length:
        return {"signal": "INSUFFICIENT_DATA", "strength": 0, "stop_loss": None}
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df, period=rsi_period)
    
    # Calculate MACD
    df = calculate_macd(df)
    
    # Calculate trend filter
    df['trend_sma'] = df['close'].rolling(window=trend_period).mean()
    
    # Multiple timeframe analysis
    mtf_analysis = analyze_multiple_timeframes(df)
    
    # Detect divergence with improved precision
    bullish_div, bearish_div, div_strength = detect_divergence(df, lookback=30)
    
    # Add volume analysis if volume column exists
    has_volume = 'volume' in df.columns
    if has_volume:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        volume_surge = df['volume_ratio'].iloc[-1] > volume_factor
    else:
        volume_surge = True  # If no volume data, don't use as filter
    
    # Check for signal persistence
    rsi_oversold_count = sum(df['rsi'].iloc[-confirmation_days:] < rsi_oversold)
    rsi_overbought_count = sum(df['rsi'].iloc[-confirmation_days:] > rsi_overbought)
    
    # Current values
    latest_rsi = df['rsi'].iloc[-1]
    latest_macd = df['MACD'].iloc[-1]
    latest_signal_line = df['Signal_Line'].iloc[-1]
    latest_histogram = df['Histogram'].iloc[-1]
    latest_price = df['close'].iloc[-1]
    latest_trend = df['trend_sma'].iloc[-1]
    
    # Calculate simple ATR for stop loss
    atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    latest_atr = atr.iloc[-1]
    
    # Calculate signal strength (0-100)
    def calculate_strength(base_confidence, signal_type):
        # Start with base confidence 
        strength = base_confidence
        
        # Add for multi-timeframe alignment (0-20)
        if signal_type == "BUY" and mtf_analysis['aligned_bullish']:
            strength += 20
        elif signal_type == "BUY" and mtf_analysis['partially_aligned_bullish']:
            strength += 10
        elif signal_type == "SELL" and mtf_analysis['aligned_bearish']:
            strength += 20
        elif signal_type == "SELL" and mtf_analysis['partially_aligned_bearish']:
            strength += 10
        
        # Add for volume confirmation (0-15)
        if volume_surge:
            strength += 15
        
        # Add for divergence (0-15)
        if (signal_type == "BUY" and bullish_div) or (signal_type == "SELL" and bearish_div):
            strength += min(15, div_strength/2)
        
        # Cap at 100
        return min(100, strength)
    
    # Generate signals with enhanced precision
    if (
        latest_rsi < rsi_oversold and  # RSI oversold
        rsi_oversold_count >= confirmation_days * 0.5 and  # Consistent oversold signals
        latest_macd > latest_signal_line and  # MACD bullish crossover
        latest_histogram > 0 and  # Positive momentum
        (bullish_div or mtf_analysis['partially_aligned_bullish']) and  # Divergence or MTF confirmation
        volume_surge  # Significant volume
    ):
        # Calculate base confidence from RSI and MACD (40-70)
        base_confidence = 40 + min(30, (rsi_oversold - latest_rsi) * 1.5)
        
        # Calculate signal strength
        strength = calculate_strength(base_confidence, "BUY")
        
        # Calculate stop loss and take profit
        stop_loss = latest_price - latest_atr * 1.5
        take_profit = latest_price + latest_atr * 3
        
        return {
            "signal": "BUY", 
            "strength": strength, 
            "stop_loss": stop_loss, 
            "take_profit": take_profit,
            "divergence": bullish_div,
            "multi_timeframe": mtf_analysis['partially_aligned_bullish']
        }
        
    elif (
        latest_rsi > rsi_overbought and  # RSI overbought
        rsi_overbought_count >= confirmation_days * 0.5 and  # Consistent overbought signals
        latest_macd < latest_signal_line and  # MACD bearish crossover
        latest_histogram < 0 and  # Negative momentum
        (bearish_div or mtf_analysis['partially_aligned_bearish']) and  # Divergence or MTF confirmation
        volume_surge  # Significant volume
    ):
        # Calculate base confidence from RSI and MACD (40-70)
        base_confidence = 40 + min(30, (latest_rsi - rsi_overbought) * 1.5)
        
        # Calculate signal strength
        strength = calculate_strength(base_confidence, "SELL")
        
        # Calculate stop loss and take profit
        stop_loss = latest_price + latest_atr * 1.5
        take_profit = latest_price - latest_atr * 3
        
        return {
            "signal": "SELL", 
            "strength": strength, 
            "stop_loss": stop_loss, 
            "take_profit": take_profit,
            "divergence": bearish_div,
            "multi_timeframe": mtf_analysis['partially_aligned_bearish']
        }
        
    else:
        # If close to generating a signal, return the potential signal type with low strength
        potential_signal = "NONE"
        potential_strength = 0
        
        if latest_rsi < rsi_oversold + 5:
            potential_signal = "BUY"
            potential_strength = 30 - min(30, (latest_rsi - rsi_oversold) * 6)
        elif latest_rsi > rsi_overbought - 5:
            potential_signal = "SELL"
            potential_strength = 30 - min(30, (rsi_overbought - latest_rsi) * 6)
        
        return {"signal": "HOLD", "potential": potential_signal, "strength": potential_strength}

def get_rsi_zones(rsi_value):
    """
    Get a descriptive zone for the RSI value.
    """
    if rsi_value > 80:
        return "Extremely overbought"
    elif rsi_value > 70:
        return "Overbought"
    elif rsi_value > 60:
        return "Moderate strength"
    elif rsi_value > 40:
        return "Neutral"
    elif rsi_value > 30:
        return "Moderate weakness"
    elif rsi_value > 20:
        return "Oversold"
    else:
        return "Extremely oversold"
