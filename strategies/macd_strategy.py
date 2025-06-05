import pandas as pd
import numpy as np

def calculate_rsi(df, period=14, ema=True):
    """
    Calculate the Relative Strength Index (RSI) for the given DataFrame.
    
    Parameters:
    - df: DataFrame containing price data
    - period: Period for RSI calculation
    - ema: If True, use exponential moving average, else use simple moving average
    """
    delta = df['close'].diff()
    
    # Handle first NaN value
    delta = delta.dropna()
    
    # Separate gains and losses
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate avg gain and loss using EMA or SMA
    if ema:
        # First values are simple averages
        avg_gain = gain[:period].mean()
        avg_loss = loss[:period].mean()
        
        # Calculate subsequent values with smoothing
        for i in range(period, len(gain)):
            avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
            
            # Store in DataFrame
            if i == period:
                df.loc[delta.index[i], 'RSI'] = 100 - (100 / (1 + (avg_gain / max(avg_loss, 1e-9))))
            else:
                df.loc[delta.index[i], 'RSI'] = 100 - (100 / (1 + (avg_gain / max(avg_loss, 1e-9))))
    else:
        # Simple moving average
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RSI
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def macd_strategy(df):
    """
    MACD Strategy: Generates BUY/SELL signals based on MACD line and Signal line crossovers.
    """
    df['12_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
        return "BUY"
    elif df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] >= df['Signal_Line'].iloc[-2]:
        return "SELL"
    else:
        return "HOLD"

def macd_strategy_with_filters(df, fast_period=12, slow_period=26, signal_period=9, 
                             rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                             volume_factor=1.5, trend_period=50, confirmation_days=2):
    """
    Enhanced MACD Strategy with multiple filters for precision.
    
    Parameters:
    - df: DataFrame with OHLCV data
    - fast_period: Fast EMA period
    - slow_period: Slow EMA period
    - signal_period: Signal line period
    - rsi_period: RSI calculation period
    - rsi_overbought: RSI overbought threshold
    - rsi_oversold: RSI oversold threshold
    - volume_factor: Volume increase factor for confirmation
    - trend_period: Period for trend filter
    - confirmation_days: Number of consecutive days needed for signal confirmation
    
    Returns:
    - Signal: "BUY", "SELL", "HOLD", or "INSUFFICIENT_DATA"
    """
    # Handle edge cases - check if we have sufficient data
    required_length = max(fast_period, slow_period, signal_period, rsi_period, trend_period) + confirmation_days + 10
    if len(df) < required_length:
        return "INSUFFICIENT_DATA"
    
    # Calculate MACD and Signal Line
    df['Fast_EMA'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['Slow_EMA'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['Fast_EMA'] - df['Slow_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Calculate histogram divergence (important for trend reversal detection)
    df['Histogram_Diff'] = df['Histogram'].diff(3)  # 3-day change in histogram
    
    # Add trend filter - SMA for trend direction
    df['Trend_SMA'] = df['close'].rolling(window=trend_period).mean()
    
    # Calculate RSI
    df = calculate_rsi(df, period=rsi_period)
    
    # Add volume analysis if volume column exists
    has_volume = 'volume' in df.columns
    if has_volume:
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Surge'] = df['volume'] > df['Volume_MA'] * volume_factor
    
    # Signal confirmation variables
    current_idx = len(df) - 1
    
    # BUY signal confirmation - check for consistent signals over multiple days
    buy_signals = 0
    for i in range(confirmation_days):
        idx = current_idx - i
        if idx < 0:
            continue
        
        # Check core MACD conditions for buy
        if (df['MACD'].iloc[idx] > df['Signal_Line'].iloc[idx] and 
            df['MACD'].iloc[idx-1] <= df['Signal_Line'].iloc[idx-1]):
            buy_signals += 1
    
    # SELL signal confirmation - check for consistent signals over multiple days
    sell_signals = 0
    for i in range(confirmation_days):
        idx = current_idx - i
        if idx < 0:
            continue
        
        # Check core MACD conditions for sell
        if (df['MACD'].iloc[idx] < df['Signal_Line'].iloc[idx] and 
            df['MACD'].iloc[idx-1] >= df['Signal_Line'].iloc[idx-1]):
            sell_signals += 1
    
    # Current data points
    current = {
        'macd': df['MACD'].iloc[-1],
        'signal': df['Signal_Line'].iloc[-1],
        'histogram': df['Histogram'].iloc[-1],
        'histogram_diff': df['Histogram_Diff'].iloc[-1],
        'rsi': df['RSI'].iloc[-1],
        'close': df['close'].iloc[-1],
        'trend_sma': df['Trend_SMA'].iloc[-1],
        'volume_surge': df['Volume_Surge'].iloc[-1] if has_volume else True
    }
    
    # Enhanced Buy Signal with precision filters
    if (buy_signals >= confirmation_days * 0.7 and  # At least 70% of days show buy signal
        current['macd'] > 0 and  # MACD is positive
        current['histogram'] > 0 and  # Histogram is positive
        current['rsi'] < rsi_oversold + 10 and  # RSI indicates not overbought
        (not has_volume or current['volume_surge']) and  # Volume confirmation if data available
        (current['histogram_diff'] > 0 or  # Rising histogram (momentum)
         (current['close'] > current['trend_sma'] and current['macd'] > 0))):  # or strong uptrend
        return "BUY"
    
    # Enhanced Sell Signal with precision filters
    elif (sell_signals >= confirmation_days * 0.7 and  # At least 70% of days show sell signal
          current['macd'] < 0 and  # MACD is negative
          current['histogram'] < 0 and  # Histogram is negative
          current['rsi'] > rsi_overbought - 10 and  # RSI indicates not oversold
          (not has_volume or current['volume_surge']) and  # Volume confirmation if data available
          (current['histogram_diff'] < 0 or  # Falling histogram (momentum)
           (current['close'] < current['trend_sma'] and current['macd'] < 0))):  # or strong downtrend
        return "SELL"
    
    # No clear signal
    else:
        return "HOLD"

def detect_macd_divergence(df, lookback=10):
    """
    Detect bullish and bearish divergences between MACD and price.
    Divergence often precedes significant price movements.
    
    Parameters:
    - df: DataFrame with price and MACD data
    - lookback: Period to look back for divergence patterns
    
    Returns:
    - dict with detected divergence information
    """
    # Make sure we have enough data
    if len(df) < lookback + 5:
        return {"bullish": False, "bearish": False}
    
    # Slice dataframe to relevant period
    recent = df.iloc[-lookback:]
    
    # Find local price lows and highs
    price_lows = (recent['close'] < recent['close'].shift(1)) & (recent['close'] < recent['close'].shift(-1))
    price_highs = (recent['close'] > recent['close'].shift(1)) & (recent['close'] > recent['close'].shift(-1))
    
    # Find local MACD lows and highs
    macd_lows = (recent['MACD'] < recent['MACD'].shift(1)) & (recent['MACD'] < recent['MACD'].shift(-1))
    macd_highs = (recent['MACD'] > recent['MACD'].shift(1)) & (recent['MACD'] > recent['MACD'].shift(-1))
    
    # Check for bullish divergence (price making lower lows but MACD making higher lows)
    bullish = False
    for i in range(1, len(recent) - 1):
        if not price_lows.iloc[i]:
            continue
            
        for j in range(i + 1, len(recent) - 1):
            if not price_lows.iloc[j]:
                continue
                
            # If price made lower low but MACD made higher low (bullish divergence)
            if (recent['close'].iloc[i] > recent['close'].iloc[j] and 
                recent['MACD'].iloc[i] < recent['MACD'].iloc[j]):
                bullish = True
                break
    
    # Check for bearish divergence (price making higher highs but MACD making lower highs)
    bearish = False
    for i in range(1, len(recent) - 1):
        if not price_highs.iloc[i]:
            continue
            
        for j in range(i + 1, len(recent) - 1):
            if not price_highs.iloc[j]:
                continue
                
            # If price made higher high but MACD made lower high (bearish divergence)
            if (recent['close'].iloc[i] < recent['close'].iloc[j] and 
                recent['MACD'].iloc[i] > recent['MACD'].iloc[j]):
                bearish = True
                break
    
    return {"bullish": bullish, "bearish": bearish}

def get_macd_signal_strength(df):
    """
    Calculate the strength of the MACD signal (0-100%).
    Higher percentages indicate stronger signals.
    """
    # Get relevant values
    macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    histogram = df['Histogram'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    # Calculate crossover strength (0-100)
    crossover_strength = min(100, abs(macd - signal) * 100)
    
    # Calculate momentum strength (0-100)
    momentum_strength = min(100, abs(histogram) * 100)
    
    # RSI confirmation strength
    if macd > signal:  # Buy
        rsi_strength = max(0, min(100, 100 - rsi))  # Lower RSI = stronger buy
    else:  # Sell
        rsi_strength = max(0, min(100, rsi))  # Higher RSI = stronger sell
    
    # Calculate overall signal strength (weighted average)
    signal_strength = (crossover_strength * 0.4) + (momentum_strength * 0.4) + (rsi_strength * 0.2)
    
    # Return signal strength with sign
    if macd > signal:
        return signal_strength  # Positive for buy
    else:
        return -signal_strength  # Negative for sell