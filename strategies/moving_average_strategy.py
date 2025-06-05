import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR) for the given DataFrame.
    """
    df['High-Low'] = df['high'] - df['low']
    df['High-Close'] = abs(df['high'] - df['close'].shift(1))
    df['Low-Close'] = abs(df['low'] - df['close'].shift(1))
    df['True_Range'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=period).mean()
    return df

def calculate_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI) for the given DataFrame.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    loss = loss.replace(0, 0.00001)
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def moving_average_strategy_with_filters(df, 
                                        short_window=5, 
                                        long_window=20, 
                                        trend_window=50,
                                        atr_period=14,
                                        atr_multiplier=1.5,
                                        volume_factor=1.5,
                                        rsi_period=14,
                                        confirmation_days=2):
    """
    Enhanced Moving Average Strategy with multiple filters for precision.
    
    Parameters:
    - df: DataFrame with OHLCV data
    - short_window: Period for short EMA
    - long_window: Period for long EMA
    - trend_window: Period for trend filter SMA
    - atr_period: Period for ATR calculation
    - atr_multiplier: Multiplier for stop loss calculation
    - volume_factor: Factor to determine significant volume
    - rsi_period: Period for RSI calculation
    - confirmation_days: Days required to confirm a signal
    
    Returns:
    - Dictionary with signal, stop loss, and additional info
    """
    # Check for sufficient data
    required_length = max(short_window, long_window, trend_window, atr_period) + confirmation_days + 5
    if len(df) < required_length:
        return {"signal": "INSUFFICIENT_DATA", "stop_loss": None, "confidence": 0}
    
    # Calculate EMAs with configurable windows
    df['short_ema'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate crossover points
    df['ema_crossover'] = ((df['short_ema'] > df['long_ema']) & 
                           (df['short_ema'].shift() <= df['long_ema'].shift())).astype(int)
    df['ema_crossunder'] = ((df['short_ema'] < df['long_ema']) & 
                            (df['short_ema'].shift() >= df['long_ema'].shift())).astype(int)
    
    # Calculate multiple timeframe trend filters
    df['trend_sma'] = df['close'].rolling(window=trend_window).mean()
    df['trend_sma_slope'] = df['trend_sma'].diff(5) > 0  # Positive slope in trend
    
    # Calculate ATR for volatility-based stop loss
    df = calculate_atr(df, period=atr_period)
    
    # Add RSI for momentum confirmation
    df = calculate_rsi(df, period=rsi_period)
    
    # Add volume analysis if volume column exists
    has_volume = 'volume' in df.columns
    if has_volume:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Variables for signal confirmation
    current_idx = len(df) - 1
    
    # Check for consistent buy signals (crossovers) over confirmation period
    buy_confirmed = sum(df['ema_crossover'].iloc[-(confirmation_days+1):]) >= 1
    # Check for consistent sell signals (crossunders) over confirmation period
    sell_confirmed = sum(df['ema_crossunder'].iloc[-(confirmation_days+1):]) >= 1
    
    # Current values for signal generation
    current = {
        'close': df['close'].iloc[-1],
        'short_ema': df['short_ema'].iloc[-1],
        'long_ema': df['long_ema'].iloc[-1],
        'trend_sma': df['trend_sma'].iloc[-1],
        'trend_slope': df['trend_sma_slope'].iloc[-1],
        'atr': df['ATR'].iloc[-1],
        'rsi': df['RSI'].iloc[-1],
        'volume_high': has_volume and df['volume_ratio'].iloc[-1] > volume_factor
    }
    
    # Calculate price to EMA distance (for overbought/oversold)
    ema_distance_pct = abs(current['close'] - current['long_ema']) / current['long_ema'] * 100
    
    # Calculate dynamic ATR multiplier based on volatility
    # Higher volatility = larger stop to avoid getting stopped out by noise
    volatility_ratio = df['ATR'].iloc[-1] / df['ATR'].rolling(window=100).mean().iloc[-1]
    dynamic_atr_multiplier = min(3.0, max(1.0, atr_multiplier * volatility_ratio))
    
    # Calculate confidence score (0-100%)
    def calculate_confidence(signal_type):
        base_score = 50
        
        # Trend alignment score (0-20)
        if signal_type == "BUY":
            trend_score = 20 if current['trend_slope'] else 0
            rsi_score = max(0, min(20, (50 - current['rsi']) / 2.5))
            ema_distance_score = max(0, min(10, (10 - ema_distance_pct)))  # Not too extended
        else:  # SELL
            trend_score = 20 if not current['trend_slope'] else 0
            rsi_score = max(0, min(20, (current['rsi'] - 50) / 2.5))
            ema_distance_score = max(0, min(10, (10 - ema_distance_pct)))  # Not too extended
        
        # Volume confirmation (0-10)
        volume_score = 10 if current['volume_high'] else 0
        
        # Calculate total confidence
        confidence = base_score + trend_score + rsi_score + ema_distance_score + volume_score
        return min(100, confidence)
    
    # Enhanced signal generation with multiple confirmation filters
    if (buy_confirmed and 
        current['short_ema'] > current['long_ema'] and
        current['close'] > current['trend_sma'] and
        # Not extremely overbought
        current['rsi'] < 70 and 
        # Not too extended from moving average
        ema_distance_pct < 10):
        
        # Calculate more sophisticated stop loss
        # Use larger ATR multiplier for higher volatility
        stop_loss = current['close'] - (current['atr'] * dynamic_atr_multiplier)
        
        # Calculate take profit levels based on ATR
        take_profit_1 = current['close'] + (current['atr'] * 2)
        take_profit_2 = current['close'] + (current['atr'] * 3)
        
        # Calculate confidence score
        confidence = calculate_confidence("BUY")
        
        return {
            "signal": "BUY", 
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "confidence": confidence,
            "atr_multiplier": dynamic_atr_multiplier
        }
        
    elif (sell_confirmed and 
          current['short_ema'] < current['long_ema'] and
          current['close'] < current['trend_sma'] and
          # Not extremely oversold
          current['rsi'] > 30 and
          # Not too extended from moving average
          ema_distance_pct < 10):
        
        # Calculate more sophisticated stop loss
        # Use larger ATR multiplier for higher volatility
        stop_loss = current['close'] + (current['atr'] * dynamic_atr_multiplier)
        
        # Calculate take profit levels based on ATR
        take_profit_1 = current['close'] - (current['atr'] * 2)
        take_profit_2 = current['close'] - (current['atr'] * 3)
        
        # Calculate confidence score
        confidence = calculate_confidence("SELL")
        
        return {
            "signal": "SELL", 
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "confidence": confidence,
            "atr_multiplier": dynamic_atr_multiplier
        }
        
    else:
        return {"signal": "HOLD", "stop_loss": None, "confidence": 0}

def detect_ma_squeeze(df, bb_length=20, kc_length=20, kc_mult=2.0):
    """
    Detect a moving average squeeze, which often precedes significant price moves.
    Combines Bollinger Bands and Keltner Channels.
    
    A squeeze occurs when Bollinger Bands contract inside Keltner Channels.
    """
    # Calculate Bollinger Bands
    df['ma'] = df['close'].rolling(window=bb_length).mean()
    df['bb_std'] = df['close'].rolling(window=bb_length).std()
    df['bb_upper'] = df['ma'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['ma'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma']
    
    # Calculate Keltner Channels
    df['kc_atr'] = calculate_atr(df, period=kc_length)['ATR']
    df['kc_upper'] = df['ma'] + (df['kc_atr'] * kc_mult)
    df['kc_lower'] = df['ma'] - (df['kc_atr'] * kc_mult)
    
    # Identify squeeze
    df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
    df['squeeze_off'] = (~df['squeeze_on']) & (df['squeeze_on'].shift(1))
    
    # Return if we're currently in a squeeze or just exiting one
    current_squeeze = df['squeeze_on'].iloc[-1]
    exiting_squeeze = df['squeeze_off'].iloc[-1]
    
    return {
        "in_squeeze": current_squeeze,
        "exiting_squeeze": exiting_squeeze,
        "bb_width": df['bb_width'].iloc[-1],
        "days_in_squeeze": df['squeeze_on'].rolling(window=100).sum().iloc[-1] if current_squeeze else 0
    }
