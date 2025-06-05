import pandas as pd
import numpy as np

def calculate_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI) for the given DataFrame.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

def bollinger_bands_strategy_with_rsi(df, bb_period=20, bb_std=2, rsi_period=14, 
                                     rsi_overbought=70, rsi_oversold=30, sma_period=50, 
                                     volume_factor=1.5, confirmation_days=2):
    """
    Enhanced Bollinger Bands Strategy with RSI, SMA trend filter, volume confirmation,
    and additional precision indicators.
    
    Parameters:
    - bb_period: Period for Bollinger Bands calculation
    - bb_std: Standard deviation multiplier for bands
    - rsi_period: Period for RSI calculation
    - rsi_overbought: RSI threshold for overbought condition
    - rsi_oversold: RSI threshold for oversold condition
    - sma_period: Period for SMA trend filter
    - volume_factor: Factor to determine significant volume increase
    - confirmation_days: Number of days required to confirm a signal
    """
    # Handle edge cases
    if len(df) < max(bb_period, rsi_period, sma_period) + confirmation_days + 1:
        return "INSUFFICIENT_DATA"
    
    # Calculate Bollinger Bands
    df['SMA'] = df['close'].rolling(window=bb_period).mean()
    std_dev = df['close'].rolling(window=bb_period).std()
    df['Upper_Band'] = df['SMA'] + (std_dev * bb_std)
    df['Lower_Band'] = df['SMA'] - (std_dev * bb_std)
    
    # Calculate Band Width - measure of volatility
    df['Band_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA']
    
    # Calculate %B - position of price relative to bands (0 to 1)
    df['%B'] = (df['close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # Calculate Squeeze - identifies when bands are narrowing
    df['Squeeze'] = df['Band_Width'].rolling(window=20).mean() > df['Band_Width']
    
    # Calculate trend filter
    df[f'{sma_period}_SMA'] = df['close'].rolling(window=sma_period).mean()
    
    # Calculate RSI with the specified period
    df = calculate_rsi(df, period=rsi_period)
    
    # Calculate volume baseline
    df['Volume_MA'] = df['volume'].rolling(window=20).mean() if 'volume' in df.columns else None
    df['Volume_Surge'] = df['volume'] > (df['Volume_MA'] * volume_factor) if 'volume' in df.columns else False
    
    # Generate signals with confirmation
    current_idx = len(df) - 1
    
    # Check for consistent sell signals over confirmation period
    sell_confirmed = True
    for i in range(confirmation_days):
        idx = current_idx - i
        if idx < 0 or not (df['close'].iloc[idx] > df['Upper_Band'].iloc[idx] and 
                          df['RSI'].iloc[idx] > rsi_overbought and 
                          df['close'].iloc[idx] > df[f'{sma_period}_SMA'].iloc[idx]):
            sell_confirmed = False
            break
    
    # Check for consistent buy signals over confirmation period
    buy_confirmed = True
    for i in range(confirmation_days):
        idx = current_idx - i
        if idx < 0 or not (df['close'].iloc[idx] < df['Lower_Band'].iloc[idx] and 
                          df['RSI'].iloc[idx] < rsi_oversold and 
                          df['close'].iloc[idx] < df[f'{sma_period}_SMA'].iloc[idx]):
            buy_confirmed = False
            break
    
    # Strong signals with volume confirmation (if volume data is available)
    volume_confirmation = True if 'volume' not in df.columns else df['Volume_Surge'].iloc[-1]
    
    # Decision logic with enhanced precision
    if (sell_confirmed and 
        df['%B'].iloc[-1] > 1 and  # Price extremely above upper band
        volume_confirmation):
        return "SELL"
    elif (buy_confirmed and 
          df['%B'].iloc[-1] < 0 and  # Price extremely below lower band
          volume_confirmation and
          df['Squeeze'].iloc[-1]):  # Coming out of a squeeze (potential for strong move)
        return "BUY"
    else:
        return "HOLD"

def get_signal_strength(df):
    """
    Calculate the strength of the signal (0-100%).
    """
    # Latest values
    rsi = df['RSI'].iloc[-1]
    percent_b = df['%B'].iloc[-1]
    
    if percent_b > 1:  # Sell signal
        # Scale from 0-100% where 100% is strongest sell
        rsi_strength = min(100, max(0, (rsi - 70) * (100/30)))  # RSI 70->100 maps to 0->100%
        band_strength = min(100, max(0, (percent_b - 1) * 100))  # %B 1->2 maps to 0->100%
        return -1 * ((rsi_strength + band_strength) / 2)  # Negative for sell
    
    elif percent_b < 0:  # Buy signal
        # Scale from 0-100% where 100% is strongest buy
        rsi_strength = min(100, max(0, (30 - rsi) * (100/30)))  # RSI 30->0 maps to 0->100%
        band_strength = min(100, max(0, -percent_b * 100))  # %B 0->-1 maps to 0->100%
        return (rsi_strength + band_strength) / 2  # Positive for buy
    
    return 0  # Neutral/hold