def calculate_vwap(df):
    """
    Calculate VWAP for the given DataFrame.
    """
    df['Cumulative_TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['Cumulative_Volume'] = df['volume'].cumsum()
    df['Cumulative_TPV'] = (df['Cumulative_TP'] * df['volume']).cumsum()
    df['VWAP'] = df['Cumulative_TPV'] / df['Cumulative_Volume']
    return df

def calculate_macd(df):
    """
    Calculate MACD and Signal Line for the given DataFrame.
    """
    df['12_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def vwap_strategy_with_filters(df, volume_threshold_multiplier=1.5):
    """
    Enhanced VWAP Strategy with multi-timeframe VWAP, MACD confirmation, and volume thresholds.
    """
    # Calculate 5-minute VWAP
    df = calculate_vwap(df)

    # Calculate 1-hour VWAP (resample to 1-hour timeframe)
    df['1H_Cumulative_TP'] = df['Cumulative_TP'].resample('1H').sum().fillna(0)
    df['1H_Cumulative_Volume'] = df['volume'].resample('1H').sum().cumsum().fillna(0)
    df['1H_Cumulative_TPV'] = df['Cumulative_TPV'].resample('1H').sum().cumsum().fillna(0)
    df['1H_VWAP'] = df['1H_Cumulative_TPV'] / df['1H_Cumulative_Volume']

    # Calculate MACD
    df = calculate_macd(df)

    # Calculate average volume for volume threshold
    df['Average_Volume'] = df['volume'].rolling(window=20).mean()
    current_volume = df['volume'].iloc[-1]
    volume_threshold = df['Average_Volume'].iloc[-1] * volume_threshold_multiplier

    # Generate signals with filters
    if (
        df['close'].iloc[-1] > df['VWAP'].iloc[-1] and  # Above 5-minute VWAP
        df['close'].iloc[-1] > df['1H_VWAP'].iloc[-1] and  # Above 1-hour VWAP
        df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and  # MACD bullish crossover
        current_volume > volume_threshold  # Volume exceeds threshold
    ):
        return "BUY"
    elif (
        df['close'].iloc[-1] < df['VWAP'].iloc[-1] and  # Below 5-minute VWAP
        df['close'].iloc[-1] < df['1H_VWAP'].iloc[-1] and  # Below 1-hour VWAP
        df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and  # MACD bearish crossover
        current_volume > volume_threshold  # Volume exceeds threshold
    ):
        return "SELL"
    else:
        return "HOLD"