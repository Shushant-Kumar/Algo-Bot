import pandas as pd

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

def bollinger_bands_strategy_with_rsi(df):
    """
    Bollinger Bands Strategy with RSI and 50-SMA trend filter.
    """
    # Calculate Bollinger Bands
    df['20_SMA'] = df['close'].rolling(window=20).mean()
    df['Upper_Band'] = df['20_SMA'] + (df['close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['20_SMA'] - (df['close'].rolling(window=20).std() * 2)

    # Calculate 50-SMA as a trend filter
    df['50_SMA'] = df['close'].rolling(window=50).mean()

    # Calculate RSI
    df = calculate_rsi(df)

    # Generate signals
    if df['close'].iloc[-1] > df['Upper_Band'].iloc[-1] and df['RSI'].iloc[-1] > 70 and df['close'].iloc[-1] > df['50_SMA'].iloc[-1]:
        return "SELL"
    elif df['close'].iloc[-1] < df['Lower_Band'].iloc[-1] and df['RSI'].iloc[-1] < 30 and df['close'].iloc[-1] < df['50_SMA'].iloc[-1]:
        return "BUY"
    else:
        return "HOLD"