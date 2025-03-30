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

def macd_strategy_with_filters(df):
    """
    Enhanced MACD Strategy with zero-line filter, histogram momentum confirmation, and RSI filter.
    """
    # Calculate MACD and Signal Line
    df['12_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal_Line']

    # Calculate RSI
    df = calculate_rsi(df)

    # Generate signals with filters
    if (
        df['MACD'].iloc[-1] > 0 and  # MACD above zero
        df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and  # MACD crossover above Signal Line
        df['Histogram'].iloc[-1] > 0 and  # Positive momentum
        df['RSI'].iloc[-1] < 30  # RSI indicates oversold
    ):
        return "BUY"
    elif (
        df['MACD'].iloc[-1] < 0 and  # MACD below zero
        df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and  # MACD crossover below Signal Line
        df['Histogram'].iloc[-1] < 0 and  # Negative momentum
        df['RSI'].iloc[-1] > 70  # RSI indicates overbought
    ):
        return "SELL"
    else:
        return "HOLD"