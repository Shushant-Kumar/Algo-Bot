# strategies/rsi_strategy.py

import pandas as pd

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

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

def detect_divergence(df, window=3):
    """
    Detect divergence using pivot points over a specified window.
    """
    price_highs, price_lows = find_pivots(df['close'], window)
    rsi_highs, rsi_lows = find_pivots(df['rsi'], window)

    bullish_divergence = (
        price_lows & rsi_lows &  # Both price and RSI have pivot lows
        (df['close'] > df['close'].shift(window)) &  # Higher low in price
        (df['rsi'] < df['rsi'].shift(window))  # Lower low in RSI
    )
    bearish_divergence = (
        price_highs & rsi_highs &  # Both price and RSI have pivot highs
        (df['close'] < df['close'].shift(window)) &  # Lower high in price
        (df['rsi'] > df['rsi'].shift(window))  # Higher high in RSI
    )

    return bullish_divergence.any(), bearish_divergence.any()

def rsi_strategy_with_filters(df):
    """
    Enhanced RSI Strategy with divergence, MACD confirmation, and trend filter.
    """
    # Calculate RSI
    df['rsi'] = calculate_rsi(df)

    # Calculate MACD
    df = calculate_macd(df)

    # Calculate trend filter (50-SMA)
    df['50_SMA'] = df['close'].rolling(window=50).mean()

    # Detect divergence using pivot points
    bullish_divergence, bearish_divergence = detect_divergence(df)

    # Take the latest RSI and MACD values
    latest_rsi = df['rsi'].iloc[-1]
    latest_macd = df['MACD'].iloc[-1]
    latest_signal_line = df['Signal_Line'].iloc[-1]
    latest_histogram = df['Histogram'].iloc[-1]
    latest_price = df['close'].iloc[-1]
    latest_trend = df['50_SMA'].iloc[-1]

    # Generate signals with filters
    if (
        latest_rsi < 30 and  # RSI oversold
        latest_macd > latest_signal_line and  # MACD bullish crossover
        latest_histogram > 0 and  # Positive momentum
        latest_price > latest_trend and  # Above trend filter
        bullish_divergence  # Confirmed bullish divergence
    ):
        return "BUY"
    elif (
        latest_rsi > 70 and  # RSI overbought
        latest_macd < latest_signal_line and  # MACD bearish crossover
        latest_histogram < 0 and  # Negative momentum
        latest_price < latest_trend and  # Below trend filter
        bearish_divergence  # Confirmed bearish divergence
    ):
        return "SELL"
    else:
        return "HOLD"
