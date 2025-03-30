import pandas as pd

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

def moving_average_strategy_with_filters(df):
    """
    Enhanced Moving Average Strategy with EMA, ATR-based stop loss, and trend filter.
    """
    short_window = 5
    long_window = 20
    trend_window = 50  # Trend filter using 50-SMA

    # Calculate EMAs
    df['short_ema'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=long_window, adjust=False).mean()

    # Calculate trend filter (50-SMA)
    df['trend_sma'] = df['close'].rolling(window=trend_window).mean()

    # Calculate ATR
    df = calculate_atr(df)

    # Generate signals with filters
    if (
        df['short_ema'].iloc[-1] > df['long_ema'].iloc[-1] and  # EMA crossover
        df['close'].iloc[-1] > df['trend_sma'].iloc[-1]  # Above trend filter
    ):
        stop_loss = df['close'].iloc[-1] - df['ATR'].iloc[-1]  # ATR-based stop loss
        return {"signal": "BUY", "stop_loss": stop_loss}
    elif (
        df['short_ema'].iloc[-1] < df['long_ema'].iloc[-1] and  # EMA crossover
        df['close'].iloc[-1] < df['trend_sma'].iloc[-1]  # Below trend filter
    ):
        stop_loss = df['close'].iloc[-1] + df['ATR'].iloc[-1]  # ATR-based stop loss
        return {"signal": "SELL", "stop_loss": stop_loss}
    else:
        return {"signal": "HOLD", "stop_loss": None}
