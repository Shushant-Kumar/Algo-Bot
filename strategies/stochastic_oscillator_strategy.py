def stochastic_oscillator_strategy_with_filters(df, support_level, resistance_level):
    """
    Enhanced Stochastic Oscillator Strategy with support/resistance levels, trend filter, and divergence signals.
    """
    # Calculate Stochastic Oscillator
    df['14_High'] = df['high'].rolling(window=14).max()
    df['14_Low'] = df['low'].rolling(window=14).min()
    df['%K'] = (df['close'] - df['14_Low']) / (df['14_High'] - df['14_Low']) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Calculate 50-SMA as a trend filter
    df['50_SMA'] = df['close'].rolling(window=50).mean()

    # Check for divergence
    bullish_divergence = (
        df['close'].iloc[-2] < df['close'].iloc[-1] and  # Higher low in price
        df['%K'].iloc[-2] > df['%K'].iloc[-1]  # Lower low in %K
    )
    bearish_divergence = (
        df['close'].iloc[-2] > df['close'].iloc[-1] and  # Lower high in price
        df['%K'].iloc[-2] < df['%K'].iloc[-1]  # Higher high in %K
    )

    # Check if price is near support or resistance
    near_support = df['close'].iloc[-1] <= support_level * 1.02  # Within 2% of support
    near_resistance = df['close'].iloc[-1] >= resistance_level * 0.98  # Within 2% of resistance

    # Generate signals with filters
    if (
        df['%K'].iloc[-1] > df['%D'].iloc[-1] and  # %K crosses above %D
        df['%K'].iloc[-2] <= df['%D'].iloc[-2] and
        df['close'].iloc[-1] > df['50_SMA'].iloc[-1] and  # Above trend filter
        near_support and  # Near support level
        bullish_divergence  # Confirmed bullish divergence
    ):
        return "BUY"
    elif (
        df['%K'].iloc[-1] < df['%D'].iloc[-1] and  # %K crosses below %D
        df['%K'].iloc[-2] >= df['%D'].iloc[-2] and
        df['close'].iloc[-1] < df['50_SMA'].iloc[-1] and  # Below trend filter
        near_resistance and  # Near resistance level
        bearish_divergence  # Confirmed bearish divergence
    ):
        return "SELL"
    else:
        return "HOLD"