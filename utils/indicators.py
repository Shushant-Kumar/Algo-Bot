def calculate_atr(df, period=14):
    # Ensure required columns exist
    if not {'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
    
    # Calculate True Range (TR) components
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    
    # Calculate TR and ATR
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    # Drop intermediate columns to keep the DataFrame clean
    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True)
    
    return df