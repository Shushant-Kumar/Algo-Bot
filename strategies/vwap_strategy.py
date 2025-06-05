import pandas as pd
import numpy as np

def calculate_vwap(df, reset_period='D'):
    """
    Calculate VWAP for the given DataFrame with proper period reset.
    
    Parameters:
    - df: DataFrame with OHLCV data and datetime index
    - reset_period: Frequency to reset VWAP calculation ('D' for daily, 'W' for weekly)
    
    Returns:
    - DataFrame with VWAP calculated
    """
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index for VWAP calculation")
    
    # Create a reset marker based on the reset period
    df['period_marker'] = df.index.to_period(reset_period)
    
    # Calculate typical price
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP components
    df['tp_x_vol'] = df['tp'] * df['volume']
    
    # Group by reset period and calculate cumulative values
    groups = df.groupby('period_marker')
    df['cum_vol'] = groups['volume'].cumsum()
    df['cum_tp_x_vol'] = groups['tp_x_vol'].cumsum()
    
    # Calculate VWAP
    df['VWAP'] = df['cum_tp_x_vol'] / df['cum_vol']
    
    # Clean up intermediate columns
    df.drop(['period_marker', 'tp', 'tp_x_vol', 'cum_vol', 'cum_tp_x_vol'], axis=1, inplace=True)
    
    return df

def calculate_vwap_multi_timeframe(df, primary_tf='D', higher_tf='W'):
    """
    Calculate VWAP for multiple timeframes.
    
    Parameters:
    - df: DataFrame with OHLCV data and datetime index
    - primary_tf: Primary timeframe for VWAP ('D' for daily)
    - higher_tf: Higher timeframe for VWAP ('W' for weekly)
    
    Returns:
    - DataFrame with VWAP for both timeframes
    """
    # Calculate primary timeframe VWAP
    df = calculate_vwap(df, reset_period=primary_tf)
    
    # Calculate higher timeframe VWAP
    df = calculate_vwap(df.copy(), reset_period=higher_tf)
    df.rename(columns={'VWAP': 'Higher_VWAP'}, inplace=True)
    
    return df

def calculate_macd(df):
    """
    Calculate MACD and Signal Line for the given DataFrame.
    """
    df['12_EMA'] = df['close'].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

def calculate_vwap_slope(df, window=10):
    """
    Calculate the slope of VWAP to determine trend direction.
    
    Parameters:
    - df: DataFrame with VWAP calculated
    - window: Window size for slope calculation
    
    Returns:
    - DataFrame with VWAP slope added
    """
    # Calculate VWAP change over window
    df['VWAP_Change'] = df['VWAP'].diff(window)
    
    # Calculate percentage change for normalization
    df['VWAP_Slope'] = df['VWAP_Change'] / df['VWAP'].shift(window) * 100
    
    return df

def vwap_strategy_with_filters(df, volume_threshold_multiplier=1.5, vwap_distance_limit=1.5,
                             macd_span_fast=12, macd_span_slow=26, macd_span_signal=9):
    """
    Enhanced VWAP Strategy with multi-timeframe VWAP, MACD confirmation, and volume thresholds.
    
    Parameters:
    - df: DataFrame with OHLCV data and datetime index
    - volume_threshold_multiplier: Multiplier for volume threshold
    - vwap_distance_limit: Maximum distance from VWAP as percentage
    - macd_span_fast: Fast period for MACD
    - macd_span_slow: Slow period for MACD
    - macd_span_signal: Signal period for MACD
    
    Returns:
    - dict: Signal details including type, strength, and stop levels
    """
    # Check for sufficient data
    required_rows = max(50, macd_span_slow * 2)
    if len(df) < required_rows:
        return {"signal": "INSUFFICIENT_DATA", "strength": 0}
    
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            return {"signal": "ERROR", "message": "DataFrame requires datetime index"}
    
    try:
        # Calculate VWAP for both daily and weekly timeframes
        df = calculate_vwap(df.copy(), 'D')
        
        # Try to calculate weekly VWAP if we have enough data
        if len(df) >= 10:  # Need reasonable amount of data for weekly
            df_weekly = calculate_vwap(df.copy(), 'W')
            df['Weekly_VWAP'] = df_weekly['VWAP']
        else:
            df['Weekly_VWAP'] = df['VWAP']  # Fallback
        
        # Calculate VWAP slope for trend direction
        df = calculate_vwap_slope(df)
        
        # Calculate MACD with customizable spans
        df['Fast_EMA'] = df['close'].ewm(span=macd_span_fast, adjust=False).mean()
        df['Slow_EMA'] = df['close'].ewm(span=macd_span_slow, adjust=False).mean()
        df['MACD'] = df['Fast_EMA'] - df['Slow_EMA']
        df['Signal_Line'] = df['MACD'].ewm(span=macd_span_signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Calculate average volume for threshold
        df['Average_Volume'] = df['volume'].rolling(window=20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_threshold = df['Average_Volume'].iloc[-1] * volume_threshold_multiplier
        
        # Calculate distance from VWAP as percentage
        df['VWAP_Distance'] = (df['close'] - df['VWAP']) / df['VWAP'] * 100
        
        # Calculate standard deviation of VWAP distance for volatility-adjusted decisions
        df['VWAP_Dist_StdDev'] = df['VWAP_Distance'].rolling(window=20).std()
        
        # Detect crossovers
        df['Above_VWAP'] = df['close'] > df['VWAP']
        df['VWAP_Crossover_Up'] = (df['Above_VWAP'] & ~df['Above_VWAP'].shift(1))
        df['VWAP_Crossover_Down'] = (~df['Above_VWAP'] & df['Above_VWAP'].shift(1))
        
        df['MACD_Above_Signal'] = df['MACD'] > df['Signal_Line']
        df['MACD_Crossover_Up'] = (df['MACD_Above_Signal'] & ~df['MACD_Above_Signal'].shift(1))
        df['MACD_Crossover_Down'] = (~df['MACD_Above_Signal'] & df['MACD_Above_Signal'].shift(1))
        
        # Current metrics
        current = {
            'close': df['close'].iloc[-1],
            'vwap': df['VWAP'].iloc[-1],
            'weekly_vwap': df['Weekly_VWAP'].iloc[-1],
            'vwap_distance': df['VWAP_Distance'].iloc[-1],
            'vwap_dist_std': df['VWAP_Dist_StdDev'].iloc[-1],
            'vwap_slope': df['VWAP_Slope'].iloc[-1],
            'macd': df['MACD'].iloc[-1],
            'signal': df['Signal_Line'].iloc[-1],
            'histogram': df['MACD_Histogram'].iloc[-1],
            'volume': current_volume,
            'avg_volume': df['Average_Volume'].iloc[-1],
            'recent_vwap_cross_up': df['VWAP_Crossover_Up'].iloc[-5:].any(),
            'recent_vwap_cross_down': df['VWAP_Crossover_Down'].iloc[-5:].any(),
            'recent_macd_cross_up': df['MACD_Crossover_Up'].iloc[-5:].any(),
            'recent_macd_cross_down': df['MACD_Crossover_Down'].iloc[-5:].any(),
        }
        
        # Calculate volatility for stop loss (ATR-like)
        df['hl_range'] = df['high'] - df['low']
        volatility = df['hl_range'].rolling(window=14).mean().iloc[-1]
        
        # Calculate signal strength (0-100%)
        def calculate_strength(signal_type):
            base_score = 50
            
            # VWAP alignment score (0-20)
            if signal_type == "BUY":
                vwap_alignment = 20 if current['close'] > current['weekly_vwap'] else 10
                vwap_trend_score = min(15, max(0, current['vwap_slope'] * 3))
            else:
                vwap_alignment = 20 if current['close'] < current['weekly_vwap'] else 10
                vwap_trend_score = min(15, max(0, -current['vwap_slope'] * 3))
            
            # MACD strength (0-15)
            macd_strength = min(15, abs(current['histogram']) * 10)
            
            # Volume confirmation (0-15)
            volume_ratio = current['volume'] / current['avg_volume']
            volume_score = min(15, (volume_ratio - 1) * 10)
            
            # Distance from VWAP - not too extended (0-15)
            # Lower score if too far from VWAP
            if abs(current['vwap_distance']) > vwap_distance_limit * 2:
                distance_score = 0
            elif abs(current['vwap_distance']) > vwap_distance_limit:
                distance_score = 7
            else:
                distance_score = 15
                
            total = base_score + vwap_alignment + vwap_trend_score + macd_strength + volume_score + distance_score
            return min(100, total)
        
        # Generate signals with enhanced filters
        if (
            current['close'] > current['vwap'] and  # Above primary VWAP
            (current['close'] > current['weekly_vwap'] or current['recent_vwap_cross_up']) and  # Above higher timeframe VWAP or recent crossover
            current['macd'] > current['signal'] and  # MACD above signal line
            (current['recent_macd_cross_up'] or current['histogram'] > 0) and  # Recent MACD crossover or positive momentum
            abs(current['vwap_distance']) < vwap_distance_limit * current['vwap_dist_std'] and  # Not too extended from VWAP
            current['volume'] > volume_threshold and  # Volume confirmation
            current['vwap_slope'] > 0  # Upward VWAP slope
        ):
            strength = calculate_strength("BUY")
            
            # Calculate stop loss and take profit
            stop_loss = min(current['vwap'], current['close'] - volatility * 1.5)
            take_profit = current['close'] + (current['close'] - stop_loss) * 2  # 2:1 reward-risk ratio
            
            return {
                "signal": "BUY", 
                "strength": strength, 
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "vwap_distance": current['vwap_distance'],
                "vwap_slope": current['vwap_slope'],
                "volume_ratio": current['volume'] / current['avg_volume']
            }
            
        elif (
            current['close'] < current['vwap'] and  # Below primary VWAP
            (current['close'] < current['weekly_vwap'] or current['recent_vwap_cross_down']) and  # Below higher timeframe VWAP or recent crossover
            current['macd'] < current['signal'] and  # MACD below signal line
            (current['recent_macd_cross_down'] or current['histogram'] < 0) and  # Recent MACD crossover or negative momentum
            abs(current['vwap_distance']) < vwap_distance_limit * current['vwap_dist_std'] and  # Not too extended from VWAP
            current['volume'] > volume_threshold and  # Volume confirmation
            current['vwap_slope'] < 0  # Downward VWAP slope
        ):
            strength = calculate_strength("SELL")
            
            # Calculate stop loss and take profit
            stop_loss = max(current['vwap'], current['close'] + volatility * 1.5)
            take_profit = current['close'] - (stop_loss - current['close']) * 2  # 2:1 reward-risk ratio
            
            return {
                "signal": "SELL", 
                "strength": strength, 
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "vwap_distance": current['vwap_distance'],
                "vwap_slope": current['vwap_slope'],
                "volume_ratio": current['volume'] / current['avg_volume']
            }
            
        else:
            # Check if we're close to a signal
            potential = "NONE"
            potential_strength = 0
            
            # If price is near VWAP and other conditions are aligning
            if abs(current['vwap_distance']) < 0.5 and current['volume'] > current['avg_volume']:
                if current['macd'] > current['signal'] and current['vwap_slope'] > 0:
                    potential = "BUY"
                    potential_strength = 30
                elif current['macd'] < current['signal'] and current['vwap_slope'] < 0:
                    potential = "SELL"
                    potential_strength = 30
            
            return {
                "signal": "HOLD", 
                "potential": potential, 
                "strength": potential_strength
            }
            
    except Exception as e:
        return {"signal": "ERROR", "message": str(e)}