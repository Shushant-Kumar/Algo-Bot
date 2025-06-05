from config import PER_STOCK_CAPITAL_LIMIT, MIN_SIGNAL_STRENGTH
from decimal import Decimal, ROUND_DOWN
from utils.logger import Logger

# Initialize enhanced logger
logger = Logger()

def extract_signal_strength(signal_data):
    """
    Extract signal strength from various signal formats.
    
    Parameters:
    - signal_data: Can be float/int value, string signal, or dict with signal details
    
    Returns:
    - tuple: (signal_value, confidence_score)
    """
    # If it's a numeric value, use directly
    if isinstance(signal_data, (int, float, Decimal)):
        return float(signal_data), 100.0
    
    # If it's a string (BUY/SELL/HOLD)
    elif isinstance(signal_data, str):
        if signal_data == "BUY":
            return 1.0, 100.0
        elif signal_data == "SELL":
            return -1.0, 100.0
        else:  # HOLD or other
            return 0.0, 0.0
    
    # If it's a dictionary (from enhanced strategies)
    elif isinstance(signal_data, dict):
        # Extract signal type
        signal_type = signal_data.get('signal', signal_data.get('signal_type', 'HOLD'))
        
        # Extract confidence/strength
        confidence = signal_data.get('strength', signal_data.get('confidence', 50.0))
        
        # Convert to numeric value
        if signal_type == "BUY":
            signal_value = 1.0
        elif signal_type == "SELL":
            signal_value = -1.0
        else:  # HOLD or other
            signal_value = 0.0
            
        # Apply confidence scaling
        return signal_value, float(confidence)
    
    # Default case
    else:
        return 0.0, 0.0

def calculate_allocation(signal, total_weight, remaining_capital):
    """Calculate the allocation for a stock based on its signal weight and remaining capital."""
    return (Decimal(signal) / Decimal(total_weight)) * remaining_capital if total_weight else Decimal(0)

def allocate_capital(stock_signals, total_capital, risk_adjusted=True):
    """
    Allocate capital among stocks based on their signal strength.
    
    Parameters:
    - stock_signals: Dict of {stock: signal_data} where signal_data can be numeric or dict
    - total_capital: Total capital to allocate
    - risk_adjusted: Whether to adjust allocation based on signal confidence
    
    Returns:
    - Dict of {stock: allocation} with float values
    """
    allocations = {}
    remaining_capital = Decimal(total_capital).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    MIN_ALLOCATION_THRESHOLD = Decimal("0.01")  
    
    # Process and normalize signals
    normalized_signals = {}
    signal_confidences = {}
    
    for stock, signal_data in stock_signals.items():
        signal_value, confidence = extract_signal_strength(signal_data)
        
        # Only consider BUY signals with sufficient confidence
        if signal_value > 0 and confidence >= MIN_SIGNAL_STRENGTH:
            # Scale signal by confidence if risk adjustment is enabled
            if risk_adjusted:
                normalized_signals[stock] = Decimal(str(signal_value * (confidence / 100.0)))
            else:
                normalized_signals[stock] = Decimal(str(signal_value))
                
            signal_confidences[stock] = confidence
            
    # Log the normalized signals
    logger.debug(f"Normalized signals: {normalized_signals}")
    
    # ✅ **Step 1: Remove zero-signal stocks early**
    stock_signals = {stock: signal for stock, signal in normalized_signals.items() if signal > 0}
    if not stock_signals:
        logger.warning("⚠️ All stock signals are zero or removed. No allocation will be made.")
        return {stock: 0.0 for stock in normalized_signals}  

    # ✅ **Step 2: Compute Total Weight & Exit Early if Needed**
    total_weight = sum(stock_signals.values())
    if total_weight == 0:
        return {stock: 0.0 for stock in stock_signals}  # No allocation possible

    # ✅ **Step 3: First Pass – Allocate Capital Based on Weights**
    initial_allocations = {
        stock: min(
            calculate_allocation(signal, total_weight, remaining_capital).quantize(Decimal("0.01"), rounding=ROUND_DOWN),
            Decimal(PER_STOCK_CAPITAL_LIMIT)
        )
        for stock, signal in stock_signals.items()
    }

    # ✅ **Step 4: Update Remaining Capital**
    allocated_capital = sum(initial_allocations.values())
    remaining_capital = (remaining_capital - allocated_capital).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    # ✅ **Step 5: Find Eligible Stocks for Redistribution**
    if risk_adjusted:
        # Prioritize highest confidence signals when redistributing
        eligible_stocks = sorted(
            (stock for stock in stock_signals if initial_allocations[stock] < Decimal(PER_STOCK_CAPITAL_LIMIT)),
            key=lambda s: signal_confidences.get(s, 0),  # Prioritize highest confidence signals
            reverse=True
        )
    else:
        # Original prioritization logic
        eligible_stocks = sorted(
            (stock for stock in stock_signals if initial_allocations[stock] < Decimal(PER_STOCK_CAPITAL_LIMIT)),
            key=lambda s: stock_signals[s] / total_weight,  # Prioritize highest signal-to-weight ratio
            reverse=True
        )

    # ✅ **Step 6: Efficient Capital Redistribution**
    for stock in eligible_stocks:
        if remaining_capital < MIN_ALLOCATION_THRESHOLD:
            break  

        max_possible_allocation = Decimal(PER_STOCK_CAPITAL_LIMIT) - initial_allocations[stock]
        additional_allocation = min(max_possible_allocation, remaining_capital)

        if additional_allocation >= MIN_ALLOCATION_THRESHOLD:
            initial_allocations[stock] += additional_allocation
            remaining_capital = (remaining_capital - additional_allocation).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    # ✅ **Step 7: Assign Any Leftover Capital to the Most Undercapitalized Stock**
    if remaining_capital > 0:
        stock_with_lowest_allocation = min(eligible_stocks, key=initial_allocations.get, default=None)
        if stock_with_lowest_allocation:
            assignable_amount = min(remaining_capital, Decimal(PER_STOCK_CAPITAL_LIMIT) - initial_allocations[stock_with_lowest_allocation])
            initial_allocations[stock_with_lowest_allocation] += assignable_amount
            remaining_capital = (remaining_capital - assignable_amount).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    # Log the final allocations
    logger.info(f"Final capital allocations: {initial_allocations}")
    logger.info(f"Remaining unallocated capital: {remaining_capital}")
    
    # ✅ **Step 9: Add signal confidence to output for reference**
    final_allocations = {
        stock: {
            "amount": float(allocation),
            "confidence": signal_confidences.get(stock, 0),
            "pct_of_total": float((allocation / Decimal(total_capital)) * 100) if total_capital > 0 else 0
        }
        for stock, allocation in initial_allocations.items()
    }
    
    # For backward compatibility, also return simple dict if needed
    if not risk_adjusted:
        return {stock: float(allocation) for stock, allocation in initial_allocations.items()}
    
    return final_allocations
