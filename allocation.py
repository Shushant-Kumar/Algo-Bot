"""
Production-Ready Capital Allocation Module

This module provides advanced capital allocation strategies for algorithmic trading,
optimized for high-frequency intraday trading with enhanced risk management.

Features:
- Dynamic signal strength extraction from multiple formats
- Risk-adjusted allocation based on signal confidence
- Precision decimal arithmetic for accurate calculations
- Comprehensive logging and monitoring
- Production-grade error handling and validation
- Integration with FastStrategy signal formats
"""

from config import PER_STOCK_CAPITAL_LIMIT, MIN_SIGNAL_STRENGTH
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from utils.logger import Logger
import time
from typing import Dict, Union, Tuple, Any, Optional

# Initialize enhanced logger
logger = Logger(console_output=True, file_output=True)

def extract_signal_strength(signal_data: Any) -> Tuple[float, float]:
    """
    Extract signal strength from various signal formats with enhanced error handling.
    
    Parameters:
    - signal_data: Can be float/int value, string signal, or dict with signal details
    
    Returns:
    - tuple: (signal_value, confidence_score)
    """
    try:
        # If it's a numeric value, use directly
        if isinstance(signal_data, (int, float, Decimal)):
            value = float(signal_data)
            # Clamp value between -1 and 1
            value = max(-1.0, min(1.0, value))
            return value, 100.0
        
        # If it's a string (BUY/SELL/HOLD)
        elif isinstance(signal_data, str):
            signal_upper = signal_data.upper().strip()
            if signal_upper == "BUY":
                return 1.0, 100.0
            elif signal_upper == "SELL":
                return -1.0, 100.0
            else:  # HOLD or other
                return 0.0, 0.0
        
        # If it's a dictionary (from enhanced FastStrategies)
        elif isinstance(signal_data, dict):
            # Extract signal type with fallbacks
            signal_type = (signal_data.get('signal') or 
                          signal_data.get('signal_type') or 
                          signal_data.get('action', 'HOLD')).upper().strip()
            
            # Extract confidence/strength with fallbacks
            confidence = (signal_data.get('strength') or 
                         signal_data.get('confidence') or 
                         signal_data.get('certainty', 50.0))
            
            # Ensure confidence is valid
            confidence = max(0.0, min(100.0, float(confidence)))
            
            # Convert to numeric value
            if signal_type == "BUY":
                signal_value = 1.0
            elif signal_type == "SELL":
                signal_value = -1.0
            else:  # HOLD or other
                signal_value = 0.0
                
            return signal_value, confidence
        
        # Default case for unknown types
        else:
            logger.warning(f"Unknown signal format: {type(signal_data)} - {signal_data}")
            return 0.0, 0.0
            
    except (ValueError, TypeError, InvalidOperation) as e:
        logger.error(f"Error extracting signal strength from {signal_data}: {e}")
        return 0.0, 0.0

def calculate_allocation(signal: Union[Decimal, float], total_weight: Union[Decimal, float], 
                        remaining_capital: Decimal) -> Decimal:
    """
    Calculate the allocation for a stock based on its signal weight and remaining capital.
    
    Parameters:
    - signal: Signal strength for the stock
    - total_weight: Total weight of all signals
    - remaining_capital: Available capital to allocate
    
    Returns:
    - Decimal: Calculated allocation amount
    """
    try:
        if total_weight <= 0:
            return Decimal("0.00")
        
        signal_decimal = Decimal(str(signal)) if not isinstance(signal, Decimal) else signal
        total_weight_decimal = Decimal(str(total_weight)) if not isinstance(total_weight, Decimal) else total_weight
        
        allocation = (signal_decimal / total_weight_decimal) * remaining_capital
        return allocation.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
    except (ValueError, TypeError, InvalidOperation) as e:
        logger.error(f"Error calculating allocation: signal={signal}, total_weight={total_weight}, error={e}")
        return Decimal("0.00")

def allocate_capital(stock_signals: Dict[str, Any], total_capital: Union[float, int, Decimal], 
                    risk_adjusted: bool = True) -> Dict[str, Any]:
    """
    Allocate capital among stocks based on their signal strength with enhanced production features.
    
    Parameters:
    - stock_signals: Dict of {stock: signal_data} where signal_data can be numeric or dict
    - total_capital: Total capital to allocate
    - risk_adjusted: Whether to adjust allocation based on signal confidence
    
    Returns:
    - Dict of {stock: allocation} with enhanced metadata
    """
    start_time = time.perf_counter()
    
    try:
        # Input validation
        if not stock_signals:
            logger.warning("No stock signals provided for allocation")
            return {}
        
        if total_capital <= 0:
            logger.warning(f"Invalid total capital: {total_capital}")
            return {stock: 0.0 for stock in stock_signals}
        
        allocations = {}
        remaining_capital = Decimal(str(total_capital)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        MIN_ALLOCATION_THRESHOLD = Decimal("0.01")
        
        # Process and normalize signals with performance tracking
        normalized_signals = {}
        signal_confidences = {}
        rejected_signals = {}
        
        logger.debug(f"Processing {len(stock_signals)} stock signals...")
        
        for stock, signal_data in stock_signals.items():
            try:
                signal_value, confidence = extract_signal_strength(signal_data)
                
                # Only consider BUY signals with sufficient confidence
                if signal_value > 0 and confidence >= MIN_SIGNAL_STRENGTH:
                    # Scale signal by confidence if risk adjustment is enabled
                    if risk_adjusted:
                        normalized_signals[stock] = Decimal(str(signal_value * (confidence / 100.0)))
                    else:
                        normalized_signals[stock] = Decimal(str(signal_value))
                        
                    signal_confidences[stock] = confidence
                else:
                    rejected_signals[stock] = {
                        'signal_value': signal_value,
                        'confidence': confidence,
                        'reason': 'Insufficient confidence' if confidence < MIN_SIGNAL_STRENGTH else 'Non-BUY signal'
                    }
                    
            except Exception as e:
                logger.error(f"Error processing signal for {stock}: {e}")
                rejected_signals[stock] = {'error': str(e)}
        
        # Log processing results
        logger.info(f"Signal processing: {len(normalized_signals)} accepted, {len(rejected_signals)} rejected")
        if rejected_signals:
            logger.debug(f"Rejected signals: {rejected_signals}")
        
        # ✅ **Step 1: Remove zero-signal stocks early**
        stock_signals_filtered = {stock: signal for stock, signal in normalized_signals.items() if signal > 0}
        if not stock_signals_filtered:
            logger.warning("⚠️ All stock signals are zero or removed. No allocation will be made.")
            return {stock: 0.0 for stock in stock_signals}
        
        # ✅ **Step 2: Compute Total Weight & Exit Early if Needed**
        total_weight = sum(stock_signals_filtered.values())
        if total_weight == 0:
            logger.warning("Total signal weight is zero")
            return {stock: 0.0 for stock in stock_signals_filtered}
        
        logger.debug(f"Total signal weight: {total_weight}")
        
        # ✅ **Step 3: First Pass – Allocate Capital Based on Weights**
        initial_allocations = {}
        for stock, signal in stock_signals_filtered.items():
            try:
                calculated_allocation = calculate_allocation(signal, total_weight, remaining_capital)
                capped_allocation = min(calculated_allocation, Decimal(str(PER_STOCK_CAPITAL_LIMIT)))
                initial_allocations[stock] = capped_allocation
            except Exception as e:
                logger.error(f"Error calculating initial allocation for {stock}: {e}")
                initial_allocations[stock] = Decimal("0.00")
        
        # ✅ **Step 4: Update Remaining Capital**
        allocated_capital = sum(initial_allocations.values())
        remaining_capital = (remaining_capital - allocated_capital).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        logger.debug(f"Initial allocation complete. Allocated: {allocated_capital}, Remaining: {remaining_capital}")
        
        # ✅ **Step 5: Find Eligible Stocks for Redistribution**
        if risk_adjusted:
            # Prioritize highest confidence signals when redistributing
            eligible_stocks = sorted(
                (stock for stock in stock_signals_filtered if initial_allocations[stock] < Decimal(str(PER_STOCK_CAPITAL_LIMIT))),
                key=lambda s: signal_confidences.get(s, 0),
                reverse=True
            )
        else:
            # Original prioritization logic
            eligible_stocks = sorted(
                (stock for stock in stock_signals_filtered if initial_allocations[stock] < Decimal(str(PER_STOCK_CAPITAL_LIMIT))),
                key=lambda s: float(stock_signals_filtered[s]) / float(total_weight),
                reverse=True
            )
        
        logger.debug(f"Redistribution eligible stocks: {len(eligible_stocks)}")
        
        # ✅ **Step 6: Efficient Capital Redistribution**
        redistributed_amount = Decimal("0.00")
        for stock in eligible_stocks:
            if remaining_capital < MIN_ALLOCATION_THRESHOLD:
                break
            
            max_possible_allocation = Decimal(str(PER_STOCK_CAPITAL_LIMIT)) - initial_allocations[stock]
            additional_allocation = min(max_possible_allocation, remaining_capital)
            
            if additional_allocation >= MIN_ALLOCATION_THRESHOLD:
                initial_allocations[stock] += additional_allocation
                remaining_capital = (remaining_capital - additional_allocation).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
                redistributed_amount += additional_allocation

        # ✅ **Step 7: Assign Any Leftover Capital to the Most Undercapitalized Stock**
        if remaining_capital > 0 and eligible_stocks:
            stock_with_lowest_allocation = min(eligible_stocks, key=lambda s: initial_allocations[s])
            if stock_with_lowest_allocation:
                assignable_amount = min(remaining_capital, Decimal(str(PER_STOCK_CAPITAL_LIMIT)) - initial_allocations[stock_with_lowest_allocation])
                if assignable_amount >= MIN_ALLOCATION_THRESHOLD:
                    initial_allocations[stock_with_lowest_allocation] += assignable_amount
                    remaining_capital = (remaining_capital - assignable_amount).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        # Log the final allocations
        logger.info(f"Final capital allocations: {initial_allocations}")
        logger.info(f"Remaining unallocated capital: {remaining_capital}")
        
        # Performance tracking
        execution_time = time.perf_counter() - start_time
        logger.debug(f"Capital allocation completed in {execution_time:.4f} seconds")
        
        # ✅ **Step 8: Add signal confidence to output for reference**
        final_allocations = {
            stock: {
                "amount": float(allocation),
                "confidence": float(signal_confidences.get(stock, 0)),
                "pct_of_total": float((allocation / Decimal(str(total_capital))) * 100) if total_capital > 0 else 0
            }
            for stock, allocation in initial_allocations.items()
        }
        
        # For backward compatibility, also return simple dict if needed
        if not risk_adjusted:
            return {stock: float(allocation) for stock, allocation in initial_allocations.items()}
        
        return final_allocations
        
    except Exception as e:
        logger.error(f"Critical error in capital allocation: {e}")
        execution_time = time.perf_counter() - start_time
        logger.error(f"Allocation failed after {execution_time:.4f} seconds")
        # Return empty allocations on critical failure
        return {stock: 0.0 for stock in stock_signals}
