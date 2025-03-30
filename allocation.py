from config import PER_STOCK_CAPITAL_LIMIT
from decimal import Decimal, ROUND_DOWN
import logging  

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_allocation(signal, total_weight, remaining_capital):
    """Calculate the allocation for a stock based on its signal weight and remaining capital."""
    return (Decimal(signal) / Decimal(total_weight)) * remaining_capital if total_weight else Decimal(0)

def allocate_capital(stock_signals, total_capital):
    allocations = {}
    remaining_capital = Decimal(total_capital).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    MIN_ALLOCATION_THRESHOLD = Decimal("0.01")  

    # ✅ **Step 1: Remove zero-signal stocks early**
    stock_signals = {stock: Decimal(signal) for stock, signal in stock_signals.items() if signal > 0}
    if not stock_signals:
        logging.warning("⚠️ All stock signals are zero or removed. No allocation will be made.")
        return {stock: 0.0 for stock in stock_signals}  

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

    # ✅ **Step 8: Convert allocations to float for JSON/External Use**
    return {stock: float(allocation) for stock, allocation in initial_allocations.items()}
