class OrderExecution:
    def __init__(self, slippage_tolerance=0.01):
        """
        Initialize the order execution with a slippage tolerance.
        :param slippage_tolerance: Maximum allowable slippage as a fraction (e.g., 0.01 for 1%).
        """
        self.slippage_tolerance = slippage_tolerance

    def execute_order(self, order_type, quantity, ltp, market_depth):
        """
        Execute an order with slippage consideration.
        :param order_type: 'buy' or 'sell'
        :param quantity: Quantity to execute
        :param ltp: Last traded price
        :param market_depth: List of price levels with available quantities
        :return: Execution price or None if order cannot be executed
        """
        # ...existing code...

        # Calculate the maximum allowable price based on slippage tolerance
        if order_type == 'buy':
            max_price = ltp * (1 + self.slippage_tolerance)
        elif order_type == 'sell':
            max_price = ltp * (1 - self.slippage_tolerance)
        else:
            raise ValueError("Invalid order type")

        # Simulate order execution considering market depth
        execution_price = self._simulate_execution(order_type, quantity, market_depth, max_price)
        if execution_price is None:
            print("Order could not be executed within slippage tolerance.")
            return None

        print(f"Order executed at price: {execution_price}")
        return execution_price

    def _simulate_execution(self, order_type, quantity, market_depth, max_price):
        """
        Simulate order execution considering market depth and slippage tolerance.
        :param order_type: 'buy' or 'sell'
        :param quantity: Quantity to execute
        :param market_depth: List of price levels with available quantities
        :param max_price: Maximum allowable price for execution
        :return: Execution price or None if order cannot be executed
        """
        remaining_quantity = quantity
        total_cost = 0
        total_quantity = 0

        for price, available_quantity in market_depth:
            if (order_type == 'buy' and price > max_price) or (order_type == 'sell' and price < max_price):
                break

            trade_quantity = min(remaining_quantity, available_quantity)
            total_cost += trade_quantity * price
            total_quantity += trade_quantity
            remaining_quantity -= trade_quantity

            if remaining_quantity <= 0:
                break

        if remaining_quantity > 0:
            return None  # Not enough liquidity within slippage tolerance

        return total_cost / total_quantity  # Average execution price
