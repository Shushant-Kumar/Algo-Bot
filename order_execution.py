import time
from datetime import datetime
import uuid
from enum import Enum
import traceback
from utils.logger import Logger, LogLevel
from utils.risk_manager import get_risk_exposure
from manager import kite_manager

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderExecution:
    def __init__(self, slippage_tolerance=0.01, max_retries=3, retry_delay=5, use_market_orders=True, simulation_mode=True):
        """
        Initialize the order execution with a slippage tolerance.
        :param slippage_tolerance: Maximum allowable slippage as a fraction (e.g., 0.01 for 1%).
        :param max_retries: Maximum number of retries on execution failure
        :param retry_delay: Delay in seconds between retries
        :param use_market_orders: Whether to use market orders by default
        :param simulation_mode: Whether to simulate orders or place real orders via Zerodha
        """
        self.slippage_tolerance = slippage_tolerance
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_market_orders = use_market_orders
        self.simulation_mode = simulation_mode
        self.logger = Logger()
        self.active_orders = {}
        self.order_history = {}

    def execute_order(self, order_type, quantity, ltp, market_depth, symbol=None, strategy_name=None, signal_data=None):
        """
        Execute an order with slippage consideration.
        :param order_type: 'buy' or 'sell'
        :param quantity: Quantity to execute
        :param ltp: Last traded price
        :param market_depth: List of price levels with available quantities
        :param symbol: Symbol/ticker for the order
        :param strategy_name: Name of the strategy generating the order
        :param signal_data: Additional signal data with confidence, etc.
        :return: Order details dict or None if order cannot be executed
        """
        # Generate order ID
        order_id = str(uuid.uuid4())[:10]
        
        # Log order attempt
        order_details = {
            'order_id': order_id,
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'ltp': ltp,
            'timestamp': datetime.now(),
            'strategy': strategy_name
        }
        
        self.logger.info(f"Attempting to execute {order_type} order for {quantity} {symbol} at ~{ltp}", 
                        strategy_name, order_details)

        # Calculate the maximum allowable price based on slippage tolerance
        if order_type.lower() == 'buy':
            max_price = ltp * (1 + self.slippage_tolerance)
        elif order_type.lower() == 'sell':
            max_price = ltp * (1 - self.slippage_tolerance)
        else:
            self.logger.error(f"Invalid order type: {order_type}", strategy_name)
            raise ValueError("Invalid order type")

        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                # Simulate order execution considering market depth
                execution_price = self._simulate_execution(order_type, quantity, market_depth, max_price)
                if execution_price is None:
                    self.logger.warning(f"Order could not be executed within slippage tolerance (attempt {attempt+1})", strategy_name)
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        self.logger.error(f"Order execution failed after {self.max_retries} attempts", strategy_name)
                        order_details['status'] = OrderStatus.REJECTED.value
                        order_details['reason'] = "Slippage tolerance exceeded"
                        self.order_history.append(order_details)
                        return None

                # Order executed successfully
                order_details['execution_price'] = execution_price
                order_details['status'] = OrderStatus.FILLED.value
                order_details['slippage'] = abs((execution_price - ltp) / ltp) * 100  # Slippage as percentage
                
                # Add to order history
                self.active_orders[order_id] = order_details
                self.order_history.append(order_details)
                
                # Log successful execution
                self.logger.info(f"Order executed: {order_type} {quantity} {symbol} at {execution_price} (slippage: {order_details['slippage']:.2f}%)", 
                                strategy_name, order_details)
                
                return order_details
                
            except Exception as e:
                self.logger.error(f"Error executing order (attempt {attempt+1}): {str(e)}", strategy_name)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Order execution failed after {self.max_retries} attempts: {traceback.format_exc()}", strategy_name)
                    order_details['status'] = OrderStatus.REJECTED.value
                    order_details['reason'] = str(e)
                    self.order_history.append(order_details)
                    return None

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
            if (order_type.lower() == 'buy' and price > max_price) or (order_type.lower() == 'sell' and price < max_price):
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
    
    def execute_order_from_signal(self, signal_data, symbol, quantity, ltp, market_depth, strategy_name=None):
        """
        Execute an order based on a strategy signal.
        
        Parameters:
        - signal_data: Signal data from strategy (string or dict)
        - symbol: Symbol/ticker
        - quantity: Quantity to execute
        - ltp: Last traded price
        - market_depth: Market depth data
        - strategy_name: Name of the strategy
        
        Returns:
        - Order details or None if no action taken
        """
        # Handle different signal formats
        if isinstance(signal_data, str):
            signal = signal_data
            confidence = 100
        elif isinstance(signal_data, dict):
            signal = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('strength', signal_data.get('confidence', 50))
        else:
            self.logger.warning(f"Unsupported signal format: {type(signal_data)}", strategy_name)
            return None
            
        # Check if signal is actionable
        if signal not in ["BUY", "SELL"] or confidence < 60:  # Minimum confidence threshold
            return None
            
        # Map signal to order type
        order_type = signal.lower()
        
        # Check if we should use market or limit orders
        if self.use_market_orders:
            return self.execute_order(order_type, quantity, ltp, market_depth, symbol, strategy_name, signal_data)
        else:
            # For limit orders, set price slightly better than LTP
            limit_price = ltp * 0.99 if order_type == 'buy' else ltp * 1.01
            return self.place_limit_order(order_type, quantity, limit_price, symbol, strategy_name, signal_data)
    
    def place_limit_order(self, order_type, quantity, limit_price, symbol, strategy_name=None, signal_data=None):
        """
        Place a limit order.
        
        Parameters:
        - order_type: 'buy' or 'sell'
        - quantity: Quantity to execute
        - limit_price: Limit price for the order
        - symbol: Symbol/ticker
        - strategy_name: Name of the strategy
        - signal_data: Additional signal data
        
        Returns:
        - Order details dict
        """
        # Generate order ID
        order_id = str(uuid.uuid4())[:10]
        
        # Create order details
        order_details = {
            'order_id': order_id,
            'symbol': symbol,
            'type': order_type,
            'order_type': OrderType.LIMIT.value,
            'quantity': quantity,
            'limit_price': limit_price,
            'timestamp': datetime.now(),
            'strategy': strategy_name,
            'status': OrderStatus.PENDING.value
        }
        
        # Log limit order placement
        self.logger.info(f"Placed limit {order_type} order for {quantity} {symbol} at {limit_price}", 
                        strategy_name, order_details)
        
        if not self.simulation_mode:
            # Place real order with Zerodha via kite_manager
            try:
                zerodha_order_id = kite_manager.place_order(
                    symbol=symbol,
                    exchange="NSE",
                    transaction_type=order_type.upper(),
                    quantity=quantity,
                    price=round(limit_price, 2),
                    product="MIS",  # Intraday product
                    order_type="LIMIT",
                    tag=f"strat_{strategy_name}" if strategy_name else None
                )
                order_details['zerodha_order_id'] = zerodha_order_id
                self.logger.info(f"Order placed with Zerodha, Order ID: {zerodha_order_id}")
                
                # Store order ID mapping for future reference
                order_details['broker_order_id'] = zerodha_order_id
                
            except Exception as e:
                self.logger.error(f"Error placing order with Zerodha: {e}")
                order_details['status'] = OrderStatus.REJECTED.value
                order_details['reason'] = str(e)
                self.order_history.append(order_details)
                return order_details
        else:
            # Simulate order execution for testing
            filled_quantity = int(quantity * 0.8)  # 80% fill for simulation
            if filled_quantity > 0:
                execution_price = limit_price
                order_details['execution_price'] = execution_price
                order_details['filled_quantity'] = filled_quantity
                order_details['status'] = OrderStatus.PARTIALLY_FILLED.value if filled_quantity < quantity else OrderStatus.FILLED.value
                
                self.logger.info(f"Simulating: Limit order partially filled: {filled_quantity}/{quantity} {symbol} at {execution_price}", 
                                strategy_name, order_details)
        
        # Store in active orders
        self.active_orders[order_id] = order_details
        self.order_history.append(order_details)
        return order_details
    
    def cancel_order(self, order_id):
        """
        Cancel an existing order.
        
        Parameters:
        - order_id: ID of the order to cancel
        
        Returns:
        - True if successful, False otherwise
        """
        if order_id not in self.active_orders:
            self.logger.warning(f"Order {order_id} not found")
            return False
            
        order = self.active_orders[order_id]
        
        # Only cancel if in a cancellable state
        if order['status'] not in [OrderStatus.PENDING.value, OrderStatus.PARTIALLY_FILLED.value]:
            self.logger.warning(f"Cannot cancel order {order_id} in status {order['status']}")
            return False
            
        if not self.simulation_mode and 'broker_order_id' in order:
            # Cancel real order with Zerodha
            try:
                kite_manager.cancel_order(
                    order_id=order['broker_order_id'], 
                    variety="regular"
                )
                self.logger.info(f"Order {order_id} cancelled with broker")
            except Exception as e:
                self.logger.error(f"Error cancelling order with Zerodha: {e}")
                return False
                
        # Update order status
        order['status'] = OrderStatus.CANCELLED.value
        order['cancel_time'] = datetime.now()
        
        self.logger.info(f"Cancelled order {order_id} for {order['symbol']}", order.get('strategy'))
        
        # Update in history
        for hist_order in self.order_history:
            if hist_order['order_id'] == order_id:
                hist_order['status'] = OrderStatus.CANCELLED.value
                hist_order['cancel_time'] = order['cancel_time']
        
        return True
    
    def sync_order_status(self, order_id=None):
        """
        Sync order status with broker for real orders.
        
        Parameters:
        - order_id: Specific order to sync, or all active orders if None
        
        Returns:
        - Updated order details
        """
        if self.simulation_mode:
            return  # No syncing needed in simulation mode
            
        orders_to_sync = []
        if order_id:
            if order_id in self.active_orders:
                orders_to_sync.append(order_id)
            else:
                return None
        else:
            orders_to_sync = list(self.active_orders.keys())
            
        for o_id in orders_to_sync:
            order = self.active_orders[o_id]
            if 'broker_order_id' not in order:
                continue
                
            try:
                # Get latest status from Zerodha
                broker_order = kite_manager.get_order_history(order['broker_order_id'])
                if not broker_order:
                    continue
                    
                latest_status = broker_order[-1]['status']
                
                # Map Zerodha status to our status enum
                status_map = {
                    'COMPLETE': OrderStatus.FILLED.value,
                    'CANCELLED': OrderStatus.CANCELLED.value,
                    'REJECTED': OrderStatus.REJECTED.value,
                    'OPEN': OrderStatus.PENDING.value,
                    'PENDING': OrderStatus.PENDING.value
                }
                
                order['status'] = status_map.get(latest_status, order['status'])
                
                if latest_status == 'COMPLETE':
                    order['execution_price'] = broker_order[-1]['average_price']
                    order['filled_quantity'] = broker_order[-1]['filled_quantity']
                    
                    # Calculate slippage if we have limit price
                    if 'limit_price' in order:
                        order['slippage'] = abs((order['execution_price'] - order['limit_price']) / order['limit_price']) * 100
                
                # Update in history
                for hist_order in self.order_history:
                    if hist_order['order_id'] == o_id:
                        hist_order.update(order)
                        
                self.logger.info(f"Synced order {o_id} status: {latest_status}")
                
            except Exception as e:
                self.logger.error(f"Error syncing order {o_id} status: {e}")
                
        return self.active_orders

    def get_order_status(self, order_id):
        """Get the status of a specific order."""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check in history
        for order in self.order_history:
            if order['order_id'] == order_id:
                return order
        
        return None
    
    def get_active_orders(self, symbol=None, strategy=None):
        """
        Get active orders, optionally filtered by symbol or strategy.
        """
        if symbol and strategy:
            return {oid: order for oid, order in self.active_orders.items() 
                   if order['symbol'] == symbol and order['strategy'] == strategy}
        elif symbol:
            return {oid: order for oid, order in self.active_orders.items() 
                   if order['symbol'] == symbol}
        elif strategy:
            return {oid: order for oid, order in self.active_orders.items() 
                   if order['strategy'] == strategy}
        else:
            return self.active_orders
    
    def get_order_history(self, symbol=None, strategy=None, status=None):
        """
        Get order history, optionally filtered by symbol, strategy or status.
        """
        filtered_history = self.order_history
        
        if symbol:
            filtered_history = [order for order in filtered_history if order['symbol'] == symbol]
        
        if strategy:
            filtered_history = [order for order in filtered_history if order.get('strategy') == strategy]
            
        if status:
            filtered_history = [order for order in filtered_history if order.get('status') == status]
            
        return filtered_history
    
    def get_execution_statistics(self):
        """
        Get statistics about order executions.
        """
        total_orders = len(self.order_history)
        if total_orders == 0:
            return {"total_orders": 0}
            
        filled_orders = len([o for o in self.order_history if o.get('status') == OrderStatus.FILLED.value])
        rejected_orders = len([o for o in self.order_history if o.get('status') == OrderStatus.REJECTED.value])
        cancelled_orders = len([o for o in self.order_history if o.get('status') == OrderStatus.CANCELLED.value])
        
        # Calculate average slippage
        slippages = [o.get('slippage', 0) for o in self.order_history if 'slippage' in o]
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        
        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "rejected_orders": rejected_orders,
            "cancelled_orders": cancelled_orders,
            "fill_rate": (filled_orders / total_orders) * 100 if total_orders > 0 else 0,
            "average_slippage": avg_slippage
        }
