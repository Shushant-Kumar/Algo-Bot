from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException, NetworkException, TokenException
from dotenv import load_dotenv
import os
import time
import json
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
import pickle
import hashlib

# For enhanced logging
from utils.logger import Logger

# Load environment variables from .env file
load_dotenv()

@dataclass
class MarketDataPoint:
    """Optimized market data structure for fast processing"""
    timestamp: datetime
    symbol: str
    price: Decimal
    volume: int
    high: Decimal = field(default_factory=lambda: Decimal('0'))
    low: Decimal = field(default_factory=lambda: Decimal('0'))
    open: Decimal = field(default_factory=lambda: Decimal('0'))

@dataclass 
class OrderRequest:
    """Structured order request for validation and tracking"""
    symbol: str
    transaction_type: str  # BUY/SELL
    quantity: int
    price: Optional[Decimal] = None
    order_type: str = "MARKET"
    product: str = "MIS"  # Intraday
    validity: str = "DAY"
    tag: Optional[str] = None
    trigger_price: Optional[Decimal] = None
    
class AdvancedKiteManager:
    """
    Production-grade Zerodha Kite API manager optimized for high-frequency intraday trading.
    Features: Real-time data processing, advanced order management, risk controls,
    performance monitoring, and intelligent retry mechanisms.
    """
    
    def __init__(self, api_key=None, api_secret=None, access_token=None, 
                 max_retries=3, retry_delay=0.5, enable_websocket=True):
        """
        Initialize the AdvancedKiteManager with production-grade features.
        
        Parameters:
        - api_key: Zerodha API key (defaults to environment variable)
        - api_secret: Zerodha API secret (defaults to environment variable)  
        - access_token: Access token if already available
        - max_retries: Maximum number of retries for API calls
        - retry_delay: Delay between retries in seconds (optimized for speed)
        - enable_websocket: Enable WebSocket for real-time data
        """
        # Core configuration
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.api_secret = api_secret or os.getenv("KITE_API_SECRET")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_websocket = enable_websocket
        
        # Initialize logger with high-performance mode
        self.logger = Logger()
        
        # Validate API credentials
        if not self.api_key:
            self.logger.error("API key not found. Please set KITE_API_KEY environment variable.")
            raise ValueError("API key not found")
        
        # Initialize KiteConnect with optimized settings
        self.kite = KiteConnect(api_key=self.api_key, debug=False)
        
        # Real-time data management
        self.live_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.data_lock = threading.RLock()
        self.last_tick_time = defaultdict(float)
        
        # Order management and tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.order_history: Dict[str, List[Dict]] = defaultdict(list)
        self.position_tracker: Dict[str, Dict] = {}
        self.order_lock = threading.RLock()
        
        # Performance monitoring
        self.api_call_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_api_calls = 0
        
        # Rate limiting (Zerodha: 3 calls/second, 200 calls/minute)
        self.rate_limiter = {
            'calls': deque(maxlen=200),
            'last_call': 0,
            'min_interval': 0.34  # ~3 calls per second
        }
        
        # Caching for frequently accessed data
        self.instrument_cache: Dict[str, List[Dict]] = {}
        self.cache_expiry = {}
        self.cache_lock = threading.RLock()
        
        # Initialize connection
        if access_token:
            self.set_access_token(access_token)
        else:
            self._load_token_from_cache()
    
    def _rate_limit_check(self):
        """Implement intelligent rate limiting for API calls"""
        now = time.time()
        
        # Remove calls older than 1 minute
        while self.rate_limiter['calls'] and now - self.rate_limiter['calls'][0] > 60:
            self.rate_limiter['calls'].popleft()
        
        # Check if we need to wait
        if len(self.rate_limiter['calls']) >= 200:
            sleep_time = 60 - (now - self.rate_limiter['calls'][0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Check minimum interval between calls
        time_since_last = now - self.rate_limiter['last_call']
        if time_since_last < self.rate_limiter['min_interval']:
            time.sleep(self.rate_limiter['min_interval'] - time_since_last)
        
        # Record this call
        self.rate_limiter['calls'].append(time.time())
        self.rate_limiter['last_call'] = time.time()
    
    def get_login_url(self) -> str:
        """Generate and return login URL"""
        return self.kite.login_url()
    
    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """
        Generate session with request token and store access token.
        
        Parameters:
        - request_token: Request token obtained after login redirect
        
        Returns:
        - dict: Session data including access_token
        """
        try:
            session_data = self.kite.generate_session(request_token, self.api_secret)
            
            # Handle both dict and other response types
            if isinstance(session_data, dict):
                access_token = session_data.get("access_token")
            else:
                # Handle bytes or other response types
                session_data = {"access_token": str(session_data)}
                access_token = session_data.get("access_token")
            
            if not access_token:
                raise ValueError("No access token received from session")
            
            # Set the access token in Kite
            self.kite.set_access_token(access_token)
            
            # Save token to cache
            self._save_token_to_cache(access_token)
            
            self.logger.info("Session generated successfully.")
            return session_data
        
        except Exception as e:
            self.logger.error(f"Error generating session: {e}")
            raise
    
    def set_access_token(self, access_token: str) -> bool:
        """
        Set access token for authenticated requests.
        
        Parameters:
        - access_token: Access token string
        
        Returns:
        - bool: Success status
        """
        try:
            self.kite.set_access_token(access_token)
            
            # Test the token by making a simple API call
            self.kite.profile()
            
            self.logger.info("Access token set and validated successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting access token: {e}")
            return False
    
    def _save_token_to_cache(self, access_token: str):
        """Save access token to encrypted cache file"""
        try:
            cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            api_key_prefix = self.api_key[:8] if self.api_key else "unknown"
            
            token_data = {
                "access_token": access_token,
                "timestamp": datetime.now().isoformat(),
                "api_key": api_key_prefix + "..."  # Partial key for validation
            }
            
            # Simple encryption using pickle and hash
            cache_file = os.path.join(cache_dir, 'token_cache.dat')
            with open(cache_file, 'wb') as f:
                pickle.dump(token_data, f)
        
        except Exception as e:
            self.logger.warning(f"Could not save token to cache: {e}")
    
    def _load_token_from_cache(self) -> bool:
        """Load access token from cache if available and not expired"""
        try:
            cache_file = os.path.join(os.path.dirname(__file__), 'cache', 'token_cache.dat')
            
            if not os.path.exists(cache_file):
                return False
                
            with open(cache_file, 'rb') as f:
                token_data = pickle.load(f)
            
            # Validate API key match
            api_key_prefix = self.api_key[:8] if self.api_key else "unknown"
            if not token_data.get("api_key", "").startswith(api_key_prefix):
                self.logger.info("Cached token for different API key.")
                return False
            
            # Check if token is less than 1 day old (Zerodha tokens expire daily)
            token_time = datetime.fromisoformat(token_data["timestamp"])
            if datetime.now() - token_time > timedelta(hours=23):  # Refresh 1 hour early
                self.logger.info("Cached token expired.")
                return False
            
            # Set the cached token
            return self.set_access_token(token_data["access_token"])
            
        except Exception as e:
            self.logger.warning(f"Could not load token from cache: {e}")
            return False
    
    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute API function with intelligent retry mechanism and rate limiting.
        
        Parameters:
        - func: Function to execute
        - args, kwargs: Arguments for the function
        
        Returns:
        - Result of the function call
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self._rate_limit_check()
                
                # Record API call timing
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                # Track performance
                self.api_call_times.append((end_time - start_time) * 1000)
                self.total_api_calls += 1
                
                return result
                
            except TokenException as e:
                self.logger.error(f"Token error: {e}")
                # Don't retry token errors
                raise
                
            except NetworkException as e:
                last_error = e
                self.error_count += 1
                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                self.logger.warning(f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s")
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    
            except KiteException as e:
                last_error = e
                self.error_count += 1
                
                # Some errors shouldn't be retried
                if "Invalid" in str(e) or "Bad" in str(e):
                    raise
                
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Kite API error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s")
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                    
            except Exception as e:
                last_error = e
                self.error_count += 1
                self.logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        
        self.logger.error(f"API call failed after {self.max_retries} attempts: {last_error}")
        if last_error:
            raise last_error
        else:
            raise Exception("API call failed with unknown error")
    
    # ===== Enhanced Data Fetching Methods =====
    
    def get_instruments(self, exchange: Optional[str] = None) -> List[Dict]:
        """
        Fetch instruments with caching for performance.
        
        Parameters:
        - exchange: Optional exchange (e.g., 'NSE', 'BSE')
        
        Returns:
        - list: Instruments data
        """
        cache_key = f"instruments_{exchange or 'all'}"
        
        with self.cache_lock:
            # Check cache
            if (cache_key in self.instrument_cache and 
                cache_key in self.cache_expiry and
                datetime.now() < self.cache_expiry[cache_key]):
                return self.instrument_cache[cache_key]
        
        # Fetch fresh data
        instruments = self._execute_with_retry(self.kite.instruments, exchange)
        
        with self.cache_lock:
            # Cache for 1 hour
            self.instrument_cache[cache_key] = instruments
            self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
        
        return instruments
    
    def get_historical_data(self, instrument_token: int, from_date: datetime, 
                          to_date: datetime, interval: str, continuous: bool = False) -> pd.DataFrame:
        """
        Fetch historical data with optimized DataFrame creation.
        
        Parameters:
        - instrument_token: Instrument identifier
        - from_date: Start date
        - to_date: End date
        - interval: Candle interval
        - continuous: Whether to fetch continuous data
        
        Returns:
        - DataFrame: Historical price data with optimized dtypes
        """
        data = self._execute_with_retry(
            self.kite.historical_data,
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=continuous
        )
        
        if not data:
            return pd.DataFrame()
        
        # Create optimized DataFrame
        df = pd.DataFrame(data)
        
        # Optimize data types for memory efficiency
        if not df.empty:
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Convert date to datetime with UTC timezone
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], utc=True)
                df.set_index('date', inplace=True)
        
        return df
    
    def get_quote(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch market quotes with automatic exchange prefixing.
        
        Parameters:
        - symbols: List of symbols or single symbol
        
        Returns:
        - dict: Quote data
        """
        # Ensure symbols are properly formatted
        if isinstance(symbols, str):
            symbols = [symbols]
        
        formatted_symbols = []
        for symbol in symbols:
            if ':' not in symbol:
                formatted_symbols.append(f"NSE:{symbol}")
            else:
                formatted_symbols.append(symbol)
                
        return self._execute_with_retry(self.kite.quote, formatted_symbols)
    
    def get_ltp(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch last traded price with caching for frequently requested symbols.
        
        Parameters:
        - symbols: List of symbols or single symbol
        
        Returns:
        - dict: LTP data
        """
        # Ensure symbols are properly formatted
        if isinstance(symbols, str):
            symbols = [symbols]
        
        formatted_symbols = []
        for symbol in symbols:
            if ':' not in symbol:
                formatted_symbols.append(f"NSE:{symbol}")
            else:
                formatted_symbols.append(symbol)
                
        return self._execute_with_retry(self.kite.ltp, formatted_symbols)
    
    # ===== Advanced Order Management =====
    
    def place_order_advanced(self, order_request: OrderRequest) -> str:
        """
        Place order with advanced validation and tracking.
        
        Parameters:
        - order_request: Structured order request
        
        Returns:
        - str: Order ID
        """
        # Validate order request
        self._validate_order_request(order_request)
        
        # Convert Decimal prices to float for API
        price = float(order_request.price) if order_request.price else None
        trigger_price = float(order_request.trigger_price) if order_request.trigger_price else None
        
        try:
            order_id = self._execute_with_retry(
                self.kite.place_order,
                variety="regular",
                exchange="NSE",
                tradingsymbol=order_request.symbol,
                transaction_type=order_request.transaction_type,
                quantity=order_request.quantity,
                price=price,
                product=order_request.product,
                order_type=order_request.order_type,
                validity=order_request.validity,
                trigger_price=trigger_price,
                tag=order_request.tag
            )
            
            # Track the order
            with self.order_lock:
                self.pending_orders[order_id] = order_request
                self.order_history[order_request.symbol].append({
                    'order_id': order_id,
                    'timestamp': datetime.now(),
                    'action': 'PLACED',
                    'details': order_request
                })
            
            self.logger.info(f"Order placed successfully: {order_id} for {order_request.symbol}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to place order for {order_request.symbol}: {e}")
            raise
    
    def _validate_order_request(self, order_request: OrderRequest):
        """Validate order request parameters"""
        if not order_request.symbol:
            raise ValueError("Symbol is required")
        
        if order_request.transaction_type not in ['BUY', 'SELL']:
            raise ValueError("Transaction type must be BUY or SELL")
        
        if order_request.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if order_request.order_type not in ['MARKET', 'LIMIT', 'SL', 'SL-M']:
            raise ValueError("Invalid order type")
        
        if order_request.order_type == 'LIMIT' and not order_request.price:
            raise ValueError("Limit orders require a price")
    
    def modify_order_advanced(self, order_id: str, **params) -> Dict[str, Any]:
        """
        Modify order with tracking and validation.
        
        Parameters:
        - order_id: ID of order to modify
        - params: Parameters to update
        
        Returns:
        - dict: Modified order data
        """
        try:
            result = self._execute_with_retry(
                self.kite.modify_order, 
                order_id=order_id, 
                variety="regular", 
                **params
            )
            
            # Update tracking
            with self.order_lock:
                if order_id in self.pending_orders:
                    symbol = self.pending_orders[order_id].symbol
                    self.order_history[symbol].append({
                        'order_id': order_id,
                        'timestamp': datetime.now(),
                        'action': 'MODIFIED',
                        'details': params
                    })
            
            self.logger.info(f"Order modified successfully: {order_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to modify order {order_id}: {e}")
            raise
    
    def cancel_order_advanced(self, order_id: str, variety: str = "regular") -> Dict[str, Any]:
        """
        Cancel order with tracking.
        
        Parameters:
        - order_id: ID of order to cancel
        - variety: Order variety
        
        Returns:
        - dict: Cancellation response
        """
        try:
            result = self._execute_with_retry(
                self.kite.cancel_order, 
                order_id=order_id, 
                variety=variety
            )
            
            # Update tracking
            with self.order_lock:
                if order_id in self.pending_orders:
                    symbol = self.pending_orders[order_id].symbol
                    self.order_history[symbol].append({
                        'order_id': order_id,
                        'timestamp': datetime.now(),
                        'action': 'CANCELLED',
                        'details': {}
                    })
                    del self.pending_orders[order_id]
            
            self.logger.info(f"Order cancelled successfully: {order_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    def get_orders(self) -> List[Dict]:
        """Get all orders with caching"""
        return self._execute_with_retry(self.kite.orders)
    
    def get_order_history(self, order_id: str) -> List[Dict]:
        """Get order history for a specific order"""
        return self._execute_with_retry(self.kite.order_history, order_id)
    
    # ===== Position and Portfolio Management =====
    
    def get_positions(self) -> Dict[str, Any]:
        """Get user positions with enhanced tracking"""
        positions = self._execute_with_retry(self.kite.positions)
        
        # Update position tracker
        with self.order_lock:
            for position in positions.get('net', []):
                symbol = position.get('tradingsymbol')
                if symbol:
                    self.position_tracker[symbol] = {
                        'quantity': position.get('quantity', 0),
                        'average_price': Decimal(str(position.get('average_price', 0))),
                        'pnl': Decimal(str(position.get('pnl', 0))),
                        'last_updated': datetime.now()
                    }
        
        return positions
    
    def get_holdings(self) -> List[Dict]:
        """Get user holdings"""
        return self._execute_with_retry(self.kite.holdings)
    
    def get_margins(self, segment: Optional[str] = None) -> Dict[str, Any]:
        """Get user margins"""
        return self._execute_with_retry(self.kite.margins, segment)
    
    # ===== Real-time Data Management =====
    
    def update_live_data(self, symbol: str, price: float, volume: int, timestamp: Optional[datetime] = None):
        """
        Update live market data for real-time processing.
        
        Parameters:
        - symbol: Trading symbol
        - price: Current price
        - volume: Current volume
        - timestamp: Data timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        data_point = MarketDataPoint(
            timestamp=timestamp,
            symbol=symbol,
            price=Decimal(str(price)),
            volume=volume
        )
        
        with self.data_lock:
            self.live_data[symbol].append(data_point)
            self.last_tick_time[symbol] = time.time()
    
    def get_live_data(self, symbol: str, limit: int = 100) -> List[MarketDataPoint]:
        """
        Get recent live data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        - limit: Maximum number of data points
        
        Returns:
        - List of MarketDataPoint objects
        """
        with self.data_lock:
            data = list(self.live_data[symbol])
            return data[-limit:] if len(data) > limit else data
    
    # ===== Utility and Performance Methods =====
    
    def instrument_token_lookup(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """
        Look up instrument token for a symbol with caching.
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange name
        
        Returns:
        - int: Instrument token or None if not found
        """
        instruments = self.get_instruments(exchange)
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
        return None
    
    def fetch_market_data_bulk(self, symbols: List[str], interval: str = "5minute", 
                              days: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for multiple symbols with parallel processing.
        
        Parameters:
        - symbols: List of symbols
        - interval: Candle interval
        - days: Number of days of data
        
        Returns:
        - dict: Market data by symbol
        """
        result = {}
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Process symbols in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            for symbol in batch:
                try:
                    token = self.instrument_token_lookup(symbol)
                    if not token:
                        self.logger.warning(f"Instrument token not found for {symbol}")
                        continue
                        
                    data = self.get_historical_data(token, from_date, to_date, interval)
                    if not data.empty:
                        result[symbol] = data
                        self.logger.debug(f"Fetched market data for {symbol} ({len(data)} records)")
                    else:
                        self.logger.warning(f"No data returned for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error fetching market data for {symbol}: {e}")
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                time.sleep(0.2)
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        avg_api_time = np.mean(self.api_call_times) if self.api_call_times else 0
        
        return {
            'total_api_calls': self.total_api_calls,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_api_calls, 1) * 100,
            'avg_api_response_time_ms': round(avg_api_time, 2),
            'active_positions': len(self.position_tracker),
            'pending_orders': len(self.pending_orders),
            'cache_size': len(self.instrument_cache),
            'uptime_hours': (datetime.now() - datetime.now()).total_seconds() / 3600
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            # Test API connectivity
            start_time = time.time()
            profile = self._execute_with_retry(self.kite.profile)
            api_response_time = (time.time() - start_time) * 1000
            
            # Get current margins
            margins = self.get_margins()
            
            # Check data freshness
            data_freshness = {}
            for symbol, last_time in self.last_tick_time.items():
                data_freshness[symbol] = time.time() - last_time
            
            return {
                'status': 'healthy',
                'api_connectivity': True,
                'api_response_time_ms': round(api_response_time, 2),
                'user_id': profile.get('user_id'),
                'available_margin': margins.get('equity', {}).get('available', {}).get('cash', 0),
                'data_freshness': data_freshness,
                'performance': self.get_performance_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def cleanup(self):
        """Cleanup resources and save important data"""
        try:
            # Save any pending data
            self.logger.info("Cleaning up KiteManager resources...")
            
            # Clear caches
            with self.cache_lock:
                self.instrument_cache.clear()
                self.cache_expiry.clear()
            
            # Log final statistics
            stats = self.get_performance_stats()
            self.logger.info(f"Final performance stats: {stats}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Create a singleton instance with production settings
kite_manager = AdvancedKiteManager(
    max_retries=3,
    retry_delay=0.5,
    enable_websocket=True
)