from kiteconnect import KiteConnect
from dotenv import load_dotenv
import os
import time
import json
from datetime import datetime, timedelta
import pandas as pd

# For enhanced logging
from utils.logger import Logger

# Load environment variables from .env file
load_dotenv()

class KiteManager:
    """
    Centralized manager for Zerodha Kite API interactions with robust error handling
    and retry mechanisms.
    """
    
    def __init__(self, api_key=None, api_secret=None, access_token=None, max_retries=3, retry_delay=5):
        """
        Initialize the KiteManager.
        
        Parameters:
        - api_key: Zerodha API key (defaults to environment variable)
        - api_secret: Zerodha API secret (defaults to environment variable)
        - access_token: Access token if already available
        - max_retries: Maximum number of retries for API calls
        - retry_delay: Delay between retries in seconds
        """
        # Use provided values or get from environment
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.api_secret = api_secret or os.getenv("KITE_API_SECRET")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize logger
        self.logger = Logger()
        
        # Initialize KiteConnect
        if not self.api_key:
            self.logger.error("API key not found. Please set KITE_API_KEY environment variable.")
            raise ValueError("API key not found")
        
        self.kite = KiteConnect(api_key=self.api_key)
        
        # Set access token if provided
        if access_token:
            self.set_access_token(access_token)
        else:
            # Try to load from token cache
            self._load_token_from_cache()
    
    def get_login_url(self):
        """Generate and return login URL"""
        return self.kite.login_url()
    
    def generate_session(self, request_token):
        """
        Generate session with request token and store access token.
        
        Parameters:
        - request_token: Request token obtained after login redirect
        
        Returns:
        - dict: Session data including access_token
        """
        try:
            session = self.kite.generate_session(request_token, self.api_secret)
            access_token = session["access_token"]
            
            # Set the access token in Kite
            self.kite.set_access_token(access_token)
            
            # Save token to cache
            self._save_token_to_cache(access_token)
            
            self.logger.info("Session generated successfully.")
            return session
        
        except Exception as e:
            self.logger.error(f"Error generating session: {e}")
            raise
    
    def set_access_token(self, access_token):
        """
        Set access token for authenticated requests.
        
        Parameters:
        - access_token: Access token string
        
        Returns:
        - bool: Success status
        """
        try:
            self.kite.set_access_token(access_token)
            self.logger.info("Access token set successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting access token: {e}")
            return False
    
    def _save_token_to_cache(self, access_token):
        """Save access token to cache file"""
        try:
            cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            token_data = {
                "access_token": access_token,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(os.path.join(cache_dir, 'token_cache.json'), 'w') as f:
                json.dump(token_data, f)
        
        except Exception as e:
            self.logger.warning(f"Could not save token to cache: {e}")
    
    def _load_token_from_cache(self):
        """Load access token from cache if available and not expired"""
        try:
            cache_file = os.path.join(os.path.dirname(__file__), 'cache', 'token_cache.json')
            
            if not os.path.exists(cache_file):
                return False
                
            with open(cache_file, 'r') as f:
                token_data = json.load(f)
            
            # Check if token is less than 1 day old (Zerodha tokens expire daily)
            token_time = datetime.fromisoformat(token_data["timestamp"])
            if datetime.now() - token_time > timedelta(days=1):
                self.logger.info("Cached token expired.")
                return False
            
            # Set the cached token
            return self.set_access_token(token_data["access_token"])
            
        except Exception as e:
            self.logger.warning(f"Could not load token from cache: {e}")
            return False
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """
        Execute API function with retry mechanism.
        
        Parameters:
        - func: Function to execute
        - args, kwargs: Arguments for the function
        
        Returns:
        - Result of the function call
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        self.logger.error(f"API call failed after {self.max_retries} attempts: {last_error}")
        raise last_error
    
    # ===== Data Fetching Methods =====
    
    def get_instruments(self, exchange=None):
        """
        Fetch instruments from Zerodha.
        
        Parameters:
        - exchange: Optional exchange (e.g., 'NSE', 'BSE')
        
        Returns:
        - list: Instruments data
        """
        return self._execute_with_retry(self.kite.instruments, exchange)
    
    def get_historical_data(self, instrument_token, from_date, to_date, interval, continuous=False):
        """
        Fetch historical data for an instrument.
        
        Parameters:
        - instrument_token: Instrument identifier
        - from_date: Start date
        - to_date: End date
        - interval: Candle interval
        - continuous: Whether to fetch continuous data
        
        Returns:
        - DataFrame: Historical price data
        """
        data = self._execute_with_retry(
            self.kite.historical_data,
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=continuous
        )
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def get_quote(self, symbols):
        """
        Fetch market quotes for symbols.
        
        Parameters:
        - symbols: List of symbols or single symbol
        
        Returns:
        - dict: Quote data
        """
        # Ensure symbols are prefixed with exchange
        if isinstance(symbols, str):
            symbols = [f"NSE:{symbols}"]
        else:
            symbols = [f"NSE:{s}" if ':' not in s else s for s in symbols]
            
        return self._execute_with_retry(self.kite.quote, symbols)
    
    def get_ltp(self, symbols):
        """
        Fetch last traded price for symbols.
        
        Parameters:
        - symbols: List of symbols or single symbol
        
        Returns:
        - dict: LTP data
        """
        # Ensure symbols are prefixed with exchange
        if isinstance(symbols, str):
            symbols = [f"NSE:{symbols}"]
        else:
            symbols = [f"NSE:{s}" if ':' not in s else s for s in symbols]
            
        return self._execute_with_retry(self.kite.ltp, symbols)
    
    # ===== Order Management Methods =====
    
    def place_order(self, symbol, exchange="NSE", transaction_type=None, 
                   quantity=None, price=None, product=None, order_type=None,
                   validity=None, disclosed_quantity=None, trigger_price=None,
                   squareoff=None, stoploss=None, trailing_stoploss=None, tag=None):
        """
        Place an order.
        
        Parameters:
        - Multiple order parameters (see Zerodha API docs)
        
        Returns:
        - str: Order ID
        """
        return self._execute_with_retry(
            self.kite.place_order,
            variety="regular",
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            product=product,
            order_type=order_type,
            validity=validity,
            disclosed_quantity=disclosed_quantity,
            trigger_price=trigger_price,
            squareoff=squareoff,
            stoploss=stoploss,
            trailing_stoploss=trailing_stoploss,
            tag=tag
        )
    
    def modify_order(self, order_id, **params):
        """
        Modify an existing order.
        
        Parameters:
        - order_id: ID of order to modify
        - params: Parameters to update
        
        Returns:
        - dict: Modified order data
        """
        return self._execute_with_retry(self.kite.modify_order, order_id=order_id, variety="regular", **params)
    
    def cancel_order(self, order_id, variety="regular"):
        """
        Cancel an order.
        
        Parameters:
        - order_id: ID of order to cancel
        - variety: Order variety
        
        Returns:
        - dict: Cancellation response
        """
        return self._execute_with_retry(self.kite.cancel_order, order_id=order_id, variety=variety)
    
    def get_orders(self):
        """
        Get all orders.
        
        Returns:
        - list: Order data
        """
        return self._execute_with_retry(self.kite.orders)
    
    def get_order_history(self, order_id):
        """
        Get order history for a specific order.
        
        Parameters:
        - order_id: ID of the order
        
        Returns:
        - list: Order history
        """
        return self._execute_with_retry(self.kite.order_history, order_id)
    
    # ===== Position and Holdings Methods =====
    
    def get_positions(self):
        """
        Get user positions.
        
        Returns:
        - dict: Position data
        """
        return self._execute_with_retry(self.kite.positions)
    
    def get_holdings(self):
        """
        Get user holdings.
        
        Returns:
        - list: Holdings data
        """
        return self._execute_with_retry(self.kite.holdings)
    
    def get_margins(self, segment=None):
        """
        Get user margins.
        
        Parameters:
        - segment: Optional segment (e.g., 'equity', 'commodity')
        
        Returns:
        - dict: Margin data
        """
        return self._execute_with_retry(self.kite.margins, segment)
    
    # ===== Utility Methods =====
    
    def instrument_token_lookup(self, symbol, exchange="NSE"):
        """
        Look up instrument token for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange name
        
        Returns:
        - int: Instrument token
        """
        instruments = self.get_instruments(exchange)
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
        return None
    
    def fetch_market_data(self, symbols, interval="5minute", days=1):
        """
        Fetch market data for multiple symbols.
        
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
        
        for symbol in symbols:
            try:
                token = self.instrument_token_lookup(symbol)
                if not token:
                    self.logger.warning(f"Instrument token not found for {symbol}")
                    continue
                    
                data = self.get_historical_data(token, from_date, to_date, interval)
                if not data.empty:
                    result[symbol] = data
                    self.logger.debug(f"Fetched market data for {symbol}")
                else:
                    self.logger.warning(f"No data returned for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching market data for {symbol}: {e}")
        
        return result

# Create a singleton instance
kite_manager = KiteManager()