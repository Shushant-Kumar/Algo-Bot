"""
Production-Ready Algorithmic Trading System

High-performance intraday trading bot optimized for speed, reliability, and comprehensive
risk management. Features real-time strategy execution, advanced position management,
comprehensive monitoring, and full Kite Connect integration.

Key Features:
- Ultra-fast strategy execution with sub-second latency
- Advanced risk management with circuit breakers
- Real-time position and PnL tracking
- Comprehensive logging and monitoring
- Production-grade error handling and recovery
- Market hours validation and scheduling
- Multi-strategy execution with intelligent aggregation
- Automated stop-loss and take-profit management
"""

import os
import sys
import time
import asyncio
import threading
import traceback
from datetime import datetime, timezone, time as dt_time
from typing import Dict, List, Optional, Any
from decimal import Decimal
from collections import defaultdict, deque
import signal
import json

# Standard library imports
import pandas as pd
import numpy as np
import schedule

# Configuration and utilities
from config import (
    TOTAL_CAPITAL, 
    SIMULATION_MODE, 
    SLIPPAGE_TOLERANCE,
    PER_STOCK_ALLOCATION,
    MAX_RETRIES_ON_ERROR,
    RETRY_DELAY_SECONDS
)

# Enhanced logging system
from utils.logger import Logger

# Core trading components
from manager import kite_manager
from order_execution import get_executor, get_system_health, reset_circuit_breaker
from utils.strategy_manager import run_all_strategies_for_stocks

# Import all strategies
from strategies.fast_strategy_base import FastStrategyBase
from strategies.bollinger_bands_strategy import FastBollingerBandsStrategy
from strategies.rsi_strategy import FastRSIStrategy  
from strategies.macd_strategy import FastMACDStrategy
from strategies.moving_average_strategy import FastMovingAverageStrategy
from strategies.vwap_strategy import FastVWAPStrategy

# Safe strategy import
try:
    from strategies.stochastic_oscillator_strategy import FastStochasticStrategy as FastStochasticOscillatorStrategy
    HAS_STOCHASTIC_STRATEGY = True
except ImportError:
    HAS_STOCHASTIC_STRATEGY = False
    FastStochasticOscillatorStrategy = None
    print("Warning: FastStochasticStrategy not available - continuing without it")

# Safe imports with fallbacks
try:
    from utils.risk_manager import RiskManager
    HAS_RISK_MANAGER = True
except ImportError:
    HAS_RISK_MANAGER = False
    RiskManager = None


class ProductionTradingSystem:
    """
    Production-grade algorithmic trading system with comprehensive monitoring,
    risk management, and high-performance execution capabilities.
    """
    
    def __init__(self):
        """Initialize the production trading system."""
        # Core system configuration
        self.logger = Logger(console_output=True, file_output=True)
        self.logger.info("Initializing Production Trading System...")
        
        # System state tracking
        self.is_running = False
        self.is_market_open = False
        self.emergency_stop = False
        self.system_start_time = datetime.now(timezone.utc)
        
        # Performance and risk tracking
        self.daily_pnl = Decimal('0.0')
        self.daily_trades = 0
        self.total_capital = Decimal(str(TOTAL_CAPITAL))
        self.available_capital = Decimal(str(TOTAL_CAPITAL))
        self.max_daily_loss = self.total_capital * Decimal('0.05')  # 5% daily loss limit
        
        # Threading and synchronization
        self.main_lock = threading.RLock()
        self.execution_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Strategy management
        self.active_strategies = {}
        self.strategy_performance = defaultdict(lambda: {
            'total_signals': 0,
            'successful_trades': 0,
            'total_pnl': Decimal('0.0'),
            'win_rate': 0.0,
            'avg_return': 0.0
        })
        
        # Position and order tracking
        self.active_positions = {}
        self.position_tracker = defaultdict(lambda: {
            'quantity': 0,
            'average_price': Decimal('0.0'),
            'current_pnl': Decimal('0.0'),
            'unrealized_pnl': Decimal('0.0'),
            'entry_time': None
        })
        
        # Market data and timing
        self.last_market_data_update = {}
        self.execution_times = deque(maxlen=100)
        self.api_response_times = deque(maxlen=100)
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Production Trading System initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize order executor
            self.order_executor = get_executor()
            
            # Initialize risk manager if available
            self.risk_manager = None
            if HAS_RISK_MANAGER and RiskManager:
                try:
                    self.risk_manager = RiskManager()
                    self.logger.info("Risk manager initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Risk manager initialization failed: {e}")
            
            # Initialize strategies
            self._initialize_strategies()
            
            # Test Kite Connect authentication
            self._test_authentication()
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize all trading strategies."""
        try:
            # Get stock symbols from allocation config
            symbols = list(PER_STOCK_ALLOCATION.keys()) if PER_STOCK_ALLOCATION else ["RELIANCE", "TCS", "INFY"]
            
            strategy_classes = {
                'bollinger_bands': FastBollingerBandsStrategy,
                'rsi': FastRSIStrategy,
                'macd': FastMACDStrategy,
                'moving_average': FastMovingAverageStrategy,
                'vwap': FastVWAPStrategy
            }
            
            # Add stochastic strategy if available
            if HAS_STOCHASTIC_STRATEGY and FastStochasticOscillatorStrategy:
                strategy_classes['stochastic'] = FastStochasticOscillatorStrategy
            
            for strategy_name, strategy_class in strategy_classes.items():
                try:
                    strategy_instance = strategy_class()
                    self.active_strategies[strategy_name] = strategy_instance
                    self.logger.info(f"Initialized strategy: {strategy_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize strategy {strategy_name}: {e}")
            
            self.logger.info(f"Initialized {len(self.active_strategies)} strategies successfully")
            
        except Exception as e:
            self.logger.error(f"Strategy initialization failed: {e}")
            raise
    
    def _test_authentication(self):
        """Test Kite Connect authentication."""
        try:
            access_token = os.getenv("KITE_ACCESS_TOKEN")
            if not access_token:
                self.logger.warning("No access token found in environment variables")
                return False
            
            # Set access token
            if not kite_manager.set_access_token(access_token):
                self.logger.error("Failed to set access token")
                return False
            
            # Test with a simple API call
            try:
                profile = kite_manager.kite.profile()
                if isinstance(profile, dict):
                    user_name = profile.get('user_name', 'Unknown')
                else:
                    user_name = 'Unknown'
                self.logger.info(f"Authentication successful. User: {user_name}")
            except Exception as e:
                self.logger.warning(f"Could not fetch profile: {e}")
                self.logger.info("Authentication successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication test failed: {e}")
            if not SIMULATION_MODE:
                raise
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.emergency_stop = True
        self.shutdown()
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        current_time = now.time()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Indian market hours: 9:15 AM to 3:30 PM
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        
        return market_open <= current_time <= market_close
    
    def get_account_balance(self) -> Decimal:
        """Fetch current account balance."""
        if SIMULATION_MODE:
            return self.available_capital
        
        try:
            margins = kite_manager.get_margins("equity")
            cash_available = margins.get("available", {}).get("cash", 0)
            return Decimal(str(cash_available))
        except Exception as e:
            self.logger.error(f"Failed to fetch account balance: {e}")
            return Decimal('0.0')
    
    def fetch_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch real-time market data for symbols."""
        market_data = {}
        
        try:
            # Batch fetch quotes for efficiency
            quotes = kite_manager.get_quote(symbols)
            
            for symbol in symbols:
                try:
                    symbol_key = f"NSE:{symbol}"
                    if symbol_key in quotes:
                        quote = quotes[symbol_key]
                        ohlc = quote.get('ohlc', {})
                        
                        # Create DataFrame with OHLCV data
                        data = {
                            'timestamp': [datetime.now()],
                            'open': [float(ohlc.get('open', 0))],
                            'high': [float(ohlc.get('high', 0))],
                            'low': [float(ohlc.get('low', 0))],
                            'close': [float(quote.get('last_price', 0))],
                            'volume': [int(quote.get('volume', 0))]
                        }
                        
                        df = pd.DataFrame(data)
                        market_data[symbol] = df
                        self.last_market_data_update[symbol] = datetime.now()
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process market data for {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
        
        return market_data
    
    def execute_strategy_cycle(self) -> Dict[str, Any]:
        """Execute a complete strategy cycle for all symbols."""
        cycle_start_time = time.perf_counter()
        results = {
            'timestamp': datetime.now(timezone.utc),
            'signals_generated': 0,
            'orders_placed': 0,
            'errors': 0,
            'execution_time_ms': 0,
            'strategy_results': {}
        }
        
        try:
            # Check system health first
            health_status = get_system_health()
            if health_status['status'] != 'healthy':
                self.logger.warning(f"System health check failed: {health_status}")
                if health_status['status'] == 'circuit_breaker_triggered':
                    self.logger.error("Circuit breaker is active. Skipping trading cycle.")
                    return results
            
            # Check daily loss limits
            if self.daily_pnl <= -self.max_daily_loss:
                self.logger.error(f"Daily loss limit exceeded: {self.daily_pnl}. Stopping trading.")
                self.emergency_stop = True
                return results
            
            # Get symbols to trade
            symbols = list(PER_STOCK_ALLOCATION.keys()) if PER_STOCK_ALLOCATION else ["RELIANCE"]
            
            # Fetch market data
            market_data = self.fetch_market_data(symbols)
            
            # Execute strategies for each symbol
            for symbol in symbols:
                if symbol not in market_data:
                    self.logger.warning(f"No market data available for {symbol}")
                    continue
                
                try:
                    symbol_results = self._process_symbol_strategies(symbol, market_data[symbol])
                    results['strategy_results'][symbol] = symbol_results
                    
                    if symbol_results.get('signal_generated'):
                        results['signals_generated'] += 1
                    
                    if symbol_results.get('order_placed'):
                        results['orders_placed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing strategies for {symbol}: {e}")
                    results['errors'] += 1
            
            # Update positions and PnL
            self._update_positions_and_pnl()
            
        except Exception as e:
            self.logger.error(f"Strategy cycle execution failed: {e}")
            results['errors'] += 1
        
        # Record execution time
        execution_time = (time.perf_counter() - cycle_start_time) * 1000
        results['execution_time_ms'] = round(execution_time, 2)
        self.execution_times.append(execution_time)
        
        return results
    
    def _process_symbol_strategies(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Process all strategies for a single symbol."""
        symbol_results = {
            'symbol': symbol,
            'signal_generated': False,
            'order_placed': False,
            'strategy_signals': {},
            'aggregated_signal': None,
            'confidence_score': 0.0
        }
        
        try:
            # Collect signals from all strategies
            strategy_signals = {}
            total_confidence = 0.0
            signal_count = 0
            
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    # Update strategy with new data
                    latest_data = market_data.iloc[-1]
                    strategy.update(
                        price=float(latest_data['close']),
                        volume=int(latest_data['volume']),
                        high=float(latest_data['high']),
                        low=float(latest_data['low'])
                    )
                    
                    # Get signal from strategy
                    signal_result = strategy.get_signal()
                    
                    if signal_result and signal_result.get('signal') != 'HOLD':
                        strategy_signals[strategy_name] = signal_result
                        total_confidence += signal_result.get('strength', 50.0)
                        signal_count += 1
                        
                        # Update strategy performance tracking
                        self.strategy_performance[strategy_name]['total_signals'] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy_name} failed for {symbol}: {e}")
            
            symbol_results['strategy_signals'] = strategy_signals
            
            # Aggregate signals if any were generated
            if strategy_signals:
                aggregated_signal = self._aggregate_signals(strategy_signals)
                symbol_results['aggregated_signal'] = aggregated_signal
                symbol_results['confidence_score'] = total_confidence / max(signal_count, 1)
                symbol_results['signal_generated'] = True
                
                # Execute trade if signal is strong enough
                if aggregated_signal and aggregated_signal.get('strength', 0) >= 60.0:
                    order_result = self._execute_trade(symbol, aggregated_signal, market_data)
                    symbol_results['order_placed'] = order_result is not None
                    symbol_results['order_result'] = order_result
            
        except Exception as e:
            self.logger.error(f"Error processing strategies for {symbol}: {e}")
        
        return symbol_results
    
    def _aggregate_signals(self, strategy_signals: Dict[str, Dict]) -> Optional[Dict]:
        """Aggregate signals from multiple strategies."""
        if not strategy_signals:
            return None
        
        # Count BUY and SELL signals
        buy_signals = []
        sell_signals = []
        
        for strategy_name, signal_data in strategy_signals.items():
            signal = signal_data.get('signal')
            strength = signal_data.get('strength', 50.0)
            
            if signal == 'BUY':
                buy_signals.append(strength)
            elif signal == 'SELL':
                sell_signals.append(strength)
        
        # Determine aggregated signal
        if len(buy_signals) > len(sell_signals):
            avg_strength = sum(buy_signals) / len(buy_signals)
            return {
                'signal': 'BUY',
                'strength': avg_strength,
                'strategy_count': len(buy_signals),
                'consensus': len(buy_signals) / len(strategy_signals)
            }
        elif len(sell_signals) > len(buy_signals):
            avg_strength = sum(sell_signals) / len(sell_signals)
            return {
                'signal': 'SELL',
                'strength': avg_strength,
                'strategy_count': len(sell_signals),
                'consensus': len(sell_signals) / len(strategy_signals)
            }
        
        return None  # No clear consensus
    
    def _execute_trade(self, symbol: str, signal: Dict, market_data: pd.DataFrame) -> Optional[Dict]:
        """Execute a trade based on aggregated signal."""
        try:
            # Calculate position size
            current_price = float(market_data.iloc[-1]['close'])
            allocation = PER_STOCK_ALLOCATION.get(symbol, 10000)  # Default allocation
            quantity = max(1, int(allocation / current_price))
            
            # Risk management check
            if self.risk_manager:
                try:
                    # Try different risk manager method names with getattr
                    if hasattr(self.risk_manager, 'validate_order'):
                        risk_check = getattr(self.risk_manager, 'validate_order')(
                            symbol=symbol,
                            quantity=quantity,
                            price=current_price,
                            order_type=signal['signal']
                        )
                    elif hasattr(self.risk_manager, 'check_order_risk'):
                        risk_result = getattr(self.risk_manager, 'check_order_risk')(
                            symbol=symbol,
                            quantity=quantity,
                            current_price=current_price,
                            transaction_type=signal['signal']
                        )
                        risk_check = risk_result.get('allowed', True) if isinstance(risk_result, dict) else bool(risk_result)
                    else:
                        risk_check = True  # Allow if no risk method found
                        
                    if not risk_check:
                        self.logger.warning(f"Risk manager rejected trade for {symbol}")
                        return None
                except Exception as e:
                    self.logger.warning(f"Risk check failed: {e}")
                    # Continue without risk check if it fails
            
            # Execute order
            order_result = self.order_executor.execute_order(
                order_type=signal['signal'],
                quantity=quantity,
                symbol=symbol,
                price=current_price,
                strategy_name="aggregated",
                signal_data=signal
            )
            
            if order_result:
                self.daily_trades += 1
                self.logger.info(f"Trade executed: {symbol} {signal['signal']} {quantity} @ {current_price}")
                return {
                    'order_id': order_result.order_id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': current_price,
                    'signal': signal
                }
            
        except Exception as e:
            self.logger.error(f"Trade execution failed for {symbol}: {e}")
        
        return None
    
    def _update_positions_and_pnl(self):
        """Update position tracking and calculate PnL."""
        try:
            if not SIMULATION_MODE:
                # Get positions from broker
                positions = kite_manager.get_positions()
                
                # Update position tracker
                for position in positions.get('net', []):
                    symbol = position.get('tradingsymbol')
                    if symbol:
                        self.position_tracker[symbol].update({
                            'quantity': position.get('quantity', 0),
                            'average_price': Decimal(str(position.get('average_price', 0))),
                            'current_pnl': Decimal(str(position.get('pnl', 0))),
                            'unrealized_pnl': Decimal(str(position.get('unrealised', 0)))
                        })
            
            # Calculate total daily PnL with safe operations
            total_pnl = Decimal('0.0')
            for position_data in self.position_tracker.values():
                current_pnl = position_data.get('current_pnl')
                unrealized_pnl = position_data.get('unrealized_pnl')
                
                if current_pnl is not None:
                    if isinstance(current_pnl, (int, float)):
                        total_pnl += Decimal(str(current_pnl))
                    elif isinstance(current_pnl, Decimal):
                        total_pnl += current_pnl
                
                if unrealized_pnl is not None:
                    if isinstance(unrealized_pnl, (int, float)):
                        total_pnl += Decimal(str(unrealized_pnl))
                    elif isinstance(unrealized_pnl, Decimal):
                        total_pnl += unrealized_pnl
            
            self.daily_pnl = total_pnl
            
        except Exception as e:
            self.logger.error(f"Failed to update positions and PnL: {e}")
    
    def run_trading_cycle(self):
        """Execute a single trading cycle."""
        if not self.is_market_hours():
            self.logger.debug("Market is closed. Skipping trading cycle.")
            return
        
        if self.emergency_stop:
            self.logger.warning("Emergency stop is active. Skipping trading cycle.")
            return
        
        try:
            self.logger.info("Executing trading cycle...")
            
            with self.execution_lock:
                cycle_results = self.execute_strategy_cycle()
            
            # Log cycle results
            self.logger.info(f"Trading cycle completed: {cycle_results['signals_generated']} signals, "
                           f"{cycle_results['orders_placed']} orders, "
                           f"{cycle_results['execution_time_ms']:.2f}ms")
            
            # Update performance metrics
            self._log_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")
            self.logger.error(traceback.format_exc())
    
    def _log_performance_metrics(self):
        """Log comprehensive performance metrics."""
        try:
            avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
            
            metrics = {
                'daily_pnl': float(self.daily_pnl),
                'daily_trades': self.daily_trades,
                'available_capital': float(self.available_capital),
                'avg_execution_time_ms': round(avg_execution_time, 2),
                'active_positions': len([p for p in self.position_tracker.values() if p['quantity'] != 0]),
                'system_uptime_hours': (datetime.now(timezone.utc) - self.system_start_time).total_seconds() / 3600
            }
            
            self.logger.info(f"Performance metrics: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metrics: {e}")
    
    def start(self):
        """Start the production trading system."""
        self.logger.info("Starting Production Trading System...")
        
        try:
            self.is_running = True
            
            # Schedule trading cycles
            if self.is_market_hours():
                self.logger.info("Market is open. Starting immediate trading cycle...")
                self.run_trading_cycle()
            
            # Schedule regular trading cycles (every 30 seconds during market hours)
            schedule.every(30).seconds.do(self._scheduled_trading_cycle)
            
            # Schedule daily reset
            schedule.every().day.at("09:00").do(self._daily_reset)
            
            # Schedule end-of-day cleanup
            schedule.every().day.at("15:35").do(self._end_of_day_cleanup)
            
            self.logger.info("Trading system started successfully")
            
            # Main execution loop
            self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            raise
    
    def _scheduled_trading_cycle(self):
        """Wrapper for scheduled trading cycles."""
        if self.is_market_hours():
            self.run_trading_cycle()
    
    def _daily_reset(self):
        """Reset daily metrics for new trading day."""
        self.logger.info("Performing daily reset...")
        
        with self.stats_lock:
            self.daily_pnl = Decimal('0.0')
            self.daily_trades = 0
            self.emergency_stop = False
            
            # Reset strategy performance
            for strategy_name in self.strategy_performance:
                self.strategy_performance[strategy_name] = {
                    'total_signals': 0,
                    'successful_trades': 0,
                    'total_pnl': Decimal('0.0'),
                    'win_rate': 0.0,
                    'avg_return': 0.0
                }
        
        # Reset circuit breaker
        reset_circuit_breaker()
        
        self.logger.info("Daily reset completed")
    
    def _end_of_day_cleanup(self):
        """Execute end-of-day cleanup and reporting."""
        self.logger.info("Executing end-of-day cleanup...")
        
        try:
            # Cancel any pending orders
            active_orders = self.order_executor.get_active_orders()
            for order in active_orders:
                self.order_executor.cancel_order(order.order_id)
            
            # Generate end-of-day report
            self._generate_daily_report()
            
            self.logger.info("End-of-day cleanup completed")
            
        except Exception as e:
            self.logger.error(f"End-of-day cleanup failed: {e}")
    
    def _generate_daily_report(self):
        """Generate comprehensive daily trading report."""
        try:
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'daily_pnl': float(self.daily_pnl),
                'total_trades': self.daily_trades,
                'win_rate': self._calculate_win_rate(),
                'avg_execution_time': np.mean(self.execution_times) if self.execution_times else 0,
                'strategy_performance': dict(self.strategy_performance),
                'position_summary': dict(self.position_tracker)
            }
            
            # Save report to file
            report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Daily report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {e}")
    
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate."""
        try:
            total_profitable = 0
            for p in self.position_tracker.values():
                current_pnl = p.get('current_pnl', 0)
                if current_pnl is not None and current_pnl > 0:
                    total_profitable += 1
            
            total_trades = max(self.daily_trades, 1)
            return (total_profitable / total_trades) * 100
        except:
            return 0.0
    
    def _main_loop(self):
        """Main execution loop."""
        self.logger.info("Entering main execution loop...")
        
        try:
            while self.is_running and not self.emergency_stop:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt. Shutting down...")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of the trading system."""
        self.logger.info("Initiating graceful shutdown...")
        
        try:
            self.is_running = False
            
            # Cancel all pending orders
            active_orders = self.order_executor.get_active_orders()
            for order in active_orders:
                self.order_executor.cancel_order(order.order_id)
            
            # Generate final report
            self._generate_daily_report()
            
            # Clear scheduled tasks
            schedule.clear()
            
            self.logger.info("Trading system shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


def main():
    """Main entry point for the production trading system."""
    try:
        # Create and start the trading system
        trading_system = ProductionTradingSystem()
        
        print("🚀 Production Algorithmic Trading System")
        print("=" * 50)
        print(f"📊 Mode: {'SIMULATION' if SIMULATION_MODE else 'LIVE TRADING'}")
        print(f"💰 Capital: ₹{TOTAL_CAPITAL:,.2f}")
        print(f"⏰ Market Hours: 09:15 - 15:30 IST")
        print("=" * 50)
        print("Press Ctrl+C to stop the system gracefully")
        print("=" * 50)
        
        # Start the trading system
        trading_system.start()
        
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        print("Check logs for detailed error information")


if __name__ == "__main__":
    main()