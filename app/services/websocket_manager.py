"""
WebSocket Manager for Real-time Market Data Streaming
Integrates with HyperEVM Transaction Simulator for live data feeds
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from hyperliquid.info import Info
from hyperliquid.utils import constants
from structlog import get_logger

from app.core.config import settings

logger = get_logger(__name__)


@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    price: float
    timestamp: datetime
    volume: float = 0.0
    side: Optional[str] = None  # 'buy' or 'sell'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class OrderBookUpdate:
    """Order book L2 update"""
    symbol: str
    bids: List[Dict[str, float]]  # [{'px': price, 'sz': size}]
    asks: List[Dict[str, float]]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class UserEvent:
    """User-specific event (fills, orders, etc.)"""
    user_address: str
    event_type: str  # 'fill', 'order', 'liquidation'
    data: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class OrchestratorEvent:
    """Cross-chain orchestrator event"""
    strategy_id: str
    user_address: str
    event_type: str  # 'strategy_created', 'route_analyzed', 'bridge_started', 'bridge_completed', 'strategy_completed'
    status: str
    data: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


class WebSocketDataManager:
    """
    Manages real-time WebSocket connections to Hyperliquid
    Provides live data for enhanced transaction simulation
    """
    
    def __init__(self, use_mainnet: bool = False):
        """Initialize WebSocket data manager"""
        self.use_mainnet = use_mainnet
        self.base_url = constants.MAINNET_API_URL if use_mainnet else constants.TESTNET_API_URL
        
        # WebSocket connections
        self.info = Info(self.base_url, skip_ws=False)
        self.active_subscriptions: Set[str] = set()
        
        # Data storage
        self.market_ticks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.orderbook_snapshots: Dict[str, OrderBookUpdate] = {}
        self.user_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.orchestrator_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # Strategy events by user
        self.all_mids: Dict[str, float] = {}
        
        # Subscription callbacks
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance metrics
        self.message_count = 0
        self.last_message_time = time.time()
        self.connection_start_time = time.time()
        
        # Lock for thread-safe operations
        self.lock = threading.RLock()
        
        logger.info("WebSocket data manager initialized", 
                   mainnet=use_mainnet, base_url=self.base_url)
    
    def subscribe_to_market_data(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data for multiple symbols"""
        try:
            for symbol in symbols:
                subscription_key = f"market_data_{symbol}"
                
                if subscription_key not in self.active_subscriptions:
                    # Subscribe to L2 order book
                    self.info.subscribe(
                        {"type": "l2Book", "coin": symbol},
                        lambda data, s=symbol: self._handle_l2_book_update(s, data)
                    )
                    
                    # Subscribe to trades
                    self.info.subscribe(
                        {"type": "trades", "coin": symbol},
                        lambda data, s=symbol: self._handle_trade_update(s, data)
                    )
                    
                    self.active_subscriptions.add(subscription_key)
                    logger.info("Subscribed to market data", symbol=symbol)
            
            # Subscribe to all mids for price overview
            if "all_mids" not in self.active_subscriptions:
                self.info.subscribe(
                    {"type": "allMids"},
                    self._handle_all_mids_update
                )
                self.active_subscriptions.add("all_mids")
                
        except Exception as e:
            logger.error("Error subscribing to market data", symbols=symbols, error=str(e))
    
    def subscribe_to_user_events(self, user_address: str) -> None:
        """Subscribe to user-specific events"""
        try:
            subscription_key = f"user_events_{user_address}"
            
            if subscription_key not in self.active_subscriptions:
                self.info.subscribe(
                    {"type": "userEvents", "user": user_address},
                    lambda data: self._handle_user_event(user_address, data)
                )
                
                # Also subscribe to user fills
                self.info.subscribe(
                    {"type": "userFills", "user": user_address},
                    lambda data: self._handle_user_fills(user_address, data)
                )
                
                self.active_subscriptions.add(subscription_key)
                logger.info("Subscribed to user events", user=user_address)
                
        except Exception as e:
            logger.error("Error subscribing to user events", user=user_address, error=str(e))
    
    def _handle_l2_book_update(self, symbol: str, data: Dict[str, Any]) -> None:
        """Handle L2 order book updates"""
        try:
            with self.lock:
                self.message_count += 1
                self.last_message_time = time.time()
                
                # Extract order book data
                levels = data.get('levels', [[], []])
                bids = levels[0] if len(levels) > 0 else []
                asks = levels[1] if len(levels) > 1 else []
                
                # Create order book update
                orderbook_update = OrderBookUpdate(
                    symbol=symbol,
                    bids=[{'px': float(b['px']), 'sz': float(b['sz'])} for b in bids[:10]],
                    asks=[{'px': float(a['px']), 'sz': float(a['sz'])} for a in asks[:10]],
                    timestamp=datetime.utcnow()
                )
                
                # Store latest snapshot
                self.orderbook_snapshots[symbol] = orderbook_update
                
                # Trigger callbacks
                self._trigger_callbacks(f"orderbook_{symbol}", orderbook_update)
                
        except Exception as e:
            logger.error("Error handling L2 book update", symbol=symbol, error=str(e))
    
    def _handle_trade_update(self, symbol: str, data: Any) -> None:
        """Handle trade updates"""
        try:
            with self.lock:
                self.message_count += 1
                
                # Handle different data formats from WebSocket
                if isinstance(data, dict):
                    # Single trade
                    trades = [data]
                elif isinstance(data, list):
                    trades = data
                else:
                    # Skip invalid data format
                    logger.warning("Invalid trade data format", symbol=symbol, data_type=type(data))
                    return
                
                for trade in trades:
                    if not isinstance(trade, dict):
                        logger.warning("Invalid trade item format", symbol=symbol, trade_type=type(trade))
                        continue
                        
                    # Create market tick
                    tick = MarketTick(
                        symbol=symbol,
                        price=float(trade.get('px', 0)),
                        volume=float(trade.get('sz', 0)),
                        side=trade.get('side'),
                        timestamp=datetime.utcnow()
                    )
                    
                    # Store tick
                    self.market_ticks[symbol].append(tick)
                    
                    # Trigger callbacks
                    self._trigger_callbacks(f"trade_{symbol}", tick)
                
        except Exception as e:
            logger.error("Error handling trade update", symbol=symbol, error=str(e))
    
    def _handle_all_mids_update(self, data: Any) -> None:
        """Handle all mids price update"""
        try:
            with self.lock:
                self.message_count += 1
                
                # Handle different data formats from WebSocket
                if isinstance(data, dict):
                    # Check if it's a control message
                    if 'channel' in data:
                        logger.debug("Received channel control message", data=data)
                        return
                    
                    # Handle dict format with mids data
                    mids_data = data.get('mids', data.get('data', []))
                    if isinstance(mids_data, list):
                        data = mids_data
                    else:
                        return
                
                elif not isinstance(data, list):
                    logger.warning("Invalid allMids data format", data_type=type(data))
                    return
                
                # Update all mid prices
                for i, price in enumerate(data):
                    if price and isinstance(price, (str, float, int)):
                        try:
                            price_float = float(price)
                            if price_float > 0:  # Valid price
                                self.all_mids[f"asset_{i}"] = price_float
                        except (ValueError, TypeError):
                            # Skip invalid price values
                            continue
                
                # Trigger callbacks
                self._trigger_callbacks("all_mids", self.all_mids.copy())
                
        except Exception as e:
            logger.error("Error handling all mids update", error=str(e))
    
    def _handle_user_event(self, user_address: str, data: Dict[str, Any]) -> None:
        """Handle user-specific events"""
        try:
            with self.lock:
                self.message_count += 1
                
                event = UserEvent(
                    user_address=user_address,
                    event_type="user_event",
                    data=data,
                    timestamp=datetime.utcnow()
                )
                
                self.user_events[user_address].append(event)
                
                # Trigger callbacks
                self._trigger_callbacks(f"user_{user_address}", event)
                
        except Exception as e:
            logger.error("Error handling user event", user=user_address, error=str(e))
    
    def _handle_user_fills(self, user_address: str, data: List[Dict[str, Any]]) -> None:
        """Handle user fills"""
        try:
            with self.lock:
                for fill in data:
                    event = UserEvent(
                        user_address=user_address,
                        event_type="fill",
                        data=fill,
                        timestamp=datetime.utcnow()
                    )
                    
                    self.user_events[user_address].append(event)
                    
                    # Trigger callbacks
                    self._trigger_callbacks(f"fill_{user_address}", event)
                    
        except Exception as e:
            logger.error("Error handling user fills", user=user_address, error=str(e))
    
    def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger registered callbacks for an event type"""
        try:
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(data)
                except Exception as e:
                    logger.error("Callback error", event_type=event_type, error=str(e))
                    
        except Exception as e:
            logger.error("Error triggering callbacks", event_type=event_type, error=str(e))
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback for specific event type"""
        with self.lock:
            self.callbacks[event_type].append(callback)
            logger.debug("Callback registered", event_type=event_type)
    
    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookUpdate]:
        """Get latest order book snapshot"""
        with self.lock:
            return self.orderbook_snapshots.get(symbol)
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[MarketTick]:
        """Get recent trades for a symbol"""
        with self.lock:
            trades = list(self.market_ticks.get(symbol, []))
            return trades[-limit:] if trades else []
    
    def get_user_events(self, user_address: str, limit: int = 50) -> List[UserEvent]:
        """Get recent user events"""
        with self.lock:
            events = list(self.user_events.get(user_address, []))
            return events[-limit:] if events else []
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """Get current live price for a symbol"""
        with self.lock:
            # Try to get from recent trades first
            recent_trades = self.get_recent_trades(symbol, 1)
            if recent_trades:
                return recent_trades[-1].price
            
            # Fall back to order book mid
            orderbook = self.get_latest_orderbook(symbol)
            if orderbook and orderbook.bids and orderbook.asks:
                bid = orderbook.bids[0]['px']
                ask = orderbook.asks[0]['px']
                return (bid + ask) / 2
            
            return None
    
    def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Get market depth metrics"""
        with self.lock:
            orderbook = self.get_latest_orderbook(symbol)
            if not orderbook:
                return {}
            
            # Calculate depth metrics
            total_bid_volume = sum(level['sz'] for level in orderbook.bids)
            total_ask_volume = sum(level['sz'] for level in orderbook.asks)
            
            spread = 0.0
            if orderbook.bids and orderbook.asks:
                spread = orderbook.asks[0]['px'] - orderbook.bids[0]['px']
            
            return {
                'symbol': symbol,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'spread': spread,
                'levels_count': len(orderbook.bids) + len(orderbook.asks),
                'last_update': orderbook.timestamp.isoformat()
            }
    
    def get_volatility_estimate(self, symbol: str, window_minutes: int = 5) -> float:
        """Calculate recent volatility estimate"""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            recent_trades = [
                trade for trade in self.market_ticks.get(symbol, [])
                if trade.timestamp >= cutoff_time
            ]
            
            if len(recent_trades) < 2:
                return 0.0
            
            prices = [trade.price for trade in recent_trades]
            
            # Calculate returns and standard deviation
            returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
            
            if not returns:
                return 0.0
            
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            
            # Annualized volatility estimate
            volatility = (variance ** 0.5) * (252 * 24 * 60 / window_minutes) ** 0.5
            
            return volatility
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        with self.lock:
            uptime_seconds = time.time() - self.connection_start_time
            
            return {
                'uptime_seconds': uptime_seconds,
                'messages_received': self.message_count,
                'messages_per_second': self.message_count / max(uptime_seconds, 1),
                'active_subscriptions': len(self.active_subscriptions),
                'last_message_age_seconds': time.time() - self.last_message_time,
                'symbols_tracked': len(self.orderbook_snapshots),
                'users_tracked': len(self.user_events),
                'connection_healthy': time.time() - self.last_message_time < 60
            }
    
    async def cleanup(self) -> None:
        """Cleanup WebSocket connections"""
        try:
            if hasattr(self.info, 'disconnect_websocket'):
                self.info.disconnect_websocket()
            
            self.active_subscriptions.clear()
            self.callbacks.clear()
            
            logger.info("WebSocket manager cleaned up")
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
    
    def emit_orchestrator_event(
        self,
        strategy_id: str,
        user_address: str,
        event_type: str,
        status: str,
        data: Dict[str, Any]
    ) -> None:
        """Emit orchestrator event to WebSocket subscribers"""
        try:
            event = OrchestratorEvent(
                strategy_id=strategy_id,
                user_address=user_address,
                event_type=event_type,
                status=status,
                data=data,
                timestamp=datetime.utcnow()
            )
            
            with self.lock:
                # Store event for user
                self.orchestrator_events[user_address].append(event)
                
                # Trigger callbacks for orchestrator events
                for callback in self.callbacks.get('orchestrator', []):
                    try:
                        callback(event.to_dict())
                    except Exception as e:
                        logger.error("Error in orchestrator callback", error=str(e))
                
                # Trigger user-specific callbacks
                callback_key = f"orchestrator:{user_address}"
                for callback in self.callbacks.get(callback_key, []):
                    try:
                        callback(event.to_dict())
                    except Exception as e:
                        logger.error("Error in user orchestrator callback", error=str(e))
            
            logger.debug("Orchestrator event emitted", 
                        strategy_id=strategy_id, 
                        event_type=event_type,
                        status=status)
            
        except Exception as e:
            logger.error("Failed to emit orchestrator event", error=str(e))
    
    def get_orchestrator_events(self, user_address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent orchestrator events for a user"""
        with self.lock:
            events = list(self.orchestrator_events[user_address])
            events = events[-limit:] if len(events) > limit else events
            return [event.to_dict() for event in events]
    
    def subscribe_to_orchestrator_events(self, user_address: str, callback: Callable) -> None:
        """Subscribe to orchestrator events for a specific user"""
        callback_key = f"orchestrator:{user_address}"
        self.callbacks[callback_key].append(callback)
        logger.info("Subscribed to orchestrator events", user=user_address)
    
    def unsubscribe_from_orchestrator_events(self, user_address: str, callback: Callable) -> None:
        """Unsubscribe from orchestrator events"""
        callback_key = f"orchestrator:{user_address}"
        if callback in self.callbacks[callback_key]:
            self.callbacks[callback_key].remove(callback)
            logger.info("Unsubscribed from orchestrator events", user=user_address)


# Global WebSocket manager instance
_ws_manager: Optional[WebSocketDataManager] = None
_manager_lock = threading.Lock()


def get_websocket_manager(use_mainnet: bool = None) -> WebSocketDataManager:
    """Get or create global WebSocket manager instance"""
    global _ws_manager
    
    if use_mainnet is None:
        use_mainnet = getattr(settings, 'HYPERLIQUID_NETWORK', 'testnet') == 'mainnet'
    
    with _manager_lock:
        if _ws_manager is None:
            _ws_manager = WebSocketDataManager(use_mainnet=use_mainnet)
        
        return _ws_manager


async def initialize_websocket_subscriptions():
    """Initialize common WebSocket subscriptions"""
    try:
        manager = get_websocket_manager()
        
        # Subscribe to major trading pairs
        major_symbols = ['ETH', 'BTC', 'SOL', 'ARB', 'OP', 'AVAX']
        manager.subscribe_to_market_data(major_symbols)
        
        logger.info("WebSocket subscriptions initialized", symbols=major_symbols)
        
    except Exception as e:
        logger.error("Failed to initialize WebSocket subscriptions", error=str(e))


# Utility function for enhanced simulation with live data
def enhance_simulation_with_live_data(symbol: str, base_simulation: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance transaction simulation with real-time WebSocket data"""
    try:
        manager = get_websocket_manager()
        
        # Get live market data
        live_price = manager.get_live_price(symbol)
        market_depth = manager.get_market_depth(symbol)
        volatility = manager.get_volatility_estimate(symbol)
        recent_trades = manager.get_recent_trades(symbol, 10)
        
        # Enhanced simulation data
        enhanced = base_simulation.copy()
        enhanced['live_data'] = {
            'live_price': live_price,
            'market_depth': market_depth,
            'volatility_estimate': volatility,
            'recent_trade_count': len(recent_trades),
            'data_freshness_seconds': (
                (datetime.utcnow() - recent_trades[-1].timestamp).total_seconds()
                if recent_trades else None
            )
        }
        
        # Adjust risk score based on live data
        if volatility > 0.5:  # High volatility
            enhanced['risk_score'] = min(1.0, enhanced.get('risk_score', 0.5) + 0.2)
            enhanced['warnings'] = enhanced.get('warnings', []) + ['High recent volatility detected']
        
        # Adjust slippage based on market depth
        if market_depth.get('spread', 0) > live_price * 0.005:  # Wide spread
            enhanced['slippage_estimate'] = enhanced.get('slippage_estimate', 0) * 1.5
            enhanced['warnings'] = enhanced.get('warnings', []) + ['Wide bid-ask spread']
        
        enhanced['data_sources'] = enhanced.get('data_sources', []) + ['websocket_live_data']
        
        return enhanced
        
    except Exception as e:
        logger.error("Error enhancing simulation with live data", symbol=symbol, error=str(e))
        return base_simulation