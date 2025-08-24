"""
TWAP (Time-Weighted Average Price) Order Executor Service
$5,000 Hackathon Bounty Implementation

Provides shielded TWAP execution with privacy-preserving features,
market-aware adjustments, and integration with HyperCore/HyperEVM.
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import uuid

from app.services.hyperliquid_client import HyperliquidClient
from app.adapters.lifi import LiFiAdapter
from app.core.config import settings

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class TWAPStrategy(Enum):
    EQUAL_INTERVAL = "equal_interval"
    VOLUME_WEIGHTED = "volume_weighted"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"

class PrivacyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MAXIMUM = 4

@dataclass
class TWAPConfig:
    """Configuration for TWAP execution"""
    total_amount: Decimal
    symbol: str
    side: OrderSide
    duration_minutes: int
    max_slippage: Decimal  # Percentage
    privacy_level: PrivacyLevel
    strategy: TWAPStrategy
    min_order_size: Optional[Decimal] = None
    max_order_size: Optional[Decimal] = None
    randomness_factor: Decimal = Decimal("0.1")  # 10% randomness
    pause_threshold: Decimal = Decimal("0.5")  # Pause if spread > 0.5%
    cross_chain: bool = False
    target_chain: Optional[str] = None

@dataclass
class TWAPOrder:
    """Individual order within TWAP execution"""
    order_id: str
    amount: Decimal
    expected_time: datetime
    actual_time: Optional[datetime] = None
    price: Optional[Decimal] = None
    status: str = "pending"  # pending, executed, failed, skipped
    transaction_hash: Optional[str] = None
    gas_used: Optional[int] = None
    slippage: Optional[Decimal] = None

@dataclass
class TWAPExecution:
    """Complete TWAP execution state"""
    execution_id: str
    config: TWAPConfig
    orders: List[TWAPOrder]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"  # active, completed, paused, cancelled
    executed_amount: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    total_slippage: Optional[Decimal] = None
    total_gas_used: Optional[int] = None
    benchmark_price: Optional[Decimal] = None
    performance_vs_benchmark: Optional[Decimal] = None

class MarketDataProvider:
    """Provides real-time market data for TWAP execution"""
    
    def __init__(self, hyperliquid_client: HyperliquidClient):
        self.hyperliquid_client = hyperliquid_client
        self.price_cache = {}
        self.orderbook_cache = {}
        self.volume_cache = {}
    
    async def get_current_price(self, symbol: str) -> Decimal:
        """Get current market price"""
        try:
            # Use Hyperliquid WebSocket or REST API
            market_data = await self.hyperliquid_client.get_l2_book(symbol)
            if market_data and "levels" in market_data:
                bid = Decimal(market_data["levels"][0][0]["px"]) if market_data["levels"][0] else Decimal("0")
                ask = Decimal(market_data["levels"][1][0]["px"]) if market_data["levels"][1] else Decimal("0")
                mid_price = (bid + ask) / 2
                self.price_cache[symbol] = {
                    "price": mid_price,
                    "bid": bid,
                    "ask": ask,
                    "spread": ask - bid,
                    "timestamp": datetime.now()
                }
                return mid_price
            return Decimal("0")
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return Decimal("0")
    
    async def get_spread(self, symbol: str) -> Decimal:
        """Get current bid-ask spread"""
        await self.get_current_price(symbol)  # Updates cache
        if symbol in self.price_cache:
            return self.price_cache[symbol]["spread"]
        return Decimal("0")
    
    async def get_volume_profile(self, symbol: str, lookback_minutes: int = 60) -> Dict[str, Any]:
        """Get recent volume profile for VWAP calculations"""
        try:
            # Get recent trades
            trades = await self.hyperliquid_client.get_user_fills(symbol)
            
            # Calculate volume in time buckets
            now = datetime.now()
            buckets = {}
            bucket_size = 5  # 5-minute buckets
            
            for trade in trades[-100:]:  # Last 100 trades
                trade_time = datetime.fromisoformat(trade.get("time", ""))
                if (now - trade_time).total_seconds() > lookback_minutes * 60:
                    continue
                
                bucket = int((now - trade_time).total_seconds() // (bucket_size * 60))
                if bucket not in buckets:
                    buckets[bucket] = {"volume": Decimal("0"), "count": 0}
                
                buckets[bucket]["volume"] += Decimal(trade.get("sz", "0"))
                buckets[bucket]["count"] += 1
            
            # Calculate average volume per bucket
            total_volume = sum(bucket["volume"] for bucket in buckets.values())
            avg_volume_per_bucket = total_volume / len(buckets) if buckets else Decimal("0")
            
            return {
                "total_volume": total_volume,
                "avg_volume_per_bucket": avg_volume_per_bucket,
                "buckets": buckets,
                "lookback_minutes": lookback_minutes
            }
            
        except Exception as e:
            logger.error(f"Failed to get volume profile for {symbol}: {e}")
            return {"total_volume": Decimal("0"), "avg_volume_per_bucket": Decimal("0")}

class PrivacyEnhancer:
    """Implements privacy-preserving features for TWAP execution"""
    
    def __init__(self, privacy_level: PrivacyLevel):
        self.privacy_level = privacy_level
        self.wallet_rotation_enabled = privacy_level.value >= 3
        self.wallet_pool = []
    
    def add_randomness_to_timing(self, base_interval: int) -> int:
        """Add randomness to order timing"""
        if self.privacy_level.value <= 1:
            return base_interval
        
        # Add up to 25% randomness for high privacy
        max_variance = int(base_interval * 0.25 * self.privacy_level.value / 4)
        return base_interval + random.randint(-max_variance, max_variance)
    
    def add_randomness_to_size(self, base_size: Decimal, randomness_factor: Decimal) -> Decimal:
        """Add randomness to order size"""
        if self.privacy_level.value <= 1:
            return base_size
        
        max_variance = base_size * randomness_factor * self.privacy_level.value / 4
        variance = Decimal(random.uniform(float(-max_variance), float(max_variance)))
        return max(base_size + variance, base_size * Decimal("0.1"))  # Minimum 10% of base size
    
    def should_rotate_wallet(self) -> bool:
        """Determine if wallet should be rotated"""
        if not self.wallet_rotation_enabled:
            return False
        
        # Rotate based on privacy level and randomness
        rotation_probability = self.privacy_level.value * 0.1  # 10-40% chance
        return random.random() < rotation_probability
    
    def get_privacy_delay(self) -> int:
        """Get additional delay for privacy"""
        if self.privacy_level.value <= 1:
            return 0
        
        # Add 0-30 seconds delay based on privacy level
        max_delay = self.privacy_level.value * 7.5
        return random.randint(0, int(max_delay))

class TWAPOrderExecutor:
    """Main TWAP execution engine"""
    
    def __init__(self):
        self.hyperliquid_client = HyperliquidClient()
        self.lifi_adapter = LiFiAdapter()
        self.market_data = MarketDataProvider(self.hyperliquid_client)
        self.active_executions: Dict[str, TWAPExecution] = {}
        self.signal_callbacks: List[Callable] = []
    
    def add_signal_callback(self, callback: Callable):
        """Add a signal callback for strategy adjustments"""
        self.signal_callbacks.append(callback)
    
    async def create_twap_execution(self, config: TWAPConfig) -> str:
        """Create a new TWAP execution plan"""
        execution_id = str(uuid.uuid4())
        
        # Calculate order schedule
        orders = await self._calculate_order_schedule(config)
        
        execution = TWAPExecution(
            execution_id=execution_id,
            config=config,
            orders=orders,
            start_time=datetime.now(),
            status="active"
        )
        
        self.active_executions[execution_id] = execution
        
        logger.info(f"Created TWAP execution {execution_id} with {len(orders)} orders")
        return execution_id
    
    async def _calculate_order_schedule(self, config: TWAPConfig) -> List[TWAPOrder]:
        """Calculate the schedule of orders for TWAP execution"""
        orders = []
        privacy_enhancer = PrivacyEnhancer(config.privacy_level)
        
        if config.strategy == TWAPStrategy.EQUAL_INTERVAL:
            orders = await self._calculate_equal_interval_orders(config, privacy_enhancer)
        elif config.strategy == TWAPStrategy.VOLUME_WEIGHTED:
            orders = await self._calculate_volume_weighted_orders(config, privacy_enhancer)
        elif config.strategy == TWAPStrategy.ADAPTIVE:
            orders = await self._calculate_adaptive_orders(config, privacy_enhancer)
        elif config.strategy == TWAPStrategy.HYBRID:
            orders = await self._calculate_hybrid_orders(config, privacy_enhancer)
        
        return orders
    
    async def _calculate_equal_interval_orders(
        self, 
        config: TWAPConfig, 
        privacy_enhancer: PrivacyEnhancer
    ) -> List[TWAPOrder]:
        """Calculate orders with equal time intervals"""
        orders = []
        num_orders = max(5, config.duration_minutes // 2)  # At least 5 orders, or one every 2 minutes
        base_interval = config.duration_minutes * 60 // num_orders  # seconds
        base_size = config.total_amount / num_orders
        
        current_time = datetime.now()
        
        for i in range(num_orders):
            # Add privacy randomness
            interval = privacy_enhancer.add_randomness_to_timing(base_interval)
            size = privacy_enhancer.add_randomness_to_size(base_size, config.randomness_factor)
            
            # Ensure size constraints
            if config.min_order_size and size < config.min_order_size:
                size = config.min_order_size
            if config.max_order_size and size > config.max_order_size:
                size = config.max_order_size
            
            execution_time = current_time + timedelta(seconds=interval * i)
            
            order = TWAPOrder(
                order_id=str(uuid.uuid4()),
                amount=size,
                expected_time=execution_time
            )
            orders.append(order)
        
        return orders
    
    async def _calculate_volume_weighted_orders(
        self, 
        config: TWAPConfig, 
        privacy_enhancer: PrivacyEnhancer
    ) -> List[TWAPOrder]:
        """Calculate orders weighted by historical volume"""
        volume_profile = await self.market_data.get_volume_profile(config.symbol)
        
        # If no volume data, fall back to equal interval
        if volume_profile["total_volume"] == 0:
            return await self._calculate_equal_interval_orders(config, privacy_enhancer)
        
        orders = []
        buckets = volume_profile["buckets"]
        total_weight = sum(bucket["volume"] for bucket in buckets.values())
        
        current_time = datetime.now()
        base_interval = config.duration_minutes * 60 // len(buckets)
        
        for i, (bucket_id, bucket_data) in enumerate(buckets.items()):
            # Weight order size by volume
            weight = bucket_data["volume"] / total_weight if total_weight > 0 else Decimal("1") / len(buckets)
            size = config.total_amount * weight
            
            # Add privacy randomness
            size = privacy_enhancer.add_randomness_to_size(size, config.randomness_factor)
            interval = privacy_enhancer.add_randomness_to_timing(base_interval)
            
            # Ensure size constraints
            if config.min_order_size and size < config.min_order_size:
                size = config.min_order_size
            if config.max_order_size and size > config.max_order_size:
                size = config.max_order_size
            
            execution_time = current_time + timedelta(seconds=interval * i)
            
            order = TWAPOrder(
                order_id=str(uuid.uuid4()),
                amount=size,
                expected_time=execution_time
            )
            orders.append(order)
        
        return orders
    
    async def _calculate_adaptive_orders(
        self, 
        config: TWAPConfig, 
        privacy_enhancer: PrivacyEnhancer
    ) -> List[TWAPOrder]:
        """Calculate orders that adapt to market conditions"""
        orders = []
        num_orders = max(5, config.duration_minutes // 3)  # More adaptive chunks
        base_interval = config.duration_minutes * 60 // num_orders
        base_size = config.total_amount / num_orders
        
        current_time = datetime.now()
        
        for i in range(num_orders):
            # Adaptive sizing based on time of day and market conditions
            time_factor = self._get_time_of_day_factor(current_time + timedelta(seconds=base_interval * i))
            size = base_size * time_factor
            
            # Add privacy randomness
            size = privacy_enhancer.add_randomness_to_size(size, config.randomness_factor)
            interval = privacy_enhancer.add_randomness_to_timing(base_interval)
            
            # Ensure size constraints
            if config.min_order_size and size < config.min_order_size:
                size = config.min_order_size
            if config.max_order_size and size > config.max_order_size:
                size = config.max_order_size
            
            execution_time = current_time + timedelta(seconds=interval * i)
            
            order = TWAPOrder(
                order_id=str(uuid.uuid4()),
                amount=size,
                expected_time=execution_time
            )
            orders.append(order)
        
        return orders
    
    async def _calculate_hybrid_orders(
        self, 
        config: TWAPConfig, 
        privacy_enhancer: PrivacyEnhancer
    ) -> List[TWAPOrder]:
        """Calculate orders using hybrid TWAP/VWAP approach"""
        # Combine volume weighting with time adaptiveness
        volume_profile = await self.market_data.get_volume_profile(config.symbol)
        
        orders = []
        num_orders = max(8, config.duration_minutes // 2)  # More granular for hybrid
        base_interval = config.duration_minutes * 60 // num_orders
        
        current_time = datetime.now()
        
        for i in range(num_orders):
            execution_time = current_time + timedelta(seconds=base_interval * i)
            
            # Volume factor
            volume_factor = self._get_volume_factor(volume_profile, i, num_orders)
            
            # Time factor
            time_factor = self._get_time_of_day_factor(execution_time)
            
            # Combined factor
            combined_factor = (volume_factor + time_factor) / 2
            size = (config.total_amount / num_orders) * combined_factor
            
            # Add privacy randomness
            size = privacy_enhancer.add_randomness_to_size(size, config.randomness_factor)
            interval = privacy_enhancer.add_randomness_to_timing(base_interval)
            
            # Ensure size constraints
            if config.min_order_size and size < config.min_order_size:
                size = config.min_order_size
            if config.max_order_size and size > config.max_order_size:
                size = config.max_order_size
            
            order = TWAPOrder(
                order_id=str(uuid.uuid4()),
                amount=size,
                expected_time=execution_time
            )
            orders.append(order)
        
        return orders
    
    def _get_time_of_day_factor(self, execution_time: datetime) -> Decimal:
        """Get scaling factor based on time of day (higher during active hours)"""
        hour = execution_time.hour
        
        # Higher activity during typical trading hours (9 AM - 4 PM UTC)
        if 9 <= hour <= 16:
            return Decimal("1.2")  # 20% higher
        elif 6 <= hour <= 9 or 16 <= hour <= 20:
            return Decimal("1.0")  # Normal
        else:
            return Decimal("0.8")  # 20% lower during off hours
    
    def _get_volume_factor(self, volume_profile: Dict, order_index: int, total_orders: int) -> Decimal:
        """Get scaling factor based on historical volume"""
        if not volume_profile["buckets"]:
            return Decimal("1.0")
        
        # Map order index to volume bucket
        bucket_keys = list(volume_profile["buckets"].keys())
        bucket_index = order_index * len(bucket_keys) // total_orders
        bucket_index = min(bucket_index, len(bucket_keys) - 1)
        
        bucket_key = bucket_keys[bucket_index]
        bucket_volume = volume_profile["buckets"][bucket_key]["volume"]
        avg_volume = volume_profile["avg_volume_per_bucket"]
        
        if avg_volume > 0:
            return bucket_volume / avg_volume
        return Decimal("1.0")
    
    async def execute_twap(self, execution_id: str) -> None:
        """Execute TWAP orders according to schedule"""
        if execution_id not in self.active_executions:
            raise ValueError(f"TWAP execution {execution_id} not found")
        
        execution = self.active_executions[execution_id]
        privacy_enhancer = PrivacyEnhancer(execution.config.privacy_level)
        
        logger.info(f"Starting TWAP execution {execution_id}")
        
        try:
            total_executed = Decimal("0")
            total_slippage = Decimal("0")
            total_gas = 0
            executed_count = 0
            benchmark_sum = Decimal("0")
            
            for order in execution.orders:
                if execution.status != "active":
                    break
                
                # Wait until scheduled time
                now = datetime.now()
                if order.expected_time > now:
                    wait_seconds = (order.expected_time - now).total_seconds()
                    
                    # Add privacy delay
                    privacy_delay = privacy_enhancer.get_privacy_delay()
                    await asyncio.sleep(wait_seconds + privacy_delay)
                
                # Check market conditions before execution
                if await self._should_pause_execution(execution.config):
                    logger.info(f"Pausing execution {execution_id} due to market conditions")
                    execution.status = "paused"
                    break
                
                # Apply signal callbacks for strategy adjustments
                for callback in self.signal_callbacks:
                    try:
                        adjustment = await callback(execution, order)
                        if adjustment:
                            order.amount = adjustment.get("size", order.amount)
                    except Exception as e:
                        logger.error(f"Signal callback failed: {e}")
                
                # Execute the order
                result = await self._execute_single_order(execution.config, order)
                
                if result["success"]:
                    order.status = "executed"
                    order.actual_time = datetime.now()
                    order.price = result["price"]
                    order.transaction_hash = result.get("tx_hash")
                    order.gas_used = result.get("gas_used", 0)
                    order.slippage = result.get("slippage", Decimal("0"))
                    
                    total_executed += order.amount
                    total_slippage += order.slippage
                    total_gas += order.gas_used
                    executed_count += 1
                    benchmark_sum += order.price
                    
                    logger.info(f"Executed order {order.order_id}: {order.amount} at {order.price}")
                else:
                    order.status = "failed"
                    logger.error(f"Failed to execute order {order.order_id}: {result.get('error')}")
                
                # Update execution state
                execution.executed_amount = total_executed
                if executed_count > 0:
                    execution.average_price = benchmark_sum / executed_count
                    execution.total_slippage = total_slippage / executed_count
                    execution.total_gas_used = total_gas
            
            # Mark execution as completed
            execution.status = "completed"
            execution.end_time = datetime.now()
            
            # Calculate performance vs benchmark
            if execution.average_price:
                current_price = await self.market_data.get_current_price(execution.config.symbol)
                if current_price > 0:
                    execution.benchmark_price = current_price
                    execution.performance_vs_benchmark = (
                        (execution.average_price - current_price) / current_price * 100
                    )
            
            logger.info(f"Completed TWAP execution {execution_id}")
            
        except Exception as e:
            logger.error(f"TWAP execution {execution_id} failed: {e}")
            execution.status = "failed"
            raise
    
    async def _should_pause_execution(self, config: TWAPConfig) -> bool:
        """Check if execution should be paused due to market conditions"""
        try:
            spread = await self.market_data.get_spread(config.symbol)
            current_price = await self.market_data.get_current_price(config.symbol)
            
            if current_price == 0:
                return True  # No price data
            
            spread_percentage = (spread / current_price) * 100
            
            # Pause if spread is too wide
            return spread_percentage > config.pause_threshold
            
        except Exception as e:
            logger.error(f"Failed to check market conditions: {e}")
            return True  # Pause on error
    
    async def _execute_single_order(self, config: TWAPConfig, order: TWAPOrder) -> Dict[str, Any]:
        """Execute a single order"""
        try:
            if config.cross_chain and config.target_chain:
                # Use LI.FI for cross-chain execution
                return await self._execute_cross_chain_order(config, order)
            else:
                # Use Hyperliquid direct execution
                return await self._execute_hyperliquid_order(config, order)
                
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_hyperliquid_order(self, config: TWAPConfig, order: TWAPOrder) -> Dict[str, Any]:
        """Execute order on Hyperliquid"""
        try:
            # Place order using Hyperliquid client
            order_result = await self.hyperliquid_client.place_order(
                symbol=config.symbol,
                side=config.side.value,
                amount=float(order.amount),
                order_type="Market",  # Market order for TWAP
                reduce_only=False
            )
            
            if order_result and "status" in order_result:
                # Get execution price from order result
                fills = order_result.get("fills", [])
                if fills:
                    avg_price = sum(Decimal(fill["price"]) * Decimal(fill["sz"]) for fill in fills)
                    avg_price /= sum(Decimal(fill["sz"]) for fill in fills)
                    
                    return {
                        "success": True,
                        "price": avg_price,
                        "tx_hash": order_result.get("oid"),
                        "gas_used": 0,  # No gas on Hyperliquid
                        "slippage": self._calculate_slippage(avg_price, config.symbol)
                    }
            
            return {"success": False, "error": "Order execution failed"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_cross_chain_order(self, config: TWAPConfig, order: TWAPOrder) -> Dict[str, Any]:
        """Execute cross-chain order using LI.FI"""
        try:
            # Get current price for slippage calculation
            current_price = await self.market_data.get_current_price(config.symbol)
            
            # Use LI.FI for cross-chain execution
            routes = await self.lifi_adapter.get_routes(
                from_chain="1",  # Ethereum
                to_chain=config.target_chain,
                from_token="0xA0b86a33E6047cdB6Da4e3AA67fefE76Ac109f23",  # Example token
                to_token="0x...",  # Target token
                amount=str(int(order.amount * 10**18)),  # Convert to wei
                user_address="0x..."  # User address
            )
            
            if routes:
                best_route = routes[0]
                # Execute the route
                result = await self.lifi_adapter.execute_route(best_route)
                
                if result.get("success"):
                    executed_price = Decimal(result.get("executed_price", current_price))
                    
                    return {
                        "success": True,
                        "price": executed_price,
                        "tx_hash": result.get("transactionHash"),
                        "gas_used": result.get("gasUsed", 0),
                        "slippage": self._calculate_slippage(executed_price, config.symbol)
                    }
            
            return {"success": False, "error": "No routes available"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_slippage(self, executed_price: Decimal, symbol: str) -> Decimal:
        """Calculate slippage against current market price"""
        try:
            if symbol in self.market_data.price_cache:
                market_price = self.market_data.price_cache[symbol]["price"]
                if market_price > 0:
                    return abs((executed_price - market_price) / market_price) * 100
            return Decimal("0")
        except Exception:
            return Decimal("0")
    
    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get current status of TWAP execution"""
        if execution_id not in self.active_executions:
            return {"error": "Execution not found"}
        
        execution = self.active_executions[execution_id]
        
        executed_orders = [order for order in execution.orders if order.status == "executed"]
        pending_orders = [order for order in execution.orders if order.status == "pending"]
        failed_orders = [order for order in execution.orders if order.status == "failed"]
        
        return {
            "execution_id": execution_id,
            "status": execution.status,
            "config": asdict(execution.config),
            "progress": {
                "total_orders": len(execution.orders),
                "executed_orders": len(executed_orders),
                "pending_orders": len(pending_orders),
                "failed_orders": len(failed_orders),
                "executed_amount": str(execution.executed_amount),
                "remaining_amount": str(execution.config.total_amount - execution.executed_amount)
            },
            "performance": {
                "average_price": str(execution.average_price) if execution.average_price else None,
                "total_slippage": str(execution.total_slippage) if execution.total_slippage else None,
                "total_gas_used": execution.total_gas_used,
                "benchmark_price": str(execution.benchmark_price) if execution.benchmark_price else None,
                "performance_vs_benchmark": str(execution.performance_vs_benchmark) if execution.performance_vs_benchmark else None
            },
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active TWAP execution"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status == "active":
            execution.status = "cancelled"
            execution.end_time = datetime.now()
            logger.info(f"Cancelled TWAP execution {execution_id}")
            return True
        
        return False
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active TWAP execution"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status == "active":
            execution.status = "paused"
            logger.info(f"Paused TWAP execution {execution_id}")
            return True
        
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused TWAP execution"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status == "paused":
            execution.status = "active"
            logger.info(f"Resumed TWAP execution {execution_id}")
            # Continue execution in background
            asyncio.create_task(self.execute_twap(execution_id))
            return True
        
        return False

# Example signal callback for trend-based adjustments
async def trend_signal_callback(execution: TWAPExecution, order: TWAPOrder) -> Optional[Dict[str, Any]]:
    """Example signal callback that adjusts order size based on trend"""
    try:
        # Get recent price movement
        # This is a placeholder - implement actual trend analysis
        current_price = await execution.market_data.get_current_price(execution.config.symbol)
        
        # If price is trending up and we're buying, increase order size
        # If price is trending down and we're selling, increase order size
        trend_factor = 1.0  # Placeholder calculation
        
        if abs(trend_factor - 1.0) > 0.1:  # Significant trend
            adjusted_size = order.amount * Decimal(trend_factor)
            return {"size": adjusted_size}
        
        return None
        
    except Exception as e:
        logger.error(f"Trend signal callback failed: {e}")
        return None