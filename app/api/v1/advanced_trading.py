"""
Advanced Trading Features API
Stop-loss, Take-profit, Conditional Orders - Phase 2 Implementation
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import asyncio
import uuid

from app.api.v1.auth import get_current_user
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class OrderConditionType(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    PRICE_ALERT = "price_alert"
    TIME_BASED = "time_based"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class StopLossRequest(BaseModel):
    """Stop-loss order request"""
    symbol: str = Field(..., description="Trading symbol")
    trigger_price: float = Field(..., gt=0, description="Price to trigger stop-loss")
    size: Optional[float] = Field(None, gt=0, description="Size to close (None for full position)")
    order_type: str = Field("market", description="Order type when triggered")
    reduce_only: bool = Field(True, description="Reduce only flag")
    expiry_time: Optional[datetime] = Field(None, description="Expiry time for the condition")


class TakeProfitRequest(BaseModel):
    """Take-profit order request"""
    symbol: str = Field(..., description="Trading symbol")
    trigger_price: float = Field(..., gt=0, description="Price to trigger take-profit")
    size: Optional[float] = Field(None, gt=0, description="Size to close (None for full position)")
    order_type: str = Field("limit", description="Order type when triggered")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price if using limit order")
    reduce_only: bool = Field(True, description="Reduce only flag")
    expiry_time: Optional[datetime] = Field(None, description="Expiry time for the condition")


class TrailingStopRequest(BaseModel):
    """Trailing stop order request"""
    symbol: str = Field(..., description="Trading symbol")
    trail_amount: float = Field(..., gt=0, description="Trail amount in USD")
    trail_percent: Optional[float] = Field(None, gt=0, le=100, description="Trail percentage")
    size: Optional[float] = Field(None, gt=0, description="Size to close")
    order_type: str = Field("market", description="Order type when triggered")


class ConditionalOrderResponse(BaseModel):
    """Conditional order response"""
    order_id: str
    condition_type: OrderConditionType
    symbol: str
    status: OrderStatus
    trigger_price: Optional[float]
    current_price: float
    distance_to_trigger: float
    created_at: datetime
    triggered_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    
class ConditionalOrdersList(BaseModel):
    """List of conditional orders"""
    orders: List[ConditionalOrderResponse]
    total_count: int
    active_count: int
    pending_count: int


# In-memory storage for demo (production would use database)
conditional_orders: Dict[str, Dict] = {}


@router.post("/stop-loss", response_model=ConditionalOrderResponse)
async def create_stop_loss(
    request: StopLossRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a stop-loss order"""
    try:
        order_id = str(uuid.uuid4())
        wallet_address = current_user["wallet_address"]
        
        # Get current market data for validation
        adapter = get_hyperliquid_adapter()
        market_data = await adapter.get_market_data(request.symbol)
        positions = await adapter.get_user_positions(wallet_address)
        await adapter.close()
        
        if not market_data:
            raise HTTPException(status_code=400, detail=f"No market data for {request.symbol}")
        
        current_price = market_data.mid_price
        
        # Find current position
        current_position = None
        for pos in positions:
            if pos.symbol == request.symbol:
                current_position = pos
                break
        
        if not current_position:
            raise HTTPException(status_code=400, detail=f"No position found for {request.symbol}")
        
        # Validate stop-loss logic
        if current_position.side == "long" and request.trigger_price >= current_price:
            raise HTTPException(status_code=400, detail="Stop-loss for long position must be below current price")
        elif current_position.side == "short" and request.trigger_price <= current_price:
            raise HTTPException(status_code=400, detail="Stop-loss for short position must be above current price")
        
        # Create conditional order
        order_data = {
            "order_id": order_id,
            "wallet_address": wallet_address,
            "condition_type": OrderConditionType.STOP_LOSS,
            "symbol": request.symbol,
            "trigger_price": request.trigger_price,
            "size": request.size or abs(current_position.size),
            "order_type": request.order_type,
            "reduce_only": request.reduce_only,
            "status": OrderStatus.ACTIVE,
            "created_at": datetime.utcnow(),
            "expiry_time": request.expiry_time,
            "position_side": current_position.side,
            "creation_price": current_price
        }
        
        conditional_orders[order_id] = order_data
        
        # Start background monitoring
        background_tasks.add_task(monitor_conditional_order, order_id)
        
        distance_to_trigger = abs(current_price - request.trigger_price)
        
        logger.info("Stop-loss order created", 
                   order_id=order_id,
                   symbol=request.symbol,
                   trigger_price=request.trigger_price,
                   current_price=current_price)
        
        return ConditionalOrderResponse(
            order_id=order_id,
            condition_type=OrderConditionType.STOP_LOSS,
            symbol=request.symbol,
            status=OrderStatus.ACTIVE,
            trigger_price=request.trigger_price,
            current_price=current_price,
            distance_to_trigger=distance_to_trigger,
            created_at=order_data["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating stop-loss order", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create stop-loss order")


@router.post("/take-profit", response_model=ConditionalOrderResponse)
async def create_take_profit(
    request: TakeProfitRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a take-profit order"""
    try:
        order_id = str(uuid.uuid4())
        wallet_address = current_user["wallet_address"]
        
        # Get current market data
        adapter = get_hyperliquid_adapter()
        market_data = await adapter.get_market_data(request.symbol)
        positions = await adapter.get_user_positions(wallet_address)
        await adapter.close()
        
        if not market_data:
            raise HTTPException(status_code=400, detail=f"No market data for {request.symbol}")
        
        current_price = market_data.mid_price
        
        # Find current position
        current_position = None
        for pos in positions:
            if pos.symbol == request.symbol:
                current_position = pos
                break
        
        if not current_position:
            raise HTTPException(status_code=400, detail=f"No position found for {request.symbol}")
        
        # Validate take-profit logic
        if current_position.side == "long" and request.trigger_price <= current_price:
            raise HTTPException(status_code=400, detail="Take-profit for long position must be above current price")
        elif current_position.side == "short" and request.trigger_price >= current_price:
            raise HTTPException(status_code=400, detail="Take-profit for short position must be below current price")
        
        # Create conditional order
        order_data = {
            "order_id": order_id,
            "wallet_address": wallet_address,
            "condition_type": OrderConditionType.TAKE_PROFIT,
            "symbol": request.symbol,
            "trigger_price": request.trigger_price,
            "size": request.size or abs(current_position.size),
            "order_type": request.order_type,
            "limit_price": request.limit_price,
            "reduce_only": request.reduce_only,
            "status": OrderStatus.ACTIVE,
            "created_at": datetime.utcnow(),
            "expiry_time": request.expiry_time,
            "position_side": current_position.side,
            "creation_price": current_price
        }
        
        conditional_orders[order_id] = order_data
        
        # Start background monitoring
        background_tasks.add_task(monitor_conditional_order, order_id)
        
        distance_to_trigger = abs(current_price - request.trigger_price)
        
        logger.info("Take-profit order created", 
                   order_id=order_id,
                   symbol=request.symbol,
                   trigger_price=request.trigger_price,
                   current_price=current_price)
        
        return ConditionalOrderResponse(
            order_id=order_id,
            condition_type=OrderConditionType.TAKE_PROFIT,
            symbol=request.symbol,
            status=OrderStatus.ACTIVE,
            trigger_price=request.trigger_price,
            current_price=current_price,
            distance_to_trigger=distance_to_trigger,
            created_at=order_data["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating take-profit order", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create take-profit order")


@router.post("/trailing-stop", response_model=ConditionalOrderResponse)
async def create_trailing_stop(
    request: TrailingStopRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a trailing stop order"""
    try:
        order_id = str(uuid.uuid4())
        wallet_address = current_user["wallet_address"]
        
        # Get current market data
        adapter = get_hyperliquid_adapter()
        market_data = await adapter.get_market_data(request.symbol)
        positions = await adapter.get_user_positions(wallet_address)
        await adapter.close()
        
        if not market_data:
            raise HTTPException(status_code=400, detail=f"No market data for {request.symbol}")
        
        current_price = market_data.mid_price
        
        # Find current position
        current_position = None
        for pos in positions:
            if pos.symbol == request.symbol:
                current_position = pos
                break
        
        if not current_position:
            raise HTTPException(status_code=400, detail=f"No position found for {request.symbol}")
        
        # Calculate initial trigger price
        if current_position.side == "long":
            trigger_price = current_price - request.trail_amount
        else:
            trigger_price = current_price + request.trail_amount
        
        # Create conditional order
        order_data = {
            "order_id": order_id,
            "wallet_address": wallet_address,
            "condition_type": OrderConditionType.TRAILING_STOP,
            "symbol": request.symbol,
            "trigger_price": trigger_price,
            "trail_amount": request.trail_amount,
            "trail_percent": request.trail_percent,
            "size": request.size or abs(current_position.size),
            "order_type": request.order_type,
            "status": OrderStatus.ACTIVE,
            "created_at": datetime.utcnow(),
            "position_side": current_position.side,
            "creation_price": current_price,
            "highest_price": current_price if current_position.side == "long" else None,
            "lowest_price": current_price if current_position.side == "short" else None
        }
        
        conditional_orders[order_id] = order_data
        
        # Start background monitoring
        background_tasks.add_task(monitor_conditional_order, order_id)
        
        distance_to_trigger = abs(current_price - trigger_price)
        
        logger.info("Trailing stop order created", 
                   order_id=order_id,
                   symbol=request.symbol,
                   trigger_price=trigger_price,
                   trail_amount=request.trail_amount)
        
        return ConditionalOrderResponse(
            order_id=order_id,
            condition_type=OrderConditionType.TRAILING_STOP,
            symbol=request.symbol,
            status=OrderStatus.ACTIVE,
            trigger_price=trigger_price,
            current_price=current_price,
            distance_to_trigger=distance_to_trigger,
            created_at=order_data["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating trailing stop order", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create trailing stop order")


@router.get("/conditional-orders", response_model=ConditionalOrdersList)
async def get_conditional_orders(current_user: dict = Depends(get_current_user)):
    """Get all conditional orders for user"""
    try:
        wallet_address = current_user["wallet_address"]
        user_orders = []
        
        for order_data in conditional_orders.values():
            if order_data["wallet_address"] == wallet_address:
                # Get current market price for distance calculation
                try:
                    adapter = get_hyperliquid_adapter()
                    market_data = await adapter.get_market_data(order_data["symbol"])
                    await adapter.close()
                    
                    current_price = market_data.mid_price if market_data else 0.0
                    distance_to_trigger = abs(current_price - order_data["trigger_price"])
                    
                except Exception:
                    current_price = 0.0
                    distance_to_trigger = 0.0
                
                user_orders.append(ConditionalOrderResponse(
                    order_id=order_data["order_id"],
                    condition_type=order_data["condition_type"],
                    symbol=order_data["symbol"],
                    status=order_data["status"],
                    trigger_price=order_data["trigger_price"],
                    current_price=current_price,
                    distance_to_trigger=distance_to_trigger,
                    created_at=order_data["created_at"],
                    triggered_at=order_data.get("triggered_at"),
                    executed_at=order_data.get("executed_at"),
                    execution_result=order_data.get("execution_result")
                ))
        
        # Sort by creation time
        user_orders.sort(key=lambda x: x.created_at, reverse=True)
        
        active_count = sum(1 for order in user_orders if order.status == OrderStatus.ACTIVE)
        pending_count = sum(1 for order in user_orders if order.status == OrderStatus.PENDING)
        
        return ConditionalOrdersList(
            orders=user_orders,
            total_count=len(user_orders),
            active_count=active_count,
            pending_count=pending_count
        )
        
    except Exception as e:
        logger.error("Error fetching conditional orders", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch conditional orders")


@router.delete("/conditional-orders/{order_id}")
async def cancel_conditional_order(order_id: str, current_user: dict = Depends(get_current_user)):
    """Cancel a conditional order"""
    try:
        if order_id not in conditional_orders:
            raise HTTPException(status_code=404, detail="Order not found")
        
        order_data = conditional_orders[order_id]
        
        # Verify ownership
        if order_data["wallet_address"] != current_user["wallet_address"]:
            raise HTTPException(status_code=403, detail="Not authorized to cancel this order")
        
        # Cancel only if not already executed
        if order_data["status"] in [OrderStatus.EXECUTED, OrderStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail=f"Cannot cancel order with status: {order_data['status']}")
        
        order_data["status"] = OrderStatus.CANCELLED
        order_data["cancelled_at"] = datetime.utcnow()
        
        logger.info("Conditional order cancelled", order_id=order_id)
        
        return {"message": "Order cancelled successfully", "order_id": order_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error cancelling conditional order", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cancel order")


# Background monitoring function
async def monitor_conditional_order(order_id: str):
    """Monitor conditional order and execute when triggered"""
    try:
        while order_id in conditional_orders:
            order_data = conditional_orders[order_id]
            
            # Skip if order is not active
            if order_data["status"] != OrderStatus.ACTIVE:
                break
            
            # Check expiry
            if order_data.get("expiry_time") and datetime.utcnow() > order_data["expiry_time"]:
                order_data["status"] = OrderStatus.EXPIRED
                logger.info("Conditional order expired", order_id=order_id)
                break
            
            try:
                # Get current market data
                adapter = get_hyperliquid_adapter()
                market_data = await adapter.get_market_data(order_data["symbol"])
                
                if not market_data:
                    await adapter.close()
                    await asyncio.sleep(5)  # Wait before retry
                    continue
                
                current_price = market_data.mid_price
                triggered = False
                
                # Check trigger conditions
                if order_data["condition_type"] == OrderConditionType.STOP_LOSS:
                    if order_data["position_side"] == "long":
                        triggered = current_price <= order_data["trigger_price"]
                    else:
                        triggered = current_price >= order_data["trigger_price"]
                        
                elif order_data["condition_type"] == OrderConditionType.TAKE_PROFIT:
                    if order_data["position_side"] == "long":
                        triggered = current_price >= order_data["trigger_price"]
                    else:
                        triggered = current_price <= order_data["trigger_price"]
                        
                elif order_data["condition_type"] == OrderConditionType.TRAILING_STOP:
                    # Update trailing stop logic
                    if order_data["position_side"] == "long":
                        if current_price > order_data.get("highest_price", 0):
                            order_data["highest_price"] = current_price
                            new_trigger = current_price - order_data["trail_amount"]
                            if new_trigger > order_data["trigger_price"]:
                                order_data["trigger_price"] = new_trigger
                        triggered = current_price <= order_data["trigger_price"]
                    else:
                        if current_price < order_data.get("lowest_price", float('inf')):
                            order_data["lowest_price"] = current_price
                            new_trigger = current_price + order_data["trail_amount"]
                            if new_trigger < order_data["trigger_price"]:
                                order_data["trigger_price"] = new_trigger
                        triggered = current_price >= order_data["trigger_price"]
                
                if triggered:
                    logger.info("Conditional order triggered", 
                               order_id=order_id,
                               trigger_price=order_data["trigger_price"],
                               current_price=current_price)
                    
                    order_data["status"] = OrderStatus.TRIGGERED
                    order_data["triggered_at"] = datetime.utcnow()
                    
                    # Execute the order (simplified for demo)
                    # In production, this would place actual order via SDK
                    execution_result = {
                        "success": True,
                        "execution_price": current_price,
                        "executed_size": order_data["size"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    order_data["status"] = OrderStatus.EXECUTED
                    order_data["executed_at"] = datetime.utcnow()
                    order_data["execution_result"] = execution_result
                    
                    logger.info("Conditional order executed", 
                               order_id=order_id,
                               execution_result=execution_result)
                    
                    break
                
                await adapter.close()
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error("Error monitoring conditional order", 
                           order_id=order_id, error=str(e))
                await asyncio.sleep(5)  # Wait longer on error
                
    except Exception as e:
        logger.error("Fatal error in conditional order monitoring", 
                   order_id=order_id, error=str(e))
