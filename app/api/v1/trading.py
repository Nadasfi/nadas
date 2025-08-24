"""
Trading API endpoints - Client-Side Wallet Integration
Non-custodial approach: Backend prepares data, frontend signs and executes
"""

from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from app.api.v1.auth import get_current_user
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/demo/place-order")
async def demo_place_order(
    order_request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Demo order placement for hackathon demonstration"""
    try:
        user_address = current_user["wallet_address"]
        
        # Create demo order response (for hackathon)
        demo_order = {
            "order_id": f"demo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "symbol": order_request.get("symbol", "ETH"),
            "side": order_request.get("side", "buy"),
            "size": order_request.get("size", 1.0),
            "price": order_request.get("price", 3500.0),
            "order_type": order_request.get("order_type", "limit"),
            "status": "pending_signature",
            "timestamp": datetime.utcnow().isoformat(),
            "user_address": user_address,
            "demo_mode": True,
            "estimated_gas": "0.002 ETH",
            "network": "hyperliquid_mainnet",
            "message": "This is a demo order for hackathon demonstration. In production, this would require wallet signature."
        }
        
        logger.info("Demo order created for hackathon", 
                   user=user_address,
                   symbol=order_request.get("symbol", "ETH"),
                   order_id=demo_order["order_id"])
        
        return {
            "success": True,
            "data": demo_order,
            "message": "Demo order prepared successfully"
        }
        
    except Exception as e:
        logger.error("Demo order creation failed", 
                    user=current_user.get("wallet_address"),
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Demo order failed: {str(e)}")


@router.get("/demo/market-data/{symbol}")
async def get_demo_market_data(
    symbol: str,
    current_user: dict = Depends(get_current_user)
):
    """Get market data for demo trading"""
    try:
        adapter = get_hyperliquid_adapter(use_mainnet=True)
        
        # Get real market data from Hyperliquid mainnet
        market_data = await adapter.get_market_data(symbol)
        
        if market_data:
            response = {
                "success": True,
                "data": {
                    "symbol": market_data.symbol,
                    "mid_price": market_data.mid_price,
                    "bid_price": market_data.bid_price,
                    "ask_price": market_data.ask_price,
                    "volume_24h": market_data.volume_24h,
                    "price_change_24h": market_data.price_change_24h,
                    "last_updated": market_data.last_updated.isoformat(),
                    "network": "hyperliquid_mainnet"
                },
                "message": "Live market data from Hyperliquid mainnet"
            }
        else:
            # Fallback demo data
            response = {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "mid_price": 3500.0 if symbol == "ETH" else 50000.0,
                    "bid_price": 3495.0 if symbol == "ETH" else 49950.0,
                    "ask_price": 3505.0 if symbol == "ETH" else 50050.0,
                    "volume_24h": 1000000.0,
                    "price_change_24h": 2.5,
                    "last_updated": datetime.utcnow().isoformat(),
                    "demo_data": True
                },
                "message": "Demo market data (API unavailable)"
            }
        
        return response
        
    except Exception as e:
        logger.error("Error getting market data", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Market data failed: {str(e)}")


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class PlaceOrderRequest(BaseModel):
    """Prepare order request for client-side signing"""
    symbol: str = Field(..., description="Trading symbol (e.g., ETH, BTC)")
    side: OrderSide = Field(..., description="Order side: buy or sell")
    size: float = Field(..., gt=0, description="Order size")
    price: Optional[float] = Field(None, gt=0, description="Limit price (required for limit orders)")
    order_type: OrderType = Field(OrderType.LIMIT, description="Order type")
    reduce_only: bool = Field(False, description="Reduce only order")
    post_only: bool = Field(False, description="Post only order (limit orders)")


class ClosePositionRequest(BaseModel):
    """Close position request"""
    symbol: str = Field(..., description="Symbol to close")
    size: Optional[float] = Field(None, gt=0, description="Size to close (None for full position)")
    user_address: str = Field(..., description="User wallet address")


class UpdateLeverageRequest(BaseModel):
    """Update leverage request"""
    symbol: str = Field(..., description="Symbol to update leverage for")
    leverage: int = Field(..., ge=1, le=100, description="New leverage (1-100)")


class TransferRequest(BaseModel):
    """Transfer between spot and perp request"""
    amount: float = Field(..., gt=0, description="USD amount to transfer")
    to_perp: bool = Field(..., description="True to transfer to perp, False to spot")


@router.post("/order/prepare")
async def prepare_order(
    order_request: PlaceOrderRequest,
    current_user: dict = Depends(get_current_user)
):
    """Prepare order data for client-side wallet signing (non-custodial)"""
    try:
        # Get read-only adapter (no private key needed)
        adapter = get_hyperliquid_adapter()
        
        # Validate order size limits
        max_position_size = getattr(settings, 'HYPERLIQUID_MAX_POSITION_SIZE', 1000.0)
        notional_value = order_request.size * (order_request.price or 1)
        if notional_value > max_position_size:
            raise HTTPException(
                status_code=400,
                detail=f"Order size ${notional_value:.2f} exceeds maximum ${max_position_size}"
            )
        
        # Prepare order data for client-side execution
        is_buy = order_request.side == OrderSide.BUY
        
        if order_request.order_type == OrderType.MARKET:
            result = await adapter.prepare_market_order_data(
                order_request.symbol,
                is_buy,
                order_request.size
            )
        else:
            if not order_request.price:
                raise HTTPException(
                    status_code=400,
                    detail="Price is required for limit orders"
                )
            
            result = await adapter.prepare_order_data(
                order_request.symbol,
                is_buy,
                order_request.size,
                order_request.price,
                order_request.order_type.value,
                order_request.reduce_only,
                order_request.post_only
            )
        
        await adapter.close()
        
        if result.get('success'):
            logger.info("Order prepared for client-side signing", 
                       wallet_address=current_user["wallet_address"],
                       symbol=order_request.symbol,
                       side=order_request.side,
                       size=order_request.size)
            
            return {
                "success": True,
                "order_data": result.get('order_data'),
                "message": "Order prepared for wallet signing",
                "timestamp": datetime.utcnow().isoformat(),
                "requires_wallet_signature": True
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Order preparation failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error preparing order", 
                    wallet_address=current_user["wallet_address"],
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Order preparation failed: {str(e)}")


@router.post("/position/close/prepare")
async def prepare_close_position(
    close_request: ClosePositionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Prepare position closing data for client-side signing"""
    try:
        adapter = get_hyperliquid_adapter()
        
        result = await adapter.prepare_close_position_data(
            symbol=close_request.symbol,
            user_address=close_request.user_address,
            size=close_request.size
        )
        
        await adapter.close()
        
        if result.get('success'):
            logger.info("Position close prepared for client-side signing", 
                       wallet_address=current_user["wallet_address"],
                       symbol=close_request.symbol)
            
            return {
                "success": True,
                "order_data": result.get('order_data'),
                "message": "Position close prepared for wallet signing",
                "timestamp": datetime.utcnow().isoformat(),
                "requires_wallet_signature": True
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Position close preparation failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error preparing position close", 
                    wallet_address=current_user["wallet_address"],
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Position close preparation failed: {str(e)}")


@router.post("/leverage/prepare")
async def prepare_leverage_update(
    leverage_request: UpdateLeverageRequest,
    current_user: dict = Depends(get_current_user)
):
    """Prepare leverage update for client-side signing"""
    try:
        adapter = get_hyperliquid_adapter()
        
        result = await adapter.prepare_leverage_update_data(
            symbol=leverage_request.symbol,
            leverage=leverage_request.leverage
        )
        
        await adapter.close()
        
        if result.get('success'):
            logger.info("Leverage update prepared", 
                       wallet_address=current_user["wallet_address"],
                       symbol=leverage_request.symbol,
                       leverage=leverage_request.leverage)
            
            return {
                "success": True,
                "leverage_data": result.get('leverage_data'),
                "message": f"Leverage update to {leverage_request.leverage}x prepared for wallet signing",
                "timestamp": datetime.utcnow().isoformat(),
                "requires_wallet_signature": True
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Leverage update preparation failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error preparing leverage update", 
                    wallet_address=current_user["wallet_address"],
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Leverage update preparation failed: {str(e)}")


@router.post("/transfer/prepare")
async def prepare_transfer(
    transfer_request: TransferRequest,
    current_user: dict = Depends(get_current_user)
):
    """Prepare funds transfer for client-side signing"""
    try:
        adapter = get_hyperliquid_adapter()
        
        result = await adapter.prepare_transfer_data(
            usd_amount=transfer_request.amount,
            to_perp=transfer_request.to_perp
        )
        
        await adapter.close()
        
        if result.get('success'):
            direction = "to perpetual" if transfer_request.to_perp else "to spot"
            
            logger.info("Transfer prepared", 
                       wallet_address=current_user["wallet_address"],
                       amount=transfer_request.amount,
                       direction=direction)
            
            return {
                "success": True,
                "transfer_data": result.get('transfer_data'),
                "message": f"${transfer_request.amount} transfer {direction} prepared for wallet signing",
                "timestamp": datetime.utcnow().isoformat(),
                "requires_wallet_signature": True
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Transfer preparation failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error preparing transfer", 
                    wallet_address=current_user["wallet_address"],
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Transfer preparation failed: {str(e)}")


# READ-ONLY endpoints (these don't require client-side signing)

@router.get("/open-orders")
async def get_open_orders(
    user_address: str,
    current_user: dict = Depends(get_current_user)
):
    """Get open orders for a user address"""
    try:
        adapter = get_hyperliquid_adapter()
        
        orders = await adapter.get_open_orders(user_address)
        
        await adapter.close()
        
        logger.info("Open orders fetched", 
                   wallet_address=user_address,
                   count=len(orders))
        
        return {
            "success": True,
            "orders": orders,
            "count": len(orders),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error fetching open orders", 
                    wallet_address=user_address,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch open orders")


@router.get("/trade-history")
async def get_trade_history(
    user_address: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get recent trade history for a user"""
    try:
        adapter = get_hyperliquid_adapter()
        
        fills = await adapter.get_user_fills(user_address)
        
        await adapter.close()
        
        # Limit results
        limited_fills = fills[:limit] if fills else []
        
        logger.info("Trade history fetched", 
                   wallet_address=user_address,
                   count=len(limited_fills))
        
        return {
            "success": True,
            "fills": limited_fills,
            "count": len(limited_fills),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error fetching trade history", 
                    wallet_address=user_address,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch trade history")


@router.get("/market-prices")
async def get_market_prices():
    """Get current market prices for all symbols (public data)"""
    try:
        adapter = get_hyperliquid_adapter()
        
        prices = await adapter.get_all_mid_prices()
        
        await adapter.close()
        
        logger.info("Market prices fetched", count=len(prices))
        
        return {
            "success": True,
            "prices": prices,
            "count": len(prices),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error fetching market prices", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch market prices")


@router.get("/market-data/{symbol}")
async def get_symbol_market_data(symbol: str):
    """Get detailed market data for a specific symbol (public data)"""
    try:
        adapter = get_hyperliquid_adapter()
        
        market_data = await adapter.get_market_data(symbol.upper())
        
        await adapter.close()
        
        if not market_data:
            raise HTTPException(status_code=404, detail=f"Market data not found for {symbol}")
        
        logger.info("Market data fetched", symbol=symbol)
        
        return {
            "success": True,
            "symbol": symbol.upper(),
            "data": {
                "mid_price": market_data.mid_price,
                "bid_price": market_data.bid_price,
                "ask_price": market_data.ask_price,
                "volume_24h": market_data.volume_24h,
                "price_change_24h": market_data.price_change_24h,
                "timestamp": market_data.last_updated.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching market data", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch market data")


@router.get("/health")
async def trading_health_check():
    """Health check for trading services"""
    try:
        adapter = get_hyperliquid_adapter()
        
        # Test basic connectivity
        prices = await adapter.get_all_mid_prices()
        connectivity_ok = len(prices) > 0
        
        await adapter.close()
        
        return {
            "success": True,
            "trading_mode": "client_side_wallet",
            "api_connectivity": connectivity_ok,
            "network": "testnet" if not adapter.use_mainnet else "mainnet",
            "prices_available": len(prices),
            "timestamp": datetime.utcnow().isoformat(),
            "security_model": "non_custodial"
        }
        
    except Exception as e:
        logger.error("Trading health check failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }