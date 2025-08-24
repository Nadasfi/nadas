"""
Live Trading API Endpoints
Real trading execution with comprehensive risk management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from app.services.live_trading import get_trading_service, TradeOrder, TradeResult
from app.adapters.alchemy import get_alchemy_adapter
from app.api.v1.auth import get_current_user
from app.core.response import create_response
from structlog import get_logger

logger = get_logger(__name__)
router = APIRouter()


class LiveTradeRequest(BaseModel):
    """Request model for live trade execution"""
    symbol: str = Field(..., description="Trading symbol (e.g., ETH, BTC)")
    side: str = Field(..., description="Trade side: 'buy' or 'sell'")
    size: float = Field(..., gt=0, description="Trade size in USD")
    order_type: str = Field("market", description="Order type: 'market' or 'limit'")
    price: Optional[float] = Field(None, description="Price for limit orders")
    reduce_only: bool = Field(False, description="Reduce only flag")
    post_only: bool = Field(False, description="Post only flag for limit orders")
    time_in_force: str = Field("GTC", description="Time in force: GTC, IOC, FOK")
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "ETH",
                "side": "buy",
                "size": 1000.0,
                "order_type": "market",
                "price": None,
                "reduce_only": False,
                "post_only": False,
                "time_in_force": "GTC"
            }
        }


class RiskCheckRequest(BaseModel):
    """Request model for risk assessment"""
    symbol: str
    side: str
    size: float
    price: Optional[float] = None


class CancelOrderRequest(BaseModel):
    """Request model for order cancellation"""
    order_id: str = Field(..., description="Order ID to cancel")


@router.post("/execute-trade")
async def execute_live_trade(
    request: LiveTradeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute live trade with risk management"""
    
    try:
        user_address = current_user["wallet_address"]
        trading_service = get_trading_service()
        
        # Create trade order
        order = TradeOrder(
            symbol=request.symbol,
            side=request.side,
            size=request.size,
            order_type=request.order_type,
            price=request.price,
            reduce_only=request.reduce_only,
            post_only=request.post_only,
            time_in_force=request.time_in_force
        )
        
        # Execute trade
        result = await trading_service.execute_trade(order, user_address)
        
        if result.success:
            logger.info("Live trade executed successfully", 
                       user=user_address, 
                       order_id=result.order_id,
                       symbol=request.symbol,
                       side=request.side)
            
            return create_response(
                success=True,
                data=result.to_dict(),
                message="Trade executed successfully"
            )
        else:
            logger.warning("Trade execution failed", 
                          user=user_address, 
                          symbol=request.symbol,
                          error=result.error)
            
            return create_response(
                success=False,
                error=result.error or result.message,
                data=result.to_dict()
            )
            
    except Exception as e:
        user_address = current_user.get("wallet_address", "unknown")
        logger.error("Live trade execution error", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trade execution failed: {str(e)}"
        )


@router.post("/risk-check")
async def perform_risk_assessment(
    request: RiskCheckRequest,
    current_user: dict = Depends(get_current_user)
):
    """Perform risk assessment for potential trade"""
    
    try:
        trading_service = get_trading_service()
        
        # Create order for risk check
        order = TradeOrder(
            symbol=request.symbol,
            side=request.side,
            size=request.size,
            order_type="market",  # Default for risk check
            price=request.price
        )
        
        # Perform risk assessment
        risk_check = await trading_service.perform_risk_check(order, user_address)
        
        return create_response(
            success=True,
            data=risk_check.to_dict(),
            message="Risk assessment completed"
        )
        
    except Exception as e:
        logger.error("Risk check failed", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk assessment failed: {str(e)}"
        )


@router.get("/open-orders")
async def get_open_orders(
    current_user: dict = Depends(get_current_user)
):
    """Get user's open orders"""
    
    try:
        trading_service = get_trading_service()
        orders = await trading_service.get_open_orders(user_address)
        
        return create_response(
            success=True,
            data={
                "orders": orders,
                "count": len(orders),
                "timestamp": datetime.utcnow().isoformat()
            },
            message="Open orders retrieved successfully"
        )
        
    except Exception as e:
        logger.error("Failed to get open orders", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orders: {str(e)}"
        )


@router.post("/cancel-order")
async def cancel_order(
    request: CancelOrderRequest,
    current_user: dict = Depends(get_current_user)
):
    """Cancel an open order"""
    
    try:
        trading_service = get_trading_service()
        success = await trading_service.cancel_order(request.order_id, user_address)
        
        if success:
            logger.info("Order cancelled successfully", 
                       user=user_address, 
                       order_id=request.order_id)
            
            return create_response(
                success=True,
                data={
                    "order_id": request.order_id,
                    "status": "cancelled",
                    "timestamp": datetime.utcnow().isoformat()
                },
                message="Order cancelled successfully"
            )
        else:
            return create_response(
                success=False,
                error="Failed to cancel order",
                data={"order_id": request.order_id}
            )
            
    except Exception as e:
        logger.error("Order cancellation failed", 
                    user=user_address, 
                    order_id=request.order_id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Order cancellation failed: {str(e)}"
        )


@router.get("/trading-stats")
async def get_trading_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get trading statistics and limits"""
    
    try:
        trading_service = get_trading_service()
        stats = await trading_service.get_trading_stats()
        
        return create_response(
            success=True,
            data=stats,
            message="Trading statistics retrieved"
        )
        
    except Exception as e:
        logger.error("Failed to get trading stats", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/market-data/{symbol}")
async def get_enhanced_market_data(
    symbol: str,
    current_user: dict = Depends(get_current_user)
):
    """Get enhanced market data using Alchemy"""
    
    try:
        alchemy = get_alchemy_adapter()
        market_data = await alchemy.get_enhanced_market_data(symbol)
        
        return create_response(
            success=True,
            data=market_data,
            message="Enhanced market data retrieved"
        )
        
    except Exception as e:
        logger.error("Failed to get market data", symbol=symbol, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market data: {str(e)}"
        )


@router.get("/transaction-analysis/{tx_hash}")
async def analyze_transaction(
    tx_hash: str,
    current_user: dict = Depends(get_current_user)
):
    """Analyze transaction with Alchemy enhanced data"""
    
    try:
        alchemy = get_alchemy_adapter()
        analysis = await alchemy.get_transaction_analysis(tx_hash)
        
        return create_response(
            success=True,
            data=analysis,
            message="Transaction analysis completed"
        )
        
    except Exception as e:
        logger.error("Transaction analysis failed", tx_hash=tx_hash, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transaction analysis failed: {str(e)}"
        )


@router.get("/gas-tracker")
async def get_gas_information():
    """Get current gas price information via Alchemy"""
    
    try:
        alchemy = get_alchemy_adapter()
        gas_info = await alchemy.get_gas_price()
        
        return create_response(
            success=True,
            data=gas_info,
            message="Gas information retrieved"
        )
        
    except Exception as e:
        logger.error("Failed to get gas info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get gas information: {str(e)}"
        )


@router.post("/emergency-stop")
async def emergency_stop_trading(
    current_user: dict = Depends(get_current_user)
):
    """Emergency stop all trading activity"""
    
    try:
        trading_service = get_trading_service()
        
        # Get and cancel all open orders
        open_orders = await trading_service.get_open_orders(user_address)
        cancelled_orders = []
        
        for order in open_orders:
            order_id = order.get('oid')  # Order ID from Hyperliquid
            if order_id:
                success = await trading_service.cancel_order(str(order_id), user_address)
                if success:
                    cancelled_orders.append(order_id)
        
        logger.warning("Emergency stop executed", 
                      user=user_address, 
                      cancelled_orders=len(cancelled_orders))
        
        return create_response(
            success=True,
            data={
                "cancelled_orders": cancelled_orders,
                "total_cancelled": len(cancelled_orders),
                "timestamp": datetime.utcnow().isoformat()
            },
            message=f"Emergency stop executed - {len(cancelled_orders)} orders cancelled"
        )
        
    except Exception as e:
        logger.error("Emergency stop failed", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency stop failed: {str(e)}"
        )