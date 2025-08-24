"""
Simple Portfolio API - Basic endpoints for portfolio management
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from app.api.v1.auth import get_current_user
from app.services.portfolio_service_simple import get_portfolio_summary
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/overview")
async def portfolio_overview(current_user: dict = Depends(get_current_user)):
    """Get complete portfolio overview from Hyperliquid mainnet"""
    try:
        user_address = current_user["wallet_address"]
        
        # Get Hyperliquid adapter (use mainnet)
        adapter = get_hyperliquid_adapter(use_mainnet=True)
        
        # Fetch real portfolio data
        positions = await adapter.get_user_positions(user_address)
        balances = await adapter.get_spot_balances(user_address)
        portfolio_stats = await adapter.get_portfolio_summary(user_address)
        
        # Calculate totals
        total_equity = portfolio_stats.get("total_equity", 0.0)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        margin_ratio = portfolio_stats.get("margin_ratio", 0.0)
        buying_power = portfolio_stats.get("buying_power", 0.0)
        
        overview = {
            "success": True,
            "data": {
                "total_equity": total_equity,
                "unrealized_pnl": unrealized_pnl,
                "margin_ratio": margin_ratio,
                "buying_power": buying_power,
                "positions_count": len(positions),
                "spot_balances_count": len(balances),
                "positions": [
                    {
                        "symbol": pos.symbol,
                        "size": pos.size,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "mark_price": pos.mark_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "leverage": pos.leverage
                    }
                    for pos in positions
                ],
                "balances": [
                    {
                        "coin": balance.token,
                        "total": balance.total_balance,
                        "available": balance.available_balance,
                        "locked": balance.locked_balance
                    }
                    for balance in balances
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        logger.info("Portfolio overview retrieved from Hyperliquid", 
                   wallet_address=user_address,
                   positions=len(positions),
                   total_equity=total_equity)
        
        return overview
        
    except Exception as e:
        logger.error("Error getting portfolio overview from Hyperliquid", 
                    wallet_address=current_user.get("wallet_address"),
                    error=str(e))
        
        # Return mock data if Hyperliquid is unavailable (for development)
        return {
            "success": True,
            "data": {
                "total_equity": 2437892.0,
                "unrealized_pnl": 125000.0,
                "margin_ratio": 0.85,
                "buying_power": 567890.0,
                "positions_count": 0,
                "spot_balances_count": 0,
                "positions": [],
                "balances": [],
                "timestamp": datetime.utcnow().isoformat(),
                "mock_data": True,
                "error": str(e)
            }
        }


@router.get("/health")
async def portfolio_health():
    """Portfolio service health check"""
    return {
        "status": "healthy",
        "service": "portfolio",
        "timestamp": datetime.utcnow().isoformat()
    }