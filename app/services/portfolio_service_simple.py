"""
Simple Portfolio Service - Basic Implementation
Temporary simplified version for backend completion
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


class SimplePortfolioService:
    """Simplified portfolio service"""
    
    def __init__(self):
        self.hyperliquid_adapter = None
    
    async def get_portfolio_summary(self, wallet_address: str) -> Dict[str, Any]:
        """Get basic portfolio summary"""
        try:
            self.hyperliquid_adapter = get_hyperliquid_adapter()
            
            # Get account value
            account_value = await self.hyperliquid_adapter.get_account_value(wallet_address)
            
            # Get positions
            positions = await self.hyperliquid_adapter.get_user_positions(wallet_address)
            
            # Calculate basic metrics
            total_positions = len([p for p in positions if p.size != 0])
            total_unrealized_pnl = sum(p.unrealized_pnl for p in positions if hasattr(p, 'unrealized_pnl'))
            
            await self.hyperliquid_adapter.close()
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "total_equity": account_value.get("total_equity", 0.0),
                "available_margin": account_value.get("available_margin", 0.0),
                "margin_used": account_value.get("margin_used", 0.0),
                "unrealized_pnl": total_unrealized_pnl,
                "total_positions": total_positions,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting portfolio summary", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "wallet_address": wallet_address,
                "timestamp": datetime.utcnow().isoformat()
            }


# Global instance
_portfolio_service = None


def get_simple_portfolio_service() -> SimplePortfolioService:
    """Get portfolio service instance"""
    global _portfolio_service
    if _portfolio_service is None:
        _portfolio_service = SimplePortfolioService()
    return _portfolio_service


async def get_portfolio_summary(wallet_address: str) -> Dict[str, Any]:
    """Get portfolio summary for wallet"""
    service = get_simple_portfolio_service()
    return await service.get_portfolio_summary(wallet_address)