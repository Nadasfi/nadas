"""
Live Trading Service - Real Execution Engine
Enables actual trading operations with risk management
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from decimal import Decimal

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from structlog import get_logger

from app.core.config import settings
from app.adapters.alchemy import get_alchemy_adapter
from app.services.websocket_manager import get_websocket_manager

logger = get_logger(__name__)


@dataclass
class TradeOrder:
    """Live trade order"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: str  # 'market' or 'limit'
    price: Optional[float] = None
    reduce_only: bool = False
    post_only: bool = False
    time_in_force: str = "GTC"  # Good Till Cancel
    client_order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[str] = None
    transaction_hash: Optional[str] = None
    message: str = ""
    filled_size: float = 0.0
    filled_price: float = 0.0
    fees_paid: float = 0.0
    timestamp: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskCheck:
    """Risk management check result"""
    approved: bool
    risk_score: float
    warnings: List[str]
    max_allowed_size: float
    estimated_liquidation_price: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveTradingService:
    """Real trading execution with risk management"""
    
    def __init__(self):
        self.use_mainnet = settings.HYPERLIQUID_NETWORK == "mainnet"
        self.private_key = settings.HYPERLIQUID_PRIVATE_KEY
        self.paper_trading = settings.HYPERLIQUID_ENABLE_PAPER_TRADING
        
        # Initialize if private key is available
        if self.private_key:
            base_url = constants.MAINNET_API_URL if self.use_mainnet else constants.TESTNET_API_URL
            self.info = Info(base_url)
            self.exchange = Exchange(self.private_key, base_url)
            logger.info("Live trading initialized", mainnet=self.use_mainnet, paper_trading=self.paper_trading)
        else:
            self.info = None
            self.exchange = None
            logger.warning("Live trading disabled - private key not configured")
        
        # Risk management settings
        self.max_position_size = settings.HYPERLIQUID_MAX_POSITION_SIZE
        self.default_slippage = settings.HYPERLIQUID_DEFAULT_SLIPPAGE
        self.risk_limits = {
            'max_daily_volume': 50000.0,  # USD
            'max_position_percent': 0.20,  # 20% of portfolio
            'min_liquidity_ratio': 0.05,   # 5% of orderbook
            'max_leverage': 10.0,
            'stop_loss_required_above': 10000.0  # USD position size
        }
        
        # Performance tracking
        self.daily_volume = 0.0
        self.last_reset_date = datetime.utcnow().date()
        self.active_orders: Dict[str, TradeOrder] = {}
        
    def _is_trading_enabled(self) -> bool:
        """Check if trading is enabled"""
        return self.exchange is not None or self.paper_trading
    
    async def perform_risk_check(self, order: TradeOrder, user_address: str) -> RiskCheck:
        """Comprehensive risk assessment before trade execution"""
        warnings = []
        risk_score = 0.0
        
        try:
            # 1. Check daily volume limits
            if self.daily_volume + (order.size * (order.price or 0)) > self.risk_limits['max_daily_volume']:
                warnings.append("Exceeds daily volume limit")
                risk_score += 30.0
            
            # 2. Get current portfolio
            if self.info:
                try:
                    user_state = self.info.user_state(user_address)
                    account_value = float(user_state.get('marginSummary', {}).get('accountValue', 0))
                    
                    # Position size check
                    position_value = order.size * (order.price or 1)
                    if position_value > account_value * self.risk_limits['max_position_percent']:
                        warnings.append("Position too large relative to portfolio")
                        risk_score += 25.0
                    
                    # Leverage check
                    total_ntl = float(user_state.get('marginSummary', {}).get('totalNtlPos', 0))
                    if total_ntl > 0 and account_value > 0:
                        leverage = total_ntl / account_value
                        if leverage > self.risk_limits['max_leverage']:
                            warnings.append("High leverage detected")
                            risk_score += 20.0
                
                except Exception as e:
                    logger.warning("Could not fetch user state for risk check", error=str(e))
                    warnings.append("Unable to verify current portfolio state")
                    risk_score += 15.0
            
            # 3. Market liquidity check (simplified)
            ws_manager = get_websocket_manager()
            orderbook = ws_manager.get_latest_orderbook(order.symbol)
            
            if orderbook:
                # Check if order size is reasonable relative to market depth
                total_liquidity = sum(level['sz'] for level in orderbook.bids[:5]) + sum(level['sz'] for level in orderbook.asks[:5])
                if order.size > total_liquidity * self.risk_limits['min_liquidity_ratio']:
                    warnings.append("Low market liquidity for trade size")
                    risk_score += 15.0
            else:
                warnings.append("No market data available")
                risk_score += 10.0
            
            # 4. Market hours / volatility check
            volatility = ws_manager.get_volatility_estimate(order.symbol)
            if volatility > 0.5:  # High volatility
                warnings.append("High market volatility detected")
                risk_score += 10.0
            
            # 5. Determine max allowed size  
            remaining_daily_limit = self.risk_limits['max_daily_volume'] - self.daily_volume
            max_allowed_size = min(
                order.size,
                self.max_position_size,
                max(0, remaining_daily_limit)  # Ensure non-negative
            )
            
            # Final approval decision
            approved = risk_score < 50.0 and len(warnings) < 3
            
            return RiskCheck(
                approved=approved,
                risk_score=risk_score,
                warnings=warnings,
                max_allowed_size=max_allowed_size
            )
            
        except Exception as e:
            logger.error("Risk check failed", order=order.to_dict(), error=str(e))
            return RiskCheck(
                approved=False,
                risk_score=100.0,
                warnings=[f"Risk check error: {str(e)}"],
                max_allowed_size=0.0
            )
    
    async def execute_trade(self, order: TradeOrder, user_address: str) -> TradeResult:
        """Execute live trade with risk management"""
        
        # Check if trading is enabled
        if not self._is_trading_enabled():
            return TradeResult(
                success=False,
                message="Live trading not configured",
                error="Private key required for live trading",
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Perform risk check
        risk_check = await self.perform_risk_check(order, user_address)
        
        if not risk_check.approved:
            return TradeResult(
                success=False,
                message="Trade rejected by risk management",
                error=f"Risk score: {risk_check.risk_score}, Warnings: {', '.join(risk_check.warnings)}",
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Adjust order size if needed
        if order.size > risk_check.max_allowed_size:
            order.size = risk_check.max_allowed_size
            logger.warning("Order size reduced by risk management", 
                          original_size=order.size, 
                          adjusted_size=risk_check.max_allowed_size)
        
        try:
            if self.paper_trading:
                # Paper trading simulation
                return await self._execute_paper_trade(order, risk_check)
            else:
                # Real trade execution
                return await self._execute_real_trade(order, user_address, risk_check)
                
        except Exception as e:
            logger.error("Trade execution failed", order=order.to_dict(), error=str(e))
            return TradeResult(
                success=False,
                message="Trade execution failed",
                error=str(e),
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _execute_paper_trade(self, order: TradeOrder, risk_check: RiskCheck) -> TradeResult:
        """Execute paper trade simulation"""
        
        # Get current market price
        ws_manager = get_websocket_manager()
        current_price = ws_manager.get_live_price(order.symbol)
        
        if not current_price:
            current_price = order.price or 100.0  # Fallback
        
        # Simulate execution
        filled_price = current_price
        
        # Add simulated slippage for market orders
        if order.order_type == "market":
            slippage_factor = 1 + (self.default_slippage if order.side == "buy" else -self.default_slippage)
            filled_price = current_price * slippage_factor
        
        # Simulate fees (0.05% maker, 0.055% taker)
        fee_rate = 0.0005 if order.order_type == "limit" and order.post_only else 0.00055
        fees_paid = order.size * filled_price * fee_rate
        
        # Generate mock order ID
        order_id = f"paper_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{order.symbol}_{order.side}"
        
        # Update daily volume tracking
        self.daily_volume += order.size * filled_price
        
        logger.info("Paper trade executed", 
                   order_id=order_id, 
                   symbol=order.symbol, 
                   side=order.side, 
                   size=order.size, 
                   price=filled_price)
        
        return TradeResult(
            success=True,
            order_id=order_id,
            message=f"Paper trade executed: {order.side} {order.size} {order.symbol} at ${filled_price:.4f}",
            filled_size=order.size,
            filled_price=filled_price,
            fees_paid=fees_paid,
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def _execute_real_trade(self, order: TradeOrder, user_address: str, risk_check: RiskCheck) -> TradeResult:
        """Execute real trade on Hyperliquid"""
        
        if not self.exchange:
            raise Exception("Exchange not initialized")
        
        try:
            # Convert order to Hyperliquid format
            is_buy = order.side.lower() == "buy"
            sz = order.size
            
            # Prepare order parameters
            order_params = {
                "a": user_address,  # User address
                "b": is_buy,        # Buy or sell
                "p": str(order.price) if order.price else "0",  # Price (0 for market)
                "s": str(sz),       # Size
                "r": order.reduce_only,
                "t": {"limit": {"tif": order.time_in_force}, "market": {}}[order.order_type]
            }
            
            # Execute order
            if order.order_type == "market":
                result = self.exchange.market_order(
                    coin=order.symbol,
                    is_buy=is_buy,
                    sz=sz,
                    px=None,  # Market price
                    reduce_only=order.reduce_only
                )
            else:  # Limit order
                result = self.exchange.order(
                    coin=order.symbol,
                    is_buy=is_buy,
                    sz=sz,
                    limit_px=order.price,
                    reduce_only=order.reduce_only,
                    post_only=order.post_only
                )
            
            # Parse result
            if result and 'status' in result:
                if result['status'] == 'ok':
                    # Extract order details
                    order_id = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('resting', {}).get('oid')
                    
                    # Update daily volume
                    if order.price:
                        self.daily_volume += order.size * order.price
                    
                    logger.info("Real trade executed", 
                               order_id=order_id, 
                               symbol=order.symbol, 
                               result=result)
                    
                    return TradeResult(
                        success=True,
                        order_id=str(order_id) if order_id else "unknown",
                        message=f"Live trade executed: {order.side} {order.size} {order.symbol}",
                        filled_size=order.size,
                        filled_price=order.price or 0.0,
                        timestamp=datetime.utcnow().isoformat()
                    )
                else:
                    error_msg = result.get('response', 'Unknown error')
                    return TradeResult(
                        success=False,
                        message="Trade rejected by exchange",
                        error=str(error_msg),
                        timestamp=datetime.utcnow().isoformat()
                    )
            else:
                return TradeResult(
                    success=False,
                    message="Invalid response from exchange",
                    error="No status in response",
                    timestamp=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            logger.error("Real trade execution failed", error=str(e))
            return TradeResult(
                success=False,
                message="Real trade execution failed",
                error=str(e),
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_open_orders(self, user_address: str) -> List[Dict[str, Any]]:
        """Get user's open orders"""
        if not self.info:
            return []
        
        try:
            open_orders = self.info.open_orders(user_address)
            return open_orders or []
        except Exception as e:
            logger.error("Failed to get open orders", user=user_address, error=str(e))
            return []
    
    async def cancel_order(self, order_id: str, user_address: str) -> bool:
        """Cancel an open order"""
        if not self.exchange:
            return False
        
        try:
            # For paper trading
            if self.paper_trading:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                    logger.info("Paper order cancelled", order_id=order_id)
                    return True
                return False
            
            # Real cancellation
            result = self.exchange.cancel(order_id, user_address)
            
            if result and result.get('status') == 'ok':
                logger.info("Order cancelled", order_id=order_id)
                return True
            else:
                logger.error("Failed to cancel order", order_id=order_id, result=result)
                return False
                
        except Exception as e:
            logger.error("Cancel order failed", order_id=order_id, error=str(e))
            return False
    
    async def get_trading_stats(self) -> Dict[str, Any]:
        """Get trading statistics and limits"""
        return {
            'daily_volume_usd': self.daily_volume,
            'daily_limit_usd': self.risk_limits['max_daily_volume'],
            'daily_usage_percent': (self.daily_volume / self.risk_limits['max_daily_volume']) * 100,
            'max_position_size': self.max_position_size,
            'paper_trading_enabled': self.paper_trading,
            'live_trading_enabled': self.exchange is not None,
            'active_orders_count': len(self.active_orders),
            'risk_limits': self.risk_limits,
            'last_reset_date': self.last_reset_date.isoformat(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def reset_daily_limits(self):
        """Reset daily trading limits (called by scheduler)"""
        today = datetime.utcnow().date()
        if today > self.last_reset_date:
            self.daily_volume = 0.0
            self.last_reset_date = today
            logger.info("Daily trading limits reset")


# Global instance
_trading_service: Optional[LiveTradingService] = None

def get_trading_service() -> LiveTradingService:
    """Get or create trading service instance"""
    global _trading_service
    if _trading_service is None:
        _trading_service = LiveTradingService()
    return _trading_service