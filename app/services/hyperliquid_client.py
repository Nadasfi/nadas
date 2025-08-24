"""
HyperliquidClient Service - Real Trading Client
Production-ready service for actual Hyperliquid trading operations
Wraps hyperliquid-python-sdk for consistent API
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import eth_account
from eth_account.signers.local import LocalAccount

from app.adapters.hyperliquid import HyperliquidAdapter, get_hyperliquid_adapter
from app.core.config import settings
from app.core.error_handling import circuit_breaker_protected, track_errors, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP_MARKET = "Stop Market"
    STOP_LIMIT = "Stop Limit"
    TAKE_PROFIT = "Take Profit"

class OrderSide(Enum):
    BUY = "A"  # Hyperliquid uses "A" for buy
    SELL = "B"  # Hyperliquid uses "B" for sell

@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    filled_size: Optional[float] = None
    average_price: Optional[float] = None
    total_fee: Optional[float] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    size: float
    side: str
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: float
    liquidation_price: float
    margin_used: float

class HyperliquidClient:
    """
    Production-ready Hyperliquid trading client
    Provides real trading capabilities for TWAP executor and other services
    """
    
    def __init__(self, private_key: Optional[str] = None, use_testnet: bool = True):
        """
        Initialize Hyperliquid client for real trading
        
        Args:
            private_key: Wallet private key for trading (optional for read-only)
            use_testnet: Use testnet or mainnet
        """
        self.use_testnet = use_testnet
        self.base_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
        
        # Initialize Info API (always available)
        self.info = Info(self.base_url, skip_ws=False)
        
        # Initialize Exchange API only if private key provided
        self.exchange = None
        self.account = None
        self.wallet_address = None
        
        if private_key:
            self._initialize_trading(private_key)
        
        # Initialize adapter for read operations with lazy loading
        self.adapter = None
        self._use_mainnet = not use_testnet
        self._adapter_initialized = False
        
        logger.info(f"HyperliquidClient initialized - trading_enabled: {self.exchange is not None}, network: {'testnet' if use_testnet else 'mainnet'}")
    
    def _ensure_adapter(self):
        """Lazy initialization of HyperliquidAdapter"""
        if not self._adapter_initialized:
            try:
                self.adapter = HyperliquidAdapter(use_mainnet=self._use_mainnet)
                self._adapter_initialized = True
                logger.info("HyperliquidAdapter initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize HyperliquidAdapter", error=str(e))
                # Don't re-raise here, let the calling method handle it
        return self.adapter
    
    def _initialize_trading(self, private_key: str):
        """Initialize trading capabilities"""
        try:
            # Create account from private key
            self.account: LocalAccount = eth_account.Account.from_key(private_key)
            self.wallet_address = self.account.address
            
            # Initialize Exchange with account
            self.exchange = Exchange(self.account, self.base_url, account_address=self.wallet_address)
            
            logger.info(f"Trading initialized - wallet_address: {self.wallet_address}")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading: {str(e)}")
            raise
    
    @circuit_breaker_protected("hyperliquid_orders", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60))
    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "Limit",
        reduce_only: bool = False,
        post_only: bool = False,
        time_in_force: str = "Gtc"
    ) -> OrderResult:
        """
        Place an order on Hyperliquid
        
        Args:
            symbol: Trading pair (e.g., "ETH")
            side: "buy" or "sell"
            amount: Order size
            price: Limit price (None for market orders)
            order_type: Order type (Market, Limit, etc.)
            reduce_only: Reduce only flag
            post_only: Post only flag
            time_in_force: Time in force (Gtc, Ioc, Alo)
        """
        async with track_errors("hyperliquid_client", {"method": "place_order", "symbol": symbol}):
            try:
                if not self.exchange:
                    return OrderResult(
                        success=False,
                        error="Trading not enabled - no private key provided"
                    )
                
                # Convert side to Hyperliquid format
                is_buy = side.lower() == "buy"
                hl_side = "A" if is_buy else "B"
                
                # Prepare order data
                order_data = {
                    "a": self._get_asset_id(symbol),  # Asset ID
                    "b": hl_side,  # Side (A=buy, B=sell)
                    "p": str(price) if price else "0",  # Price (0 for market)
                    "s": str(amount),  # Size
                    "r": reduce_only,  # Reduce only
                    "t": {"limit": {"tif": time_in_force}} if order_type == "Limit" else {"market": {}}
                }
                
                # Add post-only for limit orders
                if order_type == "Limit" and post_only:
                    order_data["t"]["limit"]["alo"] = True
                
                # Place order
                result = self.exchange.order(order_data, signature=None)
                
                if result and "status" in result:
                    if result["status"] == "ok":
                        return OrderResult(
                            success=True,
                            order_id=result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("resting", {}).get("oid"),
                            status="submitted",
                            timestamp=datetime.now()
                        )
                    else:
                        return OrderResult(
                            success=False,
                            error=result.get("response", "Order failed")
                        )
                
                return OrderResult(success=False, error="Unknown response format")
                
            except Exception as e:
                logger.error("Order placement failed", symbol=symbol, side=side, amount=amount, error=str(e))
                return OrderResult(success=False, error=str(e))
    
    async def place_market_order(self, symbol: str, side: str, amount: float, reduce_only: bool = False) -> OrderResult:
        """Place a market order"""
        return await self.place_order(symbol, side, amount, None, "Market", reduce_only)
    
    async def place_limit_order(self, symbol: str, side: str, amount: float, price: float, 
                              post_only: bool = False, reduce_only: bool = False) -> OrderResult:
        """Place a limit order"""
        return await self.place_order(symbol, side, amount, price, "Limit", reduce_only, post_only)
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            if not self.exchange:
                logger.error("Trading not enabled")
                return False
            
            cancel_data = {
                "a": self._get_asset_id(symbol),
                "o": order_id
            }
            
            result = self.exchange.cancel(cancel_data)
            
            if result and result.get("status") == "ok":
                logger.info("Order cancelled", order_id=order_id, symbol=symbol)
                return True
            
            logger.error("Order cancellation failed", order_id=order_id, result=result)
            return False
            
        except Exception as e:
            logger.error("Order cancellation error", order_id=order_id, error=str(e))
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders for a symbol or all symbols"""
        try:
            if not self.exchange:
                return 0
            
            if symbol:
                asset_id = self._get_asset_id(symbol)
                result = self.exchange.cancel_all(asset_id)
            else:
                result = self.exchange.cancel_all()
            
            if result and result.get("status") == "ok":
                cancelled_count = len(result.get("response", {}).get("data", {}).get("statuses", []))
                logger.info("Orders cancelled", count=cancelled_count, symbol=symbol)
                return cancelled_count
            
            return 0
            
        except Exception as e:
            logger.error("Cancel all orders failed", symbol=symbol, error=str(e))
            return 0
    
    async def modify_order(self, symbol: str, order_id: str, new_price: float, new_size: float) -> OrderResult:
        """Modify an existing order"""
        try:
            if not self.exchange:
                return OrderResult(success=False, error="Trading not enabled")
            
            modify_data = {
                "oid": order_id,
                "order": {
                    "a": self._get_asset_id(symbol),
                    "b": "A",  # Will be determined by existing order
                    "p": str(new_price),
                    "s": str(new_size),
                    "r": False,
                    "t": {"limit": {"tif": "Gtc"}}
                }
            }
            
            result = self.exchange.modify(modify_data)
            
            if result and result.get("status") == "ok":
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status="modified",
                    timestamp=datetime.now()
                )
            
            return OrderResult(success=False, error=result.get("response", "Modify failed"))
            
        except Exception as e:
            logger.error("Order modification failed", order_id=order_id, error=str(e))
            return OrderResult(success=False, error=str(e))
    
    async def get_positions(self, address: Optional[str] = None) -> List[PositionInfo]:
        """Get current positions"""
        try:
            wallet_addr = address or self.wallet_address
            if not wallet_addr:
                return []
            
            positions = await self._ensure_adapter().get_user_positions(wallet_addr)
            
            position_info = []
            for pos in positions:
                position_info.append(PositionInfo(
                    symbol=pos.symbol,
                    size=pos.size,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    mark_price=pos.mark_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    leverage=pos.leverage,
                    liquidation_price=pos.liquidation_price,
                    margin_used=0  # Calculate if needed
                ))
            
            return position_info
            
        except Exception as e:
            logger.error("Failed to get positions", error=str(e))
            return []
    
    async def get_position(self, symbol: str, address: Optional[str] = None) -> Optional[PositionInfo]:
        """Get position for specific symbol"""
        positions = await self.get_positions(address)
        return next((p for p in positions if p.symbol == symbol), None)
    
    async def close_position(self, symbol: str, size: Optional[float] = None) -> OrderResult:
        """Close a position (market order)"""
        try:
            position = await self.get_position(symbol)
            if not position or position.size == 0:
                return OrderResult(success=False, error=f"No position to close for {symbol}")
            
            close_size = size if size else abs(position.size)
            close_side = "buy" if position.size < 0 else "sell"
            
            return await self.place_market_order(symbol, close_side, close_size, reduce_only=True)
            
        except Exception as e:
            logger.error("Position closing failed", symbol=symbol, error=str(e))
            return OrderResult(success=False, error=str(e))
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            if not self.exchange:
                return False
            
            leverage_data = {
                "asset": self._get_asset_id(symbol),
                "isCross": True,
                "leverage": leverage
            }
            
            result = self.exchange.update_leverage(leverage_data)
            
            if result and result.get("status") == "ok":
                logger.info("Leverage updated", symbol=symbol, leverage=leverage)
                return True
            
            logger.error("Leverage update failed", symbol=symbol, result=result)
            return False
            
        except Exception as e:
            logger.error("Leverage update error", symbol=symbol, leverage=leverage, error=str(e))
            return False
    
    async def transfer_between_spot_and_perp(self, usd_amount: float, to_perp: bool) -> bool:
        """Transfer USDC between spot and perp accounts"""
        try:
            if not self.exchange:
                return False
            
            transfer_data = {
                "type": "spot2perp" if to_perp else "perp2spot",
                "usdc": usd_amount
            }
            
            result = self.exchange.usd_transfer(transfer_data)
            
            if result and result.get("status") == "ok":
                direction = "spot->perp" if to_perp else "perp->spot"
                logger.info("Transfer completed", amount=usd_amount, direction=direction)
                return True
            
            logger.error("Transfer failed", amount=usd_amount, to_perp=to_perp, result=result)
            return False
            
        except Exception as e:
            logger.error("Transfer error", amount=usd_amount, to_perp=to_perp, error=str(e))
            return False
    
    async def get_open_orders(self, address: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        try:
            wallet_addr = address or self.wallet_address
            if not wallet_addr:
                return []
            
            orders = await self._ensure_adapter().get_open_orders(wallet_addr)
            return orders
            
        except Exception as e:
            logger.error("Failed to get open orders", error=str(e))
            return []
    
    async def get_trade_history(self, address: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            wallet_addr = address or self.wallet_address
            if not wallet_addr:
                return []
            
            fills = await self._ensure_adapter().get_user_fills(wallet_addr)
            return fills[:limit]
            
        except Exception as e:
            logger.error("Failed to get trade history", error=str(e))
            return []
    
    async def get_account_summary(self, address: Optional[str] = None) -> Dict[str, Any]:
        """Get account summary"""
        try:
            wallet_addr = address or self.wallet_address
            if not wallet_addr:
                return {}
            
            return await self._ensure_adapter().get_account_summary(wallet_addr)
            
        except Exception as e:
            logger.error("Failed to get account summary", error=str(e))
            return {}
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            prices = await self._ensure_adapter().get_all_mid_prices()
            return prices.get(symbol)
            
        except Exception as e:
            logger.error("Failed to get price", symbol=symbol, error=str(e))
            return None
    
    async def get_l2_book(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        """Get Level 2 order book"""
        try:
            asset_id = self._get_asset_id(symbol)
            l2_data = self.info.l2_snapshot(asset_id)
            return l2_data
            
        except Exception as e:
            logger.error("Failed to get L2 book", symbol=symbol, error=str(e))
            return None
    
    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[Dict[str, Any]]:
        """Get candlestick data"""
        try:
            asset_id = self._get_asset_id(symbol)
            # Convert interval to Hyperliquid format
            hl_interval = self._convert_interval(interval)
            
            klines = self.info.candles_snapshot(asset_id, hl_interval, limit)
            return klines
            
        except Exception as e:
            logger.error("Failed to get klines", symbol=symbol, error=str(e))
            return []
    
    def _get_asset_id(self, symbol: str) -> int:
        """Get asset ID for symbol"""
        # This is a simplified mapping - in production, use the meta API
        asset_mapping = {
            "ETH": 0,
            "BTC": 1,
            "SOL": 2,
            "ARB": 3,
            "AVAX": 4,
            "OP": 5,
            "MATIC": 6,
            "LINK": 7,
            "UNI": 8,
            "AAVE": 9
        }
        
        # Remove -PERP suffix if present
        clean_symbol = symbol.replace("-PERP", "").replace("-USD", "")
        return asset_mapping.get(clean_symbol, 0)
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval to Hyperliquid format"""
        interval_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        return interval_mapping.get(interval, "1m")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the client"""
        try:
            # Test API connectivity
            prices = await self._ensure_adapter().get_all_mid_prices()
            api_healthy = len(prices) > 0
            
            # Test trading if enabled
            trading_healthy = self.exchange is not None
            
            return {
                "api_connectivity": api_healthy,
                "trading_enabled": trading_healthy,
                "wallet_address": self.wallet_address,
                "network": "testnet" if self.use_testnet else "mainnet",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "api_connectivity": False,
                "trading_enabled": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Close connections and cleanup"""
        try:
            if self.adapter:
                await self.adapter.close()
            logger.info("HyperliquidClient closed")
            
        except Exception as e:
            logger.error("Error closing HyperliquidClient", error=str(e))


# Factory functions
def create_hyperliquid_client(private_key: Optional[str] = None, use_testnet: bool = True) -> HyperliquidClient:
    """Create a new HyperliquidClient instance"""
    return HyperliquidClient(private_key=private_key, use_testnet=use_testnet)

def create_readonly_client(use_testnet: bool = True) -> HyperliquidClient:
    """Create a read-only client for market data"""
    return HyperliquidClient(private_key=None, use_testnet=use_testnet)

def create_trading_client(private_key: str, use_testnet: bool = True) -> HyperliquidClient:
    """Create a trading-enabled client"""
    return HyperliquidClient(private_key=private_key, use_testnet=use_testnet)