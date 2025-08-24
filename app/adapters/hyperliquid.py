"""
Hyperliquid Adapter - REAL Python SDK Integration
Production-ready implementation with hyperliquid-python-sdk
"""

import asyncio
import json
from decimal import Decimal
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from web3 import Web3
from web3.exceptions import ContractLogicError
from eth_abi import encode, decode

import eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from structlog import get_logger

from app.core.config import settings
from app.core.error_handling import (
    circuit_breaker_protected, track_errors, safe_execute,
    CircuitBreakerConfig, RetryConfig, get_health_monitor,
    create_health_check, RetryableException
)

logger = get_logger(__name__)

# Global adapter instances
_mainnet_adapter = None
_testnet_adapter = None

# HyperEVM Constants
PRECOMPILE_ADDRESS = "0x0000000000000000000000000000000000000800"
COREWRITER_ADDRESS = "0x3333333333333333333333333333333333333333"


@dataclass
class HyperliquidPosition:
    """Represents a position on Hyperliquid"""
    
    symbol: str
    asset_id: int
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float
    side: str  # 'long' or 'short'
    liquidation_price: float
    mark_price: float
    last_updated: datetime
    
    @classmethod
    def from_api_data(cls, position_data: Dict[str, Any]) -> 'HyperliquidPosition':
        """Create position from API response data"""
        pos = position_data.get('position', {})
        return cls(
            symbol=pos.get('coin', 'UNKNOWN'),
            asset_id=position_data.get('assetId', 0),
            size=float(pos.get('szi', 0)),
            entry_price=float(pos.get('entryPx', 0)),
            unrealized_pnl=float(pos.get('unrealizedPnl', 0)),
            leverage=float(pos.get('leverage', 1)),
            side="long" if float(pos.get('szi', 0)) > 0 else "short",
            liquidation_price=float(pos.get('liquidationPx', 0)),
            mark_price=float(pos.get('markPx', 0)),
            last_updated=datetime.utcnow()
        )


@dataclass
class SpotBalance:
    """Represents spot token balance"""
    
    token: str
    total_balance: float
    available_balance: float
    locked_balance: float
    usd_value: float
    last_updated: datetime
    
    @classmethod
    def from_api_data(cls, balance_data: Dict[str, Any]) -> 'SpotBalance':
        """Create balance from API response data"""
        return cls(
            token=balance_data.get('coin', 'UNKNOWN'),
            total_balance=float(balance_data.get('total', 0)),
            available_balance=float(balance_data.get('available', 0)),
            locked_balance=float(balance_data.get('locked', 0)),
            usd_value=float(balance_data.get('usdValue', 0)),
            last_updated=datetime.utcnow()
        )


@dataclass
class MarketData:
    """Market data for an asset"""
    
    symbol: str
    mid_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    price_change_24h: float
    last_updated: datetime
    
    @classmethod
    def from_api_data(cls, symbol: str, market_data: Dict[str, Any]) -> 'MarketData':
        """Create market data from API response"""
        return cls(
            symbol=symbol,
            mid_price=float(market_data.get('mid', 0)),
            bid_price=float(market_data.get('bid', 0)),
            ask_price=float(market_data.get('ask', 0)),
            volume_24h=float(market_data.get('volume24h', 0)),
            price_change_24h=float(market_data.get('priceChange24h', 0)),
            last_updated=datetime.utcnow()
        )


@dataclass
class TransactionSimulation:
    """Transaction simulation results"""
    
    estimated_gas: int
    gas_price: float
    estimated_cost: float
    market_impact: Dict[str, float]
    slippage_estimate: float
    risk_score: float
    execution_time_ms: int
    success_probability: float
    warnings: List[str]
    
    @classmethod
    def create_failed(cls, error_message: str) -> 'TransactionSimulation':
        """Create failed simulation result"""
        return cls(
            estimated_gas=0,
            gas_price=0,
            estimated_cost=0,
            market_impact={"bps": 0, "usd": 0},
            slippage_estimate=0,
            risk_score=1.0,
            execution_time_ms=0,
            success_probability=0,
            warnings=[f"Simulation failed: {error_message}"]
        )


class HyperliquidAdapter:
    """Production-ready Hyperliquid integration using official Python SDK"""
    
    def __init__(self, use_mainnet: bool = False):
        """
        Initialize Hyperliquid adapter in READ-ONLY mode for client-side wallet security
        
        Args:
            use_mainnet: Whether to use mainnet (default: testnet)
        """
        self.use_mainnet = use_mainnet
        
        # Use correct API URLs from settings
        if use_mainnet:
            self.base_url = settings.HYPERLIQUID_MAINNET_URL
        else:
            self.base_url = settings.HYPERLIQUID_TESTNET_URL
        
        # Initialize Info API (read-only operations only)
        self.info = Info(self.base_url, skip_ws=False)
        
        # NO Exchange API - trading is done client-side for security
        self.exchange = None
        self.wallet = None
        
        # Symbol mappings cache
        self._symbol_to_asset_id = {}
        self._asset_id_to_symbol = {}
        
        # HyperEVM configuration for precompiles
        self.evm_rpc_url = settings.HYPEREVM_MAINNET_RPC if use_mainnet else settings.HYPEREVM_TESTNET_RPC
        self.chain_id = settings.HYPEREVM_CHAIN_ID_MAINNET if use_mainnet else settings.HYPEREVM_CHAIN_ID_TESTNET
        self.web3 = Web3(Web3.HTTPProvider(self.evm_rpc_url))
        
        logger.info("Hyperliquid adapter initialized in READ-ONLY mode",
                   network="mainnet" if use_mainnet else "testnet",
                   base_url=self.base_url,
                   evm_rpc=self.evm_rpc_url,
                   security_mode="client_side_trading")

    async def _load_symbol_mappings(self) -> None:
        """Load asset ID to symbol mappings"""
        try:
            meta = self.info.meta()
            
            for asset in meta.get("universe", []):
                asset_id = asset.get("name")  # This might be the asset name
                symbol = asset.get("name", "")
                
                if asset_id and symbol:
                    self._symbol_to_asset_id[symbol] = asset_id
                    self._asset_id_to_symbol[asset_id] = symbol
                    
            logger.debug("Symbol mappings loaded", 
                        symbols_count=len(self._symbol_to_asset_id))
                        
        except Exception as e:
            logger.error("Failed to load symbol mappings", error=str(e))

    @circuit_breaker_protected("hyperliquid_positions", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30))
    async def get_user_positions(self, address: str) -> List[HyperliquidPosition]:
        """Fetch perpetual positions for a user"""
        async with track_errors("hyperliquid_adapter", {"method": "get_user_positions", "address": address}):
            try:
                await self._load_symbol_mappings()
                
                user_state = self.info.user_state(address)
                positions = []
                
                if user_state and 'assetPositions' in user_state:
                    for pos_data in user_state['assetPositions']:
                        if pos_data.get('position') and float(pos_data['position'].get('szi', 0)) != 0:
                            # Add symbol name from asset mapping
                            asset_id = int(pos_data.get('assetId', 0))
                            symbol = self._asset_id_to_symbol.get(asset_id, f"ASSET_{asset_id}")
                            pos_data['position']['coin'] = symbol
                            
                            position = HyperliquidPosition.from_api_data(pos_data)
                            positions.append(position)
                
                logger.info("Fetched user positions", address=address, count=len(positions))
                return positions
                
            except Exception as e:
                logger.error("Error fetching user positions", address=address, error=str(e))
                return []  # Return empty list on error for graceful degradation

    @circuit_breaker_protected("hyperliquid_balances", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30))
    async def get_spot_balances(self, address: str) -> List[SpotBalance]:
        """Get spot token balances for a user"""
        async with track_errors("hyperliquid_adapter", {"method": "get_spot_balances", "address": address}):
            try:
                spot_state = self.info.spot_user_state(address)
                balances = []
                
                if spot_state and 'balances' in spot_state:
                    for balance_data in spot_state['balances']:
                        if float(balance_data.get('total', 0)) > 0:
                            balance = SpotBalance.from_api_data(balance_data)
                            balances.append(balance)
                
                logger.info("Fetched spot balances", address=address, count=len(balances))
                return balances
                
            except Exception as e:
                logger.error("Error fetching spot balances", address=address, error=str(e))
                return []

    async def get_account_value(self, address: str) -> Dict[str, float]:
        """Calculate total account equity and margin info"""
        try:
            user_state = self.info.user_state(address)
            
            if not user_state:
                return {"total_equity": 0.0, "margin_used": 0.0, "available_margin": 0.0}
                
            margin_summary = user_state.get('marginSummary', {})
            
            total_equity = float(margin_summary.get('accountValue', 0))
            margin_used = float(margin_summary.get('totalMarginUsed', 0))
            available_margin = total_equity - margin_used
            
            return {
                "total_equity": total_equity,
                "margin_used": margin_used,
                "available_margin": available_margin,
                "margin_ratio": margin_used / total_equity if total_equity > 0 else 0
            }
            
        except Exception as e:
            logger.error("Error calculating account value", address=address, error=str(e))
            return {"total_equity": 0.0, "margin_used": 0.0, "available_margin": 0.0}

    async def get_portfolio_summary(self, address: str) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            account_value = await self.get_account_value(address)
            positions = await self.get_user_positions(address)
            spot_balances = await self.get_spot_balances(address)
            
            # Calculate totals
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            summary = {
                "total_equity": account_value.get("total_equity", 0.0),
                "unrealized_pnl": total_unrealized_pnl,
                "margin_ratio": account_value.get("margin_ratio", 0.0),
                "buying_power": account_value.get("available_margin", 0.0),
                "margin_used": account_value.get("margin_used", 0.0),
                "positions_count": len(positions),
                "spot_balances_count": len(spot_balances)
            }
            
            return summary
            
        except Exception as e:
            logger.error("Error getting portfolio summary", address=address, error=str(e))
            return {
                "total_equity": 0.0,
                "unrealized_pnl": 0.0,
                "margin_ratio": 0.0,
                "buying_power": 0.0,
                "margin_used": 0.0,
                "positions_count": 0,
                "spot_balances_count": 0
            }

    async def get_all_mid_prices(self) -> Dict[str, float]:
        """Get mid prices for all assets"""
        try:
            all_mids = self.info.all_mids()
            
            # Convert string prices to floats
            mid_prices = {}
            for symbol, price_str in all_mids.items():
                try:
                    mid_prices[symbol] = float(price_str)
                except (ValueError, TypeError):
                    logger.warning("Invalid price for symbol", symbol=symbol, price=price_str)
                    
            logger.debug("Fetched mid prices", symbols_count=len(mid_prices))
            return mid_prices
            
        except Exception as e:
            logger.error("Error fetching mid prices", error=str(e))
            return {}

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for a specific symbol"""
        try:
            # This would need more detailed market data API
            mid_prices = await self.get_all_mid_prices()
            
            if symbol in mid_prices:
                return MarketData(
                    symbol=symbol,
                    mid_price=mid_prices[symbol],
                    bid_price=mid_prices[symbol] * 0.999,  # Approximate
                    ask_price=mid_prices[symbol] * 1.001,  # Approximate
                    volume_24h=0,  # Would need volume data
                    price_change_24h=0,  # Would need historical data
                    last_updated=datetime.utcnow()
                )
                
            return None
            
        except Exception as e:
            logger.error("Error fetching market data", symbol=symbol, error=str(e))
            return None

    async def get_price_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price data for a specific symbol (called by AI assistant)"""
        try:
            mid_prices = await self.get_all_mid_prices()
            
            if symbol in mid_prices:
                current_price = mid_prices[symbol]
                
                return {
                    "symbol": symbol,
                    "price": current_price,
                    "change_24h": 0.0,  # Hyperliquid doesn't provide 24h change in basic API
                    "volume": "N/A",    # Would need separate volume API call
                    "last_updated": datetime.utcnow().isoformat()
                }
                
            logger.warning("Symbol not found in mid prices", symbol=symbol, available_symbols=list(mid_prices.keys()))
            return None
            
        except Exception as e:
            logger.error("Error fetching price data", symbol=symbol, error=str(e))
            return None

    async def get_account_summary(self, address: str) -> Dict[str, Any]:
        """Get account summary for portfolio data (called by AI assistant)"""
        try:
            account_value = await self.get_account_value(address)
            positions = await self.get_user_positions(address)
            spot_balances = await self.get_spot_balances(address)
            
            # Build portfolio summary
            portfolio_summary = {
                "total_equity": account_value.get("total_equity", 0.0),
                "margin_used": account_value.get("margin_used", 0.0),
                "available_margin": account_value.get("available_margin", 0.0),
                "positions_count": len(positions),
                "spot_balances_count": len(spot_balances),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            if positions:
                portfolio_summary["positions"] = [
                    {
                        "symbol": pos.symbol,
                        "size": pos.size,
                        "side": pos.side,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "entry_price": pos.entry_price
                    } for pos in positions[:5]  # Limit to first 5 positions
                ]
            
            if spot_balances:
                portfolio_summary["balances"] = [
                    {
                        "coin": balance.coin,
                        "hold": balance.hold,
                        "total": balance.total
                    } for balance in spot_balances[:10]  # Limit to first 10 balances
                ]
            
            return portfolio_summary
            
        except Exception as e:
            logger.error("Error fetching account summary", address=address, error=str(e))
            return {}

    async def get_user_fills(self, address: str) -> List[Dict[str, Any]]:
        """Get user trade fills (transaction history)"""
        try:
            fills = self.info.user_fills(address)
            logger.info("Fetched user fills", address=address, count=len(fills))
            return fills
            
        except Exception as e:
            logger.error("Error fetching user fills", address=address, error=str(e))
            return []

    async def get_open_orders(self, address: str) -> List[Dict[str, Any]]:
        """Get open orders for user"""
        try:
            orders = self.info.open_orders(address)
            logger.info("Fetched open orders", address=address, count=len(orders))
            return orders
            
        except Exception as e:
            logger.error("Error fetching open orders", address=address, error=str(e))
            return []

    async def get_symbol_precision(self, symbol: str) -> Dict[str, int]:
        """Get precision requirements for a symbol"""
        try:
            # This would need to be fetched from meta information
            # For now, return default precision
            return {
                "size_decimals": 3,
                "price_decimals": 5
            }
            
        except Exception as e:
            logger.error("Error fetching symbol precision", symbol=symbol, error=str(e))
            return {"size_decimals": 3, "price_decimals": 5}

    def _round_to_precision(self, value: float, decimals: int) -> float:
        """Round value to specified decimal places"""
        multiplier = 10 ** decimals
        return round(value * multiplier) / multiplier

    async def prepare_order_data(self, symbol: str, is_buy: bool, size: float, 
                               price: Optional[float] = None, order_type: str = "limit",
                               reduce_only: bool = False, post_only: bool = False) -> Dict[str, Any]:
        """Prepare order data for client-side signing (READ-ONLY)"""
        
        try:
            # Get precision requirements
            precision = await self.get_symbol_precision(symbol)
            
            # Round size and price to correct precision
            rounded_size = self._round_to_precision(size, precision['size_decimals'])
            rounded_price = self._round_to_precision(price, precision['price_decimals']) if price else None
            
            # Validate minimum size (typically 0.001 for most assets)
            min_size = 1 / (10 ** precision['size_decimals'])
            if rounded_size < min_size:
                raise ValueError(f"Order size {rounded_size} is below minimum {min_size} for {symbol}")
            
            # Return order data for client-side execution
            order_data = {
                "symbol": symbol,
                "is_buy": is_buy,
                "size": rounded_size,
                "price": rounded_price,
                "order_type": order_type,
                "reduce_only": reduce_only,
                "post_only": post_only,
                "precision": precision,
                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                "requires_client_signing": True
            }
            
            logger.info("Order data prepared for client-side execution", 
                       symbol=symbol, 
                       is_buy=is_buy, 
                       size=rounded_size, 
                       price=rounded_price)
            
            return {
                "success": True,
                "order_data": order_data,
                "message": "Order prepared for client-side signing"
            }
            
        except Exception as e:
            logger.error("Error preparing order data", 
                       symbol=symbol, 
                       is_buy=is_buy, 
                       size=size, 
                       error=str(e))
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    async def prepare_market_order_data(self, symbol: str, is_buy: bool, size: float, 
                                      slippage: float = 0.01) -> Dict[str, Any]:
        """Prepare market order data for client-side execution"""
        return await self.prepare_order_data(symbol, is_buy, size, None, "market")

    async def prepare_close_position_data(self, symbol: str, user_address: str, 
                                        size: Optional[float] = None) -> Dict[str, Any]:
        """Prepare position closing data for client-side execution"""
        try:
            # Get current positions for the user
            positions = await self.get_user_positions(user_address)
            position = next((p for p in positions if p.symbol == symbol), None)
            
            if not position or position.size == 0:
                return {"success": False, "error": f"No open position for {symbol}"}
            
            # Determine close size
            close_size = size if size else abs(position.size)
            is_buy = position.size < 0  # Buy to close short, sell to close long
            
            return await self.prepare_order_data(symbol, is_buy, close_size, None, "market", reduce_only=True)
            
        except Exception as e:
            logger.error("Error preparing close position data", symbol=symbol, error=str(e))
            return {"success": False, "error": str(e)}

    async def prepare_leverage_update_data(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Prepare leverage update data for client-side execution"""
        try:
            return {
                "success": True,
                "leverage_data": {
                    "symbol": symbol,
                    "leverage": leverage,
                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    "requires_client_signing": True
                },
                "message": "Leverage update prepared for client-side signing"
            }
            
        except Exception as e:
            logger.error("Error preparing leverage data", symbol=symbol, leverage=leverage, error=str(e))
            return {"success": False, "error": str(e)}

    async def prepare_transfer_data(self, usd_amount: float, to_perp: bool) -> Dict[str, Any]:
        """Prepare USDC transfer data for client-side execution"""
        try:
            direction = "spot->perp" if to_perp else "perp->spot"
            
            return {
                "success": True,
                "transfer_data": {
                    "amount": usd_amount,
                    "to_perp": to_perp,
                    "direction": direction,
                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    "requires_client_signing": True
                },
                "message": f"Transfer {direction} prepared for client-side signing"
            }
            
        except Exception as e:
            logger.error("Error preparing transfer data", amount=usd_amount, to_perp=to_perp, error=str(e))
            return {"success": False, "error": str(e)}

    async def simulate_transaction(self, 
                                 asset_id: int, 
                                 is_buy: bool, 
                                 size: float, 
                                 price: Optional[float] = None) -> TransactionSimulation:
        """Simulate transaction execution with enhanced market data"""
        start_time = datetime.utcnow()
        
        try:
            # Get current market data
            all_mids = await self.get_all_mid_prices()
            symbol = self._asset_id_to_symbol.get(asset_id, f"ASSET_{asset_id}")
            current_price = all_mids.get(symbol, 0)
            
            if current_price == 0:
                return TransactionSimulation.create_failed(f"No price data for asset {asset_id}")
            
            # Enhanced simulation with WebSocket data
            from app.services.websocket_manager import get_websocket_manager, enhance_simulation_with_live_data
            
            try:
                ws_manager = get_websocket_manager()
                enhanced_data = await enhance_simulation_with_live_data(
                    ws_manager, symbol, size, is_buy
                )
                
                # Use live volatility if available
                volatility = enhanced_data.get("volatility", 0.3)  # Default 30%
                live_price = enhanced_data.get("live_price", current_price)
                current_price = live_price  # Use most recent price
                
            except Exception as e:
                logger.warning("WebSocket enhancement failed, using basic simulation", error=str(e))
                volatility = 0.3
            
            # Calculate transaction costs
            notional_value = size * current_price
            estimated_gas = 25000  # Typical gas for Hyperliquid operations
            gas_price = 0.000001  # Very low gas price on Hyperliquid
            estimated_cost = estimated_gas * gas_price
            
            # Calculate market impact (simplified)
            market_impact_bps = min(100, (notional_value / 1000000) * 10)  # 1 bps per $100k
            
            # Estimate slippage
            slippage_estimate = market_impact_bps * 0.01  # Convert bps to percentage
            
            # Calculate risk score (0-1, where 1 is highest risk)
            risk_factors = [
                size > 1000,  # Large position size
                market_impact_bps > 10,  # High market impact
                price and abs(price - current_price) / current_price > 0.05  # Price far from market
            ]
            risk_score = sum(risk_factors) / len(risk_factors)
            
            # Success probability
            success_probability = max(0.5, 1.0 - risk_score * 0.4)
            
            # Execution time
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Generate warnings
            warnings = []
            if market_impact_bps > 20:
                warnings.append(f"High market impact: {market_impact_bps:.1f} bps")
            if risk_score > 0.7:
                warnings.append("High risk transaction")
            if price and abs(price - current_price) / current_price > 0.03:
                warnings.append("Price significantly different from market")
            
            simulation = TransactionSimulation(
                estimated_gas=estimated_gas,
                gas_price=gas_price,
                estimated_cost=estimated_cost,
                market_impact={
                    "bps": market_impact_bps,
                    "usd": notional_value * (market_impact_bps / 10000)
                },
                slippage_estimate=slippage_estimate,
                risk_score=risk_score,
                execution_time_ms=execution_time,
                success_probability=success_probability,
                warnings=warnings
            )
            
            logger.info("Transaction simulation completed", 
                       asset_id=asset_id, 
                       size=size, 
                       risk_score=risk_score,
                       success_probability=success_probability)
            
            return simulation
            
        except Exception as e:
            logger.error("Transaction simulation failed", error=str(e))
            return TransactionSimulation.create_failed(str(e))

    async def close(self) -> None:
        """Close WebSocket connections and cleanup"""
        try:
            if hasattr(self.info, 'disconnect_websocket'):
                self.info.disconnect_websocket()
            logger.info("Hyperliquid adapter closed")
            
        except Exception as e:
            logger.error("Error closing adapter", error=str(e))


# Utility functions
def get_hyperliquid_adapter(use_mainnet: bool = None) -> HyperliquidAdapter:
    """Get configured Hyperliquid adapter instance (READ-ONLY mode)"""
    if use_mainnet is None:
        use_mainnet = getattr(settings, 'HYPERLIQUID_NETWORK', 'mainnet') == 'mainnet'
    
    return HyperliquidAdapter(use_mainnet=use_mainnet)


# Health check functions
async def hyperliquid_adapter_health_check() -> Dict[str, Any]:
    """Health check for Hyperliquid adapter"""
    return await create_health_check("hyperliquid_adapter", _check_hyperliquid_adapter)


async def _check_hyperliquid_adapter() -> Dict[str, Any]:
    """Internal health check for Hyperliquid adapter"""
    try:
        adapter = get_hyperliquid_adapter()
        
        # Test basic API connectivity
        try:
            all_mids = await adapter.get_all_mid_prices()
            api_connectivity = len(all_mids) > 0
        except Exception as e:
            api_connectivity = False
            api_error = str(e)
        
        await adapter.close()
        
        return {
            "adapter_initialized": True,
            "api_connectivity": api_connectivity,
            "api_error": api_error if not api_connectivity else None,
            "network": "mainnet" if adapter.use_mainnet else "testnet",
            "trading_enabled": adapter.exchange is not None
        }
        
    except Exception as e:
        return {
            "adapter_initialized": False,
            "error": str(e)
        }


# Register health check on import
def _register_hyperliquid_health_check():
    """Register Hyperliquid adapter health check"""
    health_monitor = get_health_monitor()
    health_monitor.register_health_check("hyperliquid_adapter", hyperliquid_adapter_health_check)

# Auto-register when module is imported  
_register_hyperliquid_health_check()