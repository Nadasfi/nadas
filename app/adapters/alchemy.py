"""
Alchemy API Adapter for Enhanced Market Data
Provides real-time and historical data for Hyperliquid via Alchemy
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from structlog import get_logger

from app.core.config import settings

logger = get_logger(__name__)


class AlchemyAdapter:
    """Alchemy API client for Hyperliquid data"""
    
    def __init__(self):
        self.api_key = settings.ALCHEMY_API_KEY
        self.base_url = settings.ALCHEMY_HYPERLIQUID_URL
        self.session: Optional[aiohttp.ClientSession] = None
        
        if not self.api_key or not self.base_url:
            logger.warning("Alchemy API key or URL not configured")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Nadas.fi/1.0.0'
                },
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def _make_request(self, method: str, params: List[Any] = None) -> Dict[str, Any]:
        """Make JSON-RPC request to Alchemy"""
        if not self.api_key:
            raise ValueError("Alchemy API key not configured")
        
        session = await self._get_session()
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": 1
        }
        
        try:
            async with session.post(self.base_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'error' in data:
                        logger.error("Alchemy API error", error=data['error'])
                        raise Exception(f"Alchemy API error: {data['error']}")
                    return data.get('result', {})
                else:
                    logger.error("HTTP error from Alchemy", status=response.status)
                    raise Exception(f"HTTP {response.status} from Alchemy")
                    
        except Exception as e:
            logger.error("Failed to make Alchemy request", method=method, error=str(e))
            raise
    
    async def get_block_number(self) -> int:
        """Get latest block number"""
        try:
            result = await self._make_request("eth_blockNumber")
            return int(result, 16) if isinstance(result, str) else result
        except Exception as e:
            logger.error("Failed to get block number", error=str(e))
            return 0
    
    async def get_block_by_number(self, block_number: int = None) -> Dict[str, Any]:
        """Get block details by number"""
        try:
            block_param = hex(block_number) if block_number else "latest"
            result = await self._make_request("eth_getBlockByNumber", [block_param, True])
            return result or {}
        except Exception as e:
            logger.error("Failed to get block", block_number=block_number, error=str(e))
            return {}
    
    async def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction receipt"""
        try:
            result = await self._make_request("eth_getTransactionReceipt", [tx_hash])
            return result or {}
        except Exception as e:
            logger.error("Failed to get transaction receipt", tx_hash=tx_hash, error=str(e))
            return {}
    
    async def get_balance(self, address: str, block: str = "latest") -> float:
        """Get ETH balance for address"""
        try:
            result = await self._make_request("eth_getBalance", [address, block])
            if result:
                # Convert from wei to ETH
                return int(result, 16) / 10**18
            return 0.0
        except Exception as e:
            logger.error("Failed to get balance", address=address, error=str(e))
            return 0.0
    
    async def get_gas_price(self) -> Dict[str, Any]:
        """Get current gas price information"""
        try:
            # Get base fee from latest block
            latest_block = await self.get_block_by_number()
            base_fee = int(latest_block.get('baseFeePerGas', '0x0'), 16) if latest_block else 0
            
            # Get gas price estimate
            gas_price_result = await self._make_request("eth_gasPrice")
            gas_price = int(gas_price_result, 16) if gas_price_result else 0
            
            return {
                'gas_price': gas_price,
                'base_fee': base_fee,
                'gas_price_gwei': gas_price / 10**9,
                'base_fee_gwei': base_fee / 10**9,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error("Failed to get gas price", error=str(e))
            return {
                'gas_price': 0,
                'base_fee': 0,
                'gas_price_gwei': 0,
                'base_fee_gwei': 0,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_transaction_count(self, address: str, block: str = "latest") -> int:
        """Get transaction count (nonce) for address"""
        try:
            result = await self._make_request("eth_getTransactionCount", [address, block])
            return int(result, 16) if result else 0
        except Exception as e:
            logger.error("Failed to get transaction count", address=address, error=str(e))
            return 0
    
    async def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas for transaction"""
        try:
            result = await self._make_request("eth_estimateGas", [transaction])
            return int(result, 16) if result else 21000  # Default gas limit
        except Exception as e:
            logger.error("Failed to estimate gas", transaction=transaction, error=str(e))
            return 21000
    
    async def send_raw_transaction(self, raw_tx: str) -> str:
        """Send raw signed transaction"""
        try:
            result = await self._make_request("eth_sendRawTransaction", [raw_tx])
            return result or ""
        except Exception as e:
            logger.error("Failed to send transaction", error=str(e))
            raise
    
    async def get_logs(self, filter_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get logs matching filter"""
        try:
            result = await self._make_request("eth_getLogs", [filter_params])
            return result or []
        except Exception as e:
            logger.error("Failed to get logs", filter_params=filter_params, error=str(e))
            return []
    
    async def trace_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction trace (if supported)"""
        try:
            # Try trace_transaction method
            result = await self._make_request("trace_transaction", [tx_hash])
            return result or {}
        except Exception:
            try:
                # Fallback to debug_traceTransaction
                result = await self._make_request("debug_traceTransaction", [tx_hash])
                return result or {}
            except Exception as e:
                logger.error("Failed to trace transaction", tx_hash=tx_hash, error=str(e))
                return {}
    
    async def get_token_metadata(self, contract_address: str) -> Dict[str, Any]:
        """Get ERC-20 token metadata"""
        try:
            # This would require contract calls - implementing basic structure
            return {
                'address': contract_address,
                'name': 'Unknown Token',
                'symbol': 'UNK',
                'decimals': 18,
                'total_supply': 0,
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error("Failed to get token metadata", address=contract_address, error=str(e))
            return {}
    
    async def get_enhanced_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get enhanced market data combining multiple sources"""
        try:
            current_block = await self.get_block_number()
            gas_info = await self.get_gas_price()
            
            return {
                'symbol': symbol,
                'current_block': current_block,
                'gas_info': gas_info,
                'network_health': {
                    'block_time': 'normal',  # Would calculate from recent blocks
                    'congestion_level': 'low' if gas_info['gas_price_gwei'] < 20 else 'high'
                },
                'timestamp': datetime.utcnow().isoformat(),
                'data_source': 'alchemy'
            }
        except Exception as e:
            logger.error("Failed to get enhanced market data", symbol=symbol, error=str(e))
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_transaction_analysis(self, tx_hash: str) -> Dict[str, Any]:
        """Analyze transaction with enhanced data"""
        try:
            receipt = await self.get_transaction_receipt(tx_hash)
            trace = await self.trace_transaction(tx_hash)
            
            if not receipt:
                return {'error': 'Transaction not found'}
            
            gas_used = int(receipt.get('gasUsed', '0x0'), 16)
            gas_price = int(receipt.get('effectiveGasPrice', '0x0'), 16)
            
            return {
                'transaction_hash': tx_hash,
                'block_number': int(receipt.get('blockNumber', '0x0'), 16),
                'gas_used': gas_used,
                'gas_price': gas_price,
                'gas_cost_eth': (gas_used * gas_price) / 10**18,
                'status': 'success' if receipt.get('status') == '0x1' else 'failed',
                'trace_data': trace,
                'timestamp': datetime.utcnow().isoformat(),
                'analysis': {
                    'gas_efficiency': 'optimal' if gas_used < 100000 else 'high',
                    'cost_usd': 0,  # Would need price feed
                }
            }
        except Exception as e:
            logger.error("Failed to analyze transaction", tx_hash=tx_hash, error=str(e))
            return {'error': str(e), 'transaction_hash': tx_hash}
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


# Global instance
_alchemy_adapter: Optional[AlchemyAdapter] = None

def get_alchemy_adapter() -> AlchemyAdapter:
    """Get or create Alchemy adapter instance"""
    global _alchemy_adapter
    if _alchemy_adapter is None:
        _alchemy_adapter = AlchemyAdapter()
    return _alchemy_adapter