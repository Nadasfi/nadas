"""
Web3 Provider Service for HyperEVM
Real blockchain connection for transaction simulation and execution
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
import json

try:
    from web3 import Web3, AsyncWeb3
    try:
        # Try new middleware import first (web3.py v6+)
        from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
    except ImportError:
        try:
            # Fallback to old import (web3.py v5)
            from web3.middleware import geth_poa_middleware
        except ImportError:
            # If both fail, set to None (we'll handle this case)
            geth_poa_middleware = None
    
    from web3.exceptions import (
        ContractLogicError, 
        TransactionNotFound, 
        BlockNotFound,
        Web3Exception
    )
    from eth_abi import encode, decode
    from eth_utils import to_checksum_address, keccak
    WEB3_AVAILABLE = True
except ImportError:
    Web3 = None
    AsyncWeb3 = None
    geth_poa_middleware = None
    WEB3_AVAILABLE = False

from app.core.config import settings
from app.core.error_handling import circuit_breaker_protected, track_errors, CircuitBreakerConfig
from app.core.logging import get_logger

logger = get_logger(__name__)

# HyperEVM Network Configuration
HYPEREVM_NETWORKS = {
    "testnet": {
        "rpc_url": "https://api.hyperliquid-testnet.xyz/evm",
        "chain_id": 998,
        "name": "HyperEVM Testnet",
        "block_explorer": "https://hyperliquid-testnet.xyz",
        "native_token": "ETH"
    },
    "mainnet": {
        "rpc_url": "https://api.hyperliquid.xyz/evm", 
        "chain_id": 42161,  # This might be different for HyperEVM mainnet
        "name": "HyperEVM Mainnet",
        "block_explorer": "https://hyperliquid.xyz",
        "native_token": "ETH"
    }
}

# HyperEVM Precompile Addresses
PRECOMPILE_ADDRESSES = {
    "hypercore_bridge": "0x0000000000000000000000000000000000000800",
    "name_service": "0x0000000000000000000000000000000000000801", 
    "oracle_bridge": "0x0000000000000000000000000000000000000802",
    "corewriter": "0x3333333333333333333333333333333333333333"
}

# Standard precompiles (available on all EVM chains)
STANDARD_PRECOMPILES = {
    "ecrecover": "0x0000000000000000000000000000000000000001",
    "sha256": "0x0000000000000000000000000000000000000002", 
    "ripemd160": "0x0000000000000000000000000000000000000003",
    "identity": "0x0000000000000000000000000000000000000004",
    "modexp": "0x0000000000000000000000000000000000000005",
    "ecadd": "0x0000000000000000000000000000000000000006",
    "ecmul": "0x0000000000000000000000000000000000000007",
    "ecpairing": "0x0000000000000000000000000000000000000008",
    "blake2f": "0x0000000000000000000000000000000000000009"
}

# HyperEVM-specific precompiles  
HYPEREVM_PRECOMPILES = {
    "hyperliquid_perp_order": "0x0000000000000000000000000000000000000100",
    "hyperliquid_spot_order": "0x0000000000000000000000000000000000000101",
    "hyperliquid_cancel_order": "0x0000000000000000000000000000000000000102",
    "hyperliquid_transfer": "0x0000000000000000000000000000000000000103",
    "spot_router": "0x0000000000000000000000000000000000000800",
    "perp_router": "0x0000000000000000000000000000000000000801",
    "oracle_bridge": "0x0000000000000000000000000000000000000802"
}

class HyperEVMProvider:
    """
    HyperEVM Web3 Provider for real blockchain interaction
    Supports transaction simulation, state queries, and precompile detection
    """
    
    def __init__(self, network: str = "testnet", custom_rpc: Optional[str] = None):
        """
        Initialize HyperEVM provider
        
        Args:
            network: "testnet" or "mainnet"
            custom_rpc: Custom RPC URL (overrides default)
        """
        if not WEB3_AVAILABLE:
            raise ImportError("web3 package not available. Install with: pip install web3")
        
        self.network = network
        self.network_config = HYPEREVM_NETWORKS[network]
        
        # Use custom RPC if provided
        rpc_url = custom_rpc or self.network_config["rpc_url"]
        
        # Initialize Web3 instances
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Add PoA middleware for some networks
        if network == "testnet" and geth_poa_middleware is not None:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Connection state
        self._connected = False
        self._latest_block = None
        self._precompiles_checked = False
        self._available_precompiles = {}
        
        logger.info("HyperEVM provider initialized", 
                   network=network, 
                   rpc_url=rpc_url,
                   chain_id=self.network_config["chain_id"])
    
    async def connect(self) -> bool:
        """Establish connection and verify network"""
        try:
            # Check if connected
            if not self.w3.is_connected():
                logger.error("Failed to connect to HyperEVM RPC")
                return False
            
            # Verify chain ID
            chain_id = self.w3.eth.chain_id
            expected_chain_id = self.network_config["chain_id"]
            
            if chain_id != expected_chain_id:
                logger.warning("Chain ID mismatch", 
                             expected=expected_chain_id, 
                             actual=chain_id)
            
            # Get latest block
            self._latest_block = self.w3.eth.block_number
            
            # Check precompiles
            await self._check_precompiles()
            
            self._connected = True
            logger.info("Connected to HyperEVM", chain_id=chain_id, latest_block=self._latest_block)
            return True
            
        except Exception as e:
            logger.error("HyperEVM connection failed", error=str(e))
            return False
    
    async def _check_precompiles(self):
        """Check which precompiles are available"""
        self._available_precompiles = {}
        
        # Check standard precompiles
        for name, address in STANDARD_PRECOMPILES.items():
            try:
                code = self.w3.eth.get_code(Web3.to_checksum_address(address))
                self._available_precompiles[name] = {
                    "address": address,
                    "available": len(code) > 0 or address in STANDARD_PRECOMPILES.values(),
                    "type": "standard"
                }
            except:
                self._available_precompiles[name] = {
                    "address": address,
                    "available": True,  # Standard precompiles should always be available
                    "type": "standard"
                }
        
        # Check HyperEVM-specific precompiles
        for name, address in PRECOMPILE_ADDRESSES.items():
            try:
                # Try to call the precompile with empty data
                result = self.w3.eth.call({
                    "to": Web3.to_checksum_address(address),
                    "data": "0x"
                })
                self._available_precompiles[name] = {
                    "address": address,
                    "available": True,
                    "type": "hyperevm"
                }
            except Exception as e:
                # Some precompiles might revert on empty call, but still be available
                self._available_precompiles[name] = {
                    "address": address,
                    "available": "execution reverted" not in str(e).lower(),
                    "type": "hyperevm"
                }
        
        self._precompiles_checked = True
        logger.info("Precompiles checked", 
                   available_count=sum(1 for p in self._available_precompiles.values() if p["available"]))
    
    @circuit_breaker_protected("hyperevm_simulation", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60))
    async def simulate_transaction(
        self,
        transaction: Dict[str, Any],
        block_number: Optional[Union[int, str]] = "latest",
        state_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate transaction execution
        
        Args:
            transaction: Transaction parameters
            block_number: Block to simulate against
            state_overrides: State modifications for simulation
        """
        async with track_errors("hyperevm_provider", {"method": "simulate_transaction"}):
            try:
                if not self._connected:
                    await self.connect()
                
                # Prepare transaction
                tx_params = {
                    "to": Web3.to_checksum_address(transaction.get("to", "0x0")),
                    "data": transaction.get("data", "0x"),
                    "value": int(transaction.get("value", 0)),
                    "gas": int(transaction.get("gas", 30_000_000)),
                    "gasPrice": int(transaction.get("gasPrice", 1_000_000_000))
                }
                
                if transaction.get("from"):
                    tx_params["from"] = Web3.to_checksum_address(transaction["from"])
                
                # Apply state overrides if provided
                if state_overrides:
                    # This would require debug_traceCall or similar advanced RPC method
                    logger.warning("State overrides not fully supported without debug RPC")
                
                # Estimate gas
                try:
                    estimated_gas = self.w3.eth.estimate_gas(tx_params, block_identifier=block_number)
                except Exception as gas_error:
                    logger.warning("Gas estimation failed", error=str(gas_error))
                    estimated_gas = tx_params["gas"]
                
                # Call the transaction (read-only execution)
                try:
                    call_result = self.w3.eth.call(tx_params, block_identifier=block_number)
                    success = True
                    revert_reason = None
                except ContractLogicError as e:
                    success = False
                    revert_reason = str(e)
                    call_result = "0x"
                except Exception as e:
                    success = False
                    revert_reason = str(e)
                    call_result = "0x"
                
                # Get current gas price
                try:
                    current_gas_price = self.w3.eth.gas_price
                except:
                    current_gas_price = tx_params["gasPrice"]
                
                # Get block info
                try:
                    block_info = self.w3.eth.get_block(block_number)
                    block_timestamp = block_info.timestamp
                    block_hash = block_info.hash.hex()
                except:
                    block_timestamp = int(datetime.now().timestamp())
                    block_hash = "0x" + "0" * 64
                
                # Check for precompile interactions
                precompile_calls = self._detect_precompile_calls(transaction)
                
                # Build simulation result
                simulation_result = {
                    "success": success,
                    "gas_used": estimated_gas,
                    "gas_limit": tx_params["gas"],
                    "gas_price": current_gas_price,
                    "effective_gas_price": current_gas_price,
                    "return_value": call_result.hex() if isinstance(call_result, bytes) else call_result,
                    "revert_reason": revert_reason,
                    "block_number": self.w3.eth.block_number if block_number == "latest" else block_number,
                    "block_timestamp": block_timestamp,
                    "block_hash": block_hash,
                    "precompile_calls": precompile_calls,
                    "logs": [],  # Would need receipt for logs
                    "state_changes": {},  # Would need debug trace for state changes
                    "network": self.network,
                    "chain_id": self.network_config["chain_id"]
                }
                
                logger.info("Transaction simulated", 
                           success=success, 
                           gas_used=estimated_gas,
                           to=tx_params["to"])
                
                return simulation_result
                
            except Exception as e:
                logger.error("Transaction simulation failed", error=str(e))
                return {
                    "success": False,
                    "error": str(e),
                    "gas_used": 0,
                    "revert_reason": str(e)
                }
    
    def _detect_precompile_calls(self, transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect precompile calls in transaction"""
        precompile_calls = []
        
        to_address = transaction.get("to", "").lower()
        
        # Check if calling a precompile
        for name, info in self._available_precompiles.items():
            if info["address"].lower() == to_address:
                precompile_calls.append({
                    "name": name,
                    "address": info["address"],
                    "type": info["type"],
                    "input_data": transaction.get("data", "0x"),
                    "available": info["available"]
                })
                break
        
        return precompile_calls
    
    async def get_gas_breakdown(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed gas breakdown for transaction"""
        try:
            simulation = await self.simulate_transaction(transaction)
            
            if not simulation["success"]:
                return {
                    "total_gas": 0,
                    "breakdown": {},
                    "error": simulation.get("revert_reason")
                }
            
            total_gas = simulation["gas_used"]
            
            # Basic gas breakdown (would need debug trace for detailed breakdown)
            breakdown = {
                "intrinsic_gas": 21000,  # Base transaction cost
                "execution_gas": max(0, total_gas - 21000),
                "total_gas": total_gas,
                "gas_price": simulation["gas_price"],
                "estimated_cost_wei": total_gas * simulation["gas_price"],
                "estimated_cost_eth": (total_gas * simulation["gas_price"]) / 10**18
            }
            
            return breakdown
            
        except Exception as e:
            logger.error("Gas breakdown failed", error=str(e))
            return {"total_gas": 0, "error": str(e)}
    
    async def get_state(self, address: str, block_number: Optional[Union[int, str]] = "latest") -> Dict[str, Any]:
        """Get account state"""
        try:
            checksum_addr = Web3.to_checksum_address(address)
            
            balance = self.w3.eth.get_balance(checksum_addr, block_number)
            nonce = self.w3.eth.get_transaction_count(checksum_addr, block_number)
            code = self.w3.eth.get_code(checksum_addr, block_number)
            
            return {
                "address": checksum_addr,
                "balance": str(balance),
                "nonce": nonce,
                "code": code.hex(),
                "is_contract": len(code) > 0,
                "block_number": self.w3.eth.block_number if block_number == "latest" else block_number
            }
            
        except Exception as e:
            logger.error("Failed to get state", address=address, error=str(e))
            return {}
    
    async def get_storage(self, address: str, slot: str, block_number: Optional[Union[int, str]] = "latest") -> str:
        """Get storage slot value"""
        try:
            checksum_addr = Web3.to_checksum_address(address)
            storage_value = self.w3.eth.get_storage_at(checksum_addr, slot, block_number)
            return storage_value.hex()
            
        except Exception as e:
            logger.error("Failed to get storage", address=address, slot=slot, error=str(e))
            return "0x" + "0" * 64
    
    async def get_block_info(self, block_number: Optional[Union[int, str]] = "latest") -> Dict[str, Any]:
        """Get block information"""
        try:
            block = self.w3.eth.get_block(block_number)
            
            return {
                "number": block.number,
                "hash": block.hash.hex(),
                "timestamp": block.timestamp,
                "gas_limit": block.gasLimit,
                "gas_used": block.gasUsed,
                "base_fee_per_gas": getattr(block, "baseFeePerGas", None),
                "difficulty": block.difficulty,
                "total_difficulty": getattr(block, "totalDifficulty", None),
                "transaction_count": len(block.transactions)
            }
            
        except Exception as e:
            logger.error("Failed to get block info", block_number=block_number, error=str(e))
            return {}
    
    async def create_access_list(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Create access list for transaction (if supported)"""
        try:
            # This requires eth_createAccessList RPC method
            # Not all HyperEVM nodes might support this
            
            tx_params = {
                "to": Web3.to_checksum_address(transaction.get("to", "0x0")),
                "data": transaction.get("data", "0x"),
                "value": int(transaction.get("value", 0))
            }
            
            if transaction.get("from"):
                tx_params["from"] = Web3.to_checksum_address(transaction["from"])
            
            # Try to create access list
            try:
                # This is a custom RPC call that might not be available
                access_list = self.w3.manager.request_blocking("eth_createAccessList", [tx_params])
                return {
                    "access_list": access_list.get("accessList", []),
                    "gas_used": access_list.get("gasUsed", 0),
                    "supported": True
                }
            except Exception as e:
                logger.warning("Access list creation not supported", error=str(e))
                return {
                    "access_list": [],
                    "gas_used": 0,
                    "supported": False,
                    "reason": "RPC method not supported"
                }
                
        except Exception as e:
            logger.error("Access list creation failed", error=str(e))
            return {"access_list": [], "error": str(e)}
    
    def get_precompiles_info(self) -> Dict[str, Any]:
        """Get information about available precompiles"""
        # Always provide known precompiles, even if not checked via RPC
        if not self._precompiles_checked:
            # Provide static precompiles info
            static_precompiles = {}
            for name, address in STANDARD_PRECOMPILES.items():
                static_precompiles[name] = {
                    "address": address,
                    "type": "standard",
                    "available": "unknown"  # Unknown without RPC check
                }
            for name, address in HYPEREVM_PRECOMPILES.items():
                static_precompiles[name] = {
                    "address": address,
                    "type": "hyperevm",
                    "available": "unknown"  # Unknown without RPC check
                }
            
            return {
                "precompiles": static_precompiles,
                "checked": False,
                "network": self.network
            }
        
        return {
            "precompiles": self._available_precompiles,
            "checked": True,
            "network": self.network,
            "hyperevm_specific": {
                name: info for name, info in self._available_precompiles.items() 
                if info["type"] == "hyperevm"
            },
            "standard": {
                name: info for name, info in self._available_precompiles.items() 
                if info["type"] == "standard"
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for provider"""
        try:
            if not self._connected:
                await self.connect()
            
            # Test basic connectivity
            latest_block = self.w3.eth.block_number
            chain_id = self.w3.eth.chain_id
            
            # Test a simple call
            balance = self.w3.eth.get_balance("0x0000000000000000000000000000000000000001")
            
            return {
                "connected": True,
                "latest_block": latest_block,
                "chain_id": chain_id,
                "network": self.network,
                "rpc_url": self.network_config["rpc_url"],
                "precompiles_available": len([p for p in self._available_precompiles.values() if p["available"]]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "network": self.network,
                "timestamp": datetime.now().isoformat()
            }
    
    def is_connected(self) -> bool:
        """Check if provider is connected"""
        return self._connected and self.w3.is_connected()
    
    async def close(self):
        """Close provider connections"""
        try:
            self._connected = False
            logger.info("HyperEVM provider closed")
        except Exception as e:
            logger.error("Error closing provider", error=str(e))


# Factory functions
def create_hyperevm_provider(network: str = "testnet", custom_rpc: Optional[str] = None) -> HyperEVMProvider:
    """Create HyperEVM provider instance"""
    return HyperEVMProvider(network=network, custom_rpc=custom_rpc)

def get_testnet_provider() -> HyperEVMProvider:
    """Get testnet provider"""
    return HyperEVMProvider(network="testnet")

def get_mainnet_provider() -> HyperEVMProvider:
    """Get mainnet provider"""
    return HyperEVMProvider(network="mainnet")

# Global provider instances (lazy initialized)
_testnet_provider = None
_mainnet_provider = None

async def get_default_provider(network: str = "testnet") -> HyperEVMProvider:
    """Get default provider instance (singleton)"""
    global _testnet_provider, _mainnet_provider
    
    if network == "testnet":
        if _testnet_provider is None:
            _testnet_provider = get_testnet_provider()
            await _testnet_provider.connect()
        return _testnet_provider
    else:
        if _mainnet_provider is None:
            _mainnet_provider = get_mainnet_provider()
            await _mainnet_provider.connect()
        return _mainnet_provider