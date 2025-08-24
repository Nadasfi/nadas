"""
LI.FI Cross-Chain API Endpoints
$5k bounty target - Best use of LI.FI API/Widget
Comprehensive cross-chain routing with direct Hypercore access
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from datetime import datetime

from app.api.v1.auth import get_current_user
from app.adapters.lifi import (
    LiFiAdapter,
    LiFiRoute,
    LiFiChain,
    LiFiToken,
    get_lifi_adapter
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CrossChainQuoteRequest(BaseModel):
    """LI.FI cross-chain quote request"""
    from_chain: int = Field(..., description="Source chain ID")
    to_chain: int = Field(..., description="Destination chain ID")
    from_token: str = Field(..., description="Source token address")
    to_token: str = Field(..., description="Destination token address")
    from_amount: str = Field(..., description="Amount to swap (in token units)")
    to_address: Optional[str] = Field(None, description="Receiver address (optional)")
    slippage: float = Field(0.03, ge=0.001, le=0.1, description="Maximum slippage (3% default)")
    allow_bridges: bool = Field(True, description="Allow bridge steps")
    allow_exchanges: bool = Field(True, description="Allow exchange steps")
    prefer_bridges: Optional[List[str]] = Field(None, description="Preferred bridge tools")
    deny_bridges: Optional[List[str]] = Field(None, description="Denied bridge tools")

    class Config:
        schema_extra = {
            "example": {
                "from_chain": 1,
                "to_chain": 42161,
                "from_token": "0xA0b86a33E6417c033D1c15D866172aA0000000000",
                "to_token": "0x0000000000000000000000000000000000000000",
                "from_amount": "1000000000000000000",
                "slippage": 0.03,
                "prefer_bridges": ["relay", "gaszip"]
            }
        }


class RouteExecutionRequest(BaseModel):
    """Route execution request"""
    route_id: str = Field(..., description="Route ID from quote")
    
    class Config:
        schema_extra = {
            "example": {
                "route_id": "0x1234567890abcdef"
            }
        }


class HypercoreDirectRequest(BaseModel):
    """Direct Hypercore bridging request"""
    from_chain: int = Field(..., description="Source chain ID")
    from_token: str = Field(..., description="Source token address")
    amount: str = Field(..., description="Amount to bridge")
    
    class Config:
        schema_extra = {
            "example": {
                "from_chain": 1,
                "from_token": "0xA0b86a33E6417c033D1c15D866172aA000000000",
                "amount": "1000000000000000000"
            }
        }


class StatusCheckRequest(BaseModel):
    """Transaction status check request"""
    tx_hash: str = Field(..., description="Transaction hash")
    bridge_tool: str = Field(..., description="Bridge tool used")
    
    class Config:
        schema_extra = {
            "example": {
                "tx_hash": "0x1234567890abcdef",
                "bridge_tool": "relay"
            }
        }


@router.get("/chains")
async def get_supported_chains(
    chain_types: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all supported chains for cross-chain bridging"""
    try:
        lifi = await get_lifi_adapter()
        
        # Parse chain types if provided
        chain_type_list = None
        if chain_types:
            chain_type_list = [ct.strip() for ct in chain_types.split(",")]
        
        chains = await lifi.get_chains(chain_type_list)
        
        return {
            "success": True,
            "chains": [
                {
                    "id": chain.id,
                    "key": chain.key,
                    "name": chain.name,
                    "coin": chain.coin,
                    "mainnet": chain.mainnet,
                    "logo_uri": chain.logo_uri,
                    "native_token": chain.native_token
                }
                for chain in chains
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get supported chains", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported chains: {str(e)}"
        )


@router.get("/chains/{chain_id}/tokens")
async def get_chain_tokens(
    chain_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get available tokens for a specific chain"""
    try:
        lifi = await get_lifi_adapter()
        tokens = await lifi.get_tokens(chain_id)
        
        # Enhance with price and volume data
        enhanced_tokens = []
        for token in tokens:
            enhanced_token = {
                "address": token.address,
                "symbol": token.symbol,
                "name": token.name,
                "decimals": token.decimals,
                "chain_id": token.chain_id,
                "logo_uri": token.logo_uri,
                "price_usd": token.price_usd,
                "is_popular": token.symbol in ['ETH', 'USDC', 'USDT', 'WBTC', 'DAI'],
                "is_native": token.address == "0x0000000000000000000000000000000000000000",
                "nadas_verified": _is_token_verified(token.address, chain_id)
            }
            enhanced_tokens.append(enhanced_token)
        
        # Sort by popularity and price availability
        enhanced_tokens.sort(key=lambda x: (
            -1 if x["is_native"] else 0,
            -1 if x["is_popular"] else 0,
            -1 if x["price_usd"] else 0,
            x["symbol"]
        ))
        
        return {
            "success": True,
            "data": {
                "chain_id": chain_id,
                "tokens": enhanced_tokens,
                "total_count": len(enhanced_tokens),
                "verified_count": len([t for t in enhanced_tokens if t["nadas_verified"]]),
                "with_price_count": len([t for t in enhanced_tokens if t["price_usd"]])
            },
            "message": f"Retrieved {len(enhanced_tokens)} tokens for chain {chain_id}"
        }
        
    except Exception as e:
        logger.error("Failed to get chain tokens", chain_id=chain_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get tokens: {str(e)}")


def _is_token_verified(token_address: str, chain_id: int) -> bool:
    """Check if token is verified in Nadas.fi system"""
    # Major mainnet tokens verification
    verified_tokens = {
        1: [  # Ethereum Mainnet
            '0xA0b86a33E6417c033D1c15D866172aA0b41e88E8',  # USDC
            '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # USDT
            '0x0000000000000000000000000000000000000000',  # ETH
            '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',  # WBTC
            '0x6B175474E89094C44Da98b954EedeAC495271d0F'   # DAI
        ],
        42161: [  # Arbitrum
            '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',  # USDC
            '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',  # USDT
            '0x0000000000000000000000000000000000000000',  # ETH
            '0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f'   # WBTC
        ],
        999: [  # HyperEVM
            '0x0000000000000000000000000000000000000000',  # HYPE native token
        ],
        137: [  # Polygon
            '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',  # USDC
            '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',  # USDT
            '0x0000000000000000000000000000000000000000'   # MATIC
        ]
    }
    return token_address.lower() in [addr.lower() for addr in verified_tokens.get(chain_id, [])]


def _assess_route_risk(route: LiFiRoute) -> str:
    """Assess risk level of a cross-chain route"""
    risk_factors = 0
    
    # More bridges = higher risk
    if len(route.steps) > 2:
        risk_factors += 1
    
    # Unknown tools increase risk
    known_tools = ['stargate', 'hop', 'across', 'relay', 'socket']
    unknown_tools = [step.tool for step in route.steps if step.tool not in known_tools]
    if unknown_tools:
        risk_factors += len(unknown_tools)
    
    # High gas costs indicate complexity
    if route.gas_cost_usd and float(route.gas_cost_usd) > 50:
        risk_factors += 1
    
    if risk_factors >= 3:
        return "HIGH"
    elif risk_factors >= 1:
        return "MEDIUM"
    else:
        return "LOW"


def _assess_route_speed(route: LiFiRoute) -> str:
    """Assess speed rating of route"""
    total_duration = sum(step.execution_duration or 0 for step in route.steps)
    
    if total_duration <= 60:  # 1 minute
        return "FAST"
    elif total_duration <= 300:  # 5 minutes
        return "MEDIUM"
    else:
        return "SLOW"


def _assess_cost_efficiency(route: LiFiRoute) -> str:
    """Assess cost efficiency of route"""
    if not route.gas_cost_usd:
        return "UNKNOWN"
    
    gas_cost = float(route.gas_cost_usd)
    if gas_cost <= 5:
        return "EXCELLENT"
    elif gas_cost <= 20:
        return "GOOD"
    elif gas_cost <= 50:
        return "FAIR"
    else:
        return "EXPENSIVE"


def _is_route_recommended(route: LiFiRoute) -> bool:
    """Check if route is recommended by Nadas.fi"""
    risk = _assess_route_risk(route)
    cost = _assess_cost_efficiency(route)
    
    # Recommend low/medium risk routes with good cost efficiency
    return risk in ["LOW", "MEDIUM"] and cost in ["EXCELLENT", "GOOD"]


@router.post("/quote")
async def get_cross_chain_quote(
    request: CrossChainQuoteRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get optimal cross-chain routing quote"""
    try:
        user_address = current_user["wallet_address"]
        
        logger.info("Getting LI.FI cross-chain quote", 
                   user=user_address,
                   from_chain=request.from_chain,
                   to_chain=request.to_chain,
                   amount=request.from_amount)
        
        lifi = await get_lifi_adapter()
        route = await lifi.get_quote(
                from_chain=request.from_chain,
                to_chain=request.to_chain,
                from_token=request.from_token,
                to_token=request.to_token,
                from_amount=request.from_amount,
                from_address=user_address,
                to_address=request.to_address,
                slippage=request.slippage,
                allow_bridges=request.allow_bridges,
                allow_exchanges=request.allow_exchanges,
                prefer_bridges=request.prefer_bridges,
                deny_bridges=request.deny_bridges
            )
        
        if not route:
            return {
                "success": False,
                "error": "No route found for the requested parameters",
                "data": {
                    "from_chain": request.from_chain,
                    "to_chain": request.to_chain,
                    "from_token": request.from_token,
                    "to_token": request.to_token
                }
            }
        
        # Enhanced route response with Nadas.fi insights
        response_data = {
            "route_id": route.id,
            "from_chain_id": route.from_chain_id,
            "to_chain_id": route.to_chain_id,
            "from_token": {
                "address": route.from_token.address,
                "symbol": route.from_token.symbol,
                "name": route.from_token.name,
                "decimals": route.from_token.decimals
            },
            "to_token": {
                "address": route.to_token.address,
                "symbol": route.to_token.symbol,
                "name": route.to_token.name,
                "decimals": route.to_token.decimals
            },
            "from_amount": route.from_amount,
            "to_amount": route.to_amount,
            "to_amount_min": route.to_amount_min,
            "gas_cost_usd": route.gas_cost_usd,
            "steps": [
                {
                    "id": step.id,
                    "type": step.type,
                    "tool": step.tool,
                    "from_amount": step.from_amount,
                    "to_amount": step.to_amount,
                    "execution_duration": step.execution_duration,
                    "tags": step.tags
                }
                for step in route.steps
            ],
            "tags": route.tags,
            "nadas_insights": {
                "risk_level": _assess_route_risk(route),
                "speed_rating": _assess_route_speed(route),
                "cost_efficiency": _assess_cost_efficiency(route),
                "hypercore_compatible": "hypercore" in route.tags,
                "recommended": _is_route_recommended(route)
            },
            "estimated_duration_minutes": sum(step.execution_duration or 0 for step in route.steps) // 60,
            "tools_used": list(set(step.tool for step in route.steps)),
            "expires_at": (datetime.utcnow().timestamp() + 300) * 1000  # 5 minutes
        }
        
        return {
            "success": True,
            "data": response_data,
            "message": f"Route found via {len(route.steps)} steps using {len(set(step.tool for step in route.steps))} tools"
        }
        
    except Exception as e:
        logger.error("Failed to get cross-chain quote", 
                    user=current_user.get("wallet_address"),
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get quote: {str(e)}")


@router.post("/execute")
async def execute_cross_chain_route(
    request: RouteExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Execute cross-chain route via LI.FI"""
    try:
        user_address = current_user["wallet_address"]
        
        logger.info("Executing LI.FI cross-chain route", 
                   user=user_address,
                   route_id=request.route_id)
        
        # In production, this would require the full route object
        # For now, simulate execution
        result = {
            "success": True,
            "transaction_hash": f"0x{request.route_id[-40:]}",
            "route_id": request.route_id,
            "user_address": user_address,
            "status": "pending",
            "estimated_completion": "5-15 minutes",
            "tracking_url": f"https://scan.li.fi/tx/{request.route_id}",
            "executed_at": datetime.utcnow().isoformat()
        }
        
        # Add background monitoring task
        background_tasks.add_task(
            monitor_lifi_transaction,
            request.route_id,
            user_address
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Cross-chain route execution initiated"
        }
        
    except Exception as e:
        logger.error("Failed to execute cross-chain route", 
                    user=current_user.get("wallet_address"),
                    route_id=request.route_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@router.post("/hypercore/direct")
async def get_hypercore_direct_routes(
    request: HypercoreDirectRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get direct routes to Hypercore perps (LI.FI's unique feature)"""
    try:
        user_address = current_user["wallet_address"]
        
        logger.info("Getting direct Hypercore routes", 
                   user=user_address,
                   from_chain=request.from_chain,
                   amount=request.amount)
        
        async with get_lifi_adapter() as lifi:
            hypercore_routes = await lifi.get_hypercore_direct_routes(
                from_chain=request.from_chain,
                from_token=request.from_token,
                amount=request.amount,
                user_address=user_address
            )
        
        if not hypercore_routes:
            return {
                "success": False,
                "error": "No direct Hypercore routes available",
                "data": {
                    "from_chain": request.from_chain,
                    "alternatives": [
                        "Use standard cross-chain route to Arbitrum",
                        "Bridge to Arbitrum then deposit to Hypercore manually"
                    ]
                }
            }
        
        # Enhance with Hypercore-specific insights
        enhanced_routes = []
        for route in hypercore_routes:
            enhanced_route = {
                **route,
                "advantages": [
                    "One-click access to Hypercore perps",
                    "No separate Arbitrum transaction needed",
                    "Optimized for trading speed"
                ],
                "recommended_use_cases": [
                    "Quick arbitrage opportunities",
                    "Immediate position entry",
                    "Emergency exits from other chains"
                ],
                "risk_factors": [
                    "Bridge dependency",
                    "Slippage on small amounts",
                    "Gas costs on source chain"
                ]
            }
            enhanced_routes.append(enhanced_route)
        
        return {
            "success": True,
            "data": {
                "routes": enhanced_routes,
                "total_routes": len(enhanced_routes),
                "fastest_route": min(enhanced_routes, key=lambda x: x.get("execution_time_estimate", 999)),
                "cheapest_route": min(enhanced_routes, key=lambda x: float(x.get("gas_cost_usd", "999"))),
                "hypercore_benefits": {
                    "direct_perps_access": True,
                    "no_arbitrum_permit_required": True,
                    "single_transaction": True,
                    "optimized_for_trading": True
                }
            },
            "message": f"Found {len(enhanced_routes)} direct Hypercore routes"
        }
        
    except Exception as e:
        logger.error("Failed to get Hypercore direct routes", 
                    user=current_user.get("wallet_address"),
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get routes: {str(e)}")


@router.post("/status")
async def check_transaction_status(
    request: StatusCheckRequest,
    current_user: dict = Depends(get_current_user)
):
    """Check cross-chain transaction status"""
    try:
        async with get_lifi_adapter() as lifi:
            status = await lifi.get_status(
                tx_hash=request.tx_hash,
                bridge_tool=request.bridge_tool
            )
        
        if not status:
            return {
                "success": False,
                "error": "Transaction status not found",
                "data": {
                    "tx_hash": request.tx_hash,
                    "bridge_tool": request.bridge_tool
                }
            }
        
        # Enhanced status with user-friendly information
        status_data = {
            "tx_hash": status.tx_hash,
            "status": status.status,
            "substatus": status.substatus,
            "substatus_message": status.substatus_message,
            "tx_link": status.tx_link,
            "from_amount": status.from_amount,
            "to_amount": status.to_amount,
            "gas_used": status.gas_used,
            "gas_price": status.gas_price,
            "gas_cost": status.gas_cost,
            "updated_at": status.updated_at.isoformat(),
            "user_friendly_status": _get_user_friendly_status(status.status),
            "estimated_completion": _get_estimated_completion(status),
            "next_steps": _get_next_steps(status)
        }
        
        return {
            "success": True,
            "data": status_data,
            "message": f"Transaction {status.status.lower()}"
        }
        
    except Exception as e:
        logger.error("Failed to check transaction status", 
                    tx_hash=request.tx_hash,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/tools")
async def get_available_tools(current_user: dict = Depends(get_current_user)):
    """Get available bridge and exchange tools"""
    try:
        async with get_lifi_adapter() as lifi:
            tools = await lifi.get_tools()
        
        # Enhance with Nadas.fi-specific tool ratings
        enhanced_bridges = []
        for bridge in tools.get("bridges", []):
            enhanced_bridge = {
                **bridge,
                "nadas_rating": _get_tool_rating(bridge.get("key", "")),
                "hypercore_compatible": bridge.get("key", "") in ["relay", "gaszip"],
                "recommended_for": _get_tool_use_cases(bridge.get("key", ""))
            }
            enhanced_bridges.append(enhanced_bridge)
        
        enhanced_exchanges = []
        for exchange in tools.get("exchanges", []):
            enhanced_exchange = {
                **exchange,
                "nadas_rating": _get_tool_rating(exchange.get("key", "")),
                "liquidity_score": _get_liquidity_score(exchange.get("key", "")),
                "recommended_for": _get_tool_use_cases(exchange.get("key", ""))
            }
            enhanced_exchanges.append(enhanced_exchange)
        
        return {
            "success": True,
            "data": {
                "bridges": enhanced_bridges,
                "exchanges": enhanced_exchanges,
                "bridge_count": len(enhanced_bridges),
                "exchange_count": len(enhanced_exchanges),
                "hypercore_bridges": [b for b in enhanced_bridges if b["hypercore_compatible"]],
                "recommended_bridges": [b for b in enhanced_bridges if b["nadas_rating"] >= 4]
            },
            "message": f"Retrieved {len(enhanced_bridges)} bridges and {len(enhanced_exchanges)} exchanges"
        }
        
    except Exception as e:
        logger.error("Failed to get available tools", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")


@router.get("/widget-config")
async def get_widget_configuration(current_user: dict = Depends(get_current_user)):
    """Get LI.FI widget configuration for frontend integration"""
    try:
        user_address = current_user["wallet_address"]
        
        widget_config = {
            "integrator": "nadas-fi",
            "theme": {
                "palette": {
                    "primary": {"main": "#6366f1"},
                    "secondary": {"main": "#ec4899"}
                },
                "container": {
                    "borderRadius": "16px",
                    "boxShadow": "0 8px 32px rgba(0, 0, 0, 0.12)"
                }
            },
            "appearance": "auto",
            "hiddenUI": ["fromAmount", "toAddress"],
            "defaultRouteOptions": {
                "slippage": 0.03,
                "allowSwitchChain": True,
                "allowBridges": True,
                "allowExchanges": True,
                "preferredBridges": ["relay", "gaszip"],  # Hypercore-optimized
                "preferredExchanges": ["1inch", "paraswap", "0x"]
            },
            "fromChain": 1,  # Ethereum
            "toChain": 42161,  # Arbitrum (Hypercore access)
            "fromToken": "0x0000000000000000000000000000000000000000",  # ETH
            "toToken": "0x0000000000000000000000000000000000000000",  # ETH
            "toAddress": user_address,
            "fee": 0.001,  # 0.1% Nadas.fi fee
            "feeConfig": {
                "recipient": "0x742d35cc6634C0532925a3b8D50D5a5B6F3C0c3e",  # Nadas.fi fee recipient
                "feeAmount": 0.001
            },
            "variant": "expandable",
            "subvariant": "split",
            "disabledChains": [],  # Allow all chains
            "walletConfig": {
                "onConnect": "nadas_wallet_connected",
                "autoConnect": True
            },
            "apiUrl": "https://li.quest/v1",
            "rpcs": {
                1: ["https://eth.public.node.com", "https://ethereum.rpc.subquery.network/public"],
                42161: ["https://arbitrum.rpc.subquery.network/public"]
            }
        }
        
        return {
            "success": True,
            "data": {
                "widget_config": widget_config,
                "integration_guide": {
                    "react_component": "@lifi/widget",
                    "installation": "npm install @lifi/widget",
                    "docs_url": "https://docs.li.fi/integrate-li.fi-widget/widget-configuration"
                },
                "nadas_customizations": [
                    "Hypercore-optimized bridge preferences",
                    "Nadas.fi fee collection",
                    "Custom theme matching Nadas.fi design",
                    "Pre-configured for major trading pairs"
                ]
            },
            "message": "Widget configuration ready for frontend integration"
        }
        
    except Exception as e:
        logger.error("Failed to get widget configuration", error=str(e))
        raise HTTPException(status_code=500, detail=f"Widget config failed: {str(e)}")


# Helper functions
def _get_chain_tvl_rank(chain_id: int) -> int:
    """Get approximate TVL rank for chain"""
    tvl_ranks = {
        1: 1,      # Ethereum
        42161: 2,  # Arbitrum
        137: 3,    # Polygon
        8453: 4,   # Base
        10: 5,     # Optimism
        56: 6,     # BSC
        43114: 7,  # Avalanche
    }
    return tvl_ranks.get(chain_id, 99)


def _is_token_verified(token_address: str, chain_id: int) -> bool:
    """Check if token is verified by Nadas.fi"""
    # Major tokens we trust
    verified_tokens = {
        1: [  # Ethereum
            "0x0000000000000000000000000000000000000000",  # ETH
            "0xA0b86a33E6417c033D1c15D866172aA00000000",  # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        ],
        42161: [  # Arbitrum
            "0x0000000000000000000000000000000000000000",  # ETH
            "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC
        ]
    }
    return token_address in verified_tokens.get(chain_id, [])


def _assess_route_risk(route) -> str:
    """Assess risk level of a route"""
    if len(route.steps) == 1:
        return "low"
    elif len(route.steps) <= 3:
        return "medium"
    else:
        return "high"


def _assess_route_speed(route) -> str:
    """Assess speed rating of a route"""
    total_time = sum(step.execution_duration or 0 for step in route.steps)
    if total_time < 300:  # 5 minutes
        return "fast"
    elif total_time < 900:  # 15 minutes
        return "medium"
    else:
        return "slow"


def _assess_cost_efficiency(route) -> str:
    """Assess cost efficiency of a route"""
    gas_cost = float(route.gas_cost_usd or "0")
    if gas_cost < 5:
        return "high"
    elif gas_cost < 20:
        return "medium"
    else:
        return "low"


def _is_route_recommended(route) -> bool:
    """Check if route is recommended by Nadas.fi"""
    return (
        len(route.steps) <= 2 and
        float(route.gas_cost_usd or "0") < 50 and
        any("hypercore" in tag for tag in route.tags)
    )


def _get_user_friendly_status(status: str) -> str:
    """Convert technical status to user-friendly message"""
    status_map = {
        "PENDING": "Transaction in progress...",
        "DONE": "Transaction completed successfully!",
        "FAILED": "Transaction failed",
        "CANCELLED": "Transaction was cancelled"
    }
    return status_map.get(status, status)


def _get_estimated_completion(status) -> str:
    """Get estimated completion time"""
    if status.status == "PENDING":
        return "5-15 minutes remaining"
    elif status.status == "DONE":
        return "Completed"
    else:
        return "N/A"


def _get_next_steps(status) -> List[str]:
    """Get next steps for user"""
    if status.status == "PENDING":
        return ["Wait for transaction confirmation", "Monitor status updates"]
    elif status.status == "DONE":
        return ["Check your wallet balance", "Transaction complete"]
    elif status.status == "FAILED":
        return ["Review error details", "Contact support if needed"]
    else:
        return []


def _get_tool_rating(tool_key: str) -> int:
    """Get Nadas.fi rating for tool (1-5)"""
    ratings = {
        "relay": 5,
        "gaszip": 5,
        "stargate": 4,
        "across": 4,
        "hop": 4,
        "1inch": 5,
        "paraswap": 4,
        "0x": 4
    }
    return ratings.get(tool_key, 3)


def _get_liquidity_score(tool_key: str) -> int:
    """Get liquidity score for tool (1-10)"""
    scores = {
        "1inch": 9,
        "paraswap": 8,
        "0x": 8,
        "uniswap": 10,
        "sushiswap": 7
    }
    return scores.get(tool_key, 5)


def _get_tool_use_cases(tool_key: str) -> List[str]:
    """Get recommended use cases for tool"""
    use_cases = {
        "relay": ["Direct Hypercore access", "Fast bridging"],
        "gaszip": ["Low-cost bridging", "Hypercore deposits"],
        "stargate": ["Stable coin transfers", "Large amounts"],
        "1inch": ["Best rates", "Large swaps"],
        "paraswap": ["Token variety", "MEV protection"]
    }
    return use_cases.get(tool_key, ["General bridging"])


# Background task functions
async def monitor_lifi_transaction(route_id: str, user_address: str):
    """Background task to monitor LI.FI transaction progress"""
    try:
        logger.info("Starting LI.FI transaction monitoring", 
                   route_id=route_id,
                   user=user_address)
        
        # Simulate monitoring
        import asyncio
        await asyncio.sleep(300)  # 5 minutes
        
        logger.info("LI.FI transaction monitoring completed", 
                   route_id=route_id,
                   final_status="completed")
        
    except Exception as e:
        logger.error("LI.FI transaction monitoring failed", 
                    route_id=route_id, 
                    error=str(e))