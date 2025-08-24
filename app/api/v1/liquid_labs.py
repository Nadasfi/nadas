"""
Liquid Labs API Endpoints
$7k Bounty Implementation - LiquidSwap DEX Aggregator + LiquidLaunch Token Platform
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from datetime import datetime

from app.api.v1.auth import get_current_user
from app.adapters.liquid_labs import (
    LiquidLabsAdapter, 
    SwapQuoteRequest, 
    get_liquid_labs_adapter
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SwapRequest(BaseModel):
    """LiquidSwap routing request"""
    token_in: str = Field(..., description="Input token address")
    token_out: str = Field(..., description="Output token address")
    amount_in: Optional[str] = Field(None, description="Input amount (for exact input)")
    amount_out: Optional[str] = Field(None, description="Output amount (for exact output)")
    multi_hop: bool = Field(True, description="Enable multi-hop routing")
    slippage: float = Field(1.0, ge=0.1, le=50.0, description="Slippage tolerance percentage")
    unwrap_whype: bool = Field(False, description="Unwrap WHYPE to native HYPE")
    exclude_dexes: Optional[str] = Field(None, description="Comma-separated DEX indices to exclude")
    fee_bps: int = Field(0, ge=0, le=100, description="Fee in basis points (max 1%)")
    fee_recipient: Optional[str] = Field(None, description="Fee recipient address")

    @validator('amount_in', 'amount_out')
    def validate_amounts(cls, v, values):
        amount_in = values.get('amount_in')
        amount_out = v if 'amount_out' in cls.__fields__ else values.get('amount_out')
        
        # Exactly one of amount_in or amount_out must be provided
        if bool(amount_in) == bool(amount_out):
            raise ValueError('Exactly one of amount_in or amount_out must be provided')
        return v


class SwapExecutionRequest(BaseModel):
    """Swap execution request"""
    swap_route_id: str = Field(..., description="Route ID from previous quote")
    user_address: str = Field(..., description="User wallet address")


class TokenCreationRequest(BaseModel):
    """LiquidLaunch token creation request"""
    name: str = Field(..., min_length=1, max_length=100, description="Token name")
    symbol: str = Field(..., min_length=1, max_length=20, description="Token symbol")
    description: str = Field(..., min_length=1, max_length=1000, description="Token description")
    image_url: str = Field(..., description="Token image URL")
    initial_buy_hype: float = Field(0.0, ge=0.0, le=100.0, description="Initial HYPE to buy tokens")

    @validator('symbol')
    def validate_symbol(cls, v):
        if not v.isupper():
            raise ValueError('Symbol must be uppercase')
        return v


class BondingCurvePurchaseRequest(BaseModel):
    """Bonding curve token purchase request"""
    token_address: str = Field(..., description="Token contract address")
    hype_amount: float = Field(..., gt=0, le=1000, description="Amount of HYPE to spend")
    slippage: float = Field(1.0, ge=0.1, le=50.0, description="Slippage tolerance percentage")


class SwapRouteResponse(BaseModel):
    """LiquidSwap routing response"""
    success: bool
    tokens: Dict[str, Any]
    amount_in: str
    amount_out: str
    average_price_impact: str
    execution_data: Optional[Dict[str, Any]]
    dex_breakdown: List[str]
    route_id: str
    expires_at: datetime


class TokenCreationResponse(BaseModel):
    """Token creation response"""
    success: bool
    token_address: Optional[str] = None
    transaction_hash: Optional[str] = None
    name: str
    symbol: str
    total_supply: str
    bonding_curve_address: str
    initial_price: str
    created_at: datetime
    error: Optional[str] = None


class BondingCurveInfoResponse(BaseModel):
    """Bonding curve information response"""
    token_address: str
    virtual_hype_liquidity: str
    tokens_sold: str
    tokens_remaining: str
    current_price: str
    market_cap: str
    bonding_complete: bool
    progress_percentage: float
    creator_fees_earned: str
    total_volume: str


class DEXStatsResponse(BaseModel):
    """DEX aggregator statistics"""
    total_dexes: int
    total_volume_24h: str
    total_trades_24h: int
    total_liquidity: str
    top_dexes_by_volume: List[Dict[str, str]]
    supported_tokens: int
    active_pools: int
    last_updated: datetime


@router.post("/swap/quote", response_model=SwapRouteResponse)
async def get_swap_quote(
    request: SwapRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get optimal swap route from LiquidSwap aggregator"""
    try:
        swap_request = SwapQuoteRequest(
            token_in=request.token_in,
            token_out=request.token_out,
            amount_in=request.amount_in,
            amount_out=request.amount_out,
            multi_hop=request.multi_hop,
            slippage=request.slippage,
            unwrap_whype=request.unwrap_whype,
            exclude_dexes=request.exclude_dexes,
            fee_bps=request.fee_bps,
            fee_recipient=request.fee_recipient
        )
        
        async with get_liquid_labs_adapter() as liquid_labs:
            route = await liquid_labs.get_swap_route(swap_request)
        
        if not route or not route.success:
            raise HTTPException(status_code=400, detail="Failed to get swap route")
        
        # Extract DEX information
        dex_breakdown = liquid_labs._extract_dex_info(route)
        route_id = f"route_{datetime.utcnow().timestamp()}"
        
        return SwapRouteResponse(
            success=route.success,
            tokens=route.tokens,
            amount_in=route.amount_in,
            amount_out=route.amount_out,
            average_price_impact=route.average_price_impact,
            execution_data=route.execution,
            dex_breakdown=dex_breakdown,
            route_id=route_id,
            expires_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get swap quote", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get swap quote: {str(e)}")


@router.post("/swap/execute")
async def execute_swap(
    request: SwapExecutionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute swap using LiquidSwap MultiHopRouter"""
    try:
        # In production, this would validate the route_id and execute the transaction
        # For now, return simulation result
        user_address = current_user["wallet_address"]
        
        result = {
            "success": True,
            "transaction_hash": f"0x{request.swap_route_id[-40:]}",
            "user_address": user_address,
            "executed_at": datetime.utcnow().isoformat(),
            "status": "simulated"
        }
        
        logger.info("Swap execution simulated", 
                   route_id=request.swap_route_id,
                   user=user_address)
        
        return result
        
    except Exception as e:
        logger.error("Swap execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Swap execution failed: {str(e)}")


@router.get("/tokens")
async def get_token_list(current_user: dict = Depends(get_current_user)):
    """Get list of available tokens on LiquidSwap"""
    try:
        async with get_liquid_labs_adapter() as liquid_labs:
            tokens = await liquid_labs.get_token_list()
        
        return {
            "success": True,
            "tokens": [
                {
                    "address": token.address,
                    "symbol": token.symbol,
                    "name": token.name,
                    "decimals": token.decimals
                }
                for token in tokens
            ],
            "count": len(tokens),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get token list", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get token list: {str(e)}")


@router.get("/tokens/{token_address}/pools")
async def get_token_pools(
    token_address: str,
    current_user: dict = Depends(get_current_user)
):
    """Get available pools for a specific token"""
    try:
        async with get_liquid_labs_adapter() as liquid_labs:
            pools = await liquid_labs.get_token_pools(token_address)
        
        return {
            "success": True,
            "token_address": token_address,
            "pools": pools,
            "count": len(pools),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get token pools", token=token_address, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get token pools: {str(e)}")


@router.get("/balances/{wallet_address}")
async def get_wallet_balances(
    wallet_address: str,
    current_user: dict = Depends(get_current_user)
):
    """Get token balances for a wallet address"""
    try:
        async with get_liquid_labs_adapter() as liquid_labs:
            balances = await liquid_labs.get_token_balances(wallet_address)
        
        return {
            "success": True,
            "wallet_address": wallet_address,
            "balances": balances,
            "tokens_count": len(balances),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get wallet balances", wallet=wallet_address, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get wallet balances: {str(e)}")


# LiquidLaunch Endpoints
@router.post("/launch/create-token", response_model=TokenCreationResponse)
async def create_token(
    request: TokenCreationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a new token via LiquidLaunch"""
    try:
        creator_address = current_user["wallet_address"]
        
        async with get_liquid_labs_adapter() as liquid_labs:
            result = await liquid_labs.create_token(
                name=request.name,
                symbol=request.symbol,
                description=request.description,
                image_url=request.image_url,
                creator_address=creator_address,
                initial_buy_hype=request.initial_buy_hype
            )
        
        return TokenCreationResponse(
            success=result["success"],
            token_address=result.get("token_address"),
            transaction_hash=result.get("transaction_hash"),
            name=result["name"],
            symbol=result["symbol"],
            total_supply=result["total_supply"],
            bonding_curve_address=result["bonding_curve_address"],
            initial_price=result["initial_price"],
            created_at=datetime.fromisoformat(result["created_at"]),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error("Token creation failed", error=str(e))
        return TokenCreationResponse(
            success=False,
            name=request.name,
            symbol=request.symbol,
            total_supply="0",
            bonding_curve_address="",
            initial_price="0",
            created_at=datetime.utcnow(),
            error=str(e)
        )


@router.get("/launch/tokens")
async def get_liquidlaunch_tokens(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get list of tokens created via LiquidLaunch"""
    try:
        async with get_liquid_labs_adapter() as liquid_labs:
            tokens = await liquid_labs.get_liquidlaunch_tokens(limit)
        
        return {
            "success": True,
            "tokens": [
                {
                    "address": token.address,
                    "symbol": token.symbol,
                    "name": token.name,
                    "creator": token.creator,
                    "created_at": token.created_at.isoformat(),
                    "total_supply": token.total_supply,
                    "current_price": token.current_price,
                    "market_cap": token.market_cap,
                    "bonding_complete": token.bonding_complete
                }
                for token in tokens
            ],
            "count": len(tokens),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get LiquidLaunch tokens", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get LiquidLaunch tokens: {str(e)}")


@router.get("/launch/tokens/{token_address}", response_model=BondingCurveInfoResponse)
async def get_bonding_curve_info(
    token_address: str,
    current_user: dict = Depends(get_current_user)
):
    """Get bonding curve information for a token"""
    try:
        async with get_liquid_labs_adapter() as liquid_labs:
            info = await liquid_labs.get_bonding_curve_info(token_address)
        
        if not info:
            raise HTTPException(status_code=404, detail="Token not found or not a LiquidLaunch token")
        
        return BondingCurveInfoResponse(
            token_address=info["token_address"],
            virtual_hype_liquidity=info["virtual_hype_liquidity"],
            tokens_sold=info["tokens_sold"],
            tokens_remaining=info["tokens_remaining"],
            current_price=info["current_price"],
            market_cap=info["market_cap"],
            bonding_complete=info["bonding_complete"],
            progress_percentage=info["progress_percentage"],
            creator_fees_earned=info["creator_fees_earned"],
            total_volume=info["total_volume"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get bonding curve info", token=token_address, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get bonding curve info: {str(e)}")


@router.post("/launch/buy")
async def buy_token_bonding_curve(
    request: BondingCurvePurchaseRequest,
    current_user: dict = Depends(get_current_user)
):
    """Buy tokens from bonding curve"""
    try:
        buyer_address = current_user["wallet_address"]
        
        async with get_liquid_labs_adapter() as liquid_labs:
            result = await liquid_labs.buy_token_bonding_curve(
                token_address=request.token_address,
                hype_amount=request.hype_amount,
                buyer_address=buyer_address,
                slippage=request.slippage
            )
        
        return result
        
    except Exception as e:
        logger.error("Bonding curve purchase failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Bonding curve purchase failed: {str(e)}")


@router.get("/dex/stats", response_model=DEXStatsResponse)
async def get_dex_stats(current_user: dict = Depends(get_current_user)):
    """Get aggregated DEX statistics"""
    try:
        async with get_liquid_labs_adapter() as liquid_labs:
            stats = await liquid_labs.get_dex_stats()
        
        return DEXStatsResponse(
            total_dexes=stats["total_dexes"],
            total_volume_24h=stats["total_volume_24h"],
            total_trades_24h=stats["total_trades_24h"],
            total_liquidity=stats["total_liquidity"],
            top_dexes_by_volume=stats["top_dexes_by_volume"],
            supported_tokens=stats["supported_tokens"],
            active_pools=stats["active_pools"],
            last_updated=datetime.fromisoformat(stats["last_updated"])
        )
        
    except Exception as e:
        logger.error("Failed to get DEX stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get DEX stats: {str(e)}")


@router.get("/dex/supported")
async def get_supported_dexes(current_user: dict = Depends(get_current_user)):
    """Get list of supported DEXes on HyperEVM"""
    try:
        async with get_liquid_labs_adapter() as liquid_labs:
            dexes = liquid_labs.supported_dexes
        
        return {
            "success": True,
            "dexes": [
                {
                    "index": dex.index,
                    "name": dex.name,
                    "router_name": dex.router_name
                }
                for dex in dexes.values()
            ],
            "count": len(dexes),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get supported DEXes", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get supported DEXes: {str(e)}")


@router.get("/status")
async def get_liquid_labs_status(current_user: dict = Depends(get_current_user)):
    """Get Liquid Labs integration status and health"""
    try:
        # Test basic connectivity
        async with get_liquid_labs_adapter() as liquid_labs:
            tokens = await liquid_labs.get_token_list()
            stats = await liquid_labs.get_dex_stats()
            api_healthy = len(tokens) >= 0 and bool(stats)
        
        return {
            "success": True,
            "status": "healthy" if api_healthy else "degraded",
            "api_connectivity": api_healthy,
            "supported_dexes": 14,  # KittenSwap, HyperSwap, Laminar, etc.
            "features": {
                "liquidswap_aggregation": True,
                "multi_hop_routing": True,
                "liquidlaunch_tokens": True,
                "bonding_curves": True,
                "revenue_sharing": True
            },
            "endpoints": {
                "api_base": "https://api.liqd.ag",
                "multihop_router": "0x744489ee3d540777a66f2cf297479745e0852f7a",
                "liquidlaunch_contract": "0xDEC3540f5BA6f2aa3764583A9c29501FeB020030"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Liquid Labs status check failed", error=str(e))
        return {
            "success": False,
            "status": "unhealthy",
            "api_connectivity": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
