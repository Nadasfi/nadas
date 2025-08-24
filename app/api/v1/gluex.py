"""
GlueX Cross-Chain API Endpoints
$7k Bounty Implementation - Cross-chain deposits and portfolio management
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from datetime import datetime

from app.api.v1.auth import get_current_user
from app.adapters.gluex import GlueXAdapter, CrossChainDeposit, get_gluex_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class QuoteRequest(BaseModel):
    """Request for GlueX routing quote"""
    input_token: str = Field(..., description="Input token address or symbol")
    output_token: str = Field(..., description="Output token address or symbol")
    input_amount: str = Field(..., description="Amount to swap (in token units)")
    chain_id: str = Field("ethereum", description="Target blockchain")
    slippage_tolerance: float = Field(0.005, ge=0.001, le=0.1, description="Slippage tolerance (0.5% default)")
    is_permit2: bool = Field(False, description="Use Permit2 for approvals")


class CrossChainDepositRequest(BaseModel):
    """Request for cross-chain deposit"""
    source_chain: str = Field(..., description="Source blockchain")
    target_chain: str = Field(..., description="Target blockchain")
    source_token: str = Field(..., description="Source token address/symbol")
    target_token: str = Field(..., description="Target token address/symbol")
    amount: str = Field(..., description="Amount to deposit")
    slippage_tolerance: float = Field(0.005, description="Slippage tolerance")

    @validator('source_chain', 'target_chain')
    def validate_chains(cls, v):
        supported_chains = ["ethereum", "arbitrum", "polygon", "base"]
        if v not in supported_chains:
            raise ValueError(f"Chain {v} not supported. Supported: {supported_chains}")
        return v


class ExchangeRateRequest(BaseModel):
    """Request for exchange rates"""
    token_pairs: List[Dict[str, str]] = Field(..., description="List of token pairs")

    @validator('token_pairs')
    def validate_token_pairs(cls, v):
        required_fields = ["domestic_blockchain", "domestic_token", "foreign_blockchain", "foreign_token"]
        for pair in v:
            for field in required_fields:
                if field not in pair:
                    raise ValueError(f"Missing required field: {field}")
        return v


class QuoteResponse(BaseModel):
    """GlueX routing quote response"""
    quote_id: str
    input_token: str
    output_token: str
    input_amount: str
    output_amount: str
    chain_id: str
    gas_estimate: Optional[str]
    price_impact: Optional[float]
    route_description: str
    expires_at: datetime
    estimated_fee_usd: Optional[float] = None


class CrossChainDepositResponse(BaseModel):
    """Cross-chain deposit response"""
    success: bool
    quote_id: Optional[str] = None
    transaction_hash: Optional[str] = None
    source_chain: str
    target_chain: str
    input_amount: str
    expected_output: Optional[str] = None
    price_impact: Optional[float] = None
    gas_estimate: Optional[str] = None
    route_description: Optional[str] = None
    timestamp: datetime
    error: Optional[str] = None


class PortfolioSummaryResponse(BaseModel):
    """Cross-chain portfolio summary"""
    total_value_usd: float
    chains: Dict[str, Any]
    top_tokens: List[Dict[str, Any]]
    last_updated: datetime


class BridgeRouteResponse(BaseModel):
    """Bridge route information"""
    route_id: str
    source_chain: str
    target_chain: str
    token: str
    amount: str
    estimated_fee: str
    estimated_time_minutes: int
    bridge_provider: str
    security_level: str
    liquidity_available: bool


@router.post("/quote", response_model=QuoteResponse)
async def get_routing_quote(
    request: QuoteRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get optimal routing quote from GlueX"""
    try:
        user_address = current_user["wallet_address"]
        
        async with get_gluex_adapter() as gluex:
            quote = await gluex.get_quote(
                input_token=request.input_token,
                output_token=request.output_token,
                input_amount=request.input_amount,
                user_address=user_address,
                chain_id=request.chain_id,
                slippage_tolerance=request.slippage_tolerance,
                is_permit2=request.is_permit2
            )
        
        if not quote:
            raise HTTPException(status_code=400, detail="Failed to get routing quote")
        
        # Calculate estimated fee in USD (simplified)
        estimated_fee_usd = None
        if quote.gas_estimate:
            try:
                gas_cost_eth = float(quote.gas_estimate) * 20e-9  # 20 gwei
                estimated_fee_usd = gas_cost_eth * 3000  # ETH price approximation
            except:
                pass
        
        return QuoteResponse(
            quote_id=quote.quote_id,
            input_token=quote.input_token,
            output_token=quote.output_token,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            chain_id=quote.chain_id,
            gas_estimate=quote.gas_estimate,
            price_impact=quote.price_impact,
            route_description=quote.route_description,
            expires_at=quote.expires_at,
            estimated_fee_usd=estimated_fee_usd
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get routing quote", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get routing quote: {str(e)}")


@router.post("/cross-chain-deposit", response_model=CrossChainDepositResponse)
async def execute_cross_chain_deposit(
    request: CrossChainDepositRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Execute cross-chain deposit using GlueX Router"""
    try:
        user_address = current_user["wallet_address"]
        
        deposit_config = CrossChainDeposit(
            source_chain=request.source_chain,
            target_chain=request.target_chain,
            source_token=request.source_token,
            target_token=request.target_token,
            amount=request.amount,
            user_address=user_address,
            slippage_tolerance=request.slippage_tolerance
        )
        
        async with get_gluex_adapter() as gluex:
            result = await gluex.execute_cross_chain_deposit(deposit_config)
        
        return CrossChainDepositResponse(
            success=result["success"],
            quote_id=result.get("quote_id"),
            transaction_hash=result.get("transaction_hash"),
            source_chain=request.source_chain,
            target_chain=request.target_chain,
            input_amount=request.amount,
            expected_output=result.get("expected_output"),
            price_impact=result.get("price_impact"),
            gas_estimate=result.get("gas_estimate"),
            route_description=result.get("route_description"),
            timestamp=datetime.fromisoformat(result["timestamp"]),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error("Cross-chain deposit failed", error=str(e))
        return CrossChainDepositResponse(
            success=False,
            source_chain=request.source_chain,
            target_chain=request.target_chain,
            input_amount=request.amount,
            timestamp=datetime.utcnow(),
            error=str(e)
        )


@router.post("/exchange-rates")
async def get_exchange_rates(
    request: ExchangeRateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get real-time exchange rates for token pairs"""
    try:
        async with get_gluex_adapter() as gluex:
            rates = await gluex.get_exchange_rates(request.token_pairs)
        
        return {
            "success": True,
            "rates": rates,
            "timestamp": datetime.utcnow().isoformat(),
            "pairs_count": len(request.token_pairs)
        }
        
    except Exception as e:
        logger.error("Failed to get exchange rates", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get exchange rates: {str(e)}")


@router.get("/portfolio/cross-chain", response_model=PortfolioSummaryResponse)
async def get_cross_chain_portfolio(
    target_currency: str = "USD",
    current_user: dict = Depends(get_current_user)
):
    """Get portfolio value across all supported chains"""
    try:
        user_address = current_user["wallet_address"]
        
        async with get_gluex_adapter() as gluex:
            portfolio = await gluex.get_portfolio_value_across_chains(
                user_address=user_address,
                target_currency=target_currency
            )
        
        return PortfolioSummaryResponse(
            total_value_usd=portfolio["total_value_usd"],
            chains=portfolio["chains"],
            top_tokens=portfolio["top_tokens"],
            last_updated=datetime.fromisoformat(portfolio["last_updated"])
        )
        
    except Exception as e:
        logger.error("Failed to get cross-chain portfolio", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get cross-chain portfolio: {str(e)}")


@router.get("/chains/{chain_id}/tokens")
async def get_supported_tokens(
    chain_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get supported tokens for a specific chain"""
    try:
        async with get_gluex_adapter() as gluex:
            tokens = await gluex.get_supported_tokens(chain_id)
        
        return {
            "success": True,
            "chain_id": chain_id,
            "tokens": tokens,
            "count": len(tokens),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get supported tokens", chain_id=chain_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get supported tokens: {str(e)}")


@router.get("/bridge-routes", response_model=List[BridgeRouteResponse])
async def get_bridge_routes(
    source_chain: str,
    target_chain: str,
    token_symbol: str,
    amount: str,
    current_user: dict = Depends(get_current_user)
):
    """Find optimal bridge routes between chains"""
    try:
        async with get_gluex_adapter() as gluex:
            routes = await gluex.get_optimal_bridge_routes(
                source_chain=source_chain,
                target_chain=target_chain,
                token_symbol=token_symbol,
                amount=amount
            )
        
        return [
            BridgeRouteResponse(
                route_id=route["route_id"],
                source_chain=route["source_chain"],
                target_chain=route["target_chain"],
                token=route["token"],
                amount=route["amount"],
                estimated_fee=route["estimated_fee"],
                estimated_time_minutes=route["estimated_time_minutes"],
                bridge_provider=route["bridge_provider"],
                security_level=route["security_level"],
                liquidity_available=route["liquidity_available"]
            )
            for route in routes
        ]
        
    except Exception as e:
        logger.error("Failed to get bridge routes", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get bridge routes: {str(e)}")


@router.get("/chains")
async def get_supported_chains(current_user: dict = Depends(get_current_user)):
    """Get all supported blockchain networks"""
    try:
        async with get_gluex_adapter() as gluex:
            chains = gluex.supported_chains
        
        return {
            "success": True,
            "chains": [
                {
                    "chain_id": chain_id,
                    "name": info.name,
                    "native_token": info.native_token,
                    "explorer_url": info.explorer_url,
                    "supported_tokens_count": len(info.supported_tokens)
                }
                for chain_id, info in chains.items()
            ],
            "count": len(chains),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get supported chains", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get supported chains: {str(e)}")


@router.get("/status")
async def get_gluex_status(current_user: dict = Depends(get_current_user)):
    """Get GlueX integration status and health"""
    try:
        # Test basic connectivity
        async with get_gluex_adapter() as gluex:
            # Try to get supported tokens for Ethereum as health check
            tokens = await gluex.get_supported_tokens("ethereum")
            api_healthy = len(tokens) > 0
        
        return {
            "success": True,
            "status": "healthy" if api_healthy else "degraded",
            "api_connectivity": api_healthy,
            "supported_chains": 4,  # ethereum, arbitrum, polygon, base
            "features": {
                "cross_chain_deposits": True,
                "exchange_rates": True,
                "portfolio_aggregation": True,
                "bridge_routing": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("GlueX status check failed", error=str(e))
        return {
            "success": False,
            "status": "unhealthy",
            "api_connectivity": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
