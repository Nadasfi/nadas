"""
Cross-Chain Orchestrator API Endpoints
Unified API for cross-chain strategies with AI integration
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, validator
import structlog

from app.core.dependencies import get_current_user_address
from app.core.response import create_response
from app.services.cross_chain_orchestrator import get_cross_chain_orchestrator, CrossChainOrchestrator
from app.services.ai_strategy_generator import get_ai_strategy_generator, AIStrategyGenerator

logger = structlog.get_logger(__name__)
router = APIRouter()

# Request Models

class CreateStrategyRequest(BaseModel):
    """Request to create cross-chain strategy"""
    source_chain: str = Field(..., description="Source blockchain")
    target_chain: str = Field(default="hyperliquid", description="Target blockchain")
    source_token: str = Field(..., description="Source token symbol")
    target_token: str = Field(..., description="Target token symbol")
    amount: float = Field(..., gt=0, description="Amount to transfer")
    automation_rules: Optional[List[Dict[str, Any]]] = Field(default=None, description="Automation rules to setup after transfer")
    risk_tolerance: str = Field(default="medium", description="Risk tolerance level")
    
    @validator('source_chain', 'target_chain')
    def validate_chains(cls, v):
        supported_chains = ["ethereum", "polygon", "arbitrum", "optimism", "bsc", "avalanche", "hyperliquid"]
        if v.lower() not in supported_chains:
            raise ValueError(f"Unsupported chain: {v}")
        return v.lower()
    
    @validator('amount')
    def validate_amount(cls, v):
        if v > 1000000:  # $1M limit
            raise ValueError("Amount exceeds maximum limit of $1,000,000")
        return v

class AIGenerateStrategyRequest(BaseModel):
    """Request to generate strategy using AI"""
    user_input: str = Field(..., description="Natural language strategy request")
    portfolio_context: Optional[Dict[str, Any]] = Field(default=None, description="User's current portfolio")
    
    @validator('user_input')
    def validate_input(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Strategy request too short. Please provide more details.")
        if len(v) > 1000:
            raise ValueError("Strategy request too long. Please keep under 1000 characters.")
        return v.strip()

class SelectRouteRequest(BaseModel):
    """Request to select optimal route"""
    strategy_id: str = Field(..., description="Strategy ID")
    route_index: int = Field(default=0, description="Index of selected route")
    
    @validator('route_index')
    def validate_route_index(cls, v):
        if v < 0:
            raise ValueError("Route index cannot be negative")
        return v

# API Endpoints

@router.post("/create-strategy")
async def create_cross_chain_strategy(
    request: CreateStrategyRequest,
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Create new cross-chain strategy"""
    try:
        strategy_config = {
            "source_chain": request.source_chain,
            "target_chain": request.target_chain,
            "source_token": request.source_token,
            "target_token": request.target_token,
            "amount": request.amount,
            "automation_rules": request.automation_rules or [],
            "risk_tolerance": request.risk_tolerance
        }
        
        strategy = await orchestrator.create_strategy(user_address, strategy_config)
        
        logger.info("Cross-chain strategy created via API",
                   strategy_id=strategy.id,
                   user=user_address,
                   source_chain=request.source_chain,
                   target_chain=request.target_chain)
        
        return create_response(
            success=True,
            data={
                "strategy_id": strategy.id,
                "status": strategy.status.value,
                "config": strategy.strategy_config,
                "created_at": strategy.created_at.isoformat(),
                "progress_percentage": 0
            },
            message="Cross-chain strategy created successfully"
        )
        
    except Exception as e:
        logger.error("Failed to create cross-chain strategy",
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create strategy: {str(e)}"
        )

@router.post("/ai-generate-strategy")
async def ai_generate_strategy(
    request: AIGenerateStrategyRequest,
    user_address: str = Depends(get_current_user_address),
    ai_generator: AIStrategyGenerator = Depends(get_ai_strategy_generator),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Generate cross-chain strategy using AI from natural language"""
    try:
        # Generate strategy using AI
        ai_result = await ai_generator.generate_strategy_from_text(
            user_input=request.user_input,
            user_address=user_address,
            portfolio_context=request.portfolio_context
        )
        
        if not ai_result.get("success"):
            return create_response(
                success=False,
                error=ai_result.get("error", "AI strategy generation failed"),
                data={
                    "fallback_suggestions": ai_result.get("fallback_suggestions", [])
                }
            )
        
        # Create strategy from AI-generated plan
        strategy_plan = ai_result["strategy"]
        strategy_config = {
            "source_chain": strategy_plan["source_chain"],
            "target_chain": strategy_plan["target_chain"],
            "source_token": strategy_plan["source_token"],
            "target_token": strategy_plan["target_token"],
            "amount": strategy_plan["amount"],
            "automation_rules": strategy_plan.get("automation_rules", []),
            "risk_tolerance": ai_result["intent"].get("risk_tolerance", "medium"),
            "ai_generated": True
        }
        
        strategy = await orchestrator.create_strategy(user_address, strategy_config)
        
        logger.info("AI-generated cross-chain strategy created",
                   strategy_id=strategy.id,
                   user=user_address,
                   user_input=request.user_input[:100])
        
        return create_response(
            success=True,
            data={
                "strategy_id": strategy.id,
                "ai_analysis": ai_result,
                "strategy": strategy.to_dict(),
                "explanation": await ai_generator.explain_strategy(strategy.id, strategy_plan)
            },
            message="AI-generated strategy created successfully"
        )
        
    except Exception as e:
        logger.error("AI strategy generation failed",
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI strategy generation failed: {str(e)}"
        )

@router.post("/analyze-routes/{strategy_id}")
async def analyze_cross_chain_routes(
    strategy_id: str,
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Analyze available cross-chain routes for strategy"""
    try:
        strategy = orchestrator.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        if strategy.user_address != user_address:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - not your strategy"
            )
        
        routes = await orchestrator.analyze_routes(strategy_id)
        
        logger.info("Routes analyzed for strategy",
                   strategy_id=strategy_id,
                   user=user_address,
                   routes_found=len(routes))
        
        return create_response(
            success=True,
            data={
                "strategy_id": strategy_id,
                "routes": routes,
                "recommended_route": routes[0] if routes else None,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "strategy_status": strategy.status.value
            },
            message=f"Found {len(routes)} available routes"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Route analysis failed",
                    strategy_id=strategy_id,
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route analysis failed: {str(e)}"
        )

@router.post("/select-route")
async def select_optimal_route(
    request: SelectRouteRequest,
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Select optimal route for strategy execution"""
    try:
        strategy = orchestrator.get_strategy(request.strategy_id)
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        if strategy.user_address != user_address:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - not your strategy"
            )
        
        selected_route = await orchestrator.select_optimal_route(
            request.strategy_id, 
            request.route_index
        )
        
        logger.info("Route selected for strategy",
                   strategy_id=request.strategy_id,
                   user=user_address,
                   route_index=request.route_index,
                   provider=selected_route.get("provider"))
        
        return create_response(
            success=True,
            data={
                "strategy_id": request.strategy_id,
                "selected_route": selected_route,
                "estimated_completion": strategy.estimated_completion.isoformat() if strategy.estimated_completion else None,
                "total_fees_usd": strategy.total_fees_usd
            },
            message="Route selected successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Route selection failed",
                    strategy_id=request.strategy_id,
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route selection failed: {str(e)}"
        )

@router.post("/execute-strategy/{strategy_id}")
async def execute_cross_chain_strategy(
    strategy_id: str,
    background_tasks: BackgroundTasks,
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Execute cross-chain strategy"""
    try:
        strategy = orchestrator.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        if strategy.user_address != user_address:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - not your strategy"
            )
        
        execution_result = await orchestrator.execute_strategy(strategy_id)
        
        logger.info("Cross-chain strategy execution started",
                   strategy_id=strategy_id,
                   user=user_address,
                   bridge_tx_hash=execution_result.get("bridge_tx_hash"))
        
        return create_response(
            success=True,
            data=execution_result,
            message="Strategy execution started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Strategy execution failed",
                    strategy_id=strategy_id,
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy execution failed: {str(e)}"
        )

@router.get("/strategy/{strategy_id}")
async def get_strategy_status(
    strategy_id: str,
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Get strategy status and details"""
    try:
        strategy = orchestrator.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        if strategy.user_address != user_address:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - not your strategy"
            )
        
        return create_response(
            success=True,
            data=strategy.to_dict(),
            message="Strategy status retrieved"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get strategy status",
                    strategy_id=strategy_id,
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get strategy status: {str(e)}"
        )

@router.get("/track-bridge/{tx_hash}")
async def track_bridge_transaction(
    tx_hash: str,
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Track bridge transaction status"""
    try:
        # Find strategy by transaction hash
        user_strategies = orchestrator.get_user_strategies(user_address)
        target_strategy = None
        
        for strategy in user_strategies:
            for bridge_tx in strategy.bridge_transactions:
                if bridge_tx.get("tx_hash") == tx_hash:
                    target_strategy = strategy
                    break
            if target_strategy:
                break
        
        if not target_strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bridge transaction not found"
            )
        
        # Get bridge transaction details
        bridge_tx = None
        for tx in target_strategy.bridge_transactions:
            if tx.get("tx_hash") == tx_hash:
                bridge_tx = tx
                break
        
        return create_response(
            success=True,
            data={
                "tx_hash": tx_hash,
                "strategy_id": target_strategy.id,
                "bridge_details": bridge_tx,
                "strategy_status": target_strategy.status.value,
                "progress_percentage": target_strategy._calculate_progress(),
                "execution_log": target_strategy.execution_log[-5:],  # Last 5 entries
                "estimated_completion": target_strategy.estimated_completion.isoformat() if target_strategy.estimated_completion else None
            },
            message="Bridge transaction status retrieved"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Bridge tracking failed",
                    tx_hash=tx_hash,
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bridge tracking failed: {str(e)}"
        )

@router.get("/user-strategies")
async def get_user_strategies(
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Get all strategies for current user"""
    try:
        strategies = orchestrator.get_user_strategies(user_address)
        
        strategy_summaries = []
        for strategy in strategies:
            summary = {
                "id": strategy.id,
                "status": strategy.status.value,
                "source_chain": strategy.strategy_config.get("source_chain"),
                "target_chain": strategy.strategy_config.get("target_chain"),
                "source_token": strategy.strategy_config.get("source_token"),
                "target_token": strategy.strategy_config.get("target_token"),
                "amount": strategy.strategy_config.get("amount"),
                "created_at": strategy.created_at.isoformat(),
                "progress_percentage": strategy._calculate_progress(),
                "total_fees_usd": strategy.total_fees_usd,
                "ai_generated": strategy.strategy_config.get("ai_generated", False)
            }
            strategy_summaries.append(summary)
        
        # Sort by creation time (newest first)
        strategy_summaries.sort(key=lambda x: x["created_at"], reverse=True)
        
        return create_response(
            success=True,
            data={
                "strategies": strategy_summaries,
                "total_count": len(strategy_summaries),
                "active_count": len([s for s in strategy_summaries if s["status"] not in ["completed", "failed", "cancelled"]]),
                "completed_count": len([s for s in strategy_summaries if s["status"] == "completed"])
            },
            message=f"Found {len(strategy_summaries)} strategies"
        )
        
    except Exception as e:
        logger.error("Failed to get user strategies",
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user strategies: {str(e)}"
        )

@router.post("/cancel-strategy/{strategy_id}")
async def cancel_strategy(
    strategy_id: str,
    user_address: str = Depends(get_current_user_address),
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Cancel pending strategy"""
    try:
        strategy = orchestrator.get_strategy(strategy_id)
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        if strategy.user_address != user_address:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied - not your strategy"
            )
        
        cancelled = await orchestrator.cancel_strategy(strategy_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Strategy cannot be cancelled in current status"
            )
        
        logger.info("Strategy cancelled by user",
                   strategy_id=strategy_id,
                   user=user_address)
        
        return create_response(
            success=True,
            data={
                "strategy_id": strategy_id,
                "status": strategy.status.value
            },
            message="Strategy cancelled successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Strategy cancellation failed",
                    strategy_id=strategy_id,
                    user=user_address,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy cancellation failed: {str(e)}"
        )

@router.get("/statistics")
async def get_orchestrator_statistics(
    orchestrator: CrossChainOrchestrator = Depends(get_cross_chain_orchestrator)
):
    """Get orchestrator performance statistics (public endpoint)"""
    try:
        stats = orchestrator.get_execution_statistics()
        
        return create_response(
            success=True,
            data={
                **stats,
                "timestamp": datetime.utcnow().isoformat(),
                "service_status": "operational"
            },
            message="Orchestrator statistics retrieved"
        )
        
    except Exception as e:
        logger.error("Failed to get orchestrator statistics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )

@router.get("/health")
async def orchestrator_health_check():
    """Health check for orchestrator service"""
    try:
        orchestrator = get_cross_chain_orchestrator()
        ai_generator = get_ai_strategy_generator()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "orchestrator": "operational",
                "ai_generator": "operational" if ai_generator.ai_service else "degraded",
                "lifi_adapter": "operational" if orchestrator.lifi_adapter else "unavailable",
                "gluex_adapter": "operational" if orchestrator.gluex_adapter else "unavailable",
                "liquid_labs_adapter": "operational" if orchestrator.liquid_labs_adapter else "unavailable"
            },
            "active_strategies": len(orchestrator.active_strategies),
            "version": "1.0.0"
        }
        
        return create_response(
            success=True,
            data=health_data,
            message="Orchestrator service healthy"
        )
        
    except Exception as e:
        logger.error("Orchestrator health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service health check failed: {str(e)}"
        )