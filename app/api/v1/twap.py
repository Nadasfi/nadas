"""
TWAP Order Executor API Endpoints
$5,000 Hackathon Bounty Implementation

Provides REST API for shielded TWAP execution with privacy-preserving features.
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime
import logging

from app.services.twap_executor import (
    TWAPOrderExecutor, 
    TWAPConfig, 
    OrderSide, 
    TWAPStrategy, 
    PrivacyLevel,
    trend_signal_callback
)
from app.api.v1.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()

# Global TWAP executor instance
twap_executor = TWAPOrderExecutor()

class TWAPRequest(BaseModel):
    total_amount: float = Field(..., gt=0, description="Total amount to execute")
    symbol: str = Field(..., description="Trading symbol (e.g., 'ETH-PERP')")
    side: str = Field(..., description="Order side: 'buy' or 'sell'")
    duration_minutes: int = Field(..., ge=5, le=1440, description="Execution duration in minutes (5-1440)")
    max_slippage: float = Field(default=0.5, ge=0, le=10, description="Maximum slippage percentage")
    privacy_level: int = Field(default=2, ge=1, le=4, description="Privacy level (1-4, higher is more private)")
    strategy: str = Field(default="equal_interval", description="TWAP strategy")
    min_order_size: Optional[float] = Field(None, gt=0, description="Minimum order size")
    max_order_size: Optional[float] = Field(None, gt=0, description="Maximum order size")
    randomness_factor: float = Field(default=0.1, ge=0, le=0.5, description="Randomness factor (0-0.5)")
    pause_threshold: float = Field(default=0.5, ge=0, le=5, description="Pause threshold for spread %")
    cross_chain: bool = Field(default=False, description="Enable cross-chain execution")
    target_chain: Optional[str] = Field(None, description="Target chain for cross-chain execution")
    
    @validator('side')
    def validate_side(cls, v):
        if v.lower() not in ['buy', 'sell']:
            raise ValueError('Side must be "buy" or "sell"')
        return v.lower()
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['equal_interval', 'volume_weighted', 'adaptive', 'hybrid']
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v
    
    @validator('max_order_size')
    def validate_order_sizes(cls, v, values):
        if v is not None and 'min_order_size' in values and values['min_order_size'] is not None:
            if v <= values['min_order_size']:
                raise ValueError('max_order_size must be greater than min_order_size')
        return v

class TWAPConfigUpdate(BaseModel):
    max_slippage: Optional[float] = Field(None, ge=0, le=10)
    pause_threshold: Optional[float] = Field(None, ge=0, le=5)
    randomness_factor: Optional[float] = Field(None, ge=0, le=0.5)

class SignalConfig(BaseModel):
    enable_trend_signals: bool = Field(default=False, description="Enable trend-based order adjustments")
    trend_lookback_minutes: int = Field(default=15, ge=5, le=60, description="Trend analysis lookback period")
    trend_threshold: float = Field(default=0.02, ge=0.01, le=0.1, description="Trend detection threshold")

@router.post("/create", response_model=Dict[str, Any])
async def create_twap_execution(
    request: TWAPRequest,
    signal_config: Optional[SignalConfig] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new TWAP execution plan.
    
    This creates the execution schedule but does not start execution.
    Use /execute/{execution_id} to start execution.
    """
    try:
        # Convert request to TWAPConfig
        config = TWAPConfig(
            total_amount=Decimal(str(request.total_amount)),
            symbol=request.symbol,
            side=OrderSide(request.side),
            duration_minutes=request.duration_minutes,
            max_slippage=Decimal(str(request.max_slippage)),
            privacy_level=PrivacyLevel(request.privacy_level),
            strategy=TWAPStrategy(request.strategy),
            min_order_size=Decimal(str(request.min_order_size)) if request.min_order_size else None,
            max_order_size=Decimal(str(request.max_order_size)) if request.max_order_size else None,
            randomness_factor=Decimal(str(request.randomness_factor)),
            pause_threshold=Decimal(str(request.pause_threshold)),
            cross_chain=request.cross_chain,
            target_chain=request.target_chain
        )
        
        # Add signal callbacks if enabled
        if signal_config and signal_config.enable_trend_signals:
            twap_executor.add_signal_callback(trend_signal_callback)
        
        execution_id = await twap_executor.create_twap_execution(config)
        
        # Get initial status
        status_info = await twap_executor.get_execution_status(execution_id)
        
        return {
            "success": True,
            "execution_id": execution_id,
            "message": "TWAP execution plan created successfully",
            "status": status_info,
            "estimated_orders": status_info["progress"]["total_orders"],
            "privacy_features": {
                "timing_randomization": request.privacy_level >= 2,
                "size_randomization": request.privacy_level >= 2,
                "wallet_rotation": request.privacy_level >= 3,
                "maximum_privacy": request.privacy_level == 4
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create TWAP execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create TWAP execution: {str(e)}"
        )

@router.post("/execute/{execution_id}", response_model=Dict[str, Any])
async def execute_twap(
    execution_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Start executing a TWAP plan.
    
    Execution runs in the background according to the configured schedule.
    """
    try:
        # Verify execution exists
        status_info = await twap_executor.get_execution_status(execution_id)
        if "error" in status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TWAP execution not found"
            )
        
        if status_info["status"] != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot execute TWAP in status: {status_info['status']}"
            )
        
        # Start execution in background
        background_tasks.add_task(twap_executor.execute_twap, execution_id)
        
        return {
            "success": True,
            "execution_id": execution_id,
            "message": "TWAP execution started",
            "status": "executing",
            "estimated_completion": status_info.get("estimated_completion"),
            "total_orders": status_info["progress"]["total_orders"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start TWAP execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start TWAP execution: {str(e)}"
        )

@router.get("/status/{execution_id}", response_model=Dict[str, Any])
async def get_twap_status(
    execution_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get current status and progress of a TWAP execution.
    
    Returns detailed information about execution progress, performance metrics,
    and individual order status.
    """
    try:
        status_info = await twap_executor.get_execution_status(execution_id)
        
        if "error" in status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TWAP execution not found"
            )
        
        # Calculate additional metrics
        progress = status_info["progress"]
        completion_percentage = (
            progress["executed_orders"] / progress["total_orders"] * 100
            if progress["total_orders"] > 0 else 0
        )
        
        return {
            "success": True,
            "execution_id": execution_id,
            "status": status_info["status"],
            "progress": {
                **progress,
                "completion_percentage": round(completion_percentage, 2),
                "execution_rate": f"{progress['executed_orders']}/{progress['total_orders']}"
            },
            "performance": status_info["performance"],
            "config": status_info["config"],
            "timing": {
                "start_time": status_info["start_time"],
                "end_time": status_info["end_time"],
                "duration_so_far": None  # Calculate if needed
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get TWAP status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get TWAP status: {str(e)}"
        )

@router.post("/pause/{execution_id}", response_model=Dict[str, Any])
async def pause_twap_execution(
    execution_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Pause an active TWAP execution.
    
    Stops scheduling new orders but does not cancel already submitted orders.
    """
    try:
        success = await twap_executor.pause_execution(execution_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot pause execution (not found or not active)"
            )
        
        return {
            "success": True,
            "execution_id": execution_id,
            "message": "TWAP execution paused",
            "status": "paused"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause TWAP execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause TWAP execution: {str(e)}"
        )

@router.post("/resume/{execution_id}", response_model=Dict[str, Any])
async def resume_twap_execution(
    execution_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Resume a paused TWAP execution.
    
    Continues execution from where it was paused.
    """
    try:
        success = await twap_executor.resume_execution(execution_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot resume execution (not found or not paused)"
            )
        
        return {
            "success": True,
            "execution_id": execution_id,
            "message": "TWAP execution resumed",
            "status": "active"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume TWAP execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume TWAP execution: {str(e)}"
        )

@router.post("/cancel/{execution_id}", response_model=Dict[str, Any])
async def cancel_twap_execution(
    execution_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Cancel an active TWAP execution.
    
    Stops all future orders but does not cancel already submitted orders.
    """
    try:
        success = await twap_executor.cancel_execution(execution_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel execution (not found or not active)"
            )
        
        return {
            "success": True,
            "execution_id": execution_id,
            "message": "TWAP execution cancelled",
            "status": "cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel TWAP execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel TWAP execution: {str(e)}"
        )

@router.put("/config/{execution_id}", response_model=Dict[str, Any])
async def update_twap_config(
    execution_id: str,
    config_update: TWAPConfigUpdate,
    current_user: User = Depends(get_current_user)
):
    """
    Update configuration of an active TWAP execution.
    
    Only certain parameters can be updated while execution is running.
    """
    try:
        # Get current execution
        if execution_id not in twap_executor.active_executions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TWAP execution not found"
            )
        
        execution = twap_executor.active_executions[execution_id]
        
        # Update allowed parameters
        if config_update.max_slippage is not None:
            execution.config.max_slippage = Decimal(str(config_update.max_slippage))
        
        if config_update.pause_threshold is not None:
            execution.config.pause_threshold = Decimal(str(config_update.pause_threshold))
        
        if config_update.randomness_factor is not None:
            execution.config.randomness_factor = Decimal(str(config_update.randomness_factor))
        
        return {
            "success": True,
            "execution_id": execution_id,
            "message": "TWAP configuration updated",
            "updated_config": {
                "max_slippage": str(execution.config.max_slippage),
                "pause_threshold": str(execution.config.pause_threshold),
                "randomness_factor": str(execution.config.randomness_factor)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update TWAP config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update TWAP config: {str(e)}"
        )

@router.get("/list", response_model=Dict[str, Any])
async def list_twap_executions(
    status_filter: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    List TWAP executions for the current user.
    
    Optionally filter by status and limit results.
    """
    try:
        executions = []
        
        for execution_id, execution in twap_executor.active_executions.items():
            if status_filter and execution.status != status_filter:
                continue
            
            status_info = await twap_executor.get_execution_status(execution_id)
            executions.append({
                "execution_id": execution_id,
                "symbol": execution.config.symbol,
                "side": execution.config.side.value,
                "total_amount": str(execution.config.total_amount),
                "status": execution.status,
                "progress": status_info["progress"]["completion_percentage"] if "progress" in status_info else 0,
                "start_time": execution.start_time.isoformat(),
                "end_time": execution.end_time.isoformat() if execution.end_time else None
            })
            
            if len(executions) >= limit:
                break
        
        return {
            "success": True,
            "executions": executions,
            "total_count": len(executions),
            "applied_filters": {
                "status": status_filter,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list TWAP executions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list TWAP executions: {str(e)}"
        )

@router.get("/strategies", response_model=Dict[str, Any])
async def get_twap_strategies():
    """
    Get information about available TWAP strategies.
    
    Returns details about each strategy and when to use them.
    """
    return {
        "success": True,
        "strategies": {
            "equal_interval": {
                "name": "Equal Interval",
                "description": "Orders spaced equally in time with consistent sizing",
                "best_for": "Stable markets with predictable volume",
                "privacy_features": ["timing_randomization", "size_randomization"]
            },
            "volume_weighted": {
                "name": "Volume Weighted (VWAP-style)",
                "description": "Order sizes weighted by historical volume patterns",
                "best_for": "Markets with clear volume patterns",
                "privacy_features": ["timing_randomization", "size_randomization", "volume_adaptation"]
            },
            "adaptive": {
                "name": "Adaptive",
                "description": "Adapts order sizing to time-of-day and market conditions",
                "best_for": "Markets with clear daily patterns",
                "privacy_features": ["market_adaptation", "timing_randomization", "size_randomization"]
            },
            "hybrid": {
                "name": "Hybrid TWAP/VWAP",
                "description": "Combines time and volume weighting for optimal execution",
                "best_for": "Most market conditions, balanced approach",
                "privacy_features": ["timing_randomization", "size_randomization", "volume_adaptation", "market_adaptation"]
            }
        },
        "privacy_levels": {
            "1": {"name": "Low", "features": ["basic_execution"]},
            "2": {"name": "Medium", "features": ["timing_randomization", "size_randomization"]},
            "3": {"name": "High", "features": ["timing_randomization", "size_randomization", "wallet_rotation"]},
            "4": {"name": "Maximum", "features": ["timing_randomization", "size_randomization", "wallet_rotation", "maximum_privacy_delays"]}
        }
    }

@router.get("/health", response_model=Dict[str, Any])
async def twap_health_check():
    """
    Check health of TWAP execution service.
    
    Returns status of market data connections and execution engine.
    """
    try:
        # Check market data connectivity
        market_health = True
        try:
            await twap_executor.market_data.get_current_price("ETH-PERP")
        except Exception:
            market_health = False
        
        # Check active executions
        active_count = len([
            ex for ex in twap_executor.active_executions.values() 
            if ex.status == "active"
        ])
        
        return {
            "success": True,
            "status": "healthy" if market_health else "degraded",
            "market_data_connected": market_health,
            "active_executions": active_count,
            "total_executions": len(twap_executor.active_executions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"TWAP health check failed: {str(e)}")
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }