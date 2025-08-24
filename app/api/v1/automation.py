"""
Automation API Endpoints
DCA, Portfolio Rebalancing, Liquidation Monitoring - Real Hyperliquid SDK Implementation
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from app.api.v1.auth import get_current_user
from app.services.automation_engine import automation_engine, AutomationRuleType, AutomationStatus
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class RuleTypeEnum(str, Enum):
    DCA = "dca"
    REBALANCING = "rebalancing"
    LIQUIDATION_MONITOR = "liquidation_monitor"


class DCARequest(BaseModel):
    """DCA automation request"""
    symbol: str = Field(..., description="Trading symbol (e.g., ETH, BTC)")
    amount_usd: float = Field(..., gt=0, le=10000, description="USD amount per execution")
    interval_hours: int = Field(..., ge=1, le=168, description="Hours between executions")
    is_buy: bool = Field(True, description="True for buy orders, False for sell")
    max_executions: Optional[int] = Field(None, ge=1, description="Maximum number of executions")


class RebalancingRequest(BaseModel):
    """Portfolio rebalancing request"""
    target_allocation: Dict[str, float] = Field(..., description="Target allocation percentages")
    rebalance_threshold: float = Field(0.05, ge=0.01, le=0.5, description="Rebalancing threshold")
    check_interval_hours: int = Field(24, ge=1, le=168, description="Check interval in hours")
    
    @validator('target_allocation')
    def validate_allocation(cls, v):
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError('Target allocation must sum to 1.0')
        for symbol, percent in v.items():
            if not (0 < percent <= 1.0):
                raise ValueError(f'Allocation for {symbol} must be between 0 and 1.0')
        return v


class LiquidationMonitorRequest(BaseModel):
    """Liquidation monitoring request"""
    warning_threshold: float = Field(0.15, ge=0.05, le=0.5, description="Warning threshold from liquidation")
    emergency_threshold: float = Field(0.05, ge=0.01, le=0.2, description="Emergency threshold from liquidation")


class AutomationRuleResponse(BaseModel):
    """Automation rule response"""
    rule_id: str
    user_id: str
    wallet_address: str
    rule_type: str
    status: str
    config: Dict[str, Any]
    created_at: datetime
    last_executed: Optional[datetime]
    execution_count: int
    last_error: Optional[str]


class ExecutionHistoryResponse(BaseModel):
    """Execution history response"""
    rule_id: str
    type: str
    timestamp: str
    details: Dict[str, Any]


@router.post("/dca", response_model=AutomationRuleResponse)
async def create_dca_automation(
    request: DCARequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create Dollar Cost Averaging automation"""
    try:
        user_id = current_user["user_id"]
        wallet_address = current_user["wallet_address"]
        
        rule_id = await automation_engine.create_dca_rule(
            user_id=user_id,
            wallet_address=wallet_address,
            symbol=request.symbol,
            amount_usd=request.amount_usd,
            interval_hours=request.interval_hours,
            is_buy=request.is_buy,
            max_executions=request.max_executions
        )
        
        # Start automation engine if not running
        if not automation_engine.is_running:
            background_tasks.add_task(automation_engine.start_automation_engine)
        
        rule = automation_engine.get_rule_status(rule_id)
        
        return AutomationRuleResponse(
            rule_id=rule.rule_id,
            user_id=rule.user_id,
            wallet_address=rule.wallet_address,
            rule_type=rule.rule_type,
            status=rule.status,
            config=rule.config,
            created_at=rule.created_at,
            last_executed=rule.last_executed,
            execution_count=rule.execution_count,
            last_error=rule.last_error
        )
        
    except Exception as e:
        logger.error("Failed to create DCA automation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create DCA automation: {str(e)}")


@router.post("/rebalancing", response_model=AutomationRuleResponse)
async def create_rebalancing_automation(
    request: RebalancingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create portfolio rebalancing automation"""
    try:
        user_id = current_user["user_id"]
        wallet_address = current_user["wallet_address"]
        
        rule_id = await automation_engine.create_rebalancing_rule(
            user_id=user_id,
            wallet_address=wallet_address,
            target_allocation=request.target_allocation,
            rebalance_threshold=request.rebalance_threshold,
            check_interval_hours=request.check_interval_hours
        )
        
        # Start automation engine if not running
        if not automation_engine.is_running:
            background_tasks.add_task(automation_engine.start_automation_engine)
        
        rule = automation_engine.get_rule_status(rule_id)
        
        return AutomationRuleResponse(
            rule_id=rule.rule_id,
            user_id=rule.user_id,
            wallet_address=rule.wallet_address,
            rule_type=rule.rule_type,
            status=rule.status,
            config=rule.config,
            created_at=rule.created_at,
            last_executed=rule.last_executed,
            execution_count=rule.execution_count,
            last_error=rule.last_error
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create rebalancing automation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create rebalancing automation: {str(e)}")


@router.post("/liquidation-monitor", response_model=AutomationRuleResponse)
async def create_liquidation_monitor(
    request: LiquidationMonitorRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create liquidation monitoring automation"""
    try:
        user_id = current_user["user_id"]
        wallet_address = current_user["wallet_address"]
        
        rule_id = await automation_engine.create_liquidation_monitor(
            user_id=user_id,
            wallet_address=wallet_address,
            warning_threshold=request.warning_threshold,
            emergency_threshold=request.emergency_threshold
        )
        
        # Start automation engine if not running
        if not automation_engine.is_running:
            background_tasks.add_task(automation_engine.start_automation_engine)
        
        rule = automation_engine.get_rule_status(rule_id)
        
        return AutomationRuleResponse(
            rule_id=rule.rule_id,
            user_id=rule.user_id,
            wallet_address=rule.wallet_address,
            rule_type=rule.rule_type,
            status=rule.status,
            config=rule.config,
            created_at=rule.created_at,
            last_executed=rule.last_executed,
            execution_count=rule.execution_count,
            last_error=rule.last_error
        )
        
    except Exception as e:
        logger.error("Failed to create liquidation monitor", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create liquidation monitor: {str(e)}")


@router.get("/rules", response_model=List[AutomationRuleResponse])
async def get_automation_rules(current_user: dict = Depends(get_current_user)):
    """Get all automation rules for user"""
    try:
        user_id = current_user["user_id"]
        user_rules = []
        
        for rule in automation_engine.active_rules.values():
            if rule.user_id == user_id:
                user_rules.append(AutomationRuleResponse(
                    rule_id=rule.rule_id,
                    user_id=rule.user_id,
                    wallet_address=rule.wallet_address,
                    rule_type=rule.rule_type,
                    status=rule.status,
                    config=rule.config,
                    created_at=rule.created_at,
                    last_executed=rule.last_executed,
                    execution_count=rule.execution_count,
                    last_error=rule.last_error
                ))
        
        return sorted(user_rules, key=lambda x: x.created_at, reverse=True)
        
    except Exception as e:
        logger.error("Failed to get automation rules", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch automation rules")


@router.get("/rules/{rule_id}", response_model=AutomationRuleResponse)
async def get_automation_rule(
    rule_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific automation rule"""
    try:
        rule = automation_engine.get_rule_status(rule_id)
        
        if not rule:
            raise HTTPException(status_code=404, detail="Automation rule not found")
        
        # Verify ownership
        if rule.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to access this rule")
        
        return AutomationRuleResponse(
            rule_id=rule.rule_id,
            user_id=rule.user_id,
            wallet_address=rule.wallet_address,
            rule_type=rule.rule_type,
            status=rule.status,
            config=rule.config,
            created_at=rule.created_at,
            last_executed=rule.last_executed,
            execution_count=rule.execution_count,
            last_error=rule.last_error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get automation rule", rule_id=rule_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch automation rule")


@router.put("/rules/{rule_id}/pause")
async def pause_automation_rule(
    rule_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Pause automation rule"""
    try:
        rule = automation_engine.get_rule_status(rule_id)
    
    if not rule:
            raise HTTPException(status_code=404, detail="Automation rule not found")
        
        # Verify ownership
        if rule.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to modify this rule")
        
        success = automation_engine.pause_rule(rule_id)
        
        if success:
            return {"message": "Automation rule paused successfully", "rule_id": rule_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to pause automation rule")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pause automation rule", rule_id=rule_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to pause automation rule")


@router.put("/rules/{rule_id}/resume")
async def resume_automation_rule(
    rule_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Resume automation rule"""
    try:
        rule = automation_engine.get_rule_status(rule_id)
        
        if not rule:
            raise HTTPException(status_code=404, detail="Automation rule not found")
        
        # Verify ownership
        if rule.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to modify this rule")
        
        success = automation_engine.resume_rule(rule_id)
        
        if success:
            return {"message": "Automation rule resumed successfully", "rule_id": rule_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to resume automation rule")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume automation rule", rule_id=rule_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to resume automation rule")


@router.delete("/rules/{rule_id}")
async def delete_automation_rule(
    rule_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete automation rule"""
    try:
        rule = automation_engine.get_rule_status(rule_id)
        
        if not rule:
            raise HTTPException(status_code=404, detail="Automation rule not found")
        
        # Verify ownership
        if rule.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to delete this rule")
        
        success = automation_engine.delete_rule(rule_id)
        
        if success:
            return {"message": "Automation rule deleted successfully", "rule_id": rule_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete automation rule")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete automation rule", rule_id=rule_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete automation rule")


@router.get("/rules/{rule_id}/history", response_model=List[ExecutionHistoryResponse])
async def get_rule_execution_history(
    rule_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get execution history for specific rule"""
    try:
        rule = automation_engine.get_rule_status(rule_id)
        
        if not rule:
            raise HTTPException(status_code=404, detail="Automation rule not found")
        
        # Verify ownership
        if rule.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to access this rule")
        
        history = automation_engine.get_execution_history(rule_id)
        
        return [
            ExecutionHistoryResponse(
                rule_id=record["rule_id"],
                type=record["type"],
                timestamp=record["timestamp"],
                details={k: v for k, v in record.items() if k not in ["rule_id", "type", "timestamp"]}
            )
            for record in history[-50:]  # Last 50 executions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get execution history", rule_id=rule_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch execution history")


@router.get("/engine/status")
async def get_engine_status(current_user: dict = Depends(get_current_user)):
    """Get automation engine status"""
    try:
        total_rules = len(automation_engine.active_rules)
        active_rules = sum(1 for rule in automation_engine.active_rules.values() 
                          if rule.status == AutomationStatus.ACTIVE)
        
        return {
            "engine_running": automation_engine.is_running,
            "total_rules": total_rules,
            "active_rules": active_rules,
            "total_executions": len(automation_engine.execution_history),
            "last_cycle": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get engine status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get engine status")