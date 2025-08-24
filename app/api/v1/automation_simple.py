"""
Simple Notification-Based Automation API
Non-custodial approach: Backend monitors conditions, frontend executes trades
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from app.api.v1.auth import get_current_user
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class AutomationRuleType(str, Enum):
    DCA = "dca"
    STOP_LOSS = "stop_loss" 
    TAKE_PROFIT = "take_profit"
    REBALANCING = "rebalancing"
    PRICE_ALERT = "price_alert"


class AutomationStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TRIGGERED = "triggered"
    COMPLETED = "completed"


class CreateAutomationRuleRequest(BaseModel):
    """Create automation rule request"""
    rule_type: AutomationRuleType
    symbol: str = Field(..., description="Trading symbol")
    user_address: str = Field(..., description="User wallet address")
    
    # DCA Config
    dca_amount_usd: Optional[float] = Field(None, description="DCA amount in USD")
    dca_interval_hours: Optional[int] = Field(None, description="DCA interval in hours")
    
    # Stop Loss / Take Profit Config
    trigger_price: Optional[float] = Field(None, description="Trigger price")
    close_percentage: Optional[float] = Field(100.0, description="Percentage of position to close")
    
    # Price Alert Config
    alert_price: Optional[float] = Field(None, description="Alert price")
    alert_condition: Optional[str] = Field("above", description="Alert when price is above/below")
    
    # Rebalancing Config
    target_allocation: Optional[Dict[str, float]] = Field(None, description="Target allocation percentages")


class AutomationRuleResponse(BaseModel):
    """Automation rule response"""
    rule_id: str
    rule_type: str
    symbol: str
    user_address: str
    status: str
    config: Dict[str, Any]
    created_at: datetime
    last_checked: Optional[datetime] = None
    trigger_count: int = 0
    next_check: Optional[datetime] = None


class AutomationTrigger(BaseModel):
    """Automation trigger notification"""
    rule_id: str
    rule_type: str
    symbol: str
    trigger_reason: str
    current_price: float
    recommended_action: Dict[str, Any]
    triggered_at: datetime
    user_needs_to_act: bool = True


@router.post("/rules", response_model=AutomationRuleResponse)
async def create_automation_rule(
    request: CreateAutomationRuleRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new automation rule"""
    try:
        # Generate rule ID
        rule_id = f"{request.rule_type}_{request.symbol}_{int(datetime.utcnow().timestamp())}"
        
        # Build config based on rule type
        config = {}
        
        if request.rule_type == AutomationRuleType.DCA:
            if not request.dca_amount_usd or not request.dca_interval_hours:
                raise HTTPException(400, "DCA rules require amount_usd and interval_hours")
            config = {
                "amount_usd": request.dca_amount_usd,
                "interval_hours": request.dca_interval_hours,
                "next_execution": datetime.utcnow()
            }
            
        elif request.rule_type in [AutomationRuleType.STOP_LOSS, AutomationRuleType.TAKE_PROFIT]:
            if not request.trigger_price:
                raise HTTPException(400, f"{request.rule_type} rules require trigger_price")
            config = {
                "trigger_price": request.trigger_price,
                "close_percentage": request.close_percentage or 100.0,
                "condition": "below" if request.rule_type == AutomationRuleType.STOP_LOSS else "above"
            }
            
        elif request.rule_type == AutomationRuleType.PRICE_ALERT:
            if not request.alert_price:
                raise HTTPException(400, "Price alert rules require alert_price")
            config = {
                "alert_price": request.alert_price,
                "condition": request.alert_condition or "above",
                "notified": False
            }
            
        elif request.rule_type == AutomationRuleType.REBALANCING:
            if not request.target_allocation:
                raise HTTPException(400, "Rebalancing rules require target_allocation")
            config = {
                "target_allocation": request.target_allocation,
                "tolerance_percentage": 5.0,  # 5% deviation triggers rebalance
                "check_interval_hours": 6
            }
        
        # Save to database (simplified - would use SQLAlchemy in production)
        rule_data = {
            "rule_id": rule_id,
            "rule_type": request.rule_type,
            "symbol": request.symbol,
            "user_address": request.user_address,
            "status": AutomationStatus.ACTIVE,
            "config": config,
            "created_at": datetime.utcnow()
        }
        
        logger.info("Automation rule created", 
                   rule_id=rule_id,
                   rule_type=request.rule_type,
                   symbol=request.symbol,
                   user_address=request.user_address)
        
        return AutomationRuleResponse(**rule_data, trigger_count=0)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating automation rule", error=str(e))
        raise HTTPException(500, f"Failed to create automation rule: {str(e)}")


@router.get("/rules")
async def get_automation_rules(
    user_address: str,
    current_user: dict = Depends(get_current_user)
):
    """Get automation rules for a user"""
    try:
        # In production, this would query the database
        # For demo, return example rules
        
        example_rules = [
            {
                "rule_id": f"dca_ETH_{int(datetime.utcnow().timestamp())}",
                "rule_type": "dca",
                "symbol": "ETH",
                "user_address": user_address,
                "status": "active",
                "config": {
                    "amount_usd": 100,
                    "interval_hours": 24,
                    "next_execution": datetime.utcnow()
                },
                "created_at": datetime.utcnow(),
                "trigger_count": 0
            }
        ]
        
        logger.info("Automation rules fetched", 
                   user_address=user_address,
                   count=len(example_rules))
        
        return {
            "success": True,
            "rules": example_rules,
            "count": len(example_rules),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error fetching automation rules", 
                    user_address=user_address, 
                    error=str(e))
        raise HTTPException(500, "Failed to fetch automation rules")


@router.get("/check-triggers")
async def check_automation_triggers(
    user_address: str,
    current_user: dict = Depends(get_current_user)
):
    """Check for automation rule triggers that need user action"""
    try:
        adapter = get_hyperliquid_adapter()
        
        # Get current market data
        prices = await adapter.get_all_mid_prices()
        
        # Get user positions
        positions = await adapter.get_user_positions(user_address)
        
        await adapter.close()
        
        # Check for example triggers (in production, this would check actual rules)
        triggers = []
        
        # Example: ETH stop-loss trigger
        eth_price = prices.get('ETH', 0)
        if eth_price > 0:
            for position in positions:
                if position.symbol == 'ETH' and position.unrealized_pnl < -50:
                    triggers.append({
                        "rule_id": f"stop_loss_ETH_{int(datetime.utcnow().timestamp())}",
                        "rule_type": "stop_loss",
                        "symbol": "ETH",
                        "trigger_reason": f"Position down ${abs(position.unrealized_pnl):.2f}",
                        "current_price": eth_price,
                        "recommended_action": {
                            "action": "close_position",
                            "symbol": "ETH",
                            "size": abs(position.size),
                            "reason": "Stop loss triggered"
                        },
                        "triggered_at": datetime.utcnow(),
                        "user_needs_to_act": True
                    })
        
        logger.info("Automation triggers checked", 
                   user_address=user_address,
                   triggers_found=len(triggers))
        
        return {
            "success": True,
            "triggers": triggers,
            "count": len(triggers),
            "requires_user_action": len(triggers) > 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error checking automation triggers", 
                    user_address=user_address, 
                    error=str(e))
        raise HTTPException(500, "Failed to check automation triggers")


@router.post("/rules/{rule_id}/acknowledge")
async def acknowledge_trigger(
    rule_id: str,
    action_taken: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge that user has seen and acted on a trigger"""
    try:
        # In production, this would update the database
        
        logger.info("Automation trigger acknowledged", 
                   rule_id=rule_id,
                   action_taken=action_taken,
                   user_address=current_user["wallet_address"])
        
        return {
            "success": True,
            "message": "Trigger acknowledged",
            "rule_id": rule_id,
            "action_taken": action_taken,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error acknowledging trigger", 
                    rule_id=rule_id, 
                    error=str(e))
        raise HTTPException(500, "Failed to acknowledge trigger")


@router.get("/health")
async def automation_health_check():
    """Health check for automation system"""
    try:
        return {
            "success": True,
            "automation_mode": "notification_based",
            "security_model": "non_custodial",
            "user_controls_execution": True,
            "backend_monitors_conditions": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Automation health check failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }