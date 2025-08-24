"""
User Wallet + Automation API Endpoints
Manages delegation, automation rules, and user wallet integration
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from app.services.user_wallet_integration import (
    get_wallet_manager, 
    get_automation_engine,
    AutomationRuleType,
    DelegationPermission,
    AutomationRule
)
from app.core.dependencies import get_current_user_address
from app.core.response import create_response
from structlog import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CreateDelegationRequest(BaseModel):
    """Request to create delegation permission"""
    rule_types: List[str] = Field(..., description="Allowed automation types")
    max_amount_per_trade: float = Field(100.0, description="Max USD per trade")
    max_daily_volume: float = Field(1000.0, description="Max daily USD volume")
    valid_days: int = Field(30, description="Delegation valid for N days")


class SignDelegationRequest(BaseModel):
    """Request to sign delegation"""
    message: str = Field(..., description="Message to sign")
    signature: str = Field(..., description="User signature")
    permissions: Dict[str, Any] = Field(..., description="Delegation permissions")


class CreateAutomationRuleRequest(BaseModel):
    """Request to create automation rule"""
    rule_type: str = Field(..., description="Rule type (dca, stop_loss, rebalancing, etc.)")
    config: Dict[str, Any] = Field(..., description="Rule configuration")
    
    class Config:
        schema_extra = {
            "example": {
                "rule_type": "dca",
                "config": {
                    "symbol": "ETH",
                    "amount_usd": 100,
                    "interval": "daily",
                    "max_executions": 30,
                    "price_range": {"min": 2000, "max": 4000}
                }
            }
        }


class UpdateRuleStatusRequest(BaseModel):
    """Request to update rule status"""
    status: str = Field(..., description="New status (active, paused, completed)")


@router.post("/create-delegation")
async def create_delegation_message(
    request: CreateDelegationRequest,
    user_address: str = Depends(get_current_user_address)
):
    """Create delegation message for user to sign"""
    
    try:
        wallet_manager = get_wallet_manager()
        
        # Prepare permissions
        valid_until = datetime.utcnow() + timedelta(days=request.valid_days)
        permissions = {
            "rule_types": request.rule_types,
            "max_amount_per_trade": request.max_amount_per_trade,
            "max_daily_volume": request.max_daily_volume,
            "valid_until": valid_until.isoformat()
        }
        
        # Create message for user to sign
        message = wallet_manager.create_delegation_message(user_address, permissions)
        
        return create_response(
            success=True,
            data={
                "message": message,
                "permissions": permissions,
                "user_address": user_address
            },
            message="Delegation message created - please sign in your wallet"
        )
        
    except Exception as e:
        logger.error("Failed to create delegation message", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create delegation: {str(e)}"
        )


@router.post("/sign-delegation")
async def sign_delegation(
    request: SignDelegationRequest,
    user_address: str = Depends(get_current_user_address)
):
    """Process signed delegation from user"""
    
    try:
        wallet_manager = get_wallet_manager()
        
        # Verify signature
        is_valid = await wallet_manager.verify_delegation_signature(
            user_address, 
            request.message, 
            request.signature
        )
        
        if not is_valid:
            return create_response(
                success=False,
                error="Invalid signature",
                data={"user_address": user_address}
            )
        
        # Store delegation
        delegation = await wallet_manager.store_delegation(
            user_address,
            request.permissions,
            request.signature
        )
        
        logger.info("Delegation signed and stored", 
                   user=user_address, 
                   rule_types=request.permissions.get('rule_types'))
        
        return create_response(
            success=True,
            data=delegation.to_dict(),
            message="Delegation successfully stored - automation enabled!"
        )
        
    except Exception as e:
        logger.error("Failed to process delegation signature", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process delegation: {str(e)}"
        )


@router.get("/delegation-status")
async def get_delegation_status(
    user_address: str = Depends(get_current_user_address)
):
    """Get user's current delegation status"""
    
    try:
        wallet_manager = get_wallet_manager()
        delegation = wallet_manager.active_delegations.get(user_address)
        
        if not delegation:
            return create_response(
                success=True,
                data={
                    "has_delegation": False,
                    "user_address": user_address
                },
                message="No active delegation found"
            )
        
        # Check if expired
        is_expired = datetime.utcnow() > delegation.valid_until
        
        return create_response(
            success=True,
            data={
                "has_delegation": True,
                "delegation": delegation.to_dict(),
                "is_expired": is_expired,
                "days_remaining": (delegation.valid_until - datetime.utcnow()).days if not is_expired else 0
            },
            message="Delegation status retrieved"
        )
        
    except Exception as e:
        logger.error("Failed to get delegation status", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get delegation status: {str(e)}"
        )


@router.post("/create-automation-rule")
async def create_automation_rule(
    request: CreateAutomationRuleRequest,
    user_address: str = Depends(get_current_user_address)
):
    """Create new automation rule"""
    
    try:
        wallet_manager = get_wallet_manager()
        
        # Check if user has valid delegation
        if not wallet_manager.active_delegations.get(user_address):
            return create_response(
                success=False,
                error="No delegation found - please sign delegation first",
                data={"user_address": user_address}
            )
        
        # Validate rule type
        try:
            rule_type = AutomationRuleType(request.rule_type)
        except ValueError:
            return create_response(
                success=False,
                error=f"Invalid rule type: {request.rule_type}",
                data={"available_types": [t.value for t in AutomationRuleType]}
            )
        
        # Check delegation permission for this rule type
        if not wallet_manager.is_delegation_valid(user_address, request.rule_type, 0):
            return create_response(
                success=False,
                error=f"Delegation does not permit {request.rule_type} rules",
                data={"rule_type": request.rule_type}
            )
        
        # Create automation rule
        rule = await wallet_manager.create_automation_rule(
            user_address,
            rule_type,
            request.config
        )
        
        logger.info("Automation rule created", 
                   rule_id=rule.rule_id, 
                   user=user_address, 
                   type=request.rule_type)
        
        return create_response(
            success=True,
            data=rule.to_dict(),
            message=f"Automation rule created: {rule.rule_id}"
        )
        
    except Exception as e:
        logger.error("Failed to create automation rule", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create rule: {str(e)}"
        )


@router.get("/automation-rules")
async def get_user_automation_rules(
    user_address: str = Depends(get_current_user_address)
):
    """Get all automation rules for user"""
    
    try:
        wallet_manager = get_wallet_manager()
        rules = wallet_manager.get_user_rules(user_address)
        
        return create_response(
            success=True,
            data={
                "rules": [rule.to_dict() for rule in rules],
                "count": len(rules),
                "user_address": user_address
            },
            message=f"Found {len(rules)} automation rules"
        )
        
    except Exception as e:
        logger.error("Failed to get automation rules", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rules: {str(e)}"
        )


@router.put("/automation-rules/{rule_id}/status")
async def update_rule_status(
    rule_id: str,
    request: UpdateRuleStatusRequest,
    user_address: str = Depends(get_current_user_address)
):
    """Update automation rule status"""
    
    try:
        wallet_manager = get_wallet_manager()
        
        # Find rule
        rule = wallet_manager.automation_rules.get(rule_id)
        if not rule:
            return create_response(
                success=False,
                error="Rule not found",
                data={"rule_id": rule_id}
            )
        
        # Check ownership
        if rule.user_address != user_address:
            return create_response(
                success=False,
                error="Access denied - not your rule",
                data={"rule_id": rule_id}
            )
        
        # Update status
        old_status = rule.status
        rule.status = request.status
        
        logger.info("Rule status updated", 
                   rule_id=rule_id, 
                   user=user_address, 
                   old_status=old_status,
                   new_status=request.status)
        
        return create_response(
            success=True,
            data={
                "rule_id": rule_id,
                "old_status": old_status,
                "new_status": request.status,
                "updated_rule": rule.to_dict()
            },
            message=f"Rule status updated to {request.status}"
        )
        
    except Exception as e:
        logger.error("Failed to update rule status", rule_id=rule_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update rule: {str(e)}"
        )


@router.delete("/automation-rules/{rule_id}")
async def delete_automation_rule(
    rule_id: str,
    user_address: str = Depends(get_current_user_address)
):
    """Delete automation rule"""
    
    try:
        wallet_manager = get_wallet_manager()
        
        # Find rule
        rule = wallet_manager.automation_rules.get(rule_id)
        if not rule:
            return create_response(
                success=False,
                error="Rule not found",
                data={"rule_id": rule_id}
            )
        
        # Check ownership
        if rule.user_address != user_address:
            return create_response(
                success=False,
                error="Access denied - not your rule",
                data={"rule_id": rule_id}
            )
        
        # Delete rule
        del wallet_manager.automation_rules[rule_id]
        
        logger.info("Automation rule deleted", rule_id=rule_id, user=user_address)
        
        return create_response(
            success=True,
            data={"rule_id": rule_id, "deleted_rule": rule.to_dict()},
            message="Automation rule deleted successfully"
        )
        
    except Exception as e:
        logger.error("Failed to delete automation rule", rule_id=rule_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete rule: {str(e)}"
        )


@router.get("/automation-engine/status")
async def get_automation_engine_status():
    """Get automation engine status - public endpoint"""
    
    try:
        automation_engine = get_automation_engine()
        wallet_manager = get_wallet_manager()
        
        # Count rules by status
        total_rules = len(wallet_manager.automation_rules)
        active_rules = len([r for r in wallet_manager.automation_rules.values() if r.status == "active"])
        paused_rules = len([r for r in wallet_manager.automation_rules.values() if r.status == "paused"])
        completed_rules = len([r for r in wallet_manager.automation_rules.values() if r.status == "completed"])
        
        return create_response(
            success=True,
            data={
                "engine_running": automation_engine.running,
                "total_rules": total_rules,
                "active_rules": active_rules,
                "paused_rules": paused_rules,
                "completed_rules": completed_rules,
                "total_delegations": len(wallet_manager.active_delegations),
                "timestamp": datetime.utcnow().isoformat()
            },
            message="Automation engine status retrieved"
        )
        
    except Exception as e:
        logger.error("Failed to get engine status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get engine status: {str(e)}"
        )


@router.get("/rule-templates")
async def get_rule_templates():
    """Get automation rule templates and examples - public endpoint"""
    
    templates = {
        "dca": {
            "name": "Dollar Cost Averaging",
            "description": "Buy fixed USD amount at regular intervals",
            "example_config": {
                "symbol": "ETH",
                "amount_usd": 100,
                "interval": "daily",  # hourly, daily, weekly, monthly
                "max_executions": 30,
                "price_range": {"min": 2000, "max": 4000},
                "market_hours_only": False
            }
        },
        "stop_loss": {
            "name": "Stop Loss",
            "description": "Automatically sell when price drops below threshold",
            "example_config": {
                "symbol": "ETH",
                "trigger_price": 2800,
                "sell_percentage": 100,  # Sell all
                "trailing": False,
                "market_hours_only": False
            }
        },
        "take_profit": {
            "name": "Take Profit",
            "description": "Automatically sell when price rises above threshold",
            "example_config": {
                "symbol": "ETH",
                "trigger_price": 3500,
                "sell_percentage": 50,  # Sell half
                "trailing": False
            }
        },
        "rebalancing": {
            "name": "Portfolio Rebalancing",
            "description": "Maintain target allocation percentages",
            "example_config": {
                "target_allocation": {
                    "ETH": 40,  # 40%
                    "BTC": 35,  # 35%
                    "SOL": 25   # 25%
                },
                "threshold": 5,  # Rebalance when 5% off target
                "check_interval": "daily"
            }
        },
        "grid_trading": {
            "name": "Grid Trading",
            "description": "Place buy/sell orders in a price grid",
            "example_config": {
                "symbol": "ETH",
                "price_range": {"min": 2800, "max": 3200},
                "grid_levels": 10,
                "amount_per_level": 50,
                "take_profit_percent": 1.0
            }
        }
    }
    
    return create_response(
        success=True,
        data=templates,
        message="Rule templates retrieved"
    )