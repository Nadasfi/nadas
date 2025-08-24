"""
Sub-Account Automation API Endpoints
Secure automation management via Hyperliquid sub-accounts
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.services.sub_account_automation import get_sub_account_manager, SubAccountStatus
from app.core.dependencies import get_current_user_address
from app.core.response import create_response
from structlog import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SubAccountCreationRequest(BaseModel):
    """Request to initiate sub-account creation"""
    automation_request: Dict[str, Any] = Field(..., description="The automation rule that requires sub-account")
    
    class Config:
        schema_extra = {
            "example": {
                "automation_request": {
                    "rule_type": "cross_asset_trigger",
                    "config": {
                        "trigger_asset": "BTC",
                        "trigger_price": 120000,
                        "action": "sell_all",
                        "from_asset": "ETH", 
                        "to_asset": "USDC"
                    }
                }
            }
        }


class SubAccountLimitRequest(BaseModel):
    """Request to create sub-account with limit"""
    limit_usd: float = Field(..., gt=0, description="Maximum USD limit for sub-account")
    automation_rules: List[Dict[str, Any]] = Field(..., description="Automation rules to be enabled")
    
    class Config:
        schema_extra = {
            "example": {
                "limit_usd": 5000.0,
                "automation_rules": [{
                    "rule_type": "cross_asset_trigger",
                    "config": {
                        "trigger_asset": "BTC",
                        "trigger_price": 120000,
                        "action": "sell_all",
                        "from_asset": "ETH",
                        "to_asset": "USDC"
                    }
                }]
            }
        }


class SecurityCheckRequest(BaseModel):
    """Request for security validation"""
    sub_account_id: str = Field(..., description="Sub-account ID")
    operation: Dict[str, Any] = Field(..., description="Operation to validate")
    
    class Config:
        schema_extra = {
            "example": {
                "sub_account_id": "sub_0x7b3B5B_1755775691",
                "operation": {
                    "type": "swap",
                    "amount_usd": 500.0,
                    "trading_pair": "ETH/USDC"
                }
            }
        }


@router.post("/initiate-sub-account")
async def initiate_sub_account_creation(
    request: SubAccountCreationRequest,
    user_address: str = Depends(get_current_user_address)
):
    """Initiate sub-account creation process with educational content"""
    
    try:
        manager = get_sub_account_manager()
        result = await manager.initiate_sub_account_creation(
            user_address, 
            request.automation_request
        )
        
        if result["success"]:
            logger.info("Sub-account creation initiated", 
                       user=user_address,
                       automation_type=request.automation_request.get('rule_type'))
            
            return create_response(
                success=True,
                data=result,
                message="Sub-account creation initiated - please review and set limit"
            )
        else:
            return create_response(
                success=False,
                error=result.get("error", "Failed to initiate sub-account creation"),
                data=result
            )
            
    except Exception as e:
        logger.error("Sub-account initiation failed", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate sub-account creation: {str(e)}"
        )


@router.post("/create-sub-account")
async def create_sub_account_with_limit(
    request: SubAccountLimitRequest,
    user_address: str = Depends(get_current_user_address)
):
    """Create sub-account with user-defined limit and automation rules"""
    
    try:
        manager = get_sub_account_manager()
        result = await manager.create_sub_account_with_limit(
            user_address,
            request.limit_usd,
            request.automation_rules
        )
        
        if result["success"]:
            logger.info("Sub-account created successfully", 
                       user=user_address,
                       sub_account_id=result["sub_account"]["sub_account_id"],
                       limit=request.limit_usd)
            
            return create_response(
                success=True,
                data=result,
                message=f"Sub-account created with ${request.limit_usd:.2f} limit"
            )
        else:
            return create_response(
                success=False,
                error=result.get("error", "Failed to create sub-account"),
                data=result
            )
            
    except Exception as e:
        logger.error("Sub-account creation failed", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create sub-account: {str(e)}"
        )


@router.get("/sub-accounts")
async def get_user_sub_accounts(
    user_address: str = Depends(get_current_user_address)
):
    """Get all sub-accounts for the user"""
    
    try:
        manager = get_sub_account_manager()
        sub_accounts = await manager.get_user_sub_accounts(user_address)
        
        return create_response(
            success=True,
            data={
                "sub_accounts": sub_accounts,
                "count": len(sub_accounts),
                "user_address": user_address
            },
            message=f"Found {len(sub_accounts)} sub-account(s)"
        )
        
    except Exception as e:
        logger.error("Failed to get user sub-accounts", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sub-accounts: {str(e)}"
        )


@router.get("/sub-accounts/{sub_account_id}/status")
async def get_sub_account_status(
    sub_account_id: str,
    user_address: str = Depends(get_current_user_address)
):
    """Get detailed status of a specific sub-account"""
    
    try:
        manager = get_sub_account_manager()
        
        # Verify ownership
        user_sub_accounts = await manager.get_user_sub_accounts(user_address)
        user_sub_account_ids = [acc["sub_account_id"] for acc in user_sub_accounts]
        
        if sub_account_id not in user_sub_account_ids:
            return create_response(
                success=False,
                error="Sub-account not found or access denied",
                data={"sub_account_id": sub_account_id}
            )
        
        status_info = manager.get_sub_account_status(sub_account_id)
        
        if status_info:
            return create_response(
                success=True,
                data=status_info,
                message="Sub-account status retrieved"
            )
        else:
            return create_response(
                success=False,
                error="Sub-account not found",
                data={"sub_account_id": sub_account_id}
            )
            
    except Exception as e:
        logger.error("Failed to get sub-account status", 
                    sub_account_id=sub_account_id, 
                    user=user_address, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sub-account status: {str(e)}"
        )


@router.post("/security-check")
async def perform_security_check(
    request: SecurityCheckRequest,
    user_address: str = Depends(get_current_user_address)
):
    """Perform security validation for a sub-account operation"""
    
    try:
        manager = get_sub_account_manager()
        
        # Verify user owns this sub-account
        user_sub_accounts = await manager.get_user_sub_accounts(user_address)
        user_sub_account_ids = [acc["sub_account_id"] for acc in user_sub_accounts]
        
        if request.sub_account_id not in user_sub_account_ids:
            return create_response(
                success=False,
                error="Sub-account not found or access denied"
            )
        
        # Perform security check
        security_check = await manager.validate_sub_account_operation(
            request.sub_account_id,
            request.operation
        )
        
        return create_response(
            success=True,
            data=security_check.to_dict(),
            message="Security check completed"
        )
        
    except Exception as e:
        logger.error("Security check failed", 
                    sub_account_id=request.sub_account_id,
                    user=user_address, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security check failed: {str(e)}"
        )


@router.post("/emergency-stop")
async def emergency_stop_all(
    user_address: str = Depends(get_current_user_address)
):
    """Emergency stop all user's sub-accounts and automations"""
    
    try:
        manager = get_sub_account_manager()
        result = await manager.emergency_stop(user_address)
        
        if result["success"]:
            logger.warning("Emergency stop executed by user", 
                          user=user_address,
                          stopped_accounts=result["stopped_accounts"])
            
            return create_response(
                success=True,
                data=result,
                message=result["message"]
            )
        else:
            return create_response(
                success=False,
                error=result.get("error", "Emergency stop failed"),
                data=result
            )
            
    except Exception as e:
        logger.error("Emergency stop failed", user=user_address, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency stop failed: {str(e)}"
        )


@router.get("/limits-info")
async def get_limits_information():
    """Get information about sub-account limits and constraints - public endpoint"""
    
    try:
        manager = get_sub_account_manager()
        
        limits_info = {
            "min_limit_usd": manager.min_limit_usd,
            "max_limit_usd": manager.max_limit_usd,
            "max_sub_accounts_per_user": manager.max_sub_accounts_per_user,
            "allowed_trading_pairs": manager.allowed_trading_pairs,
            "security_features": [
                "Daily spending limits",
                "Per-trade limits", 
                "Emergency stop button",
                "Parent account isolation",
                "Real-time monitoring"
            ],
            "risk_disclosures": [
                "Sub-account funds are exposed to automation risks",
                "Market volatility can cause losses",
                "Technical issues may affect execution",
                "Smart contract risks apply"
            ],
            "recommended_limits": {
                "conservative": "2-5% of total portfolio",
                "moderate": "5-15% of total portfolio", 
                "aggressive": "15-25% of total portfolio"
            }
        }
        
        return create_response(
            success=True,
            data=limits_info,
            message="Sub-account limits information retrieved"
        )
        
    except Exception as e:
        logger.error("Failed to get limits info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get limits information: {str(e)}"
        )