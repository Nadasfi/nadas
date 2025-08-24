"""
Sub-Account Automation Management
Secure automation system using Hyperliquid sub-accounts for trustless trading
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from structlog import get_logger

from app.core.config import settings
from app.core.supabase import get_supabase_client
from app.services.user_wallet_integration import get_wallet_manager, AutomationRuleType

logger = get_logger(__name__)


class SubAccountStatus(Enum):
    """Sub-account status types"""
    PENDING_CREATION = "pending_creation"
    PENDING_FUNDING = "pending_funding"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EMERGENCY_STOPPED = "emergency_stopped"
    CLOSED = "closed"


@dataclass
class SubAccountConfig:
    """Sub-account configuration"""
    user_address: str
    limit_usd: float
    daily_limit_usd: float
    per_trade_limit_usd: float
    allowed_operations: List[str]
    allowed_pairs: List[str]
    emergency_stop_enabled: bool = True
    auto_rebalance: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class SubAccount:
    """Sub-account data model"""
    sub_account_id: str
    user_address: str
    hyperliquid_sub_address: str
    config: SubAccountConfig
    status: SubAccountStatus
    created_at: datetime
    last_activity: Optional[datetime] = None
    total_value_usd: float = 0.0
    used_limit_usd: float = 0.0
    active_automations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


@dataclass
class SecurityCheck:
    """Security validation result"""
    passed: bool
    checks: Dict[str, bool]
    warnings: List[str]
    blocked_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SubAccountAutomationManager:
    """Manages user sub-accounts for secure automation"""
    
    def __init__(self):
        self.use_mainnet = settings.HYPERLIQUID_NETWORK == "mainnet"
        self.private_key = settings.HYPERLIQUID_PRIVATE_KEY
        
        # Initialize Hyperliquid clients if available
        if self.private_key:
            base_url = constants.MAINNET_API_URL if self.use_mainnet else constants.TESTNET_API_URL
            self.info = Info(base_url)
            self.exchange = Exchange(self.private_key, base_url)
        else:
            self.info = None
            self.exchange = None
            
        # Supabase client for persistent storage
        self.supabase = get_supabase_client()
        
        # Security settings
        self.max_sub_accounts_per_user = 3
        self.min_limit_usd = 100.0
        self.max_limit_usd = 100000.0
        self.allowed_trading_pairs = [
            "ETH/USDC", "BTC/USDC", "SOL/USDC", 
            "ARB/USDC", "OP/USDC", "AVAX/USDC"
        ]
    
    async def initiate_sub_account_creation(
        self, 
        user_address: str,
        automation_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initiate sub-account creation process"""
        
        try:
            # Check if user already has maximum sub-accounts
            existing_accounts = await self.supabase.get_user_sub_accounts(user_address)
            if len(existing_accounts) >= self.max_sub_accounts_per_user:
                return {
                    "success": False,
                    "error": f"Maximum {self.max_sub_accounts_per_user} sub-accounts allowed",
                    "existing_count": len(existing_accounts)
                }
            
            # Analyze automation request to suggest optimal limit
            suggested_limit = self._calculate_suggested_limit(automation_request)
            
            # Create notification data
            notification = {
                "type": "SUB_ACCOUNT_REQUIRED",
                "title": "🔐 Güvenli Otomasyon İçin Alt Hesap",
                "message": "Bu otomasyon tipi için güvenli alt hesap oluşturmanız gerekiyor",
                "automation_details": automation_request,
                "suggested_limit_usd": suggested_limit,
                "min_limit_usd": self.min_limit_usd,
                "max_limit_usd": self.max_limit_usd,
                "benefits": [
                    "Ana cüzdanınız %100 güvende kalır",
                    "Her işlem için imza gerekmez",
                    "İstediğiniz zaman iptal edebilirsiniz",
                    "Sadece belirlediğiniz miktar kullanılır"
                ],
                "risks": [
                    "Alt hesaptaki fonlar otomasyon riskine maruz kalır",
                    "Piyasa volatilitesi kayıplara neden olabilir",
                    "Teknik hatalar olabilir"
                ],
                "required_actions": [
                    "set_limit",
                    "approve_creation", 
                    "transfer_funds"
                ]
            }
            
            logger.info("Sub-account creation initiated", 
                       user=user_address,
                       automation_type=automation_request.get('rule_type'),
                       suggested_limit=suggested_limit)
            
            return {
                "success": True,
                "notification": notification,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to initiate sub-account creation", 
                        user=user_address, error=str(e))
            return {
                "success": False,
                "error": f"Failed to initiate creation: {str(e)}"
            }
    
    def _calculate_suggested_limit(self, automation_request: Dict[str, Any]) -> float:
        """Calculate suggested limit based on automation type"""
        
        rule_type = automation_request.get('rule_type', 'dca')
        config = automation_request.get('config', {})
        
        # Base suggestions by rule type
        base_limits = {
            'dca': 1000.0,  # DCA typically needs sustained funding
            'stop_loss': 500.0,  # One-time protection
            'take_profit': 500.0,  # One-time execution
            'rebalancing': 2000.0,  # Portfolio management
            'grid_trading': 1500.0,  # Multiple levels
            'cross_asset_trigger': 800.0  # Cross-asset swaps
        }
        
        suggested = base_limits.get(rule_type, 1000.0)
        
        # Adjust based on configuration
        if 'amount_usd' in config:
            # For amount-based rules, suggest 5-10x for safety
            suggested = max(suggested, config['amount_usd'] * 7)
        
        if 'max_executions' in config and config['max_executions'] > 0:
            # For limited executions, calculate total exposure
            max_exec = config['max_executions']
            amount = config.get('amount_usd', 100)
            suggested = max(suggested, amount * max_exec * 1.2)  # 20% buffer
        
        # Ensure within bounds
        suggested = max(self.min_limit_usd, min(suggested, self.max_limit_usd))
        
        # Round to nice numbers
        if suggested < 1000:
            suggested = round(suggested / 100) * 100  # Round to hundreds
        else:
            suggested = round(suggested / 500) * 500  # Round to 500s
            
        return suggested
    
    async def create_sub_account_with_limit(
        self,
        user_address: str,
        limit_usd: float,
        automation_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create sub-account with user-defined limit"""
        
        try:
            # Validate limit
            if limit_usd < self.min_limit_usd or limit_usd > self.max_limit_usd:
                return {
                    "success": False,
                    "error": f"Limit must be between ${self.min_limit_usd} and ${self.max_limit_usd}"
                }
            
            # Create sub-account configuration
            config = SubAccountConfig(
                user_address=user_address,
                limit_usd=limit_usd,
                daily_limit_usd=limit_usd / 30,  # Monthly limit / 30
                per_trade_limit_usd=limit_usd * 0.1,  # 10% per trade max
                allowed_operations=["trade", "swap", "transfer_internal"],
                allowed_pairs=self.allowed_trading_pairs,
                emergency_stop_enabled=True,
                auto_rebalance=False
            )
            
            # Generate unique sub-account ID
            sub_account_id = f"sub_{user_address[:8]}_{int(datetime.utcnow().timestamp())}"
            
            # For now, mock the Hyperliquid sub-account creation
            # In production, this would call Hyperliquid's sub-account API
            mock_sub_address = f"0x{sub_account_id.replace('_', '')[:40]}"
            
            # Create sub-account object
            sub_account = SubAccount(
                sub_account_id=sub_account_id,
                user_address=user_address,
                hyperliquid_sub_address=mock_sub_address,
                config=config,
                status=SubAccountStatus.PENDING_FUNDING,
                created_at=datetime.utcnow()
            )
            
            # Store sub-account in Supabase
            sub_account_data = {
                "user_address": user_address,
                "sub_account_id": sub_account_id,
                "hyperliquid_address": mock_sub_address,
                "limit_usd": limit_usd,
                "daily_limit_usd": config.daily_limit_usd,
                "per_trade_limit_usd": config.per_trade_limit_usd,
                "used_limit_usd": 0.0,
                "total_value_usd": 0.0,
                "status": SubAccountStatus.PENDING_FUNDING.value,
                "allowed_operations": config.allowed_operations,
                "allowed_pairs": config.allowed_pairs,
                "emergency_stop_enabled": config.emergency_stop_enabled,
                "auto_rebalance": config.auto_rebalance
            }
            
            result = await self.supabase.create_sub_account(sub_account_data)
            if not result.get("success"):
                raise Exception(f"Database error: {result.get('error')}")
            
            # Create automation rules in database
            for rule in automation_rules:
                rule_id = f"rule_{sub_account_id}_{uuid.uuid4().hex[:8]}"
                rule_data = {
                    "rule_id": rule_id,
                    "sub_account_id": result["data"]["id"],  # Use DB UUID
                    "user_address": user_address,
                    "rule_type": rule.get("rule_type"),
                    "config": rule.get("config", {}),
                    "status": "active",
                    "execution_count": 0
                }
                await self.supabase.create_automation_rule(rule_data)
            
            # Generate transfer instructions
            transfer_instructions = {
                "sub_account_address": mock_sub_address,
                "required_amount_usd": limit_usd,
                "transfer_message": f"Transfer ${limit_usd} to sub-account for automation",
                "estimated_gas": "~$2-5 in ETH for gas fees"
            }
            
            logger.info("Sub-account created", 
                       user=user_address,
                       sub_account_id=sub_account_id,
                       limit=limit_usd)
            
            return {
                "success": True,
                "sub_account": sub_account.to_dict(),
                "transfer_instructions": transfer_instructions,
                "next_steps": [
                    "Transfer funds to sub-account",
                    "Activate automation rules",
                    "Monitor automation dashboard"
                ]
            }
            
        except Exception as e:
            logger.error("Failed to create sub-account", 
                        user=user_address, error=str(e))
            return {
                "success": False,
                "error": f"Failed to create sub-account: {str(e)}"
            }
    
    async def validate_sub_account_operation(
        self,
        sub_account_id: str,
        operation: Dict[str, Any]
    ) -> SecurityCheck:
        """Validate operation against security rules"""
        
        checks = {}
        warnings = []
        blocked_reason = None
        
        try:
            # Get sub-account
            sub_account = self.sub_accounts.get(sub_account_id)
            if not sub_account:
                return SecurityCheck(
                    passed=False,
                    checks={"sub_account_exists": False},
                    warnings=[],
                    blocked_reason="Sub-account not found"
                )
            
            checks["sub_account_exists"] = True
            
            # Check if sub-account is active
            checks["is_active"] = sub_account.status == SubAccountStatus.ACTIVE
            if not checks["is_active"]:
                blocked_reason = f"Sub-account status: {sub_account.status.value}"
            
            # Check emergency stop
            checks["emergency_stop_ok"] = sub_account.status != SubAccountStatus.EMERGENCY_STOPPED
            if not checks["emergency_stop_ok"]:
                blocked_reason = "Emergency stop is active"
            
            # Check operation limits
            operation_amount = operation.get('amount_usd', 0)
            
            # Daily limit check
            daily_used = await self._get_daily_usage(sub_account_id)
            daily_remaining = sub_account.config.daily_limit_usd - daily_used
            checks["within_daily_limit"] = operation_amount <= daily_remaining
            
            if not checks["within_daily_limit"]:
                blocked_reason = f"Exceeds daily limit. Used: ${daily_used:.2f}, Limit: ${sub_account.config.daily_limit_usd:.2f}"
            
            # Per-trade limit check
            checks["within_trade_limit"] = operation_amount <= sub_account.config.per_trade_limit_usd
            if not checks["within_trade_limit"]:
                blocked_reason = f"Exceeds per-trade limit of ${sub_account.config.per_trade_limit_usd:.2f}"
            
            # Total limit check
            checks["within_total_limit"] = (sub_account.used_limit_usd + operation_amount) <= sub_account.config.limit_usd
            if not checks["within_total_limit"]:
                blocked_reason = f"Exceeds total limit of ${sub_account.config.limit_usd:.2f}"
            
            # Check allowed operations
            op_type = operation.get('type', 'unknown')
            checks["operation_allowed"] = op_type in sub_account.config.allowed_operations
            if not checks["operation_allowed"]:
                blocked_reason = f"Operation '{op_type}' not allowed"
            
            # Check trading pairs
            trading_pair = operation.get('trading_pair')
            if trading_pair:
                checks["pair_allowed"] = trading_pair in sub_account.config.allowed_pairs
                if not checks["pair_allowed"]:
                    blocked_reason = f"Trading pair '{trading_pair}' not allowed"
            else:
                checks["pair_allowed"] = True
            
            # Generate warnings
            if daily_remaining < sub_account.config.daily_limit_usd * 0.2:
                warnings.append(f"Daily limit almost reached. ${daily_remaining:.2f} remaining")
            
            if sub_account.used_limit_usd > sub_account.config.limit_usd * 0.8:
                warnings.append("Total limit 80% used")
            
            # Overall pass/fail
            passed = all(checks.values()) and not blocked_reason
            
            return SecurityCheck(
                passed=passed,
                checks=checks,
                warnings=warnings,
                blocked_reason=blocked_reason
            )
            
        except Exception as e:
            logger.error("Security check failed", 
                        sub_account_id=sub_account_id, error=str(e))
            return SecurityCheck(
                passed=False,
                checks={"error": False},
                warnings=[],
                blocked_reason=f"Security check error: {str(e)}"
            )
    
    async def _get_daily_usage(self, sub_account_id: str) -> float:
        """Get daily usage for sub-account"""
        # In production, this would query database for today's transactions
        # For now, return mock data
        return 0.0
    
    async def emergency_stop(self, user_address: str) -> Dict[str, Any]:
        """Emergency stop all user's sub-accounts"""
        
        try:
            sub_account_ids = self.user_sub_accounts.get(user_address, [])
            stopped_accounts = []
            
            for sub_account_id in sub_account_ids:
                sub_account = self.sub_accounts.get(sub_account_id)
                if sub_account and sub_account.status == SubAccountStatus.ACTIVE:
                    # Stop the sub-account
                    sub_account.status = SubAccountStatus.EMERGENCY_STOPPED
                    stopped_accounts.append(sub_account_id)
                    
                    # Cancel all active automations
                    await self._cancel_sub_account_automations(sub_account_id)
            
            logger.warning("Emergency stop activated", 
                          user=user_address,
                          stopped_accounts=len(stopped_accounts))
            
            return {
                "success": True,
                "stopped_accounts": len(stopped_accounts),
                "message": f"🛑 {len(stopped_accounts)} sub-account(s) stopped. All automations cancelled.",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Emergency stop failed", 
                        user=user_address, error=str(e))
            return {
                "success": False,
                "error": f"Emergency stop failed: {str(e)}"
            }
    
    async def _cancel_sub_account_automations(self, sub_account_id: str):
        """Cancel all automations for a sub-account"""
        # This would integrate with the automation engine
        # to cancel all active rules for this sub-account
        wallet_manager = get_wallet_manager()
        
        # Find and cancel automations for this sub-account
        for rule_id, rule in wallet_manager.automation_rules.items():
            if hasattr(rule, 'sub_account_id') and rule.sub_account_id == sub_account_id:
                rule.status = "cancelled"
                logger.info("Automation cancelled", rule_id=rule_id, sub_account=sub_account_id)
    
    async def get_user_sub_accounts(self, user_address: str) -> List[Dict[str, Any]]:
        """Get all sub-accounts for a user"""
        
        return await self.supabase.get_user_sub_accounts(user_address)
    
    def get_sub_account_status(self, sub_account_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a sub-account"""
        
        sub_account = self.sub_accounts.get(sub_account_id)
        if not sub_account:
            return None
        
        return {
            **sub_account.to_dict(),
            "daily_usage_usd": 0.0,  # Would be calculated from transactions
            "remaining_daily_limit": sub_account.config.daily_limit_usd,
            "utilization_percent": (sub_account.used_limit_usd / sub_account.config.limit_usd) * 100,
            "can_trade": sub_account.status == SubAccountStatus.ACTIVE
        }


# Global instance
_sub_account_manager: Optional[SubAccountAutomationManager] = None

def get_sub_account_manager() -> SubAccountAutomationManager:
    """Get or create sub-account manager instance"""
    global _sub_account_manager
    if _sub_account_manager is None:
        _sub_account_manager = SubAccountAutomationManager()
    return _sub_account_manager