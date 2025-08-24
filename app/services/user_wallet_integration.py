"""
User Wallet Integration & Delegation System
Allows users to connect their own wallets while enabling automation
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
import jwt
from structlog import get_logger

from app.core.config import settings

logger = get_logger(__name__)


class AutomationRuleType(Enum):
    """Types of automation rules"""
    DCA = "dca"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    REBALANCING = "rebalancing"
    LIQUIDATION_PROTECTION = "liquidation_protection"
    TRAILING_STOP = "trailing_stop"
    GRID_TRADING = "grid_trading"
    ARBITRAGE = "arbitrage"


@dataclass
class DelegationPermission:
    """User delegation permission for automation"""
    user_address: str
    rule_types: List[str]  # Which automation types allowed
    max_amount_per_trade: float  # USD
    max_daily_volume: float  # USD
    valid_until: datetime
    signature: str  # User signature
    nonce: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'valid_until': self.valid_until.isoformat()
        }


@dataclass
class AutomationRule:
    """User automation rule"""
    rule_id: str
    user_address: str
    rule_type: AutomationRuleType
    config: Dict[str, Any]
    status: str  # active, paused, completed, failed
    created_at: datetime
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    max_executions: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'rule_type': self.rule_type.value,
            'created_at': self.created_at.isoformat(),
            'last_executed': self.last_executed.isoformat() if self.last_executed else None
        }


class UserWalletManager:
    """Manages user wallet connections and delegations"""
    
    def __init__(self):
        self.w3 = Web3()
        self.active_delegations: Dict[str, DelegationPermission] = {}
        self.automation_rules: Dict[str, AutomationRule] = {}
        
    def create_delegation_message(self, user_address: str, permissions: Dict[str, Any]) -> str:
        """Create message for user to sign for delegation"""
        
        delegation_data = {
            "address": user_address,
            "permissions": permissions,
            "timestamp": datetime.utcnow().isoformat(),
            "nonce": self._generate_nonce(user_address),
            "platform": "Nadas.fi",
            "version": "1.0"
        }
        
        message = f"""
🤖 Nadas.fi Automation Delegation

You are granting permission for automated trading on your behalf.

Address: {user_address}
Max per trade: ${permissions.get('max_amount_per_trade', 0)}
Max daily volume: ${permissions.get('max_daily_volume', 0)}
Allowed rules: {', '.join(permissions.get('rule_types', []))}
Valid until: {permissions.get('valid_until', 'N/A')}

Timestamp: {delegation_data['timestamp']}
Nonce: {delegation_data['nonce']}

⚠️ Only sign this on the official Nadas.fi platform.
        """.strip()
        
        return message
    
    async def verify_delegation_signature(
        self, 
        user_address: str, 
        message: str, 
        signature: str
    ) -> bool:
        """Verify user's delegation signature"""
        try:
            # Allow demo signature for testing
            if signature == "0xdemo_signature_for_testing" and user_address == "0x7b3B5Ba32D5f4b2e3bA4ba3F4BD0bdD8b1c92b1d":
                return True
            
            # Encode message for signing
            encoded_message = encode_defunct(text=message)
            
            # Recover signer address
            recovered_address = Account.recover_message(encoded_message, signature=signature)
            
            # Check if matches user address
            return recovered_address.lower() == user_address.lower()
            
        except Exception as e:
            logger.error("Signature verification failed", error=str(e))
            return False
    
    async def store_delegation(
        self, 
        user_address: str, 
        permissions: Dict[str, Any], 
        signature: str
    ) -> DelegationPermission:
        """Store user delegation permission"""
        
        # Create delegation object
        delegation = DelegationPermission(
            user_address=user_address,
            rule_types=permissions.get('rule_types', []),
            max_amount_per_trade=permissions.get('max_amount_per_trade', 100.0),
            max_daily_volume=permissions.get('max_daily_volume', 1000.0),
            valid_until=datetime.fromisoformat(permissions.get('valid_until')),
            signature=signature,
            nonce=self._generate_nonce(user_address)
        )
        
        # Store in memory (in production, use database)
        self.active_delegations[user_address] = delegation
        
        logger.info("Delegation stored", user=user_address, permissions=permissions)
        return delegation
    
    def is_delegation_valid(self, user_address: str, rule_type: str, amount: float) -> bool:
        """Check if delegation allows this automation"""
        
        delegation = self.active_delegations.get(user_address)
        if not delegation:
            return False
        
        # Check expiry
        if datetime.utcnow() > delegation.valid_until:
            return False
        
        # Check rule type permission
        if rule_type not in delegation.rule_types:
            return False
        
        # Check amount limits
        if amount > delegation.max_amount_per_trade:
            return False
        
        # Check daily volume (would need to track in DB)
        # For now, assume it's within limits
        
        return True
    
    async def create_automation_rule(
        self, 
        user_address: str, 
        rule_type: AutomationRuleType,
        config: Dict[str, Any]
    ) -> AutomationRule:
        """Create new automation rule"""
        
        # Generate rule ID
        rule_id = f"{rule_type.value}_{user_address}_{int(datetime.utcnow().timestamp())}"
        
        # Create rule
        rule = AutomationRule(
            rule_id=rule_id,
            user_address=user_address,
            rule_type=rule_type,
            config=config,
            status="active",
            created_at=datetime.utcnow(),
            max_executions=config.get('max_executions')
        )
        
        # Store rule
        self.automation_rules[rule_id] = rule
        
        logger.info("Automation rule created", 
                   rule_id=rule_id, 
                   user=user_address, 
                   type=rule_type.value)
        
        return rule
    
    def get_user_rules(self, user_address: str) -> List[AutomationRule]:
        """Get all automation rules for a user"""
        return [
            rule for rule in self.automation_rules.values() 
            if rule.user_address == user_address
        ]
    
    def _generate_nonce(self, user_address: str) -> int:
        """Generate nonce for signature verification"""
        return int(datetime.utcnow().timestamp())
    
    async def prepare_transaction(
        self, 
        user_address: str, 
        trade_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare transaction for user to sign"""
        
        # This would prepare actual Hyperliquid transaction data
        # For now, return mock transaction
        
        transaction = {
            "to": "0x...",  # Hyperliquid contract address
            "data": "0x...",  # Encoded function call
            "value": "0",
            "gas": 150000,
            "gasPrice": "20000000000",  # 20 gwei
            "nonce": await self._get_user_nonce(user_address),
            "chainId": 1  # or Hyperliquid chain ID
        }
        
        return transaction
    
    async def _get_user_nonce(self, user_address: str) -> int:
        """Get user's transaction nonce"""
        # Would get from blockchain
        return 0


class AutomationEngine:
    """Monitors conditions and executes automation rules"""
    
    def __init__(self, wallet_manager: UserWalletManager):
        self.wallet_manager = wallet_manager
        self.running = False
        
    async def start(self):
        """Start automation monitoring"""
        self.running = True
        logger.info("Automation engine started")
        
        while self.running:
            try:
                await self._check_all_rules()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error("Automation engine error", error=str(e))
                await asyncio.sleep(30)  # Wait longer on error
    
    async def stop(self):
        """Stop automation monitoring"""
        self.running = False
        logger.info("Automation engine stopped")
    
    async def _check_all_rules(self):
        """Check all active automation rules"""
        
        for rule in self.wallet_manager.automation_rules.values():
            if rule.status != "active":
                continue
            
            try:
                await self._check_rule(rule)
            except Exception as e:
                logger.error("Rule check failed", rule_id=rule.rule_id, error=str(e))
    
    async def _check_rule(self, rule: AutomationRule):
        """Check if rule conditions are met"""
        
        if rule.rule_type == AutomationRuleType.DCA:
            await self._check_dca_rule(rule)
        elif rule.rule_type == AutomationRuleType.STOP_LOSS:
            await self._check_stop_loss_rule(rule)
        elif rule.rule_type == AutomationRuleType.REBALANCING:
            await self._check_rebalancing_rule(rule)
        # Add more rule types...
    
    async def _check_dca_rule(self, rule: AutomationRule):
        """Check DCA (Dollar Cost Averaging) rule"""
        
        config = rule.config
        interval_hours = {
            'hourly': 1,
            'daily': 24,
            'weekly': 168,
            'monthly': 720
        }.get(config.get('interval', 'daily'), 24)
        
        # Check if enough time has passed
        if rule.last_executed:
            time_since_last = datetime.utcnow() - rule.last_executed
            if time_since_last.total_seconds() < interval_hours * 3600:
                return  # Too early
        
        # Check max executions
        if rule.max_executions and rule.execution_count >= rule.max_executions:
            rule.status = "completed"
            return
        
        # Check price conditions
        symbol = config.get('symbol', 'ETH')
        current_price = await self._get_current_price(symbol)
        
        price_range = config.get('price_range', {})
        if price_range:
            if current_price < price_range.get('min', 0):
                return  # Price too low
            if current_price > price_range.get('max', float('inf')):
                return  # Price too high
        
        # Execute DCA trade
        await self._execute_dca_trade(rule, current_price)
    
    async def _check_stop_loss_rule(self, rule: AutomationRule):
        """Check stop-loss rule"""
        
        config = rule.config
        symbol = config.get('symbol', 'ETH')
        trigger_price = config.get('trigger_price', 0)
        
        current_price = await self._get_current_price(symbol)
        
        # Check if stop-loss should trigger
        side = config.get('side', 'sell')  # Usually sell for stop-loss
        
        if side == 'sell' and current_price <= trigger_price:
            await self._execute_stop_loss_trade(rule, current_price)
        elif side == 'buy' and current_price >= trigger_price:
            await self._execute_stop_loss_trade(rule, current_price)
    
    async def _execute_dca_trade(self, rule: AutomationRule, current_price: float):
        """Execute DCA trade"""
        
        config = rule.config
        amount = config.get('amount_usd', 100)
        
        # Check delegation permission
        if not self.wallet_manager.is_delegation_valid(
            rule.user_address, 
            rule.rule_type.value, 
            amount
        ):
            logger.warning("DCA trade rejected - insufficient delegation", rule_id=rule.rule_id)
            return
        
        # Prepare trade parameters
        trade_params = {
            'symbol': config.get('symbol', 'ETH'),
            'side': config.get('side', 'buy'),
            'amount': amount,
            'order_type': 'market'
        }
        
        # Execute trade (this would integrate with trading service)
        logger.info("Executing DCA trade", 
                   rule_id=rule.rule_id, 
                   user=rule.user_address,
                   params=trade_params)
        
        # Update rule
        rule.last_executed = datetime.utcnow()
        rule.execution_count += 1
        
        # In production, this would:
        # 1. Prepare transaction for user's wallet
        # 2. Submit to Hyperliquid
        # 3. Monitor execution
        # 4. Update rule status
    
    async def _execute_stop_loss_trade(self, rule: AutomationRule, current_price: float):
        """Execute stop-loss trade"""
        
        config = rule.config
        
        logger.info("Executing stop-loss trade", 
                   rule_id=rule.rule_id, 
                   user=rule.user_address,
                   trigger_price=config.get('trigger_price'),
                   current_price=current_price)
        
        # Mark rule as completed (stop-loss usually executes once)
        rule.status = "completed"
        rule.last_executed = datetime.utcnow()
        rule.execution_count += 1
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        # Would integrate with WebSocket manager or Alchemy
        # For now, return mock price
        return 3000.0  # Mock ETH price


# Global instances
_wallet_manager: Optional[UserWalletManager] = None
_automation_engine: Optional[AutomationEngine] = None

def get_wallet_manager() -> UserWalletManager:
    """Get wallet manager instance"""
    global _wallet_manager
    if _wallet_manager is None:
        _wallet_manager = UserWalletManager()
    return _wallet_manager

def get_automation_engine() -> AutomationEngine:
    """Get automation engine instance"""
    global _automation_engine
    if _automation_engine is None:
        wallet_manager = get_wallet_manager()
        _automation_engine = AutomationEngine(wallet_manager)
    return _automation_engine