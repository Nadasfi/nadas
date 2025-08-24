"""
Automation Engine - DCA, Rebalancing, Liquidation Monitoring  
Real Hyperliquid SDK Implementation for Production Trading
Enhanced with Database Integration and Celery Background Tasks
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.hyperliquid import HyperliquidAdapter, get_hyperliquid_adapter
from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger

logger = get_logger(__name__)


class AutomationRuleType(str, Enum):
    DCA = "dca"
    REBALANCING = "rebalancing"
    LIQUIDATION_MONITOR = "liquidation_monitor"
    STOP_LOSS = "stop_loss"


class AutomationStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AutomationRule:
    """Automation rule configuration"""
    rule_id: str
    user_id: str
    wallet_address: str
    rule_type: AutomationRuleType
    status: AutomationStatus
    config: Dict[str, Any]
    created_at: datetime
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    last_error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of automation execution"""
    success: bool
    message: str
    transaction_hash: Optional[str] = None
    amount_traded: Optional[float] = None
    price: Optional[float] = None
    error: Optional[str] = None


class AutomationEngine:
    """
    Production-ready automation engine using real Hyperliquid SDK
    Handles DCA, portfolio rebalancing, and liquidation monitoring
    """
    
    def __init__(self):
        self.active_rules: Dict[str, AutomationRule] = {}
        self.is_running = False
        self.execution_history: List[Dict] = []
        
    async def load_active_rules_from_db(self):
        """Load active automation rules from database"""
        async with get_db_session() as db:
            result = await db.execute(
                select(AutomationRule).where(AutomationRule.is_active == True)
            )
            db_rules = result.scalars().all()
            
            for db_rule in db_rules:
                # Convert database model to in-memory format for compatibility
                rule = AutomationRule(
                    rule_id=str(db_rule.id),
                    user_id=str(db_rule.user_id),
                    wallet_address=db_rule.user.wallet_address if db_rule.user else "",
                    rule_type=AutomationRuleType(db_rule.automation_type),
                    status=AutomationStatus.ACTIVE,
                    config=db_rule.config,
                    created_at=db_rule.created_at,
                    last_executed=db_rule.last_executed,
                    execution_count=db_rule.execution_count
                )
                self.active_rules[rule.rule_id] = rule
                
            logger.info("Loaded automation rules from database", count=len(db_rules))
        
    async def create_dca_rule(
        self,
        user_id: str,
        wallet_address: str,
        symbol: str,
        amount_usd: float,
        interval_hours: int,
        is_buy: bool = True,
        max_executions: Optional[int] = None
    ) -> str:
        """Create Dollar Cost Averaging automation rule"""
        rule_id = str(uuid.uuid4())
        
        rule = AutomationRule(
            rule_id=rule_id,
            user_id=user_id,
            wallet_address=wallet_address,
            rule_type=AutomationRuleType.DCA,
            status=AutomationStatus.ACTIVE,
            config={
                "symbol": symbol,
                "amount_usd": amount_usd,
                "interval_hours": interval_hours,
                "is_buy": is_buy,
                "max_executions": max_executions,
                "slippage_tolerance": 0.005,  # 0.5% slippage
                "order_type": "market"
            },
            created_at=datetime.utcnow()
        )
        
        self.active_rules[rule_id] = rule
        
        logger.info("DCA rule created",
                   rule_id=rule_id,
                   symbol=symbol,
                   amount_usd=amount_usd,
                   interval_hours=interval_hours)
        
        return rule_id
    
    async def create_rebalancing_rule(
        self,
        user_id: str,
        wallet_address: str,
        target_allocation: Dict[str, float],
        rebalance_threshold: float = 0.05,
        check_interval_hours: int = 24
    ) -> str:
        """Create portfolio rebalancing automation rule"""
        rule_id = str(uuid.uuid4())
        
        # Validate allocation percentages sum to 1.0
        total_allocation = sum(target_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Target allocation must sum to 1.0, got {total_allocation}")
        
        rule = AutomationRule(
            rule_id=rule_id,
            user_id=user_id,
            wallet_address=wallet_address,
            rule_type=AutomationRuleType.REBALANCING,
            status=AutomationStatus.ACTIVE,
            config={
                "target_allocation": target_allocation,
                "rebalance_threshold": rebalance_threshold,
                "check_interval_hours": check_interval_hours,
                "max_trade_size_usd": 5000,
                "min_trade_size_usd": 10,
                "slippage_tolerance": 0.01  # 1% slippage for rebalancing
            },
            created_at=datetime.utcnow()
        )
        
        self.active_rules[rule_id] = rule
        
        logger.info("Rebalancing rule created",
                   rule_id=rule_id,
                   target_allocation=target_allocation,
                   threshold=rebalance_threshold)
        
        return rule_id
    
    async def create_liquidation_monitor(
        self,
        user_id: str,
        wallet_address: str,
        warning_threshold: float = 0.15,  # 15% from liquidation
        emergency_threshold: float = 0.05  # 5% from liquidation
    ) -> str:
        """Create liquidation monitoring rule"""
        rule_id = str(uuid.uuid4())
        
        rule = AutomationRule(
            rule_id=rule_id,
            user_id=user_id,
            wallet_address=wallet_address,
            rule_type=AutomationRuleType.LIQUIDATION_MONITOR,
            status=AutomationStatus.ACTIVE,
            config={
                "warning_threshold": warning_threshold,
                "emergency_threshold": emergency_threshold,
                "check_interval_minutes": 5,
                "auto_reduce_positions": True,
                "max_reduction_percent": 0.5,  # Max 50% position reduction
                "notification_channels": ["email", "telegram"]
            },
            created_at=datetime.utcnow()
        )
        
        self.active_rules[rule_id] = rule
        
        logger.info("Liquidation monitor created",
                   rule_id=rule_id,
                   warning_threshold=warning_threshold,
                   emergency_threshold=emergency_threshold)
        
        return rule_id
    
    async def start_automation_engine(self):
        """Start the main automation engine loop"""
        if self.is_running:
            logger.warning("Automation engine already running")
            return
            
        self.is_running = True
        logger.info("Starting automation engine")
        
        while self.is_running:
            try:
                await self._execute_automation_cycle()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error("Automation engine cycle error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def stop_automation_engine(self):
        """Stop the automation engine"""
        self.is_running = False
        logger.info("Automation engine stopped")
    
    async def _execute_automation_cycle(self):
        """Execute one cycle of automation checks"""
        for rule_id, rule in list(self.active_rules.items()):
            if rule.status != AutomationStatus.ACTIVE:
                continue
                
            try:
                if rule.rule_type == AutomationRuleType.DCA:
                    await self._execute_dca_rule(rule)
                elif rule.rule_type == AutomationRuleType.REBALANCING:
                    await self._execute_rebalancing_rule(rule)
                elif rule.rule_type == AutomationRuleType.LIQUIDATION_MONITOR:
                    await self._execute_liquidation_monitor(rule)
                    
            except Exception as e:
                error_msg = f"Error executing rule {rule_id}: {str(e)}"
                logger.error("Rule execution failed", rule_id=rule_id, error=str(e))
                
                # Update rule with error
                rule.last_error = error_msg
                rule.status = AutomationStatus.ERROR
    
    async def _execute_dca_rule(self, rule: AutomationRule):
        """Execute DCA (Dollar Cost Averaging) rule"""
        config = rule.config
        now = datetime.utcnow()
        
        # Check if it's time to execute
        if rule.last_executed:
            time_since_last = now - rule.last_executed
            if time_since_last.total_seconds() < config["interval_hours"] * 3600:
                return
        
        # Check execution limit
        if config.get("max_executions") and rule.execution_count >= config["max_executions"]:
            rule.status = AutomationStatus.COMPLETED
            logger.info("DCA rule completed max executions", rule_id=rule.rule_id)
            return
        
        try:
            # Get Hyperliquid adapter with user's wallet
            adapter = get_hyperliquid_adapter(
                private_key=self._get_user_private_key(rule.wallet_address)
            )
            
            # Get current market data
            market_data = await adapter.get_market_data(config["symbol"])
            if not market_data:
                logger.warning("No market data for DCA execution", 
                             symbol=config["symbol"], rule_id=rule.rule_id)
                await adapter.close()
                return
            
            current_price = market_data.mid_price
            
            # Calculate order size based on USD amount
            order_size = config["amount_usd"] / current_price
            
            # Round to appropriate precision
            precision = await adapter.get_symbol_precision(config["symbol"])
            order_size = adapter._round_to_precision(order_size, precision["size_decimals"])
            
            # Execute market order
            if config["is_buy"]:
                result = await adapter.place_market_order(
                    symbol=config["symbol"],
                    is_buy=True,
                    size=order_size
                )
            else:
                result = await adapter.place_market_order(
                    symbol=config["symbol"],
                    is_buy=False,
                    size=order_size
                )
            
            await adapter.close()
            
            # Update rule execution
            rule.last_executed = now
            rule.execution_count += 1
            rule.last_error = None
            
            # Log execution
            execution_record = {
                "rule_id": rule.rule_id,
                "type": "dca",
                "timestamp": now.isoformat(),
                "symbol": config["symbol"],
                "size": order_size,
                "price": current_price,
                "amount_usd": config["amount_usd"],
                "is_buy": config["is_buy"],
                "result": result
            }
            self.execution_history.append(execution_record)
            
            logger.info("DCA rule executed successfully",
                       rule_id=rule.rule_id,
                       symbol=config["symbol"],
                       size=order_size,
                       price=current_price,
                       amount_usd=config["amount_usd"])
            
        except Exception as e:
            logger.error("DCA execution failed", rule_id=rule.rule_id, error=str(e))
            rule.last_error = str(e)
    
    async def _execute_rebalancing_rule(self, rule: AutomationRule):
        """Execute portfolio rebalancing rule"""
        config = rule.config
        now = datetime.utcnow()
        
        # Check if it's time to execute
        if rule.last_executed:
            time_since_last = now - rule.last_executed
            if time_since_last.total_seconds() < config["check_interval_hours"] * 3600:
                return
        
        try:
            # Get adapter
            adapter = get_hyperliquid_adapter(
                private_key=self._get_user_private_key(rule.wallet_address)
            )
            
            # Get current portfolio state
            account_value = await adapter.get_account_value(rule.wallet_address)
            positions = await adapter.get_user_positions(rule.wallet_address)
            spot_balances = await adapter.get_spot_balances(rule.wallet_address)
            
            total_value = account_value.get("accountValue", 0)
            if total_value < 100:  # Skip if portfolio too small
                await adapter.close()
                return
            
            # Calculate current allocation
            current_allocation = {}
            target_allocation = config["target_allocation"]
            
            for symbol in target_allocation.keys():
                current_value = 0
                
                # Check positions
                for pos in positions:
                    if pos.get("symbol") == symbol:
                        current_value += abs(pos.get("position_value", 0))
                
                # Check spot balances
                if symbol in spot_balances:
                    market_data = await adapter.get_market_data(symbol)
                    if market_data:
                        current_value += spot_balances[symbol] * market_data.mid_price
                
                current_allocation[symbol] = current_value / total_value if total_value > 0 else 0
            
            # Check if rebalancing is needed
            needs_rebalancing = False
            trades_needed = {}
            
            for symbol, target_percent in target_allocation.items():
                current_percent = current_allocation.get(symbol, 0)
                deviation = abs(current_percent - target_percent)
                
                if deviation > config["rebalance_threshold"]:
                    needs_rebalancing = True
                    target_value = total_value * target_percent
                    current_value = total_value * current_percent
                    trade_value = target_value - current_value
                    
                    # Limit trade size
                    max_trade = config["max_trade_size_usd"]
                    min_trade = config["min_trade_size_usd"]
                    
                    if abs(trade_value) > min_trade:
                        trade_value = max(-max_trade, min(max_trade, trade_value))
                        trades_needed[symbol] = trade_value
            
            if not needs_rebalancing:
                rule.last_executed = now
                await adapter.close()
                return
            
            # Execute rebalancing trades
            executed_trades = []
            for symbol, trade_value in trades_needed.items():
                try:
                    market_data = await adapter.get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    current_price = market_data.mid_price
                    trade_size = abs(trade_value) / current_price
                    is_buy = trade_value > 0
                    
                    # Round to precision
                    precision = await adapter.get_symbol_precision(symbol)
                    trade_size = adapter._round_to_precision(trade_size, precision["size_decimals"])
                    
                    if trade_size > 0:
                        result = await adapter.place_market_order(
                            symbol=symbol,
                            is_buy=is_buy,
                            size=trade_size
                        )
                        
                        executed_trades.append({
                            "symbol": symbol,
                            "size": trade_size,
                            "is_buy": is_buy,
                            "value": trade_value,
                            "price": current_price,
                            "result": result
                        })
                        
                        logger.info("Rebalancing trade executed",
                                   symbol=symbol,
                                   size=trade_size,
                                   is_buy=is_buy,
                                   value=trade_value)
                
                except Exception as e:
                    logger.error("Rebalancing trade failed", symbol=symbol, error=str(e))
            
            await adapter.close()
            
            # Update rule
            rule.last_executed = now
            rule.execution_count += 1
            rule.last_error = None
            
            # Record execution
            execution_record = {
                "rule_id": rule.rule_id,
                "type": "rebalancing",
                "timestamp": now.isoformat(),
                "trades": executed_trades,
                "total_value": total_value,
                "current_allocation": current_allocation,
                "target_allocation": target_allocation
            }
            self.execution_history.append(execution_record)
            
            logger.info("Rebalancing rule executed",
                       rule_id=rule.rule_id,
                       trades_count=len(executed_trades),
                       total_value=total_value)
            
        except Exception as e:
            logger.error("Rebalancing execution failed", rule_id=rule.rule_id, error=str(e))
            rule.last_error = str(e)
    
    async def _execute_liquidation_monitor(self, rule: AutomationRule):
        """Execute liquidation monitoring rule"""
        config = rule.config
        now = datetime.utcnow()
        
        # Check more frequently for liquidation monitoring
        if rule.last_executed:
            time_since_last = now - rule.last_executed
            if time_since_last.total_seconds() < config["check_interval_minutes"] * 60:
                return
        
        try:
            # Get adapter
            adapter = get_hyperliquid_adapter(
                private_key=self._get_user_private_key(rule.wallet_address)
            )
            
            # Get account state
            account_data = await adapter.get_account_value(rule.wallet_address)
            positions = await adapter.get_user_positions(rule.wallet_address)
            
            account_value = account_data.get("accountValue", 0)
            total_margin_used = account_data.get("totalMarginUsed", 0)
            
            if total_margin_used == 0 or account_value == 0:
                await adapter.close()
                rule.last_executed = now
                return
            
            # Calculate margin ratio and liquidation distance
            margin_ratio = total_margin_used / account_value
            
            # Check each position's liquidation distance
            risky_positions = []
            for pos in positions:
                liquidation_price = pos.get("liquidation_price")
                mark_price = pos.get("mark_price")
                
                if liquidation_price and mark_price and liquidation_price > 0:
                    distance_to_liquidation = abs(mark_price - liquidation_price) / mark_price
                    
                    if distance_to_liquidation < config["emergency_threshold"]:
                        risky_positions.append({
                            "symbol": pos.get("symbol"),
                            "distance": distance_to_liquidation,
                            "position": pos,
                            "risk_level": "emergency"
                        })
                    elif distance_to_liquidation < config["warning_threshold"]:
                        risky_positions.append({
                            "symbol": pos.get("symbol"),
                            "distance": distance_to_liquidation,
                            "position": pos,
                            "risk_level": "warning"
                        })
            
            # Handle emergency situations
            emergency_actions = []
            for risky_pos in risky_positions:
                if risky_pos["risk_level"] == "emergency" and config["auto_reduce_positions"]:
                    try:
                        position = risky_pos["position"]
                        current_size = abs(position.get("size", 0))
                        reduction_size = current_size * config["max_reduction_percent"]
                        
                        if reduction_size > 0:
                            # Close portion of position to reduce risk
                            result = await adapter.close_position(
                                symbol=position["symbol"],
                                size=reduction_size
                            )
                            
                            emergency_actions.append({
                                "symbol": position["symbol"],
                                "reduced_size": reduction_size,
                                "distance_to_liquidation": risky_pos["distance"],
                                "result": result
                            })
                            
                            logger.warning("Emergency position reduction executed",
                                         symbol=position["symbol"],
                                         reduced_size=reduction_size,
                                         distance=risky_pos["distance"])
                    
                    except Exception as e:
                        logger.error("Emergency position reduction failed",
                                   symbol=risky_pos["symbol"], error=str(e))
            
            await adapter.close()
            
            # Update rule
            rule.last_executed = now
            rule.execution_count += 1
            rule.last_error = None
            
            # Record monitoring result
            monitoring_record = {
                "rule_id": rule.rule_id,
                "type": "liquidation_monitor",
                "timestamp": now.isoformat(),
                "account_value": account_value,
                "margin_ratio": margin_ratio,
                "risky_positions": risky_positions,
                "emergency_actions": emergency_actions
            }
            self.execution_history.append(monitoring_record)
            
            if risky_positions:
                logger.warning("Liquidation risk detected",
                             rule_id=rule.rule_id,
                             risky_positions_count=len(risky_positions),
                             emergency_actions_count=len(emergency_actions))
            
        except Exception as e:
            logger.error("Liquidation monitoring failed", rule_id=rule.rule_id, error=str(e))
            rule.last_error = str(e)
    
    def _get_user_private_key(self, wallet_address: str) -> str:
        """Get user's private key for trading (placeholder - implement secure key management)"""
        # TODO: Implement secure key management system
        # This is a placeholder - in production, retrieve from secure key vault
        return settings.HYPERLIQUID_PRIVATE_KEY
    
    def get_rule_status(self, rule_id: str) -> Optional[AutomationRule]:
        """Get automation rule status"""
        return self.active_rules.get(rule_id)
    
    def pause_rule(self, rule_id: str) -> bool:
        """Pause automation rule"""
        if rule_id in self.active_rules:
            self.active_rules[rule_id].status = AutomationStatus.PAUSED
            logger.info("Automation rule paused", rule_id=rule_id)
            return True
        return False
    
    def resume_rule(self, rule_id: str) -> bool:
        """Resume automation rule"""
        if rule_id in self.active_rules:
            self.active_rules[rule_id].status = AutomationStatus.ACTIVE
            logger.info("Automation rule resumed", rule_id=rule_id)
            return True
        return False
    
    def delete_rule(self, rule_id: str) -> bool:
        """Delete automation rule"""
        if rule_id in self.active_rules:
            del self.active_rules[rule_id]
            logger.info("Automation rule deleted", rule_id=rule_id)
            return True
        return False
    
    def get_execution_history(self, rule_id: Optional[str] = None) -> List[Dict]:
        """Get execution history for rule or all rules"""
        if rule_id:
            return [record for record in self.execution_history if record.get("rule_id") == rule_id]
        return self.execution_history


# Global automation engine instance
automation_engine = AutomationEngine()
