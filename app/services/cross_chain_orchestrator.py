"""
Cross-Chain Orchestrator Service
Coordinates LI.FI, GlueX, and Liquid Labs for seamless cross-chain operations
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import structlog

from app.adapters.lifi import get_lifi_adapter
from app.adapters.gluex import get_gluex_adapter  
from app.adapters.liquid_labs import get_liquid_labs_adapter
# from app.services.automation_engine import get_automation_engine  # TODO: Implement later
from app.services.websocket_manager import get_websocket_manager
from app.core.config import settings

logger = structlog.get_logger(__name__)

class StrategyStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    QUOTE_READY = "quote_ready"
    EXECUTING_BRIDGE = "executing_bridge"
    WAITING_CONFIRMATION = "waiting_confirmation"
    BRIDGE_COMPLETED = "bridge_completed"
    EXECUTING_TARGET = "executing_target"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BridgeProvider(str, Enum):
    LIFI = "lifi"
    GLUEX = "gluex"
    LIQUID_LABS = "liquid_labs"

class CrossChainStrategy:
    """Represents a cross-chain strategy execution"""
    
    def __init__(self, user_address: str, strategy_config: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.user_address = user_address
        self.strategy_config = strategy_config
        self.status = StrategyStatus.PENDING
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.execution_log: List[Dict] = []
        self.route_quotes: List[Dict] = []
        self.selected_route: Optional[Dict] = None
        self.bridge_transactions: List[Dict] = []
        self.automation_rules: List[str] = []
        self.estimated_completion: Optional[datetime] = None
        self.actual_completion: Optional[datetime] = None
        self.total_fees_usd: float = 0.0
        self.error_message: Optional[str] = None

    def add_log_entry(self, message: str, level: str = "info", data: Optional[Dict] = None):
        """Add log entry to execution history"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "level": level,
            "data": data or {}
        }
        self.execution_log.append(entry)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for API responses"""
        return {
            "id": self.id,
            "user_address": self.user_address,
            "status": self.status.value,
            "strategy_config": self.strategy_config,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "execution_log": self.execution_log[-10:],  # Last 10 entries
            "route_quotes": self.route_quotes,
            "selected_route": self.selected_route,
            "bridge_transactions": self.bridge_transactions,
            "automation_rules": self.automation_rules,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "actual_completion": self.actual_completion.isoformat() if self.actual_completion else None,
            "total_fees_usd": self.total_fees_usd,
            "error_message": self.error_message,
            "progress_percentage": self._calculate_progress()
        }

    def _calculate_progress(self) -> int:
        """Calculate completion percentage based on status"""
        progress_map = {
            StrategyStatus.PENDING: 0,
            StrategyStatus.ANALYZING: 10,
            StrategyStatus.QUOTE_READY: 20,
            StrategyStatus.EXECUTING_BRIDGE: 40,
            StrategyStatus.WAITING_CONFIRMATION: 60,
            StrategyStatus.BRIDGE_COMPLETED: 80,
            StrategyStatus.EXECUTING_TARGET: 90,
            StrategyStatus.COMPLETED: 100,
            StrategyStatus.FAILED: 0,
            StrategyStatus.CANCELLED: 0
        }
        return progress_map.get(self.status, 0)

class CrossChainOrchestrator:
    """Main orchestrator for cross-chain operations"""
    
    def __init__(self):
        self.active_strategies: Dict[str, CrossChainStrategy] = {}
        self.lifi_adapter = None
        self.gluex_adapter = None
        self.liquid_labs_adapter = None
        self.automation_engine = None
        self.websocket_manager = None
        self._initialize_adapters()

    def _initialize_adapters(self):
        """Initialize all adapters lazily"""
        try:
            self.lifi_adapter = get_lifi_adapter()
            self.gluex_adapter = get_gluex_adapter()
            self.liquid_labs_adapter = get_liquid_labs_adapter()
            self.automation_engine = None  # TODO: Implement automation engine integration
            self.websocket_manager = get_websocket_manager()
            logger.info("Cross-chain orchestrator adapters initialized")
        except Exception as e:
            logger.error("Failed to initialize orchestrator adapters", error=str(e))
    
    def _emit_strategy_event(self, strategy: CrossChainStrategy, event_type: str, data: Dict[str, Any] = None):
        """Emit WebSocket event for strategy updates"""
        try:
            if self.websocket_manager:
                self.websocket_manager.emit_orchestrator_event(
                    strategy_id=strategy.id,
                    user_address=strategy.user_address,
                    event_type=event_type,
                    status=strategy.status.value,
                    data=data or {}
                )
        except Exception as e:
            logger.warning("Failed to emit strategy event", error=str(e))

    async def create_strategy(self, user_address: str, strategy_config: Dict[str, Any]) -> CrossChainStrategy:
        """Create new cross-chain strategy"""
        try:
            strategy = CrossChainStrategy(user_address, strategy_config)
            self.active_strategies[strategy.id] = strategy
            
            strategy.add_log_entry("Strategy created", "info", {
                "source_chain": strategy_config.get("source_chain"),
                "target_chain": strategy_config.get("target_chain"),
                "asset": strategy_config.get("asset"),
                "amount": strategy_config.get("amount")
            })
            
            logger.info("Cross-chain strategy created", 
                       strategy_id=strategy.id,
                       user=user_address)
            
            # Emit strategy creation event
            self._emit_strategy_event(strategy, "strategy_created", {
                "source_chain": strategy_config.get("source_chain"),
                "target_chain": strategy_config.get("target_chain"),
                "amount": strategy_config.get("amount")
            })
            
            return strategy
            
        except Exception as e:
            # Emit strategy creation failed event if strategy was created
            if 'strategy' in locals():
                self._emit_strategy_event(strategy, "strategy_creation_failed", {
                    "error": str(e)
                })
            
            logger.error("Failed to create cross-chain strategy", 
                        user=user_address, error=str(e))
            raise

    async def analyze_routes(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Analyze and compare routes from multiple providers"""
        strategy = self.active_strategies.get(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        try:
            strategy.status = StrategyStatus.ANALYZING
            strategy.add_log_entry("Starting route analysis")
            
            config = strategy.strategy_config
            quotes = []
            
            # Get LI.FI quote
            try:
                lifi_quote = await self._get_lifi_quote(config)
                if lifi_quote:
                    quotes.append({
                        "provider": BridgeProvider.LIFI.value,
                        "route": lifi_quote,
                        "estimated_fee_usd": lifi_quote.get("fee_usd", 0),
                        "estimated_time_minutes": lifi_quote.get("time_minutes", 10),
                        "confidence_score": 85
                    })
                    strategy.add_log_entry("LI.FI quote received", "info", {"fee": lifi_quote.get("fee_usd")})
            except Exception as e:
                strategy.add_log_entry(f"LI.FI quote failed: {str(e)}", "warning")
                logger.warning("LI.FI quote failed", error=str(e))
            
            # Get GlueX quote  
            try:
                gluex_quote = await self._get_gluex_quote(config)
                if gluex_quote:
                    quotes.append({
                        "provider": BridgeProvider.GLUEX.value,
                        "route": gluex_quote,
                        "estimated_fee_usd": gluex_quote.get("fee_usd", 0),
                        "estimated_time_minutes": gluex_quote.get("time_minutes", 5),
                        "confidence_score": 90
                    })
                    strategy.add_log_entry("GlueX quote received", "info", {"fee": gluex_quote.get("fee_usd")})
            except Exception as e:
                strategy.add_log_entry(f"GlueX quote failed: {str(e)}", "warning")
                logger.warning("GlueX quote failed", error=str(e))
            
            if not quotes:
                strategy.status = StrategyStatus.FAILED
                strategy.error_message = "No routes available from any provider"
                strategy.add_log_entry("Route analysis failed - no quotes available", "error")
                
                # Emit route analysis failed event
                self._emit_strategy_event(strategy, "routes_analysis_failed", {
                    "error": "No routes available from any provider"
                })
                
                return []
            
            # Sort by best combination of fee and time
            quotes.sort(key=lambda x: (x["estimated_fee_usd"] * 0.7 + x["estimated_time_minutes"] * 0.3))
            
            strategy.route_quotes = quotes
            strategy.status = StrategyStatus.QUOTE_READY
            strategy.add_log_entry(f"Route analysis completed - {len(quotes)} options found")
            
            # Emit routes analyzed event
            self._emit_strategy_event(strategy, "routes_analyzed", {
                "quote_count": len(quotes),
                "best_provider": quotes[0]["provider"] if quotes else None,
                "best_fee_usd": quotes[0]["estimated_fee_usd"] if quotes else None
            })
            
            return quotes
            
        except Exception as e:
            strategy.status = StrategyStatus.FAILED
            strategy.error_message = str(e)
            strategy.add_log_entry(f"Route analysis failed: {str(e)}", "error")
            
            # Emit route analysis failed event
            self._emit_strategy_event(strategy, "routes_analysis_failed", {
                "error": str(e)
            })
            
            logger.error("Route analysis failed", strategy_id=strategy_id, error=str(e))
            raise

    async def _get_lifi_quote(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get quote from LI.FI adapter"""
        if not self.lifi_adapter:
            return None
            
        return await self.lifi_adapter.get_quote(
            from_chain=config.get("source_chain", "ethereum"),
            to_chain=config.get("target_chain", "hyperliquid"),
            from_token=config.get("source_token", "ETH"),
            to_token=config.get("target_token", "USDC"),
            amount=str(config.get("amount", 1000))
        )

    async def _get_gluex_quote(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get quote from GlueX adapter"""
        if not self.gluex_adapter:
            return None
            
        return await self.gluex_adapter.get_best_route(
            asset=config.get("target_token", "USDC"),
            target_chain="hyperliquid",
            amount=config.get("amount", 1000)
        )

    async def select_optimal_route(self, strategy_id: str, route_index: int = 0) -> Dict[str, Any]:
        """Select and prepare optimal route for execution"""
        strategy = self.active_strategies.get(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        if not strategy.route_quotes or route_index >= len(strategy.route_quotes):
            raise ValueError("Invalid route selection")
        
        selected_quote = strategy.route_quotes[route_index]
        strategy.selected_route = selected_quote
        
        # Calculate estimated completion time
        estimated_minutes = selected_quote["estimated_time_minutes"]
        strategy.estimated_completion = datetime.utcnow() + timedelta(minutes=estimated_minutes)
        strategy.total_fees_usd = selected_quote["estimated_fee_usd"]
        
        strategy.add_log_entry(f"Route selected: {selected_quote['provider']}", "info", {
            "provider": selected_quote["provider"],
            "fee_usd": selected_quote["estimated_fee_usd"],
            "time_minutes": selected_quote["estimated_time_minutes"]
        })
        
        logger.info("Route selected for strategy", 
                   strategy_id=strategy_id, 
                   provider=selected_quote["provider"])
        
        # Emit route selected event
        self._emit_strategy_event(strategy, "route_selected", {
            "provider": selected_quote["provider"],
            "fee_usd": selected_quote["estimated_fee_usd"],
            "time_minutes": selected_quote["estimated_time_minutes"]
        })
        
        return selected_quote

    async def execute_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Execute the complete cross-chain strategy"""
        strategy = self.active_strategies.get(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        if not strategy.selected_route:
            raise ValueError("No route selected for execution")
        
        try:
            strategy.status = StrategyStatus.EXECUTING_BRIDGE
            strategy.add_log_entry("Starting cross-chain execution")
            
            # Emit execution started event
            self._emit_strategy_event(strategy, "execution_started", {
                "provider": strategy.selected_route["provider"]
            })
            
            # Execute bridge transaction
            bridge_result = await self._execute_bridge_transaction(strategy)
            
            if bridge_result.get("success"):
                strategy.status = StrategyStatus.WAITING_CONFIRMATION
                strategy.add_log_entry("Bridge transaction submitted", "info", {
                    "tx_hash": bridge_result.get("tx_hash"),
                    "provider": strategy.selected_route["provider"]
                })
                
                # Emit bridge submitted event
                self._emit_strategy_event(strategy, "bridge_submitted", {
                    "tx_hash": bridge_result.get("tx_hash"),
                    "provider": strategy.selected_route["provider"]
                })
                
                # Start monitoring in background
                asyncio.create_task(self._monitor_bridge_completion(strategy_id, bridge_result))
                
                return {
                    "success": True,
                    "strategy_id": strategy_id,
                    "bridge_tx_hash": bridge_result.get("tx_hash"),
                    "status": strategy.status.value,
                    "estimated_completion": strategy.estimated_completion.isoformat() if strategy.estimated_completion else None
                }
            else:
                strategy.status = StrategyStatus.FAILED
                strategy.error_message = bridge_result.get("error", "Bridge execution failed")
                strategy.add_log_entry(f"Bridge execution failed: {strategy.error_message}", "error")
                
                # Emit bridge execution failed event
                self._emit_strategy_event(strategy, "bridge_execution_failed", {
                    "error": strategy.error_message,
                    "provider": strategy.selected_route["provider"]
                })
                
                raise Exception(strategy.error_message)
                
        except Exception as e:
            strategy.status = StrategyStatus.FAILED
            strategy.error_message = str(e)
            strategy.add_log_entry(f"Strategy execution failed: {str(e)}", "error")
            
            # Emit strategy execution failed event
            self._emit_strategy_event(strategy, "strategy_execution_failed", {
                "error": str(e)
            })
            
            logger.error("Strategy execution failed", strategy_id=strategy_id, error=str(e))
            raise

    async def _execute_bridge_transaction(self, strategy: CrossChainStrategy) -> Dict[str, Any]:
        """Execute bridge transaction based on selected provider"""
        provider = strategy.selected_route["provider"]
        
        if provider == BridgeProvider.LIFI.value:
            return await self._execute_lifi_bridge(strategy)
        elif provider == BridgeProvider.GLUEX.value:
            return await self._execute_gluex_bridge(strategy)
        else:
            raise ValueError(f"Unsupported bridge provider: {provider}")

    async def _execute_lifi_bridge(self, strategy: CrossChainStrategy) -> Dict[str, Any]:
        """Execute LI.FI bridge transaction"""
        try:
            if not self.lifi_adapter:
                raise Exception("LI.FI adapter not available")
            
            route_data = strategy.selected_route["route"]
            config = strategy.strategy_config
            
            # For demo purposes, simulate transaction
            # In production, this would call actual LI.FI execution
            bridge_tx = {
                "success": True,
                "tx_hash": f"0x{uuid.uuid4().hex[:40]}",  # Mock tx hash
                "provider": BridgeProvider.LIFI.value,
                "source_chain": config.get("source_chain"),
                "target_chain": config.get("target_chain"),
                "amount": config.get("amount"),
                "fee_usd": route_data.get("fee_usd", 0),
                "estimated_time": route_data.get("time_minutes", 10)
            }
            
            strategy.bridge_transactions.append(bridge_tx)
            return bridge_tx
            
        except Exception as e:
            logger.error("LI.FI bridge execution failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _execute_gluex_bridge(self, strategy: CrossChainStrategy) -> Dict[str, Any]:
        """Execute GlueX bridge transaction"""
        try:
            if not self.gluex_adapter:
                raise Exception("GlueX adapter not available")
            
            route_data = strategy.selected_route["route"]
            config = strategy.strategy_config
            
            # For demo purposes, simulate transaction
            # In production, this would call actual GlueX execution
            bridge_tx = {
                "success": True,
                "tx_hash": f"0x{uuid.uuid4().hex[:40]}",  # Mock tx hash
                "provider": BridgeProvider.GLUEX.value,
                "source_chain": config.get("source_chain"),
                "target_chain": config.get("target_chain"),
                "amount": config.get("amount"),
                "fee_usd": route_data.get("fee_usd", 0),
                "estimated_time": route_data.get("time_minutes", 5)
            }
            
            strategy.bridge_transactions.append(bridge_tx)
            return bridge_tx
            
        except Exception as e:
            logger.error("GlueX bridge execution failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _monitor_bridge_completion(self, strategy_id: str, bridge_result: Dict[str, Any]):
        """Monitor bridge transaction completion"""
        strategy = self.active_strategies.get(strategy_id)
        if not strategy:
            return
        
        try:
            # Simulate monitoring - in production, poll actual bridge status
            estimated_time = bridge_result.get("estimated_time", 10)
            await asyncio.sleep(min(estimated_time * 60, 300))  # Max 5 minutes for demo
            
            # Simulate successful completion
            strategy.status = StrategyStatus.BRIDGE_COMPLETED
            strategy.add_log_entry("Bridge transaction confirmed", "info", {
                "tx_hash": bridge_result.get("tx_hash")
            })
            
            # Emit bridge completed event
            self._emit_strategy_event(strategy, "bridge_completed", {
                "tx_hash": bridge_result.get("tx_hash")
            })
            
            # Setup automation rules if configured
            if strategy.strategy_config.get("automation_rules"):
                await self._setup_automation_rules(strategy)
            
            strategy.status = StrategyStatus.COMPLETED
            strategy.actual_completion = datetime.utcnow()
            strategy.add_log_entry("Cross-chain strategy completed successfully", "info")
            
            # Emit strategy completed event
            self._emit_strategy_event(strategy, "strategy_completed", {
                "total_time_minutes": (datetime.utcnow() - strategy.created_at).total_seconds() / 60,
                "total_fees_usd": strategy.total_fees_usd,
                "automation_rules_count": len(strategy.automation_rules)
            })
            
            logger.info("Cross-chain strategy completed", strategy_id=strategy_id)
            
        except Exception as e:
            strategy.status = StrategyStatus.FAILED
            strategy.error_message = f"Bridge monitoring failed: {str(e)}"
            strategy.add_log_entry(strategy.error_message, "error")
            
            # Emit bridge monitoring failed event
            self._emit_strategy_event(strategy, "bridge_monitoring_failed", {
                "error": str(e)
            })
            
            logger.error("Bridge monitoring failed", strategy_id=strategy_id, error=str(e))

    async def _setup_automation_rules(self, strategy: CrossChainStrategy):
        """Setup automation rules after successful bridge"""
        try:
            if not self.automation_engine:
                return
            
            automation_config = strategy.strategy_config.get("automation_rules", {})
            
            # Create automation rules based on strategy config
            for rule_config in automation_config.get("rules", []):
                rule_id = await self.automation_engine.create_rule(
                    user_address=strategy.user_address,
                    rule_type=rule_config.get("type"),
                    config=rule_config.get("config", {})
                )
                
                strategy.automation_rules.append(rule_id)
                strategy.add_log_entry(f"Automation rule created: {rule_config.get('type')}", "info", {
                    "rule_id": rule_id
                })
            
        except Exception as e:
            strategy.add_log_entry(f"Automation setup failed: {str(e)}", "warning")
            logger.warning("Automation setup failed", strategy_id=strategy.id, error=str(e))

    def get_strategy(self, strategy_id: str) -> Optional[CrossChainStrategy]:
        """Get strategy by ID"""
        return self.active_strategies.get(strategy_id)

    def get_user_strategies(self, user_address: str) -> List[CrossChainStrategy]:
        """Get all strategies for a user"""
        return [
            strategy for strategy in self.active_strategies.values()
            if strategy.user_address == user_address
        ]

    async def cancel_strategy(self, strategy_id: str) -> bool:
        """Cancel a pending strategy"""
        strategy = self.active_strategies.get(strategy_id)
        if not strategy:
            return False
        
        if strategy.status in [StrategyStatus.PENDING, StrategyStatus.ANALYZING, StrategyStatus.QUOTE_READY]:
            strategy.status = StrategyStatus.CANCELLED
            strategy.add_log_entry("Strategy cancelled by user", "info")
            
            # Emit strategy cancelled event
            self._emit_strategy_event(strategy, "strategy_cancelled", {})
            
            return True
        
        return False

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics"""
        total_strategies = len(self.active_strategies)
        completed = sum(1 for s in self.active_strategies.values() if s.status == StrategyStatus.COMPLETED)
        failed = sum(1 for s in self.active_strategies.values() if s.status == StrategyStatus.FAILED)
        
        provider_usage = {}
        for strategy in self.active_strategies.values():
            if strategy.selected_route:
                provider = strategy.selected_route.get("provider", "unknown")
                provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        return {
            "total_strategies": total_strategies,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total_strategies * 100) if total_strategies > 0 else 0,
            "provider_usage": provider_usage,
            "average_fee_usd": sum(s.total_fees_usd for s in self.active_strategies.values()) / total_strategies if total_strategies > 0 else 0
        }

# Global orchestrator instance
_orchestrator = None

def get_cross_chain_orchestrator() -> CrossChainOrchestrator:
    """Get global cross-chain orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CrossChainOrchestrator()
    return _orchestrator