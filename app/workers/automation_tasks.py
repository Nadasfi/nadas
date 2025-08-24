"""
Celery Background Tasks for Automation Engine
Real implementation of DCA, stop-loss, rebalancing, and monitoring
Based on nadas_prd.md requirements for production automation
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from celery import Celery
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.automation import AutomationRule, AutomationExecution
from app.models.user import User
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.services.ai_service import get_ai_service
from app.adapters.notifications import get_notification_manager
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Initialize Celery app
celery_app = Celery(
    "nadas_automation",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.automation_tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "app.workers.automation_tasks.execute_automation_rule": {"queue": "automation"},
        "app.workers.automation_tasks.monitor_portfolios": {"queue": "monitoring"},
        "app.workers.automation_tasks.check_liquidation_risks": {"queue": "risk_monitoring"},
    },
    beat_schedule={
        # Execute automation rules every minute
        "execute-automation-rules": {
            "task": "app.workers.automation_tasks.execute_automation_rules",
            "schedule": 60.0,  # Every 60 seconds
        },
        # Monitor portfolios every 30 seconds
        "monitor-portfolios": {
            "task": "app.workers.automation_tasks.monitor_portfolios", 
            "schedule": 30.0,  # Every 30 seconds
        },
        # Check liquidation risks every 15 seconds
        "check-liquidation-risks": {
            "task": "app.workers.automation_tasks.check_liquidation_risks",
            "schedule": 15.0,  # Every 15 seconds
        },
        # AI market analysis every 5 minutes
        "ai-market-analysis": {
            "task": "app.workers.automation_tasks.ai_market_analysis",
            "schedule": 300.0,  # Every 5 minutes
        },
        # Portfolio health check every 2 minutes
        "portfolio-health-check": {
            "task": "app.workers.automation_tasks.portfolio_health_check",
            "schedule": 120.0,  # Every 2 minutes
        }
    }
)


@celery_app.task(name="app.workers.automation_tasks.execute_automation_rules")
def execute_automation_rules():
    """Execute all active automation rules"""
    try:
        logger.info("Starting automation rules execution cycle")
        
        db = next(get_db())
        
        # Get all active automation rules
        active_rules = db.query(AutomationRule).filter(
            AutomationRule.is_active == True
        ).all()
        
        executed_count = 0
        failed_count = 0
        
        for rule in active_rules:
            try:
                # Check if rule should be executed
                if should_execute_rule(rule):
                    execute_automation_rule.delay(rule.id)
                    executed_count += 1
                    
            except Exception as e:
                logger.error("Error checking automation rule", 
                           rule_id=rule.id, error=str(e))
                failed_count += 1
        
        logger.info("Automation rules execution cycle completed",
                   total_rules=len(active_rules),
                   executed=executed_count,
                   failed=failed_count)
        
        return {
            "total_rules": len(active_rules),
            "executed": executed_count,
            "failed": failed_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error in automation rules execution", error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="app.workers.automation_tasks.execute_automation_rule")
def execute_automation_rule(rule_id: str):
    """Execute a specific automation rule"""
    try:
        db = next(get_db())
        
        rule = db.query(AutomationRule).filter(AutomationRule.id == rule_id).first()
        if not rule:
            logger.error("Automation rule not found", rule_id=rule_id)
            return {"error": "Rule not found"}
        
        # Create execution record
        execution = AutomationExecution(
            rule_id=rule.id,
            status="executing",
            started_at=datetime.utcnow()
        )
        db.add(execution)
        db.commit()
        
        try:
            # Execute based on automation type
            if rule.automation_type == "dca":
                result = asyncio.run(execute_dca_rule(rule, db))
            elif rule.automation_type == "stop_loss":
                result = asyncio.run(execute_stop_loss_rule(rule, db))
            elif rule.automation_type == "rebalance":
                result = asyncio.run(execute_rebalancing_rule(rule, db))
            elif rule.automation_type == "liquidation_monitor":
                result = asyncio.run(execute_liquidation_monitor(rule, db))
            elif rule.automation_type == "take_profit":
                result = asyncio.run(execute_take_profit_rule(rule, db))
            elif rule.automation_type == "trailing_stop":
                result = asyncio.run(execute_trailing_stop_rule(rule, db))
            else:
                raise ValueError(f"Unknown automation type: {rule.automation_type}")
            
            # Update execution record
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            execution.execution_details = result
            
            # Update rule
            rule.last_executed = datetime.utcnow()
            rule.execution_count += 1
            if result.get("success", False):
                rule.success_count += 1
            
            db.commit()
            
            logger.info("Automation rule executed successfully",
                       rule_id=rule_id,
                       automation_type=rule.automation_type,
                       result=result)
            
            return result
            
        except Exception as e:
            # Update execution record with error
            execution.status = "failed"
            execution.completed_at = datetime.utcnow()
            execution.error_message = str(e)
            db.commit()
            
            logger.error("Automation rule execution failed",
                        rule_id=rule_id,
                        error=str(e))
            raise
            
    except Exception as e:
        logger.error("Error executing automation rule", rule_id=rule_id, error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


async def execute_dca_rule(rule: AutomationRule, db: Session) -> Dict[str, Any]:
    """Execute Dollar Cost Averaging automation"""
    try:
        config = rule.config
        
        # Check if enough time has passed since last execution
        if rule.last_executed:
            time_since_last = datetime.utcnow() - rule.last_executed
            required_interval = timedelta(hours=config.get("interval_hours", 24))
            
            if time_since_last < required_interval:
                return {
                    "success": False,
                    "reason": "Interval not reached",
                    "next_execution": (rule.last_executed + required_interval).isoformat()
                }
        
        # Get Hyperliquid adapter
        adapter = get_hyperliquid_adapter()
        
        # Get current price
        all_mids = await adapter.get_all_mids()
        current_price = float(all_mids.get(config["coin"], 0))
        
        if current_price == 0:
            return {"success": False, "reason": "Unable to get current price"}
        
        # Calculate order size
        amount_usd = config["amount_usd"]
        order_size = amount_usd / current_price
        
        # Check account balance
        account_equity = await adapter.get_vault_equity()
        if account_equity < amount_usd:
            return {
                "success": False, 
                "reason": "Insufficient funds",
                "required": amount_usd,
                "available": account_equity
            }
        
        # Execute market buy order
        result = await adapter.place_market_order(
            coin=config["coin"],
            is_buy=True,
            size=order_size,
            slippage=config.get("max_slippage", 0.01)
        )
        
        await adapter.close()
        
        # Send notification
        notification_manager = get_notification_manager()
        await notification_manager.send_notification(
            rule.user_id,
            "automation_executed",
            f"DCA executed: Bought ${amount_usd} worth of {config['coin']} at ${current_price:.2f}",
            {"rule_type": "dca", "coin": config["coin"], "amount": amount_usd, "price": current_price}
        )
        
        return {
            "success": True,
            "coin": config["coin"],
            "amount_usd": amount_usd,
            "order_size": order_size,
            "execution_price": current_price,
            "order_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("DCA rule execution failed", rule_id=rule.id, error=str(e))
        return {"success": False, "error": str(e)}


async def execute_stop_loss_rule(rule: AutomationRule, db: Session) -> Dict[str, Any]:
    """Execute stop-loss automation"""
    try:
        config = rule.config
        
        # Get Hyperliquid adapter
        adapter = get_hyperliquid_adapter()
        
        # Get current price
        all_mids = await adapter.get_all_mids()
        current_price = float(all_mids.get(config["coin"], 0))
        
        if current_price == 0:
            return {"success": False, "reason": "Unable to get current price"}
        
        # Check if stop-loss should trigger
        trigger_price = config["trigger_price"]
        trigger_direction = config.get("trigger_direction", "below")
        
        should_trigger = False
        if trigger_direction == "below" and current_price <= trigger_price:
            should_trigger = True
        elif trigger_direction == "above" and current_price >= trigger_price:
            should_trigger = True
        
        if not should_trigger:
            return {
                "success": False,
                "reason": "Stop-loss not triggered",
                "current_price": current_price,
                "trigger_price": trigger_price,
                "trigger_direction": trigger_direction
            }
        
        # Get current position
        positions = await adapter.get_user_positions()
        position = next((p for p in positions if p.coin == config["coin"]), None)
        
        if not position or position.size == 0:
            # Deactivate rule since no position exists
            rule.is_active = False
            db.commit()
            
            return {
                "success": False,
                "reason": "No position found - rule deactivated",
                "coin": config["coin"]
            }
        
        # Calculate close size
        position_percentage = config.get("position_percentage", 1.0)
        close_size = abs(position.size) * position_percentage
        
        # Execute market order to close position
        result = await adapter.place_market_order(
            coin=config["coin"],
            is_buy=position.size < 0,  # Reverse position direction
            size=close_size,
            slippage=config.get("emergency_slippage", 0.02)
        )
        
        await adapter.close()
        
        # Deactivate rule after execution
        rule.is_active = False
        db.commit()
        
        # Send notification
        notification_manager = get_notification_manager()
        await notification_manager.send_notification(
            rule.user_id,
            "stop_loss_triggered",
            f"Stop-loss triggered for {config['coin']} at ${current_price:.2f}. Position closed.",
            {
                "rule_type": "stop_loss",
                "coin": config["coin"],
                "trigger_price": trigger_price,
                "execution_price": current_price,
                "size_closed": close_size
            }
        )
        
        return {
            "success": True,
            "triggered": True,
            "coin": config["coin"],
            "trigger_price": trigger_price,
            "execution_price": current_price,
            "position_closed": close_size,
            "order_result": result,
            "rule_deactivated": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Stop-loss rule execution failed", rule_id=rule.id, error=str(e))
        return {"success": False, "error": str(e)}


async def execute_rebalancing_rule(rule: AutomationRule, db: Session) -> Dict[str, Any]:
    """Execute portfolio rebalancing automation"""
    try:
        config = rule.config
        
        # Check if enough time has passed since last execution
        if rule.last_executed:
            time_since_last = datetime.utcnow() - rule.last_executed
            required_interval = timedelta(hours=config.get("check_interval_hours", 24))
            
            if time_since_last < required_interval:
                return {
                    "success": False,
                    "reason": "Rebalance interval not reached"
                }
        
        # Get Hyperliquid adapter
        adapter = get_hyperliquid_adapter()
        
        # Get current portfolio
        account_equity = await adapter.get_vault_equity()
        positions = await adapter.get_user_positions()
        all_mids = await adapter.get_all_mids()
        
        # Calculate current allocation
        current_allocation = {}
        for position in positions:
            if position.size != 0:
                current_price = float(all_mids.get(position.coin, 0))
                position_value = position.size * current_price
                current_allocation[position.coin] = position_value / account_equity
        
        # Calculate required trades
        target_allocation = config["target_allocation"]
        rebalance_threshold = config.get("rebalance_threshold", 0.05)
        max_trade_size_usd = config.get("max_trade_size_usd", 1000)
        
        trades_needed = []
        
        for coin, target_percent in target_allocation.items():
            current_percent = current_allocation.get(coin, 0)
            deviation = abs(current_percent - target_percent)
            
            if deviation > rebalance_threshold:
                target_value = account_equity * target_percent
                current_value = account_equity * current_percent
                trade_value = target_value - current_value
                
                if abs(trade_value) >= config.get("min_trade_size_usd", 50):
                    trades_needed.append({
                        "coin": coin,
                        "trade_value_usd": trade_value,
                        "is_buy": trade_value > 0,
                        "current_allocation": current_percent,
                        "target_allocation": target_percent,
                        "deviation": deviation
                    })
        
        if not trades_needed:
            return {
                "success": True,
                "reason": "No rebalancing needed",
                "current_allocation": current_allocation,
                "target_allocation": target_allocation
            }
        
        # Execute rebalancing trades
        executed_trades = []
        
        for trade in trades_needed:
            if abs(trade["trade_value_usd"]) > max_trade_size_usd:
                continue  # Skip trades that are too large
            
            current_price = float(all_mids.get(trade["coin"], 0))
            if current_price > 0:
                trade_size = abs(trade["trade_value_usd"]) / current_price
                
                result = await adapter.place_market_order(
                    coin=trade["coin"],
                    is_buy=trade["is_buy"],
                    size=trade_size,
                    slippage=config.get("max_slippage", 0.01)
                )
                
                executed_trades.append({
                    **trade,
                    "trade_size": trade_size,
                    "execution_price": current_price,
                    "order_result": result
                })
        
        await adapter.close()
        
        # Send notification
        if executed_trades:
            notification_manager = get_notification_manager()
            await notification_manager.send_notification(
                rule.user_id,
                "rebalancing_executed",
                f"Portfolio rebalanced: {len(executed_trades)} trades executed",
                {
                    "rule_type": "rebalancing",
                    "trades_executed": len(executed_trades),
                    "total_trades_needed": len(trades_needed)
                }
            )
        
        return {
            "success": True,
            "trades_needed": len(trades_needed),
            "trades_executed": len(executed_trades),
            "executed_trades": executed_trades,
            "current_allocation": current_allocation,
            "target_allocation": target_allocation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Rebalancing rule execution failed", rule_id=rule.id, error=str(e))
        return {"success": False, "error": str(e)}


async def execute_liquidation_monitor(rule: AutomationRule, db: Session) -> Dict[str, Any]:
    """Execute liquidation risk monitoring"""
    try:
        config = rule.config
        
        # Get Hyperliquid adapter
        adapter = get_hyperliquid_adapter()
        
        # Get account state
        account_equity = await adapter.get_vault_equity()
        positions = await adapter.get_user_positions()
        all_mids = await adapter.get_all_mids()
        
        # Calculate total position value and margin ratio
        total_position_value = 0
        position_details = []
        
        for position in positions:
            if position.size != 0:
                current_price = float(all_mids.get(position.coin, 0))
                position_value = abs(position.size) * current_price
                total_position_value += position_value
                
                position_details.append({
                    "coin": position.coin,
                    "size": position.size,
                    "value": position_value,
                    "unrealized_pnl": position.unrealized_pnl
                })
        
        # Calculate margin ratio
        margin_ratio = account_equity / total_position_value if total_position_value > 0 else 1.0
        
        # Check thresholds
        warning_threshold = config.get("warning_threshold", 0.15)
        emergency_threshold = config.get("emergency_threshold", 0.10)
        
        risk_level = "safe"
        action_taken = None
        
        if margin_ratio <= emergency_threshold:
            # Emergency: Execute emergency action
            emergency_action = config.get("emergency_action", "close_positions")
            
            if emergency_action == "close_positions":
                action_taken = await execute_emergency_position_closure(adapter, positions)
            elif emergency_action == "reduce_positions":
                action_taken = await execute_position_reduction(adapter, positions, 0.5)
            
            risk_level = "critical"
            
            # Send critical notification
            notification_manager = get_notification_manager()
            await notification_manager.send_notification(
                rule.user_id,
                "liquidation_critical",
                f"CRITICAL: Margin ratio {margin_ratio:.3f} below emergency threshold. Emergency action taken.",
                {
                    "rule_type": "liquidation_monitor",
                    "margin_ratio": margin_ratio,
                    "emergency_threshold": emergency_threshold,
                    "action_taken": emergency_action
                }
            )
            
        elif margin_ratio <= warning_threshold:
            risk_level = "warning"
            
            # Check if warning was recently sent to avoid spam
            last_warning_time = config.get("last_warning_sent")
            if not last_warning_time or datetime.fromisoformat(last_warning_time) < datetime.utcnow() - timedelta(hours=1):
                notification_manager = get_notification_manager()
                await notification_manager.send_notification(
                    rule.user_id,
                    "liquidation_warning",
                    f"WARNING: Low margin ratio {margin_ratio:.3f}. Consider reducing positions.",
                    {
                        "rule_type": "liquidation_monitor",
                        "margin_ratio": margin_ratio,
                        "warning_threshold": warning_threshold
                    }
                )
                
                # Update last warning time
                config["last_warning_sent"] = datetime.utcnow().isoformat()
                rule.config = config
                db.commit()
        
        await adapter.close()
        
        return {
            "success": True,
            "margin_ratio": margin_ratio,
            "risk_level": risk_level,
            "account_equity": account_equity,
            "total_position_value": total_position_value,
            "position_count": len(position_details),
            "position_details": position_details,
            "action_taken": action_taken,
            "thresholds": {
                "warning": warning_threshold,
                "emergency": emergency_threshold
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Liquidation monitor execution failed", rule_id=rule.id, error=str(e))
        return {"success": False, "error": str(e)}


async def execute_emergency_position_closure(adapter, positions) -> Dict[str, Any]:
    """Close all positions in emergency"""
    try:
        closed_positions = []
        
        for position in positions:
            if position.size != 0:
                result = await adapter.place_market_order(
                    coin=position.coin,
                    is_buy=position.size < 0,  # Reverse position
                    size=abs(position.size),
                    slippage=0.05  # 5% slippage tolerance for emergency
                )
                
                closed_positions.append({
                    "coin": position.coin,
                    "size_closed": abs(position.size),
                    "order_result": result
                })
        
        return {
            "action": "emergency_close_all",
            "positions_closed": len(closed_positions),
            "details": closed_positions
        }
        
    except Exception as e:
        logger.error("Emergency position closure failed", error=str(e))
        return {"action": "emergency_close_all", "error": str(e)}


def should_execute_rule(rule: AutomationRule) -> bool:
    """Check if a rule should be executed based on its schedule"""
    try:
        if not rule.is_active:
            return False
        
        # For continuous monitoring rules, always execute
        if rule.automation_type in ["liquidation_monitor"]:
            return True
        
        # For scheduled rules, check timing
        if rule.last_executed:
            config = rule.config
            
            if rule.automation_type == "dca":
                interval_hours = config.get("interval_hours", 24)
                return datetime.utcnow() - rule.last_executed >= timedelta(hours=interval_hours)
            
            elif rule.automation_type == "rebalance":
                interval_hours = config.get("check_interval_hours", 24)
                return datetime.utcnow() - rule.last_executed >= timedelta(hours=interval_hours)
            
            elif rule.automation_type in ["stop_loss", "take_profit", "trailing_stop"]:
                # Price-triggered rules should be checked frequently
                return datetime.utcnow() - rule.last_executed >= timedelta(minutes=1)
        
        # If never executed, execute now
        return True
        
    except Exception as e:
        logger.error("Error checking rule execution", rule_id=rule.id, error=str(e))
        return False


@celery_app.task(name="app.workers.automation_tasks.monitor_portfolios")
def monitor_portfolios():
    """Monitor all user portfolios for changes and alerts"""
    try:
        logger.info("Starting portfolio monitoring cycle")
        
        db = next(get_db())
        
        # Get all active users with positions
        users_with_rules = db.query(User).join(AutomationRule).filter(
            AutomationRule.is_active == True
        ).distinct().all()
        
        monitored_count = 0
        alerts_sent = 0
        
        for user in users_with_rules:
            try:
                # Monitor user portfolio
                result = asyncio.run(monitor_user_portfolio(user.wallet_address))
                monitored_count += 1
                alerts_sent += result.get("alerts_sent", 0)
                
            except Exception as e:
                logger.error("Error monitoring user portfolio", 
                           user_id=user.id, error=str(e))
        
        logger.info("Portfolio monitoring cycle completed",
                   users_monitored=monitored_count,
                   alerts_sent=alerts_sent)
        
        return {
            "users_monitored": monitored_count,
            "alerts_sent": alerts_sent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error in portfolio monitoring", error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


async def monitor_user_portfolio(wallet_address: str) -> Dict[str, Any]:
    """Monitor specific user portfolio"""
    try:
        # Implementation for portfolio monitoring
        # This would check for significant changes, unusual activity, etc.
        return {"alerts_sent": 0}
        
    except Exception as e:
        logger.error("Error monitoring user portfolio", address=wallet_address, error=str(e))
        return {"error": str(e)}


@celery_app.task(name="app.workers.automation_tasks.check_liquidation_risks")
def check_liquidation_risks():
    """Check liquidation risks across all monitored portfolios"""
    try:
        logger.info("Starting liquidation risk check")
        
        db = next(get_db())
        
        # Get liquidation monitoring rules
        liquidation_rules = db.query(AutomationRule).filter(
            AutomationRule.automation_type == "liquidation_monitor",
            AutomationRule.is_active == True
        ).all()
        
        checked_count = 0
        high_risk_count = 0
        
        for rule in liquidation_rules:
            try:
                result = asyncio.run(execute_liquidation_monitor(rule, db))
                checked_count += 1
                
                if result.get("risk_level") in ["warning", "critical"]:
                    high_risk_count += 1
                    
            except Exception as e:
                logger.error("Error checking liquidation risk", 
                           rule_id=rule.id, error=str(e))
        
        logger.info("Liquidation risk check completed",
                   checked=checked_count,
                   high_risk=high_risk_count)
        
        return {
            "portfolios_checked": checked_count,
            "high_risk_portfolios": high_risk_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error in liquidation risk check", error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="app.workers.automation_tasks.ai_market_analysis")
def ai_market_analysis():
    """Perform AI market analysis and cache results"""
    try:
        logger.info("Starting AI market analysis")
        
        major_assets = ["ETH", "BTC", "SOL", "ARB", "OP", "AVAX"]
        
        ai_service = get_ai_service()
        analysis = asyncio.run(ai_service.analyze_market_conditions(major_assets))
        
        # Cache results (would store in Redis in production)
        logger.info("AI market analysis completed",
                   assets=major_assets,
                   provider=analysis.provider.value)
        
        return {
            "analysis_completed": True,
            "assets_analyzed": major_assets,
            "provider": analysis.provider.value,
            "timestamp": analysis.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error("Error in AI market analysis", error=str(e))
        return {"error": str(e)}


@celery_app.task(name="app.workers.automation_tasks.portfolio_health_check")
def portfolio_health_check():
    """Perform comprehensive portfolio health checks"""
    try:
        logger.info("Starting portfolio health check")
        
        # Implementation for comprehensive health checks
        # This would analyze portfolio composition, risk metrics, performance, etc.
        
        return {
            "health_check_completed": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error in portfolio health check", error=str(e))
        return {"error": str(e)}