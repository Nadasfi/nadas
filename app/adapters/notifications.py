"""
Notification System Adapter
$3k Bounty Implementation - Node Info API integration with real-time alerts
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from app.core.config import settings
from app.core.logging import get_logger
from app.adapters.hyperliquid import HyperliquidAdapter

logger = get_logger(__name__)


class NotificationType(str, Enum):
    """Notification types"""
    PORTFOLIO_ALERT = "portfolio_alert"
    PRICE_ALERT = "price_alert"
    POSITION_ALERT = "position_alert"
    TRADE_EXECUTION = "trade_execution"
    LIQUIDATION_WARNING = "liquidation_warning"
    MARKET_NEWS = "market_news"
    SYSTEM_ALERT = "system_alert"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    PUSH = "push"
    WEBSOCKET = "websocket"
    SMS = "sms"


@dataclass
class NotificationRule:
    """Notification rule configuration"""
    rule_id: str
    user_id: str
    name: str
    notification_type: NotificationType
    conditions: Dict[str, Any]
    channels: List[NotificationChannel]
    priority: NotificationPriority = NotificationPriority.MEDIUM
    is_active: bool = True
    created_at: datetime = None
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Notification:
    """Notification message structure"""
    notification_id: str
    user_id: str
    rule_id: Optional[str]
    notification_type: NotificationType
    title: str
    message: str
    data: Dict[str, Any]
    priority: NotificationPriority
    channels: List[NotificationChannel]
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    is_read: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class NotificationProcessor:
    """Process and evaluate notification conditions"""
    
    def __init__(self, hyperliquid_adapter: HyperliquidAdapter):
        self.hyperliquid = hyperliquid_adapter
        
    async def evaluate_portfolio_alerts(self, rule: NotificationRule, user_data: Dict[str, Any]) -> Optional[Notification]:
        """Evaluate portfolio-based notification conditions"""
        try:
            conditions = rule.conditions
            
            # Get current portfolio data
            portfolio_data = await self.hyperliquid.get_account_value(user_data["wallet_address"])
            
            if not portfolio_data:
                return None
            
            current_value = float(portfolio_data.get("accountValue", "0"))
            
            # Check portfolio value thresholds
            if "min_portfolio_value" in conditions:
                min_value = conditions["min_portfolio_value"]
                if current_value < min_value:
                    return Notification(
                        notification_id=str(uuid.uuid4()),
                        user_id=rule.user_id,
                        rule_id=rule.rule_id,
                        notification_type=NotificationType.PORTFOLIO_ALERT,
                        title="Portfolio Value Alert",
                        message=f"Portfolio value ${current_value:,.2f} is below threshold ${min_value:,.2f}",
                        data={
                            "current_value": current_value,
                            "threshold": min_value,
                            "wallet_address": user_data["wallet_address"]
                        },
                        priority=rule.priority,
                        channels=rule.channels
                    )
            
            if "max_portfolio_value" in conditions:
                max_value = conditions["max_portfolio_value"]
                if current_value > max_value:
                    return Notification(
                        notification_id=str(uuid.uuid4()),
                        user_id=rule.user_id,
                        rule_id=rule.rule_id,
                        notification_type=NotificationType.PORTFOLIO_ALERT,
                        title="Portfolio Value Alert",
                        message=f"Portfolio value ${current_value:,.2f} exceeded threshold ${max_value:,.2f}",
                        data={
                            "current_value": current_value,
                            "threshold": max_value,
                            "wallet_address": user_data["wallet_address"]
                        },
                        priority=rule.priority,
                        channels=rule.channels
                    )
            
            return None
            
        except Exception as e:
            logger.error("Error evaluating portfolio alerts", rule_id=rule.rule_id, error=str(e))
            return None
    
    async def evaluate_price_alerts(self, rule: NotificationRule) -> Optional[Notification]:
        """Evaluate price-based notification conditions"""
        try:
            conditions = rule.conditions
            symbol = conditions.get("symbol")
            
            if not symbol:
                return None
            
            # Get current market data
            market_data = await self.hyperliquid.get_market_data(symbol)
            
            if not market_data:
                return None
            
            current_price = float(market_data.get("mid_price", "0"))
            
            # Check price thresholds
            if "target_price_above" in conditions:
                target_price = conditions["target_price_above"]
                if current_price > target_price:
                    return Notification(
                        notification_id=str(uuid.uuid4()),
                        user_id=rule.user_id,
                        rule_id=rule.rule_id,
                        notification_type=NotificationType.PRICE_ALERT,
                        title=f"{symbol} Price Alert",
                        message=f"{symbol} price ${current_price:,.4f} exceeded target ${target_price:,.4f}",
                        data={
                            "symbol": symbol,
                            "current_price": current_price,
                            "target_price": target_price,
                            "direction": "above"
                        },
                        priority=rule.priority,
                        channels=rule.channels
                    )
            
            if "target_price_below" in conditions:
                target_price = conditions["target_price_below"]
                if current_price < target_price:
                    return Notification(
                        notification_id=str(uuid.uuid4()),
                        user_id=rule.user_id,
                        rule_id=rule.rule_id,
                        notification_type=NotificationType.PRICE_ALERT,
                        title=f"{symbol} Price Alert",
                        message=f"{symbol} price ${current_price:,.4f} dropped below target ${target_price:,.4f}",
                        data={
                            "symbol": symbol,
                            "current_price": current_price,
                            "target_price": target_price,
                            "direction": "below"
                        },
                        priority=rule.priority,
                        channels=rule.channels
                    )
            
            return None
            
        except Exception as e:
            logger.error("Error evaluating price alerts", rule_id=rule.rule_id, error=str(e))
            return None
    
    async def evaluate_liquidation_warnings(self, rule: NotificationRule, user_data: Dict[str, Any]) -> Optional[Notification]:
        """Evaluate liquidation risk notification conditions"""
        try:
            conditions = rule.conditions
            warning_threshold = conditions.get("liquidation_distance_threshold", 0.10)  # 10% default
            
            # Get user positions
            positions = await self.hyperliquid.get_user_positions(user_data["wallet_address"])
            
            high_risk_positions = []
            
            for position in positions:
                if not position.get("liquidation_price") or not position.get("mark_price"):
                    continue
                
                liquidation_price = float(position["liquidation_price"])
                mark_price = float(position["mark_price"])
                
                # Calculate distance to liquidation
                if liquidation_price > 0 and mark_price > 0:
                    distance = abs(mark_price - liquidation_price) / mark_price
                    
                    if distance < warning_threshold:
                        high_risk_positions.append({
                            "symbol": position["coin"],
                            "size": position["szi"],
                            "liquidation_price": liquidation_price,
                            "mark_price": mark_price,
                            "distance_percentage": distance * 100
                        })
            
            if high_risk_positions:
                return Notification(
                    notification_id=str(uuid.uuid4()),
                    user_id=rule.user_id,
                    rule_id=rule.rule_id,
                    notification_type=NotificationType.LIQUIDATION_WARNING,
                    title="Liquidation Risk Warning",
                    message=f"{len(high_risk_positions)} position(s) at risk of liquidation",
                    data={
                        "positions": high_risk_positions,
                        "threshold": warning_threshold,
                        "wallet_address": user_data["wallet_address"]
                    },
                    priority=NotificationPriority.CRITICAL,
                    channels=rule.channels
                )
            
            return None
            
        except Exception as e:
            logger.error("Error evaluating liquidation warnings", rule_id=rule.rule_id, error=str(e))
            return None


class NotificationManager:
    """Main notification management system"""
    
    def __init__(self, hyperliquid_adapter: HyperliquidAdapter):
        self.hyperliquid = hyperliquid_adapter
        self.processor = NotificationProcessor(hyperliquid_adapter)
        
        # In-memory storage for demo (in production, use database)
        self.rules: Dict[str, NotificationRule] = {}
        self.notifications: Dict[str, Notification] = {}
        self.user_subscriptions: Dict[str, List[str]] = {}  # user_id -> rule_ids
        
        # WebSocket connections for real-time delivery
        self.websocket_connections: Dict[str, Any] = {}  # user_id -> websocket
        
        # Background task control
        self.is_running = False
        self.evaluation_task: Optional[asyncio.Task] = None
        
        logger.info("Notification manager initialized")
    
    async def create_rule(self, rule: NotificationRule) -> bool:
        """Create a new notification rule"""
        try:
            self.rules[rule.rule_id] = rule
            
            # Add to user subscriptions
            if rule.user_id not in self.user_subscriptions:
                self.user_subscriptions[rule.user_id] = []
            self.user_subscriptions[rule.user_id].append(rule.rule_id)
            
            logger.info("Notification rule created", 
                       rule_id=rule.rule_id, 
                       user_id=rule.user_id,
                       notification_type=rule.notification_type)
            
            return True
            
        except Exception as e:
            logger.error("Error creating notification rule", error=str(e))
            return False
    
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing notification rule"""
        try:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules[rule_id]
            
            # Update rule properties
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            logger.info("Notification rule updated", rule_id=rule_id)
            return True
            
        except Exception as e:
            logger.error("Error updating notification rule", rule_id=rule_id, error=str(e))
            return False
    
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a notification rule"""
        try:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules[rule_id]
            
            # Remove from user subscriptions
            if rule.user_id in self.user_subscriptions:
                self.user_subscriptions[rule.user_id] = [
                    rid for rid in self.user_subscriptions[rule.user_id] if rid != rule_id
                ]
            
            # Delete rule
            del self.rules[rule_id]
            
            logger.info("Notification rule deleted", rule_id=rule_id)
            return True
            
        except Exception as e:
            logger.error("Error deleting notification rule", rule_id=rule_id, error=str(e))
            return False
    
    async def get_user_rules(self, user_id: str) -> List[NotificationRule]:
        """Get all notification rules for a user"""
        user_rule_ids = self.user_subscriptions.get(user_id, [])
        return [self.rules[rule_id] for rule_id in user_rule_ids if rule_id in self.rules]
    
    async def get_user_notifications(self, user_id: str, limit: int = 50) -> List[Notification]:
        """Get notifications for a user"""
        user_notifications = [
            notification for notification in self.notifications.values()
            if notification.user_id == user_id
        ]
        
        # Sort by creation time (newest first)
        user_notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        return user_notifications[:limit]
    
    async def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        try:
            if notification_id in self.notifications:
                notification = self.notifications[notification_id]
                notification.is_read = True
                notification.read_at = datetime.utcnow()
                return True
            return False
            
        except Exception as e:
            logger.error("Error marking notification as read", notification_id=notification_id, error=str(e))
            return False
    
    async def start_monitoring(self):
        """Start the notification monitoring system"""
        if self.is_running:
            logger.info("Notification monitoring already running")
            return
        
        self.is_running = True
        self.evaluation_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Notification monitoring started")
    
    async def stop_monitoring(self):
        """Stop the notification monitoring system"""
        self.is_running = False
        
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Notification monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for evaluating notification rules"""
        while self.is_running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(settings.NOTIFICATION_CHECK_INTERVAL_SECONDS)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in notification monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _evaluate_all_rules(self):
        """Evaluate all active notification rules"""
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            try:
                # Check cooldown period to avoid spam
                if rule.last_triggered:
                    cooldown_minutes = 15  # 15 minutes cooldown
                    if (datetime.utcnow() - rule.last_triggered).total_seconds() < cooldown_minutes * 60:
                        continue
                
                # Get user data (you'd typically fetch this from database)
                user_data = {"wallet_address": "0x1234..."}  # Placeholder
                
                notification = None
                
                # Evaluate based on notification type
                if rule.notification_type == NotificationType.PORTFOLIO_ALERT:
                    notification = await self.processor.evaluate_portfolio_alerts(rule, user_data)
                elif rule.notification_type == NotificationType.PRICE_ALERT:
                    notification = await self.processor.evaluate_price_alerts(rule)
                elif rule.notification_type == NotificationType.LIQUIDATION_WARNING:
                    notification = await self.processor.evaluate_liquidation_warnings(rule, user_data)
                
                # Send notification if triggered
                if notification:
                    await self._send_notification(notification)
                    
                    # Update rule trigger info
                    rule.last_triggered = datetime.utcnow()
                    rule.trigger_count += 1
                
            except Exception as e:
                logger.error("Error evaluating notification rule", rule_id=rule.rule_id, error=str(e))
    
    async def _send_notification(self, notification: Notification):
        """Send notification through configured channels"""
        try:
            # Store notification
            self.notifications[notification.notification_id] = notification
            notification.sent_at = datetime.utcnow()
            
            # Send through each configured channel
            for channel in notification.channels:
                if channel == NotificationChannel.WEBSOCKET:
                    await self._send_websocket_notification(notification)
                elif channel == NotificationChannel.IN_APP:
                    await self._send_in_app_notification(notification)
                # Add other channels as needed
            
            logger.info("Notification sent", 
                       notification_id=notification.notification_id,
                       user_id=notification.user_id,
                       channels=notification.channels)
            
        except Exception as e:
            logger.error("Error sending notification", 
                        notification_id=notification.notification_id, 
                        error=str(e))
    
    async def _send_websocket_notification(self, notification: Notification):
        """Send notification via WebSocket"""
        try:
            if notification.user_id in self.websocket_connections:
                websocket = self.websocket_connections[notification.user_id]
                message = {
                    "type": "notification",
                    "data": asdict(notification)
                }
                await websocket.send(json.dumps(message))
                
        except Exception as e:
            logger.error("Error sending WebSocket notification", error=str(e))
    
    async def _send_in_app_notification(self, notification: Notification):
        """Send in-app notification (stored for later retrieval)"""
        # In-app notifications are already stored in self.notifications
        # This method could trigger UI updates or push to a queue
        pass
    
    def register_websocket(self, user_id: str, websocket):
        """Register a WebSocket connection for a user"""
        self.websocket_connections[user_id] = websocket
        logger.info("WebSocket registered for user", user_id=user_id)
    
    def unregister_websocket(self, user_id: str):
        """Unregister a WebSocket connection for a user"""
        if user_id in self.websocket_connections:
            del self.websocket_connections[user_id]
            logger.info("WebSocket unregistered for user", user_id=user_id)


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager(hyperliquid_adapter: HyperliquidAdapter = None) -> NotificationManager:
    """Factory function to get notification manager instance"""
    global _notification_manager
    
    if _notification_manager is None:
        if hyperliquid_adapter is None:
            from app.adapters.hyperliquid import get_hyperliquid_adapter
            hyperliquid_adapter = get_hyperliquid_adapter()
        
        _notification_manager = NotificationManager(hyperliquid_adapter)
    
    return _notification_manager
