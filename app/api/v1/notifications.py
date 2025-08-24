"""
Notification System API Endpoints
$3k Bounty Implementation - Real-time notification system with Node Info API integration
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid

from app.api.v1.auth import get_current_user
from app.adapters.notifications import (
    NotificationManager,
    NotificationRule,
    NotificationType,
    NotificationPriority,
    NotificationChannel,
    get_notification_manager
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CreateNotificationRuleRequest(BaseModel):
    """Request to create a notification rule"""
    name: str = Field(..., min_length=1, max_length=100, description="Rule name")
    notification_type: NotificationType = Field(..., description="Type of notification")
    conditions: Dict[str, Any] = Field(..., description="Trigger conditions")
    channels: List[NotificationChannel] = Field(..., description="Delivery channels")
    priority: NotificationPriority = Field(NotificationPriority.MEDIUM, description="Notification priority")


class NotificationRuleResponse(BaseModel):
    """Notification rule response"""
    rule_id: str
    user_id: str
    name: str
    notification_type: NotificationType
    conditions: Dict[str, Any]
    channels: List[NotificationChannel]
    priority: NotificationPriority
    is_active: bool
    created_at: datetime
    last_triggered: Optional[datetime]
    trigger_count: int


@router.post("/rules", response_model=NotificationRuleResponse)
async def create_notification_rule(
    request: CreateNotificationRuleRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    notification_manager: NotificationManager = Depends(get_notification_manager)
):
    """Create a new notification rule"""
    try:
        user_id = current_user["user_id"]
        
        # Create rule
        rule = NotificationRule(
            rule_id=str(uuid.uuid4()),
            user_id=user_id,
            name=request.name,
            notification_type=request.notification_type,
            conditions=request.conditions,
            channels=request.channels,
            priority=request.priority
        )
        
        success = await notification_manager.create_rule(rule)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create notification rule")
        
        # Start monitoring if not already running
        background_tasks.add_task(notification_manager.start_monitoring)
        
        return NotificationRuleResponse(
            rule_id=rule.rule_id,
            user_id=rule.user_id,
            name=rule.name,
            notification_type=rule.notification_type,
            conditions=rule.conditions,
            channels=rule.channels,
            priority=rule.priority,
            is_active=rule.is_active,
            created_at=rule.created_at,
            last_triggered=rule.last_triggered,
            trigger_count=rule.trigger_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create notification rule", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create notification rule: {str(e)}")


@router.get("/rules", response_model=List[NotificationRuleResponse])
async def get_notification_rules(
    current_user: dict = Depends(get_current_user),
    notification_manager: NotificationManager = Depends(get_notification_manager)
):
    """Get all notification rules for the current user"""
    try:
        user_id = current_user["user_id"]
        rules = await notification_manager.get_user_rules(user_id)
        
        return [
            NotificationRuleResponse(
                rule_id=rule.rule_id,
                user_id=rule.user_id,
                name=rule.name,
                notification_type=rule.notification_type,
                conditions=rule.conditions,
                channels=rule.channels,
                priority=rule.priority,
                is_active=rule.is_active,
                created_at=rule.created_at,
                last_triggered=rule.last_triggered,
                trigger_count=rule.trigger_count
            )
            for rule in rules
        ]
        
    except Exception as e:
        logger.error("Failed to get notification rules", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get notification rules: {str(e)}")


@router.get("/stats")
async def get_notification_stats(
    current_user: dict = Depends(get_current_user),
    notification_manager: NotificationManager = Depends(get_notification_manager)
):
    """Get notification statistics for the current user"""
    try:
        user_id = current_user["user_id"]
        
        rules = await notification_manager.get_user_rules(user_id)
        notifications = await notification_manager.get_user_notifications(user_id, 1000)
        
        return {
            "total_rules": len(rules),
            "active_rules": len([r for r in rules if r.is_active]),
            "total_notifications": len(notifications),
            "unread_notifications": len([n for n in notifications if not n.is_read])
        }
        
    except Exception as e:
        logger.error("Failed to get notification stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get notification stats: {str(e)}")


# WebSocket endpoint for real-time notifications
@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    notification_manager: NotificationManager = Depends(get_notification_manager)
):
    """WebSocket endpoint for real-time notifications"""
    await websocket.accept()
    user_id = None
    
    try:
        # Expect authentication message first
        auth_message = await websocket.receive_json()
        
        if auth_message.get("type") != "auth":
            await websocket.close(code=4001, reason="Authentication required")
            return
        
        # In production, validate the token properly
        user_id = "user_123"  # Placeholder - extract from validated token
        
        # Register WebSocket connection
        notification_manager.register_websocket(user_id, websocket)
        
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connection established"
        })
        
        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_json()
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        # Unregister WebSocket connection
        if user_id:
            notification_manager.unregister_websocket(user_id)