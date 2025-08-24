"""
Notification Delivery Tasks
Background tasks for sending emails, push notifications, and alerts
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from celery import Celery
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.notification import NotificationRule, NotificationLog
from app.models.user import User
from app.core.config import settings
from app.core.logging import get_logger
from app.core.error_handling import (
    circuit_breaker_protected, track_errors,
    CircuitBreakerConfig
)

logger = get_logger(__name__)

# Use same Celery app
from app.workers.automation_tasks import celery_app


@celery_app.task(name="app.workers.notification_tasks.send_email_notification")
@circuit_breaker_protected("email_notification", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=300))
def send_email_notification(user_email: str, subject: str, message: str, notification_data: Dict[str, Any] = None):
    """Send email notification"""
    with track_errors("notification_tasks", {"task": "send_email", "email": user_email}):
        try:
        if not hasattr(settings, 'SMTP_SERVER') or not settings.SMTP_SERVER:
            logger.warning("SMTP not configured, skipping email notification")
            return {"success": False, "reason": "SMTP not configured"}
        
        # Create email message
        msg = MimeMultipart()
        msg['From'] = settings.SMTP_USERNAME
        msg['To'] = user_email
        msg['Subject'] = f"Nadas.fi - {subject}"
        
        # Create HTML email body
        html_body = f"""
        <html>
        <body>
            <h2>Nadas.fi Notification</h2>
            <h3>{subject}</h3>
            <p>{message}</p>
            
            {f'<pre>{json.dumps(notification_data, indent=2)}</pre>' if notification_data else ''}
            
            <hr>
            <p>Best regards,<br>Nadas.fi Team</p>
            <p><small>This is an automated notification from your Nadas.fi automation system.</small></p>
        </body>
        </html>
        """
        
        msg.attach(MimeText(html_body, 'html'))
        
        # Send email
        server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        logger.info("Email notification sent successfully", 
                   email=user_email, subject=subject)
        
        return {
            "success": True,
            "channel": "email",
            "recipient": user_email,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error sending email notification", 
                    email=user_email, error=str(e))
        return {"success": False, "error": str(e)}


@celery_app.task(name="app.workers.notification_tasks.send_push_notification")
def send_push_notification(user_device_token: str, title: str, message: str, notification_data: Dict[str, Any] = None):
    """Send push notification"""
    try:
        # This would integrate with Firebase FCM or similar service
        # For now, just log the notification
        
        logger.info("Push notification sent (simulated)", 
                   title=title, message=message)
        
        return {
            "success": True,
            "channel": "push",
            "title": title,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error sending push notification", error=str(e))
        return {"success": False, "error": str(e)}


@celery_app.task(name="app.workers.notification_tasks.send_telegram_notification")
def send_telegram_notification(chat_id: str, message: str, notification_data: Dict[str, Any] = None):
    """Send Telegram notification"""
    try:
        # This would integrate with Telegram Bot API
        # For now, just log the notification
        
        logger.info("Telegram notification sent (simulated)", 
                   chat_id=chat_id, message=message)
        
        return {
            "success": True,
            "channel": "telegram",
            "chat_id": chat_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error sending Telegram notification", error=str(e))
        return {"success": False, "error": str(e)}


@celery_app.task(name="app.workers.notification_tasks.send_discord_notification")
def send_discord_notification(webhook_url: str, message: str, notification_data: Dict[str, Any] = None):
    """Send Discord notification via webhook"""
    try:
        import requests
        
        payload = {
            "content": f"**Nadas.fi Notification**\\n{message}",
            "embeds": [{
                "title": "Automation Alert",
                "description": message,
                "color": 0x00ff00,  # Green
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Nadas.fi Automation System"}
            }]
        }
        
        if notification_data:
            payload["embeds"][0]["fields"] = [
                {"name": "Details", "value": f"```json\\n{json.dumps(notification_data, indent=2)}\\n```"}
            ]
        
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        
        logger.info("Discord notification sent successfully")
        
        return {
            "success": True,
            "channel": "discord",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error sending Discord notification", error=str(e))
        return {"success": False, "error": str(e)}


@celery_app.task(name="app.workers.notification_tasks.process_notification_queue")
def process_notification_queue():
    """Process pending notifications from the queue"""
    try:
        logger.info("Processing notification queue")
        
        db = next(get_db())
        
        # Get pending notifications (would typically be stored in a queue table)
        # For now, check for notification rules that need processing
        
        pending_notifications = db.query(NotificationLog).filter(
            NotificationLog.status == "pending"
        ).limit(100).all()
        
        processed_count = 0
        failed_count = 0
        
        for notification in pending_notifications:
            try:
                # Process notification based on channel
                result = None
                
                if notification.channel == "email":
                    # Get user email
                    rule = db.query(NotificationRule).filter(
                        NotificationRule.id == notification.rule_id
                    ).first()
                    
                    if rule:
                        user = db.query(User).filter(User.id == rule.user_id).first()
                        if user and user.email:
                            result = send_email_notification.delay(
                                user.email,
                                "Automation Alert",
                                notification.message
                            )
                
                elif notification.channel == "push":
                    result = send_push_notification.delay(
                        "device_token",  # Would get from user preferences
                        "Automation Alert",
                        notification.message
                    )
                
                elif notification.channel == "telegram":
                    result = send_telegram_notification.delay(
                        "chat_id",  # Would get from user preferences
                        notification.message
                    )
                
                elif notification.channel == "discord":
                    result = send_discord_notification.delay(
                        "webhook_url",  # Would get from user preferences
                        notification.message
                    )
                
                if result:
                    notification.status = "sent"
                    notification.sent_at = datetime.utcnow()
                    processed_count += 1
                else:
                    notification.status = "failed"
                    notification.error_message = "No suitable channel configuration"
                    failed_count += 1
                
            except Exception as e:
                notification.status = "failed"
                notification.error_message = str(e)
                failed_count += 1
                logger.error("Error processing notification", 
                           notification_id=notification.id, error=str(e))
        
        db.commit()
        
        logger.info("Notification queue processing completed",
                   processed=processed_count,
                   failed=failed_count)
        
        return {
            "processed": processed_count,
            "failed": failed_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error processing notification queue", error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="app.workers.notification_tasks.cleanup_old_notifications")
def cleanup_old_notifications():
    """Clean up old notification logs"""
    try:
        logger.info("Cleaning up old notifications")
        
        db = next(get_db())
        
        # Delete notifications older than 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        deleted_count = db.query(NotificationLog).filter(
            NotificationLog.sent_at < cutoff_date
        ).delete()
        
        db.commit()
        
        logger.info("Old notifications cleaned up", deleted_count=deleted_count)
        
        return {
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error cleaning up notifications", error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


# Add notification tasks to beat schedule
celery_app.conf.beat_schedule.update({
    # Process notification queue every 30 seconds
    "process-notification-queue": {
        "task": "app.workers.notification_tasks.process_notification_queue",
        "schedule": 30.0,
    },
    # Clean up old notifications daily
    "cleanup-old-notifications": {
        "task": "app.workers.notification_tasks.cleanup_old_notifications",
        "schedule": 86400.0,  # Every 24 hours
    }
})