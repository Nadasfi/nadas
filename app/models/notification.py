"""
Notification system models
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class NotificationRule(Base):
    """Notification rule model"""
    
    __tablename__ = "notification_rules"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "price_alert", "liquidation_risk", etc.
    trigger_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    notification_channels: Mapped[list] = mapped_column(JSON, nullable=False)  # ["email", "telegram", "discord"]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="notification_rules")
    logs = relationship("NotificationLog", back_populates="rule", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<NotificationRule(trigger_type='{self.trigger_type}')>"


class NotificationLog(Base):
    """Notification delivery log model"""
    
    __tablename__ = "notification_logs"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    rule_id: Mapped[UUID] = mapped_column(ForeignKey("notification_rules.id", ondelete="CASCADE"), nullable=False)
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # "sent", "failed", "pending"
    message: Mapped[str] = mapped_column(String(2000), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    
    # Relationships
    rule = relationship("NotificationRule", back_populates="logs")
    
    def __repr__(self) -> str:
        return f"<NotificationLog(channel='{self.channel}', status='{self.status}')>"
