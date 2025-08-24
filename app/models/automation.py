"""
Simple automation models (no complex workflows)
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, DateTime, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class AutomationRule(Base):
    """Simple automation rule model"""
    
    __tablename__ = "automation_rules"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    automation_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "dca", "stop_loss", etc.
    config: Mapped[dict] = mapped_column(JSON, nullable=False)  # JSON configuration
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_executed: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="automation_rules")
    executions = relationship("AutomationExecution", back_populates="rule", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<AutomationRule(name='{self.name}', type='{self.automation_type}')>"


class AutomationExecution(Base):
    """Automation execution tracking model"""
    
    __tablename__ = "automation_executions"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    rule_id: Mapped[UUID] = mapped_column(ForeignKey("automation_rules.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # "pending", "executing", "completed", "failed"
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    transaction_hash: Mapped[Optional[str]] = mapped_column(String(66), nullable=True)
    gas_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    execution_details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    
    # Relationships
    rule = relationship("AutomationRule", back_populates="executions")
    
    def __repr__(self) -> str:
        return f"<AutomationExecution(rule_id='{self.rule_id}', status='{self.status}')>"
