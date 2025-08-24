"""
AI interaction models
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class AIConversation(Base):
    """AI conversation tracking model"""
    
    __tablename__ = "ai_conversations"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    messages: Mapped[list] = mapped_column(JSON, default=list, nullable=False)  # JSON array of messages
    context_type: Mapped[str] = mapped_column(String(50), default="general", nullable=False)  # "general", "portfolio", "strategy", "risk"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="ai_conversations")
    
    def __repr__(self) -> str:
        return f"<AIConversation(user_id='{self.user_id}', context='{self.context_type}')>"
