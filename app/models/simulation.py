"""
Transaction simulation models
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, DateTime, Integer, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class SimulationJob(Base):
    """Transaction simulation job model"""
    
    __tablename__ = "simulation_jobs"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    simulation_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "order", "strategy", "portfolio_stress"
    input_parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)  # "pending", "running", "completed", "failed"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="simulation_jobs")
    
    def __repr__(self) -> str:
        return f"<SimulationJob(type='{self.simulation_type}', status='{self.status}')>"
