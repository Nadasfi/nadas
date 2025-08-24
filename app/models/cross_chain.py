"""
Cross-Chain Models for Database Integration
SQLAlchemy models for orchestrator and strategy management
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()

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

class CrossChainStrategy(Base):
    """Cross-chain strategy execution records"""
    
    __tablename__ = "cross_chain_strategies"
    
    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_address = Column(String(42), nullable=False, index=True)
    
    # Strategy configuration
    source_chain = Column(String(50), nullable=False)
    target_chain = Column(String(50), nullable=False, default="hyperliquid")
    source_token = Column(String(20), nullable=False)
    target_token = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    risk_tolerance = Column(String(20), default="medium")
    
    # Status and execution
    status = Column(String(30), default=StrategyStatus.PENDING.value, index=True)
    strategy_config = Column(JSON, default=dict)
    automation_rules_config = Column(JSON, default=list)
    ai_generated = Column(Boolean, default=False)
    
    # Route and execution data
    route_quotes = Column(JSON, default=list)
    selected_route_data = Column(JSON, nullable=True)
    selected_route_index = Column(Integer, nullable=True)
    
    # Costs and timing
    total_fees_usd = Column(Float, default=0.0)
    estimated_completion = Column(DateTime, nullable=True)
    actual_completion = Column(DateTime, nullable=True)
    
    # AI analysis results
    ai_confidence_score = Column(Float, nullable=True)
    ai_analysis_data = Column(JSON, nullable=True)
    risk_assessment_data = Column(JSON, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    bridge_transactions = relationship("BridgeTransaction", back_populates="strategy", cascade="all, delete-orphan")
    execution_logs = relationship("StrategyExecutionLog", back_populates="strategy", cascade="all, delete-orphan")
    automation_rules = relationship("StrategyAutomationRule", back_populates="strategy", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_strategy_user_status', 'user_address', 'status'),
        Index('idx_strategy_created_at', 'created_at'),
        Index('idx_strategy_chains', 'source_chain', 'target_chain'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary"""
        return {
            "id": str(self.id),
            "user_address": self.user_address,
            "status": self.status,
            "strategy_config": {
                "source_chain": self.source_chain,
                "target_chain": self.target_chain,
                "source_token": self.source_token,
                "target_token": self.target_token,
                "amount": self.amount,
                "risk_tolerance": self.risk_tolerance,
                "automation_rules": self.automation_rules_config,
                "ai_generated": self.ai_generated
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "route_quotes": self.route_quotes,
            "selected_route": self.selected_route_data,
            "total_fees_usd": self.total_fees_usd,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "actual_completion": self.actual_completion.isoformat() if self.actual_completion else None,
            "ai_confidence_score": self.ai_confidence_score,
            "error_message": self.error_message,
            "progress_percentage": self._calculate_progress()
        }
    
    def _calculate_progress(self) -> int:
        """Calculate completion percentage based on status"""
        progress_map = {
            StrategyStatus.PENDING.value: 0,
            StrategyStatus.ANALYZING.value: 10,
            StrategyStatus.QUOTE_READY.value: 20,
            StrategyStatus.EXECUTING_BRIDGE.value: 40,
            StrategyStatus.WAITING_CONFIRMATION.value: 60,
            StrategyStatus.BRIDGE_COMPLETED.value: 80,
            StrategyStatus.EXECUTING_TARGET.value: 90,
            StrategyStatus.COMPLETED.value: 100,
            StrategyStatus.FAILED.value: 0,
            StrategyStatus.CANCELLED.value: 0
        }
        return progress_map.get(self.status, 0)

class BridgeTransaction(Base):
    """Bridge transaction records for cross-chain transfers"""
    
    __tablename__ = "bridge_transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('cross_chain_strategies.id'), nullable=False, index=True)
    
    # Transaction details
    source_tx_hash = Column(String(66), nullable=True, index=True)
    target_tx_hash = Column(String(66), nullable=True, index=True)
    bridge_provider = Column(String(50), nullable=False)
    
    # Transaction data
    source_chain = Column(String(50), nullable=False)
    target_chain = Column(String(50), nullable=False)
    source_token = Column(String(20), nullable=False)
    target_token = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    
    # Status and costs
    status = Column(String(30), default="pending", index=True)
    fee_usd = Column(Float, default=0.0)
    gas_fee_usd = Column(Float, default=0.0)
    actual_output_amount = Column(Float, nullable=True)
    
    # Timing
    estimated_time_minutes = Column(Integer, nullable=True)
    execution_started_at = Column(DateTime, nullable=True)
    execution_completed_at = Column(DateTime, nullable=True)
    
    # Technical details
    route_data = Column(JSON, nullable=True)
    slippage_tolerance = Column(Float, default=0.01)
    price_impact = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    strategy = relationship("CrossChainStrategy", back_populates="bridge_transactions")
    
    # Indexes
    __table_args__ = (
        Index('idx_bridge_tx_hashes', 'source_tx_hash', 'target_tx_hash'),
        Index('idx_bridge_provider_status', 'bridge_provider', 'status'),
        Index('idx_bridge_created_at', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bridge transaction to dictionary"""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "source_tx_hash": self.source_tx_hash,
            "target_tx_hash": self.target_tx_hash,
            "bridge_provider": self.bridge_provider,
            "source_chain": self.source_chain,
            "target_chain": self.target_chain,
            "source_token": self.source_token,
            "target_token": self.target_token,
            "amount": self.amount,
            "status": self.status,
            "fee_usd": self.fee_usd,
            "gas_fee_usd": self.gas_fee_usd,
            "actual_output_amount": self.actual_output_amount,
            "estimated_time_minutes": self.estimated_time_minutes,
            "execution_started_at": self.execution_started_at.isoformat() if self.execution_started_at else None,
            "execution_completed_at": self.execution_completed_at.isoformat() if self.execution_completed_at else None,
            "slippage_tolerance": self.slippage_tolerance,
            "price_impact": self.price_impact,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class StrategyExecutionLog(Base):
    """Execution log entries for strategies"""
    
    __tablename__ = "strategy_execution_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('cross_chain_strategies.id'), nullable=False, index=True)
    
    # Log details
    log_level = Column(String(20), nullable=False, default="info", index=True)  # info, warning, error
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    
    # Context
    execution_step = Column(String(50), nullable=True)
    provider = Column(String(50), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    strategy = relationship("CrossChainStrategy", back_populates="execution_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_log_strategy_level', 'strategy_id', 'log_level'),
        Index('idx_log_created_at', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary"""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "log_level": self.log_level,
            "message": self.message,
            "details": self.details,
            "execution_step": self.execution_step,
            "provider": self.provider,
            "created_at": self.created_at.isoformat()
        }

class StrategyAutomationRule(Base):
    """Automation rules linked to cross-chain strategies"""
    
    __tablename__ = "strategy_automation_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('cross_chain_strategies.id'), nullable=False, index=True)
    
    # Rule details
    rule_type = Column(String(50), nullable=False)  # dca, stop_loss, take_profit, etc.
    rule_config = Column(JSON, nullable=False)
    
    # Status
    status = Column(String(30), default="pending", index=True)  # pending, active, paused, completed, failed
    external_rule_id = Column(String(100), nullable=True, index=True)  # ID from automation engine
    
    # Execution tracking
    execution_count = Column(Integer, default=0)
    last_execution = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    strategy = relationship("CrossChainStrategy", back_populates="automation_rules")
    
    # Indexes
    __table_args__ = (
        Index('idx_automation_strategy_type', 'strategy_id', 'rule_type'),
        Index('idx_automation_status', 'status'),
        Index('idx_automation_external_id', 'external_rule_id'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert automation rule to dictionary"""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "rule_type": self.rule_type,
            "rule_config": self.rule_config,
            "status": self.status,
            "external_rule_id": self.external_rule_id,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class OrchestratorStatistics(Base):
    """Daily aggregated statistics for the orchestrator"""
    
    __tablename__ = "orchestrator_statistics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Date and metrics
    date = Column(DateTime, nullable=False, index=True)
    
    # Strategy counts
    total_strategies = Column(Integer, default=0)
    completed_strategies = Column(Integer, default=0)
    failed_strategies = Column(Integer, default=0)
    cancelled_strategies = Column(Integer, default=0)
    
    # Provider usage
    lifi_usage_count = Column(Integer, default=0)
    gluex_usage_count = Column(Integer, default=0)
    liquid_labs_usage_count = Column(Integer, default=0)
    
    # Financial metrics
    total_volume_usd = Column(Float, default=0.0)
    total_fees_usd = Column(Float, default=0.0)
    average_fee_usd = Column(Float, default=0.0)
    
    # Performance metrics
    average_execution_time_minutes = Column(Float, default=0.0)
    success_rate_percentage = Column(Float, default=0.0)
    
    # AI metrics
    ai_generated_strategies = Column(Integer, default=0)
    average_ai_confidence = Column(Float, default=0.0)
    
    # User metrics
    unique_users = Column(Integer, default=0)
    new_users = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_stats_date', 'date'),
        Index('idx_stats_created_at', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            "id": str(self.id),
            "date": self.date.isoformat(),
            "total_strategies": self.total_strategies,
            "completed_strategies": self.completed_strategies,
            "failed_strategies": self.failed_strategies,
            "cancelled_strategies": self.cancelled_strategies,
            "provider_usage": {
                "lifi": self.lifi_usage_count,
                "gluex": self.gluex_usage_count,
                "liquid_labs": self.liquid_labs_usage_count
            },
            "total_volume_usd": self.total_volume_usd,
            "total_fees_usd": self.total_fees_usd,
            "average_fee_usd": self.average_fee_usd,
            "average_execution_time_minutes": self.average_execution_time_minutes,
            "success_rate_percentage": self.success_rate_percentage,
            "ai_generated_strategies": self.ai_generated_strategies,
            "average_ai_confidence": self.average_ai_confidence,
            "unique_users": self.unique_users,
            "new_users": self.new_users,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }