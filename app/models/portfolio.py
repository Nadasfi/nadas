"""
Portfolio tracking models - Updated for Real Hyperliquid SDK Integration
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import String, Float, DateTime, ForeignKey, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Portfolio(Base):
    """Portfolio tracking model - Updated for Hyperliquid SDK"""
    
    __tablename__ = "portfolios"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    hyperliquid_address: Mapped[str] = mapped_column(String(42), nullable=False, unique=True)
    
    # Account Value (from marginSummary)
    account_value: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_margin_used: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_ntl_pos: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_raw_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Deprecated fields (kept for backward compatibility)
    total_equity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    margin_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Metadata
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_sync_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    spot_balances = relationship("SpotBalance", back_populates="portfolio", cascade="all, delete-orphan")
    trade_history = relationship("TradeHistory", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Portfolio(address='{self.hyperliquid_address}', account_value={self.account_value})>"


class Position(Base):
    """Individual perpetual position model - Updated for Hyperliquid SDK"""
    
    __tablename__ = "positions"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    
    # Hyperliquid specific fields
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., "ETH", "BTC"
    asset_id: Mapped[int] = mapped_column(Integer, nullable=False)  # Hyperliquid asset ID
    
    # Position data (from assetPositions API response)
    size: Mapped[float] = mapped_column(Float, nullable=False)  # szi field
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)  # entryPx
    mark_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False)  # unrealizedPnl
    leverage: Mapped[float] = mapped_column(Float, nullable=False)  # leverage.value
    liquidation_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # liquidationPx
    
    # Computed fields
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # "long" or "short"
    position_value: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Raw API response for debugging
    raw_api_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Metadata
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    
    def __repr__(self) -> str:
        return f"<Position(symbol='{self.symbol}', size={self.size}, side='{self.side}')>"


class SpotBalance(Base):
    """Spot token balance model - New for Hyperliquid SDK"""
    
    __tablename__ = "spot_balances"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    
    # Balance data (from spotClearinghouseState API response)
    coin: Mapped[str] = mapped_column(String(20), nullable=False)  # Token symbol
    total: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # Total balance
    hold: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # Amount on hold
    available: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # Available for use
    
    # Metadata
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="spot_balances")
    
    def __repr__(self) -> str:
        return f"<SpotBalance(coin='{self.coin}', total={self.total})>"


class TradeHistory(Base):
    """Trade history model - New for Hyperliquid SDK"""
    
    __tablename__ = "trade_history"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    
    # Trade data (from user_fills API response)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # "buy" or "sell"
    size: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Hyperliquid specific
    fill_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)  # Unique fill ID
    order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    liquidation: Mapped[bool] = mapped_column(default=False, nullable=False)
    
    # Timestamps
    trade_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Raw API response for debugging
    raw_fill_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="trade_history")
    
    def __repr__(self) -> str:
        return f"<TradeHistory(symbol='{self.symbol}', side='{self.side}', size={self.size})>"


class MarketData(Base):
    """Market data model - New for real-time price tracking"""
    
    __tablename__ = "market_data"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    
    # Market data
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    mid_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    bid: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    ask: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mark_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    index_price: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    funding_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    open_interest: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    volume_24h: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Metadata
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        return f"<MarketData(symbol='{self.symbol}', mid_price={self.mid_price})>"
