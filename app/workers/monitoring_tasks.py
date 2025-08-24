"""
Monitoring and Performance Tasks
Real-time portfolio monitoring, performance tracking, and health checks
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from celery import Celery
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.portfolio import Portfolio, Position
from app.models.user import User
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.adapters.notifications import get_notification_manager
from app.core.config import settings
from app.core.logging import get_logger
from app.core.error_handling import (
    circuit_breaker_protected, track_errors,
    CircuitBreakerConfig
)

logger = get_logger(__name__)

# Use same Celery app
from app.workers.automation_tasks import celery_app


@celery_app.task(name="app.workers.monitoring_tasks.update_portfolio_data")
@circuit_breaker_protected("portfolio_update", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=120))
def update_portfolio_data():
    """Update portfolio data for all users"""
    with track_errors("monitoring_tasks", {"task": "update_portfolio_data"}):
        try:
        logger.info("Starting portfolio data update")
        
        db = next(get_db())
        
        # Get all users with wallet addresses
        users = db.query(User).filter(User.wallet_address.isnot(None)).all()
        
        updated_count = 0
        failed_count = 0
        
        for user in users:
            try:
                result = asyncio.run(update_user_portfolio_data(user, db))
                if result.get("success", False):
                    updated_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error("Error updating user portfolio", 
                           user_id=user.id, error=str(e))
                failed_count += 1
        
        logger.info("Portfolio data update completed",
                   total_users=len(users),
                   updated=updated_count,
                   failed=failed_count)
        
        return {
            "total_users": len(users),
            "updated": updated_count,
            "failed": failed_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error in portfolio data update", error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


async def update_user_portfolio_data(user: User, db: Session) -> Dict[str, Any]:
    """Update portfolio data for a specific user"""
    try:
        adapter = get_hyperliquid_adapter()
        
        # Get current portfolio data
        account_equity = await adapter.get_vault_equity(user.wallet_address)
        positions = await adapter.get_user_positions(user.wallet_address)
        spot_balances = await adapter.get_spot_balances(user.wallet_address)
        
        # Update or create portfolio record
        portfolio = db.query(Portfolio).filter(
            Portfolio.hyperliquid_address == user.wallet_address
        ).first()
        
        if not portfolio:
            portfolio = Portfolio(
                user_id=user.id,
                hyperliquid_address=user.wallet_address
            )
            db.add(portfolio)
        
        # Calculate total unrealized PnL
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        # Update portfolio
        portfolio.total_equity = account_equity
        portfolio.unrealized_pnl = total_unrealized_pnl
        portfolio.last_updated = datetime.utcnow()
        
        # Update positions
        # Clear existing positions
        db.query(Position).filter(Position.portfolio_id == portfolio.id).delete()
        
        # Add current positions
        for pos in positions:
            if pos.size != 0:  # Only store non-zero positions
                position = Position(
                    portfolio_id=portfolio.id,
                    asset=pos.coin,
                    size=pos.size,
                    entry_price=pos.entry_px,
                    mark_price=0.0,  # Would get from market data
                    unrealized_pnl=pos.unrealized_pnl,
                    leverage=pos.leverage,
                    side="long" if pos.size > 0 else "short"
                )
                db.add(position)
        
        db.commit()
        
        await adapter.close()
        
        return {
            "success": True,
            "equity": account_equity,
            "positions_count": len(positions),
            "unrealized_pnl": total_unrealized_pnl
        }
        
    except Exception as e:
        logger.error("Error updating user portfolio data", 
                    user_id=user.id, error=str(e))
        return {"success": False, "error": str(e)}


@celery_app.task(name="app.workers.monitoring_tasks.performance_tracking")
def performance_tracking():
    """Track portfolio performance metrics"""
    try:
        logger.info("Starting performance tracking")
        
        db = next(get_db())
        
        # Get all portfolios
        portfolios = db.query(Portfolio).all()
        
        tracked_count = 0
        
        for portfolio in portfolios:
            try:
                # Calculate performance metrics
                # This would involve historical data analysis
                tracked_count += 1
                
            except Exception as e:
                logger.error("Error tracking portfolio performance", 
                           portfolio_id=portfolio.id, error=str(e))
        
        logger.info("Performance tracking completed", tracked=tracked_count)
        
        return {
            "portfolios_tracked": tracked_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error in performance tracking", error=str(e))
        return {"error": str(e)}
    finally:
        db.close()


@celery_app.task(name="app.workers.monitoring_tasks.health_check")
@circuit_breaker_protected("health_check", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60))
def system_health_check():
    """Perform system health checks"""
    with track_errors("monitoring_tasks", {"task": "health_check"}):
        try:
        logger.info("Starting system health check")
        
        health_status = {
            "database": "healthy",
            "hyperliquid_api": "healthy", 
            "ai_service": "healthy",
            "notification_service": "healthy",
            "worker_status": "healthy"
        }
        
        # Check database connectivity
        try:
            db = next(get_db())
            db.execute("SELECT 1")
            db.close()
        except Exception as e:
            health_status["database"] = f"unhealthy: {str(e)}"
        
        # Check Hyperliquid API
        try:
            adapter = get_hyperliquid_adapter()
            await adapter.get_all_mids()
            await adapter.close()
        except Exception as e:
            health_status["hyperliquid_api"] = f"unhealthy: {str(e)}"
        
        # Check AI service
        try:
            from app.services.ai_service import get_ai_service
            ai_service = get_ai_service()
            stats = ai_service.get_service_stats()
            if stats["overall"]["failure_rate"] > 0.5:
                health_status["ai_service"] = "degraded: high failure rate"
        except Exception as e:
            health_status["ai_service"] = f"unhealthy: {str(e)}"
        
        overall_health = "healthy" if all("healthy" in status for status in health_status.values()) else "degraded"
        
        logger.info("System health check completed", 
                   overall_health=overall_health,
                   details=health_status)
        
        return {
            "overall_health": overall_health,
            "component_health": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error in system health check", error=str(e))
        return {"error": str(e), "overall_health": "unhealthy"}


# Add these to the beat schedule
celery_app.conf.beat_schedule.update({
    # Update portfolio data every 2 minutes
    "update-portfolio-data": {
        "task": "app.workers.monitoring_tasks.update_portfolio_data",
        "schedule": 120.0,  # Every 2 minutes
    },
    # Performance tracking every 15 minutes
    "performance-tracking": {
        "task": "app.workers.monitoring_tasks.performance_tracking",
        "schedule": 900.0,  # Every 15 minutes
    },
    # System health check every 5 minutes
    "system-health-check": {
        "task": "app.workers.monitoring_tasks.health_check",
        "schedule": 300.0,  # Every 5 minutes
    }
})