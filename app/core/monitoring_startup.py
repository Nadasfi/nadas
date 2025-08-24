"""
Monitoring System Startup
Initialize health monitoring, circuit breakers, and error tracking on application startup
"""

import asyncio
import logging
from typing import Dict, Any

from app.core.error_handling import get_health_monitor, get_error_tracker
from app.core.logging import get_logger

logger = get_logger(__name__)


async def initialize_monitoring_system():
    """Initialize the complete monitoring system"""
    logger.info("Initializing monitoring system...")
    
    try:
        # Initialize global instances
        health_monitor = get_health_monitor()
        error_tracker = get_error_tracker()
        
        # Import all services to register their health checks
        logger.info("Registering service health checks...")
        
        # Portfolio service
        try:
            from app.services.portfolio_service import _register_portfolio_health_check
            logger.info("Portfolio service health check registered")
        except Exception as e:
            logger.warning("Failed to register portfolio service health check", error=str(e))
        
        # AI service  
        try:
            from app.services.ai_service import _register_ai_health_check
            logger.info("AI service health check registered")
        except Exception as e:
            logger.warning("Failed to register AI service health check", error=str(e))
        
        # Hyperliquid adapter
        try:
            from app.adapters.hyperliquid import _register_hyperliquid_health_check
            logger.info("Hyperliquid adapter health check registered")
        except Exception as e:
            logger.warning("Failed to register Hyperliquid adapter health check", error=str(e))
        
        # Database health check
        try:
            await register_database_health_check()
            logger.info("Database health check registered")
        except Exception as e:
            logger.warning("Failed to register database health check", error=str(e))
        
        # Redis health check (for Celery)
        try:
            await register_redis_health_check()
            logger.info("Redis health check registered")
        except Exception as e:
            logger.warning("Failed to register Redis health check", error=str(e))
        
        # Perform initial system health check
        logger.info("Performing initial system health check...")
        initial_health = await health_monitor.get_system_health()
        
        logger.info("Monitoring system initialization completed",
                   total_services=len(initial_health["services"]),
                   overall_status=initial_health["overall_status"])
        
        # Start background monitoring (optional)
        # asyncio.create_task(health_monitor.start_monitoring(interval=300))  # 5 minutes
        
        return {
            "success": True,
            "services_registered": len(initial_health["services"]),
            "system_status": initial_health["overall_status"]
        }
        
    except Exception as e:
        logger.error("Failed to initialize monitoring system", error=str(e))
        return {
            "success": False,
            "error": str(e)
        }


async def register_database_health_check():
    """Register database connectivity health check"""
    from app.core.error_handling import create_health_check
    from app.core.database import get_db
    
    async def check_database():
        """Check database connectivity"""
        try:
            db = next(get_db())
            result = db.execute("SELECT 1 as health_check").fetchone()
            db.close()
            return {
                "query_success": result is not None,
                "connection_pool": "active"
            }
        except Exception as e:
            raise e
    
    async def database_health_check():
        return await create_health_check("database", check_database)
    
    health_monitor = get_health_monitor()
    health_monitor.register_health_check("database", database_health_check)


async def register_redis_health_check():
    """Register Redis connectivity health check"""
    from app.core.error_handling import create_health_check
    import redis
    from app.core.config import settings
    
    async def check_redis():
        """Check Redis connectivity"""
        try:
            # Parse Redis URL from Celery broker URL
            broker_url = getattr(settings, 'CELERY_BROKER_URL', 'redis://localhost:6379/0')
            
            if broker_url.startswith('redis://'):
                # Extract Redis URL
                redis_client = redis.from_url(broker_url)
                ping_result = redis_client.ping()
                info = redis_client.info()
                redis_client.close()
                
                return {
                    "ping_success": ping_result,
                    "connected_clients": info.get("connected_clients", 0),
                    "memory_usage": info.get("used_memory_human", "unknown")
                }
            else:
                return {"error": "Non-Redis broker configured"}
                
        except Exception as e:
            raise e
    
    async def redis_health_check():
        return await create_health_check("redis", check_redis)
    
    health_monitor = get_health_monitor()
    health_monitor.register_health_check("redis", redis_health_check)


# Convenience function for FastAPI startup
def create_startup_handler():
    """Create FastAPI startup event handler"""
    async def startup_handler():
        await initialize_monitoring_system()
    
    return startup_handler


# Health check endpoints data
def get_monitoring_endpoints() -> Dict[str, str]:
    """Get monitoring endpoint descriptions"""
    return {
        "/api/v1/portfolio/system-health": "Comprehensive system health status",
        "/health": "Basic application health check",  # Would need to be created
        "/metrics": "Prometheus-style metrics"  # Would need to be created
    }


# Circuit breaker configurations for different services
def get_default_circuit_breaker_configs() -> Dict[str, Dict[str, Any]]:
    """Get recommended circuit breaker configurations"""
    return {
        "portfolio_service": {
            "failure_threshold": 3,
            "recovery_timeout": 30,
            "timeout": 30
        },
        "ai_service": {
            "failure_threshold": 3,
            "recovery_timeout": 60,
            "timeout": 45
        },
        "hyperliquid_adapter": {
            "failure_threshold": 5,
            "recovery_timeout": 20,
            "timeout": 15
        },
        "database": {
            "failure_threshold": 2,
            "recovery_timeout": 10,
            "timeout": 5
        },
        "notification_service": {
            "failure_threshold": 3,
            "recovery_timeout": 300,  # 5 minutes for email issues
            "timeout": 30
        }
    }


if __name__ == "__main__":
    # Run monitoring initialization standalone for testing
    asyncio.run(initialize_monitoring_system())