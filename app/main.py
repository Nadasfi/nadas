"""
Nadas.fi - Hyperliquid DeFi Automation Platform
Main FastAPI application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.redis_client import get_redis_health
from app.api.v1 import auth, live_trading, automation_engine, sub_account_automation, ai_assistant, trading
from app.api.v1 import automation_simple, automation_wallets, simulator, lifi, twap, demo
from app.api.v1 import portfolio_simple as portfolio, websocket_api, orchestrator  # notifications, 
# , automation, simulation, ai, admin, advanced_trading, gluex, liquid_labs
from app.services.websocket_manager import initialize_websocket_subscriptions

# Configure structured logging
configure_logging()
logger = structlog.get_logger()

# Create FastAPI application
app = FastAPI(
    title="Nadas.fi API",
    description="Hyperliquid DeFi Automation Platform API",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.ALLOWED_HOSTS == "*" else settings.ALLOWED_HOSTS.split(",")
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["authentication"]
)

# Portfolio router re-enabled
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])

# Live Trading router - Real execution with risk management  
app.include_router(live_trading.router, prefix="/api/v1/live-trading", tags=["live-trading"])

# Automation Engine router - User wallet delegation and automation
app.include_router(automation_engine.router, prefix="/api/v1/automation", tags=["automation"])

# Sub-Account Automation router - Secure sub-account based automation
app.include_router(sub_account_automation.router, prefix="/api/v1/sub-accounts", tags=["sub-accounts"])

# AI Assistant router - Claude 3.5 Sonnet powered automation analysis
app.include_router(ai_assistant.router, prefix="/api/v1/ai", tags=["ai-assistant"])

# WebSocket API router - Real-time data streaming
app.include_router(websocket_api.router, prefix="/api/v1/websocket", tags=["websocket-realtime"])

# Notifications router - Temporarily disabled due to dependency injection issue
# app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["notifications"])

# Trading router - Client-side wallet integration
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])

# Simple Automation router - Notification-based automation
app.include_router(automation_simple.router, prefix="/api/v1/automation-simple", tags=["automation-simple"])

# Automation Wallets router - Hybrid wallet management
app.include_router(automation_wallets.router, prefix="/api/v1/automation-wallets", tags=["automation-wallets"])

# HyperEVM Transaction Simulator router - $30k hackathon bounty
app.include_router(simulator.router, prefix="/api/v1/simulator", tags=["hyperevm-simulator"])

# LI.FI Cross-chain router - $5k hackathon bounty
app.include_router(lifi.router, prefix="/api/v1/lifi", tags=["lifi-cross-chain"])

# TWAP Order Executor router - $5k hackathon bounty
app.include_router(twap.router, prefix="/api/v1/twap", tags=["twap-executor"])
app.include_router(demo.router, prefix="/api/v1/demo", tags=["demo"])

# Cross-Chain Orchestrator router - AI-powered cross-chain strategies
app.include_router(orchestrator.router, prefix="/api/v1/orchestrator", tags=["cross-chain-orchestrator"])

# Other routers temporarily disabled for debugging
# app.include_router(automation.router, prefix="/api/v1/automation", tags=["automation"])
# app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["simulation"])
# app.include_router(ai.router, prefix="/api/v1/ai", tags=["ai"])
# app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
# app.include_router(advanced_trading.router, prefix="/api/v1/advanced-trading", tags=["advanced-trading"])
# app.include_router(gluex.router, prefix="/api/v1/gluex", tags=["gluex-cross-chain"])
# app.include_router(liquid_labs.router, prefix="/api/v1/liquid-labs", tags=["liquid-labs-hyperevm"])


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    try:
        logger.info("Nadas.fi API started successfully")
        
        # Initialize WebSocket subscriptions for real-time data
        await initialize_websocket_subscriptions()
        
    except Exception as e:
        logger.error("Error during application startup", error=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    try:
        logger.info("Application shutdown initiated")
        
        # WebSocket cleanup
        from app.services.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        await ws_manager.cleanup()
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error("Error during application shutdown", error=str(e))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Nadas.fi API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    from app.core.supabase import supabase_health_check
    
    supabase_health = await supabase_health_check()
    redis_health = await get_redis_health()
    
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
        "services": {
            "supabase": supabase_health,
            "redis": redis_health
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    logger.error("HTTP exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("Unhandled exception", exception=str(exc), type=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "code": 500
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_config=None  # Use our custom logging config
    )
