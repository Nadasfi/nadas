"""
AI Assistant API Endpoints
Claude 3.5 Sonnet powered automation analysis
"""

from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import structlog

from app.core.dependencies import get_current_user_address
from app.services.ai_service import get_ai_service, ClaudeAIService
from app.adapters.hyperliquid import HyperliquidAdapter

logger = structlog.get_logger(__name__)

router = APIRouter()

class AnalyzeAutomationRequest(BaseModel):
    """Request to analyze automation setup"""
    user_request: str = Field(..., description="User's automation request in natural language")
    market_data: Dict[str, Any] = Field(default_factory=dict, description="Current market data")
    portfolio_data: Dict[str, Any] = Field(default_factory=dict, description="User's portfolio data")

class RiskWarningRequest(BaseModel):
    """Request to generate risk warning"""
    automation_config: Dict[str, Any] = Field(..., description="Automation configuration")
    market_conditions: Dict[str, Any] = Field(default_factory=dict, description="Current market conditions")

@router.post("/analyze-automation")
async def analyze_automation_request(
    request: AnalyzeAutomationRequest,
    user_address: str = Depends(get_current_user_address),
    ai_service: ClaudeAIService = Depends(get_ai_service)
):
    """
    Analyze user's automation request using Claude 3.5 Sonnet
    Provides intelligent recommendations and risk assessment
    """
    try:
        analysis = await ai_service.analyze_automation_request(
            user_request=request.user_request,
            market_data=request.market_data,
            portfolio_data=request.portfolio_data
        )
        
        return {
            "success": True,
            "data": analysis,
            "user_address": user_address
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze automation request: {str(e)}"
        )

@router.post("/risk-warning")
async def generate_risk_warning(
    request: RiskWarningRequest,
    user_address: str = Depends(get_current_user_address),
    ai_service: ClaudeAIService = Depends(get_ai_service)
):
    """
    Generate dynamic risk warning based on automation config and market conditions
    """
    try:
        warning = await ai_service.generate_risk_warning(
            automation_config=request.automation_config,
            current_market_conditions=request.market_conditions
        )
        
        return {
            "success": True,
            "data": {
                "warning": warning,
                "timestamp": "2025-08-21T20:55:00Z",
                "user_address": user_address
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate risk warning: {str(e)}"
        )

@router.get("/market-explanation")
async def explain_market_conditions(
    timeframe: str = "1h",
    user_address: str = Depends(get_current_user_address),
    ai_service: ClaudeAIService = Depends(get_ai_service)
):
    """
    Get AI-powered market condition explanation
    """
    try:
        # Mock market data - in production, get from real-time data sources
        market_data = {
            "ETH": {"price": 3456.78, "change_24h": 2.3, "volume": "1.2B"},
            "BTC": {"price": 98234.56, "change_24h": -1.2, "volume": "890M"},
            "SOL": {"price": 234.12, "change_24h": 4.1, "volume": "450M"}
        }
        
        explanation = await ai_service.explain_market_conditions(
            market_data=market_data,
            timeframe=timeframe
        )
        
        return {
            "success": True,
            "data": {
                "explanation": explanation,
                "timeframe": timeframe,
                "market_data": market_data,
                "user_address": user_address
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to explain market conditions: {str(e)}"
        )

@router.get("/health")
async def ai_health_check(
    ai_service: ClaudeAIService = Depends(get_ai_service)
):
    """
    Check AI service health and Bedrock connectivity
    """
    health_status = await ai_service.health_check()
    
    return {
        "success": True,
        "data": health_status
    }

@router.get("/capabilities")
async def get_ai_capabilities():
    """
    Get available AI capabilities and features
    """
    return {
        "success": True,
        "data": {
            "models": {
                "primary": "Claude 3.5 Sonnet v2",
                "fast": "Claude 3 Haiku",
                "provider": "AWS Bedrock"
            },
            "capabilities": [
                "Automation strategy analysis",
                "Risk assessment and warnings", 
                "Market condition explanation",
                "Portfolio rebalancing suggestions",
                "Educational content generation",
                "Real-time market insights"
            ],
            "languages": ["English", "Turkish"],
            "max_tokens": 4000,
            "response_time": "1-3 seconds"
        }
    }


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's chat message")

@router.post("/chat-enhanced")
async def chat_with_ai_enhanced(
    request: ChatRequest,
    user_address: str = Depends(get_current_user_address),
    ai_service: ClaudeAIService = Depends(get_ai_service)
):
    """
    Enhanced AI chat with Nadas.fi context and real market data
    """
    try:
        # Get real market data from Hyperliquid
        hl_adapter = HyperliquidAdapter()
        
        # Fetch current market prices
        market_data = {}
        try:
            for symbol in ['ETH', 'BTC', 'SOL', 'ARB', 'OP', 'AVAX']:
                try:
                    price_info = await hl_adapter.get_price_data(symbol)
                    if price_info:
                        market_data[symbol] = price_info
                except:
                    continue
        except Exception as e:
            logger.warning("Could not fetch all market data", error=str(e))
        
        # Get user portfolio data if available
        portfolio_data = {}
        try:
            if user_address:
                portfolio_data = await hl_adapter.get_account_summary(user_address)
        except Exception as e:
            logger.warning("Could not fetch portfolio data", error=str(e))
        
        # Chat with AI using context
        response = await ai_service.chat_with_context(
            user_message=request.message,
            market_data=market_data,
            portfolio_data=portfolio_data
        )
        
        return {
            "success": True,
            "response": response,
            "context": {
                "market_data_available": len(market_data) > 0,
                "portfolio_data_available": len(portfolio_data) > 0,
                "user_address": user_address
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Enhanced chat error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"AI chat error: {str(e)}"
        )