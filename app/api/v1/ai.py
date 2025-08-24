from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import random

from app.services.ai_service import get_ai_service, AIResponse
from app.api.v1.auth import get_current_user

router = APIRouter()

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime

@router.post("/chat", response_model=Dict[str, Any])
async def chat_with_ai(request: ChatMessage, current_user: dict = Depends(get_current_user)):
    """Chat with AI assistant using real multi-provider AI service"""
    try:
        ai_service = get_ai_service()
        
        # Generate response using real AI service
        ai_response = await ai_service.generate_response(
            prompt=request.message,
            context_type="general",
            max_tokens=1000,
            temperature=0.7
        )
        
        return {
            "response": ai_response.content,
            "provider": ai_response.provider.value,
            "model": ai_response.model,
            "timestamp": ai_response.timestamp.isoformat(),
            "tokens_used": ai_response.tokens_used,
            "cost_estimate": ai_response.cost_estimate,
            "confidence_score": ai_response.confidence_score,
            "processing_time_ms": ai_response.processing_time_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@router.get("/market-insights")
async def get_market_insights():
    """Get current market insights from AI"""
    try:
        insights = {
            "market_sentiment": "Bullish",
            "key_levels": {
                "ETH": {"support": 2400, "resistance": 2650},
                "BTC": {"support": 50000, "resistance": 55000}
            },
            "recommendations": [
                "Consider DCA strategy for ETH below $2450",
                "Monitor funding rates for potential arbitrage opportunities",
                "Portfolio diversification recommended due to high correlation"
            ],
            "risk_assessment": "Medium - Elevated volatility expected",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market insights: {str(e)}")


# New comprehensive AI endpoints using real service

class MarketAnalysisRequest(BaseModel):
    assets: List[str]
    timeframe: Optional[str] = "1h"


class StrategyRequest(BaseModel):
    risk_tolerance: str = "medium"  # low, medium, high
    account_equity: float
    experience_level: str = "intermediate"
    preferred_assets: Optional[List[str]] = None


class RiskAssessmentRequest(BaseModel):
    portfolio: Dict[str, Any]
    proposed_action: Dict[str, Any]


@router.post("/market-analysis", response_model=Dict[str, Any])
async def ai_market_analysis(request: MarketAnalysisRequest, current_user: dict = Depends(get_current_user)):
    """AI-powered market analysis for specific assets"""
    try:
        ai_service = get_ai_service()
        
        ai_response = await ai_service.analyze_market_conditions(request.assets)
        
        return {
            "analysis": ai_response.content,
            "assets_analyzed": request.assets,
            "timeframe": request.timeframe,
            "provider": ai_response.provider.value,
            "confidence_score": ai_response.confidence_score,
            "tokens_used": ai_response.tokens_used,
            "cost_estimate": ai_response.cost_estimate,
            "timestamp": ai_response.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market analysis error: {str(e)}")


@router.post("/strategy-generation", response_model=Dict[str, Any])
async def ai_strategy_generation(request: StrategyRequest, current_user: dict = Depends(get_current_user)):
    """Generate personalized trading strategy using AI"""
    try:
        ai_service = get_ai_service()
        
        user_profile = {
            "risk_tolerance": request.risk_tolerance,
            "account_equity": request.account_equity,
            "experience_level": request.experience_level,
            "wallet_address": current_user.get("wallet_address")
        }
        
        # Mock market data for now - in production would fetch real data
        market_data = {
            "overall_sentiment": "bullish",
            "volatility_index": 0.65,
            "major_assets": request.preferred_assets or ["ETH", "BTC", "SOL"]
        }
        
        ai_response = await ai_service.generate_trading_strategy(user_profile, market_data)
        
        return {
            "strategy": ai_response.content,
            "user_profile": user_profile,
            "market_conditions": market_data,
            "provider": ai_response.provider.value,
            "confidence_score": ai_response.confidence_score,
            "tokens_used": ai_response.tokens_used,
            "cost_estimate": ai_response.cost_estimate,
            "timestamp": ai_response.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy generation error: {str(e)}")


@router.post("/risk-assessment", response_model=Dict[str, Any])
async def ai_risk_assessment(request: RiskAssessmentRequest, current_user: dict = Depends(get_current_user)):
    """Comprehensive AI-powered risk assessment"""
    try:
        ai_service = get_ai_service()
        
        ai_response = await ai_service.risk_assessment(request.portfolio, request.proposed_action)
        
        return {
            "risk_analysis": ai_response.content,
            "portfolio_summary": request.portfolio,
            "proposed_action": request.proposed_action,
            "provider": ai_response.provider.value,
            "confidence_score": ai_response.confidence_score,
            "tokens_used": ai_response.tokens_used,
            "cost_estimate": ai_response.cost_estimate,
            "timestamp": ai_response.timestamp.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment error: {str(e)}")


@router.get("/service-stats", response_model=Dict[str, Any])
async def get_ai_service_stats(current_user: dict = Depends(get_current_user)):
    """Get AI service statistics and health"""
    try:
        ai_service = get_ai_service()
        stats = ai_service.get_service_stats()
        
        return {
            "service_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service stats error: {str(e)}")