"""
Demo Transaction Flows for Hackathon Demonstration
Showcases Nadas.fi capabilities with Hyperliquid mainnet integration
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

from app.api.v1.auth import get_current_user
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.adapters.lifi import get_lifi_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class DemoScenarioRequest(BaseModel):
    """Demo scenario request"""
    scenario_type: str = Field(..., description="Type of demo: 'trading', 'cross_chain', 'automation'")
    parameters: Dict = Field(default={}, description="Scenario parameters")


@router.post("/scenarios/trading")
async def demo_trading_scenario(
    current_user: dict = Depends(get_current_user)
):
    """Demo trading scenario with Hyperliquid mainnet"""
    try:
        user_address = current_user["wallet_address"]
        
        # Demo trading workflow
        scenario = {
            "scenario_id": f"trading_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "title": "Hyperliquid Mainnet Trading Demo",
            "network": "hyperliquid_mainnet",
            "user_address": user_address,
            "timestamp": datetime.utcnow().isoformat(),
            "steps": [
                {
                    "step": 1,
                    "action": "Connect to Hyperliquid mainnet",
                    "status": "completed",
                    "api_endpoint": "https://api.hyperliquid.xyz",
                    "description": "Establish connection to Hyperliquid mainnet API"
                },
                {
                    "step": 2,
                    "action": "Fetch portfolio data",
                    "status": "completed", 
                    "api_call": "/api/v1/portfolio/overview",
                    "description": "Get real-time portfolio information"
                },
                {
                    "step": 3,
                    "action": "Get market data",
                    "status": "ready",
                    "symbol": "ETH-PERP",
                    "api_call": "/api/v1/trading/demo/market-data/ETH",
                    "description": "Fetch live ETH perpetual market data"
                },
                {
                    "step": 4,
                    "action": "Place demo order",
                    "status": "ready",
                    "order_details": {
                        "symbol": "ETH",
                        "side": "buy",
                        "size": 100.0,
                        "order_type": "limit",
                        "price": 3500.0
                    },
                    "api_call": "/api/v1/trading/demo/place-order",
                    "description": "Place a small limit order for demonstration"
                },
                {
                    "step": 5,
                    "action": "Monitor position",
                    "status": "pending",
                    "description": "Track position and P&L in real-time"
                }
            ],
            "expected_duration_minutes": 5,
            "demo_mode": True,
            "hackathon_category": "Frontier Track ($50k)"
        }
        
        logger.info("Trading demo scenario created", 
                   user=user_address,
                   scenario_id=scenario["scenario_id"])
        
        return {
            "success": True,
            "data": scenario,
            "message": "Trading demo scenario ready for execution"
        }
        
    except Exception as e:
        logger.error("Trading demo scenario failed", 
                    user=current_user.get("wallet_address"),
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Demo scenario failed: {str(e)}")


@router.post("/scenarios/cross-chain")
async def demo_cross_chain_scenario(
    current_user: dict = Depends(get_current_user)
):
    """Demo cross-chain scenario with LI.FI integration"""
    try:
        user_address = current_user["wallet_address"]
        
        # Demo cross-chain workflow
        scenario = {
            "scenario_id": f"crosschain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "title": "LI.FI Cross-Chain Bridge Demo",
            "user_address": user_address,
            "timestamp": datetime.utcnow().isoformat(),
            "steps": [
                {
                    "step": 1,
                    "action": "Connect to LI.FI API",
                    "status": "completed",
                    "api_endpoint": "https://li.quest/v1",
                    "description": "Establish connection to LI.FI cross-chain API"
                },
                {
                    "step": 2,
                    "action": "Get supported chains",
                    "status": "completed",
                    "api_call": "/api/v1/lifi/chains",
                    "supported_chains": ["Ethereum", "Arbitrum", "Polygon", "Base", "HyperEVM"],
                    "description": "Fetch list of supported blockchain networks"
                },
                {
                    "step": 3,
                    "action": "Get bridge quote",
                    "status": "ready",
                    "bridge_details": {
                        "from_chain": "Ethereum",
                        "to_chain": "HyperEVM",
                        "token": "USDC",
                        "amount": 1000
                    },
                    "api_call": "/api/v1/lifi/quote",
                    "description": "Get optimal cross-chain routing quote"
                },
                {
                    "step": 4,
                    "action": "Execute bridge transaction",
                    "status": "ready",
                    "description": "Execute cross-chain bridge with wallet signature"
                },
                {
                    "step": 5,
                    "action": "Monitor bridge status",
                    "status": "pending",
                    "description": "Track cross-chain transaction completion"
                }
            ],
            "expected_duration_minutes": 10,
            "demo_mode": True,
            "hackathon_category": "LI.FI Integration ($7k)"
        }
        
        logger.info("Cross-chain demo scenario created", 
                   user=user_address,
                   scenario_id=scenario["scenario_id"])
        
        return {
            "success": True,
            "data": scenario,
            "message": "Cross-chain demo scenario ready for execution"
        }
        
    except Exception as e:
        logger.error("Cross-chain demo scenario failed", 
                    user=current_user.get("wallet_address"),
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Demo scenario failed: {str(e)}")


@router.post("/scenarios/automation")
async def demo_automation_scenario(
    current_user: dict = Depends(get_current_user)
):
    """Demo AI-powered automation scenario"""
    try:
        user_address = current_user["wallet_address"]
        
        # Demo automation workflow
        scenario = {
            "scenario_id": f"automation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "title": "AI-Powered DeFi Automation Demo",
            "user_address": user_address,
            "timestamp": datetime.utcnow().isoformat(),
            "ai_assistant": "Claude 3.5 Sonnet",
            "steps": [
                {
                    "step": 1,
                    "action": "Analyze portfolio",
                    "status": "completed",
                    "ai_analysis": "Portfolio analysis using Claude 3.5 Sonnet",
                    "description": "AI analyzes current positions and market conditions"
                },
                {
                    "step": 2,
                    "action": "Generate trading signals",
                    "status": "ready",
                    "signals": [
                        {"asset": "ETH", "signal": "BULLISH", "confidence": 0.75},
                        {"asset": "BTC", "signal": "NEUTRAL", "confidence": 0.60}
                    ],
                    "description": "AI generates trading recommendations"
                },
                {
                    "step": 3,
                    "action": "Create automation rules",
                    "status": "ready",
                    "rules": [
                        "Take profit at +10% gains",
                        "Stop loss at -5% losses",
                        "DCA on 15% dips"
                    ],
                    "description": "Set up automated trading rules"
                },
                {
                    "step": 4,
                    "action": "Execute automation",
                    "status": "ready",
                    "description": "Run AI-powered trading automation"
                },
                {
                    "step": 5,
                    "action": "Monitor performance",
                    "status": "pending", 
                    "description": "Track automation performance and adjustments"
                }
            ],
            "expected_duration_minutes": 15,
            "demo_mode": True,
            "hackathon_category": "AI Integration"
        }
        
        logger.info("Automation demo scenario created", 
                   user=user_address,
                   scenario_id=scenario["scenario_id"])
        
        return {
            "success": True,
            "data": scenario,
            "message": "Automation demo scenario ready for execution"
        }
        
    except Exception as e:
        logger.error("Automation demo scenario failed", 
                    user=current_user.get("wallet_address"),
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Demo scenario failed: {str(e)}")


@router.get("/health")
async def demo_health_check():
    """Demo API health check"""
    return {
        "status": "healthy",
        "service": "demo_api",
        "mainnet_ready": True,
        "integrations": {
            "hyperliquid_mainnet": True,
            "lifi_crosschain": True,
            "ai_assistant": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }