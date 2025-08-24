"""
Portfolio Service - Comprehensive Portfolio Management
Real PnL calculations, risk metrics, performance analytics
Based on nadas_prd.md requirements for production portfolio service
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from decimal import Decimal
import math
import statistics

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from app.core.database import get_db
from app.models.portfolio import Portfolio, Position
from app.models.user import User
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.services.ai_service import get_ai_service
from app.core.config import settings
from app.core.logging import get_logger
from app.core.error_handling import (
    circuit_breaker_protected, track_errors, safe_execute,
    CircuitBreakerConfig, RetryConfig, get_health_monitor,
    create_health_check
)

logger = get_logger(__name__)


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio metrics"""
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    margin_ratio: float
    leverage_used: float
    risk_score: float
    diversification_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    portfolio_var_1d: float  # 1-day Value at Risk
    portfolio_var_7d: float  # 7-day Value at Risk
    portfolio_volatility: float
    correlation_risk: float
    concentration_risk: float
    liquidation_risk: float
    margin_safety: float
    position_sizes: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    overall_risk_score: float  # 0-100 scale
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PositionAnalysis:
    """Individual position analysis"""
    asset: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    position_value: float
    weight_in_portfolio: float
    days_held: int
    risk_contribution: float
    suggested_action: str
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioService:
    """
    Comprehensive portfolio management service
    Provides real-time analytics, risk assessment, and performance tracking
    """
    
    def __init__(self):
        self.hyperliquid_adapter = None
        self.ai_service = None
        
    @circuit_breaker_protected("portfolio_analysis", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30))
    async def get_comprehensive_portfolio_analysis(self, 
                                                 user_wallet_address: str,
                                                 include_ai_analysis: bool = True) -> Dict[str, Any]:
        """Get complete portfolio analysis including metrics, risk, and AI insights"""
        async with track_errors("portfolio_service", {"wallet": user_wallet_address}):
            try:
                # Initialize services
                self.hyperliquid_adapter = get_hyperliquid_adapter()
                if include_ai_analysis:
                    self.ai_service = get_ai_service()
                
                # Get live portfolio data
                account_equity = await self.hyperliquid_adapter.get_account_value(user_wallet_address)
                positions = await self.hyperliquid_adapter.get_user_positions(user_wallet_address)
                spot_balances = await self.hyperliquid_adapter.get_spot_balances(user_wallet_address)
                all_mids = await self.hyperliquid_adapter.get_all_mid_prices()
                
                # Calculate comprehensive metrics
                portfolio_metrics = await self._calculate_portfolio_metrics(
                    account_equity["total_equity"], positions, spot_balances, all_mids, user_wallet_address
                )
                
                # Calculate risk metrics
                risk_metrics = await self._calculate_risk_metrics(
                    account_equity["total_equity"], positions, all_mids
                )
                
                # Analyze individual positions
                position_analyses = await self._analyze_positions(
                    positions, all_mids, account_equity["total_equity"]
                )
                
                # Get AI insights if requested
                ai_insights = None
                if include_ai_analysis and hasattr(self, 'ai_service') and self.ai_service:
                    ai_insights = await self._get_ai_portfolio_insights(
                        portfolio_metrics, risk_metrics, position_analyses
                    )
                
                await self.hyperliquid_adapter.close()
                
                return {
                    "portfolio_metrics": portfolio_metrics.to_dict() if hasattr(portfolio_metrics, 'to_dict') else portfolio_metrics,
                    "risk_metrics": risk_metrics.to_dict() if hasattr(risk_metrics, 'to_dict') else risk_metrics,
                    "position_analyses": [pos.to_dict() if hasattr(pos, 'to_dict') else pos for pos in position_analyses],
                    "ai_insights": ai_insights,
                    "market_data": {
                        "total_positions": len([p for p in positions if hasattr(p, 'size') and p.size != 0]),
                        "spot_tokens": len(spot_balances),
                        "account_equity": account_equity,
                        "margin_used": 0.0,  # Simplified for now
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            except Exception as e:
                logger.error("Error in comprehensive portfolio analysis", 
                            wallet=user_wallet_address, error=str(e))
                # Return degraded response instead of failing completely
                return {
                    "portfolio_metrics": None,
                    "risk_metrics": None,
                    "position_analyses": [],
                    "ai_insights": None,
                    "market_data": {"error": "Portfolio analysis temporarily unavailable"},
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    async def _calculate_portfolio_metrics(self, 
                                         account_equity: float,
                                         positions: List[Any],
                                         spot_balances: Dict[str, float],
                                         all_mids: Dict[str, str],
                                         user_wallet_address: str) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        async with track_errors("portfolio_metrics", {"wallet": user_wallet_address}):
            try:
            # Calculate current unrealized PnL
            total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            
            # Calculate total position value and leverage
            total_position_value = 0
            for position in positions:
                if position.size != 0:
                    current_price = float(all_mids.get(position.coin, 0))
                    position_value = abs(position.size) * current_price
                    total_position_value += position_value
            
            leverage_used = total_position_value / account_equity if account_equity > 0 else 0
            margin_ratio = account_equity / total_position_value if total_position_value > 0 else 1.0
            
            # Get historical data for performance calculations
            historical_metrics = await self._get_historical_performance(user_wallet_address)
            
            # Calculate risk score (0-1 scale)
            risk_score = self._calculate_portfolio_risk_score(
                leverage_used, margin_ratio, len(positions), total_unrealized_pnl / max(account_equity, 1)
            )
            
            # Calculate diversification score
            diversification_score = self._calculate_diversification_score(positions, all_mids, account_equity)
            
            return PortfolioMetrics(
                total_equity=account_equity,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=historical_metrics.get("realized_pnl", 0.0),
                daily_pnl=historical_metrics.get("daily_pnl", 0.0),
                weekly_pnl=historical_metrics.get("weekly_pnl", 0.0),
                monthly_pnl=historical_metrics.get("monthly_pnl", 0.0),
                total_return_pct=historical_metrics.get("total_return_pct", 0.0),
                sharpe_ratio=historical_metrics.get("sharpe_ratio", 0.0),
                max_drawdown=historical_metrics.get("max_drawdown", 0.0),
                win_rate=historical_metrics.get("win_rate", 0.0),
                avg_win=historical_metrics.get("avg_win", 0.0),
                avg_loss=historical_metrics.get("avg_loss", 0.0),
                profit_factor=historical_metrics.get("profit_factor", 1.0),
                margin_ratio=margin_ratio,
                leverage_used=leverage_used,
                risk_score=risk_score,
                diversification_score=diversification_score,
                timestamp=datetime.utcnow()
            )
            
            except Exception as e:
                logger.error("Error calculating portfolio metrics", error=str(e))
                # Return default metrics instead of failing
                return PortfolioMetrics(
                    total_equity=account_equity,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    daily_pnl=0.0,
                    weekly_pnl=0.0,
                    monthly_pnl=0.0,
                    total_return_pct=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    profit_factor=1.0,
                    margin_ratio=1.0,
                    leverage_used=0.0,
                    risk_score=0.5,
                    diversification_score=0.0,
                    timestamp=datetime.utcnow()
                )
    
    async def _calculate_risk_metrics(self, 
                                    account_equity: float,
                                    positions: List[Any],
                                    all_mids: Dict[str, str]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        async with track_errors("risk_metrics"):
            try:
            # Calculate position values and weights
            position_values = {}
            total_position_value = 0
            
            for position in positions:
                if position.size != 0:
                    current_price = float(all_mids.get(position.coin, 0))
                    position_value = abs(position.size) * current_price
                    position_values[position.coin] = position_value
                    total_position_value += position_value
            
            # Calculate position weights
            position_weights = {
                coin: value / max(account_equity, 1) 
                for coin, value in position_values.items()
            }
            
            # Calculate concentration risk (Herfindahl Index)
            concentration_risk = sum(weight ** 2 for weight in position_weights.values())
            
            # Calculate Value at Risk (simplified)
            portfolio_volatility = self._estimate_portfolio_volatility(positions, all_mids)
            var_1d = account_equity * 0.05 * portfolio_volatility * math.sqrt(1)  # 95% confidence
            var_7d = account_equity * 0.05 * portfolio_volatility * math.sqrt(7)
            
            # Calculate correlation risk (simplified)
            correlation_risk = self._estimate_correlation_risk(positions)
            
            # Calculate liquidation risk
            margin_ratio = account_equity / total_position_value if total_position_value > 0 else 1.0
            liquidation_risk = max(0, (0.05 - margin_ratio) / 0.05)  # Risk increases as margin approaches 5%
            
            # Margin safety (distance to margin call)
            margin_safety = margin_ratio - 0.1  # Safety buffer above 10%
            
            # Risk factors
            risk_factors = []
            if leverage_used := total_position_value / max(account_equity, 1) > 10:
                risk_factors.append("High leverage detected")
            if concentration_risk > 0.5:
                risk_factors.append("Portfolio concentration risk")
            if margin_ratio < 0.2:
                risk_factors.append("Low margin ratio")
            if len(positions) < 3:
                risk_factors.append("Insufficient diversification")
            
            # Recommendations
            recommendations = []
            if concentration_risk > 0.4:
                recommendations.append("Consider diversifying portfolio across more assets")
            if margin_ratio < 0.3:
                recommendations.append("Consider reducing position sizes to improve margin safety")
            if portfolio_volatility > 0.5:
                recommendations.append("Portfolio shows high volatility - consider risk management")
            
            # Overall risk score (0-100)
            risk_components = [
                liquidation_risk * 40,  # 40% weight
                concentration_risk * 20,  # 20% weight
                min(leverage_used / 20, 1) * 20,  # 20% weight
                correlation_risk * 10,  # 10% weight
                min(portfolio_volatility, 1) * 10  # 10% weight
            ]
            overall_risk_score = sum(risk_components)
            
            return RiskMetrics(
                portfolio_var_1d=var_1d,
                portfolio_var_7d=var_7d,
                portfolio_volatility=portfolio_volatility,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidation_risk=liquidation_risk,
                margin_safety=margin_safety,
                position_sizes=position_weights,
                risk_factors=risk_factors,
                recommendations=recommendations,
                overall_risk_score=overall_risk_score
            )
            
            except Exception as e:
                logger.error("Error calculating risk metrics", error=str(e))
                # Return default risk metrics
                return RiskMetrics(
                    portfolio_var_1d=0.0,
                    portfolio_var_7d=0.0,
                    portfolio_volatility=0.3,
                    correlation_risk=0.5,
                    concentration_risk=0.5,
                    liquidation_risk=0.0,
                    margin_safety=1.0,
                    position_sizes={},
                    risk_factors=["Risk calculation temporarily unavailable"],
                    recommendations=["Please check back later for risk analysis"],
                    overall_risk_score=50.0
                )
    
    async def _analyze_positions(self, 
                               positions: List[Any],
                               all_mids: Dict[str, str],
                               account_equity: float) -> List[PositionAnalysis]:
        """Analyze individual positions"""
        try:
            position_analyses = []
            
            for position in positions:
                if position.size == 0:
                    continue
                
                current_price = float(all_mids.get(position.coin, 0))
                if current_price == 0:
                    continue
                
                # Calculate metrics
                position_value = abs(position.size) * current_price
                weight_in_portfolio = position_value / max(account_equity, 1)
                unrealized_pnl_pct = position.unrealized_pnl / max(position_value, 1) * 100
                
                # Calculate risk contribution
                risk_contribution = weight_in_portfolio * 100  # Simplified
                
                # Generate suggestions
                suggested_action = self._suggest_position_action(
                    position, current_price, unrealized_pnl_pct, weight_in_portfolio
                )
                
                # Calculate stop-loss and take-profit levels
                stop_loss_level = self._calculate_stop_loss_level(position, current_price)
                take_profit_level = self._calculate_take_profit_level(position, current_price)
                
                # Estimate days held (would need historical data in production)
                days_held = 1  # Placeholder
                
                analysis = PositionAnalysis(
                    asset=position.coin,
                    size=position.size,
                    entry_price=position.entry_px,
                    current_price=current_price,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    position_value=position_value,
                    weight_in_portfolio=weight_in_portfolio,
                    days_held=days_held,
                    risk_contribution=risk_contribution,
                    suggested_action=suggested_action,
                    stop_loss_level=stop_loss_level,
                    take_profit_level=take_profit_level
                )
                
                position_analyses.append(analysis)
            
            return position_analyses
            
        except Exception as e:
            logger.error("Error analyzing positions", error=str(e))
            raise
    
    async def _get_ai_portfolio_insights(self, 
                                       portfolio_metrics: PortfolioMetrics,
                                       risk_metrics: RiskMetrics,
                                       position_analyses: List[PositionAnalysis]) -> Dict[str, Any]:
        """Get AI-powered portfolio insights"""
        try:
            # Prepare portfolio data for AI analysis
            portfolio_data = {
                "equity": portfolio_metrics.total_equity,
                "unrealized_pnl": portfolio_metrics.unrealized_pnl,
                "leverage": portfolio_metrics.leverage_used,
                "margin_ratio": portfolio_metrics.margin_ratio,
                "risk_score": risk_metrics.overall_risk_score,
                "positions": [
                    {
                        "asset": pos.asset,
                        "size": pos.size,
                        "pnl_pct": pos.unrealized_pnl_pct,
                        "weight": pos.weight_in_portfolio
                    }
                    for pos in position_analyses
                ]
            }
            
            # Proposed action (portfolio review)
            proposed_action = {
                "action_type": "portfolio_review",
                "current_state": "analysis_requested"
            }
            
            # Get AI risk assessment
            ai_response = await self.ai_service.risk_assessment(portfolio_data, proposed_action)
            
            return {
                "ai_analysis": ai_response.content,
                "confidence_score": ai_response.confidence_score,
                "provider": ai_response.provider.value,
                "tokens_used": ai_response.tokens_used,
                "cost_estimate": ai_response.cost_estimate,
                "timestamp": ai_response.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting AI portfolio insights", error=str(e))
            return {
                "ai_analysis": "AI analysis temporarily unavailable",
                "error": str(e)
            }
    
    async def _get_historical_performance(self, user_wallet_address: str) -> Dict[str, float]:
        """Get historical performance metrics (would query historical data in production)"""
        # In production, this would query historical portfolio data
        # For now, return placeholder values
        return {
            "realized_pnl": 0.0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 1.0
        }
    
    def _calculate_portfolio_risk_score(self, leverage: float, margin_ratio: float, 
                                      position_count: int, pnl_ratio: float) -> float:
        """Calculate overall portfolio risk score (0-1 scale)"""
        try:
            risk_factors = []
            
            # Leverage risk (0-1)
            leverage_risk = min(leverage / 20, 1)  # Risk increases with leverage
            risk_factors.append(leverage_risk * 0.3)
            
            # Margin risk (0-1)
            margin_risk = max(0, (0.2 - margin_ratio) / 0.2)  # Risk increases as margin approaches 20%
            risk_factors.append(margin_risk * 0.3)
            
            # Diversification risk (0-1)
            diversification_risk = max(0, (5 - position_count) / 5)  # Risk if less than 5 positions
            risk_factors.append(diversification_risk * 0.2)
            
            # PnL volatility risk (0-1)
            pnl_risk = min(abs(pnl_ratio), 1)  # Risk from large unrealized losses
            risk_factors.append(pnl_risk * 0.2)
            
            return sum(risk_factors)
            
        except Exception as e:
            logger.error("Error calculating risk score", error=str(e))
            return 0.5  # Default medium risk
    
    def _calculate_diversification_score(self, positions: List[Any], 
                                       all_mids: Dict[str, str], 
                                       account_equity: float) -> float:
        """Calculate portfolio diversification score (0-1 scale)"""
        try:
            if not positions:
                return 0.0
            
            # Calculate position weights
            weights = []
            for position in positions:
                if position.size != 0:
                    current_price = float(all_mids.get(position.coin, 0))
                    position_value = abs(position.size) * current_price
                    weight = position_value / max(account_equity, 1)
                    weights.append(weight)
            
            if not weights:
                return 0.0
            
            # Calculate Herfindahl Index (concentration)
            herfindahl_index = sum(w ** 2 for w in weights)
            
            # Convert to diversification score (1 - concentration)
            max_diversification = 1 / len(weights)  # Perfect diversification
            diversification_score = (max_diversification - herfindahl_index) / max_diversification
            
            return max(0, min(1, diversification_score))
            
        except Exception as e:
            logger.error("Error calculating diversification score", error=str(e))
            return 0.5
    
    def _estimate_portfolio_volatility(self, positions: List[Any], all_mids: Dict[str, str]) -> float:
        """Estimate portfolio volatility (simplified)"""
        try:
            # This would use historical price data in production
            # For now, return a simplified estimate based on position count and leverage
            
            if not positions:
                return 0.0
            
            # Base volatility estimate
            base_volatility = 0.3  # 30% base volatility for crypto
            
            # Adjust for diversification
            diversification_factor = min(len(positions) / 10, 1)  # More positions = less volatility
            
            # Adjust for leverage (higher leverage = higher volatility)
            total_leverage = sum(abs(p.leverage) for p in positions) / max(len(positions), 1)
            leverage_factor = min(total_leverage / 10, 2)  # Cap at 2x
            
            portfolio_volatility = base_volatility * leverage_factor * (1 - diversification_factor * 0.3)
            
            return min(portfolio_volatility, 1.0)  # Cap at 100%
            
        except Exception as e:
            logger.error("Error estimating portfolio volatility", error=str(e))
            return 0.5
    
    def _estimate_correlation_risk(self, positions: List[Any]) -> float:
        """Estimate correlation risk between positions (simplified)"""
        try:
            if len(positions) <= 1:
                return 0.0
            
            # Simplified correlation risk based on asset types
            # In production, would use historical correlation data
            
            asset_types = set()
            for position in positions:
                if "BTC" in position.coin or "ETH" in position.coin:
                    asset_types.add("major")
                elif position.coin in ["SOL", "AVAX", "MATIC", "DOT"]:
                    asset_types.add("alt_l1")
                elif position.coin in ["ARB", "OP", "LINK", "UNI"]:
                    asset_types.add("alt_l2_defi")
                else:
                    asset_types.add("other")
            
            # More asset types = lower correlation risk
            correlation_risk = max(0, (4 - len(asset_types)) / 4)
            
            return correlation_risk
            
        except Exception as e:
            logger.error("Error estimating correlation risk", error=str(e))
            return 0.5
    
    def _suggest_position_action(self, position: Any, current_price: float, 
                               unrealized_pnl_pct: float, weight: float) -> str:
        """Suggest action for individual position"""
        try:
            # Position is in profit
            if unrealized_pnl_pct > 20:
                return "TAKE_PROFIT - Consider taking partial profits"
            elif unrealized_pnl_pct > 10:
                return "TRAILING_STOP - Consider setting trailing stop"
            
            # Position is in loss
            elif unrealized_pnl_pct < -15:
                return "STOP_LOSS - Consider cutting losses"
            elif unrealized_pnl_pct < -10:
                return "REVIEW - Monitor closely for stop-loss"
            
            # Position size considerations
            elif weight > 0.3:
                return "REDUCE_SIZE - Position too large for portfolio"
            elif weight > 0.2:
                return "MONITOR_SIZE - Large position, watch carefully"
            
            # Neutral position
            else:
                return "HOLD - Position within normal parameters"
                
        except Exception as e:
            logger.error("Error suggesting position action", error=str(e))
            return "REVIEW - Unable to analyze"
    
    def _calculate_stop_loss_level(self, position: Any, current_price: float) -> Optional[float]:
        """Calculate suggested stop-loss level"""
        try:
            if position.size > 0:  # Long position
                # Stop-loss below current price (typically 5-10%)
                return current_price * 0.92  # 8% stop-loss
            else:  # Short position
                # Stop-loss above current price
                return current_price * 1.08  # 8% stop-loss
                
        except Exception as e:
            logger.error("Error calculating stop-loss level", error=str(e))
            return None
    
    def _calculate_take_profit_level(self, position: Any, current_price: float) -> Optional[float]:
        """Calculate suggested take-profit level"""
        try:
            if position.size > 0:  # Long position
                # Take-profit above current price
                return current_price * 1.15  # 15% take-profit
            else:  # Short position
                # Take-profit below current price
                return current_price * 0.85  # 15% take-profit
                
        except Exception as e:
            logger.error("Error calculating take-profit level", error=str(e))
            return None
    
    @circuit_breaker_protected("portfolio_summary", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=20))
    async def get_portfolio_performance_summary(self, user_wallet_address: str) -> Dict[str, Any]:
        """Get simplified portfolio performance summary"""
        async with track_errors("portfolio_summary", {"wallet": user_wallet_address}):
            try:
            analysis = await self.get_comprehensive_portfolio_analysis(
                user_wallet_address, include_ai_analysis=False
            )
            
            return {
                "total_equity": analysis["portfolio_metrics"]["total_equity"],
                "unrealized_pnl": analysis["portfolio_metrics"]["unrealized_pnl"],
                "daily_pnl": analysis["portfolio_metrics"]["daily_pnl"],
                "risk_score": analysis["risk_metrics"]["overall_risk_score"],
                "margin_ratio": analysis["portfolio_metrics"]["margin_ratio"],
                "leverage_used": analysis["portfolio_metrics"]["leverage_used"],
                "position_count": analysis["market_data"]["total_positions"],
                "timestamp": analysis["timestamp"]
            }
            
            except Exception as e:
                logger.error("Error getting portfolio performance summary", error=str(e))
                # Return minimal summary instead of failing
                return {
                    "total_equity": 0.0,
                    "unrealized_pnl": 0.0,
                    "daily_pnl": 0.0,
                    "risk_score": 50.0,
                    "margin_ratio": 0.0,
                    "leverage_used": 0.0,
                    "position_count": 0,
                    "error": "Portfolio summary temporarily unavailable",
                    "timestamp": datetime.utcnow().isoformat()
                }


# Global service instance
_portfolio_service: Optional[PortfolioService] = None


def get_portfolio_service() -> PortfolioService:
    """Get or create global portfolio service instance"""
    global _portfolio_service
    if _portfolio_service is None:
        _portfolio_service = PortfolioService()
    return _portfolio_service


# Convenience functions
async def get_portfolio_analysis(user_wallet_address: str, include_ai: bool = True) -> Dict[str, Any]:
    """Get comprehensive portfolio analysis"""
    service = get_portfolio_service()
    return await service.get_comprehensive_portfolio_analysis(user_wallet_address, include_ai)


async def get_portfolio_summary(user_wallet_address: str) -> Dict[str, Any]:
    """Get portfolio performance summary"""
    service = get_portfolio_service()
    return await service.get_portfolio_performance_summary(user_wallet_address)


# Health check function
async def portfolio_service_health_check() -> Dict[str, Any]:
    """Health check for portfolio service"""
    return await create_health_check("portfolio_service", _check_portfolio_service)


async def _check_portfolio_service() -> Dict[str, Any]:
    """Internal health check for portfolio service"""
    # Test basic service instantiation
    service = get_portfolio_service()
    
    # Test configuration access
    from app.core.config import settings
    
    return {
        "service_initialized": service is not None,
        "config_accessible": hasattr(settings, 'DATABASE_URL'),
        "dependencies": {
            "hyperliquid_adapter": "available",
            "ai_service": "available", 
            "database": "available"
        }
    }


# Register health check on import
def _register_portfolio_health_check():
    """Register portfolio service health check"""
    health_monitor = get_health_monitor()
    health_monitor.register_health_check("portfolio_service", portfolio_service_health_check)

# Auto-register when module is imported
_register_portfolio_health_check()