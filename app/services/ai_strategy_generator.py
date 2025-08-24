"""
AI Strategy Generator Service
Uses Claude 3.5 Sonnet to generate cross-chain trading strategies from natural language
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import structlog

from app.services.ai_service import get_ai_service
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.services.cross_chain_orchestrator import CrossChainStrategy, StrategyStatus

logger = structlog.get_logger(__name__)

class StrategyIntent:
    """Represents parsed user intent for cross-chain strategy"""
    
    def __init__(self):
        self.action: str = ""  # transfer, trade, dca, arbitrage
        self.source_chain: str = ""
        self.target_chain: str = "hyperliquid"  # Default
        self.source_token: str = ""
        self.target_token: str = ""
        self.amount: float = 0.0
        self.automation_rules: List[Dict] = []
        self.risk_tolerance: str = "medium"  # low, medium, high
        self.timeframe: str = "immediate"  # immediate, short, medium, long
        self.confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source_chain": self.source_chain,
            "target_chain": self.target_chain,
            "source_token": self.source_token,
            "target_token": self.target_token,
            "amount": self.amount,
            "automation_rules": self.automation_rules,
            "risk_tolerance": self.risk_tolerance,
            "timeframe": self.timeframe,
            "confidence_score": self.confidence_score
        }

class AIStrategyGenerator:
    """AI-powered cross-chain strategy generation"""
    
    def __init__(self):
        self.ai_service = None
        self.hyperliquid_adapter = None
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize AI service and market data adapters"""
        try:
            self.ai_service = get_ai_service()
            self.hyperliquid_adapter = get_hyperliquid_adapter()
            logger.info("AI Strategy Generator initialized")
        except Exception as e:
            logger.error("Failed to initialize AI Strategy Generator", error=str(e))

    async def generate_strategy_from_text(
        self, 
        user_input: str, 
        user_address: str,
        portfolio_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate cross-chain strategy from natural language input"""
        try:
            logger.info("Generating strategy from user input", 
                       user=user_address, 
                       input_length=len(user_input))
            
            # Parse user intent with AI
            intent = await self._parse_user_intent(user_input, portfolio_context)
            
            # Get current market data for context
            market_context = await self._get_market_context(intent)
            
            # Generate detailed strategy with AI
            strategy_plan = await self._generate_strategy_plan(intent, market_context, portfolio_context)
            
            # Validate and optimize strategy
            validated_strategy = await self._validate_strategy(strategy_plan, intent)
            
            # Add risk assessment
            risk_assessment = await self._assess_strategy_risks(validated_strategy, market_context)
            
            return {
                "success": True,
                "strategy": validated_strategy,
                "risk_assessment": risk_assessment,
                "intent": intent.to_dict(),
                "market_context": market_context,
                "ai_confidence": intent.confidence_score,
                "estimated_execution_time": self._estimate_execution_time(validated_strategy),
                "estimated_costs": self._estimate_costs(validated_strategy)
            }
            
        except Exception as e:
            logger.error("Strategy generation failed", 
                        user=user_address, 
                        error=str(e))
            return {
                "success": False,
                "error": str(e),
                "fallback_suggestions": self._get_fallback_suggestions(user_input)
            }

    async def _parse_user_intent(
        self, 
        user_input: str, 
        portfolio_context: Optional[Dict] = None
    ) -> StrategyIntent:
        """Parse user intent using AI"""
        
        system_prompt = """You are a DeFi strategy analyst. Parse the user's request and extract:
        
        1. ACTION: transfer, trade, dca (dollar cost average), arbitrage, rebalance
        2. SOURCE_CHAIN: ethereum, polygon, arbitrum, optimism, bsc, avalanche
        3. TARGET_CHAIN: hyperliquid (default)
        4. SOURCE_TOKEN: ETH, USDC, BTC, etc.
        5. TARGET_TOKEN: ETH, USDC, BTC, etc.
        6. AMOUNT: numerical value
        7. AUTOMATION_RULES: any recurring or conditional actions
        8. RISK_TOLERANCE: low, medium, high
        9. TIMEFRAME: immediate, short, medium, long
        10. CONFIDENCE_SCORE: 0-100 how confident you are in the parsing
        
        Respond ONLY in valid JSON format with these exact keys."""
        
        portfolio_info = ""
        if portfolio_context:
            portfolio_info = f"\\nUser's current portfolio: {json.dumps(portfolio_context, indent=2)}"
        
        user_prompt = f"""Parse this trading request: "{user_input}"{portfolio_info}"""
        
        try:
            if self.ai_service:
                ai_response = await self.ai_service.analyze_automation_request(
                    user_request=user_prompt,
                    market_data={},
                    portfolio_data=portfolio_context or {}
                )
                
                # Parse AI response to extract intent
                intent = self._parse_ai_response_to_intent(ai_response.get("analysis", ""))
            else:
                # Fallback parsing without AI
                intent = self._fallback_parse_intent(user_input)
            
            return intent
            
        except Exception as e:
            logger.warning("AI intent parsing failed, using fallback", error=str(e))
            return self._fallback_parse_intent(user_input)

    def _parse_ai_response_to_intent(self, ai_response: str) -> StrategyIntent:
        """Parse AI response into StrategyIntent object"""
        intent = StrategyIntent()
        
        try:
            # Try to extract JSON from AI response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
            else:
                # Fallback parsing from text
                parsed_data = self._extract_data_from_text(ai_response)
            
            intent.action = parsed_data.get("action", "transfer").lower()
            intent.source_chain = parsed_data.get("source_chain", "ethereum").lower()
            intent.target_chain = parsed_data.get("target_chain", "hyperliquid").lower()
            intent.source_token = parsed_data.get("source_token", "ETH").upper()
            intent.target_token = parsed_data.get("target_token", "USDC").upper()
            intent.amount = float(parsed_data.get("amount", 1000))
            intent.automation_rules = parsed_data.get("automation_rules", [])
            intent.risk_tolerance = parsed_data.get("risk_tolerance", "medium").lower()
            intent.timeframe = parsed_data.get("timeframe", "immediate").lower()
            intent.confidence_score = float(parsed_data.get("confidence_score", 75))
            
        except Exception as e:
            logger.warning("Failed to parse AI response", error=str(e))
            # Use fallback values
            intent.confidence_score = 50
            
        return intent

    def _fallback_parse_intent(self, user_input: str) -> StrategyIntent:
        """Fallback intent parsing without AI"""
        intent = StrategyIntent()
        text = user_input.lower()
        
        # Action detection
        if any(word in text for word in ["transfer", "move", "bridge", "send"]):
            intent.action = "transfer"
        elif any(word in text for word in ["trade", "buy", "sell", "swap"]):
            intent.action = "trade"
        elif any(word in text for word in ["dca", "dollar cost", "regular"]):
            intent.action = "dca"
        else:
            intent.action = "transfer"  # Default
        
        # Chain detection
        chain_mapping = {
            "ethereum": ["ethereum", "eth", "mainnet"],
            "polygon": ["polygon", "matic"],
            "arbitrum": ["arbitrum", "arb"],
            "optimism": ["optimism", "op"],
            "bsc": ["bsc", "binance"],
            "avalanche": ["avalanche", "avax"]
        }
        
        for chain, keywords in chain_mapping.items():
            if any(keyword in text for keyword in keywords):
                intent.source_chain = chain
                break
        else:
            intent.source_chain = "ethereum"  # Default
        
        # Token detection
        tokens = ["ETH", "BTC", "USDC", "USDT", "SOL", "AVAX", "MATIC"]
        for token in tokens:
            if token.lower() in text:
                if not intent.source_token:
                    intent.source_token = token
                else:
                    intent.target_token = token
        
        if not intent.source_token:
            intent.source_token = "ETH"
        if not intent.target_token:
            intent.target_token = "USDC"
        
        # Amount detection
        amount_match = re.search(r'[\d,]+\.?\d*', text)
        if amount_match:
            amount_str = amount_match.group().replace(',', '')
            try:
                intent.amount = float(amount_str)
            except:
                intent.amount = 1000.0
        else:
            intent.amount = 1000.0
        
        intent.confidence_score = 60  # Lower confidence for fallback parsing
        return intent

    def _extract_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from AI text response"""
        data = {}
        
        # This would contain more sophisticated text parsing
        # For now, return defaults
        return {
            "action": "transfer",
            "source_chain": "ethereum",
            "target_chain": "hyperliquid",
            "source_token": "ETH",
            "target_token": "USDC",
            "amount": 1000,
            "confidence_score": 65
        }

    async def _get_market_context(self, intent: StrategyIntent) -> Dict[str, Any]:
        """Get current market data for strategy context"""
        market_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "prices": {},
            "volatility": {},
            "trends": {}
        }
        
        try:
            if self.hyperliquid_adapter:
                # Get price data for relevant tokens
                if intent.source_token:
                    price_data = await self.hyperliquid_adapter.get_market_data(intent.source_token)
                    if price_data:
                        market_context["prices"][intent.source_token] = {
                            "mid_price": price_data.mid_price,
                            "bid_price": price_data.bid_price,
                            "ask_price": price_data.ask_price,
                            "volume_24h": price_data.volume_24h,
                            "price_change_24h": price_data.price_change_24h
                        }
                        
                        # Simple volatility calculation
                        market_context["volatility"][intent.source_token] = abs(price_data.price_change_24h)
                
        except Exception as e:
            logger.warning("Failed to get market context", error=str(e))
        
        return market_context

    async def _generate_strategy_plan(
        self, 
        intent: StrategyIntent, 
        market_context: Dict[str, Any],
        portfolio_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate detailed strategy plan using AI"""
        
        strategy_prompt = f"""
        Generate a detailed cross-chain strategy plan:
        
        User Intent:
        - Action: {intent.action}
        - From: {intent.source_chain} ({intent.source_token})
        - To: {intent.target_chain} ({intent.target_token})
        - Amount: {intent.amount}
        - Risk Tolerance: {intent.risk_tolerance}
        
        Market Context:
        {json.dumps(market_context, indent=2)}
        
        Create a step-by-step execution plan with:
        1. Bridge selection rationale
        2. Timing recommendations
        3. Risk mitigation steps
        4. Automation opportunities
        5. Expected outcomes
        """
        
        try:
            if self.ai_service:
                ai_response = await self.ai_service.analyze_automation_request(
                    user_request=strategy_prompt,
                    market_data=market_context,
                    portfolio_data=portfolio_context or {}
                )
                
                strategy_plan = {
                    "steps": self._extract_strategy_steps(ai_response.get("analysis", "")),
                    "rationale": ai_response.get("analysis", ""),
                    "source_chain": intent.source_chain,
                    "target_chain": intent.target_chain,
                    "source_token": intent.source_token,
                    "target_token": intent.target_token,
                    "amount": intent.amount,
                    "automation_rules": intent.automation_rules,
                    "execution_order": ["quote", "bridge", "confirm", "automate"]
                }
            else:
                # Fallback strategy plan
                strategy_plan = self._generate_fallback_strategy(intent)
            
            return strategy_plan
            
        except Exception as e:
            logger.warning("AI strategy planning failed, using fallback", error=str(e))
            return self._generate_fallback_strategy(intent)

    def _extract_strategy_steps(self, ai_analysis: str) -> List[Dict[str, Any]]:
        """Extract strategy steps from AI analysis"""
        # This would parse AI response for structured steps
        # For now, return default steps
        return [
            {
                "step": 1,
                "action": "Get cross-chain quotes",
                "description": "Compare LI.FI and GlueX routes for optimal path",
                "estimated_time": "30 seconds"
            },
            {
                "step": 2,
                "action": "Execute bridge transaction",
                "description": "Transfer assets via selected bridge provider",
                "estimated_time": "5-10 minutes"
            },
            {
                "step": 3,
                "action": "Confirm arrival",
                "description": "Wait for assets to arrive on target chain",
                "estimated_time": "2-5 minutes"
            },
            {
                "step": 4,
                "action": "Setup automation",
                "description": "Configure trading rules if requested",
                "estimated_time": "1 minute"
            }
        ]

    def _generate_fallback_strategy(self, intent: StrategyIntent) -> Dict[str, Any]:
        """Generate fallback strategy without AI"""
        return {
            "steps": self._extract_strategy_steps(""),  # Uses default steps
            "rationale": f"Transfer {intent.amount} {intent.source_token} from {intent.source_chain} to {intent.target_chain}",
            "source_chain": intent.source_chain,
            "target_chain": intent.target_chain,
            "source_token": intent.source_token,
            "target_token": intent.target_token,
            "amount": intent.amount,
            "automation_rules": intent.automation_rules,
            "execution_order": ["quote", "bridge", "confirm", "automate"]
        }

    async def _validate_strategy(self, strategy_plan: Dict[str, Any], intent: StrategyIntent) -> Dict[str, Any]:
        """Validate and optimize strategy parameters"""
        validated = strategy_plan.copy()
        
        # Validate amount limits
        if validated["amount"] > 100000:  # $100k limit for safety
            validated["amount"] = 100000
            validated["validation_warnings"] = ["Amount capped at $100k for safety"]
        
        # Validate chain support
        supported_chains = ["ethereum", "polygon", "arbitrum", "optimism", "bsc", "avalanche"]
        if validated["source_chain"] not in supported_chains:
            validated["source_chain"] = "ethereum"
            validated.setdefault("validation_warnings", []).append("Defaulted to Ethereum as source chain")
        
        # Add optimization suggestions
        validated["optimizations"] = [
            "Consider splitting large amounts across multiple transactions",
            "Monitor gas fees for optimal execution timing",
            "Set up price alerts for better entry points"
        ]
        
        return validated

    async def _assess_strategy_risks(
        self, 
        strategy: Dict[str, Any], 
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks of the proposed strategy"""
        
        risk_assessment = {
            "overall_risk": "medium",
            "risk_factors": [],
            "mitigation_suggestions": [],
            "risk_score": 50  # 0-100
        }
        
        # Bridge risk assessment
        if strategy["amount"] > 50000:
            risk_assessment["risk_factors"].append("Large amount increases bridge risk")
            risk_assessment["risk_score"] += 15
        
        # Market volatility risk
        source_token = strategy.get("source_token", "")
        if source_token in market_context.get("volatility", {}):
            volatility = market_context["volatility"][source_token]
            if volatility > 10:  # >10% daily change
                risk_assessment["risk_factors"].append(f"{source_token} showing high volatility")
                risk_assessment["risk_score"] += 10
        
        # Time risk
        if strategy.get("execution_order", []):
            risk_assessment["risk_factors"].append("Multi-step execution introduces timing risk")
            risk_assessment["risk_score"] += 5
        
        # Determine overall risk level
        if risk_assessment["risk_score"] > 70:
            risk_assessment["overall_risk"] = "high"
        elif risk_assessment["risk_score"] < 30:
            risk_assessment["overall_risk"] = "low"
        
        # Add mitigation suggestions
        risk_assessment["mitigation_suggestions"] = [
            "Set slippage limits to protect against price movements",
            "Use reputable bridge providers with insurance",
            "Consider staging large transfers over time",
            "Monitor bridge transaction status closely"
        ]
        
        return risk_assessment

    def _estimate_execution_time(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate strategy execution time"""
        base_time = 300  # 5 minutes base
        
        # Add time for automation setup
        if strategy.get("automation_rules"):
            base_time += 60  # 1 minute per rule
        
        # Add time based on amount (larger amounts take longer)
        amount = strategy.get("amount", 1000)
        if amount > 10000:
            base_time += 120  # 2 extra minutes for large amounts
        
        return {
            "estimated_seconds": base_time,
            "estimated_minutes": base_time // 60,
            "breakdown": {
                "quote_generation": 30,
                "bridge_execution": 300,
                "confirmation": 120,
                "automation_setup": 60 if strategy.get("automation_rules") else 0
            }
        }

    def _estimate_costs(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate strategy execution costs"""
        base_fee = 0.001 * strategy.get("amount", 1000)  # 0.1% base fee
        
        # Bridge fees vary by provider and amount
        bridge_fee = max(5, 0.002 * strategy.get("amount", 1000))  # Min $5 or 0.2%
        
        # Gas fees estimate
        gas_fee = 20  # $20 average gas cost
        
        total_cost = base_fee + bridge_fee + gas_fee
        
        return {
            "total_usd": round(total_cost, 2),
            "breakdown": {
                "platform_fee": round(base_fee, 2),
                "bridge_fee": round(bridge_fee, 2),
                "gas_fee": round(gas_fee, 2)
            },
            "percentage_of_amount": round((total_cost / strategy.get("amount", 1000)) * 100, 3)
        }

    def _get_fallback_suggestions(self, user_input: str) -> List[str]:
        """Provide fallback suggestions when strategy generation fails"""
        return [
            "Try: 'Transfer 1 ETH from Ethereum to Hyperliquid'",
            "Try: 'Move 5000 USDC to Hyperliquid and set up DCA for ETH'", 
            "Try: 'Bridge my tokens to Hyperliquid and buy ETH when price drops below $3000'",
            "Use more specific amounts and token names",
            "Specify source blockchain (Ethereum, Polygon, etc.)"
        ]

    async def explain_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable explanation of strategy"""
        
        explanation_prompt = f"""
        Explain this cross-chain strategy in simple terms:
        
        Strategy: {json.dumps(strategy_data, indent=2)}
        
        Provide:
        1. What will happen step by step
        2. Why this approach is recommended  
        3. What risks to be aware of
        4. Expected timeline and costs
        
        Make it easy to understand for non-technical users.
        """
        
        try:
            if self.ai_service:
                ai_response = await self.ai_service.analyze_automation_request(
                    user_request=explanation_prompt,
                    market_data={},
                    portfolio_data={}
                )
                
                return {
                    "explanation": ai_response.get("analysis", ""),
                    "summary": f"Moving {strategy_data.get('amount', 0)} {strategy_data.get('source_token', '')} from {strategy_data.get('source_chain', '')} to {strategy_data.get('target_chain', '')}",
                    "key_points": [
                        "Assets will be bridged cross-chain securely",
                        "Multiple route options will be compared",
                        "Transaction status will be monitored",
                        "Automation rules can be set up after transfer"
                    ]
                }
            else:
                # Fallback explanation
                return {
                    "explanation": f"This strategy will transfer {strategy_data.get('amount', 0)} {strategy_data.get('source_token', '')} from {strategy_data.get('source_chain', '')} to {strategy_data.get('target_chain', '')} using the most cost-effective bridge route.",
                    "summary": "Cross-chain asset transfer with optimization",
                    "key_points": [
                        "Secure cross-chain bridge transfer",
                        "Automatic route optimization", 
                        "Real-time status monitoring",
                        "Optional automation setup"
                    ]
                }
                
        except Exception as e:
            logger.error("Strategy explanation failed", error=str(e))
            return {
                "explanation": "Unable to generate detailed explanation",
                "summary": "Cross-chain transfer strategy",
                "key_points": ["Transfer assets between blockchains"]
            }

# Global AI strategy generator instance
_ai_strategy_generator = None

def get_ai_strategy_generator() -> AIStrategyGenerator:
    """Get global AI strategy generator instance"""
    global _ai_strategy_generator
    if _ai_strategy_generator is None:
        _ai_strategy_generator = AIStrategyGenerator()
    return _ai_strategy_generator