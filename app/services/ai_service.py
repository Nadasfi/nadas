"""
AWS Bedrock Claude 3.5 Sonnet Integration
AI-powered automation analysis and recommendations
"""

import json
import boto3
from typing import Dict, List, Any, Optional
import structlog
from botocore.exceptions import ClientError

from app.core.config import settings

logger = structlog.get_logger(__name__)

class ClaudeAIService:
    """AWS Bedrock Claude 3.5 Sonnet AI Service"""
    
    def __init__(self):
        self.bedrock_client = None
        # Use inference profile IDs for cross-region models
        self.model_id = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"  # EU Inference Profile
        self.model_id_haiku = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"  # Same for both
        # Using inference profile ID instead of direct model ID
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AWS Bedrock client"""
        try:
            # Try Bearer Token first (new API key system)
            bearer_token = getattr(settings, 'AWS_BEARER_TOKEN_BEDROCK', None)
            
            if bearer_token:
                # Use Bearer Token authentication
                import boto3.session
                session = boto3.session.Session()
                self.bedrock_client = session.client(
                    'bedrock-runtime',
                    region_name='eu-north-1'  # EU region for inference profile
                )
                # Add bearer token to client config
                self.bedrock_client.meta.config.signature_version = 'UNSIGNED'
                self.bearer_token = bearer_token
                logger.info("AWS Bedrock client initialized with Bearer Token")
            else:
                # Fallback to traditional IAM credentials
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name='us-west-2',
                    aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
                    aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
                )
                self.bearer_token = None
                logger.info("AWS Bedrock client initialized with IAM credentials")
            
            # List client for model discovery
            self.bedrock_list_client = boto3.client(
                'bedrock',
                region_name='us-west-2',
                aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
                aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
            )
            
        except Exception as e:
            logger.error("Failed to initialize Bedrock client", error=str(e))
            self.bedrock_client = None
            self.bedrock_list_client = None
            self.bearer_token = None
    
    async def analyze_automation_request(
        self, 
        user_request: str, 
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze user's automation request and provide intelligent recommendations
        """
        prompt = f"""
        You are a DeFi automation expert analyzing a user's trading automation request.
        
        User Request: {user_request}
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        
        User Portfolio:
        {json.dumps(portfolio_data, indent=2)}
        
        Please analyze this request and provide:
        1. Risk assessment (Low/Medium/High)
        2. Recommended automation type (DCA, Stop-Loss, Grid Trading, etc.)
        3. Suggested parameters (limits, frequency, etc.)
        4. Risk mitigation strategies
        5. Educational insights for the user
        
        Respond in JSON format:
        {{
            "risk_level": "Medium",
            "recommended_automation": "DCA",
            "suggested_parameters": {{
                "limit_usd": 1000,
                "frequency": "daily",
                "target_allocation": {{"ETH": 60, "BTC": 40}}
            }},
            "risk_mitigation": ["Set stop-loss at 20%", "Start with small amounts"],
            "educational_content": {{
                "title": "Understanding DCA Strategy",
                "explanation": "...",
                "benefits": ["Reduces timing risk", "Smooths volatility"],
                "considerations": ["Market trends", "Asset correlation"]
            }},
            "confidence_score": 0.85
        }}
        """
        
        try:
            response = await self._call_claude(prompt, use_haiku=False)
            return json.loads(response)
        except Exception as e:
            logger.error("Failed to analyze automation request", error=str(e))
            return self._fallback_automation_analysis(user_request)
    
    async def generate_risk_warning(
        self,
        automation_config: Dict[str, Any],
        current_market_conditions: Dict[str, Any]
    ) -> str:
        """Generate dynamic risk warnings based on current conditions"""
        
        prompt = f"""
        Generate a concise risk warning for this DeFi automation setup:
        
        Automation Config:
        {json.dumps(automation_config, indent=2)}
        
        Current Market Conditions:
        {json.dumps(current_market_conditions, indent=2)}
        
        Provide a clear, actionable risk warning in 2-3 sentences that highlights:
        - Immediate risks
        - Market condition impacts
        - Recommended actions
        
        Keep it user-friendly but informative.
        """
        
        try:
            return await self._call_claude(prompt, use_haiku=True)
        except Exception as e:
            logger.error("Failed to generate risk warning", error=str(e))
            return "⚠️ Please review your automation settings and current market conditions before proceeding."
    
    async def explain_market_conditions(
        self,
        market_data: Dict[str, Any],
        timeframe: str = "1h"
    ) -> str:
        """Generate AI-powered market condition explanation"""
        
        prompt = f"""
        Analyze the current market conditions and provide a clear explanation:
        
        Market Data ({timeframe}):
        {json.dumps(market_data, indent=2)}
        
        Provide a concise market analysis covering:
        1. Overall market sentiment
        2. Key price movements and trends
        3. Notable trading volumes
        4. Risk factors to watch
        5. Opportunities for automation
        
        Keep the explanation accessible for both beginners and experienced traders.
        Limit to 200-300 words.
        """
        
        try:
            return await self._call_claude(prompt, use_haiku=False)
        except Exception as e:
            logger.error("Failed to explain market conditions", error=str(e))
            return f"Current market showing mixed signals across major assets. ETH and BTC showing typical volatility patterns. Consider conservative automation strategies during uncertain market conditions."
    
    async def chat_with_context(
        self,
        user_message: str,
        market_data: Optional[Dict[str, Any]] = None,
        portfolio_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Chat with Claude using Nadas.fi context and market data"""
        
        system_prompt = """You are the Nadas.fi AI Assistant, an expert guide for Hyperliquid DeFi automation and trading.

IDENTITY & ROLE:
- You are the official AI assistant for Nadas.fi, a secure non-custodial DeFi automation platform
- You specialize in Hyperliquid trading, automation strategies, and risk management
- You have access to real-time market data and user portfolio information

CAPABILITIES:
- Market analysis and price insights for Hyperliquid assets
- Trading strategy recommendations (DCA, grid trading, stop-loss)
- Risk assessment and portfolio optimization
- Automation rule configuration guidance
- Hyperliquid testnet trading education

KNOWLEDGE AREAS:
- Hyperliquid perpetual futures trading
- DeFi automation strategies and best practices
- Risk management and position sizing
- Market analysis and technical indicators
- Nadas.fi's hybrid wallet security model

RESPONSE STYLE:
- Be helpful, informative, and concise
- Use Turkish when user writes in Turkish
- Include specific numbers and data when available
- Provide actionable insights and recommendations
- Always prioritize user safety and risk management

SECURITY FOCUS:
- Always emphasize the importance of risk management
- Recommend starting with small amounts for automation
- Explain Nadas.fi's hybrid wallet model when relevant
- Never recommend risking more than users can afford to lose"""

        # Add market data context if available
        context_parts = []
        if market_data:
            context_parts.append(f"Current Market Data: {json.dumps(market_data, indent=2)}")
        if portfolio_data:
            context_parts.append(f"User Portfolio: {json.dumps(portfolio_data, indent=2)}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nUser: {user_message}\n\nAssistant:"
        
        try:
            return await self._call_claude(full_prompt, use_haiku=False)
        except Exception as e:
            logger.error("Failed to chat with Claude", error=str(e))
            return "Merhaba! Ben Nadas.fi AI asistanınızım. Şu anda teknik bir sorun yaşıyorum, ancak yakında size yardımcı olabileceğim. Hyperliquid trading ve otomasyon konularında sorularınız varsa tekrar deneyin."

    async def _call_claude(self, prompt: str, use_haiku: bool = False) -> str:
        """Make API call to Claude via AWS Bedrock"""
        
        if not self.bedrock_client:
            raise Exception("Bedrock client not initialized")
        
        model_id = self.model_id_haiku if use_haiku else self.model_id
        
        # Prepare the request body
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        try:
            # Use Bearer Token if available
            if hasattr(self, 'bearer_token') and self.bearer_token:
                # Use direct boto3 with Bearer Token in headers
                import botocore.awsrequest
                import botocore.endpoint
                
                # Create request manually with Bearer Token
                request = botocore.awsrequest.AWSRequest(
                    method='POST',
                    url=f"https://bedrock-runtime.eu-north-1.amazonaws.com/model/{model_id}/invoke",
                    data=body,
                    headers={
                        'Authorization': f'Bearer {self.bearer_token}',
                        'Content-Type': 'application/json'
                    }
                )
                
                # Use urllib3 directly for Bearer Token auth
                import urllib3
                http = urllib3.PoolManager()
                
                response = http.request(
                    'POST',
                    f"https://bedrock-runtime.eu-north-1.amazonaws.com/model/{model_id}/invoke",
                    body=body,
                    headers={
                        'Authorization': f'Bearer {self.bearer_token}',
                        'Content-Type': 'application/json'
                    }
                )
                
                if response.status != 200:
                    raise Exception(f"Bedrock API error: {response.status} - {response.data.decode()}")
                
                response_data = json.loads(response.data.decode())
                
                # Handle Claude 3.5 Sonnet response format
                if 'content' in response_data and len(response_data['content']) > 0:
                    return response_data['content'][0]['text']
                elif 'message' in response_data:
                    return response_data['message']
                else:
                    return str(response_data)
            else:
                # Traditional boto3 IAM credentials
                response = self.bedrock_client.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType='application/json'
                )
                
                response_body = json.loads(response['body'].read())
                return response_body['content'][0]['text']
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                logger.error("Access denied to Bedrock model - check model access permissions")
            elif error_code == 'ThrottlingException':
                logger.error("Bedrock API throttled - too many requests")
            else:
                logger.error("Bedrock API error", error_code=error_code, error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error calling Claude", error=str(e))
            raise
    
    def _fallback_automation_analysis(self, user_request: str) -> Dict[str, Any]:
        """Fallback analysis when AI is unavailable"""
        return {
            "risk_level": "Medium",
            "recommended_automation": "DCA",
            "suggested_parameters": {
                "limit_usd": 1000,
                "frequency": "daily"
            },
            "risk_mitigation": [
                "Start with small amounts",
                "Monitor market conditions",
                "Set conservative limits"
            ],
            "educational_content": {
                "title": "Getting Started with DeFi Automation",
                "explanation": "AI analysis temporarily unavailable. Using conservative defaults.",
                "benefits": ["Reduces emotional trading", "Consistent strategy execution"],
                "considerations": ["Market volatility", "Gas fees", "Smart contract risks"]
            },
            "confidence_score": 0.5
        }
    
    def list_available_models(self) -> List[str]:
        """List available models in Bedrock"""
        try:
            if not self.bedrock_list_client:
                return []
            
            response = self.bedrock_list_client.list_foundation_models()
            models = []
            for model in response.get('modelSummaries', []):
                if 'anthropic' in model.get('modelId', '').lower():
                    models.append(model['modelId'])
            return models
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Bedrock service health"""
        try:
            if not self.bedrock_client:
                return {"status": "unhealthy", "error": "Client not initialized"}
            
            # List available models first
            available_models = self.list_available_models()
            
            # Simple test call
            test_response = await self._call_claude("Hello, respond with 'OK'", use_haiku=True)
            
            return {
                "status": "healthy",
                "model_id": self.model_id,
                "model_id_haiku": self.model_id_haiku,
                "available_models": available_models,
                "test_response": test_response[:20] + "..." if len(test_response) > 20 else test_response
            }
        except Exception as e:
            available_models = self.list_available_models()
            return {
                "status": "unhealthy", 
                "error": str(e),
                "available_models": available_models
            }

# Global AI service instance
_ai_service = ClaudeAIService()

def get_ai_service() -> ClaudeAIService:
    """Get the global AI service instance"""
    return _ai_service