"""
Unit tests for AI Strategy Generator
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.ai_strategy_generator import AIStrategyGenerator


class TestAIStrategyGenerator:
    """Test cases for AIStrategyGenerator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = AIStrategyGenerator()
    
    @patch('app.services.ai_strategy_generator.get_cross_chain_orchestrator')
    def test_init_with_orchestrator(self, mock_get_orchestrator):
        """Test generator initialization with orchestrator"""
        mock_orchestrator = Mock()
        mock_get_orchestrator.return_value = mock_orchestrator
        
        generator = AIStrategyGenerator()
        
        assert generator.orchestrator == mock_orchestrator
        mock_get_orchestrator.assert_called_once()

    @pytest.mark.asyncio
    @patch('anthropic.Anthropic')
    async def test_generate_strategy_from_text_basic(self, mock_anthropic_class):
        """Test basic strategy generation from text"""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''
        {
            "strategy": {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH",
                "target_token": "USDC",
                "amount": 1.0
            },
            "risk_assessment": {
                "overall_risk": "medium",
                "risk_factors": ["bridge_risk", "slippage_risk"],
                "risk_mitigation": "Use reputable bridge provider"
            },
            "estimated_costs": {
                "bridge_fee_usd": 15.0,
                "gas_fee_usd": 25.0,
                "total_usd": 40.0
            },
            "explanation": {
                "summary": "Bridge 1 ETH to Hyperliquid",
                "key_points": ["Low slippage", "Fast execution"]
            }
        }
        '''
        mock_client.messages.create.return_value = mock_response
        
        # Mock orchestrator methods
        mock_orchestrator = Mock()
        mock_strategy = Mock()
        mock_strategy.id = "test-strategy-id"
        mock_orchestrator.create_strategy.return_value = mock_strategy
        
        with patch.object(self.generator, 'orchestrator', mock_orchestrator):
            result = await self.generator.generate_strategy_from_text(
                "Transfer 1 ETH to Hyperliquid",
                "0x123456789abcdef"
            )
        
        # Verify AI was called
        mock_client.messages.create.assert_called_once()
        
        # Verify orchestrator create_strategy was called
        mock_orchestrator.create_strategy.assert_called_once()
        
        # Verify result structure
        assert result["strategy_id"] == "test-strategy-id"
        assert "ai_analysis" in result
        assert "explanation" in result
        assert result["ai_analysis"]["strategy"]["source_chain"] == "ethereum"
        assert result["ai_analysis"]["strategy"]["target_chain"] == "hyperliquid"

    @pytest.mark.asyncio
    async def test_generate_strategy_with_portfolio_context(self):
        """Test strategy generation with portfolio context"""
        portfolio_context = {
            "balances": {
                "ethereum": {"ETH": 2.5, "USDC": 1000},
                "polygon": {"MATIC": 500}
            },
            "total_value_usd": 5000
        }
        
        with patch('anthropic.Anthropic') as mock_anthropic_class, \
             patch.object(self.generator, 'orchestrator') as mock_orchestrator:
            
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = '{"strategy": {"amount": 1.0}, "explanation": {"summary": "test"}}'
            mock_client.messages.create.return_value = mock_response
            
            mock_strategy = Mock()
            mock_strategy.id = "test-id"
            mock_orchestrator.create_strategy.return_value = mock_strategy
            
            result = await self.generator.generate_strategy_from_text(
                "Move some ETH to Hyperliquid",
                "0x123",
                portfolio_context
            )
            
            # Verify portfolio context was included in AI prompt
            call_args = mock_client.messages.create.call_args[1]
            messages = call_args["messages"]
            user_message = next(msg for msg in messages if msg["role"] == "user")
            
            assert "Portfolio context" in user_message["content"]
            assert "ETH: 2.5" in user_message["content"]
            assert "total_value_usd: 5000" in user_message["content"]

    def test_parse_ai_response_valid_json(self):
        """Test parsing valid JSON response from AI"""
        valid_json = '''
        {
            "strategy": {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "amount": 1.0
            },
            "risk_assessment": {
                "overall_risk": "low"
            }
        }
        '''
        
        result = self.generator._parse_ai_response(valid_json)
        
        assert result["strategy"]["source_chain"] == "ethereum"
        assert result["risk_assessment"]["overall_risk"] == "low"

    def test_parse_ai_response_invalid_json(self):
        """Test parsing invalid JSON response"""
        invalid_json = "This is not valid JSON"
        
        result = self.generator._parse_ai_response(invalid_json)
        
        # Should return fallback structure
        assert "strategy" in result
        assert "explanation" in result
        assert result["explanation"]["summary"] == "AI response could not be parsed"

    def test_parse_ai_response_partial_json(self):
        """Test parsing JSON with missing required fields"""
        partial_json = '''
        {
            "strategy": {
                "amount": 1.0
            }
        }
        '''
        
        result = self.generator._parse_ai_response(partial_json)
        
        # Should fill in missing fields with defaults
        assert result["strategy"]["source_chain"] == "ethereum"
        assert result["strategy"]["target_chain"] == "hyperliquid"
        assert "risk_assessment" in result
        assert "estimated_costs" in result
        assert "explanation" in result

    def test_format_portfolio_context_full(self):
        """Test formatting comprehensive portfolio context"""
        portfolio = {
            "balances": {
                "ethereum": {"ETH": 2.5, "USDC": 1000},
                "polygon": {"MATIC": 500, "USDT": 200}
            },
            "total_value_usd": 5000,
            "recent_transactions": [
                {"type": "bridge", "amount": 1.0, "token": "ETH"}
            ]
        }
        
        result = self.generator._format_portfolio_context(portfolio)
        
        assert "Portfolio context:" in result
        assert "ethereum:" in result
        assert "ETH: 2.5" in result
        assert "USDC: 1000" in result
        assert "polygon:" in result
        assert "MATIC: 500" in result
        assert "Total portfolio value: $5000" in result
        assert "Recent activity:" in result
        assert "bridge 1.0 ETH" in result

    def test_format_portfolio_context_minimal(self):
        """Test formatting minimal portfolio context"""
        portfolio = {
            "balances": {
                "ethereum": {"ETH": 1.0}
            }
        }
        
        result = self.generator._format_portfolio_context(portfolio)
        
        assert "Portfolio context:" in result
        assert "ethereum:" in result
        assert "ETH: 1.0" in result

    def test_format_portfolio_context_empty(self):
        """Test formatting empty portfolio context"""
        result = self.generator._format_portfolio_context({})
        
        assert result == ""

    def test_format_portfolio_context_none(self):
        """Test formatting None portfolio context"""
        result = self.generator._format_portfolio_context(None)
        
        assert result == ""

    def test_build_ai_prompt_basic(self):
        """Test building basic AI prompt without portfolio"""
        prompt = self.generator._build_ai_prompt(
            "Transfer 1 ETH to Hyperliquid",
            "0x123456789abcdef"
        )
        
        assert "Transfer 1 ETH to Hyperliquid" in prompt
        assert "0x123456789abcdef" in prompt
        assert "JSON format" in prompt
        assert "strategy" in prompt
        assert "risk_assessment" in prompt

    def test_build_ai_prompt_with_portfolio(self):
        """Test building AI prompt with portfolio context"""
        portfolio = {
            "balances": {"ethereum": {"ETH": 2.0}},
            "total_value_usd": 3000
        }
        
        prompt = self.generator._build_ai_prompt(
            "Move some ETH",
            "0x123",
            portfolio
        )
        
        assert "Move some ETH" in prompt
        assert "Portfolio context:" in prompt
        assert "ETH: 2.0" in prompt
        assert "$3000" in prompt

    def test_create_fallback_strategy(self):
        """Test creating fallback strategy for unparseable AI response"""
        result = self.generator._create_fallback_strategy("Bridge ETH to Hyperliquid")
        
        assert result["strategy"]["source_chain"] == "ethereum"
        assert result["strategy"]["target_chain"] == "hyperliquid"
        assert result["strategy"]["source_token"] == "ETH"
        assert result["strategy"]["amount"] == 1.0
        assert result["explanation"]["summary"] == "Bridge ETH to Hyperliquid"
        assert "fallback" in result["explanation"]["summary"].lower() or "error" in result["explanation"]["summary"].lower()

    @pytest.mark.asyncio
    async def test_generate_strategy_ai_failure(self):
        """Test handling AI API failure"""
        with patch('anthropic.Anthropic') as mock_anthropic_class, \
             patch.object(self.generator, 'orchestrator') as mock_orchestrator:
            
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")
            
            mock_strategy = Mock()
            mock_strategy.id = "fallback-id"
            mock_orchestrator.create_strategy.return_value = mock_strategy
            
            result = await self.generator.generate_strategy_from_text(
                "Transfer ETH",
                "0x123"
            )
            
            # Should still return valid result with fallback
            assert result["strategy_id"] == "fallback-id"
            assert "ai_analysis" in result
            assert result["ai_analysis"]["strategy"]["source_chain"] == "ethereum"

    @pytest.mark.asyncio
    async def test_generate_strategy_orchestrator_failure(self):
        """Test handling orchestrator creation failure"""
        with patch('anthropic.Anthropic') as mock_anthropic_class, \
             patch.object(self.generator, 'orchestrator') as mock_orchestrator:
            
            # Mock successful AI response
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = '{"strategy": {"amount": 1.0}, "explanation": {"summary": "test"}}'
            mock_client.messages.create.return_value = mock_response
            
            # Mock orchestrator failure
            mock_orchestrator.create_strategy.side_effect = Exception("Database error")
            
            with pytest.raises(Exception, match="Database error"):
                await self.generator.generate_strategy_from_text(
                    "Transfer ETH",
                    "0x123"
                )

    def test_extract_tokens_from_text(self):
        """Test token extraction from natural language"""
        test_cases = [
            ("Transfer 1 ETH to Hyperliquid", "ETH", "USDC"),
            ("Bridge USDC to get ETH on Hyperliquid", "USDC", "ETH"),
            ("Move some Bitcoin to HL", "BTC", "USDC"),
            ("Send tokens to Hyperliquid", "ETH", "USDC"),  # default
            ("Bridge 1000 USDT for ETH", "USDT", "ETH")
        ]
        
        for text, expected_source, expected_target in test_cases:
            source, target = self.generator._extract_tokens_from_text(text)
            assert source == expected_source
            assert target == expected_target

    def test_extract_amount_from_text(self):
        """Test amount extraction from natural language"""
        test_cases = [
            ("Transfer 1 ETH", 1.0),
            ("Bridge 2.5 tokens", 2.5),
            ("Move 1000 USDC", 1000.0),
            ("Send some ETH", 1.0),  # default
            ("Bridge 0.1 BTC", 0.1),
            ("Transfer 5,000 tokens", 5000.0)
        ]
        
        for text, expected_amount in test_cases:
            amount = self.generator._extract_amount_from_text(text)
            assert amount == expected_amount

    def test_extract_chains_from_text(self):
        """Test chain extraction from natural language"""
        test_cases = [
            ("From Ethereum to Hyperliquid", "ethereum", "hyperliquid"),
            ("Bridge from Polygon to HL", "polygon", "hyperliquid"),
            ("Move from Arbitrum to Hyperliquid", "arbitrum", "hyperliquid"),
            ("Transfer to Hyperliquid", "ethereum", "hyperliquid"),  # default source
            ("Bridge tokens", "ethereum", "hyperliquid")  # default both
        ]
        
        for text, expected_source, expected_target in test_cases:
            source, target = self.generator._extract_chains_from_text(text)
            assert source == expected_source
            assert target == expected_target


if __name__ == "__main__":
    pytest.main([__file__])