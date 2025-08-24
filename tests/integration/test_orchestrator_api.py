"""
Integration tests for Cross-Chain Orchestrator API endpoints
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from app.main import app
from app.services.cross_chain_orchestrator import CrossChainStrategy, StrategyStatus


client = TestClient(app)

class TestOrchestratorAPI:
    """Test cases for orchestrator API endpoints"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test fixtures"""
        # Mock authentication for all tests
        with patch('app.core.dependencies.get_current_user') as mock_auth:
            mock_auth.return_value = {
                "id": "test-user-id",
                "wallet_address": "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A",
                "email": "test@example.com"
            }
            yield

    def test_create_strategy_success(self):
        """Test successful strategy creation"""
        strategy_data = {
            "source_chain": "ethereum",
            "target_chain": "hyperliquid", 
            "source_token": "ETH",
            "target_token": "USDC",
            "amount": 1.0,
            "risk_tolerance": "medium"
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_strategy = Mock()
            mock_strategy.id = "test-strategy-id"
            mock_strategy.to_dict.return_value = {
                "id": "test-strategy-id",
                "status": "pending",
                "user_address": "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
            }
            mock_orchestrator.create_strategy.return_value = mock_strategy
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["strategy_id"] == "test-strategy-id"
            assert "strategy" in data
            
            mock_orchestrator.create_strategy.assert_called_once()

    def test_create_strategy_validation_error(self):
        """Test strategy creation with invalid data"""
        invalid_data = {
            "source_chain": "ethereum",
            # Missing required fields
        }
        
        response = client.post(
            "/api/v1/orchestrator/create-strategy",
            json=invalid_data
        )
        
        assert response.status_code == 422  # Validation error

    def test_create_strategy_orchestrator_error(self):
        """Test strategy creation with orchestrator error"""
        strategy_data = {
            "source_chain": "ethereum",
            "target_chain": "hyperliquid",
            "source_token": "ETH", 
            "target_token": "USDC",
            "amount": 1.0
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.create_strategy.side_effect = Exception("Database error")
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "error" in data

    def test_ai_generate_strategy_success(self):
        """Test successful AI strategy generation"""
        request_data = {
            "user_input": "Transfer 1 ETH to Hyperliquid",
            "portfolio_context": {
                "balances": {
                    "ethereum": {"ETH": 2.0, "USDC": 1000}
                }
            }
        }
        
        mock_ai_result = {
            "strategy_id": "ai-strategy-id",
            "ai_analysis": {
                "strategy": {
                    "source_chain": "ethereum",
                    "target_chain": "hyperliquid",
                    "amount": 1.0
                },
                "risk_assessment": {"overall_risk": "medium"},
                "estimated_costs": {"total_usd": 40.0}
            },
            "explanation": {
                "summary": "Bridge 1 ETH to Hyperliquid",
                "key_points": ["Fast execution", "Low fees"]
            }
        }
        
        with patch('app.services.ai_strategy_generator.AIStrategyGenerator.generate_strategy_from_text') as mock_generate:
            mock_generate.return_value = mock_ai_result
            
            response = client.post(
                "/api/v1/orchestrator/ai-generate-strategy",
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["strategy_id"] == "ai-strategy-id"
            assert "ai_analysis" in data
            assert "explanation" in data
            
            mock_generate.assert_called_once_with(
                "Transfer 1 ETH to Hyperliquid",
                "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A",
                request_data["portfolio_context"]
            )

    def test_ai_generate_strategy_empty_input(self):
        """Test AI generation with empty user input"""
        response = client.post(
            "/api/v1/orchestrator/ai-generate-strategy",
            json={"user_input": ""}
        )
        
        assert response.status_code == 422

    def test_ai_generate_strategy_ai_error(self):
        """Test AI generation with AI service error"""
        request_data = {
            "user_input": "Transfer ETH"
        }
        
        with patch('app.services.ai_strategy_generator.AIStrategyGenerator.generate_strategy_from_text') as mock_generate:
            mock_generate.side_effect = Exception("AI service unavailable")
            
            response = client.post(
                "/api/v1/orchestrator/ai-generate-strategy",
                json=request_data
            )
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "AI service unavailable" in data["error"]

    def test_analyze_routes_success(self):
        """Test successful route analysis"""
        mock_quotes = [
            {
                "provider": "lifi",
                "estimated_fee_usd": 15.0,
                "estimated_time_minutes": 10,
                "confidence_score": 85
            },
            {
                "provider": "gluex",
                "estimated_fee_usd": 12.0,
                "estimated_time_minutes": 5,
                "confidence_score": 90
            }
        ]
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.analyze_routes.return_value = mock_quotes
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/analyze-routes/test-strategy-id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["routes"]) == 2
            assert data["routes"][0]["provider"] == "gluex"  # Best route first
            
            mock_orchestrator.analyze_routes.assert_called_once_with("test-strategy-id")

    def test_analyze_routes_not_found(self):
        """Test route analysis for non-existent strategy"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.analyze_routes.side_effect = ValueError("Strategy not found")
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/analyze-routes/invalid-id")
            
            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False

    def test_analyze_routes_no_routes_available(self):
        """Test route analysis when no routes are available"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.analyze_routes.return_value = []
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/analyze-routes/test-strategy-id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["routes"]) == 0

    def test_select_route_success(self):
        """Test successful route selection"""
        request_data = {"route_index": 0}
        
        mock_selected_route = {
            "provider": "lifi",
            "estimated_fee_usd": 15.0,
            "estimated_time_minutes": 10
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.select_optimal_route.return_value = mock_selected_route
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post(
                "/api/v1/orchestrator/select-route/test-strategy-id",
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["selected_route"]["provider"] == "lifi"
            
            mock_orchestrator.select_optimal_route.assert_called_once_with(
                "test-strategy-id", 0
            )

    def test_select_route_invalid_index(self):
        """Test route selection with invalid index"""
        request_data = {"route_index": 99}
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.select_optimal_route.side_effect = ValueError("Invalid route selection")
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post(
                "/api/v1/orchestrator/select-route/test-strategy-id",
                json=request_data
            )
            
            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False

    def test_execute_strategy_success(self):
        """Test successful strategy execution"""
        mock_execution_result = {
            "success": True,
            "strategy_id": "test-strategy-id",
            "bridge_tx_hash": "0xabcdef123456789",
            "status": "waiting_confirmation",
            "estimated_completion": "2024-01-01T12:00:00"
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.execute_strategy.return_value = mock_execution_result
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/execute-strategy/test-strategy-id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["bridge_tx_hash"] == "0xabcdef123456789"
            assert data["status"] == "waiting_confirmation"
            
            mock_orchestrator.execute_strategy.assert_called_once_with("test-strategy-id")

    def test_execute_strategy_no_route_selected(self):
        """Test executing strategy without route selection"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.execute_strategy.side_effect = ValueError("No route selected")
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/execute-strategy/test-strategy-id")
            
            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False

    def test_execute_strategy_execution_failure(self):
        """Test strategy execution failure"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.execute_strategy.side_effect = Exception("Bridge execution failed")
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/execute-strategy/test-strategy-id")
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "Bridge execution failed" in data["error"]

    def test_get_strategy_success(self):
        """Test successful strategy retrieval"""
        mock_strategy = Mock()
        mock_strategy.to_dict.return_value = {
            "id": "test-strategy-id",
            "status": "completed",
            "user_address": "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A",
            "progress_percentage": 100
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.get_strategy.return_value = mock_strategy
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/orchestrator/strategy/test-strategy-id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["strategy"]["id"] == "test-strategy-id"
            assert data["strategy"]["status"] == "completed"

    def test_get_strategy_not_found(self):
        """Test retrieving non-existent strategy"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.get_strategy.return_value = None
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/orchestrator/strategy/invalid-id")
            
            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False

    def test_get_strategy_unauthorized_access(self):
        """Test accessing strategy belonging to different user"""
        mock_strategy = Mock()
        mock_strategy.user_address = "0x999999999999999999999999999999999999999"  # Different user
        mock_strategy.to_dict.return_value = {
            "id": "test-strategy-id",
            "user_address": "0x999999999999999999999999999999999999999"
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.get_strategy.return_value = mock_strategy
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/orchestrator/strategy/test-strategy-id")
            
            assert response.status_code == 403
            data = response.json()
            assert data["success"] is False

    def test_list_user_strategies_success(self):
        """Test listing user strategies"""
        mock_strategy1 = Mock()
        mock_strategy1.to_dict.return_value = {
            "id": "strategy-1",
            "status": "completed"
        }
        mock_strategy2 = Mock()
        mock_strategy2.to_dict.return_value = {
            "id": "strategy-2", 
            "status": "pending"
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.get_user_strategies.return_value = [mock_strategy1, mock_strategy2]
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/orchestrator/user-strategies")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["strategies"]) == 2
            assert data["strategies"][0]["id"] == "strategy-1"
            assert data["strategies"][1]["id"] == "strategy-2"

    def test_list_user_strategies_empty(self):
        """Test listing user strategies when none exist"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.get_user_strategies.return_value = []
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/orchestrator/user-strategies")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["strategies"]) == 0

    def test_cancel_strategy_success(self):
        """Test successful strategy cancellation"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.cancel_strategy.return_value = True
            
            # Mock strategy ownership check
            mock_strategy = Mock()
            mock_strategy.user_address = "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
            mock_orchestrator.get_strategy.return_value = mock_strategy
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/cancel-strategy/test-strategy-id")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Strategy cancelled successfully"

    def test_cancel_strategy_not_cancellable(self):
        """Test cancelling strategy that cannot be cancelled"""
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.cancel_strategy.return_value = False
            
            # Mock strategy ownership check
            mock_strategy = Mock()
            mock_strategy.user_address = "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
            mock_orchestrator.get_strategy.return_value = mock_strategy
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.post("/api/v1/orchestrator/cancel-strategy/test-strategy-id")
            
            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False

    def test_get_orchestrator_statistics(self):
        """Test getting orchestrator performance statistics"""
        mock_stats = {
            "total_strategies": 100,
            "completed": 85,
            "failed": 5,
            "success_rate": 85.0,
            "provider_usage": {"lifi": 40, "gluex": 60},
            "average_fee_usd": 18.5
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            mock_orchestrator.get_execution_statistics.return_value = mock_stats
            mock_get_orchestrator.return_value = mock_orchestrator
            
            response = client.get("/api/v1/orchestrator/statistics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["statistics"]["total_strategies"] == 100
            assert data["statistics"]["success_rate"] == 85.0


class TestOrchestratorAPIIntegration:
    """Integration tests for complete orchestrator workflows"""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test fixtures"""
        with patch('app.core.dependencies.get_current_user') as mock_auth:
            mock_auth.return_value = {
                "id": "test-user-id",
                "wallet_address": "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A",
                "email": "test@example.com"
            }
            yield

    def test_full_strategy_workflow(self):
        """Test complete strategy creation, analysis, selection, and execution"""
        strategy_data = {
            "source_chain": "ethereum",
            "target_chain": "hyperliquid",
            "source_token": "ETH",
            "target_token": "USDC",
            "amount": 1.0
        }
        
        mock_quotes = [
            {"provider": "lifi", "estimated_fee_usd": 15.0, "estimated_time_minutes": 10},
            {"provider": "gluex", "estimated_fee_usd": 12.0, "estimated_time_minutes": 5}
        ]
        
        mock_execution_result = {
            "success": True,
            "strategy_id": "test-strategy-id",
            "bridge_tx_hash": "0xabcdef123456789",
            "status": "waiting_confirmation"
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            mock_orchestrator = Mock()
            
            # Mock strategy creation
            mock_strategy = Mock()
            mock_strategy.id = "test-strategy-id"
            mock_strategy.to_dict.return_value = {"id": "test-strategy-id", "status": "pending"}
            mock_orchestrator.create_strategy.return_value = mock_strategy
            
            # Mock other operations
            mock_orchestrator.analyze_routes.return_value = mock_quotes
            mock_orchestrator.select_optimal_route.return_value = mock_quotes[1]  # Select GlueX
            mock_orchestrator.execute_strategy.return_value = mock_execution_result
            
            mock_get_orchestrator.return_value = mock_orchestrator
            
            # 1. Create strategy
            create_response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            assert create_response.status_code == 200
            strategy_id = create_response.json()["strategy_id"]
            
            # 2. Analyze routes
            analyze_response = client.post(f"/api/v1/orchestrator/analyze-routes/{strategy_id}")
            assert analyze_response.status_code == 200
            routes = analyze_response.json()["routes"]
            assert len(routes) == 2
            
            # 3. Select optimal route (GlueX with lower fees)
            select_response = client.post(
                f"/api/v1/orchestrator/select-route/{strategy_id}",
                json={"route_index": 1}
            )
            assert select_response.status_code == 200
            selected_route = select_response.json()["selected_route"]
            assert selected_route["provider"] == "gluex"
            
            # 4. Execute strategy
            execute_response = client.post(f"/api/v1/orchestrator/execute-strategy/{strategy_id}")
            assert execute_response.status_code == 200
            execution_result = execute_response.json()
            assert execution_result["success"] is True
            assert "bridge_tx_hash" in execution_result

    def test_ai_to_execution_workflow(self):
        """Test AI strategy generation to execution workflow"""
        ai_request = {
            "user_input": "Transfer 1 ETH to Hyperliquid",
            "portfolio_context": {"balances": {"ethereum": {"ETH": 2.0}}}
        }
        
        mock_ai_result = {
            "strategy_id": "ai-strategy-id",
            "ai_analysis": {"strategy": {"amount": 1.0}},
            "explanation": {"summary": "Bridge ETH to Hyperliquid"}
        }
        
        mock_quotes = [
            {"provider": "lifi", "estimated_fee_usd": 15.0, "estimated_time_minutes": 10}
        ]
        
        with patch('app.services.ai_strategy_generator.AIStrategyGenerator.generate_strategy_from_text') as mock_ai, \
             patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            
            mock_ai.return_value = mock_ai_result
            
            mock_orchestrator = Mock()
            mock_orchestrator.analyze_routes.return_value = mock_quotes
            mock_orchestrator.select_optimal_route.return_value = mock_quotes[0]
            mock_orchestrator.execute_strategy.return_value = {
                "success": True,
                "strategy_id": "ai-strategy-id",
                "bridge_tx_hash": "0x123abc"
            }
            mock_get_orchestrator.return_value = mock_orchestrator
            
            # 1. Generate AI strategy
            ai_response = client.post(
                "/api/v1/orchestrator/ai-generate-strategy",
                json=ai_request
            )
            assert ai_response.status_code == 200
            strategy_id = ai_response.json()["strategy_id"]
            
            # 2. Analyze routes
            analyze_response = client.post(f"/api/v1/orchestrator/analyze-routes/{strategy_id}")
            assert analyze_response.status_code == 200
            
            # 3. Auto-select best route
            select_response = client.post(
                f"/api/v1/orchestrator/select-route/{strategy_id}",
                json={"route_index": 0}
            )
            assert select_response.status_code == 200
            
            # 4. Execute
            execute_response = client.post(f"/api/v1/orchestrator/execute-strategy/{strategy_id}")
            assert execute_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])