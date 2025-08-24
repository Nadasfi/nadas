"""
End-to-End tests for Cross-Chain Orchestrator Workflow
Tests the complete flow from strategy creation to execution
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.cross_chain_orchestrator import CrossChainOrchestrator, StrategyStatus


client = TestClient(app)

class TestCrossChainWorkflowE2E:
    """End-to-end tests for complete cross-chain workflows"""

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

    def test_manual_strategy_complete_workflow(self):
        """Test complete manual strategy workflow from creation to completion"""
        
        # Mock all external dependencies
        mock_lifi_quote = {
            "fee_usd": 15.0,
            "time_minutes": 10,
            "route_data": {"path": "ethereum->hyperliquid"}
        }
        mock_gluex_quote = {
            "fee_usd": 12.0,
            "time_minutes": 5,
            "route_data": {"path": "direct"}
        }
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            
            # Mock adapter methods
            orchestrator._get_lifi_quote = AsyncMock(return_value=mock_lifi_quote)
            orchestrator._get_gluex_quote = AsyncMock(return_value=mock_gluex_quote)
            orchestrator._execute_bridge_transaction = AsyncMock(return_value={
                "success": True,
                "tx_hash": "0xabcdef123456789",
                "provider": "gluex"
            })
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Step 1: Create Strategy
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH",
                "target_token": "USDC",
                "amount": 1.0,
                "risk_tolerance": "medium"
            }
            
            create_response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            
            assert create_response.status_code == 200
            create_data = create_response.json()
            assert create_data["success"] is True
            strategy_id = create_data["strategy_id"]
            
            # Verify strategy was created
            strategy = orchestrator.get_strategy(strategy_id)
            assert strategy is not None
            assert strategy.status == StrategyStatus.PENDING
            
            # Step 2: Analyze Routes
            analyze_response = client.post(f"/api/v1/orchestrator/analyze-routes/{strategy_id}")
            
            assert analyze_response.status_code == 200
            analyze_data = analyze_response.json()
            assert analyze_data["success"] is True
            assert len(analyze_data["routes"]) == 2
            
            # Verify routes are sorted by cost (GlueX should be first with lower fee)
            best_route = analyze_data["routes"][0]
            assert best_route["provider"] == "gluex"
            assert best_route["estimated_fee_usd"] == 12.0
            
            # Verify strategy status updated
            strategy = orchestrator.get_strategy(strategy_id)
            assert strategy.status == StrategyStatus.QUOTE_READY
            
            # Step 3: Select Optimal Route
            select_response = client.post(
                f"/api/v1/orchestrator/select-route/{strategy_id}",
                json={"route_index": 0}  # Select best route
            )
            
            assert select_response.status_code == 200
            select_data = select_response.json()
            assert select_data["success"] is True
            assert select_data["selected_route"]["provider"] == "gluex"
            
            # Verify route selection
            strategy = orchestrator.get_strategy(strategy_id)
            assert strategy.selected_route is not None
            assert strategy.selected_route["provider"] == "gluex"
            assert strategy.estimated_completion is not None
            
            # Step 4: Execute Strategy
            execute_response = client.post(f"/api/v1/orchestrator/execute-strategy/{strategy_id}")
            
            assert execute_response.status_code == 200
            execute_data = execute_response.json()
            assert execute_data["success"] is True
            assert "bridge_tx_hash" in execute_data
            assert execute_data["bridge_tx_hash"] == "0xabcdef123456789"
            
            # Verify execution started
            strategy = orchestrator.get_strategy(strategy_id)
            assert strategy.status == StrategyStatus.WAITING_CONFIRMATION
            
            # Step 5: Get Strategy Status
            status_response = client.get(f"/api/v1/orchestrator/strategy/{strategy_id}")
            
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["success"] is True
            assert status_data["strategy"]["status"] == "waiting_confirmation"
            assert status_data["strategy"]["progress_percentage"] == 60

    def test_ai_generated_strategy_complete_workflow(self):
        """Test complete AI-generated strategy workflow"""
        
        # Mock AI response
        mock_ai_result = {
            "strategy_id": "ai-strategy-123",
            "ai_analysis": {
                "strategy": {
                    "source_chain": "ethereum",
                    "target_chain": "hyperliquid",
                    "source_token": "ETH",
                    "target_token": "USDC",
                    "amount": 1.5
                },
                "risk_assessment": {
                    "overall_risk": "medium",
                    "risk_factors": ["bridge_risk", "slippage_risk"]
                },
                "estimated_costs": {
                    "total_usd": 40.0
                }
            },
            "explanation": {
                "summary": "Bridge 1.5 ETH to Hyperliquid with medium risk",
                "key_points": ["Optimal route selection", "Low slippage"]
            }
        }
        
        with patch('app.services.ai_strategy_generator.AIStrategyGenerator.generate_strategy_from_text') as mock_ai, \
             patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            
            mock_ai.return_value = mock_ai_result
            
            orchestrator = CrossChainOrchestrator()
            
            # Mock the strategy created by AI
            mock_strategy = Mock()
            mock_strategy.id = "ai-strategy-123"
            mock_strategy.to_dict.return_value = {
                "id": "ai-strategy-123",
                "status": "pending",
                "user_address": "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
            }
            orchestrator.active_strategies["ai-strategy-123"] = mock_strategy
            
            # Mock route analysis and execution
            orchestrator.analyze_routes = AsyncMock(return_value=[
                {"provider": "gluex", "estimated_fee_usd": 12.0, "estimated_time_minutes": 5}
            ])
            orchestrator.select_optimal_route = AsyncMock(return_value={
                "provider": "gluex", "estimated_fee_usd": 12.0
            })
            orchestrator.execute_strategy = AsyncMock(return_value={
                "success": True,
                "strategy_id": "ai-strategy-123",
                "bridge_tx_hash": "0x987654321",
                "status": "waiting_confirmation"
            })
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Step 1: Generate AI Strategy
            ai_request = {
                "user_input": "Transfer 1.5 ETH to Hyperliquid for trading",
                "portfolio_context": {
                    "balances": {
                        "ethereum": {"ETH": 3.0, "USDC": 2000}
                    },
                    "total_value_usd": 8000
                }
            }
            
            ai_response = client.post(
                "/api/v1/orchestrator/ai-generate-strategy",
                json=ai_request
            )
            
            assert ai_response.status_code == 200
            ai_data = ai_response.json()
            assert ai_data["success"] is True
            assert ai_data["strategy_id"] == "ai-strategy-123"
            assert "ai_analysis" in ai_data
            assert "explanation" in ai_data
            
            strategy_id = ai_data["strategy_id"]
            
            # Verify AI was called with correct parameters
            mock_ai.assert_called_once_with(
                "Transfer 1.5 ETH to Hyperliquid for trading",
                "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A",
                ai_request["portfolio_context"]
            )
            
            # Step 2: Analyze Routes (AI strategy continues with same workflow)
            analyze_response = client.post(f"/api/v1/orchestrator/analyze-routes/{strategy_id}")
            assert analyze_response.status_code == 200
            
            # Step 3: Select Route
            select_response = client.post(
                f"/api/v1/orchestrator/select-route/{strategy_id}",
                json={"route_index": 0}
            )
            assert select_response.status_code == 200
            
            # Step 4: Execute Strategy
            execute_response = client.post(f"/api/v1/orchestrator/execute-strategy/{strategy_id}")
            assert execute_response.status_code == 200
            
            execute_data = execute_response.json()
            assert execute_data["success"] is True
            assert execute_data["bridge_tx_hash"] == "0x987654321"

    def test_strategy_failure_handling(self):
        """Test handling of strategy failures at different stages"""
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            
            # Scenario 1: Route analysis failure (no routes available)
            orchestrator._get_lifi_quote = AsyncMock(return_value=None)
            orchestrator._get_gluex_quote = AsyncMock(return_value=None)
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Create strategy
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "unsupported_chain",  # Cause failure
                "source_token": "ETH",
                "target_token": "UNKNOWN_TOKEN",
                "amount": 1.0
            }
            
            create_response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            
            assert create_response.status_code == 200
            strategy_id = create_response.json()["strategy_id"]
            
            # Try to analyze routes - should fail
            analyze_response = client.post(f"/api/v1/orchestrator/analyze-routes/{strategy_id}")
            
            assert analyze_response.status_code == 200
            analyze_data = analyze_response.json()
            assert analyze_data["success"] is True
            assert len(analyze_data["routes"]) == 0  # No routes available
            
            # Verify strategy marked as failed
            strategy = orchestrator.get_strategy(strategy_id)
            assert strategy.status == StrategyStatus.FAILED
            assert "No routes available" in strategy.error_message

    def test_strategy_cancellation_workflow(self):
        """Test strategy cancellation at different stages"""
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            mock_get_orchestrator.return_value = orchestrator
            
            # Create strategy
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH",
                "target_token": "USDC",
                "amount": 1.0
            }
            
            create_response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            
            strategy_id = create_response.json()["strategy_id"]
            
            # Verify strategy is cancellable in pending state
            cancel_response = client.post(f"/api/v1/orchestrator/cancel-strategy/{strategy_id}")
            
            assert cancel_response.status_code == 200
            cancel_data = cancel_response.json()
            assert cancel_data["success"] is True
            assert cancel_data["message"] == "Strategy cancelled successfully"
            
            # Verify strategy was cancelled
            strategy = orchestrator.get_strategy(strategy_id)
            assert strategy.status == StrategyStatus.CANCELLED

    def test_multiple_user_strategies(self):
        """Test handling multiple strategies for different users"""
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            mock_get_orchestrator.return_value = orchestrator
            
            # Create strategies for first user (current user)
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH",
                "target_token": "USDC",
                "amount": 1.0
            }
            
            # Create 2 strategies for user 1
            create1 = client.post("/api/v1/orchestrator/create-strategy", json=strategy_data)
            create2 = client.post("/api/v1/orchestrator/create-strategy", json=strategy_data)
            
            assert create1.status_code == 200
            assert create2.status_code == 200
            
            # List user strategies
            list_response = client.get("/api/v1/orchestrator/user-strategies")
            
            assert list_response.status_code == 200
            list_data = list_response.json()
            assert list_data["success"] is True
            assert len(list_data["strategies"]) == 2
            
            # Verify strategies belong to correct user
            for strategy in list_data["strategies"]:
                assert strategy["user_address"] == "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"

    def test_orchestrator_statistics(self):
        """Test orchestrator performance statistics"""
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            
            # Create mock strategies with different statuses
            from app.services.cross_chain_orchestrator import CrossChainStrategy
            
            strategy1 = CrossChainStrategy("0x123", {"amount": 1.0})
            strategy1.status = StrategyStatus.COMPLETED
            strategy1.selected_route = {"provider": "lifi"}
            strategy1.total_fees_usd = 15.0
            
            strategy2 = CrossChainStrategy("0x456", {"amount": 2.0})
            strategy2.status = StrategyStatus.FAILED
            strategy2.total_fees_usd = 0.0
            
            strategy3 = CrossChainStrategy("0x789", {"amount": 0.5})
            strategy3.status = StrategyStatus.PENDING
            strategy3.selected_route = {"provider": "gluex"}
            strategy3.total_fees_usd = 10.0
            
            orchestrator.active_strategies = {
                "1": strategy1,
                "2": strategy2,
                "3": strategy3
            }
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Get statistics
            stats_response = client.get("/api/v1/orchestrator/statistics")
            
            assert stats_response.status_code == 200
            stats_data = stats_response.json()
            
            assert stats_data["success"] is True
            stats = stats_data["statistics"]
            
            assert stats["total_strategies"] == 3
            assert stats["completed"] == 1
            assert stats["failed"] == 1
            assert stats["success_rate"] == 33.33333333333333  # 1/3 * 100
            assert stats["provider_usage"]["lifi"] == 1
            assert stats["provider_usage"]["gluex"] == 1
            assert stats["average_fee_usd"] == 8.333333333333334  # (15+0+10)/3

    def test_strategy_real_time_updates(self):
        """Test real-time strategy updates (simulated)"""
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            
            # Mock WebSocket manager
            mock_ws_manager = Mock()
            orchestrator.websocket_manager = mock_ws_manager
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Create strategy
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH",
                "target_token": "USDC",
                "amount": 1.0
            }
            
            create_response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            
            # Verify WebSocket event was emitted for strategy creation
            mock_ws_manager.emit_orchestrator_event.assert_called()
            
            call_args = mock_ws_manager.emit_orchestrator_event.call_args[1]
            assert call_args["event_type"] == "strategy_created"
            assert call_args["user_address"] == "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
            assert call_args["status"] == "pending"

    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution(self):
        """Test handling multiple concurrent strategies"""
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            
            # Mock async methods
            orchestrator._get_lifi_quote = AsyncMock(return_value={"fee_usd": 15.0, "time_minutes": 10})
            orchestrator._get_gluex_quote = AsyncMock(return_value={"fee_usd": 12.0, "time_minutes": 5})
            orchestrator._execute_bridge_transaction = AsyncMock(return_value={
                "success": True,
                "tx_hash": "0xabc123",
                "provider": "gluex"
            })
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Create multiple strategies concurrently
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH",
                "target_token": "USDC",
                "amount": 1.0
            }
            
            # Simulate concurrent requests
            responses = []
            for _ in range(3):
                response = client.post(
                    "/api/v1/orchestrator/create-strategy",
                    json=strategy_data
                )
                responses.append(response)
            
            # Verify all strategies created successfully
            strategy_ids = []
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                strategy_ids.append(data["strategy_id"])
            
            # Verify all strategies are unique
            assert len(set(strategy_ids)) == 3
            
            # Verify all strategies exist in orchestrator
            for strategy_id in strategy_ids:
                strategy = orchestrator.get_strategy(strategy_id)
                assert strategy is not None
                assert strategy.status == StrategyStatus.PENDING

    def test_error_recovery_and_retry(self):
        """Test error recovery and retry mechanisms"""
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            
            # First attempt fails, second succeeds
            call_count = 0
            def mock_bridge_execution(*args):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {"success": False, "error": "Network timeout"}
                else:
                    return {"success": True, "tx_hash": "0xretry123", "provider": "lifi"}
            
            orchestrator._execute_bridge_transaction = AsyncMock(side_effect=mock_bridge_execution)
            orchestrator._get_lifi_quote = AsyncMock(return_value={"fee_usd": 15.0, "time_minutes": 10})
            orchestrator._get_gluex_quote = AsyncMock(return_value={"fee_usd": 12.0, "time_minutes": 5})
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Create and setup strategy
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH",
                "target_token": "USDC",
                "amount": 1.0
            }
            
            create_response = client.post(
                "/api/v1/orchestrator/create-strategy",
                json=strategy_data
            )
            strategy_id = create_response.json()["strategy_id"]
            
            # Analyze routes
            client.post(f"/api/v1/orchestrator/analyze-routes/{strategy_id}")
            
            # Select route
            client.post(
                f"/api/v1/orchestrator/select-route/{strategy_id}",
                json={"route_index": 0}
            )
            
            # First execution attempt should fail
            execute_response = client.post(f"/api/v1/orchestrator/execute-strategy/{strategy_id}")
            
            assert execute_response.status_code == 500
            execute_data = execute_response.json()
            assert execute_data["success"] is False
            assert "Network timeout" in execute_data["error"]

    def test_webhook_integration_simulation(self):
        """Test simulation of webhook/callback integration for external services"""
        
        webhook_events = []
        
        def mock_webhook_handler(event_type, data):
            webhook_events.append({"type": event_type, "data": data})
        
        with patch('app.services.cross_chain_orchestrator.get_cross_chain_orchestrator') as mock_get_orchestrator:
            orchestrator = CrossChainOrchestrator()
            
            # Mock bridge completion callback
            async def mock_bridge_monitoring(strategy_id, bridge_result):
                # Simulate bridge completion
                strategy = orchestrator.get_strategy(strategy_id)
                if strategy:
                    strategy.status = StrategyStatus.BRIDGE_COMPLETED
                    mock_webhook_handler("bridge_completed", {
                        "strategy_id": strategy_id,
                        "tx_hash": bridge_result.get("tx_hash")
                    })
                    
                    strategy.status = StrategyStatus.COMPLETED
                    strategy.actual_completion = asyncio.get_event_loop().time()
                    mock_webhook_handler("strategy_completed", {
                        "strategy_id": strategy_id,
                        "total_time": "5 minutes"
                    })
            
            orchestrator._monitor_bridge_completion = mock_bridge_monitoring
            orchestrator._get_lifi_quote = AsyncMock(return_value={"fee_usd": 15.0, "time_minutes": 10})
            orchestrator._get_gluex_quote = AsyncMock(return_value={"fee_usd": 12.0, "time_minutes": 5})
            orchestrator._execute_bridge_transaction = AsyncMock(return_value={
                "success": True,
                "tx_hash": "0xwebhook123",
                "provider": "lifi"
            })
            
            mock_get_orchestrator.return_value = orchestrator
            
            # Execute complete workflow
            strategy_data = {
                "source_chain": "ethereum",
                "target_chain": "hyperliquid",
                "source_token": "ETH", 
                "target_token": "USDC",
                "amount": 1.0
            }
            
            create_response = client.post("/api/v1/orchestrator/create-strategy", json=strategy_data)
            strategy_id = create_response.json()["strategy_id"]
            
            client.post(f"/api/v1/orchestrator/analyze-routes/{strategy_id}")
            client.post(f"/api/v1/orchestrator/select-route/{strategy_id}", json={"route_index": 0})
            client.post(f"/api/v1/orchestrator/execute-strategy/{strategy_id}")
            
            # Verify webhook events were triggered
            assert len(webhook_events) == 2
            assert webhook_events[0]["type"] == "bridge_completed"
            assert webhook_events[1]["type"] == "strategy_completed"
            assert webhook_events[0]["data"]["tx_hash"] == "0xwebhook123"


if __name__ == "__main__":
    pytest.main([__file__])