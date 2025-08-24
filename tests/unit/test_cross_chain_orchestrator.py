"""
Unit tests for Cross-Chain Orchestrator
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from app.services.cross_chain_orchestrator import (
    CrossChainOrchestrator, 
    CrossChainStrategy, 
    StrategyStatus,
    BridgeProvider
)


class TestCrossChainStrategy:
    """Test cases for CrossChainStrategy class"""

    def test_strategy_creation(self):
        """Test strategy creation with valid config"""
        user_address = "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
        config = {
            "source_chain": "ethereum",
            "target_chain": "hyperliquid",
            "source_token": "ETH",
            "target_token": "USDC",
            "amount": 1.5
        }
        
        strategy = CrossChainStrategy(user_address, config)
        
        assert strategy.user_address == user_address
        assert strategy.strategy_config == config
        assert strategy.status == StrategyStatus.PENDING
        assert isinstance(strategy.id, str)
        assert len(strategy.id) == 36  # UUID4 length
        assert strategy.execution_log == []
        assert strategy.route_quotes == []
        assert strategy.selected_route is None
        assert strategy.total_fees_usd == 0.0

    def test_add_log_entry(self):
        """Test adding log entries to strategy"""
        strategy = CrossChainStrategy("0x123", {})
        
        strategy.add_log_entry("Test message", "info", {"key": "value"})
        
        assert len(strategy.execution_log) == 1
        log_entry = strategy.execution_log[0]
        assert log_entry["message"] == "Test message"
        assert log_entry["level"] == "info"
        assert log_entry["data"] == {"key": "value"}
        assert "timestamp" in log_entry

    def test_calculate_progress(self):
        """Test progress calculation for different statuses"""
        strategy = CrossChainStrategy("0x123", {})
        
        # Test different status progress values
        test_cases = [
            (StrategyStatus.PENDING, 0),
            (StrategyStatus.ANALYZING, 10),
            (StrategyStatus.QUOTE_READY, 20),
            (StrategyStatus.EXECUTING_BRIDGE, 40),
            (StrategyStatus.WAITING_CONFIRMATION, 60),
            (StrategyStatus.BRIDGE_COMPLETED, 80),
            (StrategyStatus.EXECUTING_TARGET, 90),
            (StrategyStatus.COMPLETED, 100),
            (StrategyStatus.FAILED, 0),
            (StrategyStatus.CANCELLED, 0)
        ]
        
        for status, expected_progress in test_cases:
            strategy.status = status
            assert strategy._calculate_progress() == expected_progress

    def test_to_dict(self):
        """Test strategy serialization to dictionary"""
        config = {
            "source_chain": "ethereum",
            "target_chain": "hyperliquid",
            "amount": 1.0
        }
        strategy = CrossChainStrategy("0x123", config)
        strategy.total_fees_usd = 25.5
        
        result = strategy.to_dict()
        
        assert result["user_address"] == "0x123"
        assert result["status"] == StrategyStatus.PENDING.value
        assert result["strategy_config"] == config
        assert result["total_fees_usd"] == 25.5
        assert result["progress_percentage"] == 0
        assert "created_at" in result
        assert "updated_at" in result


class TestCrossChainOrchestrator:
    """Test cases for CrossChainOrchestrator class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.orchestrator = CrossChainOrchestrator()
        
    @patch('app.services.cross_chain_orchestrator.get_lifi_adapter')
    @patch('app.services.cross_chain_orchestrator.get_gluex_adapter')
    @patch('app.services.cross_chain_orchestrator.get_liquid_labs_adapter')
    @patch('app.services.cross_chain_orchestrator.get_automation_engine')
    @patch('app.services.cross_chain_orchestrator.get_websocket_manager')
    def test_initialize_adapters(self, mock_ws, mock_automation, mock_liquid, mock_gluex, mock_lifi):
        """Test adapter initialization"""
        mock_lifi.return_value = Mock()
        mock_gluex.return_value = Mock()
        mock_liquid.return_value = Mock()
        mock_automation.return_value = Mock()
        mock_ws.return_value = Mock()
        
        orchestrator = CrossChainOrchestrator()
        
        assert orchestrator.lifi_adapter is not None
        assert orchestrator.gluex_adapter is not None
        assert orchestrator.liquid_labs_adapter is not None
        assert orchestrator.automation_engine is not None
        assert orchestrator.websocket_manager is not None

    @pytest.mark.asyncio
    async def test_create_strategy_success(self):
        """Test successful strategy creation"""
        user_address = "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
        config = {
            "source_chain": "ethereum",
            "target_chain": "hyperliquid",
            "source_token": "ETH",
            "target_token": "USDC",
            "amount": 1.0
        }
        
        with patch.object(self.orchestrator, '_emit_strategy_event') as mock_emit:
            strategy = await self.orchestrator.create_strategy(user_address, config)
            
            assert strategy.user_address == user_address
            assert strategy.strategy_config == config
            assert strategy.status == StrategyStatus.PENDING
            assert strategy.id in self.orchestrator.active_strategies
            
            # Verify event emission
            mock_emit.assert_called_once()
            args = mock_emit.call_args
            assert args[0][1] == "strategy_created"

    @pytest.mark.asyncio
    async def test_analyze_routes_success(self):
        """Test successful route analysis"""
        # Create strategy first
        strategy = await self.orchestrator.create_strategy(
            "0x123", 
            {"source_chain": "ethereum", "target_chain": "hyperliquid", "amount": 1000}
        )
        
        # Mock adapter responses
        mock_lifi_quote = {
            "fee_usd": 15.50,
            "time_minutes": 10,
            "route_data": {"path": "ethereum->hyperliquid"}
        }
        mock_gluex_quote = {
            "fee_usd": 12.25,
            "time_minutes": 5,
            "route_data": {"path": "direct"}
        }
        
        with patch.object(self.orchestrator, '_get_lifi_quote', return_value=mock_lifi_quote), \
             patch.object(self.orchestrator, '_get_gluex_quote', return_value=mock_gluex_quote), \
             patch.object(self.orchestrator, '_emit_strategy_event') as mock_emit:
            
            quotes = await self.orchestrator.analyze_routes(strategy.id)
            
            assert len(quotes) == 2
            assert strategy.status == StrategyStatus.QUOTE_READY
            assert len(strategy.route_quotes) == 2
            
            # Verify quotes are sorted by best combination
            best_quote = quotes[0]
            assert best_quote["provider"] == BridgeProvider.GLUEX.value
            assert best_quote["estimated_fee_usd"] == 12.25
            
            # Verify event emission
            mock_emit.assert_called()

    @pytest.mark.asyncio
    async def test_analyze_routes_no_quotes(self):
        """Test route analysis when no quotes are available"""
        strategy = await self.orchestrator.create_strategy(
            "0x123", 
            {"source_chain": "ethereum", "target_chain": "hyperliquid", "amount": 1000}
        )
        
        with patch.object(self.orchestrator, '_get_lifi_quote', return_value=None), \
             patch.object(self.orchestrator, '_get_gluex_quote', return_value=None), \
             patch.object(self.orchestrator, '_emit_strategy_event') as mock_emit:
            
            quotes = await self.orchestrator.analyze_routes(strategy.id)
            
            assert len(quotes) == 0
            assert strategy.status == StrategyStatus.FAILED
            assert strategy.error_message == "No routes available from any provider"
            
            # Verify failure event emission
            mock_emit.assert_called()
            args = mock_emit.call_args
            assert args[0][1] == "routes_analysis_failed"

    @pytest.mark.asyncio
    async def test_select_optimal_route(self):
        """Test route selection"""
        strategy = await self.orchestrator.create_strategy("0x123", {})
        strategy.route_quotes = [
            {
                "provider": BridgeProvider.LIFI.value,
                "estimated_fee_usd": 15.0,
                "estimated_time_minutes": 10
            },
            {
                "provider": BridgeProvider.GLUEX.value,
                "estimated_fee_usd": 12.0,
                "estimated_time_minutes": 8
            }
        ]
        
        with patch.object(self.orchestrator, '_emit_strategy_event') as mock_emit:
            selected_route = await self.orchestrator.select_optimal_route(strategy.id, 1)
            
            assert selected_route["provider"] == BridgeProvider.GLUEX.value
            assert strategy.selected_route == selected_route
            assert strategy.total_fees_usd == 12.0
            assert strategy.estimated_completion is not None
            
            # Verify event emission
            mock_emit.assert_called()

    @pytest.mark.asyncio
    async def test_execute_strategy_success(self):
        """Test successful strategy execution"""
        strategy = await self.orchestrator.create_strategy("0x123", {})
        strategy.selected_route = {
            "provider": BridgeProvider.LIFI.value,
            "route": {"fee_usd": 15.0}
        }
        
        mock_bridge_result = {
            "success": True,
            "tx_hash": "0xabcdef123456789",
            "provider": BridgeProvider.LIFI.value
        }
        
        with patch.object(self.orchestrator, '_execute_bridge_transaction', 
                         return_value=mock_bridge_result), \
             patch.object(self.orchestrator, '_emit_strategy_event') as mock_emit, \
             patch('asyncio.create_task') as mock_task:
            
            result = await self.orchestrator.execute_strategy(strategy.id)
            
            assert result["success"] is True
            assert result["strategy_id"] == strategy.id
            assert result["bridge_tx_hash"] == "0xabcdef123456789"
            assert strategy.status == StrategyStatus.WAITING_CONFIRMATION
            
            # Verify background monitoring task created
            mock_task.assert_called_once()
            
            # Verify events emitted
            assert mock_emit.call_count >= 2

    @pytest.mark.asyncio
    async def test_execute_strategy_no_route_selected(self):
        """Test strategy execution without route selection"""
        strategy = await self.orchestrator.create_strategy("0x123", {})
        
        with pytest.raises(ValueError, match="No route selected"):
            await self.orchestrator.execute_strategy(strategy.id)

    @pytest.mark.asyncio
    async def test_lifi_bridge_execution(self):
        """Test LI.FI bridge execution"""
        strategy = CrossChainStrategy("0x123", {
            "source_chain": "ethereum",
            "target_chain": "hyperliquid",
            "amount": 1000
        })
        strategy.selected_route = {
            "provider": BridgeProvider.LIFI.value,
            "route": {"fee_usd": 15.0, "time_minutes": 10}
        }
        
        result = await self.orchestrator._execute_lifi_bridge(strategy)
        
        assert result["success"] is True
        assert "tx_hash" in result
        assert result["provider"] == BridgeProvider.LIFI.value
        assert len(strategy.bridge_transactions) == 1

    @pytest.mark.asyncio
    async def test_gluex_bridge_execution(self):
        """Test GlueX bridge execution"""
        strategy = CrossChainStrategy("0x123", {
            "source_chain": "ethereum", 
            "target_chain": "hyperliquid",
            "amount": 1000
        })
        strategy.selected_route = {
            "provider": BridgeProvider.GLUEX.value,
            "route": {"fee_usd": 12.0, "time_minutes": 5}
        }
        
        result = await self.orchestrator._execute_gluex_bridge(strategy)
        
        assert result["success"] is True
        assert "tx_hash" in result
        assert result["provider"] == BridgeProvider.GLUEX.value
        assert len(strategy.bridge_transactions) == 1

    @pytest.mark.asyncio
    async def test_monitor_bridge_completion(self):
        """Test bridge monitoring completion"""
        strategy = await self.orchestrator.create_strategy("0x123", {
            "automation_rules": {"rules": []}
        })
        bridge_result = {
            "tx_hash": "0xabcdef123456789",
            "estimated_time": 0.1  # Very short for testing
        }
        
        with patch.object(self.orchestrator, '_emit_strategy_event') as mock_emit, \
             patch('asyncio.sleep', return_value=None):  # Skip actual sleep
            
            await self.orchestrator._monitor_bridge_completion(strategy.id, bridge_result)
            
            assert strategy.status == StrategyStatus.COMPLETED
            assert strategy.actual_completion is not None
            
            # Verify completion events emitted
            assert mock_emit.call_count >= 2

    @pytest.mark.asyncio
    async def test_cancel_strategy(self):
        """Test strategy cancellation"""
        strategy = await self.orchestrator.create_strategy("0x123", {})
        
        with patch.object(self.orchestrator, '_emit_strategy_event') as mock_emit:
            result = await self.orchestrator.cancel_strategy(strategy.id)
            
            assert result is True
            assert strategy.status == StrategyStatus.CANCELLED
            
            # Verify cancellation event
            mock_emit.assert_called()

    @pytest.mark.asyncio
    async def test_cancel_strategy_not_cancellable(self):
        """Test cancelling strategy that's not in cancellable state"""
        strategy = await self.orchestrator.create_strategy("0x123", {})
        strategy.status = StrategyStatus.EXECUTING_BRIDGE
        
        result = await self.orchestrator.cancel_strategy(strategy.id)
        
        assert result is False
        assert strategy.status == StrategyStatus.EXECUTING_BRIDGE

    def test_get_strategy(self):
        """Test retrieving strategy by ID"""
        orchestrator = CrossChainOrchestrator()
        strategy = CrossChainStrategy("0x123", {})
        orchestrator.active_strategies[strategy.id] = strategy
        
        retrieved = orchestrator.get_strategy(strategy.id)
        assert retrieved == strategy
        
        # Test non-existent strategy
        non_existent = orchestrator.get_strategy("invalid-id")
        assert non_existent is None

    def test_get_user_strategies(self):
        """Test retrieving strategies for a user"""
        orchestrator = CrossChainOrchestrator()
        user1_strategy1 = CrossChainStrategy("0x123", {})
        user1_strategy2 = CrossChainStrategy("0x123", {})
        user2_strategy = CrossChainStrategy("0x456", {})
        
        orchestrator.active_strategies[user1_strategy1.id] = user1_strategy1
        orchestrator.active_strategies[user1_strategy2.id] = user1_strategy2
        orchestrator.active_strategies[user2_strategy.id] = user2_strategy
        
        user1_strategies = orchestrator.get_user_strategies("0x123")
        assert len(user1_strategies) == 2
        assert user1_strategy1 in user1_strategies
        assert user1_strategy2 in user1_strategies
        
        user2_strategies = orchestrator.get_user_strategies("0x456")
        assert len(user2_strategies) == 1
        assert user2_strategy in user2_strategies

    def test_execution_statistics(self):
        """Test getting orchestrator performance statistics"""
        orchestrator = CrossChainOrchestrator()
        
        # Add test strategies with different statuses
        completed_strategy = CrossChainStrategy("0x123", {})
        completed_strategy.status = StrategyStatus.COMPLETED
        completed_strategy.selected_route = {"provider": BridgeProvider.LIFI.value}
        completed_strategy.total_fees_usd = 15.0
        
        failed_strategy = CrossChainStrategy("0x456", {})
        failed_strategy.status = StrategyStatus.FAILED
        failed_strategy.total_fees_usd = 10.0
        
        pending_strategy = CrossChainStrategy("0x789", {})
        pending_strategy.total_fees_usd = 12.0
        
        orchestrator.active_strategies = {
            "1": completed_strategy,
            "2": failed_strategy, 
            "3": pending_strategy
        }
        
        stats = orchestrator.execution_statistics()
        
        assert stats["total_strategies"] == 3
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["success_rate"] == 33.33333333333333  # 1/3 * 100
        assert stats["provider_usage"][BridgeProvider.LIFI.value] == 1
        assert stats["average_fee_usd"] == 12.333333333333334  # (15+10+12)/3

    def test_emit_strategy_event(self):
        """Test WebSocket event emission"""
        orchestrator = CrossChainOrchestrator()
        orchestrator.websocket_manager = Mock()
        
        strategy = CrossChainStrategy("0x123", {})
        test_data = {"key": "value"}
        
        orchestrator._emit_strategy_event(strategy, "test_event", test_data)
        
        orchestrator.websocket_manager.emit_orchestrator_event.assert_called_once_with(
            strategy_id=strategy.id,
            user_address=strategy.user_address,
            event_type="test_event",
            status=strategy.status.value,
            data=test_data
        )

    def test_emit_strategy_event_no_websocket_manager(self):
        """Test event emission when websocket manager is None"""
        orchestrator = CrossChainOrchestrator()
        orchestrator.websocket_manager = None
        
        strategy = CrossChainStrategy("0x123", {})
        
        # Should not raise exception
        orchestrator._emit_strategy_event(strategy, "test_event", {})


@pytest.mark.asyncio
async def test_integration_full_workflow():
    """Integration test for complete cross-chain workflow"""
    orchestrator = CrossChainOrchestrator()
    user_address = "0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A"
    
    config = {
        "source_chain": "ethereum",
        "target_chain": "hyperliquid", 
        "source_token": "ETH",
        "target_token": "USDC",
        "amount": 1.0
    }
    
    # Mock all external dependencies
    with patch.object(orchestrator, '_get_lifi_quote') as mock_lifi, \
         patch.object(orchestrator, '_get_gluex_quote') as mock_gluex, \
         patch.object(orchestrator, '_emit_strategy_event'), \
         patch('asyncio.create_task'):
        
        # Setup mock quotes
        mock_lifi.return_value = {"fee_usd": 15.0, "time_minutes": 10}
        mock_gluex.return_value = {"fee_usd": 12.0, "time_minutes": 5}
        
        # 1. Create strategy
        strategy = await orchestrator.create_strategy(user_address, config)
        assert strategy.status == StrategyStatus.PENDING
        
        # 2. Analyze routes
        quotes = await orchestrator.analyze_routes(strategy.id)
        assert len(quotes) == 2
        assert strategy.status == StrategyStatus.QUOTE_READY
        
        # 3. Select optimal route
        selected_route = await orchestrator.select_optimal_route(strategy.id, 0)
        assert selected_route["provider"] == BridgeProvider.GLUEX.value
        assert strategy.selected_route is not None
        
        # 4. Execute strategy
        result = await orchestrator.execute_strategy(strategy.id)
        assert result["success"] is True
        assert strategy.status == StrategyStatus.WAITING_CONFIRMATION
        
        # 5. Verify strategy is tracked
        retrieved = orchestrator.get_strategy(strategy.id)
        assert retrieved == strategy
        
        user_strategies = orchestrator.get_user_strategies(user_address)
        assert len(user_strategies) == 1
        assert strategy in user_strategies


if __name__ == "__main__":
    pytest.main([__file__])