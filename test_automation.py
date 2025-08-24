#!/usr/bin/env python3
"""
Test Script for Nadas.fi Automation System
Demonstrates end-to-end automation functionality on Hyperliquid testnet
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.automation_engine import AutomationEngine
from app.adapters.hyperliquid import get_hyperliquid_adapter
from app.core.config import settings


class AutomationTester:
    """Test automation functionality end-to-end"""
    
    def __init__(self):
        self.engine = AutomationEngine()
        self.test_results = []
        
    def log_result(self, test_name: str, success: bool, message: str, details: dict = None):
        """Log test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name} - {message}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
        print()
    
    async def test_hyperliquid_connection(self):
        """Test 1: Verify Hyperliquid adapter connectivity"""
        try:
            adapter = get_hyperliquid_adapter()
            
            # Test basic connection
            all_mids = await adapter.get_all_mids()
            
            if all_mids and len(all_mids) > 0:
                self.log_result(
                    "Hyperliquid Connection",
                    True,
                    "Successfully connected to Hyperliquid testnet",
                    {
                        "symbols_available": len(all_mids),
                        "sample_prices": dict(list(all_mids.items())[:3])
                    }
                )
            else:
                self.log_result(
                    "Hyperliquid Connection",
                    False,
                    "No market data received"
                )
            
            await adapter.close()
            return True
            
        except Exception as e:
            self.log_result(
                "Hyperliquid Connection",
                False,
                f"Connection failed: {str(e)}"
            )
            return False
    
    async def test_market_data_retrieval(self):
        """Test 2: Verify market data retrieval for automation"""
        try:
            adapter = get_hyperliquid_adapter()
            
            # Test specific symbol data
            test_symbol = "ETH"
            market_data = await adapter.get_market_data(test_symbol)
            
            if market_data and hasattr(market_data, 'mid_price'):
                self.log_result(
                    "Market Data Retrieval",
                    True,
                    f"Successfully retrieved {test_symbol} market data",
                    {
                        "symbol": test_symbol,
                        "mid_price": market_data.mid_price,
                        "bid": getattr(market_data, 'bid', 'N/A'),
                        "ask": getattr(market_data, 'ask', 'N/A')
                    }
                )
            else:
                self.log_result(
                    "Market Data Retrieval",
                    False,
                    f"No market data for {test_symbol}"
                )
            
            await adapter.close()
            return True
            
        except Exception as e:
            self.log_result(
                "Market Data Retrieval",
                False,
                f"Market data retrieval failed: {str(e)}"
            )
            return False
    
    async def test_automation_engine_setup(self):
        """Test 3: Test automation engine initialization"""
        try:
            # Test engine creation
            engine = AutomationEngine()
            
            # Test DCA rule creation (in-memory)
            rule_id = await engine.create_dca_rule(
                user_id="test-user",
                wallet_address="0x1234567890abcdef",
                symbol="ETH",
                amount_usd=100.0,
                interval_hours=24
            )
            
            # Verify rule was created
            rule = engine.get_rule_status(rule_id)
            
            if rule and rule.rule_id == rule_id:
                self.log_result(
                    "Automation Engine Setup",
                    True,
                    "Successfully created DCA automation rule",
                    {
                        "rule_id": rule_id,
                        "rule_type": rule.rule_type.value,
                        "symbol": rule.config["symbol"],
                        "amount_usd": rule.config["amount_usd"]
                    }
                )
            else:
                self.log_result(
                    "Automation Engine Setup",
                    False,
                    "Failed to create automation rule"
                )
            
            return True
            
        except Exception as e:
            self.log_result(
                "Automation Engine Setup",
                False,
                f"Automation engine setup failed: {str(e)}"
            )
            return False
    
    async def test_private_key_configuration(self):
        """Test 4: Check private key configuration for trading"""
        try:
            has_private_key = bool(settings.HYPERLIQUID_PRIVATE_KEY)
            
            if has_private_key:
                # Don't log the actual key for security
                key_length = len(settings.HYPERLIQUID_PRIVATE_KEY)
                is_valid_format = settings.HYPERLIQUID_PRIVATE_KEY.startswith('0x')
                
                self.log_result(
                    "Private Key Configuration",
                    True,
                    "Private key is configured for trading",
                    {
                        "key_length": key_length,
                        "valid_format": is_valid_format,
                        "trading_enabled": True
                    }
                )
            else:
                self.log_result(
                    "Private Key Configuration",
                    False,
                    "No private key configured - trading disabled",
                    {
                        "trading_enabled": False,
                        "note": "Add HYPERLIQUID_PRIVATE_KEY to .env for live trading"
                    }
                )
            
            return has_private_key
            
        except Exception as e:
            self.log_result(
                "Private Key Configuration",
                False,
                f"Private key check failed: {str(e)}"
            )
            return False
    
    async def test_websocket_data_manager(self):
        """Test 5: Test WebSocket data manager for real-time data"""
        try:
            from app.services.websocket_manager import get_websocket_manager
            
            ws_manager = get_websocket_manager()
            
            # Test subscribing to market data
            ws_manager.subscribe_to_market_data(["ETH", "BTC"])
            
            # Check connection stats
            stats = ws_manager.get_connection_stats()
            
            self.log_result(
                "WebSocket Data Manager",
                True,
                "WebSocket manager is operational",
                {
                    "subscribed_symbols": ["ETH", "BTC"],
                    "connection_stats": stats
                }
            )
            
            return True
            
        except Exception as e:
            self.log_result(
                "WebSocket Data Manager",
                False,
                f"WebSocket manager test failed: {str(e)}"
            )
            return False
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Nadas.fi Automation System Tests")
        print("=" * 60)
        print()
        
        test_methods = [
            self.test_hyperliquid_connection,
            self.test_market_data_retrieval,
            self.test_automation_engine_setup,
            self.test_private_key_configuration,
            self.test_websocket_data_manager
        ]
        
        for test_method in test_methods:
            await test_method()
        
        # Print summary
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print()
        
        if passed == total:
            print("✅ ALL TESTS PASSED - System ready for automation!")
        else:
            print("⚠️  Some tests failed - Check configuration before live trading")
        
        print()
        
        # Next steps
        print("🎯 NEXT STEPS FOR FULL AUTOMATION:")
        print("1. Add your testnet private key to HYPERLIQUID_PRIVATE_KEY in .env")
        print("2. Start Celery workers: celery -A app.workers.automation_tasks worker")
        print("3. Start Celery beat: celery -A app.workers.automation_tasks beat")
        print("4. Create automation rules via API endpoints")
        print("5. Monitor execution via logs and dashboard")
        print()
        
        return passed == total


async def main():
    """Main test runner"""
    try:
        tester = AutomationTester()
        success = await tester.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())