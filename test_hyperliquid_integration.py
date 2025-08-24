#!/usr/bin/env python3
"""
Test script for Hyperliquid SDK integration
Run this to verify the implementation works correctly
"""

import asyncio
import os
from datetime import datetime

# Test imports first
try:
    import eth_account
    print("✅ eth_account imported successfully")
except ImportError as e:
    print(f"❌ eth_account import failed: {e}")

try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    print("✅ hyperliquid-python-sdk imported successfully")
except ImportError as e:
    print(f"❌ hyperliquid-python-sdk import failed: {e}")

try:
    import structlog
    print("✅ structlog imported successfully")
except ImportError as e:
    print(f"❌ structlog import failed: {e}")

# Test our adapter import
try:
    from app.adapters.hyperliquid import HyperliquidAdapter, get_hyperliquid_adapter
    print("✅ HyperliquidAdapter imported successfully")
except ImportError as e:
    print(f"❌ HyperliquidAdapter import failed: {e}")


async def test_read_only_operations():
    """Test read-only operations without private key"""
    print("\n🔍 Testing read-only operations...")
    
    try:
        # Initialize adapter in read-only mode
        adapter = HyperliquidAdapter(use_mainnet=False)  # Use testnet
        print("✅ Adapter initialized in read-only mode")
        
        # Test getting all mid prices
        prices = await adapter.get_all_mid_prices()
        print(f"✅ Mid prices fetched: {len(prices)} symbols")
        if prices:
            # Show first few prices
            sample_prices = list(prices.items())[:3]
            for symbol, price in sample_prices:
                print(f"   {symbol}: ${price}")
        
        # Test market data for ETH
        if 'ETH' in prices:
            market_data = await adapter.get_market_data('ETH')
            if market_data:
                print(f"✅ ETH market data: mid=${market_data.mid_price}, bid=${market_data.bid}, ask=${market_data.ask}")
            else:
                print("⚠️ No market data returned for ETH")
        
        # Test with a test address (public)
        test_address = "0x563b377A956c80d77A7c613a9343699Ad6123911"  # Random test address
        
        positions = await adapter.get_user_positions(test_address)
        print(f"✅ User positions fetched: {len(positions)} positions")
        
        spot_balances = await adapter.get_spot_balances(test_address)
        print(f"✅ Spot balances fetched: {len(spot_balances)} balances")
        
        account_value = await adapter.get_account_value(test_address)
        print(f"✅ Account value: ${account_value['account_value']}")
        
        await adapter.close()
        print("✅ Adapter closed successfully")
        
    except Exception as e:
        print(f"❌ Read-only test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_trading_operations():
    """Test trading operations with a private key (if provided)"""
    print("\n💸 Testing trading operations...")
    
    # Check for test private key in environment
    test_private_key = os.getenv('HYPERLIQUID_TEST_PRIVATE_KEY')
    
    if not test_private_key:
        print("⚠️ No test private key provided. Skipping trading tests.")
        print("   Set HYPERLIQUID_TEST_PRIVATE_KEY environment variable to test trading.")
        return
    
    try:
        # Initialize adapter with trading capabilities
        adapter = HyperliquidAdapter(private_key=test_private_key, use_mainnet=False)
        print(f"✅ Adapter initialized with trading wallet: {adapter.wallet.address}")
        
        # Test account info for our wallet
        account_value = await adapter.get_account_value(adapter.wallet.address)
        print(f"✅ Account value: ${account_value['account_value']}")
        
        # Test getting positions
        positions = await adapter.get_user_positions(adapter.wallet.address)
        print(f"✅ Your positions: {len(positions)}")
        for pos in positions[:3]:  # Show first 3
            print(f"   {pos.symbol}: {pos.size} @ ${pos.entry_price} (PnL: ${pos.unrealized_pnl})")
        
        # Test getting open orders
        open_orders = await adapter.get_open_orders(adapter.wallet.address)
        print(f"✅ Open orders: {len(open_orders)}")
        
        # NOTE: Uncomment the following to test actual order placement (be careful!)
        # print("⚠️ Skipping actual order placement for safety")
        # Small test order (uncomment only if you want to actually trade!)
        # result = await adapter.place_order("ETH", True, 0.001, 2000.0, "limit", post_only=True)
        # print(f"✅ Test order result: {result}")
        
        await adapter.close()
        print("✅ Trading adapter closed successfully")
        
    except Exception as e:
        print(f"❌ Trading test failed: {e}")
        import traceback
        traceback.print_exc()


def test_configuration():
    """Test configuration and constants"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from hyperliquid.utils import constants
        print(f"✅ Testnet URL: {constants.TESTNET_API_URL}")
        print(f"✅ Mainnet URL: {constants.MAINNET_API_URL}")
        
        # Test our adapter factory function
        adapter = get_hyperliquid_adapter()
        print(f"✅ Adapter factory works, mainnet: {adapter.use_mainnet}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")


async def main():
    """Run all tests"""
    print("🚀 Testing Hyperliquid SDK Integration")
    print("=" * 50)
    
    # Test configuration
    test_configuration()
    
    # Test read-only operations
    await test_read_only_operations()
    
    # Test trading operations (if private key available)
    await test_trading_operations()
    
    print("\n✅ All tests completed!")
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Set up environment variables for trading tests")
    print("3. Run database migrations for new models")
    print("4. Update API endpoints to use new adapter")


if __name__ == "__main__":
    # Set up basic logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
