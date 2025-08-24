#!/usr/bin/env python3
"""
Redis Connection Test Script
Test AWS ElastiCache Redis connection
"""

import redis
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_redis_connection():
    """Test Redis connection"""
    redis_url = os.getenv('REDIS_URL_PROD') or os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    print(f"🔍 Testing Redis connection to: {redis_url}")
    
    try:
        # Connect to Redis
        r = redis.from_url(redis_url, decode_responses=True)
        
        # Test basic operations
        print("📝 Testing SET operation...")
        r.set('test_key', 'Hello from Nadas.fi!', ex=60)  # Expire in 60 seconds
        
        print("📖 Testing GET operation...")
        value = r.get('test_key')
        
        if value == 'Hello from Nadas.fi!':
            print("✅ Redis connection successful!")
            print(f"   Retrieved value: {value}")
        else:
            print("❌ Redis test failed - unexpected value")
            return False
            
        # Test Redis info
        print("\n📊 Redis Server Info:")
        info = r.info()
        print(f"   Version: {info.get('redis_version')}")
        print(f"   Memory used: {info.get('used_memory_human')}")
        print(f"   Connected clients: {info.get('connected_clients')}")
        print(f"   Uptime: {info.get('uptime_in_seconds')} seconds")
        
        # Clean up test key
        r.delete('test_key')
        print("🧹 Test key cleaned up")
        
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("\n🔧 Possible fixes:")
        print("   1. Check if ElastiCache cluster is running")
        print("   2. Verify security group allows port 6379")
        print("   3. Check if you're in the correct VPC")
        print("   4. Verify the Redis URL is correct")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_redis_connection()
    sys.exit(0 if success else 1)