"""
Redis Client Configuration for AWS ElastiCache
Production-ready Redis client with connection pooling
"""

import redis
import os
from typing import Optional
import structlog
from app.core.config import settings

logger = structlog.get_logger(__name__)

class RedisClient:
    """Redis client wrapper with production configurations"""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None
        
    def get_client(self) -> redis.Redis:
        """Get Redis client instance with connection pooling"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def _initialize_client(self):
        """Initialize Redis client with proper configuration"""
        try:
            # Determine Redis URL based on environment
            redis_url = self._get_redis_url()
            
            # Create connection pool for better performance
            self._connection_pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=20,
                retry_on_timeout=True,
                decode_responses=True,
                health_check_interval=30,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Create Redis client
            self._client = redis.Redis(
                connection_pool=self._connection_pool
            )
            
            # Test connection
            self._client.ping()
            logger.info("Redis client initialized successfully", redis_url=redis_url)
            
        except Exception as e:
            logger.error("Failed to initialize Redis client", error=str(e))
            # Fallback to in-memory caching (for development)
            self._client = None
            raise
    
    def _get_redis_url(self) -> str:
        """Get Redis URL based on environment"""
        if settings.ENVIRONMENT == "production":
            # Use AWS ElastiCache in production
            redis_url = os.getenv('REDIS_URL_PROD')
            if not redis_url:
                raise ValueError("REDIS_URL_PROD not set for production environment")
        else:
            # Use local Redis in development
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        return redis_url
    
    async def health_check(self) -> dict:
        """Check Redis health status"""
        try:
            client = self.get_client()
            
            # Test basic operation
            client.ping()
            
            # Get server info
            info = client.info()
            
            return {
                "status": "healthy",
                "version": info.get('redis_version'),
                "memory_used": info.get('used_memory_human'),
                "connected_clients": info.get('connected_clients'),
                "uptime_seconds": info.get('uptime_in_seconds')
            }
            
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def close(self):
        """Close Redis connection"""
        if self._connection_pool:
            self._connection_pool.disconnect()
            logger.info("Redis connection pool closed")

# Global Redis client instance
_redis_client = RedisClient()

def get_redis_client() -> redis.Redis:
    """Get the global Redis client instance"""
    return _redis_client.get_client()

def get_redis_health() -> dict:
    """Get Redis health status"""
    return _redis_client.health_check()

# Cache decorators and utilities
class CacheManager:
    """Cache management utilities"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def set_with_ttl(self, key: str, value: str, ttl: int = 3600):
        """Set value with TTL (Time To Live)"""
        try:
            self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Cache flush pattern failed", pattern=pattern, error=str(e))
            return 0

def get_cache_manager() -> CacheManager:
    """Get cache manager instance"""
    return CacheManager(get_redis_client())