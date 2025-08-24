"""
Configuration management with environment variables
"""

from typing import List, Optional
from pydantic import validator, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "allow"  # Allow extra fields from .env
    }

    # Application
    ENVIRONMENT: str = "development"
    API_VERSION: str = "v1"
    PROJECT_NAME: str = "Nadas.fi"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: str = "*"  # Should be restricted in production
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,https://nadas.fi,https://www.nadas.fi,https://app.nadas.fi"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://nadas:nadas123@localhost:5432/nadas"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600
    
    # JWT Security
    SECRET_KEY: str = "nadas-fi-super-secret-jwt-key-hyperliquid-hackathon-2025"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Privy Authentication
    PRIVY_APP_ID: str = "your-privy-app-id-here"
    PRIVY_APP_SECRET: Optional[str] = None
    PRIVY_VERIFICATION_KEY: Optional[str] = None
    
    # Hyperliquid Configuration - Updated for Python SDK
    HYPERLIQUID_TESTNET_URL: str = "https://api.hyperliquid-testnet.xyz"
    HYPERLIQUID_MAINNET_URL: str = "https://api.hyperliquid.xyz"
    HYPERLIQUID_NETWORK: str = "mainnet"  # "testnet" or "mainnet"
    # HYPERLIQUID_PRIVATE_KEY removed - using client-side wallet for security
    HYPERLIQUID_USE_WEBSOCKET: bool = True  # Enable real-time data
    HYPERLIQUID_MAX_POSITION_SIZE: float = 1000.0  # Max position size in USD
    HYPERLIQUID_DEFAULT_SLIPPAGE: float = 0.01  # Default slippage (1%)
    
    # HyperEVM Configuration (for precompiles and simulation)
    HYPEREVM_TESTNET_RPC: str = "https://api.hyperliquid-testnet.xyz/evm"
    HYPEREVM_MAINNET_RPC: str = "https://api.hyperliquid.xyz/evm"
    HYPEREVM_CHAIN_ID_TESTNET: int = 998
    HYPEREVM_CHAIN_ID_MAINNET: int = 999
    
    # Alchemy Configuration
    ALCHEMY_API_KEY: str = ""
    ALCHEMY_HYPERLIQUID_URL: str = ""
    
    # Supabase Configuration
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    
    # LI.FI Cross-Chain Integration ($7k Bounty)
    LIFI_API_BASE: str = "https://li.quest/v1"
    LIFI_API_KEY: Optional[str] = None  # Optional for higher rate limits
    LIFI_DEFAULT_SLIPPAGE: float = 0.03  # 3% default slippage for cross-chain
    LIFI_ENABLE_CROSS_CHAIN: bool = True

    # GlueX Cross-Chain Integration ($7k Bounty)
    GLUEX_API_KEY: str = ""  # GlueX Router API key
    GLUEX_ROUTER_URL: str = "https://router.gluex.xyz/v1"  # GlueX Router API URL
    GLUEX_EXCHANGE_RATES_URL: str = "https://exchange-rates.gluex.xyz"  # GlueX Exchange Rates API URL
    GLUEX_DEFAULT_SLIPPAGE: float = 0.005  # Default slippage for cross-chain operations (0.5%)
    GLUEX_MAX_PRICE_IMPACT: float = 0.05  # Maximum acceptable price impact (5%)
    GLUEX_ENABLE_CROSS_CHAIN: bool = True  # Enable cross-chain functionality

    # Notification System ($3k Bounty)
    NOTIFICATION_CHECK_INTERVAL_SECONDS: int = 30  # Check notification rules every 30 seconds
    NOTIFICATION_WEBSOCKET_PORT: int = 8001  # WebSocket server port
    NOTIFICATION_MAX_RULES_PER_USER: int = 50  # Maximum notification rules per user
    NOTIFICATION_COOLDOWN_MINUTES: int = 15  # Cooldown between same rule triggers
    NOTIFICATION_RETENTION_DAYS: int = 30  # Keep notifications for 30 days
    EMAIL_SMTP_SERVER: str = ""  # SMTP server for email notifications
    EMAIL_SMTP_PORT: int = 587  # SMTP port
    EMAIL_USERNAME: str = ""  # Email username
    EMAIL_PASSWORD: str = ""  # Email password
    PUSH_NOTIFICATION_API_KEY: str = ""  # Push notification service API key
    
    # AI Service Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-west-2"
    
    # AWS Bedrock Bearer Token (New API Key System)
    AWS_BEARER_TOKEN_BEDROCK: Optional[str] = None
    
    # AI Provider Priority
    AI_PRIMARY_PROVIDER: str = "openai"
    AI_SECONDARY_PROVIDER: str = "anthropic"
    AI_TERTIARY_PROVIDER: str = "bedrock"
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TIMEZONE: str = "UTC"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 200
    
    # Logging
    LOG_LEVEL: str = "INFO"


# Global settings instance
settings = Settings()