"""
FastAPI Dependencies
Authentication and authorization dependencies
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from structlog import get_logger

from app.core.config import settings

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


async def get_current_user_address(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """Get current user wallet address from JWT token"""
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # TEMPORARY: Demo token support
        if credentials.credentials == "demo-access-token":
            return "0x8dF3e4806A3320D2642b1F2835ADDA1A40719c4E"
            
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        wallet_address = payload.get("wallet_address")
        if not wallet_address:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: no wallet address"
            )
            
        return wallet_address
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_optional_user_address(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """Get current user wallet address, but don't require authentication"""
    
    if not credentials:
        return None
    
    # TEMPORARY: Demo token support
    if credentials.credentials == "demo-access-token":
        return "0x8dF3e4806A3320D2642b1F2835ADDA1A40719c4E"
        
    try:
        return await get_current_user_address(credentials)
    except HTTPException:
        return None


def require_admin():
    """Dependency that requires admin privileges"""
    def _require_admin(user_address: str = Depends(get_current_user_address)) -> str:
        # For demo, any authenticated user is admin
        # In production, check against admin list
        return user_address
    
    return _require_admin