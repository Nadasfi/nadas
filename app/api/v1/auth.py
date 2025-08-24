"""
Authentication API endpoints
Privy integration and JWT token management
"""

from datetime import timedelta, datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import json

from app.core.config import settings
from app.core.security import create_access_token, create_refresh_token, verify_token
from app.core.logging import get_logger
from app.core.database import get_sync_db
from app.models.user import User
from sqlalchemy.orm import Session

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()

PRIVY_API_BASE_URL = "https://auth.privy.io"


async def verify_privy_token(privy_access_token: str) -> Dict[str, Any]:
    """Verify Privy access token with Privy API"""
    try:
        async with httpx.AsyncClient() as client:
            # Privy token verification endpoint
            response = await client.get(
                f"{PRIVY_API_BASE_URL}/api/v1/users/me",
                headers={
                    "Authorization": f"Bearer {privy_access_token}",
                    "privy-app-id": settings.PRIVY_APP_ID
                }
            )
            
            if response.status_code != 200:
                logger.warning("Privy token verification failed", 
                             status_code=response.status_code,
                             response=response.text)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Privy access token"
                )
            
            user_data = response.json()
            logger.info("Privy token verified successfully", 
                       privy_user_id=user_data.get("id"))
            
            return user_data
            
    except httpx.HTTPError as e:
        logger.error("HTTP error during Privy verification", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable"
        )
    except Exception as e:
        logger.error("Unexpected error during Privy verification", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


def validate_wallet_address(privy_user: Dict[str, Any], wallet_address: str) -> bool:
    """Validate that wallet address matches one of user's linked wallets"""
    try:
        linked_accounts = privy_user.get("linked_accounts", [])
        
        # Check if wallet address matches any linked wallet
        for account in linked_accounts:
            if account.get("type") == "wallet":
                account_address = account.get("address", "").lower()
                if account_address == wallet_address.lower():
                    return True
        
        logger.warning("Wallet address not found in linked accounts", 
                      wallet_address=wallet_address,
                      linked_accounts_count=len(linked_accounts))
        return False
        
    except Exception as e:
        logger.error("Error validating wallet address", error=str(e))
        return False


async def store_user_info(privy_user: Dict[str, Any], wallet_address: str) -> User:
    """Store or update user information in database"""
    try:
        db = next(get_sync_db())
        
        # Check if user already exists
        existing_user = db.query(User).filter(
            User.wallet_address == wallet_address.lower()
        ).first()
        
        if existing_user:
            # Update existing user
            existing_user.privy_user_id = privy_user.get("id")
            existing_user.email = privy_user.get("email")
            existing_user.last_login = datetime.utcnow()
            
            user = existing_user
        else:
            # Create new user
            user = User(
                wallet_address=wallet_address.lower(),
                privy_user_id=privy_user.get("id"),
                email=privy_user.get("email"),
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            db.add(user)
        
        db.commit()
        db.refresh(user)
        db.close()
        
        logger.info("User information stored/updated", 
                   wallet_address=wallet_address,
                   privy_user_id=privy_user.get("id"))
        
        return user
        
    except Exception as e:
        logger.error("Error storing user information", error=str(e))
        if 'db' in locals():
            db.rollback()
            db.close()
        raise


class LoginRequest(BaseModel):
    """Login request with Privy access token"""
    privy_access_token: str
    wallet_address: str


class LoginResponse(BaseModel):
    """Login response with JWT tokens"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class UserInfo(BaseModel):
    """User information response"""
    wallet_address: str
    email: Optional[str] = None
    created_at: str
    last_login: Optional[str] = None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    
    # Demo mode support
    if token == "demo-access-token":
        return {
            "wallet_address": "0x1234567890123456789012345678901234567890",
            "email": "demo@nadas.fi",
        }
    
    payload = verify_token(token, "access")
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login with Privy access token"""
    try:
        # Verify Privy access token
        privy_user = await verify_privy_token(request.privy_access_token)
        
        # Validate wallet address matches Privy user
        if not validate_wallet_address(privy_user, request.wallet_address):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Wallet address mismatch"
            )
        
        # Store/update user in database
        await store_user_info(privy_user, request.wallet_address)
        
        # Create JWT tokens
        token_data = {
            "sub": request.wallet_address,
            "wallet_address": request.wallet_address,
            "privy_user_id": privy_user.get("id")
        }
        
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        logger.info("User logged in", 
                   wallet_address=request.wallet_address,
                   privy_user_id=privy_user.get("id"))
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(request: RefreshRequest):
    """Refresh access token using refresh token"""
    try:
        payload = verify_token(request.refresh_token, "refresh")
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Create new tokens
        token_data = {
            "sub": payload["sub"],
            "wallet_address": payload["wallet_address"]
        }
        
        access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token(token_data)
        
        logger.info("Token refreshed", wallet_address=payload["wallet_address"])
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.get("/me", response_model=UserInfo)
async def get_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return UserInfo(
        wallet_address=current_user["wallet_address"],
        email=None,  # TODO: Get from database
        created_at="2024-01-01T00:00:00Z",  # TODO: Get from database
        last_login=None  # TODO: Get from database
    )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user (client should discard tokens)"""
    logger.info("User logged out", wallet_address=current_user["wallet_address"])
    return {"message": "Logged out successfully"}
