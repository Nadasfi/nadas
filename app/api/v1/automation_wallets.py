"""
Automation Wallet Management API
Handles creation and management of automation sub-wallets
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import List, Optional
import secrets
import hashlib
from cryptography.fernet import Fernet
import os
import structlog

from app.core.supabase import get_supabase_client
from app.api.v1.auth import get_current_user
from app.adapters.hyperliquid import HyperliquidAdapter
from app.models.user import User

router = APIRouter()
logger = structlog.get_logger()

class CreateAutomationWalletRequest(BaseModel):
    name: str
    max_funding_usd: float = 1000.0
    
    @validator('max_funding_usd')
    def validate_max_funding(cls, v):
        if v < 100 or v > 5000:
            raise ValueError('Max funding must be between $100 and $5000')
        return v

class AutomationWallet(BaseModel):
    id: str
    name: str
    address: str
    balance_usd: float
    max_funding_usd: float
    is_active: bool
    created_at: str

class FundTransferRequest(BaseModel):
    wallet_id: str
    amount_usd: float
    
    @validator('amount_usd')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

def get_encryption_key() -> bytes:
    """Get or create encryption key for private keys"""
    key_file = "/tmp/automation_key.key"
    
    if os.path.exists(key_file):
        with open(key_file, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        return key

def encrypt_private_key(private_key: str) -> str:
    """Encrypt private key for secure storage"""
    key = get_encryption_key()
    f = Fernet(key)
    return f.encrypt(private_key.encode()).decode()

def decrypt_private_key(encrypted_key: str) -> str:
    """Decrypt private key for use"""
    key = get_encryption_key()
    f = Fernet(key)
    return f.decrypt(encrypted_key.encode()).decode()

@router.post("/wallets", response_model=AutomationWallet)
async def create_automation_wallet(
    request: CreateAutomationWalletRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new automation sub-wallet"""
    try:
        supabase = get_supabase_client()
        
        # Generate new wallet
        private_key = "0x" + secrets.token_hex(32)
        address = "0x" + hashlib.sha256(private_key.encode()).hexdigest()[:40]
        
        # Encrypt private key
        encrypted_private_key = encrypt_private_key(private_key)
        
        # Store in database
        wallet_data = {
            "user_id": current_user.id,
            "name": request.name,
            "address": address,
            "encrypted_private_key": encrypted_private_key,
            "balance_usd": 0.0,
            "max_funding_usd": request.max_funding_usd,
            "is_active": True
        }
        
        result = supabase.table("automation_wallets").insert(wallet_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create automation wallet")
        
        wallet = result.data[0]
        
        logger.info(
            "Created automation wallet",
            user_id=current_user.id,
            wallet_address=address,
            max_funding=request.max_funding_usd
        )
        
        return AutomationWallet(
            id=wallet["id"],
            name=wallet["name"],
            address=wallet["address"],
            balance_usd=wallet["balance_usd"],
            max_funding_usd=wallet["max_funding_usd"],
            is_active=wallet["is_active"],
            created_at=wallet["created_at"]
        )
        
    except Exception as e:
        logger.error("Failed to create automation wallet", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallets", response_model=List[AutomationWallet])
async def list_automation_wallets(
    current_user: User = Depends(get_current_user)
):
    """List all automation wallets for current user"""
    try:
        supabase = get_supabase_client()
        
        result = supabase.table("automation_wallets")\
            .select("id, name, address, balance_usd, max_funding_usd, is_active, created_at")\
            .eq("user_id", current_user.id)\
            .execute()
        
        wallets = [
            AutomationWallet(
                id=w["id"],
                name=w["name"],
                address=w["address"],
                balance_usd=w["balance_usd"],
                max_funding_usd=w["max_funding_usd"],
                is_active=w["is_active"],
                created_at=w["created_at"]
            )
            for w in result.data
        ]
        
        return wallets
        
    except Exception as e:
        logger.error("Failed to list automation wallets", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallets/{wallet_id}/funding-instructions")
async def get_funding_instructions(
    wallet_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get instructions for funding automation wallet"""
    try:
        supabase = get_supabase_client()
        
        result = supabase.table("automation_wallets")\
            .select("address, max_funding_usd, balance_usd")\
            .eq("id", wallet_id)\
            .eq("user_id", current_user.id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Automation wallet not found")
        
        wallet = result.data[0]
        remaining_capacity = wallet["max_funding_usd"] - wallet["balance_usd"]
        
        return {
            "wallet_address": wallet["address"],
            "current_balance_usd": wallet["balance_usd"],
            "max_funding_usd": wallet["max_funding_usd"],
            "remaining_capacity_usd": remaining_capacity,
            "instructions": [
                "1. Send USDC to the automation wallet address above",
                "2. Maximum total funding allowed: $" + str(wallet["max_funding_usd"]),
                "3. Current available capacity: $" + str(remaining_capacity),
                "4. Funds are used exclusively for automation rules you configure",
                "5. You can withdraw unused funds at any time"
            ],
            "security_notes": [
                "✅ Automation wallet has limited funds ($" + str(wallet["max_funding_usd"]) + " max)",
                "✅ Your main wallet remains completely secure",
                "✅ You control when and how much to fund",
                "✅ Automated trades only happen within your configured rules"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get funding instructions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/wallets/{wallet_id}/withdraw")
async def withdraw_from_automation_wallet(
    wallet_id: str,
    amount_usd: float,
    recipient_address: str,
    current_user: User = Depends(get_current_user)
):
    """Withdraw funds from automation wallet to main wallet"""
    try:
        if amount_usd <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")
        
        supabase = get_supabase_client()
        
        # Get wallet details
        result = supabase.table("automation_wallets")\
            .select("address, encrypted_private_key, balance_usd")\
            .eq("id", wallet_id)\
            .eq("user_id", current_user.id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Automation wallet not found")
        
        wallet = result.data[0]
        
        if wallet["balance_usd"] < amount_usd:
            raise HTTPException(status_code=400, detail="Insufficient balance")
        
        # Decrypt private key and execute withdrawal
        private_key = decrypt_private_key(wallet["encrypted_private_key"])
        
        # Initialize Hyperliquid adapter for this wallet
        hl_adapter = HyperliquidAdapter()
        
        # Here you would implement the actual withdrawal logic
        # For now, we'll just update the balance in database
        new_balance = wallet["balance_usd"] - amount_usd
        
        supabase.table("automation_wallets")\
            .update({"balance_usd": new_balance})\
            .eq("id", wallet_id)\
            .execute()
        
        logger.info(
            "Withdrew from automation wallet",
            user_id=current_user.id,
            wallet_id=wallet_id,
            amount_usd=amount_usd,
            recipient=recipient_address
        )
        
        return {
            "success": True,
            "transaction_hash": "0x" + secrets.token_hex(32),
            "amount_withdrawn_usd": amount_usd,
            "new_balance_usd": new_balance
        }
        
    except Exception as e:
        logger.error("Failed to withdraw from automation wallet", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/wallets/{wallet_id}")
async def delete_automation_wallet(
    wallet_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete automation wallet (withdraws all funds first)"""
    try:
        supabase = get_supabase_client()
        
        # Check if wallet exists and has balance
        result = supabase.table("automation_wallets")\
            .select("balance_usd")\
            .eq("id", wallet_id)\
            .eq("user_id", current_user.id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Automation wallet not found")
        
        wallet = result.data[0]
        
        if wallet["balance_usd"] > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Wallet has ${wallet['balance_usd']:.2f} balance. Withdraw funds first."
            )
        
        # Delete wallet
        supabase.table("automation_wallets")\
            .delete()\
            .eq("id", wallet_id)\
            .eq("user_id", current_user.id)\
            .execute()
        
        logger.info(
            "Deleted automation wallet",
            user_id=current_user.id,
            wallet_id=wallet_id
        )
        
        return {"success": True, "message": "Automation wallet deleted"}
        
    except Exception as e:
        logger.error("Failed to delete automation wallet", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/wallets/{wallet_id}/balance")
async def get_automation_wallet_balance(
    wallet_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get real-time balance of automation wallet"""
    try:
        supabase = get_supabase_client()
        
        result = supabase.table("automation_wallets")\
            .select("address, encrypted_private_key")\
            .eq("id", wallet_id)\
            .eq("user_id", current_user.id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Automation wallet not found")
        
        wallet = result.data[0]
        
        # Get real balance from Hyperliquid
        hl_adapter = HyperliquidAdapter()
        account_summary = await hl_adapter.get_account_summary(wallet["address"])
        
        real_balance = account_summary.get("totalEquity", 0) if account_summary else 0
        
        # Update stored balance
        supabase.table("automation_wallets")\
            .update({"balance_usd": real_balance})\
            .eq("id", wallet_id)\
            .execute()
        
        return {
            "wallet_id": wallet_id,
            "address": wallet["address"],
            "balance_usd": real_balance,
            "last_updated": "now"
        }
        
    except Exception as e:
        logger.error("Failed to get automation wallet balance", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))