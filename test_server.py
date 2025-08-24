#!/usr/bin/env python3
"""
Lightweight test server for authentication testing
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx
import uvicorn

app = FastAPI(title="Nadas Test Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Nadas Test Server is running", "status": "healthy"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "environment": "development",
        "network": "mainnet",
        "timestamp": "2025-08-23T21:45:00Z"
    }

@app.post("/api/v1/auth/login")
async def mock_login(request: dict):
    """Mock authentication for testing"""
    privy_token = request.get('privy_access_token')
    wallet_address = request.get('wallet_address')
    
    if not privy_token or not wallet_address:
        raise HTTPException(status_code=400, detail="Missing token or wallet address")
    
    # Mock successful response
    return {
        "success": True,
        "data": {
            "access_token": "mock_jwt_token_12345",
            "refresh_token": "mock_refresh_token_67890",
            "user": {
                "wallet_address": wallet_address,
                "created_at": "2025-08-23T21:45:00Z"
            }
        },
        "message": "Login successful"
    }

@app.get("/api/v1/trading/health")
async def trading_health():
    """Mock trading health check"""
    return {
        "success": True,
        "trading_mode": "client_side_wallet",
        "api_connectivity": True,
        "network": "mainnet",
        "security_model": "non_custodial",
        "timestamp": "2025-08-23T21:45:00Z"
    }

async def test_endpoints():
    """Test endpoints after server starts"""
    await asyncio.sleep(2)
    
    async with httpx.AsyncClient() as client:
        try:
            # Test health
            response = await client.get("http://localhost:8001/health")
            print(f"✅ Health: {response.status_code} - {response.json()}")
            
            # Test auth
            auth_response = await client.post("http://localhost:8001/api/v1/auth/login", 
                json={"privy_access_token": "test_token", "wallet_address": "0x123"})
            print(f"✅ Auth: {auth_response.status_code} - Login successful")
            
            # Test trading
            trading_response = await client.get("http://localhost:8001/api/v1/trading/health")
            print(f"✅ Trading: {trading_response.status_code} - Trading health OK")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Start background test task
    asyncio.create_task(test_endpoints())
    
    print("🚀 Starting Nadas Test Server on port 8001...")
    print("📝 Testing authentication flow...")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)