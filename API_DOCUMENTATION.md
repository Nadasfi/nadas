# Nadas.fi Backend API Documentation

## Overview

Nadas.fi backend provides a comprehensive DeFi automation platform with cross-chain support, AI-powered insights, and secure non-custodial trading on Hyperliquid.

**Base URL:** `https://api.nadas.fi`  
**API Version:** v1  
**Authentication:** JWT Bearer Token + Privy Integration  

## Quick Start

### 1. Authentication

```bash
# Login with Privy
curl -X POST https://api.nadas.fi/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "privy_token": "your_privy_token",
    "wallet_address": "0x..."
  }'
```

### 2. Get Portfolio

```bash
# Get portfolio overview
curl -X GET https://api.nadas.fi/api/v1/portfolio \
  -H "Authorization: Bearer your_jwt_token"
```

### 3. Create Automation

```bash
# Create DCA automation
curl -X POST https://api.nadas.fi/api/v1/automation/rules \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "rule_type": "dca",
    "asset": "ETH",
    "amount_usd": 100,
    "frequency": "daily",
    "conditions": {...}
  }'
```

## Core Features

### 🤖 Automation Engine
- **DCA (Dollar Cost Averaging):** Regular token purchases
- **Stop-Loss:** Automatic position protection
- **Grid Trading:** Range-bound trading strategies
- **Rebalancing:** Portfolio optimization

### 🌉 Cross-Chain Integration
- **GlueX Router:** Cross-chain swaps and deposits
- **Bridge Aggregation:** Optimal route finding
- **Multi-Chain Portfolio:** Unified view across chains

### 🔄 HyperEVM Integration
- **LiquidSwap:** DEX aggregation on HyperEVM
- **LiquidLaunch:** Token creation and bonding curves
- **Real-time Data:** WebSocket price feeds

### 🧠 AI-Powered Insights
- **Claude 3.5 Sonnet:** Advanced market analysis
- **Risk Assessment:** Intelligent automation recommendations
- **Turkish Support:** Native language AI assistance

## API Endpoints

### Authentication (`/api/v1/auth/`)

#### POST `/login`
Login with Privy wallet authentication.

**Request:**
```json
{
  "privy_token": "string",
  "wallet_address": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "string",
    "refresh_token": "string",
    "user": {
      "id": "uuid",
      "wallet_address": "string",
      "created_at": "datetime"
    }
  }
}
```

#### POST `/refresh`
Refresh JWT token.

---

### Portfolio Management (`/api/v1/portfolio/`)

#### GET `/`
Get user's portfolio overview.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_value_usd": 15420.50,
    "pnl_24h": 234.50,
    "pnl_percentage_24h": 1.54,
    "positions": [
      {
        "asset": "ETH",
        "size": "5.2",
        "value_usd": 12000.00,
        "pnl_usd": 150.00,
        "pnl_percentage": 1.25
      }
    ],
    "cross_chain_total_value": 18750.00,
    "chain_distribution": {
      "ethereum": 12000.00,
      "arbitrum": 3420.50,
      "polygon": 1875.00
    }
  }
}
```

#### GET `/performance`
Get portfolio performance metrics.

---

### Automation Engine (`/api/v1/automation/`)

#### GET `/rules`
List user's automation rules.

**Query Parameters:**
- `status`: `active`, `paused`, `completed`
- `rule_type`: `dca`, `stop_loss`, `grid_trading`, `rebalancing`
- `limit`: Number of results (default: 50)

#### POST `/rules`
Create new automation rule.

**Request:**
```json
{
  "rule_type": "dca",
  "name": "ETH Daily DCA",
  "asset": "ETH",
  "amount_usd": 100,
  "frequency": "daily",
  "conditions": {
    "max_slippage": 0.01,
    "price_threshold": null
  },
  "target_chain": "ethereum",
  "cross_chain_enabled": false
}
```

#### PUT `/rules/{rule_id}`
Update automation rule.

#### DELETE `/rules/{rule_id}`
Delete automation rule.

#### POST `/rules/{rule_id}/pause`
Pause automation rule.

#### POST `/rules/{rule_id}/resume`
Resume automation rule.

---

### Cross-Chain Integration (`/api/v1/gluex/`)

#### POST `/quote`
Get cross-chain swap quote.

**Request:**
```json
{
  "input_token": "USDC",
  "output_token": "ETH",
  "input_amount": "1000",
  "target_chain": "arbitrum",
  "slippage_tolerance": 0.005,
  "is_permit2": true
}
```

#### POST `/deposit`
Execute cross-chain deposit.

**Request:**
```json
{
  "source_chain": "polygon",
  "target_chain": "arbitrum",
  "source_token": "USDC",
  "target_token": "USDC",
  "amount": "1000",
  "slippage_tolerance": 0.01
}
```

#### GET `/portfolio`
Get cross-chain portfolio overview.

#### GET `/supported-chains`
List supported blockchain networks.

#### POST `/bridge-routes`
Find optimal bridge routes.

---

### HyperEVM Integration (`/api/v1/liquidlabs/`)

#### POST `/swap/quote`
Get LiquidSwap routing quote.

**Request:**
```json
{
  "token_in": "0x...",
  "token_out": "0x...",
  "amount_in": "1000000000",
  "multi_hop": true,
  "slippage": 1.0,
  "exclude_dexes": "1,2"
}
```

#### POST `/swap/execute`
Execute swap via LiquidSwap.

#### GET `/tokens`
List available tokens on LiquidSwap.

#### POST `/launch/create-token`
Create token via LiquidLaunch.

**Request:**
```json
{
  "name": "My Token",
  "symbol": "TOKEN",
  "description": "A new token on HyperEVM",
  "image_url": "https://...",
  "initial_buy_hype": 1.0
}
```

#### GET `/launch/tokens/{token_address}`
Get bonding curve information.

#### POST `/launch/buy`
Buy tokens from bonding curve.

---

### AI Assistant (`/api/v1/ai/`)

#### POST `/chat`
Chat with AI assistant.

**Request:**
```json
{
  "message": "ETH için DCA stratejisi öner",
  "include_market_data": true,
  "include_portfolio": true
}
```

#### POST `/analyze-automation`
Get AI analysis for automation request.

**Request:**
```json
{
  "user_request": "Want to DCA into ETH with $100 weekly",
  "risk_tolerance": "medium"
}
```

#### POST `/market-analysis`
Get AI-powered market analysis.

---

### Live Trading (`/api/v1/trading/`)

#### GET `/positions`
Get current trading positions.

#### POST `/orders`
Place new trading order.

#### GET `/orders`
List trading orders.

#### DELETE `/orders/{order_id}`
Cancel trading order.

---

### Simulation (`/api/v1/simulation/`)

#### POST `/backtest`
Run strategy backtest.

**Request:**
```json
{
  "strategy": "dca",
  "asset": "ETH",
  "parameters": {
    "amount_usd": 100,
    "frequency": "weekly"
  },
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

#### POST `/scenario`
Run scenario analysis.

---

### Notifications (`/api/v1/notifications/`)

#### GET `/`
List user notifications.

#### POST `/rules`
Create notification rule.

#### GET `/rules`
List notification rules.

## WebSocket API

### Connection
```javascript
const ws = new WebSocket('wss://api.nadas.fi/ws');
```

### Channels

#### `portfolio_updates`
Real-time portfolio value changes.

#### `automation_events`
Automation rule execution events.

#### `price_alerts`
Price-based notifications.

#### `cross_chain_status`
Cross-chain transaction updates.

### Message Format
```json
{
  "channel": "portfolio_updates",
  "type": "position_update",
  "data": {
    "asset": "ETH",
    "new_value": 12500.00,
    "change": 150.00
  },
  "timestamp": "2025-01-22T10:30:00Z"
}
```

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": "Invalid request parameters",
  "error_code": "INVALID_PARAMS",
  "details": {
    "field": "amount",
    "message": "Amount must be positive"
  }
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

### Common Error Codes
- `INVALID_PARAMS` - Invalid request parameters
- `UNAUTHORIZED` - Authentication required
- `INSUFFICIENT_FUNDS` - Not enough balance
- `RATE_LIMITED` - Too many requests
- `SLIPPAGE_EXCEEDED` - Price movement too high
- `CHAIN_NOT_SUPPORTED` - Blockchain not supported

## Rate Limiting

### Limits per IP:
- **Auth endpoints:** 5 requests/5min
- **General API:** 100 requests/min
- **Trading endpoints:** 50 requests/min
- **Upload endpoints:** 1 request/min

### Headers:
- `X-RateLimit-Limit` - Request limit
- `X-RateLimit-Remaining` - Remaining requests
- `X-RateLimit-Reset` - Reset timestamp
- `Retry-After` - Seconds to wait (when limited)

## SDKs & Libraries

### JavaScript/TypeScript
```bash
npm install @nadas-fi/sdk
```

```javascript
import { NadasSDK } from '@nadas-fi/sdk';

const nadas = new NadasSDK({
  apiKey: 'your_api_key',
  baseURL: 'https://api.nadas.fi'
});

// Get portfolio
const portfolio = await nadas.portfolio.get();

// Create automation
const rule = await nadas.automation.createRule({
  type: 'dca',
  asset: 'ETH',
  amount: 100
});
```

### Python
```bash
pip install nadas-python-sdk
```

```python
from nadas import NadasClient

client = NadasClient(api_key='your_api_key')

# Get portfolio
portfolio = client.portfolio.get()

# Create automation
rule = client.automation.create_rule(
    type='dca',
    asset='ETH',
    amount=100
)
```

## Security

### Authentication
- JWT tokens with 30-minute expiration
- Refresh tokens for long-term access
- Privy integration for wallet-based auth

### API Security
- Rate limiting per IP and user
- Request validation and sanitization
- CSRF protection for state changes
- SQL injection prevention
- XSS protection headers

### Data Protection
- All data encrypted in transit (TLS 1.3)
- Sensitive data encrypted at rest
- No private keys stored
- Client-side wallet signing

## Testing

### Testnet Environment
**Base URL:** `https://staging-api.nadas.fi`

Use Hyperliquid testnet for safe testing:
- Testnet tokens available via faucet
- All functionality available
- No real funds at risk

### Example Test Flow
```bash
# 1. Get testnet tokens
curl -X POST https://testnet-faucet.hyperliquid.xyz/api/faucet

# 2. Connect to staging API
export API_BASE=https://staging-api.nadas.fi

# 3. Test automation
curl -X POST $API_BASE/api/v1/automation/rules \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"rule_type": "dca", "asset": "ETH", "amount_usd": 10}'
```

## Support

### Documentation
- API Reference: https://docs.nadas.fi/api
- Guides: https://docs.nadas.fi/guides
- Examples: https://github.com/nadas-fi/examples

### Community
- Discord: https://discord.gg/nadas-fi
- Telegram: https://t.me/nadasfi
- Twitter: https://twitter.com/nadas_fi

### Technical Support
- Email: dev@nadas.fi
- GitHub Issues: https://github.com/nadas-fi/api/issues
- Status Page: https://status.nadas.fi