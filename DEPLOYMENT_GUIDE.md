# Nadas.fi Backend Deployment Guide

## Overview

This guide covers the deployment of Nadas.fi backend to production environment with full cross-chain integration, security, and monitoring.

## Prerequisites

- Docker & Docker Compose v3.8+
- PostgreSQL 15+
- Redis 7+
- SSL certificates (Let's Encrypt recommended)
- Domain with DNS configured
- Secrets management system (HashiCorp Vault, AWS Secrets Manager, etc.)

## Environment Setup

### 1. Environment Files

Copy and configure environment files:

```bash
# Development
cp .env.development .env

# Staging
cp .env.staging .env.staging

# Production
cp .env.production .env.production
```

### 2. Required Environment Variables

Set the following variables in your deployment environment:

```bash
# Database
DATABASE_PASSWORD=<secure-password>

# JWT & Security
JWT_SECRET_KEY=<256-bit-secret>

# External Services
PRIVY_PROD_APP_ID=<privy-app-id>
PRIVY_PROD_APP_SECRET=<privy-secret>
GLUEX_PROD_API_KEY=<gluex-api-key>

# AWS (for AI services)
AWS_PROD_ACCESS_KEY_ID=<aws-key>
AWS_PROD_SECRET_ACCESS_KEY=<aws-secret>
AWS_PROD_BEARER_TOKEN_BEDROCK=<bedrock-token>

# Monitoring
SENTRY_PROD_DSN=<sentry-dsn>
GRAFANA_ADMIN_PASSWORD=<grafana-password>
```

## Deployment Steps

### 1. Development Deployment

```bash
# Start development environment
docker-compose up -d

# Run database migrations
docker-compose exec api alembic upgrade head

# Verify services
docker-compose ps
```

### 2. Staging Deployment

```bash
# Set environment
export ENVIRONMENT=staging

# Start staging with monitoring
docker-compose -f docker-compose.yml -f docker-compose.production.yml --profile monitoring up -d

# Run migrations
docker-compose exec api alembic upgrade head

# Run health checks
curl https://staging-api.nadas.fi/health
```

### 3. Production Deployment

```bash
# Set environment
export ENVIRONMENT=production

# Start production services
docker-compose -f docker-compose.yml -f docker-compose.production.yml --profile production --profile monitoring up -d

# Run migrations
docker-compose exec api alembic upgrade head

# Verify deployment
curl https://api.nadas.fi/health
```

## SSL Configuration

### 1. Let's Encrypt Setup

```bash
# Install certbot
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone -d api.nadas.fi

# Copy certificates to nginx directory
sudo cp /etc/letsencrypt/live/api.nadas.fi/* ./nginx/ssl/

# Set up auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

### 2. Nginx Configuration

Ensure your nginx configuration uses the production SSL settings:

```bash
# Copy production nginx config
cp nginx/nginx.prod.conf nginx/nginx.conf

# Restart nginx
docker-compose restart nginx
```

## Database Management

### 1. Migrations

```bash
# Create new migration
docker-compose exec api alembic revision --autogenerate -m "Description"

# Apply migrations
docker-compose exec api alembic upgrade head

# Rollback migration
docker-compose exec api alembic downgrade -1
```

### 2. Backup & Restore

```bash
# Create backup
docker-compose exec db pg_dump -U nadas_prod nadas_production > backup_$(date +%Y%m%d).sql

# Restore from backup
docker-compose exec -T db psql -U nadas_prod nadas_production < backup_20250122.sql
```

## Monitoring & Logging

### 1. Prometheus Metrics

Access metrics at: `https://your-domain:9090`

Key metrics to monitor:
- API response times
- Database connection pool
- Redis memory usage
- Cross-chain transaction success rates
- Security events

### 2. Grafana Dashboards

Access dashboards at: `https://your-domain:3001`

Default login: `admin` / `<GRAFANA_ADMIN_PASSWORD>`

### 3. Log Aggregation

Logs are collected in:
- Application logs: `./logs/`
- Nginx access logs: `/var/log/nginx/`
- Security audit logs: Redis + structured logging

## Security Considerations

### 1. Network Security

- Use private networks for internal communication
- Restrict database access to application containers only
- Configure firewall to allow only necessary ports (80, 443, 22)

### 2. Secrets Management

Never store secrets in environment files in production:

```bash
# Use AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id prod/nadas/database

# Or HashiCorp Vault
vault kv get secret/nadas/prod/database
```

### 3. Rate Limiting

Current rate limits (per IP):
- Auth endpoints: 5 requests/5min
- General API: 100 requests/min
- Upload endpoints: 1 request/min

Adjust in `app/middleware/security.py` as needed.

## Cross-Chain Integration

### 1. GlueX Configuration

Ensure GlueX API key has sufficient quotas:
- Cross-chain quotes: 1000/day
- Exchange rates: 10000/day
- Portfolio queries: 5000/day

### 2. Liquid Labs Integration

HyperEVM endpoints:
- Testnet: `https://api.hyperliquid-testnet.xyz/evm`
- Mainnet: `https://api.hyperliquid.xyz/evm`

### 3. Monitoring Cross-Chain Operations

Key metrics:
- Cross-chain transaction success rate
- Average bridge completion time
- Failed transaction reasons
- Price slippage distribution

## Scaling & Performance

### 1. Horizontal Scaling

```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale workers
docker-compose up -d --scale celery-worker=2
```

### 2. Database Optimization

Production PostgreSQL configuration:
- `max_connections=200`
- `shared_buffers=256MB`
- `effective_cache_size=1GB`
- `work_mem=4MB`

### 3. Redis Optimization

Production Redis configuration:
- `maxmemory=1gb`
- `maxmemory-policy=allkeys-lru`
- Persistent storage enabled

## Troubleshooting

### 1. Common Issues

**Service won't start:**
```bash
# Check logs
docker-compose logs api

# Check health
docker-compose exec api python -c "import app.main; print('OK')"
```

**Database connection errors:**
```bash
# Test connection
docker-compose exec api python -c "from app.core.database import engine; print(engine.connect())"

# Check migrations
docker-compose exec api alembic current
```

**Cross-chain API failures:**
```bash
# Test GlueX connection
docker-compose exec api python -c "
from app.adapters.gluex import get_gluex_adapter
import asyncio
async def test():
    async with get_gluex_adapter() as gluex:
        chains = gluex.supported_chains
        print(f'Chains: {len(chains)}')
asyncio.run(test())
"
```

### 2. Health Checks

All services have health checks configured:

```bash
# Check all service health
docker-compose ps

# Manual health check
curl https://api.nadas.fi/health
```

### 3. Performance Issues

Monitor these metrics:
- API response time > 2000ms
- Database query time > 1000ms
- Memory usage > 80%
- CPU usage > 70%

## Maintenance

### 1. Regular Tasks

**Daily:**
- Check service health
- Monitor error rates
- Review security logs

**Weekly:**
- Database backup verification
- SSL certificate status
- Dependency updates

**Monthly:**
- Security audit
- Performance optimization
- Capacity planning

### 2. Updates

```bash
# Pull latest images
docker-compose pull

# Apply updates with zero downtime
docker-compose up -d --remove-orphans

# Run any new migrations
docker-compose exec api alembic upgrade head
```

## Support

For deployment issues:
1. Check this guide first
2. Review logs in `./logs/`
3. Check monitoring dashboards
4. Contact DevOps team

**Emergency Contacts:**
- On-call DevOps: <on-call-number>
- Security team: <security-email>
- Infrastructure team: <infra-email>