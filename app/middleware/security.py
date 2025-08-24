"""
Security middleware for Nadas.fi API
Enhanced protection against common attacks and threats
"""

import time
import hashlib
import hmac
import ipaddress
from typing import Set, Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
from datetime import datetime, timedelta
import re
import json

from app.core.config import settings
from app.core.redis_client import get_redis_client

logger = structlog.get_logger(__name__)

# Security constants
SUSPICIOUS_PATTERNS = [
    r'(?i)(union|select|insert|delete|drop|create|alter|exec|script)',  # SQL injection
    r'(?i)(<script|javascript:|on\w+\s*=)',  # XSS
    r'(?i)(\.\.\/|\.\.\\|\/etc\/|\/proc\/)',  # Path traversal
    r'(?i)(cmd|powershell|bash|sh|eval|exec)',  # Command injection
]

MALICIOUS_USER_AGENTS = [
    'sqlmap', 'nikto', 'nmap', 'masscan', 'nessus', 'burp', 'dirb', 'gobuster'
]

# Rate limiting windows (endpoint -> (requests, window_seconds))
RATE_LIMITS = {
    '/api/v1/auth/login': (5, 300),  # 5 requests per 5 minutes
    '/api/v1/auth/register': (3, 3600),  # 3 requests per hour
    '/api/v1/auth/': (10, 300),  # 10 auth requests per 5 minutes
    '/api/v1/': (100, 60),  # 100 API requests per minute
    'default': (200, 60),  # Default: 200 requests per minute
}

# Blocked countries (ISO codes)
BLOCKED_COUNTRIES = {'CN', 'RU', 'KP', 'IR'}  # Add/remove as needed

# Trusted proxy IPs (Cloudflare, AWS ALB, etc.)
TRUSTED_PROXIES = {
    '173.245.48.0/20', '103.21.244.0/22', '103.22.200.0/22',
    '103.31.4.0/22', '141.101.64.0/18', '108.162.192.0/18',
    '190.93.240.0/20', '188.114.96.0/20', '197.234.240.0/22',
    '198.41.128.0/17', '162.158.0.0/15', '104.16.0.0/13',
    '104.24.0.0/14', '172.64.0.0/13', '131.0.72.0/22'
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enhanced security middleware with multiple protection layers"""
    
    def __init__(self, app):
        super().__init__(app)
        self.trusted_proxy_networks = [ipaddress.ip_network(ip) for ip in TRUSTED_PROXIES]
        self.redis_client = None
        
    async def dispatch(self, request: Request, call_next):
        # Initialize Redis client if needed
        if not self.redis_client:
            self.redis_client = await get_redis_client()
        
        # Get real client IP
        client_ip = self._get_real_client_ip(request)
        
        # Security checks
        security_result = await self._security_checks(request, client_ip)
        if security_result:
            return security_result
        
        # Rate limiting
        rate_limit_result = await self._check_rate_limit(request, client_ip)
        if rate_limit_result:
            return rate_limit_result
        
        # Add security headers to request
        request.state.client_ip = client_ip
        request.state.security_context = {
            'request_id': hashlib.md5(f"{client_ip}{time.time()}".encode()).hexdigest()[:8],
            'timestamp': datetime.utcnow().isoformat(),
            'user_agent': request.headers.get('user-agent', '')[:500]
        }
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Add security headers to response
            response = self._add_security_headers(response)
            
            # Log successful request
            await self._log_request(request, response, time.time() - start_time, client_ip)
            
            return response
            
        except Exception as e:
            # Log failed request
            await self._log_security_event(
                client_ip=client_ip,
                action='request_error',
                details={'error': str(e), 'path': str(request.url.path)},
                risk_level='medium'
            )
            raise
    
    def _get_real_client_ip(self, request: Request) -> str:
        """Extract real client IP, handling proxies and load balancers"""
        
        # Check X-Forwarded-For header (most common)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            # Take the first IP (original client)
            ips = [ip.strip() for ip in forwarded_for.split(',')]
            for ip in ips:
                try:
                    parsed_ip = ipaddress.ip_address(ip)
                    # Skip private/local IPs unless it's the only option
                    if not parsed_ip.is_private and not parsed_ip.is_loopback:
                        return ip
                except ValueError:
                    continue
            # If only private IPs, return the first one
            if ips:
                return ips[0]
        
        # Check other common headers
        for header in ['X-Real-IP', 'X-Client-IP', 'CF-Connecting-IP']:
            ip = request.headers.get(header)
            if ip:
                try:
                    ipaddress.ip_address(ip)
                    return ip
                except ValueError:
                    continue
        
        # Fallback to direct connection IP
        return request.client.host if request.client else 'unknown'
    
    async def _security_checks(self, request: Request, client_ip: str) -> Optional[Response]:
        """Perform various security checks"""
        
        # Check for blocked IPs
        if await self._is_ip_blocked(client_ip):
            await self._log_security_event(
                client_ip=client_ip,
                action='blocked_ip_access',
                details={'path': str(request.url.path)},
                risk_level='high'
            )
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check user agent
        user_agent = request.headers.get('user-agent', '').lower()
        if any(malicious in user_agent for malicious in MALICIOUS_USER_AGENTS):
            await self._log_security_event(
                client_ip=client_ip,
                action='malicious_user_agent',
                details={'user_agent': user_agent, 'path': str(request.url.path)},
                risk_level='high'
            )
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check for suspicious patterns in URL and headers
        full_url = str(request.url)
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, full_url):
                await self._log_security_event(
                    client_ip=client_ip,
                    action='suspicious_pattern_detected',
                    details={'pattern': pattern, 'url': full_url},
                    risk_level='high'
                )
                raise HTTPException(status_code=400, detail="Invalid request")
        
        # Check request size
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > 100 * 1024 * 1024:  # 100MB limit
            await self._log_security_event(
                client_ip=client_ip,
                action='oversized_request',
                details={'size': content_length, 'path': str(request.url.path)},
                risk_level='medium'
            )
            raise HTTPException(status_code=413, detail="Request too large")
        
        # Check for directory traversal in path
        if '..' in request.url.path or '/etc/' in request.url.path:
            await self._log_security_event(
                client_ip=client_ip,
                action='directory_traversal_attempt',
                details={'path': str(request.url.path)},
                risk_level='high'
            )
            raise HTTPException(status_code=400, detail="Invalid path")
        
        return None
    
    async def _check_rate_limit(self, request: Request, client_ip: str) -> Optional[Response]:
        """Check rate limiting per IP and endpoint"""
        
        path = request.url.path
        method = request.method
        
        # Determine rate limit for this endpoint
        rate_limit = None
        for pattern, limit in RATE_LIMITS.items():
            if pattern == 'default':
                continue
            if path.startswith(pattern):
                rate_limit = limit
                break
        
        if not rate_limit:
            rate_limit = RATE_LIMITS['default']
        
        max_requests, window_seconds = rate_limit
        
        # Create Redis key
        redis_key = f"rate_limit:{client_ip}:{path}:{method}"
        
        try:
            # Get current count
            current_count = await self.redis_client.get(redis_key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= max_requests:
                # Rate limit exceeded
                await self._log_security_event(
                    client_ip=client_ip,
                    action='rate_limit_exceeded',
                    details={
                        'path': path,
                        'method': method,
                        'current_count': current_count,
                        'limit': max_requests
                    },
                    risk_level='medium'
                )
                
                # Block IP temporarily if severely rate limited
                if current_count > max_requests * 3:
                    await self._block_ip_temporarily(client_ip, 3600)  # 1 hour
                
                return Response(
                    content='{"error": "Rate limit exceeded"}',
                    status_code=429,
                    headers={
                        'Content-Type': 'application/json',
                        'Retry-After': str(window_seconds),
                        'X-RateLimit-Limit': str(max_requests),
                        'X-RateLimit-Remaining': '0',
                        'X-RateLimit-Reset': str(int(time.time()) + window_seconds)
                    }
                )
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds)
            await pipe.execute()
            
            return None
            
        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            # Allow request on Redis failure
            return None
    
    async def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is in blocklist"""
        try:
            blocked_key = f"blocked_ip:{client_ip}"
            is_blocked = await self.redis_client.get(blocked_key)
            return bool(is_blocked)
        except Exception:
            return False
    
    async def _block_ip_temporarily(self, client_ip: str, duration_seconds: int):
        """Temporarily block an IP address"""
        try:
            blocked_key = f"blocked_ip:{client_ip}"
            await self.redis_client.setex(blocked_key, duration_seconds, '1')
            
            await self._log_security_event(
                client_ip=client_ip,
                action='ip_blocked_temporarily',
                details={'duration_seconds': duration_seconds},
                risk_level='high'
            )
        except Exception as e:
            logger.error("IP blocking error", error=str(e))
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Content-Security-Policy': csp,
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            'X-Permitted-Cross-Domain-Policies': 'none'
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    async def _log_security_event(
        self, 
        client_ip: str, 
        action: str, 
        details: Dict[str, Any], 
        risk_level: str = 'low'
    ):
        """Log security events for monitoring"""
        try:
            event = {
                'client_ip': client_ip,
                'action': action,
                'details': details,
                'risk_level': risk_level,
                'timestamp': datetime.utcnow().isoformat(),
                'service': 'nadas-api'
            }
            
            # Log to structured logger
            logger.warning("Security event", **event)
            
            # Store in Redis for real-time monitoring
            security_key = f"security_events:{datetime.utcnow().strftime('%Y%m%d%H')}"
            await self.redis_client.lpush(security_key, json.dumps(event))
            await self.redis_client.expire(security_key, 86400)  # Keep for 24 hours
            
        except Exception as e:
            logger.error("Security logging error", error=str(e))
    
    async def _log_request(
        self, 
        request: Request, 
        response: Response, 
        duration: float, 
        client_ip: str
    ):
        """Log request for monitoring and analytics"""
        try:
            if request.url.path in ['/health', '/metrics']:
                return  # Skip health check logs
            
            log_data = {
                'client_ip': client_ip,
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'duration_ms': round(duration * 1000, 2),
                'user_agent': request.headers.get('user-agent', '')[:200],
                'content_length': response.headers.get('content-length'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Log based on status code
            if response.status_code >= 500:
                logger.error("Server error", **log_data)
            elif response.status_code >= 400:
                logger.warning("Client error", **log_data)
            else:
                logger.info("Request completed", **log_data)
                
        except Exception as e:
            logger.error("Request logging error", error=str(e))


class CSRFProtection:
    """CSRF protection for state-changing operations"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str) -> str:
        """Generate CSRF token for user"""
        timestamp = str(int(time.time()))
        message = f"{user_id}:{timestamp}"
        signature = hashlib.hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{timestamp}:{signature}"
    
    def verify_token(self, token: str, user_id: str, max_age: int = 3600) -> bool:
        """Verify CSRF token"""
        try:
            timestamp_str, signature = token.split(':', 1)
            timestamp = int(timestamp_str)
            
            # Check token age
            if time.time() - timestamp > max_age:
                return False
            
            # Verify signature
            message = f"{user_id}:{timestamp_str}"
            expected_signature = hashlib.hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, TypeError):
            return False


# Global instances
csrf_protection = CSRFProtection(settings.SECRET_KEY)


def get_csrf_protection() -> CSRFProtection:
    """Get CSRF protection instance"""
    return csrf_protection