"""
Production Error Handling and Monitoring
Circuit breakers, retry logic, error tracking, and health monitoring
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import traceback
import json
from contextlib import asynccontextmanager

from app.core.logging import get_logger

logger = get_logger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Circuit is open (failing)
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: int = 60      # Seconds before attempting recovery
    success_threshold: int = 3      # Successes needed to close circuit
    timeout: int = 30              # Request timeout in seconds


@dataclass
class ErrorMetrics:
    total_requests: int = 0
    failed_requests: int = 0
    success_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    
    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        return 1.0 - self.failure_rate


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = ErrorMetrics()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if circuit should be opened
            if self._should_open_circuit():
                self.metrics.circuit_state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED for {self.name}")
            
            # If circuit is open, check if we can try recovery
            elif self._should_attempt_recovery():
                self.metrics.circuit_state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker HALF-OPEN for {self.name}, attempting recovery")
            
            # Circuit is open and not ready for recovery
            elif self.metrics.circuit_state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenException(f"Circuit breaker open for {self.name}")
        
        # Execute the function
        try:
            self.metrics.total_requests += 1
            
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = func(*args, **kwargs)
            
            await self._record_success()
            return result
            
        except asyncio.TimeoutError as e:
            await self._record_failure(f"Timeout after {self.config.timeout}s: {str(e)}")
            raise
        except Exception as e:
            await self._record_failure(str(e))
            raise
    
    async def _record_success(self):
        """Record a successful request"""
        async with self._lock:
            self.metrics.success_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.utcnow()
            
            # Close circuit if enough successes in half-open state
            if (self.metrics.circuit_state == CircuitBreakerState.HALF_OPEN and 
                self.metrics.consecutive_successes >= self.config.success_threshold):
                self.metrics.circuit_state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker CLOSED for {self.name}")
    
    async def _record_failure(self, error: str):
        """Record a failed request"""
        async with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = datetime.utcnow()
            
            logger.error(f"Circuit breaker failure for {self.name}: {error}")
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened"""
        return (self.metrics.circuit_state == CircuitBreakerState.CLOSED and
                self.metrics.consecutive_failures >= self.config.failure_threshold)
    
    def _should_attempt_recovery(self) -> bool:
        """Check if we should attempt recovery"""
        if self.metrics.circuit_state != CircuitBreakerState.OPEN:
            return False
        
        if not self.metrics.last_failure_time:
            return False
        
        time_since_failure = datetime.utcnow() - self.metrics.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        return {
            "name": self.name,
            "state": self.metrics.circuit_state.value,
            "total_requests": self.metrics.total_requests,
            "success_rate": self.metrics.success_rate,
            "failure_rate": self.metrics.failure_rate,
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0      # Base delay in seconds
    max_delay: float = 60.0      # Maximum delay in seconds  
    exponential_base: float = 2.0 # Exponential backoff base
    jitter: bool = True          # Add random jitter


class RetryableException(Exception):
    """Base class for exceptions that should trigger retries"""
    pass


async def retry_with_backoff(
    func: Callable,
    config: RetryConfig = None,
    retryable_exceptions: tuple = (RetryableException, ConnectionError, TimeoutError),
    *args, **kwargs
) -> Any:
    """Retry function with exponential backoff"""
    config = config or RetryConfig()
    
    for attempt in range(config.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except retryable_exceptions as e:
            if attempt == config.max_attempts - 1:
                logger.error(f"All retry attempts failed for {func.__name__}: {str(e)}")
                raise
            
            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            
            # Add jitter to prevent thundering herd
            if config.jitter:
                import random
                delay = delay * (0.5 + random.random() * 0.5)
            
            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s: {str(e)}")
            await asyncio.sleep(delay)


class ServiceHealthMonitor:
    """Monitor service health and provide metrics"""
    
    def __init__(self):
        self.services: Dict[str, CircuitBreaker] = {}
        self.health_checks: Dict[str, Callable] = {}
        self._monitoring = False
    
    def register_service(self, name: str, circuit_breaker: CircuitBreaker):
        """Register a service with circuit breaker"""
        self.services[name] = circuit_breaker
        logger.info(f"Registered service {name} with circuit breaker")
    
    def register_health_check(self, name: str, health_check: Callable):
        """Register a health check function"""
        self.health_checks[name] = health_check
        logger.info(f"Registered health check for {name}")
    
    async def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status for a specific service"""
        health_status = {
            "service": service_name,
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breaker": None,
            "health_check": None
        }
        
        # Get circuit breaker metrics
        if service_name in self.services:
            cb_metrics = self.services[service_name].get_metrics()
            health_status["circuit_breaker"] = cb_metrics
            
            # Determine status from circuit breaker
            if cb_metrics["state"] == "open":
                health_status["status"] = "unhealthy"
            elif cb_metrics["failure_rate"] > 0.5:
                health_status["status"] = "degraded"
            else:
                health_status["status"] = "healthy"
        
        # Run health check if available
        if service_name in self.health_checks:
            try:
                health_check_result = await self.health_checks[service_name]()
                health_status["health_check"] = health_check_result
                
                # Override status if health check indicates issues
                if not health_check_result.get("healthy", True):
                    health_status["status"] = "unhealthy"
                    
            except Exception as e:
                health_status["health_check"] = {"error": str(e), "healthy": False}
                health_status["status"] = "unhealthy"
        
        return health_status
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        system_health = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "services": {},
            "summary": {
                "healthy_services": 0,
                "degraded_services": 0,
                "unhealthy_services": 0,
                "unknown_services": 0
            }
        }
        
        # Check all registered services
        service_names = set(self.services.keys()) | set(self.health_checks.keys())
        
        for service_name in service_names:
            service_health = await self.get_service_health(service_name)
            system_health["services"][service_name] = service_health
            
            # Update summary
            status = service_health["status"]
            system_health["summary"][f"{status}_services"] += 1
        
        # Determine overall system status
        if system_health["summary"]["unhealthy_services"] > 0:
            system_health["overall_status"] = "unhealthy"
        elif system_health["summary"]["degraded_services"] > 0:
            system_health["overall_status"] = "degraded"
        
        return system_health
    
    async def start_monitoring(self, interval: int = 60):
        """Start periodic health monitoring"""
        self._monitoring = True
        logger.info(f"Started health monitoring with {interval}s interval")
        
        while self._monitoring:
            try:
                health = await self.get_system_health()
                logger.info("System health check", extra=health["summary"])
                
                # Log unhealthy services
                for service_name, service_health in health["services"].items():
                    if service_health["status"] == "unhealthy":
                        logger.error(f"Service {service_name} is unhealthy", 
                                   extra=service_health)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {str(e)}")
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring = False
        logger.info("Stopped health monitoring")


class ErrorTracker:
    """Track and analyze errors across the system"""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    async def record_error(self, service: str, error: Exception, context: Dict[str, Any] = None):
        """Record an error with context"""
        async with self._lock:
            error_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "service": service,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "context": context or {}
            }
            
            # Add to errors list
            self.errors.append(error_record)
            
            # Maintain max size
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)
            
            # Update error counts
            error_key = f"{service}:{type(error).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            logger.error(f"Error recorded for {service}", extra=error_record)
    
    async def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            recent_errors = [
                error for error in self.errors
                if datetime.fromisoformat(error["timestamp"]) > cutoff_time
            ]
            
            # Group by service
            service_errors = {}
            error_type_counts = {}
            
            for error in recent_errors:
                service = error["service"]
                error_type = error["error_type"]
                
                if service not in service_errors:
                    service_errors[service] = []
                service_errors[service].append(error)
                
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            return {
                "time_period_hours": hours,
                "total_errors": len(recent_errors),
                "service_errors": {
                    service: len(errors) for service, errors in service_errors.items()
                },
                "error_type_counts": error_type_counts,
                "most_recent_errors": recent_errors[-10:] if recent_errors else [],
                "timestamp": datetime.utcnow().isoformat()
            }


# Global instances
_health_monitor: Optional[ServiceHealthMonitor] = None
_error_tracker: Optional[ErrorTracker] = None


def get_health_monitor() -> ServiceHealthMonitor:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ServiceHealthMonitor()
    return _health_monitor


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


# Decorator for circuit breaker protection
def circuit_breaker_protected(name: str, config: CircuitBreakerConfig = None):
    """Decorator to protect functions with circuit breaker"""
    def decorator(func):
        circuit_breaker = CircuitBreaker(name, config)
        
        # Register with health monitor
        health_monitor = get_health_monitor()
        health_monitor.register_service(name, circuit_breaker)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Context manager for error tracking
@asynccontextmanager
async def track_errors(service_name: str, context: Dict[str, Any] = None):
    """Context manager to automatically track errors"""
    error_tracker = get_error_tracker()
    
    try:
        yield
    except Exception as e:
        await error_tracker.record_error(service_name, e, context)
        raise


# Convenience functions
async def safe_execute(
    func: Callable,
    service_name: str,
    circuit_breaker_config: CircuitBreakerConfig = None,
    retry_config: RetryConfig = None,
    context: Dict[str, Any] = None,
    *args, **kwargs
) -> Any:
    """Safely execute a function with full error handling"""
    
    # Create circuit breaker if not exists
    circuit_breaker = CircuitBreaker(service_name, circuit_breaker_config)
    
    async with track_errors(service_name, context):
        # First apply retry logic, then circuit breaker
        async def retry_wrapper():
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return await retry_with_backoff(retry_wrapper, retry_config)


# Health check helper
async def create_health_check(service_name: str, check_func: Callable) -> Dict[str, Any]:
    """Create a standardized health check response"""
    try:
        start_time = time.time()
        result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
        response_time = time.time() - start_time
        
        return {
            "service": service_name,
            "healthy": True,
            "response_time_ms": round(response_time * 1000, 2),
            "details": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "service": service_name,
            "healthy": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }