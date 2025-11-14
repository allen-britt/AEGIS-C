"""
AEGIS‑C Shared Health & Metrics Module
========================================

Standardized health checks, metrics, and auth for all FastAPI services
"""

import os
import time
import structlog
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, Depends, Request, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configuration
API_KEY = os.getenv("API_KEY", "changeme-dev")
SERVICE_NAME = os.getenv("SERVICE_NAME", "aegis-c-service")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Structured logging
logger = structlog.get_logger()

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

ACTIVE_CONNECTIONS = Gauge(
    "active_connections",
    "Active connections"
)

SERVICE_HEALTH = Gauge(
    "service_health",
    "Service health status (1=healthy, 0=unhealthy)"
)

LAST_REQUEST_TIME = Gauge(
    "last_request_timestamp",
    "Timestamp of last request"
)

def require_api_key(request: Request):
    """Require API key authentication"""
    if request.headers.get("x-api-key") != API_KEY:
        logger.warning("Unauthorized access attempt", 
                      client_ip=request.client.host,
                      user_agent=request.headers.get("user-agent"))
        raise HTTPException(status_code=401, detail="Invalid API key")

def setup_health_metrics(app: FastAPI) -> None:
    """Setup health check and metrics endpoints for FastAPI app"""
    
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        try:
            SERVICE_HEALTH.set(1)
            return {
                "status": "healthy",
                "service": SERVICE_NAME,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        except Exception as e:
            SERVICE_HEALTH.set(0)
            logger.error("Health check failed", error=str(e))
            raise HTTPException(status_code=503, detail="Service unhealthy")
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with dependencies"""
        health_status = {
            "status": "healthy",
            "service": SERVICE_NAME,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "dependencies": {}
        }
        
        # Check database connections if configured
        if os.getenv("POSTGRES_URL"):
            try:
                # Add actual DB health check here
                health_status["dependencies"]["postgres"] = "healthy"
            except Exception as e:
                health_status["dependencies"]["postgres"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
        
        if os.getenv("REDIS_URL"):
            try:
                # Add actual Redis health check here
                health_status["dependencies"]["redis"] = "healthy"
            except Exception as e:
                health_status["dependencies"]["redis"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
        
        SERVICE_HEALTH.set(1 if health_status["status"] == "healthy" else 0)
        return health_status
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/info")
    async def service_info():
        """Service information endpoint"""
        return {
            "name": SERVICE_NAME,
            "version": "1.0.0",
            "description": "AEGIS‑C Security Service",
            "environment": os.getenv("AEGISC_ENV", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            LAST_REQUEST_TIME.set(time.time())
            
            # Log request
            logger.info(
                "HTTP request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
                client_ip=request.client.host
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            
            logger.error(
                "HTTP request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                client_ip=request.client.host
            )
            
            raise e
            
        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()

def setup_service_logging():
    """Setup structured logging for the service"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Example protected endpoint decorator
def protected_endpoint():
    """Decorator for protected endpoints"""
    return Depends(require_api_key)

# Direct callable for dependencies
protected_dependency = Depends(require_api_key)

# Utility functions for service health
def set_service_healthy():
    """Mark service as healthy"""
    SERVICE_HEALTH.set(1)
    logger.info("Service marked as healthy")

def set_service_unhealthy(reason: str = "Unknown"):
    """Mark service as unhealthy"""
    SERVICE_HEALTH.set(0)
    logger.warning("Service marked as unhealthy", reason=reason)

# Export key components
__all__ = [
    "setup_health_metrics",
    "MetricsMiddleware", 
    "require_api_key",
    "protected_endpoint",
    "setup_service_logging",
    "set_service_healthy",
    "set_service_unhealthy",
    "logger"
]