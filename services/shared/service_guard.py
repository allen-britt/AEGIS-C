"""
AEGIS‑C Service Guard
======================

Standardized authentication, metrics, and health checks for all FastAPI services.
"""

import os
import time
import structlog
from fastapi import FastAPI, Request, Depends, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import PlainTextResponse
from typing import Callable

# Configuration
API_KEY = os.getenv("API_KEY", "changeme-dev")
SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["service", "method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["service", "method", "endpoint"]
)

DETECTION_SCORES = Histogram(
    "detection_scores",
    "AI detection scores",
    ["service", "verdict"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

def require_api_key(request: Request):
    """Require valid API key for protected endpoints"""
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        logger.warning("Unauthorized access attempt", 
                      service=SERVICE_NAME,
                      client_ip=request.client.host if request.client else "unknown")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def setup_service_guard(app: FastAPI, service_name: str = None):
    """Setup standardized guard for FastAPI service"""
    global SERVICE_NAME
    SERVICE_NAME = service_name or SERVICE_NAME
    
    @app.get("/health")
    async def health_check():
        """Standard health check endpoint"""
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "version": "1.0.0",
            "timestamp": time.time()
        }
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/secure/ping", dependencies=[Depends(require_api_key)])
    async def secure_ping():
        """Protected ping endpoint for authentication testing"""
        return {"pong": True, "service": SERVICE_NAME}
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable):
        """Middleware to collect request metrics"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                service=SERVICE_NAME,
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                service=SERVICE_NAME,
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Log request
            logger.info(
                "HTTP request completed",
                service=SERVICE_NAME,
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=round(duration, 3),
                client_ip=request.client.host if request.client else "unknown"
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                service=SERVICE_NAME,
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                service=SERVICE_NAME,
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            logger.error(
                "HTTP request failed",
                service=SERVICE_NAME,
                method=request.method,
                endpoint=request.url.path,
                error=str(e),
                duration=round(duration, 3),
                client_ip=request.client.host if request.client else "unknown"
            )
            
            raise
    
    logger.info("Service guard initialized", service=SERVICE_NAME)

def record_detection_score(score: float, verdict: str):
    """Record AI detection score for metrics"""
    DETECTION_SCORES.labels(service=SERVICE_NAME, verdict=verdict).observe(score)
    logger.info("Detection score recorded", 
                service=SERVICE_NAME, 
                score=round(score, 3), 
                verdict=verdict)

# Export for use in services
__all__ = [
    "setup_service_guard",
    "require_api_key", 
    "record_detection_score",
    "logger"
]