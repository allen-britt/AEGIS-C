"""
AEGIS‑C Honeynet Service
=========================

Deception APIs with canary telemetry and standardized guard.
"""

import os
import sys
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from datetime import datetime, timezone
import uuid
import json
from typing import Dict, List, Optional, Any
import hashlib
import time

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import setup_service_guard, record_detection_score, logger, require_api_key

# Service Configuration
SERVICE_NAME = "honeynet"
SERVICE_PORT = int(os.getenv("HONEYPOT_PORT", "8012"))

# Initialize FastAPI App
app = FastAPI(title="AEGIS‑C Honeynet Service")

# Setup standardized guard
setup_service_guard(app, SERVICE_NAME)

# Canary token generator and tracker
CANARY_TOKENS = set()
HONEYPOT_DATA = {}

# Initialize honeypot datasets with embedded canaries
def initialize_honeypot_data():
    """Initialize fake datasets with embedded canary tokens."""
    
    # Fake cities dataset
    cities = []
    for i in range(20):
        canary = f"CNRY-{uuid.uuid4().hex[:8]}"
        city = {
            "id": i + 1,
            "name": f"FakeCity{i}",
            "population": 10000 + i * 1000,
            "coordinates": {"lat": 40.0 + i * 0.1, "lng": -74.0 + i * 0.1},
            "mayor": f"Mayor{i}",
            "founded": 1900 + i * 5,
            "canary": canary,
            "metadata": f"secret_data_{canary}"
        }
        cities.append(city)
        CANARY_TOKENS.add(canary)
    
    # Fake user credentials dataset
    credentials = []
    for i in range(15):
        canary = f"CNRY-{uuid.uuid4().hex[:8]}"
        cred = {
            "id": i + 1,
            "username": f"user{i}@example.com",
            "password_hash": hashlib.sha256(f"fake_password_{canary}".encode()).hexdigest(),
            "role": "user" if i % 3 != 0 else "admin",
            "last_login": "2024-01-01T00:00:00Z",
            "canary": canary,
            "api_key": f"ak_{canary}"
        }
        credentials.append(cred)
        CANARY_TOKENS.add(canary)
    
    # Fake API keys dataset
    api_keys = []
    for i in range(25):
        canary = f"CNRY-{uuid.uuid4().hex[:8]}"
        key = {
            "id": i + 1,
            "key": f"sk-{canary}",
            "name": f"API Key {i}",
            "permissions": ["read", "write"] if i % 2 == 0 else ["read"],
            "created": "2024-01-01T00:00:00Z",
            "canary": canary,
            "usage_limit": 1000 + i * 100
        }
        api_keys.append(key)
        CANARY_TOKENS.add(canary)
    
    # Fake configuration data
    configs = []
    for i in range(10):
        canary = f"CNRY-{uuid.uuid4().hex[:8]}"
        config = {
            "id": i + 1,
            "service": f"service{i}",
            "config": {
                "database_url": f"postgresql://user:pass_{canary}@localhost/db",
                "redis_host": f"redis-{canary}.example.com",
                "secret_key": f"secret_{canary}",
                "webhook_url": f"https://webhook.example.com/{canary}"
            },
            "canary": canary,
            "version": f"1.{i}.0"
        }
        configs.append(config)
        CANARY_TOKENS.add(canary)
    
    HONEYPOT_DATA.update({
        "cities": cities,
        "credentials": credentials,
        "api_keys": api_keys,
        "configs": configs
    })

# Initialize data on startup
initialize_honeypot_data()

class TelemetryEvent(BaseModel):
    timestamp: str
    source_ip: str
    user_agent: str
    path: str
    method: str
    query_params: Dict[str, Any]
    headers: Dict[str, str]
    body_size: int
    response_status: int
    response_size: int
    duration_ms: float
    canary_detected: Optional[str] = None

# In-memory telemetry storage (in production, use message bus)
TELEMETRY_BUFFER: List[TelemetryEvent] = []

def extract_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxies."""
    # Check for common proxy headers
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    
    return request.client.host if request.client else "unknown"

def detect_canary_in_request(request: Request, body: bytes) -> Optional[str]:
    """Detect if request contains any canary tokens."""
    # Check headers
    for header_name, header_value in request.headers.items():
        for canary in CANARY_TOKENS:
            if canary in header_value:
                return canary
    
    # Check query parameters
    for param_value in request.query_params.values():
        for canary in CANARY_TOKENS:
            if canary in param_value:
                return canary
    
    # Check body
    body_str = body.decode('utf-8', errors='ignore')
    for canary in CANARY_TOKENS:
        if canary in body_str:
            return canary
    
    return None

@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    """Capture telemetry for all requests."""
    start_time = time.time()
    client_ip = extract_client_ip(request)
    
    # Read request body
    body = await request.body()
    
    # Detect canaries in request
    canary = detect_canary_in_request(request, body)
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (time.time() - start_time) * 1000
    
    # Create telemetry event
    event = TelemetryEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source_ip=client_ip,
        user_agent=request.headers.get("user-agent", ""),
        path=request.url.path,
        method=request.method,
        query_params=dict(request.query_params),
        headers=dict(request.headers),
        body_size=len(body),
        response_status=response.status_code,
        response_size=response.headers.get("content-length", 0),
        duration_ms=duration,
        canary_detected=canary
    )
    
    # Store telemetry
    TELEMETRY_BUFFER.append(event)
    
    # Log suspicious activity
    if canary:
        logger.warning(f"CANARY DETECTED: {canary} from IP {client_ip} on {request.url.path}")
    
    # Keep buffer size manageable
    if len(TELEMETRY_BUFFER) > 1000:
        TELEMETRY_BUFFER[:] = TELEMETRY_BUFFER[-500:]
    
    return response

# API Endpoints

@app.get("/api/datasets")
async def list_datasets():
    """List available honeypot datasets."""
    return {"datasets": list(HONEYPOT_DATA.keys())}

@app.get("/api/datasets/{dataset_name}")
async def get_dataset(dataset_name: str, limit: Optional[int] = None):
    """Get a honeypot dataset with optional limit."""
    if dataset_name not in HONEYPOT_DATA:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    data = HONEYPOT_DATA[dataset_name]
    if limit:
        data = data[:limit]
    
    return {
        "dataset": dataset_name,
        "count": len(data),
        "data": data
    }

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get specific user (honeypot endpoint)."""
    credentials = HONEYPOT_DATA.get("credentials", [])
    
    # Simulate user lookup
    user = next((cred for cred in credentials if cred["id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.post("/api/auth/login")
async def login(username: str, password: str):
    """Fake login endpoint."""
    # Always return failure for real attempts, but log them
    logger.info(f"Login attempt: {username} from IP")
    
    # Simulate processing delay
    time.sleep(0.5)
    
    return {
        "success": False,
        "message": "Invalid credentials",
        "attempt_id": str(uuid.uuid4())
    }

@app.get("/api/config/{service_name}")
async def get_config(service_name: str):
    """Get configuration (honeypot endpoint)."""
    configs = HONEYPOT_DATA.get("configs", [])
    
    config = next((cfg for cfg in configs if cfg["service"] == service_name), None)
    if not config:
        raise HTTPException(status_code=404, detail="Service configuration not found")
    
    return config

@app.get("/api/keys/validate")
async def validate_api_key(api_key: str):
    """Validate API key (honeypot endpoint)."""
    keys = HONEYPOT_DATA.get("api_keys", [])
    
    key = next((k for k in keys if k["key"] == api_key), None)
    if not key:
        return {"valid": False, "message": "Invalid API key"}
    
    return {
        "valid": True,
        "key_id": key["id"],
        "permissions": key["permissions"],
        "usage_limit": key["usage_limit"]
    }

# Telemetry and Analysis Endpoints

@app.get("/telemetry/recent")
async def get_recent_telemetry(limit: int = 50):
    """Get recent telemetry events."""
    events = TELEMETRY_BUFFER[-limit:] if limit else TELEMETRY_BUFFER
    return {"events": events, "total": len(events)}

@app.get("/telemetry/stats")
async def get_telemetry_stats():
    """Get telemetry statistics."""
    if not TELEMETRY_BUFFER:
        return {"message": "No telemetry data available"}
    
    total_requests = len(TELEMETRY_BUFFER)
    unique_ips = len(set(event.source_ip for event in TELEMETRY_BUFFER))
    canary_hits = len([event for event in TELEMETRY_BUFFER if event.canary_detected])
    
    # Top paths
    path_counts = {}
    for event in TELEMETRY_BUFFER:
        path_counts[event.path] = path_counts.get(event.path, 0) + 1
    
    top_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_requests": total_requests,
        "unique_ips": unique_ips,
        "canary_detections": canary_hits,
        "canary_hit_rate": round(canary_hits / total_requests * 100, 2) if total_requests > 0 else 0,
        "top_paths": dict(top_paths)
    }

@app.post("/telemetry/clear")
async def clear_telemetry():
    """Clear telemetry buffer."""
    TELEMETRY_BUFFER.clear()
    return {"success": True, "message": "Telemetry cleared"}

@app.get("/canaries/check/{canary_token}")
async def check_canary(canary_token: str):
    """Check if a canary token exists in our system."""
    exists = canary_token in CANARY_TOKENS
    return {"exists": exists, "token": canary_token}

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AEGIS‑C Honeynet",
        "version": "1.0.0",
        "description": "Fake APIs with telemetry and canary tokens",
        "endpoints": {
            "datasets": "/api/datasets",
            "models": "/api/models",
            "canaries": "/canaries",
            "telemetry": "/telemetry",
            "health": "/health",
            "metrics": "/metrics",
            "secure_ping": "/secure/ping"
        },
        "datasets": len(HONEYPOT_DATA),
        "canaries": len(CANARY_TOKENS),
        "telemetry_events": len(TELEMETRY_BUFFER)
    }

# Protected endpoints
@app.post("/secure/canaries/generate", dependencies=[Depends(require_api_key)])
async def secure_generate_canary():
    """Protected canary generation endpoint"""
    return await generate_canary()

@app.get("/secure/telemetry", dependencies=[Depends(require_api_key)])
async def secure_get_telemetry():
    """Protected telemetry endpoint"""
    return await get_telemetry()

@app.delete("/secure/telemetry", dependencies=[Depends(require_api_key)])
async def secure_clear_telemetry():
    """Protected telemetry clearing endpoint"""
    return await clear_telemetry()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)