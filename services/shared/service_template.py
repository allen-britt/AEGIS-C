"""
AEGIS‑C Service Template
=========================

Template for creating standardized FastAPI services with auth, metrics, and logging.
"""

import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from service_guard import setup_service_guard, record_detection_score, logger

# Service Configuration
SERVICE_NAME = os.getenv("SERVICE_NAME", "template")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8000"))

# Initialize FastAPI App
app = FastAPI(title=f"AEGIS‑C {SERVICE_NAME.title()} Service")

# Setup standardized guard (adds /health, /metrics, /secure/ping, and middleware)
setup_service_guard(app, SERVICE_NAME)

# Example Models
class TextInput(BaseModel):
    text: str

class DetectionResult(BaseModel):
    score: float
    verdict: str
    confidence: float

# Example Business Logic
def detect_something(text: str) -> DetectionResult:
    """Example detection logic - replace with actual implementation"""
    # Simple mock logic
    score = 0.5 if "test" in text.lower() else 0.1
    verdict = "suspicious" if score > 0.4 else "benign"
    confidence = score
    
    # Record metrics
    record_detection_score(score, verdict)
    
    return DetectionResult(score=score, verdict=verdict, confidence=confidence)

# Service Endpoints
@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": SERVICE_NAME,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics", 
            "secure_ping": "/secure/ping",
            "detect": "/detect"
        }
    }

@app.post("/detect")
async def detect_endpoint(payload: TextInput):
    """Main detection endpoint - replace with actual implementation"""
    try:
        result = detect_something(payload.text)
        
        logger.info(
            "Detection completed",
            service=SERVICE_NAME,
            verdict=result.verdict,
            score=result.score
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Detection failed",
            service=SERVICE_NAME,
            error=str(e)
        )
        raise

# Protected Endpoint Example
@app.post("/secure/detect", dependencies=[Depends(require_api_key)])
async def secure_detect_endpoint(payload: TextInput):
    """Protected detection endpoint requiring API key"""
    return await detect_endpoint(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)