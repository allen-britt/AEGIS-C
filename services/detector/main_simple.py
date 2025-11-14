"""
AEGIS‑C Detector Service (Simple)
==================================

AI-generated text and agent detection with standardized guard.
"""

import os
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
import time
import sys

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import setup_service_guard, record_detection_score, logger, require_api_token

# Service Configuration
SERVICE_NAME = "detector"
SERVICE_PORT = int(os.getenv("DETECTOR_PORT", "8010"))

# Initialize FastAPI App
app = FastAPI(title="AEGIS‑C Detector Service")

# Setup standardized guard
setup_service_guard(app, SERVICE_NAME)

# Models
class TextPayload(BaseModel):
    text: str

class DetectionResult(BaseModel):
    score: float
    verdict: str
    signals: dict

class AgentPayload(BaseModel):
    user_agent: str = ""
    cookie: str = ""
    accept_lang: str = ""
    referer: str = ""

class AgentResult(BaseModel):
    score: float
    verdict: str
    signals: dict

# Detection Logic
def detect_ai_text(text: str) -> DetectionResult:
    """Simple AI text detection heuristics"""
    signals = {}
    
    # Basic signals
    length = len(text)
    word_count = len(text.split())
    avg_word_len = sum(len(word) for word in text.split()) / max(word_count, 1)
    
    signals.update({
        "char_count": length,
        "word_count": word_count,
        "avg_word_length": round(avg_word_len, 2)
    })
    
    # Simple scoring
    score = 0.0
    if length > 1000: score += 0.2
    if avg_word_len > 6: score += 0.2
    if "however" in text.lower(): score += 0.1
    if "therefore" in text.lower(): score += 0.1
    if "furthermore" in text.lower(): score += 0.1
    
    score = min(1.0, score)
    
    if score >= 0.7:
        verdict = "likely_ai"
    elif score >= 0.4:
        verdict = "uncertain"
    else:
        verdict = "likely_human"
    
    # Record metrics
    record_detection_score(score, verdict)
    
    return DetectionResult(score=round(score, 3), verdict=verdict, signals=signals)

def detect_ai_agent(request: Request, payload: AgentPayload) -> AgentResult:
    """Simple AI agent detection"""
    ua = payload.user_agent
    cookie = payload.cookie
    accept_lang = payload.accept_lang
    referer = payload.referer
    
    # Basic signals
    signals = {
        "ua_length": len(ua),
        "has_cookie": bool(cookie),
        "has_accept_language": bool(accept_lang),
        "has_referer": bool(referer)
    }
    
    # Simple scoring
    score = 0.0
    if len(ua) < 30: score += 0.3
    if not cookie: score += 0.2
    if not accept_lang: score += 0.1
    if not referer: score += 0.1
    
    score = min(1.0, score)
    
    if score >= 0.7:
        verdict = "likely_bot"
    elif score >= 0.4:
        verdict = "uncertain"
    else:
        verdict = "likely_human"
    
    # Record metrics
    record_detection_score(score, verdict)
    
    return AgentResult(score=round(score, 3), verdict=verdict, signals=signals)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AEGIS‑C Detector",
        "version": "1.0.0",
        "description": "AI-generated text and agent detection service",
        "endpoints": {
            "detect_text": "/detect/text",
            "detect_agent": "/detect/agent",
            "health": "/health",
            "metrics": "/metrics",
            "info": "/info"
        }
    }

@app.post("/detect/text")
async def detect_text(payload: TextPayload):
    """Detect AI-generated text"""
    try:
        result = detect_ai_text(payload.text)
        logger.info(f"Text detection: score={result.score}, verdict={result.verdict}")
        return result
    except Exception as e:
        logger.error(f"Text detection error: {e}")
        raise

@app.post("/detect/agent")
async def detect_agent(request: Request, payload: AgentPayload):
    """Detect AI/bot agents"""
    try:
        result = detect_ai_agent(request, payload)
        logger.info(f"Agent detection: score={result.score}, verdict={result.verdict}")
        return result
    except Exception as e:
        logger.error(f"Agent detection error: {e}")
        raise

@app.get("/info")
async def info():
    """Service information"""
    return await root()

# Protected endpoints
@app.post("/secure/detect/text", dependencies=[Depends(require_api_token)])
async def secure_detect_text(payload: TextPayload):
    """Protected text detection"""
    return await detect_text(payload)

@app.post("/secure/detect/agent", dependencies=[Depends(require_api_token)])
async def secure_detect_agent(request: Request, payload: AgentPayload):
    """Protected agent detection"""
    return await detect_agent(request, payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
