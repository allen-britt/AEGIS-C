from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AEGIS‑C Detector Service")

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

# --- Basic Detection Logic ---
def detect_ai_text(text: str) -> tuple[float, str, dict]:
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
    
    return score, verdict, signals

def detect_ai_agent(request: Request, payload: AgentPayload) -> tuple[float, str, dict]:
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
    
    return score, verdict, signals

# --- API Endpoints ---
@app.post("/detect/text")
async def detect_text(payload: TextPayload):
    """Detect AI-generated text"""
    try:
        score, verdict, signals = detect_ai_text(payload.text)
        logger.info(f"Text detection: score={score}, verdict={verdict}")
        return DetectionResult(score=round(score, 3), verdict=verdict, signals=signals)
    except Exception as e:
        logger.error(f"Text detection error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")

@app.post("/detect/agent")
async def detect_agent(request: Request, payload: AgentPayload):
    """Detect AI/bot agents"""
    try:
        score, verdict, signals = detect_ai_agent(request, payload)
        logger.info(f"Agent detection: score={score}, verdict={verdict}")
        return AgentResult(score=round(score, 3), verdict=verdict, signals=signals)
    except Exception as e:
        logger.error(f"Agent detection error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")

# --- Health and Info Endpoints ---
@app.get("/health")
async def health():
    """Basic health check"""
    return {"ok": True, "service": "detector", "version": "1.0.0"}

@app.get("/metrics")
async def metrics():
    """Simple metrics endpoint"""
    return {
        "http_requests_total": 0,
        "service_health": 1,
        "uptime_seconds": time.time()
    }

@app.get("/info")
async def info():
    """Service information"""
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

@app.get("/")
async def root():
    """Root endpoint"""
    return await info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
