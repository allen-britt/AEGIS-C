import os
import sys
from fastapi import FastAPI, Request, Depends, HTTPException
from pydantic import BaseModel
import time, math
import structlog

# Import shared health metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from health_metrics import setup_health_metrics, MetricsMiddleware, protected_endpoint

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

app = FastAPI(title="AEGIS‑C Detector Service")

# Setup health metrics and middleware
setup_health_metrics(app)
app.add_middleware(MetricsMiddleware)

class TextPayload(BaseModel):
    text: str

class DetectionResult(BaseModel):
    score: float
    verdict: str
    signals: dict

# --- Heuristics ---
COMMON_PHRASES = [
    "as an ai language model", "i cannot provide", "it is important to note",
    "however,", "moreover,", "furthermore,", "in conclusion", "additionally,",
    "therefore,", "consequently,", "as a result,", "please note",
    "i'm sorry, but", "i apologize, but", "regrettably,"
]

def phrase_score(t: str) -> float:
    """Detect common AI disclaimer phrases."""
    t_low = t.lower()
    hits = sum(1 for p in COMMON_PHRASES if p in t_low)
    return min(1.0, hits / 3.0)

def token_uniformity(t: str) -> float:
    """LLM outputs often have uniform token lengths."""
    toks = [len(x) for x in t.split() if x]
    if len(toks) < 8:
        return 0.0
    avg = sum(toks)/len(toks)
    var = sum((x-avg)**2 for x in toks)/len(toks)
    std = math.sqrt(var)
    # lower std => more uniform => higher AI‑likeness
    return max(0.0, min(1.0, (5.0 - min(5.0, std)) / 5.0))

def sentence_complexity(t: str) -> float:
    """AI often uses consistent sentence structures."""
    sentences = [s.strip() for s in t.split('.') if s.strip()]
    if len(sentences) < 3:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths)/len(lengths)
    var = sum((x-avg)**2 for x in lengths)/len(lengths)
    std = math.sqrt(var)
    # Low variance in sentence length suggests AI generation
    return max(0.0, min(1.0, (10.0 - min(10.0, std)) / 10.0))

def formality_score(t: str) -> float:
    """AI tends to be overly formal."""
    formal_indicators = ["furthermore", "moreover", "consequently", "therefore", "nevertheless"]
    informal_indicators = ["gonna", "wanna", "yeah", "nah", "cool", "awesome", "btw"]
    
    t_low = t.lower()
    formal_count = sum(1 for f in formal_indicators if f in t_low)
    informal_count = sum(1 for i in informal_indicators if i in t_low)
    
    if formal_count + informal_count == 0:
        return 0.5  # neutral
    
    return formal_count / (formal_count + informal_count)

@app.post("/detect/text", response_model=DetectionResult)
async def detect_text(p: TextPayload):
    """Detect AI-generated text using multiple heuristics."""
    s1 = phrase_score(p.text)
    s2 = token_uniformity(p.text)
    s3 = sentence_complexity(p.text)
    s4 = formality_score(p.text)
    
    # Weighted combination
    score = round(0.3*s2 + 0.25*s1 + 0.25*s3 + 0.2*s4, 3)
    
    if score >= 0.7:
        verdict = "likely_ai"
    elif score >= 0.5:
        verdict = "uncertain"
    else:
        verdict = "likely_human"
    
    signals = {
        "phrase": round(s1, 3),
        "uniformity": round(s2, 3),
        "complexity": round(s3, 3),
        "formality": round(s4, 3)
    }
    
    logger.info(f"Text detection: score={score}, verdict={verdict}, length={len(p.text)}")
    
    return DetectionResult(score=score, verdict=verdict, signals=signals)

class AgentResult(BaseModel):
    score: float
    verdict: str
    signals: dict

@app.post("/detect/agent", response_model=AgentResult)
async def detect_agent(request: Request):
    """Detect bot/agent behavior via HTTP request analysis."""
    t0 = time.time()
    body = await request.body()
    t1 = time.time()
    
    think_ms = int((t1 - t0)*1000)
    headers = dict(request.headers)
    ua = headers.get("user-agent", "")
    cookie = headers.get("cookie", "")
    accept_lang = headers.get("accept-language", "")
    referer = headers.get("referer", "")
    
    # Analyze User-Agent
    ua_indicators = {
        "bot": ["bot", "crawler", "spider", "scraper"],
        "automated": ["python", "curl", "wget", "http"],
        "browser": ["mozilla", "chrome", "firefox", "safari", "edge"]
    }
    
    ua_score = 0.0
    ua_low = ua.lower()
    for category, indicators in ua_indicators.items():
        if any(ind in ua_low for ind in indicators):
            if category == "bot":
                ua_score += 0.5
            elif category == "automated":
                ua_score += 0.3
            elif category == "browser":
                ua_score -= 0.1
    
    # Header analysis
    header_count = len(headers)
    std_headers = {"host", "user-agent", "accept", "content-type", "content-length"}
    missing_std = len(std_headers - set(h.lower() for h in headers.keys()))
    
    signals = {
        "think_ms": think_ms,
        "user_agent_len": len(ua),
        "has_cookie": bool(cookie),
        "header_count": header_count,
        "missing_std_headers": missing_std,
        "ua_score": round(ua_score, 3),
        "has_accept_language": bool(accept_lang),
        "has_referer": bool(referer)
    }
    
    # Scoring
    score = 0.0
    if think_ms < 5: score += 0.3
    if len(ua) < 30: score += 0.2
    if not cookie: score += 0.2
    if ua_score > 0.3: score += 0.3
    if missing_std_headers > 1: score += 0.2
    if not accept_lang: score += 0.1
    if not referer: score += 0.1
    
    score = min(1.0, score)
    
    if score >= 0.7:
        verdict = "likely_bot"
    elif score >= 0.4:
        verdict = "uncertain"
    else:
        verdict = "likely_human"
    
    logger.info(f"Agent detection: score={score}, verdict={verdict}, ip={request.client.host if request.client else 'unknown'}")
    
    return AgentResult(score=round(score, 3), verdict=verdict, signals=signals)

# Protected endpoints
@app.get("/secure/ping", dependencies=[Depends(protected_endpoint())])
async def secure_ping():
    """Protected ping endpoint for authentication testing."""
    return {"pong": True, "service": "detector"}

@app.get("/secure/detect", dependencies=[Depends(protected_endpoint())])
async def secure_detection():
    """Protected detection endpoint."""
    return {"status": "protected", "service": "detector"}

# Health endpoints are now handled by setup_health_metrics()
# But we'll keep the basic one for compatibility
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True, "service": "detector"}

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AEGIS‑C Detector",
        "version": "1.0.0",
        "endpoints": {
            "detect_text": "/detect/text",
            "detect_agent": "/detect/agent",
            "health": "/health",
            "secure/ping": "/secure/ping (protected)",
            "metrics": "/metrics"
        }
    }