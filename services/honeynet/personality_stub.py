#!/usr/bin/env python3
"""
AEGIS‑C Honeynet Personality Stub
==================================

Deterministic stub for adaptive honeynet personality.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import hashlib

app = FastAPI(title="AEGIS-C Honeynet Personality Stub", version="1.0.0")

class SessionTraits(BaseModel):
    retry_rate: float = Field(..., description="Request retry rate per second")
    ua_len: int = Field(..., description="User agent string length")
    cookie: bool = Field(..., description="Client supports cookies")
    depth: int = Field(default=1, description="Session depth (pages visited)")
    js_support: bool = Field(default=False, description="JavaScript support detected")
    referer: bool = Field(default=False, description="Referer header present")

class PersonalityRequest(BaseModel):
    session: str = Field(..., description="Session identifier")
    traits: SessionTraits = Field(..., description="Observed session traits")

class PersonalityResponse(BaseModel):
    template: str = Field(..., description="Response template to use")
    canary: str = Field(..., description="Canary token to deploy")
    throttle_ms: int = Field(..., description="Response throttle in milliseconds")
    reasoning: str = Field(..., description="Reasoning for personality choice")
    timestamp: str = Field(..., description="Decision timestamp")

# Personality mapping based on traits
PERSONALITY_MAP = {
    # Fast bot characteristics
    "fast_bot": {
        "template": "expensive_route",
        "canary": "CNRY-BOT-001",
        "throttle_ms": 2000,
        "reasoning": "High retry rate, simple UA, no cookies - likely automated bot"
    },
    # Browser mimic characteristics  
    "browser_mimic": {
        "template": "dom_puzzle",
        "canary": "CNRY-BR-002", 
        "throttle_ms": 500,
        "reasoning": "Complex UA, cookies present - browser mimic"
    },
    # API client characteristics
    "api_client": {
        "template": "canary_dense",
        "canary": "CNRY-API-003",
        "throttle_ms": 1000,
        "reasoning": "Medium UA, no JS - API client"
    },
    # Stealthy characteristics
    "stealthy": {
        "template": "fake_error",
        "canary": "CNRY-STL-004",
        "throttle_ms": 800,
        "reasoning": "Low retry rate, full browser - stealthy actor"
    },
    # Default characteristics
    "unknown": {
        "template": "honeypot_data",
        "canary": "CNRY-DEF-005",
        "throttle_ms": 1000,
        "reasoning": "Unknown pattern - default personality"
    }
}

def _classify_personality(traits: SessionTraits) -> str:
    """Classify personality based on traits"""
    
    # Fast bot: high retry rate, simple UA, no cookies
    if traits.retry_rate > 5.0 and traits.ua_len < 50 and not traits.cookie:
        return "fast_bot"
    
    # Browser mimic: complex UA, cookies, JS support
    elif traits.ua_len > 100 and traits.cookie and traits.js_support:
        return "browser_mimic"
    
    # API client: medium UA, no JS, moderate retry
    elif 50 <= traits.ua_len <= 100 and not traits.js_support and traits.retry_rate < 3.0:
        return "api_client"
    
    # Stealthy: low retry rate, full browser features
    elif traits.retry_rate < 1.0 and traits.cookie and traits.ua_len > 80:
        return "stealthy"
    
    # Default
    else:
        return "unknown"

def _generate_session_canary(session: str, personality: str) -> str:
    """Generate deterministic canary for session"""
    # Use session hash for consistency
    session_hash = hashlib.md5(f"{session}_{personality}".encode()).hexdigest()[:8]
    base_canary = PERSONALITY_MAP[personality]["canary"]
    return f"{base_canary}-{session_hash}"

@app.post("/policy", response_model=PersonalityResponse)
def get_personality(request: PersonalityRequest):
    """Get adaptive personality for session based on traits"""
    try:
        # Classify personality
        personality = _classify_personality(request.traits)
        config = PERSONALITY_MAP[personality]
        
        # Generate session-specific canary
        canary = _generate_session_canary(request.session, personality)
        
        # Adjust throttle based on depth (deeper sessions get more throttling)
        throttle_adjustment = min(request.traits.depth * 100, 1000)
        final_throttle = config["throttle_ms"] + throttle_adjustment
        
        return PersonalityResponse(
            template=config["template"],
            canary=canary,
            throttle_ms=final_throttle,
            reasoning=config["reasoning"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Personality selection failed: {str(e)}")

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "honeynet_personality_stub"
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "AEGIS-C Honeynet Personality Stub",
        "description": "Deterministic adaptive honeynet personality for testing",
        "personalities": list(PERSONALITY_MAP.keys()),
        "endpoints": {
            "/policy": "POST - Get personality based on session traits",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8023)