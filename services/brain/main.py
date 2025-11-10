#!/usr/bin/env python3
"""
AEGIS‑C Brain Gateway
======================

Clean, reliable API for risk assessment and policy decisions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import math
import time
from datetime import datetime, timezone

app = FastAPI(title="AEGIS-C Brain", version="1.0.0")

# ---- Schemas ----
class Signal(BaseModel):
    name: str = Field(..., description="Signal name")
    value: float = Field(..., description="Signal value (normalized)")

class RiskRequest(BaseModel):
    subject: str = Field(..., description="Subject identifier (e.g., 'session:abc123', 'node:gpu0')")
    kind: str = Field(..., description="Subject kind: 'agent', 'rag', 'hardware', 'artifact'")
    signals: List[Signal] = Field(..., description="List of normalized signals [0..1 or z-scores]")
    context: Dict[str, str] = Field(default_factory=dict, description="Additional context")

class RiskResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability 0..1")
    level: str = Field(..., description="Risk level: 'info', 'warn', 'high', 'critical'")
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")
    timestamp: str = Field(..., description="Assessment timestamp")

class PolicyRequest(BaseModel):
    subject: str = Field(..., description="Subject identifier")
    risk: float = Field(..., ge=0.0, le=1.0, description="Risk probability from /risk endpoint")
    kind: str = Field(..., description="Subject kind")
    options: List[str] = Field(..., description="Available policy options")

class PolicyResponse(BaseModel):
    action: str = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Action confidence")
    reason: str = Field(..., description="Reasoning for action selection")
    timestamp: str = Field(..., description="Decision timestamp")

# ---- Configuration ----
FEATURE_WEIGHTS = {
    "ai_text_score": 1.2,
    "probe_sim": 1.5, 
    "canary_echo": 2.0,
    "rag_injection": 1.7,
    "ecc_delta": 1.6,
    "latency_ms": 0.8,
    "agent_anomaly": 1.1,
    "data_poison_risk": 1.4,
    "threat_intel_score": 1.3,
    "hardware_temp": 0.9
}

# ---- Risk Assessment ----
def _risk(signals: List[Signal]) -> RiskResponse:
    """Calculate risk using linear model + logistic function"""
    start_time = time.time()
    
    # Linear combination with feature weights
    z = sum(FEATURE_WEIGHTS.get(s.name, 0.5) * s.value for s in signals) - 3.0
    
    # Logistic function
    probability = 1 / (1 + math.exp(-z))
    probability = max(0.0, min(1.0, probability))  # Clamp to [0,1]
    
    # Risk level classification
    if probability < 0.2:
        level = "info"
    elif probability < 0.5:
        level = "warn"
    elif probability < 0.75:
        level = "high"
    else:
        level = "critical"
    
    # Top contributing features
    contributions = [
        {"feature": s.name, "contribution": FEATURE_WEIGHTS.get(s.name, 0.5) * s.value}
        for s in signals
    ]
    top_features = sorted(contributions, key=lambda x: -abs(x["contribution"]))[:3]
    
    # Round probability for cleaner output
    probability = round(probability, 3)
    
    processing_time = (time.time() - start_time) * 1000  # ms
    
    return RiskResponse(
        probability=probability,
        level=level,
        top_features=top_features,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post("/risk", response_model=RiskResponse)
def assess_risk(request: RiskRequest):
    """Assess risk for a given subject with signals"""
    try:
        if not request.signals:
            raise HTTPException(status_code=400, detail="At least one signal required")
        
        result = _risk(request.signals)
        
        print(f"[{datetime.now().isoformat()}] Risk assessment: {request.subject} -> {result.probability} ({result.level})")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# ---- Policy Decision ----
def _policy(subject: str, risk: float, kind: str, options: List[str]) -> PolicyResponse:
    """Simple rule-based policy decision (placeholder for bandit)"""
    
    # Default action
    action = "observe"
    reason = "Low risk default"
    confidence = 0.6
    
    # Risk-based decision logic
    if risk >= 0.85:
        if "quarantine_host" in options:
            action, reason, confidence = "quarantine_host", "Critical risk", 0.9
        elif "drain_node" in options:
            action, reason, confidence = "drain_node", "Critical risk", 0.9
    elif risk >= 0.7:
        if kind == "hardware" and "reset_gpu" in options:
            action, reason, confidence = "reset_gpu", "Hardware risk high", 0.8
        elif "drain_node" in options:
            action, reason, confidence = "drain_node", "High risk", 0.8
        elif "raise_friction" in options:
            action, reason, confidence = "raise_friction", "High risk", 0.8
    elif risk >= 0.5:
        if "raise_friction" in options:
            action, reason, confidence = "raise_friction", "Medium risk", 0.75
        elif "increase_monitoring" in options:
            action, reason, confidence = "increase_monitoring", "Medium risk", 0.7
    
    # Ensure chosen action is in options
    if action not in options:
        action = options[0] if options else "observe"
        reason = f"Preferred action not available, using: {action}"
        confidence = 0.5
    
    return PolicyResponse(
        action=action,
        confidence=confidence,
        reason=reason,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post("/policy", response_model=PolicyResponse)
def decide_policy(request: PolicyRequest):
    """Decide policy action based on risk and options"""
    try:
        if not request.options:
            raise HTTPException(status_code=400, detail="At least one policy option required")
        
        result = _policy(request.subject, request.risk, request.kind, request.options)
        
        print(f"[{datetime.now().isoformat()}] Policy decision: {request.subject} (risk={request.risk}) -> {result.action}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy decision failed: {str(e)}")

# ---- Health Check ----
@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
def root():
    """Root endpoint with service info"""
    return {
        "service": "AEGIS-C Brain Gateway",
        "description": "Risk assessment and policy decision service",
        "endpoints": {
            "/risk": "POST - Assess risk from signals",
            "/policy": "POST - Decide policy action from risk",
            "/health": "GET - Health check"
        },
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)