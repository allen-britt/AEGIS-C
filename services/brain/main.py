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
import aiohttp
import json
from datetime import datetime, timezone
import sys
import os

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="AEGIS-C Brain", version="1.0.0")

# Import risk scorer from risk.py using explicit path
from risk import calculate_adaptive_risk, RiskAssessment

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
    cve_impacts: List[str] = Field(default_factory=list, description="CVE impact areas if relevant")

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
VULN_DB_URL = "http://localhost:8019"  # URL for vuln_db service
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

# ---- CVE Data Fetching ----
async def fetch_critical_cves_from_vuln_db() -> List[Dict[str, Any]]:
    """Fetch critical CVEs from vuln_db service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{VULN_DB_URL}/threats/critical") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("critical_threats", [])
                else:
                    print(f"[{datetime.now().isoformat()}] Failed to fetch CVEs: Status {response.status}")
                    return []
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Error fetching CVEs from vuln_db: {str(e)}")
        return []

# ---- Risk Assessment ----
async def _risk(signals: List[Signal]) -> RiskResponse:
    """Calculate risk using the adaptive risk scorer"""
    start_time = time.time()
    
    # Convert signals to event dictionary for risk scorer
    event = {s.name: s.value for s in signals}
    
    # Fetch critical CVEs from vuln_db service
    critical_cves = await fetch_critical_cves_from_vuln_db()
    
    # Calculate risk using the adaptive scorer from risk.py
    assessment = calculate_adaptive_risk(event, critical_cves)
    
    # Prepare response
    top_features = [
        {"feature": f[0], "contribution": f[1]}
        for f in assessment.top_features
    ]
    
    processing_time = (time.time() - start_time) * 1000  # ms
    
    return RiskResponse(
        probability=assessment.probability,
        level=assessment.risk_level,
        top_features=top_features,
        timestamp=assessment.timestamp,
        cve_impacts=assessment.cve_impacts if assessment.cve_impacts else []
    )

@app.post("/risk", response_model=RiskResponse)
async def assess_risk(request: RiskRequest):
    """Assess risk for a given subject with signals"""
    try:
        if not request.signals:
            raise HTTPException(status_code=400, detail="At least one signal required")
        
        result = await _risk(request.signals)
        
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