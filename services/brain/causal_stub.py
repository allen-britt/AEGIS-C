#!/usr/bin/env python3
"""
AEGIS‑C Causal Explainer Stub
==============================

Deterministic stub for causal incident analysis.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import hashlib

app = FastAPI(title="AEGIS-C Causal Explainer Stub", version="1.0.0")

class CausalEvent(BaseModel):
    t: str = Field(..., description="Event timestamp")
    type: str = Field(..., description="Event type")
    details: Dict[str, Any] = Field(default_factory=dict, description="Event details")

class CausalRequest(BaseModel):
    events: List[CausalEvent] = Field(..., description="Timeline of events to analyze")

class CausalResponse(BaseModel):
    root_cause: str = Field(..., description="Identified root cause")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    remedy: str = Field(..., description="Recommended remedy")
    reasoning: str = Field(..., description="Analysis reasoning")
    timestamp: str = Field(..., description="Analysis timestamp")

# Causal pattern mapping
CAUSAL_PATTERNS = {
    "retrieve_then_inject": {
        "sequence": ["retrieve", "detect"],
        "detect_details": {"flag": "rag_injection"},
        "root_cause": "new_domain_in_topk",
        "confidence": 0.68,
        "remedy": "de-rank+admission_review",
        "reasoning": "Content retrieval followed by injection detection indicates compromised source"
    },
    "hardware_then_model": {
        "sequence": ["hardware_anomaly", "model_drift"],
        "root_cause": "gpu_memory_corruption",
        "confidence": 0.82,
        "remedy": "drain_node+memory_test",
        "reasoning": "Hardware anomaly preceding model drift suggests memory corruption"
    },
    "policy_bypass": {
        "sequence": ["policy_check", "bypass_attempt"],
        "root_cause": "privilege_escalation",
        "confidence": 0.75,
        "remedy": "audit_access+reinforce_policies",
        "reasoning": "Policy check followed by bypass attempt indicates privilege escalation"
    },
    "canary_trigger": {
        "sequence": ["canary_access", "exfiltration"],
        "root_cause": "honeynet_breach",
        "confidence": 0.91,
        "remedy": "isolate_network+forensics",
        "reasoning": "Canary access followed by data exfiltration confirms honeynet breach"
    },
    "unknown_pattern": {
        "sequence": [],
        "root_cause": "unknown_causal_pattern",
        "confidence": 0.3,
        "remedy": "manual_investigation",
        "reasoning": "No recognized causal pattern detected"
    }
}

def _analyze_causal_pattern(events: List[CausalEvent]) -> Dict[str, Any]:
    """Analyze events for known causal patterns"""
    
    if len(events) < 2:
        return CAUSAL_PATTERNS["unknown_pattern"]
    
    # Extract event sequence and details
    event_sequence = [event.type for event in events]
    event_details = {event.type: event.details for event in events}
    
    # Check each known pattern
    for pattern_name, pattern_config in CAUSAL_PATTERNS.items():
        if pattern_name == "unknown_pattern":
            continue
            
        required_sequence = pattern_config["sequence"]
        
        # Check if sequence matches (allowing for other events in between)
        if _sequence_matches(event_sequence, required_sequence):
            # Check for required details if specified
            details_match = True
            if "detect_details" in pattern_config:
                required_details = pattern_config["detect_details"]
                actual_details = event_details.get("detect", {})
                
                for key, expected_value in required_details.items():
                    if actual_details.get(key) != expected_value:
                        details_match = False
                        break
            
            if details_match:
                # Add some deterministic variation based on event content
                confidence = pattern_config["confidence"]
                
                # Vary confidence slightly based on event count
                if len(events) > 3:
                    confidence = min(0.95, confidence + 0.05)
                elif len(events) < 2:
                    confidence = max(0.2, confidence - 0.1)
                
                result = pattern_config.copy()
                result["confidence"] = round(confidence, 2)
                return result
    
    # No pattern matched
    return CAUSAL_PATTERNS["unknown_pattern"]

def _sequence_matches(actual: List[str], required: List[str]) -> bool:
    """Check if required sequence exists in actual sequence"""
    if len(required) > len(actual):
        return False
    
    # Look for required sequence as subsequence
    for i in range(len(actual) - len(required) + 1):
        if actual[i:i+len(required)] == required:
            return True
    
    return False

def _generate_deterministic_id(events: List[CausalEvent]) -> str:
    """Generate deterministic analysis ID"""
    event_summary = "|".join(f"{e.type}:{e.t}" for e in events)
    return hashlib.md5(event_summary.encode()).hexdigest()[:8]

@app.post("/causal", response_model=CausalResponse)
def analyze_causality(request: CausalRequest):
    """Analyze causal relationships in event timeline"""
    try:
        if not request.events:
            raise HTTPException(status_code=400, detail="At least one event required")
        
        # Sort events by timestamp
        sorted_events = sorted(request.events, key=lambda e: e.t)
        
        # Analyze pattern
        analysis = _analyze_causal_pattern(sorted_events)
        
        # Generate analysis ID
        analysis_id = _generate_deterministic_id(sorted_events)
        
        return CausalResponse(
            root_cause=analysis["root_cause"],
            confidence=analysis["confidence"],
            remedy=analysis["remedy"],
            reasoning=analysis["reasoning"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Causal analysis failed: {str(e)}")

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "causal_explainer_stub"
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "AEGIS-C Causal Explainer Stub",
        "description": "Deterministic causal incident analysis for testing",
        "patterns": list(CAUSAL_PATTERNS.keys()),
        "endpoints": {
            "/causal": "POST - Analyze causal relationships",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)