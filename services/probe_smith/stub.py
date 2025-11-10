#!/usr/bin/env python3
"""
AEGIS‑C Probe Smith Stub
=========================

Deterministic stub for probe generation API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime, timezone
import hashlib

app = FastAPI(title="AEGIS-C Probe Smith Stub", version="1.0.0")

class NearMiss(BaseModel):
    prompt: str = Field(..., description="Original prompt")
    true_family: str = Field(..., description="Correct model family")
    guessed: str = Field(..., description="Guessed model family")

class ProbeRequest(BaseModel):
    near_misses: List[NearMiss] = Field(..., description="List of near miss events")

class Probe(BaseModel):
    id: str = Field(..., description="Probe identifier")
    prompt: str = Field(..., description="Probe prompt")
    key: str = Field(..., description="Expected response key")

class ProbeResponse(BaseModel):
    probes: List[Probe] = Field(..., description="Generated probes")
    count: int = Field(..., description="Number of probes generated")
    timestamp: str = Field(..., description="Generation timestamp")

def _generate_deterministic_probes(near_misses: List[NearMiss]) -> List[Probe]:
    """Generate deterministic probes based on near misses"""
    probes = []
    
    # Use near miss content to generate deterministic but varied probes
    for i, miss in enumerate(near_misses[:5]):  # Limit to 5 for stub
        # Create deterministic hash for consistency
        content_hash = hashlib.md5(f"{miss.prompt}_{miss.true_family}".encode()).hexdigest()[:8]
        
        # Generate probe based on miss characteristics
        if miss.guessed != miss.true_family:
            # Family misclassification - generate strict format probe
            probe = Probe(
                id=f"format_{content_hash}_{i}",
                prompt=f"Return JSON exactly: {{\"family\":\"{miss.true_family}\",\"id\":\"{content_hash}\"}}",
                key=f"{miss.true_family}"
            )
        else:
            # Other failure - generate rare word probe
            probe = Probe(
                id=f"rare_{content_hash}_{i}",
                prompt="Respond with this word only: floccinaucinihilipilification",
                key="floccinaucinihilipilification"
            )
        
        probes.append(probe)
    
    # Add some standard probes if we have fewer than 3
    while len(probes) < 3:
        probes.append(Probe(
            id=f"standard_{len(probes)}",
            prompt="What is 137 × 149? Number only.",
            key="20413"
        ))
    
    return probes[:10]  # Return max 10 probes

@app.post("/propose", response_model=ProbeResponse)
def propose_probes(request: ProbeRequest):
    """Generate new probes based on near misses"""
    try:
        if not request.near_misses:
            raise HTTPException(status_code=400, detail="At least one near miss required")
        
        probes = _generate_deterministic_probes(request.near_misses)
        
        return ProbeResponse(
            probes=probes,
            count=len(probes),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Probe generation failed: {str(e)}")

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "probe_smith_stub"
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "AEGIS-C Probe Smith Stub",
        "description": "Deterministic probe generation for testing",
        "endpoints": {
            "/propose": "POST - Generate probes from near misses",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)