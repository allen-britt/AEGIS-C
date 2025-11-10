#!/usr/bin/env python3
"""
AEGIS‑C Probe Smith Service
============================

Automated probe generation and evolution for model fingerprinting.
"""

import os
import sys
import json
import asyncio
import uvicorn
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import (
    create_app, setup_service_guard, logger, 
    record_detection_score, get_metrics
)

# Import probe smith components
from generate import generate_new_probes, evaluate_and_select_probes, get_probe_smith_status

# Initialize FastAPI app with service guard
app = create_app("AEGIS‑C Probe Smith", "1.0.0")
security = HTTPBearer()
guard = setup_service_guard(app, "/probe_smith")

# Request/Response Models
class NearMissRequest(BaseModel):
    near_misses: List[Dict[str, Any]] = Field(..., description="List of near miss events")
    count: int = Field(default=10, ge=1, le=50, description="Number of probes to generate")

class ProbeEvaluationRequest(BaseModel):
    probes: List[Dict[str, Any]] = Field(..., description="List of probes to evaluate")
    test_results: Dict[str, Any] = Field(..., description="Test results for probes")
    top_k: int = Field(default=10, ge=1, le=20, description="Number of best probes to select")

class ProbeRequest(BaseModel):
    probe_id: str = Field(..., description="Probe identifier")
    prompt: str = Field(..., description="Probe prompt")
    expected_key: str = Field(..., description="Expected response key")

# API Endpoints
@app.post("/secure/generate")
async def generate_probes(
    request: NearMissRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Generate new probes based on near misses.
    
    Analyzes recent detection failures and generates fresh probes
    targeting the specific failure modes to improve model separation.
    """
    try:
        # Generate new probes
        new_probes = generate_new_probes(request.near_misses, request.count)
        
        logger.info(
            "Probes generated",
            count=len(new_probes),
            near_misses=len(request.near_misses)
        )
        
        return {
            "probes": new_probes,
            "generated_count": len(new_probes),
            "generation_method": "adaptive_near_miss_analysis",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Probe generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/secure/evaluate")
async def evaluate_probes(
    request: ProbeEvaluationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Evaluate probe performance and select best ones.
    
    Tests probes against model families and selects those that
    provide the best separation (highest AUC).
    """
    try:
        # Evaluate and select best probes
        best_probes = evaluate_and_select_probes(
            request.probes, 
            request.test_results, 
            request.top_k
        )
        
        logger.info(
            "Probes evaluated and selected",
            total_probes=len(request.probes),
            selected=len(best_probes),
            top_k=request.top_k
        )
        
        return {
            "best_probes": best_probes,
            "selected_count": len(best_probes),
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Probe evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/secure/status")
async def get_probe_smith_status_endpoint(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get probe smith generation and evaluation status"""
    try:
        status = get_probe_smith_status()
        
        return {
            "service": "probe_smith",
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_statistics": status
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/secure/evolution/report")
async def get_evolution_report(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get probe evolution report.
    
    Shows how probes have evolved over time and their effectiveness
    in separating model families.
    """
    try:
        status = get_probe_smith_status()
        
        # Generate evolution insights
        insights = []
        
        total_probes = status.get("total_probes_generated", 0)
        if total_probes > 100:
            insights.append({
                "type": "generation_volume",
                "message": f"Generated {total_probes} probes since inception",
                "impact": "high"
            })
        
        avg_separation = status.get("avg_separation_score", 0)
        if avg_separation > 0.7:
            insights.append({
                "type": "probe_quality",
                "message": f"High probe separation score: {avg_separation:.3f}",
                "impact": "high"
            })
        
        category_dist = status.get("category_distribution", {})
        if len(category_dist) > 5:
            insights.append({
                "type": "probe_diversity",
                "message": f"Good probe diversity across {len(category_dist)} categories",
                "impact": "medium"
            })
        
        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "evolution_summary": {
                "total_probes_generated": total_probes,
                "avg_separation_score": avg_separation,
                "near_misses_analyzed": status.get("near_misses_analyzed", 0),
                "category_diversity": len(category_dist)
            },
            "best_performing_probes": status.get("best_probes", [])[:5],
            "insights": insights,
            "recommendations": [
                "Continue analyzing near misses for probe generation",
                "Focus on underrepresented probe categories",
                "Monitor probe separation trends over time"
            ]
        }
        
    except Exception as e:
        logger.error(f"Evolution report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for weekly probe evaluation
async def weekly_probe_evolution():
    """Background task to automatically evaluate and evolve probes"""
    while True:
        try:
            # This would integrate with fingerprint service
            # to get recent near misses and test probe performance
            logger.info("Weekly probe evolution cycle started")
            
            # Sleep for 7 days
            await asyncio.sleep(7 * 24 * 3600)
            
        except Exception as e:
            logger.error(f"Weekly probe evolution failed: {e}")
            await asyncio.sleep(24 * 3600)  # Retry in 24 hours

# Start background task
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(weekly_probe_evolution())
    logger.info("Probe Smith service started with weekly evolution enabled")

if __name__ == "__main__":
    port = int(os.getenv("PROBE_SMITH_PORT", 8021))
    uvicorn.run(app, host="0.0.0.0", port=port)