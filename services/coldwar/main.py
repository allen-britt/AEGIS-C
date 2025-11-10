"""
AEGIS‑C Cold War Service
=========================

Campaign analytics and defense intelligence with standardized guard.
"""

import os
import sys
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import random

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import setup_service_guard, record_detection_score, logger, require_api_key

# Service Configuration
SERVICE_NAME = "coldwar"
SERVICE_PORT = int(os.getenv("COLDWAR_PORT", "8015"))

# Initialize FastAPI App
app = FastAPI(title="AEGIS‑C Cold War Service")

# Setup standardized guard
setup_service_guard(app, SERVICE_NAME)

# Data models
class CampaignEvent(BaseModel):
    event_type: str  # detection, blocking, diversion, etc.
    timestamp: str
    source_ip: Optional[str] = None
    target_service: Optional[str] = None
    threat_type: Optional[str] = None
    severity: str  # low, medium, high, critical
    details: Optional[Dict[str, Any]] = None

class CampaignAnalysis(BaseModel):
    campaign_id: str
    start_time: str
    end_time: Optional[str] = None
    event_count: int
    threat_actors: List[str]
    attack_vectors: List[str]
    targeted_services: List[str]
    severity_distribution: Dict[str, int]
    status: str  # active, contained, resolved

class DefenseMetrics(BaseModel):
    total_events: int
    blocked_attempts: int
    diverted_attacks: int
    detection_rate: float
    response_time_avg: float
    active_campaigns: int

# In-memory storage (replace with proper database in production)
CAMPAIGN_EVENTS: List[CampaignEvent] = []
ACTIVE_CAMPAIGNS: Dict[str, CampaignAnalysis] = {}
DEFENSE_STATS = {
    "total_events": 0,
    "blocked_attempts": 0,
    "diverted_attacks": 0,
    "detection_count": 0,
    "response_times": []
}

def analyze_campaign(events: List[CampaignEvent]) -> CampaignAnalysis:
    """Analyze events to identify campaign patterns"""
    if not events:
        raise HTTPException(status_code=400, detail="No events to analyze")
    
    # Sort events by timestamp
    events.sort(key=lambda x: x.timestamp)
    
    # Extract campaign data
    start_time = events[0].timestamp
    end_time = events[-1].timestamp
    
    threat_actors = list(set(event.details.get("threat_actor", "unknown") for event in events if event.details))
    attack_vectors = list(set(event.threat_type for event in events if event.threat_type))
    targeted_services = list(set(event.target_service for event in events if event.target_service))
    
    # Severity distribution
    severity_dist = {}
    for event in events:
        severity_dist[event.severity] = severity_dist.get(event.severity, 0) + 1
    
    # Generate campaign ID
    campaign_hash = f"{start_time}:{len(events)}:{attack_vectors[0] if attack_vectors else 'unknown'}"
    campaign_id = f"CAMPAIGN_{abs(hash(campaign_hash)) % 10000:04d}"
    
    # Determine status
    recent_events = [e for e in events if 
                    datetime.fromisoformat(e.timestamp) > datetime.now(timezone.utc) - timedelta(hours=1)]
    status = "active" if len(recent_events) > 0 else "contained"
    
    return CampaignAnalysis(
        campaign_id=campaign_id,
        start_time=start_time,
        end_time=end_time if status != "active" else None,
        event_count=len(events),
        threat_actors=threat_actors,
        attack_vectors=attack_vectors,
        targeted_services=targeted_services,
        severity_distribution=severity_dist,
        status=status
    )

def calculate_defense_metrics() -> DefenseMetrics:
    """Calculate overall defense metrics"""
    total_events = len(CAMPAIGN_EVENTS)
    blocked = sum(1 for e in CAMPAIGN_EVENTS if e.event_type == "blocked")
    diverted = sum(1 for e in CAMPAIGN_EVENTS if e.event_type == "diverted")
    
    detection_rate = DEFENSE_STATS["detection_count"] / max(total_events, 1)
    response_time_avg = sum(DEFENSE_STATS["response_times"]) / max(len(DEFENSE_STATS["response_times"]), 1)
    active_campaigns = len([c for c in ACTIVE_CAMPAIGNS.values() if c.status == "active"])
    
    return DefenseMetrics(
        total_events=total_events,
        blocked_attempts=blocked,
        diverted_attacks=diverted,
        detection_rate=round(detection_rate, 3),
        response_time_avg=round(response_time_avg, 3),
        active_campaigns=active_campaigns
    )

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    metrics = calculate_defense_metrics()
    return {
        "service": "AEGIS‑C Cold War",
        "version": "1.0.0",
        "description": "Campaign analytics and defense intelligence service",
        "endpoints": {
            "events": "/events",
            "campaigns": "/campaigns",
            "metrics": "/metrics",
            "analyze": "/analyze",
            "health": "/health",
            "metrics": "/metrics",
            "secure_ping": "/secure/ping"
        },
        "total_events": metrics.total_events,
        "active_campaigns": metrics.active_campaigns
    }

@app.post("/events")
async def add_campaign_event(event: CampaignEvent):
    """Add a new campaign event"""
    try:
        CAMPAIGN_EVENTS.append(event)
        DEFENSE_STATS["total_events"] += 1
        
        # Update defense stats
        if event.event_type == "blocked":
            DEFENSE_STATS["blocked_attempts"] += 1
        elif event.event_type == "diverted":
            DEFENSE_STATS["diverted_attacks"] += 1
        elif event.event_type == "detected":
            DEFENSE_STATS["detection_count"] += 1
        
        logger.info(f"Campaign event added", event_type=event.event_type, severity=event.severity)
        return {"success": True, "event_id": len(CAMPAIGN_EVENTS)}
        
    except Exception as e:
        logger.error(f"Failed to add campaign event: {e}")
        raise

@app.get("/events")
async def list_events(limit: int = 100, severity: Optional[str] = None):
    """List campaign events with filtering"""
    events = CAMPAIGN_EVENTS
    
    if severity:
        events = [e for e in events if e.severity == severity]
    
    # Return most recent events
    events.sort(key=lambda x: x.timestamp, reverse=True)
    
    return {
        "total_events": len(events),
        "events": events[:limit]
    }

@app.get("/campaigns")
async def list_campaigns():
    """List all identified campaigns"""
    return {
        "total_campaigns": len(ACTIVE_CAMPAIGNS),
        "campaigns": [
            {
                "campaign_id": campaign.campaign_id,
                "start_time": campaign.start_time,
                "event_count": campaign.event_count,
                "threat_actors": campaign.threat_actors,
                "attack_vectors": campaign.attack_vectors,
                "status": campaign.status,
                "severity_distribution": campaign.severity_distribution
            }
            for campaign in ACTIVE_CAMPAIGNS.values()
        ]
    }

@app.get("/metrics")
async def get_metrics():
    """Get defense metrics"""
    metrics = calculate_defense_metrics()
    return metrics

@app.post("/analyze")
async def analyze_recent_events(hours: int = 24):
    """Analyze recent events to identify campaigns"""
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_events = [
            e for e in CAMPAIGN_EVENTS 
            if datetime.fromisoformat(e.timestamp) > cutoff_time
        ]
        
        if not recent_events:
            return {"campaigns": [], "message": "No recent events found"}
        
        # Group events by potential campaigns
        # Simple grouping by threat actor and attack vector
        event_groups = {}
        for event in recent_events:
            key = f"{event.details.get('threat_actor', 'unknown')}:{event.threat_type or 'unknown'}"
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # Analyze each group
        new_campaigns = {}
        for group_key, events in event_groups.items():
            if len(events) >= 3:  # Minimum events for campaign
                campaign = analyze_campaign(events)
                new_campaigns[campaign.campaign_id] = campaign
        
        # Update active campaigns
        ACTIVE_CAMPAIGNS.update(new_campaigns)
        
        logger.info(f"Campaign analysis completed", new_campaigns=len(new_campaigns))
        return {
            "campaigns_identified": len(new_campaigns),
            "campaigns": list(new_campaigns.values()),
            "events_analyzed": len(recent_events)
        }
        
    except Exception as e:
        logger.error(f"Campaign analysis failed: {e}")
        raise

# Protected endpoints
@app.post("/secure/events", dependencies=[Depends(require_api_key)])
async def secure_add_event(event: CampaignEvent):
    """Protected event addition endpoint"""
    return await add_campaign_event(event)

@app.post("/secure/analyze", dependencies=[Depends(require_api_key)])
async def secure_analyze_events(hours: int = 24):
    """Protected campaign analysis endpoint"""
    return await analyze_recent_events(hours)

@app.delete("/secure/campaigns/{campaign_id}", dependencies=[Depends(require_api_key)])
async def close_campaign(campaign_id: str):
    """Close a campaign (admin only)"""
    if campaign_id not in ACTIVE_CAMPAIGNS:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    ACTIVE_CAMPAIGNS[campaign_id].status = "resolved"
    ACTIVE_CAMPAIGNS[campaign_id].end_time = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Campaign closed", campaign_id=campaign_id)
    
    return {"success": True, "message": f"Campaign {campaign_id} marked as resolved"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)