"""
Enhanced AEGIS‑C Detector Service with Intelligence Integration
==============================================================

Integrates real-time vulnerability intelligence and threat feeds
to provide comprehensive AI security detection.
"""

import os
import sys
import requests
import json
import time
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog

# Configure structured logging
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

app = FastAPI(title="AEGIS‑C Enhanced Detector Service")

# Configuration
INTEL_SERVICE_URL = os.getenv("INTEL_SERVICE_URL", "http://localhost:8018")
VULN_DB_URL = os.getenv("VULN_DB_URL", "http://localhost:8019")
DETECTION_CACHE_TTL = int(os.getenv("DETECTION_CACHE_TTL", "300"))  # 5 minutes

class ThreatIntelligenceClient:
    """Client for threat intelligence services"""
    
    def __init__(self):
        self.intel_url = INTEL_SERVICE_URL
        self.vuln_url = VULN_DB_URL
        self.cache = {}
    
    def get_ai_vulnerabilities(self, limit: int = 50) -> List[Dict]:
        """Get AI-relevant vulnerabilities"""
        try:
            cache_key = f"ai_vulns_{limit}_{int(time.time() // DETECTION_CACHE_TTL)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            response = requests.get(f"{self.intel_url}/vulnerabilities/ai?limit={limit}", timeout=10)
            if response.status_code == 200:
                vulns = response.json()
                self.cache[cache_key] = vulns
                return vulns
        except Exception as e:
            logger.error(f"Failed to get AI vulnerabilities: {e}")
        
        return []
    
    def search_vulnerabilities(self, query: str) -> List[Dict]:
        """Search vulnerabilities by keyword"""
        try:
            response = requests.get(f"{self.vuln_url}/cves/search?query={query}&limit=20", timeout=10)
            if response.status_code == 200:
                return response.json().get("results", [])
        except Exception as e:
            logger.error(f"Failed to search vulnerabilities: {e}")
        
        return []
    
    def get_critical_threats(self) -> List[Dict]:
        """Get critical threats requiring immediate attention"""
        try:
            response = requests.get(f"{self.vuln_url}/threats/critical", timeout=10)
            if response.status_code == 200:
                return response.json().get("critical_threats", [])
        except Exception as e:
            logger.error(f"Failed to get critical threats: {e}")
        
        return []

class AIModelDetector:
    """Advanced AI model detection using threat intelligence"""
    
    def __init__(self, intel_client: ThreatIntelligenceClient):
        self.intel_client = intel_client
        self.known_vulnerabilities = {}
        self.threat_patterns = self.load_threat_patterns()
    
    def load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load known threat patterns for AI systems"""
        return {
            "prompt_injection": [
                "ignore previous instructions",
                "system prompt", "developer mode", "jailbreak",
                "act as", "pretend you are", "roleplay as"
            ],
            "data_extraction": [
                "training data", "proprietary information",
                "internal data", "confidential", "trade secrets"
            ],
            "model_extraction": [
                "model weights", "architecture", "parameters",
                "model file", "download model", "extract model"
            ],
            "adversarial_examples": [
                "adversarial", "perturbation", "gradient descent",
                "optimization attack", "evasion technique"
            ],
            "supply_chain": [
                "pip install", "conda install", "package manager",
                "dependency", "requirements.txt", "setup.py"
            ]
        }
    
    def detect_text_with_intelligence(self, text: str) -> Dict[str, Any]:
        """Enhanced text detection using threat intelligence"""
        # Basic AI detection
        base_score, base_verdict, base_signals = self.basic_ai_detection(text)
        
        # Threat intelligence enhancement
        intel_enhancements = self.analyze_threat_context(text)
        
        # Vulnerability correlation
        vuln_correlations = self.correlate_vulnerabilities(text)
        
        # Combine results
        enhanced_score = min(1.0, base_score + intel_enhancements["threat_score"] + vuln_correlations["risk_score"])
        
        if enhanced_score >= 0.8:
            enhanced_verdict = "high_risk_ai"
        elif enhanced_score >= 0.6:
            enhanced_verdict = "suspicious_ai"
        elif enhanced_score >= 0.4:
            enhanced_verdict = "uncertain"
        else:
            enhanced_verdict = "likely_human"
        
        return {
            "score": round(enhanced_score, 3),
            "verdict": enhanced_verdict,
            "signals": {
                **base_signals,
                "threat_patterns": intel_enhancements["patterns_found"],
                "vulnerability_matches": vuln_correlations["matches"],
                "risk_indicators": vuln_correlations["indicators"]
            },
            "intelligence": {
                "threat_score": intel_enhancements["threat_score"],
                "vulnerability_risk": vuln_correlations["risk_score"],
                "related_cves": vuln_correlations["related_cves"],
                "recommendations": self.generate_security_recommendations(
                    enhanced_verdict, intel_enhancements, vuln_correlations
                )
            }
        }
    
    def basic_ai_detection(self, text: str) -> tuple[float, str, dict]:
        """Basic AI text detection"""
        signals = {}
        
        # Text characteristics
        length = len(text)
        word_count = len(text.split())
        avg_word_len = sum(len(word) for word in text.split()) / max(word_count, 1)
        
        signals.update({
            "char_count": length,
            "word_count": word_count,
            "avg_word_length": round(avg_word_len, 2)
        })
        
        # AI writing patterns
        ai_indicators = [
            "however", "therefore", "furthermore", "consequently",
            "in conclusion", "it is important to note", "additionally",
            "moreover", "nevertheless", "nonetheless"
        ]
        
        ai_indicator_count = sum(1 for indicator in ai_indicators if indicator in text.lower())
        
        # Technical complexity indicators
        technical_terms = [
            "algorithm", "optimization", "implementation", "architecture",
            "framework", "methodology", "analysis", "computation"
        ]
        
        technical_count = sum(1 for term in technical_terms if term in text.lower())
        
        # Basic scoring
        score = 0.0
        if length > 1000: score += 0.2
        if avg_word_len > 6: score += 0.2
        if ai_indicator_count > 2: score += 0.3
        if technical_count > 3: score += 0.2
        if word_count > 100: score += 0.1
        
        score = min(1.0, score)
        
        if score >= 0.7:
            verdict = "likely_ai"
        elif score >= 0.4:
            verdict = "uncertain"
        else:
            verdict = "likely_human"
        
        return score, verdict, signals
    
    def analyze_threat_context(self, text: str) -> Dict[str, Any]:
        """Analyze text for threat patterns"""
        text_lower = text.lower()
        patterns_found = []
        threat_score = 0.0
        
        for threat_type, patterns in self.threat_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in text_lower)
            if matches > 0:
                patterns_found.append({
                    "type": threat_type,
                    "matches": matches,
                    "patterns": [p for p in patterns if p in text_lower]
                })
                
                # Score based on threat type severity
                if threat_type == "prompt_injection":
                    threat_score += 0.4
                elif threat_type == "data_extraction":
                    threat_score += 0.3
                elif threat_type == "model_extraction":
                    threat_score += 0.5
                elif threat_type == "adversarial_examples":
                    threat_score += 0.3
                elif threat_type == "supply_chain":
                    threat_score += 0.2
        
        return {
            "patterns_found": patterns_found,
            "threat_score": min(1.0, threat_score)
        }
    
    def correlate_vulnerabilities(self, text: str) -> Dict[str, Any]:
        """Correlate text with known vulnerabilities"""
        # Extract potential technical terms from text
        words = text.lower().split()
        tech_terms = [word for word in words if len(word) > 4 and word.isalpha()]
        
        # Search for vulnerabilities related to these terms
        related_cves = []
        risk_score = 0.0
        indicators = []
        
        for term in tech_terms[:5]:  # Limit to avoid too many API calls
            vulns = self.intel_client.search_vulnerabilities(term)
            for vuln in vulns:
                if vuln.get("cvss_score", 0) >= 7.0:  # High severity only
                    related_cves.append({
                        "cve_id": vuln.get("cve_id"),
                        "title": vuln.get("title"),
                        "severity": vuln.get("severity"),
                        "cvss_score": vuln.get("cvss_score")
                    })
                    risk_score += 0.1
        
        # Check for high-risk indicators
        high_risk_terms = ["exploit", "vulnerability", "backdoor", "malware", "rootkit"]
        for term in high_risk_terms:
            if term in text.lower():
                indicators.append(term)
                risk_score += 0.2
        
        return {
            "matches": len(related_cves),
            "related_cves": related_cves[:5],  # Limit to top 5
            "risk_score": min(1.0, risk_score),
            "indicators": indicators
        }
    
    def generate_security_recommendations(self, verdict: str, intel: Dict, vulns: Dict) -> List[str]:
        """Generate security recommendations based on detection"""
        recommendations = []
        
        if verdict in ["high_risk_ai", "suspicious_ai"]:
            recommendations.append("Review and validate user input before processing")
            recommendations.append("Implement content filtering and sanitization")
        
        if intel["patterns_found"]:
            for pattern in intel["patterns_found"]:
                if pattern["type"] == "prompt_injection":
                    recommendations.append("Deploy prompt injection detection and filtering")
                elif pattern["type"] == "data_extraction":
                    recommendations.append("Monitor for attempts to extract sensitive information")
                elif pattern["type"] == "model_extraction":
                    recommendations.append("Implement model access controls and monitoring")
        
        if vulns["related_cves"]:
            recommendations.append("Review related CVEs for applicable patches")
            recommendations.append("Monitor for exploitation attempts of known vulnerabilities")
        
        if vulns["indicators"]:
            recommendations.append("Investigate potential security threats or malicious intent")
        
        return recommendations

# API Models
class TextPayload(BaseModel):
    text: str
    include_intelligence: bool = True

class EnhancedDetectionResult(BaseModel):
    score: float
    verdict: str
    signals: Dict[str, Any]
    intelligence: Dict[str, Any]

# Initialize services
intel_client = ThreatIntelligenceClient()
ai_detector = AIModelDetector(intel_client)

# API Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True, "service": "enhanced_detector", "version": "2.0.0"}

@app.post("/detect/text", response_model=EnhancedDetectionResult)
async def detect_text_enhanced(payload: TextPayload):
    """Enhanced AI text detection with threat intelligence"""
    try:
        if payload.include_intelligence:
            result = ai_detector.detect_text_with_intelligence(payload.text)
        else:
            # Basic detection without intelligence
            score, verdict, signals = ai_detector.basic_ai_detection(payload.text)
            result = {
                "score": score,
                "verdict": verdict,
                "signals": signals,
                "intelligence": {"enabled": False}
            }
        
        logger.info(f"Enhanced text detection: score={result['score']}, verdict={result['verdict']}")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")

@app.get("/intelligence/status")
async def get_intelligence_status():
    """Get status of threat intelligence integration"""
    try:
        # Test connectivity to intelligence services
        intel_health = requests.get(f"{INTEL_SERVICE_URL}/health", timeout=5).status_code == 200
        vuln_health = requests.get(f"{VULN_DB_URL}/health", timeout=5).status_code == 200
        
        # Get current threat landscape
        critical_threats = intel_client.get_critical_threats()
        ai_vulns = intel_client.get_ai_vulnerabilities(limit=10)
        
        return {
            "services": {
                "intelligence_service": "healthy" if intel_health else "unhealthy",
                "vulnerability_database": "healthy" if vuln_health else "unhealthy"
            },
            "threat_landscape": {
                "critical_threats_count": len(critical_threats),
                "ai_vulnerabilities_count": len(ai_vulns),
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "cache_status": {
                "cached_entries": len(intel_client.cache),
                "cache_ttl_seconds": DETECTION_CACHE_TTL
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get intelligence status: {e}")
        return {
            "services": {"error": str(e)},
            "threat_landscape": {"error": str(e)},
            "cache_status": {"error": str(e)}
        }

@app.post("/update/intelligence")
async def update_intelligence(background_tasks: BackgroundTasks):
    """Trigger intelligence update"""
    try:
        # Trigger updates in intelligence services
        requests.post(f"{INTEL_SERVICE_URL}/update", timeout=5)
        requests.post(f"{VULN_DB_URL}/update", timeout=5)
        
        return {"message": "Intelligence update triggered", "status": "processing"}
        
    except Exception as e:
        logger.error(f"Failed to trigger intelligence update: {e}")
        raise HTTPException(status_code=500, detail="Failed to update intelligence")

@app.get("/stats/enhanced")
async def get_enhanced_stats():
    """Get enhanced detection statistics"""
    try:
        # Get threat landscape data
        ai_vulns = intel_client.get_ai_vulnerabilities(limit=100)
        critical_threats = intel_client.get_critical_threats()
        
        # Analyze threat patterns
        threat_types = {}
        severity_distribution = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for vuln in ai_vulns:
            severity = vuln.get("severity", "LOW")
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        for threat in critical_threats:
            for impact_area in threat.get("ai_impact_areas", []):
                threat_types[impact_area] = threat_types.get(impact_area, 0) + 1
        
        return {
            "threat_intelligence": {
                "ai_vulnerabilities_tracked": len(ai_vulns),
                "critical_threats_active": len(critical_threats),
                "severity_distribution": severity_distribution,
                "threat_type_distribution": threat_types
            },
            "detection_capabilities": {
                "threat_patterns_loaded": len(ai_detector.threat_patterns),
                "intelligence_integration": True,
                "real_time_updates": True
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get enhanced stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)