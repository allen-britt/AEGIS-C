"""
AEGIS‑C Intelligence Service
=============================

Real-time threat intelligence integration for AI security monitoring.
Connects to vulnerability databases, threat feeds, and security advisories.
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog
import hashlib
import sqlite3
from dataclasses import asdict

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

app = FastAPI(title="AEGIS‑C Intelligence Service")

# Configuration
INTEL_DB_PATH = os.getenv("INTEL_DB_PATH", "/tmp/intelligence.db")
UPDATE_INTERVAL = int(os.getenv("INTEL_UPDATE_INTERVAL", "3600"))  # 1 hour

@dataclass
class Vulnerability:
    """Vulnerability data structure"""
    cve_id: str
    title: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    cvss_score: float
    published_date: str
    modified_date: str
    affected_products: List[str]
    references: List[str]
    ai_relevant: bool = False
    ai_impact_score: float = 0.0

@dataclass
class ThreatIndicator:
    """Threat intelligence indicator"""
    indicator: str  # IP, domain, hash, etc.
    type: str  # ip, domain, url, hash, etc.
    description: str
    source: str
    confidence: float
    first_seen: str
    last_seen: str
    tags: List[str]

@dataclass
class SecurityAdvisory:
    """Security advisory data"""
    advisory_id: str
    title: str
    severity: str
    affected_systems: List[str]
    recommendations: List[str]
    published_date: str
    source: str

class IntelligenceDatabase:
    """Local intelligence database for caching and analysis"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for intelligence storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # CVEs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                cve_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                severity TEXT,
                cvss_score REAL,
                published_date TEXT,
                modified_date TEXT,
                affected_products TEXT,
                references TEXT,
                ai_relevant BOOLEAN,
                ai_impact_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Threat indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator TEXT,
                type TEXT,
                description TEXT,
                source TEXT,
                confidence REAL,
                first_seen TEXT,
                last_seen TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(indicator, type)
            )
        ''')
        
        # Security advisories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_advisories (
                advisory_id TEXT PRIMARY KEY,
                title TEXT,
                severity TEXT,
                affected_systems TEXT,
                recommendations TEXT,
                published_date TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_vulnerability(self, vuln: Vulnerability):
        """Store vulnerability in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO vulnerabilities 
            (cve_id, title, description, severity, cvss_score, published_date, 
             modified_date, affected_products, references, ai_relevant, ai_impact_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            vuln.cve_id, vuln.title, vuln.description, vuln.severity, vuln.cvss_score,
            vuln.published_date, vuln.modified_date, json.dumps(vuln.affected_products),
            json.dumps(vuln.references), vuln.ai_relevant, vuln.ai_impact_score
        ))
        
        conn.commit()
        conn.close()
    
    def get_ai_relevant_vulns(self, limit: int = 100) -> List[Vulnerability]:
        """Get AI-relevant vulnerabilities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM vulnerabilities 
            WHERE ai_relevant = TRUE OR ai_impact_score > 0.5
            ORDER BY cvss_score DESC, published_date DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        vulns = []
        for row in rows:
            vulns.append(Vulnerability(
                cve_id=row[0], title=row[1], description=row[2], severity=row[3],
                cvss_score=row[4], published_date=row[5], modified_date=row[6],
                affected_products=json.loads(row[7]), references=json.loads(row[8]),
                ai_relevant=bool(row[9]), ai_impact_score=row[10]
            ))
        
        return vulns

class ThreatIntelligenceCollector:
    """Collects threat intelligence from various sources"""
    
    def __init__(self, db: IntelligenceDatabase):
        self.db = db
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_nvd_vulnerabilities(self) -> List[Vulnerability]:
        """Collect vulnerabilities from NVD (National Vulnerability Database)"""
        logger.info("Collecting NVD vulnerabilities")
        
        try:
            # Search for AI/ML related vulnerabilities
            ai_keywords = ["machine learning", "artificial intelligence", "neural network", 
                          "tensorflow", "pytorch", "scikit-learn", "huggingface", "openai",
                          "llm", "large language model", "chatgpt", "claude", "gemini"]
            
            vulnerabilities = []
            
            for keyword in ai_keywords:
                url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={keyword}"
                
                async with self.session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get("vulnerabilities", []):
                            cve = item.get("cve", {})
                            vuln = self.parse_nvd_vulnerability(cve)
                            if vuln:
                                vulnerabilities.append(vuln)
                
                await asyncio.sleep(1)  # Rate limiting
            
            logger.info(f"Collected {len(vulnerabilities)} NVD vulnerabilities")
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Failed to collect NVD vulnerabilities: {e}")
            return []
    
    def parse_nvd_vulnerability(self, cve_data: Dict) -> Optional[Vulnerability]:
        """Parse NVD vulnerability data"""
        try:
            cve_id = cve_data.get("id", "")
            descriptions = cve_data.get("descriptions", [])
            
            # Find English description
            description = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break
            
            # Get CVSS score
            cvss_score = 0.0
            metrics = cve_data.get("metrics", {})
            if "cvssMetricV31" in metrics:
                cvss_data = metrics["cvssMetricV31"][0].get("cvssData", {})
                cvss_score = cvss_data.get("baseScore", 0.0)
            
            # Get severity
            severity = "LOW"
            if cvss_score >= 9.0:
                severity = "CRITICAL"
            elif cvss_score >= 7.0:
                severity = "HIGH"
            elif cvss_score >= 4.0:
                severity = "MEDIUM"
            
            # Get dates
            published_date = cve_data.get("published", "")
            modified_date = cve_data.get("lastModified", "")
            
            # Get affected products
            affected_products = []
            references = []
            
            return Vulnerability(
                cve_id=cve_id,
                title=cve_id,  # NVD doesn't always have titles
                description=description,
                severity=severity,
                cvss_score=cvss_score,
                published_date=published_date,
                modified_date=modified_date,
                affected_products=affected_products,
                references=references,
                ai_relevant=True,  # All from AI keyword search
                ai_impact_score=min(cvss_score / 10.0, 1.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse NVD vulnerability: {e}")
            return None
    
    async def collect_misp_indicators(self) -> List[ThreatIndicator]:
        """Collect threat indicators from MISP (Malware Information Sharing Platform)"""
        logger.info("Collecting MISP indicators")
        
        # This would integrate with a real MISP instance
        # For now, return mock data
        indicators = [
            ThreatIndicator(
                indicator="malicious-ai-model.example.com",
                type="domain",
                description="Domain hosting malicious AI models",
                source="MISP",
                confidence=0.8,
                first_seen="2024-01-01T00:00:00Z",
                last_seen="2024-11-10T12:00:00Z",
                tags=["ai", "malicious", "model"]
            )
        ]
        
        return indicators
    
    async def collect_github_advisories(self) -> List[SecurityAdvisory]:
        """Collect security advisories from GitHub"""
        logger.info("Collecting GitHub security advisories")
        
        try:
            # Search for AI/ML related security advisories
            url = "https://api.github.com/advisories?ecosystem=pip&modified=2024-01-01..2024-12-31"
            
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    advisories = []
                    for item in data[:50]:  # Limit to 50 most recent
                        advisory = self.parse_github_advisory(item)
                        if advisory:
                            advisories.append(advisory)
                    
                    logger.info(f"Collected {len(advisories)} GitHub advisories")
                    return advisories
                    
        except Exception as e:
            logger.error(f"Failed to collect GitHub advisories: {e}")
        
        return []
    
    def parse_github_advisory(self, advisory_data: Dict) -> Optional[SecurityAdvisory]:
        """Parse GitHub advisory data"""
        try:
            # Check if AI/ML related
            package_name = advisory_data.get("package", {}).get("name", "").lower()
            ai_packages = ["tensorflow", "torch", "transformers", "scikit-learn", "numpy", "pandas"]
            
            if not any(ai_pkg in package_name for ai_pkg in ai_packages):
                return None
            
            return SecurityAdvisory(
                advisory_id=advisory_data.get("ghsaId", ""),
                title=advisory_data.get("summary", ""),
                severity=advisory_data.get("severity", ""),
                affected_systems=[package_name],
                recommendations=advisory_data.get("references", []),
                published_date=advisory_data.get("publishedAt", ""),
                source="GitHub"
            )
            
        except Exception as e:
            logger.error(f"Failed to parse GitHub advisory: {e}")
            return None

class AIIntelligenceAnalyzer:
    """Analyzes intelligence data for AI-specific threats"""
    
    def __init__(self, db: IntelligenceDatabase):
        self.db = db
    
    def analyze_ai_threats(self) -> Dict[str, Any]:
        """Analyze AI-specific threats from collected intelligence"""
        ai_vulns = self.db.get_ai_relevant_vulns()
        
        analysis = {
            "total_ai_vulnerabilities": len(ai_vulns),
            "critical_vulnerabilities": len([v for v in ai_vulns if v.severity == "CRITICAL"]),
            "high_vulnerabilities": len([v for v in ai_vulns if v.severity == "HIGH"]),
            "average_cvss_score": sum(v.cvss_score for v in ai_vulns) / len(ai_vulns) if ai_vulns else 0,
            "most_affected_products": self.get_most_affected_products(ai_vulns),
            "recent_vulnerabilities": self.get_recent_vulnerabilities(ai_vulns, 7),
            "threat_trends": self.analyze_threat_trends(ai_vulns)
        }
        
        return analysis
    
    def get_most_affected_products(self, vulns: List[Vulnerability]) -> List[Dict]:
        """Get most affected AI products"""
        product_counts = {}
        
        for vuln in vulns:
            for product in vuln.affected_products:
                product_counts[product] = product_counts.get(product, 0) + 1
        
        sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{"product": product, "count": count} for product, count in sorted_products[:10]]
    
    def get_recent_vulnerabilities(self, vulns: List[Vulnerability], days: int) -> List[Vulnerability]:
        """Get recent vulnerabilities from last N days"""
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        recent_vulns = [v for v in vulns if v.published_date > cutoff_date]
        return sorted(recent_vulns, key=lambda v: v.cvss_score, reverse=True)[:10]
    
    def analyze_threat_trends(self, vulns: List[Vulnerability]) -> Dict[str, Any]:
        """Analyze threat trends over time"""
        # Group vulnerabilities by month
        monthly_counts = {}
        
        for vuln in vulns:
            if vuln.published_date:
                month = vuln.published_date[:7]  # YYYY-MM
                monthly_counts[month] = monthly_counts.get(month, 0) + 1
        
        return {
            "monthly_vulnerability_counts": monthly_counts,
            "trend_direction": "increasing" if len(monthly_counts) > 1 else "stable"
        }

# Global instances
intel_db = IntelligenceDatabase(INTEL_DB_PATH)
collector = ThreatIntelligenceCollector(intel_db)
analyzer = AIIntelligenceAnalyzer(intel_db)

# API Models
class VulnerabilityResponse(BaseModel):
    cve_id: str
    title: str
    severity: str
    cvss_score: float
    ai_impact_score: float
    published_date: str
    description: str

class ThreatAnalysisResponse(BaseModel):
    total_ai_vulnerabilities: int
    critical_vulnerabilities: int
    high_vulnerabilities: int
    average_cvss_score: float
    most_affected_products: List[Dict]
    threat_trends: Dict[str, Any]

# API Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True, "service": "intelligence", "version": "1.0.0"}

@app.get("/vulnerabilities/ai", response_model=List[VulnerabilityResponse])
async def get_ai_vulnerabilities(limit: int = 50, severity: Optional[str] = None):
    """Get AI-relevant vulnerabilities"""
    vulns = intel_db.get_ai_relevant_vulns(limit)
    
    if severity:
        vulns = [v for v in vulns if v.severity.upper() == severity.upper()]
    
    return [
        VulnerabilityResponse(
            cve_id=v.cve_id,
            title=v.title,
            severity=v.severity,
            cvss_score=v.cvss_score,
            ai_impact_score=v.ai_impact_score,
            published_date=v.published_date,
            description=v.description[:500] + "..." if len(v.description) > 500 else v.description
        )
        for v in vulns
    ]

@app.get("/analysis/ai-threats", response_model=ThreatAnalysisResponse)
async def analyze_ai_threats():
    """Analyze AI-specific threats"""
    analysis = analyzer.analyze_ai_threats()
    return ThreatAnalysisResponse(**analysis)

@app.post("/update")
async def update_intelligence(background_tasks: BackgroundTasks):
    """Trigger intelligence update"""
    background_tasks.add_task(collect_all_intelligence)
    return {"message": "Intelligence update started", "status": "processing"}

async def collect_all_intelligence():
    """Collect intelligence from all sources"""
    logger.info("Starting intelligence collection")
    
    try:
        async with collector:
            # Collect from different sources
            nvd_vulns = await collector.collect_nvd_vulnerabilities()
            misp_indicators = await collector.collect_misp_indicators()
            github_advisories = await collector.collect_github_advisories()
            
            # Store in database
            for vuln in nvd_vulns:
                intel_db.store_vulnerability(vuln)
            
            logger.info(f"Intelligence collection completed: {len(nvd_vulns)} vulnerabilities, {len(misp_indicators)} indicators, {len(github_advisories)} advisories")
            
    except Exception as e:
        logger.error(f"Intelligence collection failed: {e}")

@app.get("/search/vulnerabilities")
async def search_vulnerabilities(query: str, limit: int = 20):
    """Search vulnerabilities by keyword"""
    conn = sqlite3.connect(intel_db.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM vulnerabilities 
        WHERE title LIKE ? OR description LIKE ? OR cve_id LIKE ?
        ORDER BY cvss_score DESC
        LIMIT ?
    ''', (f"%{query}%", f"%{query}%", f"%{query}%", limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "cve_id": row[0],
            "title": row[1],
            "severity": row[3],
            "cvss_score": row[4],
            "ai_impact_score": row[10],
            "description": row[2][:300] + "..." if len(row[2]) > 300 else row[2]
        })
    
    return {"query": query, "results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8018)