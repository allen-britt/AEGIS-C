"""
AEGIS‑C Vulnerability Database Service
=======================================

Real-time vulnerability intelligence and threat feed integration.
Connects to CVE databases, exploit databases, and security advisories.
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import structlog
import sqlite3
import redis
from dataclasses import dataclass

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

app = FastAPI(title="AEGIS‑C Vulnerability Database Service")

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VULN_DB_PATH = os.getenv("VULN_DB_PATH", "/tmp/vulnerabilities.db")
UPDATE_INTERVAL = int(os.getenv("VULN_UPDATE_INTERVAL", "1800"))  # 30 minutes

@dataclass
class CVE:
    """CVE vulnerability data structure"""
    cve_id: str
    title: str
    description: str
    severity: str
    cvss_score: float
    cvss_vector: str
    published_date: str
    modified_date: str
    references: List[str]
    affected_packages: List[str]
    exploit_available: bool = False
    exploit_difficulty: str = "UNKNOWN"  # LOW, MEDIUM, HIGH
    ai_relevant: bool = False
    ai_impact_areas: List[str] = None  # model_injection, data_poisoning, etc.

@dataclass
class Exploit:
    """Exploit information"""
    exploit_id: str
    cve_ids: List[str]
    title: str
    description: str
    source: str  # exploit-db, metasploit, github, etc.
    difficulty: str
    verified: bool
    published_date: str
    download_url: Optional[str] = None

@dataclass
class ThreatFeed:
    """Threat intelligence feed data"""
    feed_id: str
    source: str
    indicator_type: str  # ip, domain, url, hash, pattern
    indicators: List[str]
    description: str
    confidence: float
    last_updated: str
    tags: List[str]

class VulnerabilityDatabase:
    """SQLite database for vulnerability storage and indexing"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize vulnerability database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # CVEs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cves (
                cve_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                severity TEXT,
                cvss_score REAL,
                cvss_vector TEXT,
                published_date TEXT,
                modified_date TEXT,
                reference_links TEXT,
                affected_packages TEXT,
                exploit_available BOOLEAN,
                exploit_difficulty TEXT,
                ai_relevant BOOLEAN,
                ai_impact_areas TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Exploits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exploits (
                exploit_id TEXT PRIMARY KEY,
                cve_ids TEXT,
                title TEXT,
                description TEXT,
                source TEXT,
                difficulty TEXT,
                verified BOOLEAN,
                published_date TEXT,
                download_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Threat feeds table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_feeds (
                feed_id TEXT PRIMARY KEY,
                source TEXT,
                indicator_type TEXT,
                indicators TEXT,
                description TEXT,
                confidence REAL,
                last_updated TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cves_severity ON cves(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cves_cvss_score ON cves(cvss_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cves_ai_relevant ON cves(ai_relevant)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cves_published_date ON cves(published_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_exploits_difficulty ON exploits(difficulty)')
        
        conn.commit()
        conn.close()
    
    def store_cve(self, cve: CVE):
        """Store CVE in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO cves (
                cve_id, title, description, severity, cvss_score, cvss_vector,
                published_date, modified_date, reference_links, affected_packages,
                exploit_available, exploit_difficulty, ai_relevant, ai_impact_areas,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cve.cve_id,
            cve.title,
            cve.description,
            cve.severity,
            cve.cvss_score,
            cve.cvss_vector,
            cve.published_date,
            cve.modified_date,
            json.dumps(cve.references),
            json.dumps(cve.affected_packages),
            cve.exploit_available,
            cve.exploit_difficulty,
            cve.ai_relevant,
            json.dumps(cve.ai_impact_areas or []),
            datetime.now(timezone.utc).isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_ai_relevant_cves(self, limit: int = 100, min_cvss: float = 5.0) -> List[CVE]:
        """Get AI-relevant CVEs with minimum CVSS score"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM cves 
            WHERE (ai_relevant = TRUE OR exploit_available = TRUE) AND cvss_score >= ?
            ORDER BY cvss_score DESC, published_date DESC
            LIMIT ?
        ''', (min_cvss, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        cves = []
        for row in rows:
            cves.append(CVE(
                cve_id=row[0], title=row[1], description=row[2], severity=row[3],
                cvss_score=row[4], cvss_vector=row[5], published_date=row[6],
                modified_date=row[7], references=json.loads(row[8]),
                affected_packages=json.loads(row[9]), exploit_available=bool(row[10]),
                exploit_difficulty=row[11], ai_relevant=bool(row[12]),
                ai_impact_areas=json.loads(row[13])
            ))
        
        return cves
    
    def search_cves(self, query: str, limit: int = 50) -> List[CVE]:
        """Search CVEs by keyword"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM cves 
            WHERE title LIKE ? OR description LIKE ? OR cve_id LIKE ?
            ORDER BY cvss_score DESC
            LIMIT ?
        ''', (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        cves = []
        for row in rows:
            cves.append(CVE(
                cve_id=row[0], title=row[1], description=row[2], severity=row[3],
                cvss_score=row[4], cvss_vector=row[5], published_date=row[6],
                modified_date=row[7], references=json.loads(row[8]),
                affected_packages=json.loads(row[9]), exploit_available=bool(row[10]),
                exploit_difficulty=row[11], ai_relevant=bool(row[12]),
                ai_impact_areas=json.loads(row[13])
            ))
        
        return cves

class VulnIntelligenceCollector:
    """Collects vulnerability intelligence from multiple sources"""
    
    def __init__(self, db: VulnerabilityDatabase):
        self.db = db
        self.session = None
        self.redis_client = None
        
        # Initialize Redis if available
        try:
            self.redis_client = redis.from_url(REDIS_URL)
            self.redis_client.ping()
            logger.info("Redis connection established for caching")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "AEGIS-C-VulnDB/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_nvd_cves(self) -> List[CVE]:
        """Collect CVEs from NVD database"""
        logger.info("Collecting CVEs from NVD")
        
        try:
            # AI/ML related keywords for targeted search
            ai_keywords = [
                "machine learning", "artificial intelligence", "neural network",
                "tensorflow", "pytorch", "scikit-learn", "huggingface", "openai",
                "llm", "large language model", "chatgpt", "claude", "gemini",
                "transformers", "pandas", "numpy", "jupyter", "colab",
                "model injection", "data poisoning", "adversarial machine learning"
            ]
            
            all_cves = []
            
            for keyword in ai_keywords:
                # Check cache first
                cache_key = f"nvd_cves_{hashlib.md5(keyword.encode()).hexdigest()}"
                if self.redis_client:
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        cached_cves = json.loads(cached)
                        all_cves.extend(cached_cves)
                        continue
                
                # Fetch from NVD API
                url = f"https://services.nvd.nist.gov/rest/json/cves/2.0"
                params = {
                    "keywordSearch": keyword,
                    "resultsPerPage": 50,
                    "startIndex": 0
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get("vulnerabilities", []):
                            cve = self.parse_nvd_cve(item.get("cve", {}))
                            if cve and self.is_ai_relevant_cve(cve):
                                all_cves.append(cve)
                
                # Cache results
                if self.redis_client and all_cves:
                    self.redis_client.setex(
                        cache_key, 
                        timedelta(hours=1).total_seconds(),
                        json.dumps([asdict(cve) for cve in all_cves[-50:]])
                    )
                
                await asyncio.sleep(0.5)  # Rate limiting
            
            logger.info(f"Collected {len(all_cves)} AI-relevant CVEs from NVD")
            return all_cves
            
        except Exception as e:
            logger.error(f"Failed to collect NVD CVEs: {e}")
            return []
    
    def parse_nvd_cve(self, cve_data: Dict) -> Optional[CVE]:
        """Parse NVD CVE data"""
        try:
            cve_id = cve_data.get("id", "")
            
            # Get description
            descriptions = cve_data.get("descriptions", [])
            description = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break
            
            # Get CVSS metrics
            cvss_score = 0.0
            cvss_vector = ""
            metrics = cve_data.get("metrics", {})
            
            if "cvssMetricV31" in metrics:
                cvss_data = metrics["cvssMetricV31"][0].get("cvssData", {})
                cvss_score = cvss_data.get("baseScore", 0.0)
                cvss_vector = cvss_data.get("vectorString", "")
            elif "cvssMetricV30" in metrics:
                cvss_data = metrics["cvssMetricV30"][0].get("cvssData", {})
                cvss_score = cvss_data.get("baseScore", 0.0)
                cvss_vector = cvss_data.get("vectorString", "")
            
            # Determine severity
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
            
            # Get references
            references = []
            for ref in cve_data.get("references", []):
                if ref.get("url"):
                    references.append(ref["url"])
            
            # Extract affected packages from description and references
            affected_packages = self.extract_affected_packages(description, references)
            
            return CVE(
                cve_id=cve_id,
                title=cve_id,
                description=description,
                severity=severity,
                cvss_score=cvss_score,
                cvss_vector=cvss_vector,
                published_date=published_date,
                modified_date=modified_date,
                references=references,
                affected_packages=affected_packages,
                ai_relevant=True,  # Will be filtered later
                ai_impact_areas=self.classify_ai_impact(description)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse NVD CVE: {e}")
            return None
    
    def extract_affected_packages(self, description: str, references: List[str]) -> List[str]:
        """Extract affected packages from description and references"""
        packages = []
        
        # Common AI/ML packages to look for
        ai_packages = [
            "tensorflow", "torch", "pytorch", "transformers", "scikit-learn",
            "numpy", "pandas", "matplotlib", "seaborn", "jupyter", "keras",
            "huggingface", "openai", "anthropic", "langchain", "gradio",
            "streamlit", "fastapi", "flask", "django"
        ]
        
        text = (description + " " + " ".join(references)).lower()
        
        for package in ai_packages:
            if package in text:
                packages.append(package)
        
        return list(set(packages))
    
    def classify_ai_impact(self, description: str) -> List[str]:
        """Classify AI impact areas based on description"""
        impact_areas = []
        desc_lower = description.lower()
        
        impact_keywords = {
            "model_injection": ["injection", "prompt injection", "model injection"],
            "data_poisoning": ["poisoning", "training data", "dataset"],
            "adversarial_attack": ["adversarial", "evasion", "attack"],
            "model_extraction": ["extraction", "stealing", "theft"],
            "privacy_leak": ["privacy", "leak", "exposure", "pii"],
            "denial_of_service": ["dos", "denial of service", "resource exhaustion"],
            "supply_chain": ["supply chain", "dependency", "third-party"]
        }
        
        for impact, keywords in impact_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                impact_areas.append(impact)
        
        return impact_areas
    
    def is_ai_relevant_cve(self, cve: CVE) -> bool:
        """Determine if CVE is relevant to AI systems"""
        if cve.ai_relevant and cve.ai_impact_areas:
            return True
        
        if cve.affected_packages:
            return True
        
        if cve.cvss_score >= 7.0:  # High severity CVEs affecting ML systems
            return True
        
        return False
    
    async def collect_exploitdb(self) -> List[Exploit]:
        """Collect exploits from Exploit Database"""
        logger.info("Collecting exploits from Exploit-DB")
        
        try:
            # Search for AI/ML related exploits
            url = "https://www.exploit-db.com/api/search"
            params = {
                "text": "machine learning OR tensorflow OR pytorch OR AI",
                "limit": 50
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    exploits = []
                    for item in data.get("results", []):
                        exploit = self.parse_exploitdb_exploit(item)
                        if exploit:
                            exploits.append(exploit)
                    
                    logger.info(f"Collected {len(exploits)} exploits from Exploit-DB")
                    return exploits
                    
        except Exception as e:
            logger.error(f"Failed to collect Exploit-DB exploits: {e}")
        
        return []
    
    def parse_exploitdb_exploit(self, exploit_data: Dict) -> Optional[Exploit]:
        """Parse Exploit-DB exploit data"""
        try:
            return Exploit(
                exploit_id=f"EDB-{exploit_data.get('id', '')}",
                cve_ids=exploit_data.get("cve_ids", []),
                title=exploit_data.get("title", ""),
                description=exploit_data.get("description", ""),
                source="exploit-db",
                difficulty=exploit_data.get("type", "UNKNOWN"),
                verified=exploit_data.get("verified", False),
                published_date=exploit_data.get("date_published", ""),
                download_url=exploit_data.get("file_url")
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Exploit-DB exploit: {e}")
            return None
    
    async def collect_github_advisories(self) -> List[CVE]:
        """Collect security advisories from GitHub"""
        logger.info("Collecting GitHub security advisories")
        
        try:
            # Get Python package advisories (most relevant to AI/ML)
            url = "https://api.github.com/advisories"
            params = {
                "ecosystem": "pip",
                "modified": "2024-01-01..2024-12-31",
                "per_page": 100
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    advisories = []
                    for item in data:
                        advisory = self.parse_github_advisory(item)
                        if advisory and self.is_ai_relevant_cve(advisory):
                            advisories.append(advisory)
                    
                    logger.info(f"Collected {len(advisories)} AI-relevant GitHub advisories")
                    return advisories
                    
        except Exception as e:
            logger.error(f"Failed to collect GitHub advisories: {e}")
        
        return []
    
    def parse_github_advisory(self, advisory_data: Dict) -> Optional[CVE]:
        """Parse GitHub advisory data"""
        try:
            package_name = advisory_data.get("package", {}).get("name", "").lower()
            
            # Check if it's an AI/ML package
            ai_packages = ["tensorflow", "torch", "transformers", "scikit-learn", "numpy", "pandas"]
            if not any(ai_pkg in package_name for ai_pkg in ai_packages):
                return None
            
            # Get severity
            severity = advisory_data.get("severity", "UNKNOWN")
            
            # Get CVSS score if available
            cvss_score = 0.0
            cvss_vector = ""
            
            if "cvss" in advisory_data:
                cvss_data = advisory_data["cvss"]
                cvss_score = cvss_data.get("score", 0.0)
                cvss_vector = cvss_data.get("vectorString", "")
            
            return CVE(
                cve_id=advisory_data.get("ghsaId", ""),
                title=advisory_data.get("summary", ""),
                description=advisory_data.get("description", ""),
                severity=severity,
                cvss_score=cvss_score,
                cvss_vector=cvss_vector,
                published_date=advisory_data.get("publishedAt", ""),
                modified_date=advisory_data.get("updatedAt", ""),
                references=[],
                affected_packages=[package_name],
                ai_relevant=True,
                ai_impact_areas=["supply_chain"]
            )
            
        except Exception as e:
            logger.error(f"Failed to parse GitHub advisory: {e}")
            return None

# Global instances
vuln_db = VulnerabilityDatabase(VULN_DB_PATH)
collector = VulnIntelligenceCollector(vuln_db)

# API Models
class CVEResponse(BaseModel):
    cve_id: str
    title: str
    severity: str
    cvss_score: float
    ai_impact_areas: List[str]
    published_date: str
    description: str
    exploit_available: bool

class VulnerabilityStatsResponse(BaseModel):
    total_cves: int
    critical_count: int
    high_count: int
    exploit_count: int
    most_affected_packages: List[Dict]
    recent_threats: List[CVEResponse]

# API Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"ok": True, "service": "vuln_db", "version": "1.0.0"}

@app.get("/cves/ai", response_model=List[CVEResponse])
async def get_ai_cves(
    limit: int = Query(default=50, le=200),
    min_cvss: float = Query(default=5.0, ge=0.0, le=10.0),
    severity: Optional[str] = Query(default=None)
):
    """Get AI-relevant CVEs"""
    cves = vuln_db.get_ai_relevant_cves(limit, min_cvss)
    
    if severity:
        cves = [cve for cve in cves if cve.severity.upper() == severity.upper()]
    
    return [
        CVEResponse(
            cve_id=cve.cve_id,
            title=cve.title,
            severity=cve.severity,
            cvss_score=cve.cvss_score,
            ai_impact_areas=cve.ai_impact_areas or [],
            published_date=cve.published_date,
            description=cve.description[:400] + "..." if len(cve.description) > 400 else cve.description,
            exploit_available=cve.exploit_available
        )
        for cve in cves
    ]

@app.get("/cves/search", response_model=List[CVEResponse])
async def search_cves(
    query: str = Query(..., min_length=3),
    limit: int = Query(default=50, le=100)
):
    """Search CVEs by keyword"""
    cves = vuln_db.search_cves(query, limit)
    
    return [
        CVEResponse(
            cve_id=cve.cve_id,
            title=cve.title,
            severity=cve.severity,
            cvss_score=cve.cvss_score,
            ai_impact_areas=cve.ai_impact_areas or [],
            published_date=cve.published_date,
            description=cve.description[:400] + "..." if len(cve.description) > 400 else cve.description,
            exploit_available=cve.exploit_available
        )
        for cve in cves
    ]

@app.get("/stats/overview", response_model=VulnerabilityStatsResponse)
async def get_vulnerability_stats():
    """Get vulnerability statistics overview"""
    conn = sqlite3.connect(vuln_db.db_path)
    cursor = conn.cursor()
    
    # Get counts
    cursor.execute("SELECT COUNT(*) FROM cves WHERE ai_relevant = TRUE")
    total_cves = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM cves WHERE ai_relevant = TRUE AND severity = 'CRITICAL'")
    critical_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM cves WHERE ai_relevant = TRUE AND severity = 'HIGH'")
    high_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM cves WHERE exploit_available = TRUE")
    exploit_count = cursor.fetchone()[0]
    
    # Get most affected packages
    cursor.execute('''
        SELECT affected_packages, COUNT(*) as count 
        FROM cves 
        WHERE ai_relevant = TRUE AND affected_packages != '[]'
        GROUP BY affected_packages 
        ORDER BY count DESC 
        LIMIT 10
    ''')
    
    most_affected = []
    for row in cursor.fetchall():
        packages = json.loads(row[0])
        for package in packages:
            most_affected.append({"package": package, "count": row[1]})
    
    # Get recent threats
    recent_cves = vuln_db.get_ai_relevant_cves(limit=10)
    recent_threats = [
        CVEResponse(
            cve_id=cve.cve_id,
            title=cve.title,
            severity=cve.severity,
            cvss_score=cve.cvss_score,
            ai_impact_areas=cve.ai_impact_areas or [],
            published_date=cve.published_date,
            description=cve.description[:300] + "..." if len(cve.description) > 300 else cve.description,
            exploit_available=cve.exploit_available
        )
        for cve in recent_cves
    ]
    
    conn.close()
    
    return VulnerabilityStatsResponse(
        total_cves=total_cves,
        critical_count=critical_count,
        high_count=high_count,
        exploit_count=exploit_count,
        most_affected_packages=most_affected[:10],
        recent_threats=recent_threats
    )

@app.post("/update")
async def update_vulnerabilities(background_tasks: BackgroundTasks):
    """Trigger vulnerability database update"""
    background_tasks.add_task(collect_all_vulnerabilities)
    return {"message": "Vulnerability update started", "status": "processing"}

async def collect_all_vulnerabilities():
    """Collect vulnerabilities from all sources"""
    logger.info("Starting vulnerability collection")
    
    try:
        async with collector:
            # Collect from different sources
            nvd_cves = await collector.collect_nvd_cves()
            github_advisories = await collector.collect_github_advisories()
            exploits = await collector.collect_exploitdb()
            
            # Store CVEs
            for cve in nvd_cves + github_advisories:
                vuln_db.store_cve(cve)
            
            logger.info(f"Vulnerability collection completed: {len(nvd_cves)} NVD CVEs, {len(github_advisories)} GitHub advisories, {len(exploits)} exploits")
            
    except Exception as e:
        logger.error(f"Vulnerability collection failed: {e}")

@app.get("/threats/critical")
async def get_critical_threats():
    """Get critical AI threats requiring immediate attention"""
    critical_cves = vuln_db.get_ai_relevant_cves(limit=20, min_cvss=8.0)
    
    # Analyze critical threats
    threats = []
    for cve in critical_cves:
        threat_analysis = {
            "cve_id": cve.cve_id,
            "title": cve.title,
            "cvss_score": cve.cvss_score,
            "ai_impact_areas": cve.ai_impact_areas,
            "affected_packages": cve.affected_packages,
            "exploit_available": cve.exploit_available,
            "risk_level": "CRITICAL",
            "recommended_actions": self.generate_recommendations(cve)
        }
        threats.append(threat_analysis)
    
    return {"critical_threats": threats, "total_count": len(threats)}

def generate_recommendations(cve: CVE) -> List[str]:
    """Generate remediation recommendations for CVE"""
    recommendations = []
    
    if "model_injection" in (cve.ai_impact_areas or []):
        recommendations.append("Implement input validation and sanitization for model inputs")
        recommendations.append("Deploy prompt injection detection and filtering")
    
    if "data_poisoning" in (cve.ai_impact_areas or []):
        recommendations.append("Validate and filter training data sources")
        recommendations.append("Implement data provenance tracking")
    
    if "adversarial_attack" in (cve.ai_impact_areas or []):
        recommendations.append("Deploy adversarial input detection")
        recommendations.append("Implement model robustness testing")
    
    if cve.exploit_available:
        recommendations.append("IMMEDIATE: Apply available patches or mitigations")
        recommendations.append("Monitor for exploitation attempts")
    
    if cve.cvss_score >= 9.0:
        recommendations.append("CRITICAL: Patch within 24 hours")
        recommendations.append("Consider temporary service disablement if patch unavailable")
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8019)