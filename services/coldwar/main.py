from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AEGIS‑C Cold War Defense")

class DocumentAnalysis(BaseModel):
    documents: List[str]
    source_metadata: Optional[Dict[str, Any]] = None

class SessionAnalysis(BaseModel):
    session_history: List[Dict[str, Any]]
    user_id: Optional[str] = None

class ToolChainValidation(BaseModel):
    tool_sequence: List[str]
    context: Optional[str] = None
    risk_threshold: float = 0.7

# Attack pattern signatures
COLD_WAR_PATTERNS = {
    "rag_injection": [
        r"when\s+asked.*?emphasize",
        r"ignore\s+earlier\s+directions",
        r"from\s+now\s+on",
        r"your\s+task\s+is\s+to",
        r"system\s+prompt",
        r"act\s+as\s+if",
        r"pretend\s+you\s+are"
    ],
    "policy_edge_surfing": [
        r"hypothetically",
        r"in\s+theory",
        r"for\s+educational\s+purposes",
        r"what\s+if",
        r"assume\s+that",
        r"imagine\s+a\s+scenario"
    ]
}

def detect_rag_injection(text: str) -> float:
    """Detect RAG injection attempts in text."""
    score = 0.0
    text_lower = text.lower()
    
    for pattern in COLD_WAR_PATTERNS["rag_injection"]:
        matches = len(re.findall(pattern, text_lower))
        score += matches * 0.2
    
    return min(1.0, score)

@app.post("/analyze/documents", response_model=Dict[str, Any])
async def analyze_documents(analysis: DocumentAnalysis):
    """Analyze documents for cold war attack patterns."""
    results = {
        "total_documents": len(analysis.documents),
        "rag_injection_score": 0.0,
        "policy_edge_score": 0.0,
        "high_risk_docs": [],
        "recommendations": []
    }
    
    for i, doc in enumerate(analysis.documents):
        rag_score = detect_rag_injection(doc)
        
        # Aggregate scores
        results["rag_injection_score"] += rag_score
        
        # Flag high-risk documents
        if rag_score > 0.6:
            results["high_risk_docs"].append({
                "document_index": i,
                "risk_score": rag_score
            })
    
    # Normalize scores
    if len(analysis.documents) > 0:
        results["rag_injection_score"] /= len(analysis.documents)
    
    # Generate recommendations
    if results["rag_injection_score"] > 0.3:
        results["recommendations"].append("Deploy context sanitization to block instruction injection")
    
    logger.info(f"Document analysis complete: {len(results['high_risk_docs'])} high-risk documents found")
    
    return results

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "service": "cold_war_defense"
    }

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AEGIS‑C Cold War Defense",
        "version": "1.0.0",
        "description": "Advanced defense against sustained AI attack campaigns"
    }