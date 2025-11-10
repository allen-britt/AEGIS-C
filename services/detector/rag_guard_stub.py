#!/usr/bin/env python3
"""
AEGIS‑C RAG Guard Stub
=======================

Deterministic stub for RAG content sanitization.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import re

app = FastAPI(title="AEGIS-C RAG Guard Stub", version="1.0.0")

class ContentChunk(BaseModel):
    text: str = Field(..., description="Content text")
    source: str = Field(..., description="Content source identifier")

class SanitizeRequest(BaseModel):
    chunks: List[ContentChunk] = Field(..., description="Content chunks to sanitize")
    mode: str = Field(default="auto", description="Sanitization mode: 'auto' or 'strict'")

class SanitizedChunk(BaseModel):
    text: str = Field(..., description="Sanitized text")
    notes: List[str] = Field(default_factory=list, description="Processing notes")

class SanitizationFlags(BaseModel):
    injection: bool = Field(default=False, description="Injection attempt detected")
    severity: str = Field(default="low", description="Severity level: 'low', 'medium', 'high'")

class SanitizeResponse(BaseModel):
    chunks: List[SanitizedChunk] = Field(..., description="Sanitized chunks")
    flags: SanitizationFlags = Field(..., description="Security flags")
    timestamp: str = Field(..., description="Processing timestamp")

# Injection detection patterns
INJECTION_PATTERNS = [
    r'(?i)ignore\s+(your|the|previous|all)\s+(instructions|training|guidelines|rules)',
    r'(?i)pretend\s+(you\s+are|you\'re)\s+(not\s+)?(an?\s+)?(ai|assistant|model)',
    r'(?i)bypass\s+(your|the|safety|security|ethical)\s+(filters|controls|restrictions)',
    r'(?i)(forget|disregard|override)\s+(everything|all\s+previous)',
    r'(?i)(system|developer|admin)\s+(message|prompt|instruction)',
    r'(?i)DAN\s+(Do\s+Anything\s+Now)',
]

def _detect_injection(text: str) -> tuple[bool, str]:
    """Detect injection attempts in text"""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True, "high"
    
    # Check for suspicious keywords
    suspicious_keywords = ['jailbreak', 'unfiltered', 'uncensored', 'override', 'bypass']
    keyword_count = sum(1 for keyword in suspicious_keywords if keyword.lower() in text.lower())
    
    if keyword_count >= 2:
        return True, "medium"
    elif keyword_count >= 1:
        return True, "low"
    
    return False, "low"

def _sanitize_text(text: str, mode: str, has_injection: bool, severity: str) -> tuple[str, List[str]]:
    """Sanitize text based on mode and injection detection"""
    notes = []
    sanitized = text
    
    if has_injection:
        if mode == "strict" or severity == "high":
            # Strict sanitization - remove instructions
            for pattern in INJECTION_PATTERNS:
                sanitized = re.sub(pattern, '[INSTRUCTION_REMOVED]', sanitized, flags=re.IGNORECASE)
            notes.append("Embedded instructions removed")
            notes.append("Strict sanitization applied")
        else:
            # Auto mode - neutralize
            sanitized = f"[SANITIZED_CONTENT: {len(text)} chars with potential injection]"
            notes.append("Content sanitized due to injection patterns")
    
    # Additional sanitization for suspicious content
    if len(sanitized) > 10000:  # Very long content
        sanitized = sanitized[:5000] + " [CONTENT_TRUNCATED]"
        notes.append("Content truncated for safety")
    
    return sanitized, notes

@app.post("/sanitize", response_model=SanitizeResponse)
def sanitize_content(request: SanitizeRequest):
    """Sanitize RAG content chunks"""
    try:
        if not request.chunks:
            raise HTTPException(status_code=400, detail="At least one chunk required")
        
        sanitized_chunks = []
        overall_injection = False
        max_severity = "low"
        
        for chunk in request.chunks:
            has_injection, severity = _detect_injection(chunk.text)
            sanitized_text, notes = _sanitize_text(chunk.text, request.mode, has_injection, severity)
            
            if has_injection:
                overall_injection = True
                if severity == "high":
                    max_severity = "high"
                elif severity == "medium" and max_severity != "high":
                    max_severity = "medium"
            
            sanitized_chunks.append(SanitizedChunk(
                text=sanitized_text,
                notes=notes
            ))
        
        flags = SanitizationFlags(
            injection=overall_injection,
            severity=max_severity
        )
        
        return SanitizeResponse(
            chunks=sanitized_chunks,
            flags=flags,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sanitization failed: {str(e)}")

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "rag_guard_stub"
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "AEGIS-C RAG Guard Stub",
        "description": "Deterministic RAG content sanitization for testing",
        "endpoints": {
            "/sanitize": "POST - Sanitize content chunks",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8022)