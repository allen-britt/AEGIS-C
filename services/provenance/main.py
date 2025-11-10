"""
AEGIS‑C Provenance Service
===========================

C2PA signing/verification stub with standardized guard.
"""

import os
import sys
import json
import hashlib
import time
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import setup_service_guard, record_detection_score, logger, require_api_key

# Service Configuration
SERVICE_NAME = "provenance"
SERVICE_PORT = int(os.getenv("PROVENANCE_PORT", "8014"))

# Initialize FastAPI App
app = FastAPI(title="AEGIS‑C Provenance Service")

# Setup standardized guard
setup_service_guard(app, SERVICE_NAME)

# Data models
class ClaimData(BaseModel):
    content: str
    author: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SignatureRequest(BaseModel):
    claim: ClaimData
    private_key: Optional[str] = None  # In production, use proper key management

class VerificationRequest(BaseModel):
    content: str
    signature: str
    public_key: Optional[str] = None

class SignatureResult(BaseModel):
    signature: str
    claim_hash: str
    timestamp: str
    verified: bool

class VerificationResult(BaseModel):
    valid: bool
    claim_hash: str
    verification_timestamp: str
    details: Dict[str, Any]

# In-memory signature storage (replace with proper database in production)
SIGNATURE_DB: Dict[str, SignatureResult] = {}

def generate_claim_hash(claim: ClaimData) -> str:
    """Generate hash for claim data"""
    claim_data = {
        "content": claim.content,
        "author": claim.author,
        "timestamp": claim.timestamp or datetime.now(timezone.utc).isoformat(),
        "metadata": claim.metadata or {}
    }
    
    claim_json = json.dumps(claim_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(claim_json.encode()).hexdigest()

def sign_claim(claim: ClaimData) -> SignatureResult:
    """Sign a claim (stub implementation)"""
    claim_hash = generate_claim_hash(claim)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # In production, use actual cryptographic signing
    # For now, create a mock signature
    signature_data = f"{claim_hash}:{timestamp}:AEGISC_STUB_SIGNATURE"
    signature = hashlib.sha256(signature_data.encode()).hexdigest()
    
    result = SignatureResult(
        signature=signature,
        claim_hash=claim_hash,
        timestamp=timestamp,
        verified=True  # In production, this would be the result of actual signing
    )
    
    # Store signature
    SIGNATURE_DB[signature] = result
    
    logger.info(f"Claim signed", claim_hash=claim_hash, author=claim.author)
    return result

def verify_signature(content: str, signature: str) -> VerificationResult:
    """Verify a signature (stub implementation)"""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Check if signature exists in our database
    if signature not in SIGNATURE_DB:
        return VerificationResult(
            valid=False,
            claim_hash="unknown",
            verification_timestamp=timestamp,
            details={"error": "Signature not found in database"}
        )
    
    stored_result = SIGNATURE_DB[signature]
    
    # In production, perform actual cryptographic verification
    # For now, just check if it exists in our database
    valid = stored_result.verified
    
    details = {
        "original_timestamp": stored_result.timestamp,
        "signature_algorithm": "SHA256_STUB",
        "verification_method": "database_lookup"
    }
    
    logger.info(f"Signature verification", valid=valid, signature=signature[:16] + "...")
    
    return VerificationResult(
        valid=valid,
        claim_hash=stored_result.claim_hash,
        verification_timestamp=timestamp,
        details=details
    )

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AEGIS‑C Provenance",
        "version": "1.0.0",
        "description": "C2PA signing/verification stub service",
        "endpoints": {
            "sign": "/sign",
            "verify": "/verify",
            "signatures": "/signatures",
            "health": "/health",
            "metrics": "/metrics",
            "secure_ping": "/secure/ping"
        },
        "stored_signatures": len(SIGNATURE_DB)
    }

@app.post("/sign")
async def sign_claim_endpoint(request: SignatureRequest):
    """Sign a claim with provenance data"""
    try:
        # Set timestamp if not provided
        if not request.claim.timestamp:
            request.claim.timestamp = datetime.now(timezone.utc).isoformat()
        
        result = sign_claim(request.claim)
        
        logger.info(f"Claim signed successfully", author=request.claim.author)
        return result
        
    except Exception as e:
        logger.error(f"Claim signing error: {e}")
        raise

@app.post("/verify")
async def verify_signature_endpoint(request: VerificationRequest):
    """Verify a signature"""
    try:
        result = verify_signature(request.content, request.signature)
        
        logger.info(f"Signature verification completed", valid=result.valid)
        return result
        
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        raise

@app.get("/signatures")
async def list_signatures():
    """List all stored signatures"""
    return {
        "total_signatures": len(SIGNATURE_DB),
        "signatures": [
            {
                "signature": sig[:16] + "...",  # Truncated for security
                "claim_hash": result.claim_hash,
                "timestamp": result.timestamp,
                "verified": result.verified
            }
            for sig, result in SIGNATURE_DB.items()
        ]
    }

@app.get("/signatures/{signature_hash}")
async def get_signature_details(signature_hash: str):
    """Get details for a specific signature"""
    # Find signature by hash (simplified search)
    for sig, result in SIGNATURE_DB.items():
        if signature_hash in sig or signature_hash == result.claim_hash:
            return {
                "signature": sig[:16] + "...",
                "claim_hash": result.claim_hash,
                "timestamp": result.timestamp,
                "verified": result.verified
            }
    
    raise HTTPException(status_code=404, detail="Signature not found")

# Protected endpoints
@app.post("/secure/sign", dependencies=[Depends(require_api_key)])
async def secure_sign_claim(request: SignatureRequest):
    """Protected claim signing endpoint"""
    return await sign_claim_endpoint(request)

@app.post("/secure/verify", dependencies=[Depends(require_api_key)])
async def secure_verify_signature(request: VerificationRequest):
    """Protected signature verification endpoint"""
    return await verify_signature_endpoint(request)

@app.delete("/secure/signatures/{signature_hash}", dependencies=[Depends(require_api_key)])
async def delete_signature(signature_hash: str):
    """Delete a signature (admin only)"""
    # Find and delete signature
    to_delete = None
    for sig, result in SIGNATURE_DB.items():
        if signature_hash in sig or signature_hash == result.claim_hash:
            to_delete = sig
            break
    
    if not to_delete:
        raise HTTPException(status_code=404, detail="Signature not found")
    
    del SIGNATURE_DB[to_delete]
    logger.info(f"Signature deleted", signature_hash=signature_hash)
    
    return {"success": True, "message": "Signature deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)