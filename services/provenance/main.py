from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import hashlib
import logging
import time
import uuid
from typing import Optional, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AEGIS‑C Provenance")

class SignRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class SignResult(BaseModel):
    algorithm: str
    signature: str
    timestamp: str
    content_hash: str
    verification_url: str

class VerifyRequest(BaseModel):
    content: str
    signature: str
    algorithm: str = "sha256"

class VerifyResult(BaseModel):
    verified: bool
    computed_hash: str
    provided_signature: str
    timestamp: str

# In-memory signature storage (in production, use secure storage)
SIGNATURE_REGISTRY: Dict[str, Dict] = {}

def compute_hash(content: str, algorithm: str = "sha256") -> str:
    """Compute hash of content."""
    if algorithm == "sha256":
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(content.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def create_c2pa_manifest(content_hash: str, metadata: Dict = None) -> Dict:
    """Create a C2PA-style manifest (stub implementation)."""
    manifest = {
        "claim_generator": "AEGIS-C Provenance Service v1.0.0",
        "claim_generator_version": "1.0.0",
        "assertions": [
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "CreativeWork",
                    "identifier": content_hash,
                    "dateCreated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "author": "AEGIS-C System",
                    "metadata": metadata or {}
                }
            },
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "type": "c2pa.created",
                            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "softwareAgent": "AEGIS-C Provenance Service"
                        }
                    ]
                }
            }
        ],
        "signature_info": {
            "algorithm": "sha256",
            "hash": content_hash
        }
    }
    return manifest

@app.post("/sign/text", response_model=SignResult)
async def sign_text(request: SignRequest):
    """Sign text content with provenance metadata."""
    start_time = time.time()
    
    # Compute content hash
    content_hash = compute_hash(request.content)
    
    # Create manifest
    manifest = create_c2pa_manifest(content_hash, request.metadata)
    
    # Create signature (in this stub, the hash is the signature)
    signature = content_hash
    
    # Store signature record
    record_id = str(uuid.uuid4())
    SIGNATURE_REGISTRY[record_id] = {
        "signature": signature,
        "content_hash": content_hash,
        "manifest": manifest,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": "sha256"
    }
    
    processing_time = (time.time() - start_time) * 1000
    
    result = SignResult(
        algorithm="sha256",
        signature=signature,
        timestamp=SIGNATURE_REGISTRY[record_id]["timestamp"],
        content_hash=content_hash,
        verification_url=f"/verify/text/{record_id}"
    )
    
    logger.info(f"Text signed: hash={content_hash[:16]}..., time={processing_time:.1f}ms")
    
    return result

@app.post("/sign/file", response_model=SignResult)
async def sign_file(file: UploadFile = File(...), metadata: Optional[str] = None):
    """Sign uploaded file content with provenance metadata."""
    start_time = time.time()
    
    # Read file content
    content = await file.read()
    content_str = content.decode('utf-8', errors='ignore')
    
    # Parse metadata if provided
    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning("Invalid metadata JSON provided")
    
    # Add file info to metadata
    file_metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content)
    }
    if parsed_metadata:
        file_metadata.update(parsed_metadata)
    
    # Compute hash and create signature
    content_hash = compute_hash(content_str)
    manifest = create_c2pa_manifest(content_hash, file_metadata)
    signature = content_hash
    
    # Store signature record
    record_id = str(uuid.uuid4())
    SIGNATURE_REGISTRY[record_id] = {
        "signature": signature,
        "content_hash": content_hash,
        "manifest": manifest,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": "sha256",
        "filename": file.filename
    }
    
    processing_time = (time.time() - start_time) * 1000
    
    result = SignResult(
        algorithm="sha256",
        signature=signature,
        timestamp=SIGNATURE_REGISTRY[record_id]["timestamp"],
        content_hash=content_hash,
        verification_url=f"/verify/file/{record_id}"
    )
    
    logger.info(f"File signed: {file.filename}, hash={content_hash[:16]}..., time={processing_time:.1f}ms")
    
    return result

@app.post("/verify/text", response_model=VerifyResult)
async def verify_text(request: VerifyRequest):
    """Verify text content against provided signature."""
    start_time = time.time()
    
    # Compute hash of provided content
    computed_hash = compute_hash(request.content, request.algorithm)
    
    # Check if signature matches
    verified = computed_hash == request.signature
    
    processing_time = (time.time() - start_time) * 1000
    
    result = VerifyResult(
        verified=verified,
        computed_hash=computed_hash,
        provided_signature=request.signature,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    
    logger.info(f"Text verification: {verified}, time={processing_time:.1f}ms")
    
    return result

@app.post("/verify/file")
async def verify_file(file: UploadFile = File(...), signature: str = ""):
    """Verify uploaded file against provided signature."""
    start_time = time.time()
    
    # Read file content
    content = await file.read()
    content_str = content.decode('utf-8', errors='ignore')
    
    # Compute hash
    computed_hash = compute_hash(content_str)
    
    # Check verification
    verified = computed_hash == signature
    
    processing_time = (time.time() - start_time) * 1000
    
    result = VerifyResult(
        verified=verified,
        computed_hash=computed_hash,
        provided_signature=signature,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    
    logger.info(f"File verification: {file.filename}, verified={verified}, time={processing_time:.1f}ms")
    
    return result

@app.get("/verify/record/{record_id}")
async def verify_record(record_id: str):
    """Verify a stored signature record."""
    if record_id not in SIGNATURE_REGISTRY:
        raise HTTPException(status_code=404, detail="Signature record not found")
    
    record = SIGNATURE_REGISTRY[record_id]
    
    return {
        "record_id": record_id,
        "verified": True,  # In stub implementation, we assume stored records are valid
        "manifest": record["manifest"],
        "timestamp": record["timestamp"],
        "algorithm": record["algorithm"]
    }

@app.get("/signatures/list")
async def list_signatures(limit: int = 50):
    """List recent signature records."""
    records = list(SIGNATURE_REGISTRY.items())[-limit:] if limit else list(SIGNATURE_REGISTRY.items())
    
    return {
        "total_signatures": len(SIGNATURE_REGISTRY),
        "showing": len(records),
        "records": [
            {
                "record_id": record_id,
                "timestamp": record["timestamp"],
                "algorithm": record["algorithm"],
                "filename": record.get("filename", "text content")
            }
            for record_id, record in records
        ]
    }

@app.delete("/signatures/clear")
async def clear_signatures():
    """Clear all signature records (for testing)."""
    SIGNATURE_REGISTRY.clear()
    logger.info("All signature records cleared")
    return {"success": True, "message": "All signatures cleared"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "service": "provenance",
        "signature_count": len(SIGNATURE_REGISTRY),
        "algorithms": ["sha256", "sha512"]
    }

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AEGIS‑C Provenance",
        "version": "1.0.0",
        "description": "Content provenance signing and verification (C2PA stub)",
        "endpoints": {
            "sign_text": "/sign/text",
            "sign_file": "/sign/file",
            "verify_text": "/verify/text",
            "verify_file": "/verify/file",
            "verify_record": "/verify/record/{record_id}",
            "list_signatures": "/signatures/list",
            "health": "/health"
        }
    }