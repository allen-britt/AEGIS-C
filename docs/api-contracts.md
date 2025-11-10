# AEGIS‑C API Contracts

## Authentication

All API endpoints require authentication via the `x-api-key` header:
```
x-api-key: changeme-dev
```

Protected endpoints are prefixed with `/secure/` and will return 401 without valid authentication.

## Core Services

### Detector Service (Port 8010)

#### Detect AI-Generated Text
```http
POST /detect/text
Content-Type: application/json

{
  "text": "This is however a comprehensive analysis..."
}
```

**Response**:
```json
{
  "score": 0.75,
  "verdict": "likely_ai",
  "signals": {
    "char_count": 45,
    "word_count": 8,
    "avg_word_length": 5.6
  }
}
```

#### Detect AI Agent
```http
POST /detect/agent
Content-Type: application/json

{
  "user_agent": "curl/7.68.0",
  "cookie": "",
  "accept_lang": "",
  "referer": ""
}
```

**Response**:
```json
{
  "score": 0.3,
  "verdict": "likely_human",
  "signals": {
    "ua_length": 12,
    "has_cookie": false,
    "has_accept_language": false,
    "has_referer": false
  }
}
```

### Fingerprint Service (Port 8011)

#### Get Challenge Probes
```http
GET /probes
```

**Response**:
```json
{
  "probes": [
    {
      "id": "math_basic",
      "prompt": "What is 2+2?",
      "category": "mathematics",
      "expected_type": "numeric"
    }
  ]
}
```

#### Fingerprint Model
```http
POST /fingerprint
Content-Type: application/json

{
  "responses": {
    "math_basic": "4",
    "reasoning_simple": "The answer is 42"
  }
}
```

**Response**:
```json
{
  "family": "GPT_Family",
  "confidence": 0.85,
  "similarity": 0.92,
  "vector": [0.9, 0.8, 0.1, 0.9, 0.2]
}
```

### Honeynet Service (Port 8012)

#### Generate Canary Token
```http
POST /canaries/generate
```

**Response**:
```json
{
  "token": "canary_abc123def456",
  "type": "data_leak",
  "expires": "2024-12-31T23:59:59Z"
}
```

#### Access Fake Dataset
```http
GET /api/datasets/cities
```

**Response**:
```json
{
  "data": [
    {"name": "New York", "population": 8000000, "canary": "canary_abc123def456"}
  ],
  "source": "fake_database",
  "contains_canaries": true
}
```

### Admission Service (Port 8013)

#### Validate Data
```http
POST /validate
Content-Type: application/json

{
  "data": "Training data content...",
  "source": "user_upload",
  "metadata": {"format": "json"}
}
```

**Response**:
```json
{
  "allowed": false,
  "risk_score": 0.8,
  "threats": [
    "Poisoning pattern: ignore previous training",
    "Suspicious term: backdoor"
  ],
  "recommendations": [
    "QUARANTINE: High risk of data poisoning",
    "Manual review required before processing"
  ]
}
```

#### List Quarantined Items
```http
GET /quarantine/list
```

**Response**:
```json
{
  "quarantined_items": 2,
  "items": [
    {
      "id": "quarantine_001",
      "timestamp": "2024-01-15T10:30:00Z",
      "risk_score": 0.8,
      "threats": ["Data poisoning"],
      "source": "user_upload"
    }
  ]
}
```

### Provenance Service (Port 8014)

#### Sign Claim
```http
POST /sign
Content-Type: application/json

{
  "claim": {
    "content": "AI model output...",
    "author": "detector_service",
    "metadata": {"model_version": "1.0"}
  }
}
```

**Response**:
```json
{
  "signature": "sha256:abc123...",
  "claim_hash": "hash_456...",
  "timestamp": "2024-01-15T10:30:00Z",
  "verified": true
}
```

#### Verify Signature
```http
POST /verify
Content-Type: application/json

{
  "content": "AI model output...",
  "signature": "sha256:abc123..."
}
```

**Response**:
```json
{
  "valid": true,
  "claim_hash": "hash_456...",
  "verification_timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "original_timestamp": "2024-01-15T10:25:00Z",
    "signature_algorithm": "SHA256_STUB",
    "verification_method": "database_lookup"
  }
}
```

### Hardware Sentinel (Port 8016)

#### Get Hardware Status
```http
GET /status
```

**Response**:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "gpu_count": 2,
  "gpus": [
    {
      "gpu_id": 0,
      "name": "NVIDIA RTX 4090",
      "temperature": 65.0,
      "utilization": 75.0,
      "memory_used": 18.5,
      "memory_total": 24.0,
      "power_usage": 350.0,
      "ecc_errors": {"uncorrected_volatile": 0}
    }
  ],
  "system_health": "healthy",
  "anomaly_score": null
}
```

#### Execute Policy Action
```http
POST /policy/execute?action=drain&target_gpu=0&reason=High%20temperature
```

**Response**:
```json
{
  "success": true,
  "action": "drain",
  "target_gpu": 0,
  "reason": "High temperature",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Intelligence Service (Port 8018)

#### Get AI Vulnerabilities
```http
GET /vulnerabilities/ai?limit=50
```

**Response**:
```json
[
  {
    "cve_id": "CVE-2024-1234",
    "title": "TensorFlow Model Injection Vulnerability",
    "severity": "HIGH",
    "cvss_score": 8.5,
    "ai_impact_score": 0.85,
    "published_date": "2024-03-15T12:00:00Z",
    "description": "Remote code execution vulnerability..."
  }
]
```

#### Analyze AI Threats
```http
GET /analysis/ai-threats
```

**Response**:
```json
{
  "total_ai_vulnerabilities": 2,
  "critical_vulnerabilities": 1,
  "high_vulnerabilities": 1,
  "average_cvss_score": 8.85,
  "most_affected_products": [
    {"product": "tensorflow", "count": 1},
    {"product": "pytorch", "count": 1}
  ],
  "threat_trends": {
    "monthly_vulnerability_counts": {"2024-03": 1, "2024-04": 1},
    "trend_direction": "increasing"
  }
}
```

## Standard Endpoints

All services implement these standard endpoints:

### Health Check
```http
GET /health
```

**Response**:
```json
{
  "ok": true,
  "service": "detector",
  "version": "1.0.0",
  "timestamp": 1642248600.0
}
```

### Metrics
```http
GET /metrics
```

Returns Prometheus-formatted metrics for monitoring.

### Secure Ping
```http
GET /secure/ping
x-api-key: changeme-dev
```

**Response**:
```json
{
  "pong": true,
  "service": "detector"
}
```

## Error Handling

### Standard Error Response
```json
{
  "error": "Invalid API key",
  "detail": "Authentication failed",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

## Rate Limiting

All services implement rate limiting:
- Default: 100 requests per minute per API key
- Burst: 10 requests per second
- Protected endpoints: 50 requests per minute

*This API contract will be expanded with detailed schemas, examples, and client libraries.*