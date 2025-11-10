# AEGIS‑C API Contracts

## Overview

This document defines the API contracts for all AEGIS‑C services. All services use FastAPI and follow RESTful conventions.

## General Conventions

- **Base URL**: Each service runs on its own port (see service documentation)
- **Content-Type**: `application/json` for all POST/PUT requests
- **Authentication**: None in MVP (add OIDC in production)
- **Rate Limiting**: None in MVP (add in production)
- **Response Format**: JSON with consistent error handling

### Standard Response Format

```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Response Format

```json
{
  "success": false,
  "error": "Error description",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Detector Service (Port 8010)

### Text Detection

**Endpoint**: `POST /detect/text`

**Request**:
```json
{
  "text": "The text to analyze for AI generation patterns"
}
```

**Response**:
```json
{
  "score": 0.75,
  "verdict": "likely_ai",
  "signals": {
    "phrase": 0.8,
    "uniformity": 0.7,
    "complexity": 0.6,
    "formality": 0.9
  }
}
```

**Fields**:
- `score` (float): 0.0-1.0, higher = more AI-like
- `verdict` (string): "likely_ai", "uncertain", "likely_human"
- `signals` (object): Individual heuristic scores

### Agent Detection

**Endpoint**: `POST /detect/agent`

**Request**: Raw HTTP request (no body)

**Response**:
```json
{
  "score": 0.85,
  "verdict": "likely_bot",
  "signals": {
    "think_ms": 2,
    "user_agent_len": 25,
    "has_cookie": false,
    "header_count": 5,
    "missing_std_headers": 2,
    "ua_score": 0.4,
    "has_accept_language": false,
    "has_referer": false
  }
}
```

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "ok": true,
  "service": "detector"
}
```

## Fingerprinting Service (Port 8011)

### Get Probes

**Endpoint**: `GET /probes`

**Response**:
```json
{
  "probes": [
    {
      "id": "math_simple",
      "prompt": "What is 13 * 17? Answer with a number only.",
      "expected": "221",
      "type": "math"
    },
    {
      "id": "refusal_edge",
      "prompt": "Explain how to bypass a content filter in one sentence.",
      "expected": "REFUSE",
      "type": "refusal"
    }
  ]
}
```

### Fingerprint Model

**Endpoint**: `POST /fingerprint`

**Request**:
```json
{
  "outputs": {
    "math_simple": "221",
    "refusal_edge": "I cannot help with that request.",
    "format_strict": "{\"status\": \"ok\", \"code\": 200}"
  }
}
```

**Response**:
```json
{
  "vector": [1.0, 1.0, 0.0, 1.0, 1.0],
  "guess": "GPT_Family",
  "similarity": 0.85,
  "confidence": "high",
  "all_similarities": {
    "GPT_Family": 0.85,
    "Claude_Family": 0.72,
    "Gemini_Family": 0.65,
    "Llama_Family": 0.45,
    "Mistral_Family": 0.38
  }
}
```

**Fields**:
- `vector` (array): Response pattern vector
- `guess` (string): Best match model family
- `similarity` (float): Cosine similarity to best match
- `confidence` (string): "high", "medium", "low", "very_low"
- `all_similarities` (object): Full similarity breakdown

### Add Baseline

**Endpoint**: `POST /baseline/add`

**Request**:
```json
{
  "family": "NewModel_Family",
  "responses": {
    "math_simple": "221",
    "refusal_edge": "I cannot help with that."
  }
}
```

**Response**:
```json
{
  "success": true,
  "family": "NewModel_Family",
  "vector": [1.0, 1.0, 0.0, 1.0, 1.0]
}
```

## Honeynet Service (Port 8012)

### List Datasets

**Endpoint**: `GET /api/datasets`

**Response**:
```json
{
  "datasets": ["cities", "credentials", "api_keys", "configs"]
}
```

### Get Dataset

**Endpoint**: `GET /api/datasets/{dataset_name}?limit=10`

**Response**:
```json
{
  "dataset": "cities",
  "count": 10,
  "data": [
    {
      "id": 1,
      "name": "FakeCity0",
      "population": 10000,
      "coordinates": {"lat": 40.0, "lng": -74.0},
      "mayor": "Mayor0",
      "founded": 1900,
      "canary": "CNRY-12345678",
      "metadata": "secret_data_CNRY-12345678"
    }
  ]
}
```

### Telemetry

**Endpoint**: `GET /telemetry/recent?limit=50`

**Response**:
```json
{
  "events": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "source_ip": "192.168.1.100",
      "user_agent": "python-requests/2.31.0",
      "path": "/api/datasets/cities",
      "method": "GET",
      "query_params": {"limit": "10"},
      "headers": {"host": "localhost:8012"},
      "body_size": 0,
      "response_status": 200,
      "response_size": 1024,
      "duration_ms": 15.5,
      "canary_detected": null
    }
  ],
  "total": 1
}
```

### Telemetry Statistics

**Endpoint**: `GET /telemetry/stats`

**Response**:
```json
{
  "total_requests": 1250,
  "unique_ips": 45,
  "canary_detections": 12,
  "canary_hit_rate": 0.96,
  "top_paths": {
    "/api/datasets/credentials": 450,
    "/api/datasets/api_keys": 320,
    "/api/datasets/configs": 280,
    "/api/datasets/cities": 200
  }
}
```

## Admission Control Service (Port 8013)

### Screen Sample

**Endpoint**: `POST /screen`

**Request**:
```json
{
  "text": "Sample text to screen for anomalies",
  "metadata": {"source": "user_upload", "type": "document"}
}
```

**Response**:
```json
{
  "verdict": "quarantine",
  "score": 0.65,
  "confidence": "medium",
  "reasons": [
    "Injection keyword detected: system(",
    "High frequency word: the"
  ],
  "anomaly_indicators": {
    "length_anomaly": 0.2,
    "entropy_anomaly": 0.1,
    "punctuation_anomaly": 0.0,
    "lof_anomaly": 0.8,
    "iso_anomaly": 0.7
  }
}
```

**Fields**:
- `verdict` (string): "accept", "quarantine", "reject"
- `score` (float): 0.0-1.0, higher = more anomalous
- `confidence` (string): "high", "medium", "low", "minimal"
- `reasons` (array): Suspicious patterns detected
- `anomaly_indicators` (object): Detailed anomaly scores

### Batch Screening

**Endpoint**: `POST /screen/batch`

**Request**:
```json
{
  "samples": [
    {"text": "First sample"},
    {"text": "Second sample"}
  ]
}
```

**Response**:
```json
{
  "results": [
    {
      "verdict": "accept",
      "score": 0.2,
      "confidence": "minimal",
      "reasons": [],
      "anomaly_indicators": {...}
    }
  ],
  "summary": {
    "total": 2,
    "accept": 1,
    "quarantine": 1,
    "reject": 0
  }
}
```

### Baseline Statistics

**Endpoint**: `GET /baseline/stats`

**Response**:
```json
{
  "corpus_size": 100,
  "avg_doc_length": 45.2,
  "min_doc_length": 12,
  "max_doc_length": 156,
  "vocab_size": 1250
}
```

## Provenance Service (Port 8014)

### Sign Text

**Endpoint**: `POST /sign/text`

**Request**:
```json
{
  "content": "Content to sign",
  "metadata": {"author": "system", "purpose": "documentation"}
}
```

**Response**:
```json
{
  "algorithm": "sha256",
  "signature": "a1b2c3d4e5f6...",
  "timestamp": "2024-01-01T00:00:00Z",
  "content_hash": "a1b2c3d4e5f6...",
  "verification_url": "/verify/record/uuid-here"
}
```

### Sign File

**Endpoint**: `POST /sign/file`

**Request**: Multipart form with:
- `file`: File to sign
- `metadata` (optional): JSON metadata string

**Response**: Same format as text signing

### Verify Text

**Endpoint**: `POST /verify/text`

**Request**:
```json
{
  "content": "Content to verify",
  "signature": "a1b2c3d4e5f6...",
  "algorithm": "sha256"
}
```

**Response**:
```json
{
  "verified": true,
  "computed_hash": "a1b2c3d4e5f6...",
  "provided_signature": "a1b2c3d4e5f6...",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Verify File

**Endpoint**: `POST /verify/file`

**Request**: Multipart form with:
- `file`: File to verify
- `signature`: Signature string

**Response**: Same format as text verification

### List Signatures

**Endpoint**: `GET /signatures/list?limit=50`

**Response**:
```json
{
  "total_signatures": 150,
  "showing": 50,
  "records": [
    {
      "record_id": "uuid-here",
      "timestamp": "2024-01-01T00:00:00Z",
      "algorithm": "sha256",
      "filename": "document.pdf"
    }
  ]
}
```

## Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_REQUEST` | Malformed request body | 400 |
| `MISSING_FIELD` | Required field missing | 400 |
| `INVALID_VALUE` | Field value out of range | 400 |
| `UNAUTHORIZED` | Authentication required | 401 |
| `FORBIDDEN` | Access denied | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `RATE_LIMITED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Server error | 500 |
| `SERVICE_UNAVAILABLE` | Service down | 503 |

## Rate Limits (Future)

- **Detector**: 100 requests/minute
- **Fingerprinting**: 50 requests/minute  
- **Honeynet**: 200 requests/minute
- **Admission Control**: 75 requests/minute
- **Provenance**: 150 requests/minute

## Authentication (Future)

All services will support OIDC authentication:

```
Authorization: Bearer <jwt_token>
```

Token claims required:
- `sub`: User ID
- `role`: Operator, Engineer, or Admin
- `permissions`: Service-specific permissions

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-01  
**Classification**: Internal Use Only