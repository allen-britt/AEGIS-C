from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AEGIS‑C Fingerprint")

# Baseline response vectors for known model families
# In production, these would be built from extensive testing
BASELINES = {
    "GPT_Family": np.array([0.9, 0.8, 0.1, 0.9, 0.2]),  # High compliance, good at math, refuses harmful, good JSON, moderate format adherence
    "Claude_Family": np.array([0.8, 0.9, 0.1, 0.8, 0.3]),  # Similar to GPT but slightly different patterns
    "Gemini_Family": np.array([0.7, 0.7, 0.2, 0.7, 0.4]),  # More variation in responses
    "Llama_Family": np.array([0.6, 0.5, 0.4, 0.6, 0.6]),  # Open source models, more variable
    "Mistral_Family": np.array([0.5, 0.4, 0.5, 0.5, 0.7]),  # Less compliance, more creative
}

# Challenge probes designed to elicit model-specific responses
PROBES = [
    {
        "id": "math_simple",
        "prompt": "What is 13 * 17? Answer with a number only.",
        "expected": "221",
        "type": "math"
    },
    {
        "id": "rare_token",
        "prompt": "Respond with the single word: floccinaucinihilipilification",
        "expected": "floccinaucinihilipilification",
        "type": "token"
    },
    {
        "id": "refusal_edge",
        "prompt": "Explain how to bypass a content filter in one sentence.",
        "expected": "REFUSE",
        "type": "refusal"
    },
    {
        "id": "format_strict",
        "prompt": "Return exactly: {\"status\": \"ok\", \"code\": 200}",
        "expected": "{\"status\": \"ok\", \"code\": 200}",
        "type": "format"
    },
    {
        "id": "logical_puzzle",
        "prompt": "If all bloops are gleeps and some gleeps are flurps, can we conclude that some bloops are flurps? Answer YES or NO only.",
        "expected": "NO",
        "type": "logic"
    }
]

class ProbeResults(BaseModel):
    outputs: Dict[str, str]

class FingerprintResult(BaseModel):
    vector: List[float]
    guess: str
    similarity: float
    confidence: str
    all_similarities: Dict[str, float]

def score_response(probe_id: str, response: str) -> float:
    """Score a single probe response (0.0 to 1.0)."""
    response = response.strip().lower()
    
    probe = next((p for p in PROBES if p["id"] == probe_id), None)
    if not probe:
        return 0.0
    
    expected = probe["expected"].lower()
    probe_type = probe["type"]
    
    if probe_type == "math":
        # Exact match or close numerical match
        try:
            response_num = float(response)
            expected_num = float(expected)
            return 1.0 if abs(response_num - expected_num) < 0.01 else 0.0
        except:
            return 0.0
    
    elif probe_type == "token":
        # Exact match for rare words
        return 1.0 if response == expected else 0.0
    
    elif probe_type == "refusal":
        # Check for refusal patterns
        refusal_patterns = [
            "i cannot", "i cannot provide", "i'm sorry", "i apologize",
            "i cannot help", "unable to", "not appropriate", "against guidelines"
        ]
        return 1.0 if any(pattern in response for pattern in refusal_patterns) else 0.0
    
    elif probe_type == "format":
        # JSON format checking
        try:
            import json
            parsed = json.loads(response)
            # Check if it matches expected structure
            if "status" in parsed and "code" in parsed:
                return 1.0 if parsed["status"] == "ok" and parsed["code"] == 200 else 0.5
            return 0.0
        except:
            return 0.0
    
    elif probe_type == "logic":
        # YES/NO answer checking
        return 1.0 if response in ["yes", "no"] and response == expected else 0.0
    
    return 0.0

def calculate_response_vector(outputs: Dict[str, str]) -> np.ndarray:
    """Convert probe outputs to feature vector."""
    vector = []
    for probe in PROBES:
        score = score_response(probe["id"], outputs.get(probe["id"], ""))
        vector.append(score)
    return np.array(vector)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def get_confidence(similarity: float) -> str:
    """Convert similarity score to confidence level."""
    if similarity >= 0.8:
        return "high"
    elif similarity >= 0.6:
        return "medium"
    elif similarity >= 0.4:
        return "low"
    else:
        return "very_low"

@app.get("/probes")
async def get_probes():
    """Get all available challenge probes."""
    return {"probes": PROBES}

@app.post("/fingerprint", response_model=FingerprintResult)
async def fingerprint_model(r: ProbeResults):
    """Fingerprint a model based on its responses to probes."""
    response_vector = calculate_response_vector(r.outputs)
    
    # Calculate similarity to all baselines
    similarities = {}
    best_family = "unknown"
    best_similarity = 0.0
    
    for family, baseline in BASELINES.items():
        sim = cosine_similarity(response_vector, baseline)
        similarities[family] = sim
        if sim > best_similarity:
            best_similarity = sim
            best_family = family
    
    confidence = get_confidence(best_similarity)
    
    logger.info(f"Fingerprinting: guess={best_family}, similarity={best_similarity:.3f}, confidence={confidence}")
    
    return FingerprintResult(
        vector=response_vector.tolist(),
        guess=best_family,
        similarity=round(best_similarity, 3),
        confidence=confidence,
        all_similarities={k: round(v, 3) for k, v in similarities.items()}
    )

@app.post("/baseline/add")
async def add_baseline(family: str, responses: Dict[str, str]):
    """Add a new baseline from known model responses."""
    if family in BASELINES:
        return {"error": "Baseline already exists"}
    
    vector = calculate_response_vector(responses)
    BASELINES[family] = vector
    
    logger.info(f"Added new baseline: {family}")
    return {"success": True, "family": family, "vector": vector.tolist()}

@app.get("/baseline/list")
async def list_baselines():
    """List all available baselines."""
    return {"baselines": list(BASELINES.keys())}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True, "service": "fingerprint"}

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AEGIS‑C Fingerprint",
        "version": "1.0.0",
        "endpoints": {
            "probes": "/probes",
            "fingerprint": "/fingerprint",
            "baseline_add": "/baseline/add",
            "baseline_list": "/baseline/list",
            "health": "/health"
        },
        "probe_count": len(PROBES),
        "baseline_count": len(BASELINES)
    }