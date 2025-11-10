from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import numpy as np
import logging
from typing import List, Dict, Optional
import hashlib
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AEGIS‑C Admission Control")

# Demo clean baseline corpus (expand in production)
BASELINE_CORPUS = [
    "Weather report: clear skies expected tomorrow with high of 75°F.",
    "System log: service started successfully on port 8080 at 14:30 UTC.",
    "User guide: click the start button to begin the installation process.",
    "Financial report: Q3 revenue increased by 12% compared to previous quarter.",
    "Technical documentation: API endpoints require authentication via Bearer token.",
    "Meeting notes: discussed project timeline and resource allocation.",
    "Security alert: unusual login activity detected from unknown IP address.",
    "Performance metrics: CPU usage at 45%, memory utilization at 67%.",
    "Customer feedback: overall satisfaction rating improved to 4.2/5.0.",
    "Update notes: patch applied to fix vulnerability in authentication module."
]

# Initialize ML models
try:
    vectorizer = TfidfVectorizer(
        min_df=1, 
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit on baseline corpus
    X_baseline = vectorizer.fit_transform(BASELINE_CORPUS)
    
    # Outlier detection models
    lof_detector = LocalOutlierFactor(
        n_neighbors=min(5, len(BASELINE_CORPUS)-1), 
        novelty=True, 
        contamination=0.1
    )
    lof_detector.fit(X_baseline)
    
    iso_detector = IsolationForest(
        contamination=0.1, 
        random_state=42
    )
    iso_detector.fit(X_baseline.toarray())
    
    logger.info("Admission Control ML models initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize ML models: {e}")
    vectorizer = None
    lof_detector = None
    iso_detector = None

class Sample(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class BatchSample(BaseModel):
    samples: List[Sample]

class ScreeningResult(BaseModel):
    verdict: str  # accept, quarantine, reject
    score: float
    confidence: str
    reasons: List[str]
    anomaly_indicators: Dict[str, float]

class BatchScreeningResult(BaseModel):
    results: List[ScreeningResult]
    summary: Dict[str, int]

def detect_suspicious_patterns(text: str) -> List[str]:
    """Detect suspicious patterns that might indicate poisoning."""
    suspicious = []
    text_lower = text.lower()
    
    # Check for injection patterns
    injection_keywords = [
        "inject", "backdoor", "trigger", "payload", "exploit",
        "system(", "exec(", "eval(", "subprocess", "os.system",
        "__import__", "getattr", "setattr", "delattr",
        "drop table", "union select", "script>", "javascript:"
    ]
    
    for keyword in injection_keywords:
        if keyword in text_lower:
            suspicious.append(f"Injection keyword detected: {keyword}")
    
    # Check for unusual repetition
    words = text_lower.split()
    if len(words) > 10:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Flag if any word appears more than 30% of the time
        for word, count in word_freq.items():
            if count / len(words) > 0.3 and len(word) > 3:
                suspicious.append(f"High frequency word: {word}")
                break
    
    # Check for encoded content
    if any(pattern in text for pattern in ["base64", "hex:", "b64:", "\\x", "%u"]):
        suspicious.append("Encoded content detected")
    
    # Check for very long tokens (might be encoded data)
    tokens = text.split()
    for token in tokens:
        if len(token) > 100:
            suspicious.append("Unusually long token detected")
            break
    
    return suspicious

def calculate_linguistic_anomalies(text: str) -> Dict[str, float]:
    """Calculate linguistic anomaly scores."""
    if not text.strip():
        return {"empty_text": 1.0}
    
    anomalies = {}
    
    # Length-based anomalies
    word_count = len(text.split())
    char_count = len(text)
    
    # Compare with baseline statistics
    baseline_lengths = [len(doc.split()) for doc in BASELINE_CORPUS]
    avg_baseline_length = np.mean(baseline_lengths)
    std_baseline_length = np.std(baseline_lengths)
    
    # Z-score for length
    if std_baseline_length > 0:
        length_z = abs(word_count - avg_baseline_length) / std_baseline_length
        anomalies["length_anomaly"] = min(1.0, length_z / 3.0)
    else:
        anomalies["length_anomaly"] = 0.0
    
    # Entropy-based anomaly
    char_freq = {}
    for char in text.lower():
        char_freq[char] = char_freq.get(char, 0) + 1
    
    entropy = -sum((freq/len(text)) * np.log2(freq/len(text)) 
                   for freq in char_freq.values() if freq > 0)
    
    # Typical text entropy is around 4-5
    anomalies["entropy_anomaly"] = min(1.0, abs(entropy - 4.5) / 2.0)
    
    # Punctuation ratio
    punct_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    if char_count > 0:
        punct_ratio = punct_count / char_count
        anomalies["punctuation_anomaly"] = min(1.0, abs(punct_ratio - 0.1) * 10)
    
    return anomalies

def get_ml_anomaly_score(text: str) -> Dict[str, float]:
    """Get ML-based anomaly scores."""
    if not vectorizer or not lof_detector or not iso_detector:
        return {"ml_unavailable": 1.0}
    
    try:
        # Vectorize text
        X = vectorizer.transform([text])
        
        # LOF score (negative because LOF returns negative for outliers)
        lof_score = float(-lof_detector.score_samples(X)[0])
        
        # Isolation Forest score
        iso_score = float(-iso_detector.decision_function(X.toarray())[0])
        
        return {
            "lof_anomaly": min(1.0, max(0.0, lof_score)),
            "iso_anomaly": min(1.0, max(0.0, iso_score))
        }
    
    except Exception as e:
        logger.error(f"ML scoring failed: {e}")
        return {"ml_error": 1.0}

def calculate_overall_score(patterns: List[str], linguistic: Dict, ml: Dict) -> float:
    """Calculate overall anomaly score."""
    score = 0.0
    
    # Pattern-based scoring (highest weight)
    if patterns:
        score += len(patterns) * 0.3
        score = min(score, 0.6)  # Cap at 0.6 for patterns
    
    # Linguistic anomalies
    if linguistic:
        avg_linguistic = np.mean(list(linguistic.values()))
        score += avg_linguistic * 0.25
    
    # ML anomalies
    if ml and "ml_unavailable" not in ml and "ml_error" not in ml:
        avg_ml = np.mean(list(ml.values()))
        score += avg_ml * 0.15
    
    return min(1.0, score)

def get_confidence_level(score: float) -> str:
    """Convert score to confidence level."""
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "medium"
    elif score >= 0.4:
        return "low"
    else:
        return "minimal"

@app.post("/screen", response_model=ScreeningResult)
async def screen_sample(sample: Sample):
    """Screen a single sample for anomalies."""
    start_time = time.time()
    
    # Run all detection methods
    suspicious_patterns = detect_suspicious_patterns(sample.text)
    linguistic_anomalies = calculate_linguistic_anomalies(sample.text)
    ml_anomalies = get_ml_anomaly_score(sample.text)
    
    # Calculate overall score
    score = calculate_overall_score(suspicious_patterns, linguistic_anomalies, ml_anomalies)
    
    # Determine verdict
    if score >= 0.7:
        verdict = "reject"
    elif score >= 0.4:
        verdict = "quarantine"
    else:
        verdict = "accept"
    
    # Build result
    result = ScreeningResult(
        verdict=verdict,
        score=round(score, 3),
        confidence=get_confidence_level(score),
        reasons=suspicious_patterns,
        anomaly_indicators={**linguistic_anomalies, **ml_anomalies}
    )
    
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Sample screened: {verdict} (score={score:.3f}, time={processing_time:.1f}ms)")
    
    return result

@app.post("/screen/batch", response_model=BatchScreeningResult)
async def screen_batch(batch: BatchSample):
    """Screen multiple samples."""
    results = []
    
    for sample in batch.samples:
        result = await screen_sample(sample)
        results.append(result)
    
    # Summary statistics
    summary = {
        "total": len(results),
        "accept": len([r for r in results if r.verdict == "accept"]),
        "quarantine": len([r for r in results if r.verdict == "quarantine"]),
        "reject": len([r for r in results if r.verdict == "reject"])
    }
    
    return BatchScreeningResult(results=results, summary=summary)

@app.post("/baseline/add")
async def add_to_baseline(texts: List[str]):
    """Add new texts to the baseline corpus."""
    global BASELINE_CORPUS, vectorizer, lof_detector, iso_detector
    
    try:
        # Add to baseline
        BASELINE_CORPUS.extend(texts)
        
        # Retrain models
        X_baseline = vectorizer.fit_transform(BASELINE_CORPUS)
        lof_detector.fit(X_baseline)
        iso_detector.fit(X_baseline.toarray())
        
        logger.info(f"Added {len(texts)} samples to baseline")
        return {"success": True, "baseline_size": len(BASELINE_CORPUS)}
    
    except Exception as e:
        logger.error(f"Failed to update baseline: {e}")
        return {"success": False, "error": str(e)}

@app.get("/baseline/stats")
async def get_baseline_stats():
    """Get baseline corpus statistics."""
    if not BASELINE_CORPUS:
        return {"error": "No baseline data"}
    
    doc_lengths = [len(doc.split()) for doc in BASELINE_CORPUS]
    
    return {
        "corpus_size": len(BASELINE_CORPUS),
        "avg_doc_length": round(np.mean(doc_lengths), 2),
        "min_doc_length": min(doc_lengths),
        "max_doc_length": max(doc_lengths),
        "vocab_size": len(vectorizer.vocabulary_) if vectorizer else 0
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "service": "admission",
        "baseline_size": len(BASELINE_CORPUS),
        "ml_models_initialized": all([vectorizer, lof_detector, iso_detector])
    }

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AEGIS‑C Admission Control",
        "version": "1.0.0",
        "description": "Data admission control for poisoning prevention",
        "endpoints": {
            "screen": "/screen",
            "screen_batch": "/screen/batch",
            "baseline_add": "/baseline/add",
            "baseline_stats": "/baseline/stats",
            "health": "/health"
        }
    }