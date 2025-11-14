"""
AEGIS‑C Adaptive Risk Scoring
=============================

Explainable probabilistic risk assessment with feature attribution.
"""

import os
import sys
import numpy as np
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
import random
import aiohttp

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
try:
    from service_guard import logger, record_detection_score
except ImportError:
    # Fallback if service_guard is not available
    logger = logging.getLogger(__name__)
    def record_detection_score(score, verdict):
        pass

# Feature configuration
FEATURES = [
    "ai_text_score",      # AI detection confidence
    "probe_sim",          # Fingerprint similarity score
    "canary_echo",        # Canary token activation count
    "rag_injection",      # RAG injection attempt score
    "ecc_delta",          # Hardware ECC error delta
    "latency_ms",         # Request latency anomaly
    "agent_anomaly",      # Agent behavior anomaly
    "data_poison_risk",   # Data poisoning risk score
    "threat_intel_score", # Threat intelligence severity (enhanced with CVE data)
    "hardware_temp"       # Hardware temperature anomaly
]

# Initial weights (learned over time)
WEIGHTS = np.array([
    1.2,   # ai_text_score
    1.5,   # probe_sim
    2.0,   # canary_echo
    1.7,   # rag_injection
    1.6,   # ecc_delta
    0.8,   # latency_ms
    1.1,   # agent_anomaly
    1.4,   # data_poison_risk
    1.3,   # threat_intel_score (will be dynamically adjusted with CVE data)
    0.9    # hardware_temp
])

BIAS = -3.0  # Base risk bias

# ---- Configuration ----
VULN_DB_URL = "http://localhost:8019"  # URL for vuln_db service

# ---- CVE Data Fetching ----
async def fetch_critical_cves_from_vuln_db() -> List[Dict[str, Any]]:
    """Fetch critical CVEs from vuln_db service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{VULN_DB_URL}/threats/critical") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("critical_threats", [])
                else:
                    print(f"[{datetime.now().isoformat()}] Failed to fetch CVEs: Status {response.status}")
                    return []
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Error fetching CVEs from vuln_db: {str(e)}")
        return []

@dataclass
class RiskAssessment:
    """Risk assessment result with explanations"""
    probability: float
    risk_level: str
    top_features: List[Tuple[str, float]]
    feature_vector: np.ndarray
    timestamp: str
    explanation: str
    cve_impacts: List[str] = None  # Added to track CVE-related impacts

class AdaptiveRiskScorer:
    """Adaptive risk scoring with learning capabilities"""
    
    def __init__(self):
        self.weights = WEIGHTS.copy()
        self.bias = BIAS
        self.feature_history = []
        self.learning_rate = 0.01
        self.update_count = 0
        self.decay_factor = 0.95  # Decay factor for older data influence
        self.min_learning_rate = 0.001  # Minimum learning rate to prevent stagnation
        
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def normalize_features(self, event: Dict[str, Any], critical_cves: List[Dict[str, Any]] = None) -> np.ndarray:
        """Normalize and extract features from event"""
        if critical_cves is None:
            critical_cves = []
        normalized = []
        
        for feature in FEATURES:
            value = float(event.get(feature, 0))
            
            # Feature-specific normalization
            if feature == "ai_text_score":
                # Already 0-1 range
                normalized.append(value)
            elif feature == "probe_sim":
                # Already 0-1 range
                normalized.append(value)
            elif feature == "canary_echo":
                # Log scale for count, capped at 10
                normalized.append(min(np.log1p(value) / np.log1p(10), 1.0))
            elif feature == "rag_injection":
                # Already 0-1 range
                normalized.append(value)
            elif feature == "ecc_delta":
                # Normalize ECC errors (0-100 range)
                normalized.append(min(value / 100.0, 1.0))
            elif feature == "latency_ms":
                # Normalize latency (0-5000ms range)
                normalized.append(min(value / 5000.0, 1.0))
            elif feature == "agent_anomaly":
                # Already 0-1 range
                normalized.append(value)
            elif feature == "data_poison_risk":
                # Already 0-1 range
                normalized.append(value)
            elif feature == "threat_intel_score":
                # Enhance with CVE data if available
                cve_enhanced_score = self.enhance_threat_intel_with_cve(event, critical_cves)
                normalized.append(min(cve_enhanced_score / 10.0, 1.0))
            elif feature == "hardware_temp":
                # Temperature delta from baseline (0-50°C range)
                normalized.append(min(value / 50.0, 1.0))
            else:
                normalized.append(value)
        
        return np.array(normalized)
    
    def enhance_threat_intel_with_cve(self, event: Dict[str, Any], critical_cves: List[Dict[str, Any]] = None) -> float:
        """Enhance threat intelligence score with CVE data if available"""
        base_score = float(event.get("threat_intel_score", 0))
        # Use passed critical_cves instead of fetching
        if critical_cves and any(cve.get('cvss_score', 0) >= 8.0 for cve in critical_cves):
            # Increase score based on critical CVEs relevant to AI
            max_cvss = max(cve.get('cvss_score', 0) for cve in critical_cves)
            return max(base_score, max_cvss)
        return base_score
    
    async def fetch_critical_cves(self) -> List[Dict[str, Any]]:
        """Fetch critical CVEs from vuln_db service"""
        return await fetch_critical_cves_from_vuln_db()
    
    def calculate_risk(self, event: Dict[str, Any], critical_cves: List[Dict[str, Any]] = None) -> RiskAssessment:
        """Calculate risk probability with explanations"""
        if critical_cves is None:
            critical_cves = []
        
        # Extract and normalize features
        feature_vector = self.normalize_features(event, critical_cves)
        
        # Calculate risk score
        z = float(np.dot(feature_vector, self.weights) + self.bias)
        probability = self.sigmoid(z)
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = "critical"
        elif probability >= 0.6:
            risk_level = "high"
        elif probability >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Calculate feature contributions (SHAP-style)
        contributions = []
        for i, (feature, value) in enumerate(zip(FEATURES, feature_vector)):
            contribution = self.weights[i] * value
            contributions.append((feature, contribution))
        
        # Get top contributing features
        top_features = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Generate explanation and CVE impacts if relevant
        cve_impacts = []
        if any("threat_intel_score" in f[0] for f in top_features) and critical_cves:
            for cve in critical_cves:
                if cve.get('cvss_score', 0) >= 8.0:
                    cve_impacts.extend(cve.get('ai_impact_areas', []))
        explanation = self._generate_explanation(probability, risk_level, top_features, cve_impacts)
        
        # Record metrics
        record_detection_score(probability, f"risk_{risk_level}")
        
        # Store for learning
        self.feature_history.append({
            "features": feature_vector,
            "probability": probability,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep history bounded
        if len(self.feature_history) > 10000:
            self.feature_history = self.feature_history[-5000:]
        
        return RiskAssessment(
            probability=round(probability, 4),
            risk_level=risk_level,
            top_features=top_features,
            feature_vector=feature_vector,
            timestamp=datetime.now(timezone.utc).isoformat(),
            explanation=explanation,
            cve_impacts=cve_impacts if cve_impacts else []
        )
    
    def _generate_explanation(self, probability: float, risk_level: str, 
                            top_features: List[Tuple[str, float]], cve_impacts: List[str] = None) -> str:
        """Generate human-readable explanation"""
        
        # Feature name mapping for readability
        feature_names = {
            "ai_text_score": "AI-generated content",
            "probe_sim": "Model fingerprint match",
            "canary_echo": "Canary token activation",
            "rag_injection": "RAG injection attempt",
            "ecc_delta": "Hardware ECC errors",
            "latency_ms": "Response latency anomaly",
            "agent_anomaly": "Agent behavior anomaly",
            "data_poison_risk": "Data poisoning indicators",
            "threat_intel_score": "Threat intelligence severity (with CVE data)",
            "hardware_temp": "Hardware temperature anomaly"
        }
        
        # Build explanation
        if risk_level == "critical":
            base = f"CRITICAL RISK (P={probability:.1%}): Multiple strong indicators detected. "
        elif risk_level == "high":
            base = f"HIGH RISK (P={probability:.1%}): Significant threat indicators present. "
        elif risk_level == "medium":
            base = f"MEDIUM RISK (P={probability:.1%}): Some concerning signals detected. "
        else:
            base = f"LOW RISK (P={probability:.1%}): Minimal threat indicators. "
        
        # Add top contributors
        contributors = []
        for feature, contribution in top_features:
            if abs(contribution) > 0.1:
                feature_name = feature_names.get(feature, feature)
                direction = "elevated" if contribution > 0 else "reduced"
                contributors.append(feature_name)
        
        if contributors:
            base += f"Primary drivers: {', '.join(contributors)}."
        
        # Add CVE impact if relevant
        if cve_impacts and "threat_intel_score" in [f[0] for f in top_features]:
            base += f" Critical vulnerabilities impact: {', '.join(set(cve_impacts))}."
        
        return base
    
    def update_from_outcome(self, event_features: Dict[str, Any], 
                          actual_threat: bool, learning_weight: float = 1.0):
        """Update model weights based on actual outcomes with adaptive learning rate"""
        
        # Get original prediction
        feature_vector = self.normalize_features(event_features)
        z = float(np.dot(feature_vector, self.weights) + self.bias)
        predicted_prob = self.sigmoid(z)
        
        # Calculate error
        target = 1.0 if actual_threat else 0.0
        error = target - predicted_prob
        
        # Adjust learning rate based on error magnitude and update count
        adjusted_learning_rate = max(self.min_learning_rate, self.learning_rate * (1.0 - self.decay_factor ** self.update_count) * (abs(error) + 0.1))
        
        # Update weights using gradient descent with adaptive rate
        gradient = error * predicted_prob * (1 - predicted_prob)
        weight_update = adjusted_learning_rate * gradient * feature_vector * learning_weight
        
        self.weights += weight_update
        self.bias += adjusted_learning_rate * gradient * learning_weight
        
        self.update_count += 1
        
        # Log update details for monitoring
        logger.info(
            "Risk model updated",
            error=round(error, 4),
            learning_rate=round(adjusted_learning_rate, 6),
            weight_change=round(np.linalg.norm(weight_update), 6),
            update_count=self.update_count
        )
        
        # Periodically re-evaluate historical data for long-term learning
        if self.update_count % 100 == 0 and self.feature_history:
            self._retrain_on_history()
    
    def _retrain_on_history(self):
        """Periodically retrain on historical data to refine weights"""
        if len(self.feature_history) < 10:
            return
        
        logger.info("Retraining risk model on historical data", history_size=len(self.feature_history))
        
        # Sample a subset of historical data for retraining
        sample_size = min(100, len(self.feature_history))
        sample_indices = random.sample(range(len(self.feature_history)), sample_size)
        
        for idx in sample_indices:
            historical_entry = self.feature_history[idx]
            historical_features = historical_entry['features']
            historical_prob = historical_entry['probability']
            # Use a pseudo-target based on historical probability trend
            pseudo_target = 1.0 if historical_prob > 0.7 else 0.0
            
            # Recalculate error with reduced weight
            z = float(np.dot(historical_features, self.weights) + self.bias)
            current_prob = self.sigmoid(z)
            error = pseudo_target - current_prob
            
            # Apply a smaller update to avoid overfitting
            gradient = error * current_prob * (1 - current_prob)
            weight_update = self.min_learning_rate * gradient * historical_features * 0.5
            
            self.weights += weight_update
            self.bias += self.min_learning_rate * gradient * 0.5
        
        logger.info("Historical retraining completed", update_count=self.update_count)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance"""
        importance = {}
        for feature, weight in zip(FEATURES, self.weights):
            importance[feature] = round(abs(weight), 4)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def export_model(self) -> Dict[str, Any]:
        """Export current model state"""
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "update_count": self.update_count,
            "feature_importance": self.get_feature_importance(),
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def import_model(self, model_data: Dict[str, Any]):
        """Import model state"""
        self.weights = np.array(model_data["weights"])
        self.bias = model_data["bias"]
        self.update_count = model_data.get("update_count", 0)
        
        logger.info(
            "Risk model imported",
            update_count=self.update_count,
            weights_norm=round(np.linalg.norm(self.weights), 4)
        )

# Global risk scorer instance
risk_scorer = AdaptiveRiskScorer()

def calculate_adaptive_risk(event: Dict[str, Any], critical_cves: List[Dict[str, Any]] = None) -> RiskAssessment:
    """Calculate adaptive risk for an event"""
    if critical_cves is None:
        critical_cves = []
    return risk_scorer.calculate_risk(event, critical_cves)

def update_risk_model(event_features: Dict[str, Any], actual_threat: bool):
    """Update risk model with actual outcome"""
    risk_scorer.update_from_outcome(event_features, actual_threat)

def get_risk_model_status() -> Dict[str, Any]:
    """Get current risk model status"""
    return {
        "model_status": risk_scorer.export_model(),
        "feature_history_size": len(risk_scorer.feature_history),
        "last_update": datetime.now(timezone.utc).isoformat()
    }

# Convenience function for backward compatibility
def risk(event: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy risk function for compatibility"""
    assessment = risk_scorer.calculate_risk(event)
    
    return {
        "p": assessment.probability,
        "risk_level": assessment.risk_level,
        "top_features": assessment.top_features,
        "explanation": assessment.explanation,
        "timestamp": assessment.timestamp,
        "cve_impacts": assessment.cve_impacts if assessment.cve_impacts else []
    }

if __name__ == "__main__":
    # Test the risk scorer
    test_event = {
        "ai_text_score": 0.8,
        "probe_sim": 0.6,
        "canary_echo": 3,
        "rag_injection": 0.2,
        "ecc_delta": 5,
        "latency_ms": 1500,
        "agent_anomaly": 0.3,
        "data_poison_risk": 0.1,
        "threat_intel_score": 7.5,
        "hardware_temp": 15
    }
    
    result = risk(test_event)
    print(json.dumps(result, indent=2))