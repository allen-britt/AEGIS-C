#!/usr/bin/env python3
"""
AEGIS‑C Brain Client
=====================

Simple client for connecting services to the Brain Gateway.
"""

import os
import requests
import json
from typing import Dict, List, Any, Optional
import logging

# Configuration
BRAIN_URL = os.getenv("BRAIN_URL", "http://localhost:8030")
REQUEST_TIMEOUT = int(os.getenv("BRAIN_TIMEOUT", "3"))

logger = logging.getLogger(__name__)

class BrainClientError(Exception):
    """Brain client specific errors"""
    pass

def assess(kind: str, subject: str, signals: Dict[str, float], 
          context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Assess risk using the Brain Gateway.
    
    Args:
        kind: Subject kind ('agent', 'rag', 'hardware', 'artifact')
        subject: Subject identifier (e.g., 'session:abc123', 'node:gpu0')
        signals: Dictionary of signal name -> value pairs
        context: Optional additional context
    
    Returns:
        Risk assessment response from brain service
        
    Raises:
        BrainClientError: If request fails
    """
    try:
        payload = {
            "subject": subject,
            "kind": kind,
            "signals": [{"name": k, "value": float(v)} for k, v in signals.items()],
            "context": context or {}
        }
        
        response = requests.post(
            f"{BRAIN_URL}/risk",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"Risk assessed for {subject}: {result.get('probability', 0):.3f} ({result.get('level', 'unknown')})")
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Brain risk assessment failed: {e}")
        raise BrainClientError(f"Risk assessment failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in risk assessment: {e}")
        raise BrainClientError(f"Unexpected error: {str(e)}")

def decide(kind: str, subject: str, risk_prob: float, 
          options: List[str]) -> Dict[str, Any]:
    """
    Get policy decision from Brain Gateway.
    
    Args:
        kind: Subject kind
        subject: Subject identifier
        risk_prob: Risk probability from /risk endpoint
        options: List of available policy options
    
    Returns:
        Policy decision response from brain service
        
    Raises:
        BrainClientError: If request fails
    """
    try:
        payload = {
            "subject": subject,
            "kind": kind,
            "risk": float(risk_prob),
            "options": options
        }
        
        response = requests.post(
            f"{BRAIN_URL}/policy",
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"Policy decision for {subject}: {result.get('action', 'unknown')} (confidence: {result.get('confidence', 0):.3f})")
        
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Brain policy decision failed: {e}")
        raise BrainClientError(f"Policy decision failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in policy decision: {e}")
        raise BrainClientError(f"Unexpected error: {str(e)}")

def is_healthy() -> bool:
    """
    Check if Brain Gateway is healthy.
    
    Returns:
        True if brain service is healthy, False otherwise
    """
    try:
        response = requests.get(f"{BRAIN_URL}/health", timeout=1)
        return response.status_code == 200
    except Exception:
        return False

# Convenience functions for common use cases
def assess_artifact_risk(artifact_id: str, ai_score: float, 
                        additional_signals: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Assess risk for an artifact (e.g., text detection)"""
    signals = {"ai_text_score": ai_score}
    if additional_signals:
        signals.update(additional_signals)
    
    return assess("artifact", f"artifact:{artifact_id}", signals)

def assess_hardware_risk(node_id: str, ecc_delta: float = 0.0, 
                        temp_delta: float = 0.0, latency_ms: float = 0.0) -> Dict[str, Any]:
    """Assess risk for hardware anomalies"""
    signals = {
        "ecc_delta": ecc_delta,
        "hardware_temp": temp_delta,
        "latency_ms": latency_ms / 5000.0  # Normalize to 0-1 range
    }
    
    context = {"node": node_id}
    return assess("hardware", f"node:{node_id}", signals, context)

def assess_rag_risk(session_id: str, injection_score: float = 0.0,
                   content_anomaly: float = 0.0) -> Dict[str, Any]:
    """Assess risk for RAG operations"""
    signals = {
        "rag_injection": injection_score,
        "agent_anomaly": content_anomaly
    }
    
    return assess("rag", f"session:{session_id}", signals)

def decide_artifact_action(artifact_id: str, risk_prob: float) -> Dict[str, Any]:
    """Decide action for artifact risk"""
    options = ["observe", "raise_friction", "block_family"]
    return decide("artifact", f"artifact:{artifact_id}", risk_prob, options)

def decide_hardware_action(node_id: str, risk_prob: float) -> Dict[str, Any]:
    """Decide action for hardware risk"""
    options = ["observe", "throttle", "drain_node", "reset_gpu", "reattest"]
    return decide("hardware", f"node:{node_id}", risk_prob, options)

def decide_rag_action(session_id: str, risk_prob: float) -> Dict[str, Any]:
    """Decide action for RAG risk"""
    options = ["observe", "increase_monitoring", "raise_friction", "quarantine_host"]
    return decide("rag", f"session:{session_id}", risk_prob, options)

# Test function
def test_brain_connection():
    """Test connection to Brain Gateway"""
    try:
        # Test health
        if not is_healthy():
            return False, "Brain service not healthy"
        
        # Test risk assessment
        risk_result = assess("test", "test:123", {"ai_text_score": 0.5})
        
        # Test policy decision
        policy_result = decide("test", "test:123", risk_result["probability"], ["observe", "raise_friction"])
        
        return True, "Brain connection successful"
        
    except Exception as e:
        return False, f"Brain connection failed: {str(e)}"

if __name__ == "__main__":
    # Test the client
    success, message = test_brain_connection()
    print(f"Brain client test: {message}")
    
    if success:
        # Example usage
        print("\nExample risk assessment:")
        risk = assess_artifact_risk("test-123", 0.8)
        print(json.dumps(risk, indent=2))
        
        print("\nExample policy decision:")
        policy = decide_artifact_action("test-123", risk["probability"])
        print(json.dumps(policy, indent=2))