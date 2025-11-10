"""
AEGIS‑C Hardware Sentinel with Intent
======================================

Connect hardware anomalies to model impact and take appropriate actions.
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import logger

class HardwareAnomalyType(Enum):
    """Types of hardware anomalies"""
    ECC_ERRORS = "ecc_errors"
    TEMPERATURE_SPIKE = "temperature_spike"
    MEMORY_CORRUPTION = "memory_corruption"
    CLOCK_DRIFT = "clock_drift"
    POWER_ANOMALY = "power_anomaly"
    UTILIZATION_ANOMALY = "utilization_anomaly"
    CHECKSUM_DRIFT = "checksum_drift"

class ModelImpactType(Enum):
    """Types of model impact"""
    ACCURACY_DEGRADATION = "accuracy_degradation"
    LATENCY_INCREASE = "latency_increase"
    OUTPUT_CORRUPTION = "output_corruption"
    INFERENCE_FAILURES = "inference_failures"
    CANARY_SCORE_DROP = "canary_score_drop"
    BEHAVIORAL_DRIFT = "behavioral_drift"

class ResponseAction(Enum):
    """Response actions for hardware anomalies"""
    MONITOR = "monitor"
    THROTTLE = "throttle"
    DRAIN = "drain"
    RESET = "reset"
    REATTEST = "reattest"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"

@dataclass
class HardwareAnomaly:
    """Hardware anomaly event"""
    anomaly_id: str
    gpu_id: int
    anomaly_type: HardwareAnomalyType
    severity: float  # 0-1 scale
    timestamp: str
    metrics: Dict[str, float]
    threshold_exceeded: bool
    duration_seconds: float

@dataclass
class ModelImpact:
    """Model impact observation"""
    impact_id: str
    gpu_id: int
    impact_type: ModelImpactType
    severity: float
    timestamp: str
    metrics: Dict[str, float]
    baseline_deviation: float

@dataclass
class IntentAnalysis:
    """Analysis of hardware-model relationship"""
    analysis_id: str
    gpu_id: int
    hardware_anomalies: List[str]
    model_impacts: List[str]
    causality_confidence: float
    primary_anomaly: HardwareAnomalyType
    primary_impact: ModelImpactType
    recommended_action: ResponseAction
    reasoning: str
    urgency_score: float

@dataclass
class ActionOutcome:
    """Result of executed action"""
    action_id: str
    gpu_id: int
    action: ResponseAction
    execution_time: str
    success: bool
    hardware_recovery: float
    model_recovery: float
    side_effects: List[str]
    effectiveness_score: float

class HardwareIntentAnalyzer:
    """Hardware intent analyzer connecting anomalies to model impact"""
    
    def __init__(self):
        # Anomaly and impact storage
        self.hardware_anomalies: Dict[str, HardwareAnomaly] = {}
        self.model_impacts: Dict[str, ModelImpact] = {}
        
        # Causal patterns learned over time
        self.causal_patterns: Dict[Tuple[HardwareAnomalyType, ModelImpactType], float] = defaultdict(float)
        self.pattern_frequencies: Dict[Tuple[HardwareAnomalyType, ModelImpactType], int] = defaultdict(int)
        
        # Action effectiveness tracking
        self.action_effectiveness: Dict[Tuple[HardwareAnomalyType, ResponseAction], float] = defaultdict(float)
        self.action_history: List[ActionOutcome] = []
        
        # Configuration
        self.causality_threshold = 0.3
        self.urgency_threshold = 0.7
        self.correlation_window_minutes = 10
        
        # Baseline metrics
        self.hardware_baselines: Dict[int, Dict[str, float]] = {}
        self.model_baselines: Dict[int, Dict[str, float]] = {}
        
    def add_hardware_anomaly(self, anomaly_data: Dict[str, Any]) -> str:
        """Add hardware anomaly observation"""
        
        # Generate anomaly ID
        anomaly_hash = hashlib.md5(
            json.dumps(anomaly_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        anomaly_id = f"hw_anomaly_{anomaly_hash}"
        
        # Create anomaly object
        anomaly = HardwareAnomaly(
            anomaly_id=anomaly_id,
            gpu_id=anomaly_data["gpu_id"],
            anomaly_type=HardwareAnomalyType(anomaly_data["anomaly_type"]),
            severity=anomaly_data["severity"],
            timestamp=anomaly_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metrics=anomaly_data.get("metrics", {}),
            threshold_exceeded=anomaly_data.get("threshold_exceeded", True),
            duration_seconds=anomaly_data.get("duration_seconds", 0.0)
        )
        
        # Store anomaly
        self.hardware_anomalies[anomaly_id] = anomaly
        
        logger.info(
            "Hardware anomaly added",
            anomaly_id=anomaly_id,
            gpu_id=anomaly.gpu_id,
            anomaly_type=anomaly.anomaly_type.value,
            severity=round(anomaly.severity, 3)
        )
        
        return anomaly_id
    
    def add_model_impact(self, impact_data: Dict[str, Any]) -> str:
        """Add model impact observation"""
        
        # Generate impact ID
        impact_hash = hashlib.md5(
            json.dumps(impact_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        impact_id = f"model_impact_{impact_hash}"
        
        # Create impact object
        impact = ModelImpact(
            impact_id=impact_id,
            gpu_id=impact_data["gpu_id"],
            impact_type=ModelImpactType(impact_data["impact_type"]),
            severity=impact_data["severity"],
            timestamp=impact_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metrics=impact_data.get("metrics", {}),
            baseline_deviation=impact_data.get("baseline_deviation", 0.0)
        )
        
        # Store impact
        self.model_impacts[impact_id] = impact
        
        logger.info(
            "Model impact added",
            impact_id=impact_id,
            gpu_id=impact.gpu_id,
            impact_type=impact.impact_type.value,
            severity=round(impact.severity, 3)
        )
        
        return impact_id
    
    def analyze_intent(self, gpu_id: int, time_window_minutes: int = 10) -> Optional[IntentAnalysis]:
        """Analyze intent behind hardware anomalies and model impacts"""
        
        # Get recent anomalies and impacts for this GPU
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=time_window_minutes)
        
        recent_anomalies = [
            anomaly for anomaly in self.hardware_anomalies.values()
            if (anomaly.gpu_id == gpu_id and 
                datetime.fromisoformat(anomaly.timestamp.replace('Z', '+00:00')) > cutoff_time)
        ]
        
        recent_impacts = [
            impact for impact in self.model_impacts.values()
            if (impact.gpu_id == gpu_id and 
                datetime.fromisoformat(impact.timestamp.replace('Z', '+00:00')) > cutoff_time)
        ]
        
        if not recent_anomalies or not recent_impacts:
            return None
        
        # Analyze causal relationships
        causality_analysis = self._analyze_causality(recent_anomalies, recent_impacts)
        
        # Determine primary anomaly and impact
        primary_anomaly = max(recent_anomalies, key=lambda a: a.severity)
        primary_impact = max(recent_impacts, key=lambda i: i.severity)
        
        # Calculate urgency score
        urgency_score = self._calculate_urgency_score(primary_anomaly, primary_impact, causality_analysis)
        
        # Recommend action
        recommended_action = self._recommend_action(
            primary_anomaly.anomaly_type, 
            primary_impact.impact_type, 
            urgency_score
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            primary_anomaly, 
            primary_impact, 
            causality_analysis,
            recommended_action
        )
        
        # Create analysis
        analysis_id = f"intent_analysis_{hashlib.md5(f'{gpu_id}_{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        analysis = IntentAnalysis(
            analysis_id=analysis_id,
            gpu_id=gpu_id,
            hardware_anomalies=[a.anomaly_id for a in recent_anomalies],
            model_impacts=[i.impact_id for i in recent_impacts],
            causality_confidence=causality_analysis["confidence"],
            primary_anomaly=primary_anomaly.anomaly_type,
            primary_impact=primary_impact.impact_type,
            recommended_action=recommended_action,
            reasoning=reasoning,
            urgency_score=urgency_score
        )
        
        logger.info(
            "Intent analysis completed",
            analysis_id=analysis_id,
            gpu_id=gpu_id,
            confidence=round(causality_analysis["confidence"], 3),
            action=recommended_action.value,
            urgency=round(urgency_score, 3)
        )
        
        return analysis
    
    def _analyze_causality(self, anomalies: List[HardwareAnomaly], 
                          impacts: List[ModelImpact]) -> Dict[str, Any]:
        """Analyze causal relationship between hardware anomalies and model impacts"""
        
        causality_scores = []
        
        for anomaly in anomalies:
            for impact in impacts:
                # Calculate temporal correlation
                anomaly_time = datetime.fromisoformat(anomaly.timestamp.replace('Z', '+00:00'))
                impact_time = datetime.fromisoformat(impact.timestamp.replace('Z', '+00:00'))
                
                time_diff = (impact_time - anomaly_time).total_seconds()
                
                # Check if impact follows anomaly within reasonable time
                if 0 < time_diff <= self.correlation_window_minutes * 60:
                    # Calculate correlation strength
                    pattern_key = (anomaly.anomaly_type, impact.impact_type)
                    
                    # Base score from learned patterns
                    base_score = self.causal_patterns.get(pattern_key, 0.5)
                    
                    # Adjust based on severity
                    severity_factor = (anomaly.severity + impact.severity) / 2
                    
                    # Adjust based on temporal proximity
                    temporal_factor = 1.0 - (time_diff / (self.correlation_window_minutes * 60))
                    
                    # Combined causality score
                    causality_score = base_score * 0.4 + severity_factor * 0.3 + temporal_factor * 0.3
                    
                    causality_scores.append(causality_score)
                    
                    # Update learned patterns
                    self._update_causal_pattern(pattern_key, causality_score)
        
        if causality_scores:
            avg_confidence = sum(causality_scores) / len(causality_scores)
            max_confidence = max(causality_scores)
        else:
            avg_confidence = 0.0
            max_confidence = 0.0
        
        return {
            "confidence": avg_confidence,
            "max_confidence": max_confidence,
            "correlations_found": len(causality_scores)
        }
    
    def _update_causal_pattern(self, pattern: Tuple[HardwareAnomalyType, ModelImpactType], 
                             score: float):
        """Update causal pattern knowledge"""
        
        # Weighted average update
        current_score = self.causal_patterns[pattern]
        current_count = self.pattern_frequencies[pattern]
        
        new_count = current_count + 1
        new_score = (current_score * current_count + score) / new_count
        
        self.causal_patterns[pattern] = new_score
        self.pattern_frequencies[pattern] = new_count
    
    def _calculate_urgency_score(self, anomaly: HardwareAnomaly, 
                               impact: ModelImpact, 
                               causality: Dict[str, Any]) -> float:
        """Calculate urgency score for the situation"""
        
        # Base urgency from severities
        hardware_urgency = anomaly.severity * 0.4
        model_urgency = impact.severity * 0.4
        
        # Add causality confidence
        causality_urgency = causality["confidence"] * 0.2
        
        # Check for critical conditions
        critical_bonus = 0.0
        
        if anomaly.anomaly_type == HardwareAnomalyType.ECC_ERRORS and anomaly.severity > 0.8:
            critical_bonus += 0.2
        
        if impact.impact_type == ModelImpactType.OUTPUT_CORRUPTION and impact.severity > 0.7:
            critical_bonus += 0.2
        
        if (anomaly.anomaly_type == HardwareAnomalyType.TEMPERATURE_SPIKE and 
            anomaly.metrics.get("temperature", 0) > 90):
            critical_bonus += 0.15
        
        urgency = hardware_urgency + model_urgency + causality_urgency + critical_bonus
        
        return min(1.0, urgency)
    
    def _recommend_action(self, anomaly_type: HardwareAnomalyType, 
                         impact_type: ModelImpactType, 
                         urgency_score: float) -> ResponseAction:
        """Recommend action based on anomaly, impact, and urgency"""
        
        # High urgency actions
        if urgency_score >= 0.8:
            if anomaly_type == HardwareAnomalyType.ECC_ERRORS:
                return ResponseAction.QUARANTINE
            elif anomaly_type == HardwareAnomalyType.MEMORY_CORRUPTION:
                return ResponseAction.DRAIN
            elif impact_type == ModelImpactType.OUTPUT_CORRUPTION:
                return ResponseAction.QUARANTINE
            else:
                return ResponseAction.RESET
        
        # Medium urgency actions
        elif urgency_score >= 0.6:
            if anomaly_type == HardwareAnomalyType.TEMPERATURE_SPIKE:
                return ResponseAction.THROTTLE
            elif anomaly_type == HardwareAnomalyType.CHECKSUM_DRIFT:
                return ResponseAction.REATTEST
            elif impact_type == ModelImpactType.ACCURACY_DEGRADATION:
                return ResponseAction.DRAIN
            else:
                return ResponseAction.THROTTLE
        
        # Low urgency actions
        elif urgency_score >= 0.4:
            if anomaly_type == HardwareAnomalyType.UTILIZATION_ANOMALY:
                return ResponseAction.MONITOR
            elif impact_type == ModelImpactType.LATENCY_INCREASE:
                return ResponseAction.THROTTLE
            else:
                return ResponseAction.MONITOR
        
        # Minimal urgency
        else:
            return ResponseAction.MONITOR
    
    def _generate_reasoning(self, anomaly: HardwareAnomaly, 
                          impact: ModelImpact, 
                          causality: Dict[str, Any],
                          action: ResponseAction) -> str:
        """Generate human-readable reasoning"""
        
        reasoning_parts = []
        
        # Anomaly description
        anomaly_desc = {
            HardwareAnomalyType.ECC_ERRORS: "ECC memory errors detected",
            HardwareAnomalyType.TEMPERATURE_SPIKE: "GPU temperature spike observed",
            HardwareAnomalyType.MEMORY_CORRUPTION: "Memory corruption indicators found",
            HardwareAnomalyType.CHECKSUM_DRIFT: "Data checksum drift detected",
            HardwareAnomalyType.POWER_ANOMALY: "Power consumption anomaly",
            HardwareAnomalyType.UTILIZATION_ANOMALY: "Unusual utilization pattern",
            HardwareAnomalyType.CLOCK_DRIFT: "Clock timing drift detected"
        }
        
        reasoning_parts.append(f"Hardware issue: {anomaly_desc.get(anomaly.anomaly_type, 'Unknown anomaly')}")
        
        # Impact description
        impact_desc = {
            ModelImpactType.ACCURACY_DEGRADATION: "Model accuracy degradation",
            ModelImpactType.LATENCY_INCREASE: "Increased inference latency",
            ModelImpactType.OUTPUT_CORRUPTION: "Output corruption detected",
            ModelImpactType.INFERENCE_FAILURES: "Inference failures occurring",
            ModelImpactType.CANARY_SCORE_DROP: "Canary detection score drop",
            ModelImpactType.BEHAVIORAL_DRIFT: "Model behavioral drift"
        }
        
        reasoning_parts.append(f"Model impact: {impact_desc.get(impact.impact_type, 'Unknown impact')}")
        
        # Causality
        if causality["confidence"] > 0.6:
            reasoning_parts.append(f"Strong causal relationship detected (confidence: {causality['confidence']:.2f})")
        elif causality["confidence"] > 0.3:
            reasoning_parts.append(f"Moderate causal relationship (confidence: {causality['confidence']:.2f})")
        else:
            reasoning_parts.append(f"Weak causal relationship (confidence: {causality['confidence']:.2f})")
        
        # Action justification
        action_desc = {
            ResponseAction.MONITOR: "Continuing monitoring to assess progression",
            ResponseAction.THROTTLE: "Throttling to reduce stress and prevent escalation",
            ResponseAction.DRAIN: "Draining traffic to isolate affected GPU",
            ResponseAction.RESET: "Resetting GPU to clear transient issues",
            ResponseAction.REATTEST: "Re-testing to verify hardware and model integrity",
            ResponseAction.QUARANTINE: "Quarantining to prevent widespread impact",
            ResponseAction.ESCALATE: "Escalating for immediate human intervention"
        }
        
        reasoning_parts.append(f"Action: {action_desc.get(action, 'Precautionary measure')}")
        
        return ". ".join(reasoning_parts) + "."
    
    def execute_action(self, analysis: IntentAnalysis) -> ActionOutcome:
        """Execute recommended action and track outcome"""
        
        action_id = f"action_{hashlib.md5(f'{analysis.analysis_id}_{analysis.recommended_action.value}'.encode()).hexdigest()[:8]}"
        
        execution_time = datetime.now(timezone.utc).isoformat()
        
        # Mock action execution (in real implementation, this would interface with hardware management)
        success = self._execute_hardware_action(analysis.gpu_id, analysis.recommended_action)
        
        # Calculate recovery metrics (mock)
        hardware_recovery = 0.8 if success else 0.0
        model_recovery = 0.7 if success else 0.0
        
        # Identify side effects
        side_effects = self._identify_side_effects(analysis.recommended_action)
        
        # Calculate effectiveness
        effectiveness = (hardware_recovery + model_recovery) / 2
        
        # Create outcome record
        outcome = ActionOutcome(
            action_id=action_id,
            gpu_id=analysis.gpu_id,
            action=analysis.recommended_action,
            execution_time=execution_time,
            success=success,
            hardware_recovery=hardware_recovery,
            model_recovery=model_recovery,
            side_effects=side_effects,
            effectiveness_score=effectiveness
        )
        
        # Store outcome
        self.action_history.append(outcome)
        
        # Update action effectiveness
        pattern_key = (analysis.primary_anomaly, analysis.recommended_action)
        current_effectiveness = self.action_effectiveness[pattern_key]
        current_count = len([h for h in self.action_history if h.action == analysis.recommended_action])
        
        new_effectiveness = (current_effectiveness * current_count + effectiveness) / (current_count + 1)
        self.action_effectiveness[pattern_key] = new_effectiveness
        
        logger.info(
            "Action executed",
            action_id=action_id,
            gpu_id=analysis.gpu_id,
            action=analysis.recommended_action.value,
            success=success,
            effectiveness=round(effectiveness, 3)
        )
        
        return outcome
    
    def _execute_hardware_action(self, gpu_id: int, action: ResponseAction) -> bool:
        """Execute hardware action (mock implementation)"""
        
        # In real implementation, this would interface with GPU management APIs
        action_success_rates = {
            ResponseAction.MONITOR: 0.95,
            ResponseAction.THROTTLE: 0.85,
            ResponseAction.DRAIN: 0.80,
            ResponseAction.RESET: 0.75,
            ResponseAction.REATTEST: 0.90,
            ResponseAction.QUARANTINE: 0.70,
            ResponseAction.ESCALATE: 0.60
        }
        
        base_success_rate = action_success_rates.get(action, 0.7)
        
        # Add some randomness
        success = base_success_rate + (hash(str(gpu_id)) % 20 - 10) / 100.0
        
        return max(0.0, min(1.0, success)) > 0.5
    
    def _identify_side_effects(self, action: ResponseAction) -> List[str]:
        """Identify potential side effects of action"""
        
        side_effects_map = {
            ResponseAction.MONITOR: ["Increased monitoring overhead"],
            ResponseAction.THROTTLE: ["Reduced performance", "Increased latency"],
            ResponseAction.DRAIN: ["Service disruption", "Load redistribution"],
            ResponseAction.RESET: ["Temporary service interruption", "State loss"],
            ResponseAction.REATTEST: ["Resource consumption", "Brief service pause"],
            ResponseAction.QUARANTINE: ["Service unavailability", "Manual intervention required"],
            ResponseAction.ESCALATE: ["Human response time", "Operational overhead"]
        }
        
        return side_effects_map.get(action, ["Unknown side effects"])
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get intent analysis statistics"""
        
        # Recent activity
        recent_anomalies = [
            a for a in self.hardware_anomalies.values()
            if (datetime.now(timezone.utc) - datetime.fromisoformat(a.timestamp.replace('Z', '+00:00'))).hours <= 24
        ]
        
        recent_impacts = [
            i for i in self.model_impacts.values()
            if (datetime.now(timezone.utc) - datetime.fromisoformat(i.timestamp.replace('Z', '+00:00'))).hours <= 24
        ]
        
        # Causal pattern strength
        strong_patterns = {
            f"{hw.value}->{ml.value}": round(confidence, 3)
            for (hw, ml), confidence in self.causal_patterns.items()
            if confidence > 0.6
        }
        
        # Action effectiveness
        action_effectiveness = {}
        for (hw, action), effectiveness in self.action_effectiveness.items():
            key = f"{hw.value}->{action.value}"
            action_effectiveness[key] = round(effectiveness, 3)
        
        # Recent action outcomes
        recent_outcomes = [
            outcome for outcome in self.action_history
            if (datetime.now(timezone.utc) - datetime.fromisoformat(outcome.execution_time.replace('Z', '+00:00'))).hours <= 24
        ]
        
        if recent_outcomes:
            avg_effectiveness = sum(o.effectiveness_score for o in recent_outcomes) / len(recent_outcomes)
            success_rate = sum(1 for o in recent_outcomes if o.success) / len(recent_outcomes)
        else:
            avg_effectiveness = 0.0
            success_rate = 0.0
        
        return {
            "recent_anomalies": len(recent_anomalies),
            "recent_impacts": len(recent_impacts),
            "strong_causal_patterns": strong_patterns,
            "action_effectiveness": action_effectiveness,
            "recent_actions": len(recent_outcomes),
            "avg_action_effectiveness": round(avg_effectiveness, 3),
            "action_success_rate": round(success_rate, 3),
            "total_patterns_learned": len(self.causal_patterns),
            "learning_maturity": min(1.0, len(self.action_history) / 100),
            "last_update": datetime.now(timezone.utc).isoformat()
        }

# Global hardware intent analyzer instance
hardware_intent = HardwareIntentAnalyzer()

def add_hardware_anomaly(anomaly_data: Dict[str, Any]) -> str:
    """Add hardware anomaly for intent analysis"""
    return hardware_intent.add_hardware_anomaly(anomaly_data)

def add_model_impact(impact_data: Dict[str, Any]) -> str:
    """Add model impact for intent analysis"""
    return hardware_intent.add_model_impact(impact_data)

def analyze_hardware_intent(gpu_id: int, time_window_minutes: int = 10) -> Dict[str, Any]:
    """Analyze hardware intent and recommend action"""
    analysis = hardware_intent.analyze_intent(gpu_id, time_window_minutes)
    
    if analysis:
        # Execute action if urgency is high
        if analysis.urgency_score >= 0.6:
            outcome = hardware_intent.execute_action(analysis)
            return {
                "analysis": asdict(analysis),
                "action_outcome": asdict(outcome)
            }
        else:
            return {"analysis": asdict(analysis)}
    else:
        return {"message": "Insufficient data for intent analysis"}

def get_hardware_intent_status() -> Dict[str, Any]:
    """Get hardware intent analysis status"""
    return hardware_intent.get_intent_statistics()

if __name__ == "__main__":
    # Test hardware intent analysis
    test_anomaly = {
        "gpu_id": 0,
        "anomaly_type": "ecc_errors",
        "severity": 0.8,
        "metrics": {
            "uncorrected_errors": 5,
            "corrected_errors": 12,
            "error_rate": 0.001
        }
    }
    
    test_impact = {
        "gpu_id": 0,
        "impact_type": "accuracy_degradation",
        "severity": 0.6,
        "metrics": {
            "accuracy_drop": 0.15,
            "error_rate_increase": 0.05
        },
        "baseline_deviation": 0.2
    }
    
    anomaly_id = add_hardware_anomaly(test_anomaly)
    impact_id = add_model_impact(test_impact)
    
    print(f"Added anomaly: {anomaly_id}")
    print(f"Added impact: {impact_id}")
    
    # Analyze intent
    result = analyze_hardware_intent(0)
    print(json.dumps(result, indent=2, default=str))
    
    # Get statistics
    stats = get_hardware_intent_status()
    print("\nHardware Intent Statistics:")
    print(json.dumps(stats, indent=2, default=str))