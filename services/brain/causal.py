"""
AEGIS‑C Causal Incident Explainer
==================================

Move beyond anomaly detection to root cause analysis.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, Counter

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import logger

class EventType(Enum):
    """Event types in the causal chain"""
    SOURCE_REQUEST = "source_request"
    CONTENT_RETRIEVAL = "content_retrieval"
    MODEL_INFERENCE = "model_inference"
    TOOL_EXECUTION = "tool_execution"
    OPERATOR_ACTION = "operator_action"
    SYSTEM_RESPONSE = "system_response"
    ANOMALY_DETECTED = "anomaly_detected"
    POLICY_TRIGGERED = "policy_triggered"

class CausalRelation(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"           # A directly causes B
    CONTRIBUTING_FACTOR = "contributing_factor"  # A contributes to B
    ENABLING_CONDITION = "enabling_condition"    # A enables B to occur
    CORRELATED = "correlated"               # A and B occur together
    SEQUENTIAL = "sequential"               # A precedes B in time

@dataclass
class EventNode:
    """Node in the causal event graph"""
    event_id: str
    event_type: EventType
    timestamp: str
    source: str
    details: Dict[str, Any]
    risk_score: float
    attributes: Dict[str, Any]

@dataclass
class CausalEdge:
    """Edge in the causal event graph"""
    from_event: str
    to_event: str
    relation_type: CausalRelation
    confidence: float
    temporal_gap: float  # seconds between events
    evidence: List[str]

@dataclass
class CausalMotif:
    """Common cause-effect pattern"""
    motif_id: str
    pattern: List[EventType]
    description: str
    frequency: int
    avg_time_to_resolution: float
    success_rate: float

@dataclass
class RootCauseAnalysis:
    """Root cause analysis result"""
    incident_id: str
    primary_causes: List[str]
    contributing_factors: List[str]
    causal_chain: List[str]
    detected_motifs: List[CausalMotif]
    confidence: float
    explanation: str
    recommended_actions: List[str]
    prevention_strategies: List[str]

class CausalIncidentExplainer:
    """Causal incident analysis and explanation system"""
    
    def __init__(self):
        # Event graph storage
        self.event_nodes: Dict[str, EventNode] = {}
        self.causal_edges: List[CausalEdge] = []
        
        # Learned motifs
        self.causal_motifs: Dict[str, CausalMotif] = {}
        self.motif_patterns = self._initialize_motif_patterns()
        
        # Analysis history
        self.incident_history: List[RootCauseAnalysis] = []
        self.pattern_frequencies: Dict[str, int] = defaultdict(int)
        
        # Learning parameters
        self.min_confidence_threshold = 0.3
        self.temporal_window = 300  # 5 minutes
        self.learning_enabled = True
        
    def _initialize_motif_patterns(self) -> Dict[str, List[EventType]]:
        """Initialize known causal motifs"""
        return {
            "rag_drift": [
                EventType.SOURCE_REQUEST,
                EventType.CONTENT_RETRIEVAL,
                EventType.MODEL_INFERENCE,
                EventType.ANOMALY_DETECTED
            ],
            "policy_evasion": [
                EventType.SOURCE_REQUEST,
                EventType.MODEL_INFERENCE,
                EventType.TOOL_EXECUTION,
                EventType.POLICY_TRIGGERED
            ],
            "supply_chain_compromise": [
                EventType.CONTENT_RETRIEVAL,
                EventType.MODEL_INFERENCE,
                EventType.SYSTEM_RESPONSE,
                EventType.ANOMALY_DETECTED
            ],
            "operator_error": [
                EventType.OPERATOR_ACTION,
                EventType.SYSTEM_RESPONSE,
                EventType.ANOMALY_DETECTED
            ],
            "resource_exhaustion": [
                EventType.SOURCE_REQUEST,
                EventType.MODEL_INFERENCE,
                EventType.TOOL_EXECUTION,
                EventType.ANOMALY_DETECTED
            ]
        }
    
    def add_event(self, event_data: Dict[str, Any]) -> str:
        """Add an event to the causal graph"""
        
        # Generate event ID
        event_hash = hashlib.md5(
            json.dumps(event_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        event_id = f"{event_data.get('event_type', 'unknown')}_{event_hash}"
        
        # Create event node
        event_node = EventNode(
            event_id=event_id,
            event_type=EventType(event_data.get("event_type", "source_request")),
            timestamp=event_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            source=event_data.get("source", "unknown"),
            details=event_data.get("details", {}),
            risk_score=event_data.get("risk_score", 0.0),
            attributes=event_data.get("attributes", {})
        )
        
        # Store event
        self.event_nodes[event_id] = event_node
        
        # Analyze causal relationships with existing events
        self._analyze_causal_relationships(event_node)
        
        logger.info(
            "Event added to causal graph",
            event_id=event_id,
            event_type=event_node.event_type.value,
            total_events=len(self.event_nodes)
        )
        
        return event_id
    
    def _analyze_causal_relationships(self, new_event: EventNode):
        """Analyze causal relationships between new event and existing events"""
        
        new_time = datetime.fromisoformat(new_event.timestamp.replace('Z', '+00:00'))
        
        for existing_id, existing_event in self.event_nodes.items():
            if existing_id == new_event.event_id:
                continue
            
            existing_time = datetime.fromisoformat(existing_event.timestamp.replace('Z', '+00:00'))
            time_diff = abs((new_time - existing_time).total_seconds())
            
            # Only consider events within temporal window
            if time_diff > self.temporal_window:
                continue
            
            # Determine causal relationship
            relation, confidence = self._determine_causal_relation(existing_event, new_event, time_diff)
            
            if confidence >= self.min_confidence_threshold:
                edge = CausalEdge(
                    from_event=existing_event.event_id,
                    to_event=new_event.event_id,
                    relation_type=relation,
                    confidence=confidence,
                    temporal_gap=time_diff,
                    evidence=self._generate_evidence(existing_event, new_event, relation)
                )
                
                self.causal_edges.append(edge)
    
    def _determine_causal_relation(self, event1: EventNode, event2: EventNode, 
                                 time_diff: float) -> Tuple[CausalRelation, float]:
        """Determine causal relationship between two events"""
        
        # Temporal ordering
        if event1.timestamp < event2.timestamp:
            earlier, later = event1, event2
        else:
            earlier, later = event2, event1
        
        # Analyze relationship based on event types and attributes
        confidence = 0.0
        relation = CausalRelation.CORRELATED
        
        # Direct cause patterns
        if (earlier.event_type == EventType.SOURCE_REQUEST and 
            later.event_type == EventType.CONTENT_RETRIEVAL):
            relation = CausalRelation.DIRECT_CAUSE
            confidence = 0.8
        
        elif (earlier.event_type == EventType.CONTENT_RETRIEVAL and 
              later.event_type == EventType.MODEL_INFERENCE):
            relation = CausalRelation.DIRECT_CAUSE
            confidence = 0.9
        
        elif (earlier.event_type == EventType.MODEL_INFERENCE and 
              later.event_type == EventType.TOOL_EXECUTION):
            relation = CausalRelation.DIRECT_CAUSE
            confidence = 0.7
        
        elif (earlier.event_type == EventType.TOOL_EXECUTION and 
              later.event_type == EventType.SYSTEM_RESPONSE):
            relation = CausalRelation.DIRECT_CAUSE
            confidence = 0.8
        
        # Contributing factor patterns
        elif (earlier.event_type == EventType.OPERATOR_ACTION and 
              later.event_type in [EventType.ANOMALY_DETECTED, EventType.POLICY_TRIGGERED]):
            relation = CausalRelation.CONTRIBUTING_FACTOR
            confidence = 0.6
        
        elif (earlier.risk_score > 0.7 and 
              later.event_type == EventType.ANOMALY_DETECTED):
            relation = CausalRelation.CONTRIBUTING_FACTOR
            confidence = 0.5
        
        # Enabling condition patterns
        elif (earlier.event_type == EventType.SOURCE_REQUEST and 
              later.event_type == EventType.ANOMALY_DETECTED):
            relation = CausalRelation.ENABLING_CONDITION
            confidence = 0.4
        
        # Adjust confidence based on temporal proximity
        if time_diff < 10:
            confidence = min(1.0, confidence + 0.2)
        elif time_diff > 60:
            confidence = max(0.1, confidence - 0.2)
        
        # Adjust confidence based on attribute similarity
        if self._attributes_correlate(earlier.attributes, later.attributes):
            confidence = min(1.0, confidence + 0.1)
        
        return relation, confidence
    
    def _attributes_correlate(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> bool:
        """Check if event attributes correlate"""
        
        # Check for common attributes
        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        
        if not common_keys:
            return False
        
        # Check value similarity for common keys
        similar_count = 0
        for key in common_keys:
            val1, val2 = attrs1[key], attrs2[key]
            if isinstance(val1, str) and isinstance(val2, str):
                if val1.lower() == val2.lower():
                    similar_count += 1
            elif val1 == val2:
                similar_count += 1
        
        return similar_count / len(common_keys) > 0.5
    
    def _generate_evidence(self, event1: EventNode, event2: EventNode, 
                          relation: CausalRelation) -> List[str]:
        """Generate evidence for causal relationship"""
        
        evidence = []
        
        # Temporal evidence
        time_diff = abs(datetime.fromisoformat(event1.timestamp.replace('Z', '+00:00')) - 
                       datetime.fromisoformat(event2.timestamp.replace('Z', '+00:00'))).total_seconds()
        evidence.append(f"Temporal gap: {time_diff:.1f}s")
        
        # Type-based evidence
        evidence.append(f"Pattern: {event1.event_type.value} → {event2.event_type.value}")
        
        # Risk score evidence
        if event1.risk_score > 0.5:
            evidence.append(f"High risk precursor: {event1.risk_score:.2f}")
        
        # Attribute-based evidence
        common_attrs = set(event1.attributes.keys()) & set(event2.attributes.keys())
        if common_attrs:
            evidence.append(f"Shared attributes: {', '.join(common_attrs)}")
        
        return evidence
    
    def analyze_incident(self, incident_events: List[str], 
                        incident_id: str = None) -> RootCauseAnalysis:
        """Analyze incident and identify root causes"""
        
        if not incident_id:
            incident_id = f"incident_{hashlib.md5(''.join(incident_events).encode()).hexdigest()[:8]}"
        
        # Get relevant events
        relevant_events = {eid: self.event_nodes[eid] for eid in incident_events if eid in self.event_nodes}
        
        if not relevant_events:
            return RootCauseAnalysis(
                incident_id=incident_id,
                primary_causes=[],
                contributing_factors=[],
                causal_chain=[],
                detected_motifs=[],
                confidence=0.0,
                explanation="No relevant events found for analysis",
                recommended_actions=[],
                prevention_strategies=[]
            )
        
        # Build causal subgraph
        subgraph_edges = [edge for edge in self.causal_edges 
                         if edge.from_event in incident_events and edge.to_event in incident_events]
        
        # Identify primary causes
        primary_causes = self._identify_primary_causes(relevant_events, subgraph_edges)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(relevant_events, subgraph_edges)
        
        # Build causal chain
        causal_chain = self._build_causal_chain(relevant_events, subgraph_edges)
        
        # Detect motifs
        detected_motifs = self._detect_causal_motifs(relevant_events)
        
        # Generate explanation
        explanation, confidence = self._generate_explanation(
            primary_causes, contributing_factors, detected_motifs, relevant_events
        )
        
        # Generate recommendations
        recommended_actions = self._generate_recommendations(primary_causes, detected_motifs)
        prevention_strategies = self._generate_prevention_strategies(detected_motifs)
        
        analysis = RootCauseAnalysis(
            incident_id=incident_id,
            primary_causes=primary_causes,
            contributing_factors=contributing_factors,
            causal_chain=causal_chain,
            detected_motifs=detected_motifs,
            confidence=confidence,
            explanation=explanation,
            recommended_actions=recommended_actions,
            prevention_strategies=prevention_strategies
        )
        
        # Store analysis
        self.incident_history.append(analysis)
        
        # Update motif frequencies
        for motif in detected_motifs:
            self.pattern_frequencies[motif.motif_id] += 1
        
        logger.info(
            "Incident analysis completed",
            incident_id=incident_id,
            primary_causes=len(primary_causes),
            confidence=round(confidence, 3),
            motifs_detected=len(detected_motifs)
        )
        
        return analysis
    
    def _identify_primary_causes(self, events: Dict[str, EventNode], 
                               edges: List[CausalEdge]) -> List[str]:
        """Identify primary causes from event graph"""
        
        # Find events with no incoming edges (root events)
        all_events = set(events.keys())
        events_with_incoming = {edge.to_event for edge in edges}
        root_events = all_events - events_with_incoming
        
        # Sort by risk score and confidence
        primary_causes = []
        for event_id in root_events:
            event = events[event_id]
            if event.risk_score > 0.3:  # Threshold for primary cause
                primary_causes.append(event_id)
        
        # Sort by risk score (highest first)
        primary_causes.sort(key=lambda eid: events[eid].risk_score, reverse=True)
        
        return primary_causes[:5]  # Top 5 primary causes
    
    def _identify_contributing_factors(self, events: Dict[str, EventNode], 
                                     edges: List[CausalEdge]) -> List[str]:
        """Identify contributing factors"""
        
        contributing = []
        
        # Events with moderate risk that lead to high-risk events
        for edge in edges:
            if edge.confidence > 0.4:
                from_event = events.get(edge.from_event)
                to_event = events.get(edge.to_event)
                
                if (from_event and to_event and 
                    from_event.risk_score > 0.2 and 
                    to_event.risk_score > 0.6 and
                    edge.from_event not in contributing):
                    contributing.append(edge.from_event)
        
        return contributing[:3]  # Top 3 contributing factors
    
    def _build_causal_chain(self, events: Dict[str, EventNode], 
                          edges: List[CausalEdge]) -> List[str]:
        """Build chronological causal chain"""
        
        # Sort events by timestamp
        sorted_events = sorted(events.items(), key=lambda x: x[1].timestamp)
        
        # Build chain following high-confidence edges
        chain = []
        current_event = sorted_events[0][0] if sorted_events else None
        
        while current_event and len(chain) < len(events):
            chain.append(current_event)
            
            # Find next event in chain
            next_event = None
            for edge in edges:
                if edge.from_event == current_event and edge.confidence > 0.5:
                    if edge.to_event not in chain:
                        next_event = edge.to_event
                        break
            
            if not next_event:
                # Find next chronological event
                for event_id, _ in sorted_events:
                    if event_id not in chain:
                        next_event = event_id
                        break
            
            current_event = next_event
        
        return chain
    
    def _detect_causal_motifs(self, events: Dict[str, EventNode]) -> List[CausalMotif]:
        """Detect known causal motifs in events"""
        
        detected_motifs = []
        event_sequence = [event.event_type for event in events.values()]
        
        for motif_id, pattern in self.motif_patterns.items():
            if self._sequence_matches_pattern(event_sequence, pattern):
                # Create motif with current statistics
                motif = CausalMotif(
                    motif_id=motif_id,
                    pattern=pattern,
                    description=self._get_motif_description(motif_id),
                    frequency=self.pattern_frequencies.get(motif_id, 0) + 1,
                    avg_time_to_resolution=self._calculate_avg_resolution_time(motif_id),
                    success_rate=self._calculate_motif_success_rate(motif_id)
                )
                detected_motifs.append(motif)
        
        return detected_motifs
    
    def _sequence_matches_pattern(self, sequence: List[EventType], 
                                pattern: List[EventType]) -> bool:
        """Check if event sequence matches pattern"""
        
        if len(sequence) < len(pattern):
            return False
        
        # Look for pattern as subsequence
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        
        return False
    
    def _get_motif_description(self, motif_id: str) -> str:
        """Get description for motif"""
        descriptions = {
            "rag_drift": "RAG injection leading to anomalous model output",
            "policy_evasion": "Attempt to bypass security policies through tool usage",
            "supply_chain_compromise": "Compromised content affecting model responses",
            "operator_error": "Human operator action leading to system anomalies",
            "resource_exhaustion": "High resource usage causing system instability"
        }
        return descriptions.get(motif_id, "Unknown causal pattern")
    
    def _calculate_avg_resolution_time(self, motif_id: str) -> float:
        """Calculate average resolution time for motif"""
        # This would be calculated from historical data
        # For now, return placeholder values
        return {
            "rag_drift": 180.0,
            "policy_evasion": 120.0,
            "supply_chain_compromise": 300.0,
            "operator_error": 60.0,
            "resource_exhaustion": 90.0
        }.get(motif_id, 150.0)
    
    def _calculate_motif_success_rate(self, motif_id: str) -> float:
        """Calculate success rate for motif detection/response"""
        # This would be calculated from historical outcomes
        return 0.75  # Placeholder
    
    def _generate_explanation(self, primary_causes: List[str], 
                             contributing_factors: List[str],
                             motifs: List[CausalMotif],
                             events: Dict[str, EventNode]) -> Tuple[str, float]:
        """Generate human-readable explanation"""
        
        if not primary_causes and not motifs:
            return "No clear root cause identified", 0.2
        
        explanation_parts = []
        confidence = 0.0
        
        # Primary causes explanation
        if primary_causes:
            cause_descriptions = []
            for cause_id in primary_causes[:3]:  # Top 3
                event = events.get(cause_id)
                if event:
                    cause_descriptions.append(f"{event.event_type.value} (risk: {event.risk_score:.2f})")
            
            if cause_descriptions:
                explanation_parts.append(f"Primary causes: {', '.join(cause_descriptions)}")
                confidence += 0.4
        
        # Motif explanation
        if motifs:
            motif_names = [motif.description for motif in motifs[:2]]
            explanation_parts.append(f"Pattern detected: {', '.join(motif_names)}")
            confidence += 0.3
        
        # Contributing factors
        if contributing_factors:
            explanation_parts.append(f"Contributing factors: {len(contributing_factors)} identified")
            confidence += 0.2
        
        # Temporal context
        if events:
            timestamps = [datetime.fromisoformat(e.timestamp.replace('Z', '+00:00')) for e in events.values()]
            duration = (max(timestamps) - min(timestamps)).total_seconds()
            explanation_parts.append(f"Incident duration: {duration:.1f} seconds")
        
        explanation = ". ".join(explanation_parts) + "."
        
        return explanation, min(1.0, confidence)
    
    def _generate_recommendations(self, primary_causes: List[str], 
                                motifs: List[CausalMotif]) -> List[str]:
        """Generate recommended actions"""
        
        recommendations = []
        
        # Based on primary causes
        if primary_causes:
            recommendations.append("Investigate and address primary cause events")
            recommendations.append("Enhance monitoring for high-risk precursor events")
        
        # Based on motifs
        motif_recommendations = {
            "rag_drift": "Implement stricter RAG content filtering",
            "policy_evasion": "Strengthen policy enforcement mechanisms",
            "supply_chain_compromise": "Validate content sources and integrity",
            "operator_error": "Provide additional operator training and safeguards",
            "resource_exhaustion": "Implement resource monitoring and throttling"
        }
        
        for motif in motifs:
            if motif.motif_id in motif_recommendations:
                recommendations.append(motif_recommendations[motif.motif_id])
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _generate_prevention_strategies(self, motifs: List[CausalMotif]) -> List[str]:
        """Generate prevention strategies"""
        
        strategies = [
            "Implement automated causal pattern detection",
            "Enhance real-time monitoring of event sequences",
            "Develop proactive intervention based on early warning signs"
        ]
        
        # Motif-specific strategies
        for motif in motifs:
            if motif.motif_id == "rag_drift":
                strategies.append("Deploy semantic content analysis before model inference")
            elif motif.motif_id == "policy_evasion":
                strategies.append "Implement multi-layer policy validation"
            elif motif.motif_id == "supply_chain_compromise":
                strategies.append("Establish content provenance verification")
        
        return strategies[:4]  # Top 4 strategies
    
    def get_causal_statistics(self) -> Dict[str, Any]:
        """Get causal analysis statistics"""
        
        # Motif frequency analysis
        motif_stats = {}
        for motif_id, frequency in self.pattern_frequencies.items():
            motif_stats[motif_id] = {
                "frequency": frequency,
                "description": self._get_motif_description(motif_id),
                "avg_resolution_time": self._calculate_avg_resolution_time(motif_id)
            }
        
        # Analysis quality metrics
        if self.incident_history:
            avg_confidence = sum(inc.confidence for inc in self.incident_history) / len(self.incident_history)
            avg_primary_causes = sum(len(inc.primary_causes) for inc in self.incident_history) / len(self.incident_history)
        else:
            avg_confidence = 0.0
            avg_primary_causes = 0.0
        
        return {
            "total_events": len(self.event_nodes),
            "causal_relationships": len(self.causal_edges),
            "incidents_analyzed": len(self.incident_history),
            "motif_frequencies": motif_stats,
            "avg_analysis_confidence": round(avg_confidence, 3),
            "avg_primary_causes": round(avg_primary_causes, 1),
            "learning_maturity": min(1.0, len(self.incident_history) / 100),
            "last_update": datetime.now(timezone.utc).isoformat()
        }

# Global causal explainer instance
causal_explainer = CausalIncidentExplainer()

def add_causal_event(event_data: Dict[str, Any]) -> str:
    """Add event to causal analysis"""
    return causal_explainer.add_event(event_data)

def analyze_incident_causality(incident_events: List[str], 
                             incident_id: str = None) -> Dict[str, Any]:
    """Analyze incident causality"""
    analysis = causal_explainer.analyze_incident(incident_events, incident_id)
    return asdict(analysis)

def get_causal_analysis_status() -> Dict[str, Any]:
    """Get causal analysis status"""
    return causal_explainer.get_causal_statistics()

if __name__ == "__main__":
    # Test causal analysis
    test_events = [
        {
            "event_type": "source_request",
            "source": "external_api",
            "details": {"domain": "malicious-site.com"},
            "risk_score": 0.8,
            "attributes": {"user_agent": "curl/7.68.0"}
        },
        {
            "event_type": "content_retrieval",
            "source": "rag_database",
            "details": {"documents_retrieved": 5},
            "risk_score": 0.6,
            "attributes": {"query_type": "injection_attempt"}
        },
        {
            "event_type": "model_inference",
            "source": "llm_service",
            "details": {"model": "gpt-4"},
            "risk_score": 0.9,
            "attributes": {"output_anomaly": True}
        }
    ]
    
    event_ids = []
    for event in test_events:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        event_id = add_causal_event(event)
        event_ids.append(event_id)
    
    # Analyze incident
    analysis = analyze_incident_causality(event_ids)
    print(json.dumps(analysis, indent=2, default=str))
    
    # Get statistics
    stats = get_causal_analysis_status()
    print("\nCausal Analysis Statistics:")
    print(json.dumps(stats, indent=2, default=str))