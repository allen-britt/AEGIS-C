"""
AEGIS‑C Honeynet Adaptive Policies
===================================

Dynamic honeynet personality that adapts based on agent behavior.
"""

import os
import sys
import json
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import logger

class AgentProfile(Enum):
    """Agent behavior profiles"""
    FAST_BOT = "fast_bot"           # High frequency, low complexity
    BROWSER_MIMIC = "browser_mimic" # Full browser headers, human-like timing
    API_CLIENT = "api_client"       # Clean API calls, possibly automated
    PROBE_HEAVY = "probe_heavy"     # Lots of reconnaissance
    STEALTHY = "stealthy"           # Low frequency, careful behavior
    UNKNOWN = "unknown"

class ResponseTemplate(Enum):
    """Honeynet response templates"""
    EXPENSIVE_ROUTE = "expensive_route"    # Slow, resource-intensive responses
    DOM_PUZZLE = "dom_puzzle"              # JavaScript challenges
    CANARY_DENSE = "canary_dense"          # Lots of canary tokens
    FAKE_ERROR = "fake_error"              # Misleading error messages
    HONEYPOT_DATA = "honeypot_data"        # Rich fake data
    REDIRECT_LOOP = "redirect_loop"        # Endless redirects
    TIMEOUT_TRAP = "timeout_trap"          # Very slow responses

@dataclass
class AgentTraits:
    """Observed agent characteristics"""
    request_frequency: float  # requests per second
    header_complexity: float  # 0-1 scale
    session_depth: int        # pages visited
    retry_pattern: str        # immediate, delayed, exponential
    javascript_support: bool  # can execute JS
    cookie_handling: bool     # accepts/returns cookies
    referer_presence: bool    # includes referer
    user_agent_length: int    # characters in UA string
    request_size_variance: float  # variance in request sizes
    timing_pattern: str       # human, bot, mixed
    discovery_attempts: int   # number of probe attempts

@dataclass
class HoneynetPolicy:
    """Dynamic honeynet policy"""
    agent_profile: AgentProfile
    response_template: ResponseTemplate
    canary_density: float     # 0-1 scale
    response_delay_ms: int
    data_complexity: str      # simple, complex, misleading
    tracking_level: str       # minimal, standard, aggressive
    adaptation_confidence: float
    last_updated: str

class AdaptiveHoneynetPolicies:
    """Adaptive honeynet policy engine"""
    
    def __init__(self):
        # Agent behavior patterns
        self.behavior_patterns = self._initialize_behavior_patterns()
        
        # Policy mappings
        self.policy_mappings = self._initialize_policy_mappings()
        
        # Learning data
        self.agent_sessions: List[Dict[str, Any]] = []
        self.policy_effectiveness: Dict[str, Dict[str, float]] = {}
        
        # Adaptation parameters
        self.adaptation_threshold = 5  # min sessions before adaptation
        self.learning_rate = 0.1
        
    def _initialize_behavior_patterns(self) -> Dict[AgentProfile, AgentTraits]:
        """Initialize typical behavior patterns for each profile"""
        return {
            AgentProfile.FAST_BOT: AgentTraits(
                request_frequency=10.0,
                header_complexity=0.2,
                session_depth=1,
                retry_pattern="immediate",
                javascript_support=False,
                cookie_handling=False,
                referer_presence=False,
                user_agent_length=20,
                request_size_variance=0.1,
                timing_pattern="bot",
                discovery_attempts=1
            ),
            AgentProfile.BROWSER_MIMIC: AgentTraits(
                request_frequency=0.5,
                header_complexity=0.9,
                session_depth=5,
                retry_pattern="delayed",
                javascript_support=True,
                cookie_handling=True,
                referer_presence=True,
                user_agent_length=150,
                request_size_variance=0.3,
                timing_pattern="human",
                discovery_attempts=3
            ),
            AgentProfile.API_CLIENT: AgentTraits(
                request_frequency=2.0,
                header_complexity=0.5,
                session_depth=2,
                retry_pattern="exponential",
                javascript_support=False,
                cookie_handling=False,
                referer_presence=False,
                user_agent_length=80,
                request_size_variance=0.2,
                timing_pattern="bot",
                discovery_attempts=5
            ),
            AgentProfile.PROBE_HEAVY: AgentTraits(
                request_frequency=3.0,
                header_complexity=0.6,
                session_depth=10,
                retry_pattern="immediate",
                javascript_support=False,
                cookie_handling=True,
                referer_presence=False,
                user_agent_length=100,
                request_size_variance=0.5,
                timing_pattern="mixed",
                discovery_attempts=15
            ),
            AgentProfile.STEALTHY: AgentTraits(
                request_frequency=0.1,
                header_complexity=0.7,
                session_depth=3,
                retry_pattern="delayed",
                javascript_support=True,
                cookie_handling=True,
                referer_presence=True,
                user_agent_length=120,
                request_size_variance=0.4,
                timing_pattern="human",
                discovery_attempts=8
            )
        }
    
    def _initialize_policy_mappings(self) -> Dict[AgentProfile, ResponseTemplate]:
        """Initialize base policy mappings"""
        return {
            AgentProfile.FAST_BOT: ResponseTemplate.EXPENSIVE_ROUTE,
            AgentProfile.BROWSER_MIMIC: ResponseTemplate.DOM_PUZZLE,
            AgentProfile.API_CLIENT: ResponseTemplate.CANARY_DENSE,
            AgentProfile.PROBE_HEAVY: ResponseTemplate.HONEYPOT_DATA,
            AgentProfile.STEALTHY: ResponseTemplate.FAKE_ERROR
        }
    
    def analyze_agent_traits(self, session_data: Dict[str, Any]) -> AgentTraits:
        """Analyze agent behavior from session data"""
        
        # Extract request patterns
        requests = session_data.get("requests", [])
        if not requests:
            return self.behavior_patterns[AgentProfile.UNKNOWN]
        
        # Calculate timing metrics
        timestamps = [r.get("timestamp", 0) for r in requests]
        if len(timestamps) > 1:
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            request_frequency = 1.0 / (sum(time_diffs) / len(time_diffs)) if time_diffs else 0
        else:
            request_frequency = 0.1
        
        # Analyze headers
        headers = requests[0].get("headers", {})
        user_agent = headers.get("user-agent", "")
        header_complexity = min(1.0, len(headers) / 20.0)
        
        # Check for specific indicators
        javascript_support = "javascript" in headers.get("accept", "").lower()
        cookie_handling = "cookie" in headers or session_data.get("cookies_returned", False)
        referer_presence = "referer" in headers and headers["referer"]
        
        # Calculate session metrics
        session_depth = len(set(r.get("path", "") for r in requests))
        
        # Analyze retry pattern
        if len(requests) > 1:
            retry_intervals = [time_diffs[i] for i in range(len(time_diffs)) if time_diffs[i] < 60]
            if retry_intervals:
                avg_interval = sum(retry_intervals) / len(retry_intervals)
                if avg_interval < 1:
                    retry_pattern = "immediate"
                elif avg_interval < 10:
                    retry_pattern = "delayed"
                else:
                    retry_pattern = "exponential"
            else:
                retry_pattern = "none"
        else:
            retry_pattern = "none"
        
        # Calculate request size variance
        request_sizes = [len(str(r.get("body", ""))) for r in requests]
        if len(request_sizes) > 1:
            avg_size = sum(request_sizes) / len(request_sizes)
            variance = sum((s - avg_size) ** 2 for s in request_sizes) / len(request_sizes)
            request_size_variance = min(1.0, variance / 10000.0)
        else:
            request_size_variance = 0.0
        
        # Determine timing pattern
        if request_frequency > 5:
            timing_pattern = "bot"
        elif request_frequency < 1:
            timing_pattern = "human"
        else:
            timing_pattern = "mixed"
        
        # Count discovery attempts (probing different endpoints)
        paths = [r.get("path", "") for r in requests]
        discovery_attempts = len(set(p for p in paths if any(keyword in p for keyword in ["api", "admin", "config", "debug"])))
        
        return AgentTraits(
            request_frequency=request_frequency,
            header_complexity=header_complexity,
            session_depth=session_depth,
            retry_pattern=retry_pattern,
            javascript_support=javascript_support,
            cookie_handling=cookie_handling,
            referer_presence=referer_presence,
            user_agent_length=len(user_agent),
            request_size_variance=request_size_variance,
            timing_pattern=timing_pattern,
            discovery_attempts=discovery_attempts
        )
    
    def classify_agent_profile(self, traits: AgentTraits) -> AgentProfile:
        """Classify agent based on observed traits"""
        
        # Calculate similarity scores for each profile
        similarity_scores = {}
        
        for profile, pattern in self.behavior_patterns.items():
            score = self._calculate_trait_similarity(traits, pattern)
            similarity_scores[profile] = score
        
        # Select best match
        best_profile = max(similarity_scores, key=similarity_scores.get)
        confidence = similarity_scores[best_profile]
        
        logger.info(
            "Agent profile classified",
            profile=best_profile.value,
            confidence=round(confidence, 3),
            traits=asdict(traits)
        )
        
        return best_profile if confidence > 0.3 else AgentProfile.UNKNOWN
    
    def _calculate_trait_similarity(self, observed: AgentTraits, pattern: AgentTraits) -> float:
        """Calculate similarity between observed traits and pattern"""
        
        # Normalize and compare each trait
        similarities = []
        
        # Frequency similarity (log scale)
        freq_sim = 1.0 - min(1.0, abs(observed.request_frequency - pattern.request_frequency) / 10.0)
        similarities.append(freq_sim)
        
        # Header complexity
        header_sim = 1.0 - abs(observed.header_complexity - pattern.header_complexity)
        similarities.append(header_sim)
        
        # Session depth
        depth_sim = 1.0 - min(1.0, abs(observed.session_depth - pattern.session_depth) / 10.0)
        similarities.append(depth_sim)
        
        # Boolean traits
        similarities.append(1.0 if observed.javascript_support == pattern.javascript_support else 0.0)
        similarities.append(1.0 if observed.cookie_handling == pattern.cookie_handling else 0.0)
        similarities.append(1.0 if observed.referer_presence == pattern.referer_presence else 0.0)
        
        # User agent length
        ua_sim = 1.0 - min(1.0, abs(observed.user_agent_length - pattern.user_agent_length) / 200.0)
        similarities.append(ua_sim)
        
        # Request size variance
        var_sim = 1.0 - abs(observed.request_size_variance - pattern.request_size_variance)
        similarities.append(var_sim)
        
        # Timing pattern
        timing_sim = 1.0 if observed.timing_pattern == pattern.timing_pattern else 0.5
        similarities.append(timing_sim)
        
        # Discovery attempts
        disc_sim = 1.0 - min(1.0, abs(observed.discovery_attempts - pattern.discovery_attempts) / 20.0)
        similarities.append(disc_sim)
        
        # Weighted average
        weights = [0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.15]
        
        return sum(s * w for s, w in zip(similarities, weights))
    
    def generate_adaptive_policy(self, agent_profile: AgentProfile, 
                               session_history: List[Dict[str, Any]]) -> HoneynetPolicy:
        """Generate adaptive policy based on agent profile and history"""
        
        # Base template
        base_template = self.policy_mappings.get(agent_profile, ResponseTemplate.HONEYPOT_DATA)
        
        # Adapt based on session history
        adaptations = self._analyze_session_adaptations(agent_profile, session_history)
        
        # Calculate canary density
        canary_density = self._calculate_canary_density(agent_profile, adaptations)
        
        # Determine response delay
        response_delay = self._calculate_response_delay(agent_profile, adaptations)
        
        # Data complexity
        data_complexity = self._calculate_data_complexity(agent_profile, adaptations)
        
        # Tracking level
        tracking_level = self._calculate_tracking_level(agent_profile, adaptations)
        
        # Adaptation confidence
        confidence = min(1.0, len(session_history) / 10.0)
        
        policy = HoneynetPolicy(
            agent_profile=agent_profile,
            response_template=base_template,
            canary_density=canary_density,
            response_delay_ms=response_delay,
            data_complexity=data_complexity,
            tracking_level=tracking_level,
            adaptation_confidence=confidence,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(
            "Adaptive policy generated",
            profile=agent_profile.value,
            template=base_template.value,
            canary_density=round(canary_density, 2),
            response_delay=response_delay,
            confidence=round(confidence, 3)
        )
        
        return policy
    
    def _analyze_session_adaptations(self, agent_profile: AgentProfile, 
                                   session_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze past sessions to determine adaptations"""
        
        adaptations = {
            "speed_factor": 1.0,
            "complexity_factor": 1.0,
            "canary_effectiveness": 0.5,
            "success_rate": 0.5
        }
        
        if not session_history:
            return adaptations
        
        # Analyze success rates
        successful_sessions = [s for s in session_history if s.get("canary_triggered", False)]
        adaptations["success_rate"] = len(successful_sessions) / len(session_history)
        
        # Analyze session depth trends
        recent_depths = [s.get("session_depth", 1) for s in session_history[-5:]]
        if recent_depths:
            avg_depth = sum(recent_depths) / len(recent_depths)
            if avg_depth > 7:
                adaptations["complexity_factor"] = 1.5  # Increase complexity
            elif avg_depth < 3:
                adaptations["complexity_factor"] = 0.8  # Reduce complexity
        
        # Analyze speed patterns
        recent_frequencies = [s.get("request_frequency", 1.0) for s in session_history[-5:]]
        if recent_frequencies:
            avg_freq = sum(recent_frequencies) / len(recent_frequencies)
            if avg_freq > 5:
                adaptations["speed_factor"] = 2.0  # Slow down responses
            elif avg_freq < 0.5:
                adaptations["speed_factor"] = 0.5  # Normal speed
        
        # Canary effectiveness
        canary_triggers = [s for s in session_history if s.get("canary_triggered", False)]
        if canary_triggers:
            adaptations["canary_effectiveness"] = len(canary_triggers) / len(session_history)
        
        return adaptations
    
    def _calculate_canary_density(self, agent_profile: AgentProfile, 
                                adaptations: Dict[str, float]) -> float:
        """Calculate optimal canary density"""
        
        base_density = {
            AgentProfile.FAST_BOT: 0.8,
            AgentProfile.BROWSER_MIMIC: 0.3,
            AgentProfile.API_CLIENT: 0.6,
            AgentProfile.PROBE_HEAVY: 0.9,
            AgentProfile.STEALTHY: 0.4,
            AgentProfile.UNKNOWN: 0.5
        }
        
        density = base_density.get(agent_profile, 0.5)
        
        # Adapt based on effectiveness
        if adaptations["canary_effectiveness"] < 0.3:
            density = min(1.0, density * 1.2)  # Increase density
        elif adaptations["canary_effectiveness"] > 0.8:
            density = max(0.1, density * 0.8)  # Decrease density
        
        return round(density, 2)
    
    def _calculate_response_delay(self, agent_profile: AgentProfile, 
                                adaptations: Dict[str, float]) -> int:
        """Calculate optimal response delay"""
        
        base_delays = {
            AgentProfile.FAST_BOT: 2000,
            AgentProfile.BROWSER_MIMIC: 500,
            AgentProfile.API_CLIENT: 1000,
            AgentProfile.PROBE_HEAVY: 3000,
            AgentProfile.STEALTHY: 800,
            AgentProfile.UNKNOWN: 1000
        }
        
        delay = base_delays.get(agent_profile, 1000)
        
        # Adapt based on speed factor
        delay = int(delay * adaptations["speed_factor"])
        
        # Add randomness
        delay += random.randint(-200, 200)
        
        return max(100, delay)
    
    def _calculate_data_complexity(self, agent_profile: AgentProfile, 
                                 adaptations: Dict[str, float]) -> str:
        """Calculate data complexity level"""
        
        base_complexity = {
            AgentProfile.FAST_BOT: "misleading",
            AgentProfile.BROWSER_MIMIC: "complex",
            AgentProfile.API_CLIENT: "simple",
            AgentProfile.PROBE_HEAVY: "complex",
            AgentProfile.STEALTHY: "misleading",
            AgentProfile.UNKNOWN: "simple"
        }
        
        complexity = base_complexity.get(agent_profile, "simple")
        
        # Adapt based on complexity factor
        if adaptations["complexity_factor"] > 1.2:
            if complexity == "simple":
                complexity = "complex"
            elif complexity == "complex":
                complexity = "misleading"
        elif adaptations["complexity_factor"] < 0.8:
            if complexity == "misleading":
                complexity = "complex"
            elif complexity == "complex":
                complexity = "simple"
        
        return complexity
    
    def _calculate_tracking_level(self, agent_profile: AgentProfile, 
                                adaptations: Dict[str, float]) -> str:
        """Calculate tracking level"""
        
        if agent_profile in [AgentProfile.PROBE_HEAVY, AgentProfile.FAST_BOT]:
            return "aggressive"
        elif agent_profile == AgentProfile.STEALTHY:
            return "minimal"
        else:
            return "standard"
    
    def update_policy_effectiveness(self, policy: HoneynetPolicy, 
                                  session_outcome: Dict[str, Any]):
        """Update policy effectiveness based on session outcome"""
        
        policy_key = f"{policy.agent_profile.value}_{policy.response_template.value}"
        
        if policy_key not in self.policy_effectiveness:
            self.policy_effectiveness[policy_key] = {
                "sessions": 0,
                "canary_triggers": 0,
                "session_depth": 0,
                "time_to_detection": []
            }
        
        stats = self.policy_effectiveness[policy_key]
        stats["sessions"] += 1
        
        if session_outcome.get("canary_triggered", False):
            stats["canary_triggers"] += 1
        
        stats["session_depth"] += session_outcome.get("session_depth", 1)
        
        if session_outcome.get("time_to_detection"):
            stats["time_to_detection"].append(session_outcome["time_to_detection"])
        
        logger.info(
            "Policy effectiveness updated",
            policy_key=policy_key,
            sessions=stats["sessions"],
            canary_triggers=stats["canary_triggers"]
        )
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation and learning statistics"""
        
        # Profile distribution
        profile_counts = {}
        for session in self.agent_sessions:
            profile = session.get("agent_profile", "unknown")
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
        
        # Policy effectiveness
        policy_stats = {}
        for policy_key, stats in self.policy_effectiveness.items():
            if stats["sessions"] > 0:
                canary_rate = stats["canary_triggers"] / stats["sessions"]
                avg_depth = stats["session_depth"] / stats["sessions"]
                
                policy_stats[policy_key] = {
                    "sessions": stats["sessions"],
                    "canary_trigger_rate": round(canary_rate, 3),
                    "avg_session_depth": round(avg_depth, 1),
                    "effectiveness_score": round(canary_rate * 0.7 + (avg_depth / 10) * 0.3, 3)
                }
        
        return {
            "total_sessions": len(self.agent_sessions),
            "profile_distribution": profile_counts,
            "policy_effectiveness": policy_stats,
            "adaptation_maturity": min(1.0, len(self.agent_sessions) / 100),
            "last_update": datetime.now(timezone.utc).isoformat()
        }

# Global honeynet policies instance
honeynet_policies = AdaptiveHoneynetPolicies()

def analyze_and_adapt(session_data: Dict[str, Any], 
                     session_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze agent and generate adaptive policy"""
    
    # Analyze traits
    traits = honeynet_policies.analyze_agent_traits(session_data)
    
    # Classify profile
    profile = honeynet_policies.classify_agent_profile(traits)
    
    # Generate adaptive policy
    history = session_history or []
    policy = honeynet_policies.generate_adaptive_policy(profile, history)
    
    # Store session for learning
    session_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "traits": asdict(traits),
        "agent_profile": profile.value,
        "session_data": session_data
    }
    honeynet_policies.agent_sessions.append(session_record)
    
    return asdict(policy)

def update_session_outcome(policy: Dict[str, Any], session_outcome: Dict[str, Any]):
    """Update policy with session outcome"""
    
    # Convert back to policy object
    policy_obj = HoneynetPolicy(
        agent_profile=AgentProfile(policy["agent_profile"]),
        response_template=ResponseTemplate(policy["response_template"]),
        canary_density=policy["canary_density"],
        response_delay_ms=policy["response_delay_ms"],
        data_complexity=policy["data_complexity"],
        tracking_level=policy["tracking_level"],
        adaptation_confidence=policy["adaptation_confidence"],
        last_updated=policy["last_updated"]
    )
    
    honeynet_policies.update_policy_effectiveness(policy_obj, session_outcome)

def get_honeynet_adaptation_status() -> Dict[str, Any]:
    """Get honeynet adaptation status"""
    return honeynet_policies.get_adaptation_statistics()

if __name__ == "__main__":
    # Test adaptive policies
    test_session = {
        "requests": [
            {
                "timestamp": 1000,
                "headers": {
                    "user-agent": "curl/7.68.0",
                    "accept": "application/json"
                },
                "path": "/api/data"
            },
            {
                "timestamp": 1001,
                "headers": {
                    "user-agent": "curl/7.68.0",
                    "accept": "application/json"
                },
                "path": "/api/users"
            }
        ]
    }
    
    policy = analyze_and_adapt(test_session)
    print(json.dumps(policy, indent=2, default=str))
    
    # Get statistics
    stats = get_honeynet_adaptation_status()
    print("\nAdaptation Statistics:")
    print(json.dumps(stats, indent=2, default=str))