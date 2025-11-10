"""
AEGIS‑C Adaptive Policy Engine
===============================

Multi-armed bandit policy selection with Thompson Sampling.
"""

import os
import sys
import random
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import logger

class ActionType(Enum):
    """Available policy actions"""
    OBSERVE = "observe"
    RAISE_FRICTION = "raise_friction"
    BLOCK_FAMILY = "block_family"
    DRAIN_NODE = "drain_node"
    RESET_GPU = "reset_gpu"
    REATTEST = "reattest"
    QUARANTINE_HOST = "quarantine_host"
    THROTTLE_REQUESTS = "throttle_requests"
    INCREASE_MONITORING = "increase_monitoring"
    NOTIFY_ANALYST = "notify_analyst"

@dataclass
class PolicyAction:
    """Policy action with metadata"""
    action: ActionType
    confidence: float
    expected_success_rate: float
    cost_estimate: float
    execution_time_ms: int
    description: str

@dataclass
class PolicyOutcome:
    """Result of policy action execution"""
    action: ActionType
    success: bool
    effectiveness_score: float  # 0-1
    cost_actual: float
    execution_time_ms: int
    false_positive: bool
    timestamp: str
    context: Dict[str, Any]

class AdaptivePolicyEngine:
    """Adaptive policy engine using multi-armed bandits"""
    
    def __init__(self):
        # Thompson Sampling priors for each action (Beta distribution: alpha, beta)
        self.action_priors = {
            ActionType.OBSERVE: [1, 1],           # Low cost, low effectiveness
            ActionType.RAISE_FRICTION: [3, 2],    # Medium cost, medium effectiveness
            ActionType.BLOCK_FAMILY: [5, 3],      # High cost, high effectiveness
            ActionType.DRAIN_NODE: [4, 2],        # High cost, high effectiveness
            ActionType.RESET_GPU: [3, 1],         # Medium cost, medium effectiveness
            ActionType.REATTEST: [2, 1],          # Low cost, low effectiveness
            ActionType.QUARANTINE_HOST: [6, 2],   # Very high cost, high effectiveness
            ActionType.THROTTLE_REQUESTS: [2, 2], # Low cost, medium effectiveness
            ActionType.INCREASE_MONITORING: [2, 1], # Low cost, low effectiveness
            ActionType.NOTIFY_ANALYST: [1, 1]     # Low cost, unknown effectiveness
        }
        
        # Action history for learning
        self.action_history: List[PolicyOutcome] = []
        self.context_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.total_decisions = 0
        self.success_rate_history = []
        
    def choose_action(self, risk_score: float, context: Dict[str, Any]) -> PolicyAction:
        """Choose best action using Thompson Sampling"""
        
        # Filter actions based on risk level and context
        available_actions = self._filter_actions(risk_score, context)
        
        if not available_actions:
            # Fallback to observe
            available_actions = [ActionType.OBSERVE]
        
        # Thompson Sampling: draw from each action's Beta distribution
        samples = {}
        for action in available_actions:
            alpha, beta = self.action_priors[action]
            sample = random.betavariate(alpha, beta)
            samples[action] = sample
        
        # Select action with highest sample
        best_action = max(samples, key=samples.get)
        
        # Calculate confidence and expected success rate
        alpha, beta = self.action_priors[best_action]
        expected_success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        confidence = min(1.0, (alpha + beta) / 10.0)  # Confidence based on sample size
        
        # Get action metadata
        action_metadata = self._get_action_metadata(best_action, context)
        
        # Store context for learning
        self.context_history.append({
            "action": best_action.value,
            "risk_score": risk_score,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        self.total_decisions += 1
        
        logger.info(
            "Policy action chosen",
            action=best_action.value,
            risk_score=risk_score,
            confidence=round(confidence, 3),
            expected_success=round(expected_success_rate, 3)
        )
        
        return PolicyAction(
            action=best_action,
            confidence=round(confidence, 3),
            expected_success_rate=round(expected_success_rate, 3),
            cost_estimate=action_metadata["cost"],
            execution_time_ms=action_metadata["execution_time"],
            description=action_metadata["description"]
        )
    
    def _filter_actions(self, risk_score: float, context: Dict[str, Any]) -> List[ActionType]:
        """Filter available actions based on risk and context"""
        actions = []
        
        # Always available
        actions.append(ActionType.OBSERVE)
        actions.append(ActionType.NOTIFY_ANALYST)
        
        # Low risk (0.0-0.4)
        if risk_score < 0.4:
            actions.extend([
                ActionType.INCREASE_MONITORING,
                ActionType.THROTTLE_REQUESTS
            ])
        
        # Medium risk (0.4-0.6)
        elif risk_score < 0.6:
            actions.extend([
                ActionType.RAISE_FRICTION,
                ActionType.THROTTLE_REQUESTS,
                ActionType.INCREASE_MONITORING,
                ActionType.REATTEST
            ])
        
        # High risk (0.6-0.8)
        elif risk_score < 0.8:
            actions.extend([
                ActionType.RAISE_FRICTION,
                ActionType.BLOCK_FAMILY,
                ActionType.RESET_GPU,
                ActionType.REATTEST
            ])
        
        # Critical risk (0.8-1.0)
        else:
            actions.extend([
                ActionType.BLOCK_FAMILY,
                ActionType.DRAIN_NODE,
                ActionType.RESET_GPU,
                ActionType.QUARANTINE_HOST
            ])
        
        # Context-specific filtering
        if context.get("hardware_involved", False):
            # Hardware-related context
            if context.get("gpu_available", False):
                actions.extend([ActionType.RESET_GPU, ActionType.DRAIN_NODE])
            else:
                # Remove GPU-specific actions if no GPU
                actions = [a for a in actions if a not in [ActionType.RESET_GPU]]
        
        if context.get("data_poisoning_detected", False):
            # Data poisoning context
            actions.extend([ActionType.QUARANTINE_HOST])
        
        return list(set(actions))  # Remove duplicates
    
    def _get_action_metadata(self, action: ActionType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for action"""
        metadata = {
            ActionType.OBSERVE: {
                "cost": 0.1,
                "execution_time": 100,
                "description": "Monitor situation without intervention"
            },
            ActionType.RAISE_FRICTION: {
                "cost": 2.0,
                "execution_time": 500,
                "description": "Increase authentication requirements and monitoring"
            },
            ActionType.BLOCK_FAMILY: {
                "cost": 5.0,
                "execution_time": 200,
                "description": "Block requests from identified model family"
            },
            ActionType.DRAIN_NODE: {
                "cost": 8.0,
                "execution_time": 2000,
                "description": "Drain traffic from affected node"
            },
            ActionType.RESET_GPU: {
                "cost": 6.0,
                "execution_time": 5000,
                "description": "Reset GPU hardware and clear memory"
            },
            ActionType.REATTEST: {
                "cost": 1.0,
                "execution_time": 1000,
                "description": "Re-run detection tests on affected system"
            },
            ActionType.QUARANTINE_HOST: {
                "cost": 10.0,
                "execution_time": 1500,
                "description": "Isolate host from network for investigation"
            },
            ActionType.THROTTLE_REQUESTS: {
                "cost": 1.5,
                "execution_time": 300,
                "description": "Rate limit requests from source"
            },
            ActionType.INCREASE_MONITORING: {
                "cost": 0.5,
                "execution_time": 200,
                "description": "Enhance logging and monitoring frequency"
            },
            ActionType.NOTIFY_ANALYST: {
                "cost": 0.2,
                "execution_time": 50,
                "description": "Send alert to security analyst"
            }
        }
        
        base_metadata = metadata.get(action, metadata[ActionType.OBSERVE])
        
        # Adjust based on context
        if context.get("emergency", False):
            base_metadata["execution_time"] = int(base_metadata["execution_time"] * 0.5)
        
        return base_metadata
    
    def update_action_outcome(self, action: ActionType, success: bool, 
                            effectiveness_score: float, cost_actual: float,
                            execution_time_ms: int, false_positive: bool = False,
                            context: Dict[str, Any] = None):
        """Update action outcome for learning"""
        
        outcome = PolicyOutcome(
            action=action,
            success=success,
            effectiveness_score=effectiveness_score,
            cost_actual=cost_actual,
            execution_time_ms=execution_time_ms,
            false_positive=false_positive,
            timestamp=datetime.now(timezone.utc).isoformat(),
            context=context or {}
        )
        
        # Store outcome
        self.action_history.append(outcome)
        
        # Update Thompson Sampling priors
        alpha, beta = self.action_priors[action]
        
        if success and not false_positive:
            # Success: increment alpha
            self.action_priors[action][0] += 1
        else:
            # Failure: increment beta
            self.action_priors[action][1] += 1
        
        # Update success rate history
        recent_outcomes = [o for o in self.action_history[-100:] if o.action == action]
        if recent_outcomes:
            success_rate = sum(1 for o in recent_outcomes if o.success and not o.false_positive) / len(recent_outcomes)
            self.success_rate_history.append({
                "action": action.value,
                "success_rate": success_rate,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        logger.info(
            "Policy action outcome recorded",
            action=action.value,
            success=success,
            effectiveness=round(effectiveness_score, 3),
            false_positive=false_positive,
            total_outcomes=len(self.action_history)
        )
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics for all actions"""
        stats = {}
        
        for action in ActionType:
            alpha, beta = self.action_priors[action]
            
            # Get recent outcomes
            recent_outcomes = [o for o in self.action_history[-50:] if o.action == action]
            
            if recent_outcomes:
                success_rate = sum(1 for o in recent_outcomes if o.success and not o.false_positive) / len(recent_outcomes)
                avg_effectiveness = sum(o.effectiveness_score for o in recent_outcomes) / len(recent_outcomes)
                avg_cost = sum(o.cost_actual for o in recent_outcomes) / len(recent_outcomes)
            else:
                success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                avg_effectiveness = 0.5
                avg_cost = 1.0
            
            stats[action.value] = {
                "success_rate": round(success_rate, 3),
                "expected_success_rate": round(alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5, 3),
                "avg_effectiveness": round(avg_effectiveness, 3),
                "avg_cost": round(avg_cost, 2),
                "total_trials": int(alpha + beta - 2),  # Subtract initial priors
                "confidence": round(min(1.0, (alpha + beta) / 10.0), 3)
            }
        
        return stats
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get overall learning progress"""
        if not self.action_history:
            return {"message": "No learning data available"}
        
        # Calculate overall metrics
        total_success_rate = sum(1 for o in self.action_history if o.success and not o.false_positive) / len(self.action_history)
        avg_effectiveness = sum(o.effectiveness_score for o in self.action_history) / len(self.action_history)
        
        # Calculate improvement over time
        if len(self.action_history) >= 100:
            early_outcomes = self.action_history[:50]
            recent_outcomes = self.action_history[-50:]
            
            early_success = sum(1 for o in early_outcomes if o.success and not o.false_positive) / len(early_outcomes)
            recent_success = sum(1 for o in recent_outcomes if o.success and not o.false_positive) / len(recent_outcomes)
            
            improvement = (recent_success - early_success) / early_success if early_success > 0 else 0
        else:
            improvement = 0
        
        return {
            "total_decisions": self.total_decisions,
            "total_outcomes": len(self.action_history),
            "overall_success_rate": round(total_success_rate, 3),
            "avg_effectiveness": round(avg_effectiveness, 3),
            "improvement_percentage": round(improvement * 100, 1),
            "learning_maturity": min(1.0, len(self.action_history) / 1000),
            "last_update": datetime.now(timezone.utc).isoformat()
        }

# Global policy engine instance
policy_engine = AdaptivePolicyEngine()

def choose_adaptive_action(risk_score: float, context: Dict[str, Any]) -> PolicyAction:
    """Choose adaptive policy action"""
    return policy_engine.choose_action(risk_score, context)

def update_policy_outcome(action: ActionType, success: bool, effectiveness_score: float,
                         cost_actual: float, execution_time_ms: int, 
                         false_positive: bool = False, context: Dict[str, Any] = None):
    """Update policy outcome for learning"""
    policy_engine.update_action_outcome(
        action, success, effectiveness_score, cost_actual, 
        execution_time_ms, false_positive, context
    )

def get_policy_status() -> Dict[str, Any]:
    """Get current policy engine status"""
    return {
        "action_statistics": policy_engine.get_action_statistics(),
        "learning_progress": policy_engine.get_learning_progress(),
        "total_decisions": policy_engine.total_decisions
    }

# Convenience functions for backward compatibility
def choose() -> str:
    """Legacy choose function"""
    action = policy_engine.choose_action(0.5, {})
    return action.action.value

def update(action: str, success: bool):
    """Legacy update function"""
    try:
        action_enum = ActionType(action)
        policy_engine.update_action_outcome(action_enum, success, 1.0 if success else 0.0, 1.0, 100)
    except ValueError:
        logger.warning(f"Unknown action: {action}")

if __name__ == "__main__":
    # Test the policy engine
    test_context = {
        "hardware_involved": True,
        "gpu_available": True,
        "emergency": False
    }
    
    action = choose_adaptive_action(0.8, test_context)
    print(json.dumps(asdict(action), indent=2, default=str))
    
    # Update outcome
    update_policy_outcome(action.action, True, 0.9, 5.0, 2000, False, test_context)
    
    # Get statistics
    stats = get_policy_status()
    print(json.dumps(stats, indent=2, default=str))