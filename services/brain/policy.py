"""
AEGIS‑C Adaptive Policy Engine
===============================

Multi-armed bandit policy selection with Thompson Sampling and CVE prioritization.
"""

import os
import sys
import random
import math
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
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
    URGENT_PATCH = "urgent_patch"
    ISOLATE_VULNERABLE_COMPONENT = "isolate_vulnerable_component"

class PolicyAction:
    """Policy action with metadata"""
    def __init__(self, action: ActionType, confidence: float, reason: str, metadata: Dict[str, Any] = None):
        self.action = action
        self.confidence = confidence
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "action": self.action.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class ContextualBandit:
    """Contextual multi-armed bandit for policy selection"""
    
    def __init__(self, actions: List[ActionType]):
        self.actions = actions
        self.counts = {action.value: 0 for action in actions}
        self.rewards = {action.value: 0.0 for action in actions}
        self.total_pulls = 0
        self.exploration_rate = 0.2
        self.min_confidence = 0.5
    
    def select_action(self, context: Dict[str, Any], risk_score: float, 
                     cve_impacts: List[str] = None) -> Tuple[ActionType, float, str]:
        """Select action using epsilon-greedy strategy with context"""
        self.total_pulls += 1
        
        # Adjust exploration rate based on total pulls
        self.exploration_rate = max(0.05, self.exploration_rate * 0.999)
        
        if random.random() < self.exploration_rate:
            # Explore: randomly select an action
            action = random.choice(self.actions)
            confidence = self.min_confidence
            reason = f"Exploration (epsilon={self.exploration_rate:.3f})"
        else:
            # Exploit: select best action based on historical rewards and context
            action, confidence, reason = self._select_best_action(context, risk_score, cve_impacts)
        
        self.counts[action.value] += 1
        return action, confidence, reason
    
    def _select_best_action(self, context: Dict[str, Any], risk_score: float, 
                           cve_impacts: List[str] = None) -> Tuple[ActionType, float, str]:
        """Select best action based on historical rewards and context"""
        best_action = None
        best_score = -float('inf')
        reason = "Default action selection"
        confidence = self.min_confidence
        
        for action in self.actions:
            action_value = action.value
            # Base score on historical reward
            if self.counts[action_value] == 0:
                score = 0.0  # Unexplored actions have neutral score
            else:
                score = self.rewards[action_value] / self.counts[action_value]
            
            # Adjust score based on risk level
            metadata = self._get_action_metadata(action, context)
            cost = metadata.get("cost", 1.0)
            
            if risk_score >= 0.85:
                # Critical risk: prefer high-impact actions
                if action in [ActionType.QUARANTINE_HOST, ActionType.DRAIN_NODE, ActionType.URGENT_PATCH, ActionType.ISOLATE_VULNERABLE_COMPONENT]:
                    score += 2.0
                    confidence = max(confidence, 0.9)
                    reason = "Critical risk requires immediate strong action"
            elif risk_score >= 0.7:
                # High risk: prefer moderate to high impact
                if action in [ActionType.RAISE_FRICTION, ActionType.DRAIN_NODE, ActionType.RESET_GPU, ActionType.URGENT_PATCH]:
                    score += 1.5
                    confidence = max(confidence, 0.8)
                    reason = "High risk warrants protective measures"
            elif risk_score >= 0.5:
                # Medium risk: prefer low to moderate impact
                if action in [ActionType.RAISE_FRICTION, ActionType.INCREASE_MONITORING, ActionType.THROTTLE_REQUESTS]:
                    score += 1.0
                    confidence = max(confidence, 0.75)
                    reason = "Medium risk suggests cautionary measures"
            else:
                # Low risk: prefer minimal impact
                if action in [ActionType.OBSERVE, ActionType.INCREASE_MONITORING]:
                    score += 0.5
                    confidence = max(confidence, 0.6)
                    reason = "Low risk allows minimal intervention"
            
            # Adjust score based on CVE impacts if available
            if cve_impacts and action in [ActionType.URGENT_PATCH, ActionType.ISOLATE_VULNERABLE_COMPONENT]:
                if any(impact in cve_impacts for impact in ["model_injection", "data_poisoning", "adversarial_attack"]):
                    score += 2.5
                    confidence = max(confidence, 0.95)
                    reason = f"Critical CVE impact ({', '.join(cve_impacts)}) requires urgent mitigation"
            
            # Adjust for cost (prefer lower cost for similar scores)
            score -= cost * 0.1
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action, confidence, reason
    
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
            },
            ActionType.URGENT_PATCH: {
                "cost": 5.0,
                "execution_time": 1000,
                "description": "Apply urgent patch to mitigate vulnerability"
            },
            ActionType.ISOLATE_VULNERABLE_COMPONENT: {
                "cost": 7.0,
                "execution_time": 1200,
                "description": "Isolate component affected by critical vulnerability"
            }
        }
        
        return metadata.get(action, {
            "cost": 1.0,
            "execution_time": 100,
            "description": f"Perform {action.value}"
        })
    
    def update_reward(self, action: ActionType, reward: float):
        """Update reward for the action"""
        self.rewards[action.value] += reward
        logger.info(
            "Bandit reward updated",
            action=action.value,
            reward=reward,
            total_count=self.counts[action.value],
            avg_reward=self.rewards[action.value] / max(1, self.counts[action.value])
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics"""
        stats = {}
        for action in self.actions:
            action_val = action.value
            count = self.counts[action_val]
            total_reward = self.rewards[action_val]
            avg_reward = total_reward / count if count > 0 else 0.0
            stats[action_val] = {
                "pulls": count,
                "total_reward": round(total_reward, 2),
                "avg_reward": round(avg_reward, 3)
            }
        return {
            "actions": stats,
            "total_pulls": self.total_pulls,
            "exploration_rate": round(self.exploration_rate, 3)
        }

class AdaptivePolicyEngine:
    """Adaptive policy engine using contextual multi-armed bandits"""
    
    def __init__(self):
        self.bandit = ContextualBandit(list(ActionType))
        self.action_history: List[PolicyAction] = []
        self.context_history: List[Dict[str, Any]] = []
        self.total_decisions = 0
        self.success_rate_history = []
    
    def choose_action(self, risk_score: float, context: Dict[str, Any], 
                     cve_impacts: List[str] = None) -> PolicyAction:
        """Choose best action using contextual bandit"""
        
        # Filter actions based on risk level and context
        available_actions = self._filter_actions(risk_score, context)
        
        if not available_actions:
            # Fallback to observe
            available_actions = [ActionType.OBSERVE]
        
        # Select action using bandit
        action, confidence, reason = self.bandit.select_action(context, risk_score, cve_impacts)
        
        # Ensure action is valid for context
        if action not in available_actions:
            action = available_actions[0]
            confidence = 0.5
            reason = f"Adjusted to valid action: {action.value}"
        
        # Get metadata for action
        metadata = self.bandit._get_action_metadata(action, context)
        metadata["risk_score"] = risk_score
        if cve_impacts:
            metadata["cve_impacts"] = cve_impacts
        
        logger.info(
            "Policy action chosen",
            action=action.value,
            risk_score=risk_score,
            confidence=round(confidence, 3),
            reason=reason
        )
        
        self.total_decisions += 1
        
        return PolicyAction(action, confidence, reason, metadata)
    
    def update_action_outcome(self, action: ActionType, success: bool, 
                            effectiveness_score: float, cost_actual: float,
                            execution_time_ms: int, false_positive: bool = False,
                            context: Dict[str, Any] = None):
        """Update action outcome for learning"""
        
        outcome = PolicyAction(action, 0.0, "Outcome update")
        outcome.metadata = {
            "success": success,
            "effectiveness_score": effectiveness_score,
            "cost_actual": cost_actual,
            "execution_time_ms": execution_time_ms,
            "false_positive": false_positive,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context or {}
        }
        
        # Store outcome
        self.action_history.append(outcome)
        
        # Update bandit reward
        reward = max(-1.0, min(1.0, effectiveness_score))
        self.bandit.update_reward(action, reward)
        
        # Update success rate history
        recent_outcomes = [o for o in self.action_history[-100:] if o.action == action]
        if recent_outcomes:
            success_rate = sum(1 for o in recent_outcomes if o.metadata["success"] and not o.metadata["false_positive"]) / len(recent_outcomes)
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
            alpha, beta = self._get_action_priors(action)
            
            # Get recent outcomes
            recent_outcomes = [o for o in self.action_history[-50:] if o.action == action]
            
            if recent_outcomes:
                success_rate = sum(1 for o in recent_outcomes if o.metadata["success"] and not o.metadata["false_positive"]) / len(recent_outcomes)
                avg_effectiveness = sum(o.metadata["effectiveness_score"] for o in recent_outcomes) / len(recent_outcomes)
                avg_cost = sum(o.metadata["cost_actual"] for o in recent_outcomes) / len(recent_outcomes)
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
        total_success_rate = sum(1 for o in self.action_history if o.metadata["success"] and not o.metadata["false_positive"]) / len(self.action_history)
        avg_effectiveness = sum(o.metadata["effectiveness_score"] for o in self.action_history) / len(self.action_history)
        
        # Calculate improvement over time
        if len(self.action_history) >= 100:
            early_outcomes = self.action_history[:50]
            recent_outcomes = self.action_history[-50:]
            
            early_success = sum(1 for o in early_outcomes if o.metadata["success"] and not o.metadata["false_positive"]) / len(early_outcomes)
            recent_success = sum(1 for o in recent_outcomes if o.metadata["success"] and not o.metadata["false_positive"]) / len(recent_outcomes)
            
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
    
    def _get_action_priors(self, action: ActionType) -> Tuple[float, float]:
        """Get Thompson Sampling priors for action"""
        # Default priors
        priors = {
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
        
        return priors.get(action, [1, 1])

# Global policy engine instance
policy_engine = AdaptivePolicyEngine()

def choose_adaptive_action(risk_score: float, context: Dict[str, Any], 
                          cve_impacts: List[str] = None) -> PolicyAction:
    """Choose adaptive policy action"""
    return policy_engine.choose_action(risk_score, context, cve_impacts)

def update_policy_outcome(action: ActionType, success: bool, 
                         effectiveness_score: float, cost_actual: float,
                         execution_time_ms: int, false_positive: bool = False,
                         context: Dict[str, Any] = None):
    """Update policy outcome for learning"""
    policy_engine.update_action_outcome(action, success, effectiveness_score, cost_actual, 
                                       execution_time_ms, false_positive, context)

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
    print(json.dumps(action.to_dict(), indent=2, default=str))
    
    # Update outcome
    update_policy_outcome(action.action, True, 0.9, 5.0, 2000, False, test_context)
    
    # Get statistics
    stats = get_policy_status()
    print(json.dumps(stats, indent=2, default=str))