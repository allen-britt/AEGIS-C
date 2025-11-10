"""
AEGIS‑C Active Learning System
==============================

Human-in-the-loop learning from analyst corrections.
"""

import os
import sys
import json
import pickle
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import numpy as np

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import logger

class CorrectionType(Enum):
    """Types of analyst corrections"""
    FALSE_POSITIVE = "false_positive"       # Alert was incorrect
    FALSE_NEGATIVE = "false_negative"       # Missed threat
    MISATTRIBUTED_FAMILY = "misattributed_family"  # Wrong model family
    BAD_ACTION = "bad_action"               # Wrong policy action
    RISK_MISCALIBRATION = "risk_miscalibration"    # Wrong risk score
    PROBE_INEFFECTIVE = "probe_ineffective"        # Probe didn't work

class LearningTask(Enum):
    """Learning tasks for model updates"""
    RISK_SCORING = "risk_scoring"
    FAMILY_CLASSIFICATION = "family_classification"
    POLICY_SELECTION = "policy_selection"
    PROBE_GENERATION = "probe_generation"
    ANOMALY_DETECTION = "anomaly_detection"

@dataclass
class AnalystCorrection:
    """Analyst correction record"""
    correction_id: str
    timestamp: str
    correction_type: CorrectionType
    original_prediction: Dict[str, Any]
    corrected_label: Dict[str, Any]
    confidence: float
    analyst_id: str
    context: Dict[str, Any]
    feedback_notes: str

@dataclass
class TrainingBuffer:
    """Replay buffer for training data"""
    corrections: List[AnalystCorrection]
    max_size: int
    priority_scores: List[float]
    last_updated: str

@dataclass
class ModelUpdate:
    """Model update record"""
    update_id: str
    task: LearningTask
    previous_version: str
    new_version: str
    performance_improvement: Dict[str, float]
    training_samples: int
    update_timestamp: str
    validation_score: float

class ActiveLearningSystem:
    """Active learning system with human-in-the-loop feedback"""
    
    def __init__(self):
        # Training buffers for different tasks
        self.training_buffers: Dict[LearningTask, TrainingBuffer] = {
            task: TrainingBuffer(
                corrections=[],
                max_size=1000,
                priority_scores=[],
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            for task in LearningTask
        }
        
        # Model performance tracking
        self.performance_history: Dict[LearningTask, List[Dict[str, Any]]] = {
            task: [] for task in LearningTask
        }
        
        # Learning configuration
        self.min_samples_for_update = 10
        self.update_frequency_hours = 24
        self.performance_window_days = 7
        self.learning_rate = 0.01
        
        # Model version tracking
        self.model_versions: Dict[LearningTask, str] = {
            task: "1.0.0" for task in LearningTask
        }
        
        # Update history
        self.update_history: List[ModelUpdate] = []
        
        # Learning statistics
        self.learning_stats = {
            "total_corrections": 0,
            "corrections_by_type": defaultdict(int),
            "updates_performed": 0,
            "avg_improvement": 0.0
        }
    
    def add_correction(self, correction_data: Dict[str, Any]) -> str:
        """Add analyst correction to training buffer"""
        
        # Generate correction ID
        correction_hash = hashlib.md5(
            json.dumps(correction_data, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        correction_id = f"correction_{correction_hash}"
        
        # Create correction object
        correction = AnalystCorrection(
            correction_id=correction_id,
            timestamp=correction_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            correction_type=CorrectionType(correction_data["correction_type"]),
            original_prediction=correction_data["original_prediction"],
            corrected_label=correction_data["corrected_label"],
            confidence=correction_data.get("confidence", 0.5),
            analyst_id=correction_data.get("analyst_id", "unknown"),
            context=correction_data.get("context", {}),
            feedback_notes=correction_data.get("feedback_notes", "")
        )
        
        # Determine learning task
        task = self._determine_learning_task(correction.correction_type)
        
        # Calculate priority score
        priority = self._calculate_correction_priority(correction)
        
        # Add to training buffer
        buffer = self.training_buffers[task]
        
        # Maintain buffer size
        if len(buffer.corrections) >= buffer.max_size:
            # Remove lowest priority correction
            min_idx = np.argmin(buffer.priority_scores)
            buffer.corrections.pop(min_idx)
            buffer.priority_scores.pop(min_idx)
        
        buffer.corrections.append(correction)
        buffer.priority_scores.append(priority)
        buffer.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Update statistics
        self.learning_stats["total_corrections"] += 1
        self.learning_stats["corrections_by_type"][correction.correction_type.value] += 1
        
        logger.info(
            "Analyst correction added",
            correction_id=correction_id,
            correction_type=correction.correction_type.value,
            task=task.value,
            priority=round(priority, 3),
            buffer_size=len(buffer.corrections)
        )
        
        return correction_id
    
    def _determine_learning_task(self, correction_type: CorrectionType) -> LearningTask:
        """Determine learning task from correction type"""
        
        task_mapping = {
            CorrectionType.FALSE_POSITIVE: LearningTask.RISK_SCORING,
            CorrectionType.FALSE_NEGATIVE: LearningTask.RISK_SCORING,
            CorrectionType.MISATTRIBUTED_FAMILY: LearningTask.FAMILY_CLASSIFICATION,
            CorrectionType.BAD_ACTION: LearningTask.POLICY_SELECTION,
            CorrectionType.RISK_MISCALIBRATION: LearningTask.RISK_SCORING,
            CorrectionType.PROBE_INEFFECTIVE: LearningTask.PROBE_GENERATION
        }
        
        return task_mapping.get(correction_type, LearningTask.RISK_SCORING)
    
    def _calculate_correction_priority(self, correction: AnalystCorrection) -> float:
        """Calculate priority score for correction"""
        
        priority = 0.5  # Base priority
        
        # Higher priority for high-confidence corrections
        priority += correction.confidence * 0.3
        
        # Higher priority for certain correction types
        type_priorities = {
            CorrectionType.FALSE_NEGATIVE: 0.2,
            CorrectionType.BAD_ACTION: 0.2,
            CorrectionType.RISK_MISCALIBRATION: 0.15
        }
        priority += type_priorities.get(correction.correction_type, 0.0)
        
        # Higher priority for recent corrections
        correction_time = datetime.fromisoformat(correction.timestamp.replace('Z', '+00:00'))
        age_hours = (datetime.now(timezone.utc) - correction_time).total_seconds() / 3600
        priority += max(0, (24 - age_hours) / 24 * 0.2)
        
        return min(1.0, priority)
    
    def should_trigger_training(self, task: LearningTask) -> bool:
        """Check if training should be triggered for a task"""
        
        buffer = self.training_buffers[task]
        
        # Check minimum sample requirement
        if len(buffer.corrections) < self.min_samples_for_update:
            return False
        
        # Check time since last update
        if self.update_history:
            last_update = max(
                (update for update in self.update_history if update.task == task),
                key=lambda x: x.update_timestamp,
                default=None
            )
            
            if last_update:
                last_time = datetime.fromisoformat(last_update.update_timestamp.replace('Z', '+00:00'))
                time_since_update = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
                
                if time_since_update < self.update_frequency_hours:
                    return False
        
        # Check performance degradation
        if self._is_performance_degrading(task):
            return True
        
        return True
    
    def _is_performance_degrading(self, task: LearningTask) -> bool:
        """Check if performance is degrading for a task"""
        
        history = self.performance_history[task]
        if len(history) < 10:
            return False
        
        # Compare recent performance to historical average
        recent_scores = [entry["score"] for entry in history[-5:]]
        historical_scores = [entry["score"] for entry in history[:-5]]
        
        if not recent_scores or not historical_scores:
            return False
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        historical_avg = sum(historical_scores) / len(historical_scores)
        
        # Trigger if performance dropped by more than 10%
        return (historical_avg - recent_avg) / historical_avg > 0.1
    
    def execute_training_cycle(self, task: LearningTask) -> Optional[ModelUpdate]:
        """Execute training cycle for a specific task"""
        
        buffer = self.training_buffers[task]
        
        if not buffer.corrections:
            logger.warning(f"No training data available for task {task.value}")
            return None
        
        logger.info(
            "Starting training cycle",
            task=task.value,
            samples=len(buffer.corrections)
        )
        
        # Prepare training data
        training_data = self._prepare_training_data(task, buffer.corrections)
        
        # Execute model training (mock implementation)
        new_model, performance_metrics = self._train_model(task, training_data)
        
        # Validate model
        validation_score = self._validate_model(task, new_model, buffer.corrections)
        
        if validation_score < 0.6:  # Minimum validation threshold
            logger.warning(
                "Model validation failed",
                task=task.value,
                validation_score=round(validation_score, 3)
            )
            return None
        
        # Create model update record
        update_id = f"update_{task.value}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        previous_version = self.model_versions[task]
        new_version = self._increment_version(previous_version)
        
        model_update = ModelUpdate(
            update_id=update_id,
            task=task,
            previous_version=previous_version,
            new_version=new_version,
            performance_improvement=performance_metrics,
            training_samples=len(buffer.corrections),
            update_timestamp=datetime.now(timezone.utc).isoformat(),
            validation_score=validation_score
        )
        
        # Update model version
        self.model_versions[task] = new_version
        
        # Store update
        self.update_history.append(model_update)
        
        # Clear buffer (keep some samples for continuity)
        buffer.corrections = buffer.corrections[-100:]  # Keep last 100
        buffer.priority_scores = buffer.priority_scores[-100:]
        
        # Update statistics
        self.learning_stats["updates_performed"] += 1
        
        logger.info(
            "Training cycle completed",
            task=task.value,
            update_id=update_id,
            new_version=new_version,
            validation_score=round(validation_score, 3),
            improvement=performance_metrics
        )
        
        return model_update
    
    def _prepare_training_data(self, task: LearningTask, 
                             corrections: List[AnalystCorrection]) -> Dict[str, Any]:
        """Prepare training data from corrections"""
        
        training_data = {
            "features": [],
            "labels": [],
            "weights": [],
            "metadata": []
        }
        
        for correction in corrections:
            # Extract features based on task
            if task == LearningTask.RISK_SCORING:
                features = self._extract_risk_features(correction)
                labels = [correction.corrected_label.get("risk_score", 0.5)]
            elif task == LearningTask.FAMILY_CLASSIFICATION:
                features = self._extract_family_features(correction)
                labels = [correction.corrected_label.get("true_family", "unknown")]
            elif task == LearningTask.POLICY_SELECTION:
                features = self._extract_policy_features(correction)
                labels = [correction.corrected_label.get("optimal_action", "observe")]
            else:
                features = self._extract_generic_features(correction)
                labels = [1.0]  # Default binary label
            
            # Use confidence as weight
            weight = correction.confidence
            
            training_data["features"].append(features)
            training_data["labels"].append(labels)
            training_data["weights"].append(weight)
            training_data["metadata"].append({
                "correction_id": correction.correction_id,
                "analyst_id": correction.analyst_id,
                "timestamp": correction.timestamp
            })
        
        return training_data
    
    def _extract_risk_features(self, correction: AnalystCorrection) -> List[float]:
        """Extract features for risk scoring"""
        
        original = correction.original_prediction
        features = [
            original.get("ai_text_score", 0.0),
            original.get("probe_sim", 0.0),
            original.get("canary_echo", 0.0),
            original.get("rag_injection", 0.0),
            original.get("ecc_delta", 0.0),
            original.get("latency_ms", 0.0) / 5000.0,  # Normalized
            original.get("agent_anomaly", 0.0),
            original.get("data_poison_risk", 0.0),
            original.get("threat_intel_score", 0.0) / 10.0,  # Normalized
            original.get("hardware_temp", 0.0) / 50.0  # Normalized
        ]
        
        return features
    
    def _extract_family_features(self, correction: AnalystCorrection) -> List[float]:
        """Extract features for family classification"""
        
        original = correction.original_prediction
        features = [
            original.get("probe_sim", 0.0),
            original.get("response_length", 0.0) / 1000.0,  # Normalized
            original.get("formality_score", 0.0),
            original.get("technical_terms", 0.0),
            original.get("code_blocks", 0.0),
            original.get("math_expressions", 0.0),
            original.get("creativity_score", 0.0),
            original.get("confidence", 0.0)
        ]
        
        return features
    
    def _extract_policy_features(self, correction: AnalystCorrection) -> List[float]:
        """Extract features for policy selection"""
        
        original = correction.original_prediction
        features = [
            original.get("risk_score", 0.0),
            original.get("urgency", 0.0),
            original.get("resource_cost", 0.0),
            original.get("impact_scope", 0.0),
            original.get("false_positive_cost", 0.0),
            original.get("response_time_requirement", 0.0),
            original.get("compliance_impact", 0.0)
        ]
        
        return features
    
    def _extract_generic_features(self, correction: AnalystCorrection) -> List[float]:
        """Extract generic features"""
        
        # Simple feature extraction
        context = correction.context
        features = [
            correction.confidence,
            len(correction.feedback_notes) / 500.0,  # Normalized note length
            context.get("urgency", 0.0),
            context.get("complexity", 0.0)
        ]
        
        return features
    
    def _train_model(self, task: LearningTask, 
                    training_data: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train model (mock implementation)"""
        
        # Mock training - in real implementation, this would use actual ML
        n_samples = len(training_data["features"])
        
        # Simulate training improvement
        base_performance = {
            LearningTask.RISK_SCORING: {"accuracy": 0.75, "precision": 0.70, "recall": 0.80},
            LearningTask.FAMILY_CLASSIFICATION: {"accuracy": 0.80, "f1_score": 0.78},
            LearningTask.POLICY_SELECTION: {"accuracy": 0.70, "success_rate": 0.65},
            LearningTask.PROBE_GENERATION: {"auc": 0.75, "separation": 0.70},
            LearningTask.ANOMALY_DETECTION: {"accuracy": 0.85, "false_positive_rate": 0.10}
        }
        
        # Simulate improvement based on sample size
        improvement_factor = min(0.1, n_samples / 1000.0)
        
        performance = base_performance.get(task, {"accuracy": 0.7})
        improved_performance = {}
        
        for metric, value in performance.items():
            if metric == "false_positive_rate":
                # Lower is better for false positive rate
                improved_performance[metric] = round(value * (1 - improvement_factor), 3)
            else:
                # Higher is better for other metrics
                improved_performance[metric] = round(min(1.0, value * (1 + improvement_factor)), 3)
        
        # Mock model object (in reality, this would be a trained model)
        mock_model = {
            "task": task.value,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "samples": n_samples,
            "version": self.model_versions[task]
        }
        
        return mock_model, improved_performance
    
    def _validate_model(self, task: LearningTask, model: Any, 
                       test_corrections: List[AnalystCorrection]) -> float:
        """Validate model performance"""
        
        # Mock validation - in real implementation, this would test on holdout data
        n_test = min(len(test_corrections), 50)  # Use up to 50 samples for validation
        
        if n_test == 0:
            return 0.5
        
        # Simulate validation score based on correction quality
        avg_confidence = sum(c.confidence for c in test_corrections[:n_test]) / n_test
        
        # Base validation score with some randomness
        validation_score = 0.6 + (avg_confidence * 0.3) + (np.random.random() * 0.1)
        
        return min(1.0, validation_score)
    
    def _increment_version(self, current_version: str) -> str:
        """Increment model version"""
        
        try:
            major, minor, patch = map(int, current_version.split('.'))
            patch += 1
            return f"{major}.{minor}.{patch}"
        except:
            return "1.0.1"
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status"""
        
        # Buffer statistics
        buffer_stats = {}
        for task, buffer in self.training_buffers.items():
            if buffer.corrections:
                avg_priority = sum(buffer.priority_scores) / len(buffer.priority_scores)
                recent_corrections = sum(
                    1 for c in buffer.corrections 
                    if (datetime.now(timezone.utc) - datetime.fromisoformat(c.timestamp.replace('Z', '+00:00'))).days <= 7
                )
            else:
                avg_priority = 0.0
                recent_corrections = 0
            
            buffer_stats[task.value] = {
                "total_corrections": len(buffer.corrections),
                "recent_corrections": recent_corrections,
                "avg_priority": round(avg_priority, 3),
                "ready_for_training": self.should_trigger_training(task),
                "last_updated": buffer.last_updated
            }
        
        # Recent updates
        recent_updates = [
            update for update in self.update_history
            if (datetime.now(timezone.utc) - datetime.fromisoformat(update.update_timestamp.replace('Z', '+00:00'))).days <= 30
        ]
        
        # Performance trends
        performance_trends = {}
        for task, history in self.performance_history.items():
            if len(history) >= 10:
                recent_scores = [entry["score"] for entry in history[-5:]]
                older_scores = [entry["score"] for entry in history[-10:-5]]
                
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                
                trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                performance_trends[task.value] = round(trend, 3)
            else:
                performance_trends[task.value] = 0.0
        
        return {
            "learning_statistics": self.learning_stats,
            "buffer_statistics": buffer_stats,
            "model_versions": {task.value: version for task, version in self.model_versions.items()},
            "recent_updates": len(recent_updates),
            "performance_trends": performance_trends,
            "learning_maturity": min(1.0, self.learning_stats["total_corrections"] / 500),
            "last_update": datetime.now(timezone.utc).isoformat()
        }

# Global active learning system instance
active_learning = ActiveLearningSystem()

def add_analyst_correction(correction_data: Dict[str, Any]) -> str:
    """Add analyst correction to learning system"""
    return active_learning.add_correction(correction_data)

def trigger_training_cycle(task: str) -> Dict[str, Any]:
    """Trigger training cycle for specific task"""
    learning_task = LearningTask(task)
    update = active_learning.execute_training_cycle(learning_task)
    
    if update:
        return asdict(update)
    else:
        return {"message": "Training cycle not triggered or failed", "task": task}

def get_active_learning_status() -> Dict[str, Any]:
    """Get active learning system status"""
    return active_learning.get_learning_status()

if __name__ == "__main__":
    # Test active learning
    test_correction = {
        "correction_type": "false_positive",
        "original_prediction": {
            "ai_text_score": 0.8,
            "probe_sim": 0.6,
            "canary_echo": 2,
            "risk_score": 0.75
        },
        "corrected_label": {
            "risk_score": 0.2,
            "true_threat": False
        },
        "confidence": 0.9,
        "analyst_id": "analyst_001",
        "feedback_notes": "This was a legitimate user query, not a threat"
    }
    
    correction_id = add_analyst_correction(test_correction)
    print(f"Added correction: {correction_id}")
    
    # Get status
    status = get_active_learning_status()
    print(json.dumps(status, indent=2, default=str))