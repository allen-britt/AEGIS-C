"""
AEGIS‑C Hardware Sentinel Service
==================================

Hardware security sentinel with pynvml collector and anomaly scoring.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from service_guard import setup_service_guard, record_detection_score, logger, require_api_key

# Try to import pynvml, fallback to mock data if not available
try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, using mock hardware data")

# Service Configuration
SERVICE_NAME = "hardware"
SERVICE_PORT = int(os.getenv("HARDWARE_PORT", "8016"))
GPU_MONITORING_ENABLED = os.getenv("GPU_MONITORING_ENABLED", "true").lower() == "true"

# Initialize FastAPI App
app = FastAPI(title="AEGIS‑C Hardware Sentinel Service")

# Setup standardized guard
setup_service_guard(app, SERVICE_NAME)

# Data models
class GPUMetrics(BaseModel):
    gpu_id: int
    name: str
    temperature: float
    utilization: float
    memory_used: float
    memory_total: float
    power_usage: float
    ecc_errors: Optional[Dict[str, int]] = None
    timestamp: str

class AnomalyScore(BaseModel):
    severity: str  # low, medium, high, critical
    score: float
    anomalies: List[str]
    recommendations: List[str]

class HardwareStatus(BaseModel):
    timestamp: str
    gpu_count: int
    gpus: List[GPUMetrics]
    system_health: str
    anomaly_score: Optional[AnomalyScore] = None

class PolicyAction(BaseModel):
    action: str  # drain, reset, retest
    target_gpu: Optional[int] = None
    reason: str
    timestamp: str
    executed: bool = False

# Hardware monitoring state
HISTORY_BUFFER: List[HardwareStatus] = []
POLICY_ACTIONS: List[PolicyAction] = []
ANOMALY_THRESHOLDS = {
    "temperature_high": 85.0,
    "temperature_critical": 95.0,
    "utilization_high": 95.0,
    "memory_usage_high": 0.95,
    "ecc_error_threshold": 10
}

def get_gpu_metrics() -> List[GPUMetrics]:
    """Collect GPU metrics using pynvml or mock data"""
    gpus = []
    timestamp = datetime.now(timezone.utc).isoformat()
    
    if PYNVML_AVAILABLE and GPU_MONITORING_ENABLED:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic GPU info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0.0
                
                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    utilization = 0.0
                
                # Memory
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used = mem_info.used / (1024**3)  # GB
                    memory_total = mem_info.total / (1024**3)  # GB
                except:
                    memory_used = 0.0
                    memory_total = 0.0
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                except:
                    power = 0.0
                
                # ECC errors
                ecc_errors = None
                try:
                    ecc_counts = pynvml.nvmlDeviceGetTotalEccErrors(handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC)
                    ecc_errors = {
                        "uncorrected_volatile": ecc_counts
                    }
                except:
                    ecc_errors = None
                
                gpu_metrics = GPUMetrics(
                    gpu_id=i,
                    name=name,
                    temperature=float(temp),
                    utilization=float(utilization),
                    memory_used=memory_used,
                    memory_total=memory_total,
                    power_usage=float(power),
                    ecc_errors=ecc_errors,
                    timestamp=timestamp
                )
                
                gpus.append(gpu_metrics)
                
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
            # Fallback to mock data
            gpus = get_mock_gpu_metrics()
    else:
        # Mock data for testing
        gpus = get_mock_gpu_metrics()
    
    return gpus

def get_mock_gpu_metrics() -> List[GPUMetrics]:
    """Generate mock GPU metrics for testing"""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return [
        GPUMetrics(
            gpu_id=0,
            name="NVIDIA RTX 4090 (Mock)",
            temperature=65.0 + random.uniform(-10, 20),
            utilization=random.uniform(20, 90),
            memory_used=20.5 + random.uniform(-5, 10),
            memory_total=24.0,
            power_usage=350.0 + random.uniform(-100, 150),
            ecc_errors={"uncorrected_volatile": random.randint(0, 5)},
            timestamp=timestamp
        ),
        GPUMetrics(
            gpu_id=1,
            name="NVIDIA A100 (Mock)",
            temperature=55.0 + random.uniform(-10, 25),
            utilization=random.uniform(10, 80),
            memory_used=35.0 + random.uniform(-10, 15),
            memory_total=40.0,
            power_usage=400.0 + random.uniform(-150, 200),
            ecc_errors={"uncorrected_volatile": random.randint(0, 3)},
            timestamp=timestamp
        )
    ]

def detect_anomalies(gpus: List[GPUMetrics]) -> Optional[AnomalyScore]:
    """Detect hardware anomalies"""
    anomalies = []
    recommendations = []
    max_score = 0.0
    
    for gpu in gpus:
        # Temperature anomalies
        if gpu.temperature > ANOMALY_THRESHOLDS["temperature_critical"]:
            anomalies.append(f"GPU {gpu.gpu_id} critical temperature: {gpu.temperature}°C")
            recommendations.append(f"CRITICAL: Drain GPU {gpu.gpu_id} immediately")
            max_score = max(max_score, 0.9)
        elif gpu.temperature > ANOMALY_THRESHOLDS["temperature_high"]:
            anomalies.append(f"GPU {gpu.gpu_id} high temperature: {gpu.temperature}°C")
            recommendations.append(f"Monitor GPU {gpu.gpu_id} temperature closely")
            max_score = max(max_score, 0.6)
        
        # Utilization anomalies
        if gpu.utilization > ANOMALY_THRESHOLDS["utilization_high"]:
            anomalies.append(f"GPU {gpu.gpu_id} extremely high utilization: {gpu.utilization}%")
            recommendations.append(f"Check for runaway processes on GPU {gpu.gpu_id}")
            max_score = max(max_score, 0.5)
        
        # Memory anomalies
        memory_usage_ratio = gpu.memory_used / max(gpu.memory_total, 1)
        if memory_usage_ratio > ANOMALY_THRESHOLDS["memory_usage_high"]:
            anomalies.append(f"GPU {gpu.gpu_id} high memory usage: {memory_usage_ratio:.1%}")
            recommendations.append(f"Monitor memory usage on GPU {gpu.gpu_id}")
            max_score = max(max_score, 0.4)
        
        # ECC errors
        if gpu.ecc_errors:
            ecc_count = gpu.ecc_errors.get("uncorrected_volatile", 0)
            if ecc_count > ANOMALY_THRESHOLDS["ecc_error_threshold"]:
                anomalies.append(f"GPU {gpu.gpu_id} high ECC errors: {ecc_count}")
                recommendations.append(f"CRITICAL: Reset GPU {gpu.gpu_id} due to ECC errors")
                max_score = max(max_score, 0.8)
    
    if not anomalies:
        return None
    
    # Determine severity
    if max_score >= 0.8:
        severity = "critical"
    elif max_score >= 0.6:
        severity = "high"
    elif max_score >= 0.4:
        severity = "medium"
    else:
        severity = "low"
    
    # Record metrics
    record_detection_score(max_score, f"hardware_anomaly_{severity}")
    
    return AnomalyScore(
        severity=severity,
        score=round(max_score, 3),
        anomalies=anomalies,
        recommendations=recommendations
    )

def execute_policy_action(action: str, target_gpu: Optional[int] = None, reason: str = "") -> bool:
    """Execute hardware policy actions (stub implementation)"""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    policy_action = PolicyAction(
        action=action,
        target_gpu=target_gpu,
        reason=reason,
        timestamp=timestamp,
        executed=False
    )
    
    # In production, these would be actual hardware actions
    if action == "drain":
        logger.warning(f"Draining GPU {target_gpu}: {reason}")
        policy_action.executed = True
    elif action == "reset":
        logger.warning(f"Resetting GPU {target_gpu}: {reason}")
        policy_action.executed = True
    elif action == "retest":
        logger.info(f"Retesting GPU {target_gpu}: {reason}")
        policy_action.executed = True
    
    POLICY_ACTIONS.append(policy_action)
    return policy_action.executed

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    current_status = get_hardware_status()
    return {
        "service": "AEGIS‑C Hardware Sentinel",
        "version": "1.0.0",
        "description": "Hardware security sentinel with GPU monitoring",
        "endpoints": {
            "status": "/status",
            "metrics": "/metrics",
            "anomalies": "/anomalies",
            "policy": "/policy",
            "health": "/health",
            "metrics": "/metrics",
            "secure_ping": "/secure/ping"
        },
        "gpu_count": current_status.gpu_count,
        "system_health": current_status.system_health,
        "pynvml_available": PYNVML_AVAILABLE
    }

@app.get("/status")
async def get_hardware_status():
    """Get current hardware status"""
    try:
        gpus = get_gpu_metrics()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Detect anomalies
        anomaly_score = detect_anomalies(gpus)
        
        # Determine overall system health
        if anomaly_score and anomaly_score.severity == "critical":
            system_health = "critical"
        elif anomaly_score and anomaly_score.severity == "high":
            system_health = "degraded"
        elif anomaly_score:
            system_health = "warning"
        else:
            system_health = "healthy"
        
        status = HardwareStatus(
            timestamp=timestamp,
            gpu_count=len(gpus),
            gpus=gpus,
            system_health=system_health,
            anomaly_score=anomaly_score
        )
        
        # Add to history (keep last 100 entries)
        HISTORY_BUFFER.append(status)
        if len(HISTORY_BUFFER) > 100:
            HISTORY_BUFFER.pop(0)
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get hardware status: {e}")
        raise

@app.get("/metrics")
async def get_hardware_metrics():
    """Get detailed hardware metrics"""
    status = await get_hardware_status()
    
    # Calculate additional metrics
    total_memory = sum(gpu.memory_total for gpu in status.gpus)
    used_memory = sum(gpu.memory_used for gpu in status.gpus)
    avg_temperature = sum(gpu.temperature for gpu in status.gpus) / len(status.gpus) if status.gpus else 0
    avg_utilization = sum(gpu.utilization for gpu in status.gpus) / len(status.gpus) if status.gpus else 0
    
    return {
        "timestamp": status.timestamp,
        "gpu_count": status.gpu_count,
        "system_health": status.system_health,
        "summary": {
            "total_memory_gb": total_memory,
            "used_memory_gb": used_memory,
            "memory_utilization": used_memory / max(total_memory, 1),
            "average_temperature": round(avg_temperature, 1),
            "average_utilization": round(avg_utilization, 1)
        },
        "gpus": [
            {
                "gpu_id": gpu.gpu_id,
                "name": gpu.name,
                "temperature_c": gpu.temperature,
                "utilization_percent": gpu.utilization,
                "memory_utilization": gpu.memory_used / max(gpu.memory_total, 1),
                "power_usage_watts": gpu.power_usage,
                "ecc_errors": gpu.ecc_errors
            }
            for gpu in status.gpus
        ],
        "anomaly_score": status.anomaly_score.dict() if status.anomaly_score else None
    }

@app.get("/anomalies")
async def get_anomaly_history(limit: int = 50):
    """Get recent anomaly detections"""
    anomalous_statuses = [
        status for status in HISTORY_BUFFER 
        if status.anomaly_score is not None
    ]
    
    anomalous_statuses.sort(key=lambda x: x.timestamp, reverse=True)
    
    return {
        "total_anomalies": len(anomalous_statuses),
        "recent_anomalies": [
            {
                "timestamp": status.timestamp,
                "system_health": status.system_health,
                "anomaly_score": status.anomaly_score.dict()
            }
            for status in anomalous_statuses[:limit]
        ]
    }

@app.post("/policy/execute")
async def execute_hardware_policy(action: str, target_gpu: Optional[int] = None, reason: str = ""):
    """Execute hardware policy actions"""
    try:
        valid_actions = ["drain", "reset", "retest"]
        if action not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")
        
        success = execute_policy_action(action, target_gpu, reason)
        
        logger.info(f"Policy action executed", action=action, target_gpu=target_gpu, success=success)
        
        return {
            "success": success,
            "action": action,
            "target_gpu": target_gpu,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Policy execution failed: {e}")
        raise

@app.get("/policy/history")
async def get_policy_history(limit: int = 50):
    """Get policy action history"""
    POLICY_ACTIONS.sort(key=lambda x: x.timestamp, reverse=True)
    
    return {
        "total_actions": len(POLICY_ACTIONS),
        "recent_actions": [
            {
                "action": action.action,
                "target_gpu": action.target_gpu,
                "reason": action.reason,
                "timestamp": action.timestamp,
                "executed": action.executed
            }
            for action in POLICY_ACTIONS[:limit]
        ]
    }

# Protected endpoints
@app.get("/secure/status", dependencies=[Depends(require_api_key)])
async def secure_get_status():
    """Protected hardware status endpoint"""
    return await get_hardware_status()

@app.post("/secure/policy/execute", dependencies=[Depends(require_api_key)])
async def secure_execute_policy(action: str, target_gpu: Optional[int] = None, reason: str = ""):
    """Protected policy execution endpoint"""
    return await execute_hardware_policy(action, target_gpu, reason)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)