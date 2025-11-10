"""
AEGIS‑C Hardware Security Service
==================================

Dynamic AI-driven hardware protection for ML/AI infrastructure
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import psutil
import GPUtil
import platform
import subprocess
import json
import time
import random
from datetime import datetime, timezone
import logging
import asyncio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AEGIS‑C Hardware Security")

class HardwareProfile(BaseModel):
    """Hardware component profile"""
    component_type: str
    vendor: str
    model: str
    driver_version: str
    firmware_version: Optional[str] = None
    serial_number: Optional[str] = None
    capabilities: List[str]
    attack_surface: Dict[str, Any]
    trust_level: str  # TRUSTED, SUSPICIOUS, COMPROMISED

class HardwareAttack(BaseModel):
    """Hardware attack simulation"""
    target_component: str
    attack_type: str
    attack_vector: str
    intensity: float  # 0.0 to 1.0
    duration: int  # seconds

class HardwareDefense(BaseModel):
    """Hardware defense mechanism"""
    component_type: str
    defense_type: str
    parameters: Dict[str, Any]
    effectiveness: float  # 0.0 to 1.0

class HardwareSecurityMonitor:
    """Dynamic hardware security monitoring system"""
    
    def __init__(self):
        self.hardware_inventory = {}
        self.attack_patterns = {}
        self.defense_strategies = {}
        self.threat_intelligence = {}
        self.adaptation_history = []
        
    async def scan_hardware_inventory(self) -> Dict[str, HardwareProfile]:
        """Comprehensive hardware discovery and profiling"""
        profiles = {}
        
        # CPU Analysis
        cpu_info = {
            "component_type": "CPU",
            "vendor": platform.processor(),
            "model": platform.machine(),
            "driver_version": "N/A",
            "capabilities": self._get_cpu_capabilities(),
            "attack_surface": {
                "side_channels": ["timing", "power", "thermal"],
                "vulnerabilities": ["spectre", "meltdown", "rowhammer"],
                "exposure_level": self._assess_cpu_exposure()
            },
            "trust_level": "TRUSTED"
        }
        profiles["cpu"] = HardwareProfile(**cpu_info)
        
        # GPU Analysis
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info = {
                    "component_type": "GPU",
                    "vendor": "NVIDIA" if "nvidia" in gpu.name.lower() else "Unknown",
                    "model": gpu.name,
                    "driver_version": self._get_gpu_driver_version(),
                    "firmware_version": self._get_gpu_firmware_version(),
                    "capabilities": [
                        "cuda", "tensor_cores", "memory_management",
                        "parallel_processing", "ml_acceleration"
                    ],
                    "attack_surface": {
                        "side_channels": ["power_analysis", "timing", "thermal"],
                        "vulnerabilities": [
                            "gpu_memory_exhaustion", "firmware_backdoor",
                            "driver_compromise", "supply_chain"
                        ],
                        "exposure_level": self._assess_gpu_exposure(gpu)
                    },
                    "trust_level": self._assess_gpu_trust(gpu)
                }
                profiles[f"gpu_{i}"] = HardwareProfile(**gpu_info)
        except Exception as e:
            logger.warning(f"GPU scanning failed: {e}")
        
        # Memory Analysis
        memory = psutil.virtual_memory()
        mem_info = {
            "component_type": "MEMORY",
            "vendor": "Unknown",
            "model": f"{memory.total // (1024**3)}GB System",
            "driver_version": "N/A",
            "capabilities": ["data_storage", "memory_management", "cache"],
            "attack_surface": {
                "side_channels": ["rowhammer", "cold_boot", "dma"],
                "vulnerabilities": ["memory_exhaustion", "data_remnant", "bus_sniffing"],
                "exposure_level": self._assess_memory_exposure(memory)
            },
            "trust_level": "TRUSTED"
        }
        profiles["memory"] = HardwareProfile(**mem_info)
        
        # Network Interface Analysis
        network_interfaces = psutil.net_if_addrs()
        for iface_name, addresses in network_interfaces.items():
            if not iface_name.startswith("lo"):
                net_info = {
                    "component_type": "NETWORK",
                    "vendor": "Unknown",
                    "model": iface_name,
                    "driver_version": "N/A",
                    "capabilities": ["data_transmission", "packet_inspection", "routing"],
                    "attack_surface": {
                        "side_channels": ["packet_timing", "traffic_analysis"],
                        "vulnerabilities": ["nic_firmware", "driver_backdoor", "dma_attack"],
                        "exposure_level": 0.7
                    },
                    "trust_level": "TRUSTED"
                }
                profiles[f"net_{iface_name}"] = HardwareProfile(**net_info)
        
        self.hardware_inventory = profiles
        return profiles
    
    def _get_cpu_capabilities(self) -> List[str]:
        """Analyze CPU capabilities and features"""
        capabilities = ["general_computing"]
        
        # Check for virtualization support
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            if 'virtualization' in result.stdout.lower():
                capabilities.append("virtualization")
        except:
            pass
        
        # Check for security features
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'aes' in cpuinfo:
                    capabilities.append("hardware_encryption")
                if 'sgx' in cpuinfo:
                    capabilities.append("secure_enclaves")
        except:
            pass
        
        return capabilities
    
    def _get_gpu_driver_version(self) -> str:
        """Get GPU driver version"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "Unknown"
    
    def _get_gpu_firmware_version(self) -> str:
        """Get GPU firmware version"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=vbios_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "Unknown"
    
    def _assess_cpu_exposure(self) -> float:
        """Assess CPU attack surface exposure (0.0 to 1.0)"""
        exposure = 0.3  # Base exposure
        
        # Check for known vulnerabilities
        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True)
            cpu_info = result.stdout.lower()
            
            if 'intel' in cpu_info:
                exposure += 0.2  # Intel CPUs historically vulnerable
            if 'amd' in cpu_info:
                exposure += 0.15  # AMD vulnerabilities
                
        except:
            exposure += 0.1  # Unknown CPU = higher risk
        
        return min(1.0, exposure)
    
    def _assess_gpu_exposure(self, gpu) -> float:
        """Assess GPU attack surface exposure"""
        exposure = 0.4  # Base GPU exposure (higher than CPU)
        
        # Check GPU utilization (higher usage = more exposure)
        if gpu.load > 0.8:
            exposure += 0.2
        
        # Check memory usage
        if gpu.memoryUtil > 0.7:
            exposure += 0.1
        
        # Check temperature
        if gpu.temperature > 80:
            exposure += 0.1  # Thermal stress increases vulnerability
        
        return min(1.0, exposure)
    
    def _assess_gpu_trust(self, gpu) -> str:
        """Assess GPU trust level"""
        trust_score = 1.0
        
        # Check for suspicious behavior
        if gpu.load > 0.95:
            trust_score -= 0.3  # Unusual high utilization
        
        if gpu.memoryUtil > 0.9:
            trust_score -= 0.2  # Memory exhaustion attempt
        
        if gpu.temperature > 85:
            trust_score -= 0.2  # Potential thermal attack
        
        if trust_score >= 0.8:
            return "TRUSTED"
        elif trust_score >= 0.5:
            return "SUSPICIOUS"
        else:
            return "COMPROMISED"
    
    def _assess_memory_exposure(self, memory) -> float:
        """Assess memory attack surface exposure"""
        exposure = 0.2  # Base memory exposure
        
        # Check memory pressure
        if memory.percent > 90:
            exposure += 0.3
        
        # Check for swap usage (indicates memory pressure)
        swap = psutil.swap_memory()
        if swap.percent > 50:
            exposure += 0.2
        
        return min(1.0, exposure)
    
    async def simulate_hardware_attack(self, attack: HardwareAttack) -> Dict[str, Any]:
        """Simulate hardware attack for red team testing"""
        results = {
            "attack_id": f"hw_attack_{int(time.time())}",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "target": attack.target_component,
            "attack_type": attack.attack_type,
            "attack_vector": attack.attack_vector,
            "intensity": attack.intensity,
            "success": False,
            "defenses_triggered": [],
            "impact_assessment": {},
            "recommendations": []
        }
        
        try:
            if attack.target_component.startswith("gpu"):
                results.update(await self._simulate_gpu_attack(attack))
            elif attack.target_component == "cpu":
                results.update(await self._simulate_cpu_attack(attack))
            elif attack.target_component == "memory":
                results.update(await self._simulate_memory_attack(attack))
            elif attack.target_component.startswith("net"):
                results.update(await self._simulate_network_attack(attack))
            
            # Update threat intelligence
            await self._update_threat_intelligence(attack, results)
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Hardware attack simulation failed: {e}")
        
        results["end_time"] = datetime.now(timezone.utc).isoformat()
        return results
    
    async def _simulate_gpu_attack(self, attack: HardwareAttack) -> Dict[str, Any]:
        """Simulate GPU-specific attacks"""
        results = {"defenses_triggered": [], "impact_assessment": {}}
        
        if attack.attack_type == "memory_exhaustion":
            # Simulate GPU memory bomb
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    original_memory = gpu.memoryUsed
                    
                    # Simulate memory allocation attack
                    simulated_allocation = int(gpu.memoryTotal * attack.intensity * 0.8)
                    
                    results["impact_assessment"] = {
                        "memory_impact": simulated_allocation,
                        "original_memory": original_memory,
                        "attack_success": simulated_allocation > (gpu.memoryTotal * 0.7)
                    }
                    
                    if attack.intensity > 0.7:
                        results["defenses_triggered"].append("gpu_memory_monitor")
                        results["success"] = True
                        results["recommendations"].append("Implement GPU memory quotas")
                    
            except Exception as e:
                results["error"] = str(e)
        
        elif attack.attack_type == "thermal_throttling":
            # Simulate thermal attack
            results["impact_assessment"] = {
                "temperature_increase": attack.intensity * 30,  # Simulate temp rise
                "performance_degradation": attack.intensity * 0.5
            }
            
            if attack.intensity > 0.6:
                results["defenses_triggered"].append("thermal_monitor")
                results["success"] = True
                results["recommendations"].append("Enhance GPU cooling and monitoring")
        
        elif attack.attack_type == "side_channel":
            # Simulate side channel attack
            results["impact_assessment"] = {
                "data_leakage_risk": attack.intensity,
                "exfiltration_rate": attack.intensity * 1000  # bits/sec
            }
            
            if attack.intensity > 0.5:
                results["defenses_triggered"].append("side_channel_monitor")
                results["success"] = True
                results["recommendations"].append("Implement noise injection for side channel protection")
        
        return results
    
    async def _simulate_cpu_attack(self, attack: HardwareAttack) -> Dict[str, Any]:
        """Simulate CPU-specific attacks"""
        results = {"defenses_triggered": [], "impact_assessment": {}}
        
        if attack.attack_type == "spectre_meltdown":
            results["impact_assessment"] = {
                "speculative_execution_risk": attack.intensity,
                "data_exposure_potential": attack.intensity * 0.8
            }
            
            if attack.intensity > 0.4:
                results["defenses_triggered"].append("speculative_execution_mitigation")
                results["success"] = True
                results["recommendations"].append("Apply latest microcode updates")
        
        elif attack.attack_type == "rowhammer":
            results["impact_assessment"] = {
                "memory_corruption_risk": attack.intensity,
                "bit_flip_probability": attack.intensity * 0.001
            }
            
            if attack.intensity > 0.8:
                results["defenses_triggered"].append("memory_error_correction")
                results["success"] = True
                results["recommendations"].append("Enable ECC memory and monitoring")
        
        return results
    
    async def _simulate_memory_attack(self, attack: HardwareAttack) -> Dict[str, Any]:
        """Simulate memory-specific attacks"""
        results = {"defenses_triggered": [], "impact_assessment": {}}
        
        if attack.attack_type == "exhaustion":
            memory = psutil.virtual_memory()
            results["impact_assessment"] = {
                "memory_pressure": memory.percent + (attack.intensity * 20),
                "swap_usage_risk": attack.intensity * 0.7
            }
            
            if attack.intensity > 0.6:
                results["defenses_triggered"].append("memory_pressure_monitor")
                results["success"] = True
                results["recommendations"].append("Implement memory usage quotas and swap monitoring")
        
        return results
    
    async def _simulate_network_attack(self, attack: HardwareAttack) -> Dict[str, Any]:
        """Simulate network interface attacks"""
        results = {"defenses_triggered": [], "impact_assessment": {}}
        
        if attack.attack_type == "packet_injection":
            results["impact_assessment"] = {
                "network_congestion": attack.intensity * 0.8,
                "packet_loss_risk": attack.intensity * 0.1
            }
            
            if attack.intensity > 0.5:
                results["defenses_triggered"].append("packet_inspection")
                results["success"] = True
                results["recommendations"].append("Enhance network interface monitoring")
        
        return results
    
    async def _update_threat_intelligence(self, attack: HardwareAttack, results: Dict[str, Any]):
        """Update threat intelligence based on attack results"""
        threat_key = f"{attack.target_component}_{attack.attack_type}"
        
        if threat_key not in self.threat_intelligence:
            self.threat_intelligence[threat_key] = {
                "attack_count": 0,
                "success_rate": 0.0,
                "last_seen": None,
                "recommended_defenses": []
            }
        
        intel = self.threat_intelligence[threat_key]
        intel["attack_count"] += 1
        intel["last_seen"] = datetime.now(timezone.utc).isoformat()
        
        # Update success rate
        if results.get("success", False):
            intel["success_rate"] = (intel["success_rate"] * (intel["attack_count"] - 1) + 1.0) / intel["attack_count"]
        else:
            intel["success_rate"] = (intel["success_rate"] * (intel["attack_count"] - 1)) / intel["attack_count"]
        
        # Update recommended defenses
        if results.get("recommendations"):
            intel["recommended_defenses"].extend(results["recommendations"])
            intel["recommended_defenses"] = list(set(intel["recommended_defenses"]))  # Remove duplicates
    
    async def generate_adaptive_defenses(self) -> List[HardwareDefense]:
        """Generate adaptive defense strategies based on threat intelligence"""
        defenses = []
        
        for threat_key, intel in self.threat_intelligence.items():
            if intel["success_rate"] > 0.5:  # High success rate = need better defenses
                component, attack_type = threat_key.split("_", 1)
                
                if component == "gpu":
                    if attack_type == "memory_exhaustion":
                        defenses.append(HardwareDefense(
                            component_type="GPU",
                            defense_type="memory_quota",
                            parameters={
                                "max_allocation_percent": 70,
                                "monitoring_interval": 1.0,
                                "auto_throttle": True
                            },
                            effectiveness=0.8
                        ))
                    elif attack_type == "thermal_throttling":
                        defenses.append(HardwareDefense(
                            component_type="GPU",
                            defense_type="thermal_protection",
                            parameters={
                                "max_temperature": 80,
                                "throttle_threshold": 75,
                                "emergency_shutdown": True
                            },
                            effectiveness=0.9
                        ))
                
                elif component == "cpu":
                    if attack_type == "spectre_meltdown":
                        defenses.append(HardwareDefense(
                            component_type="CPU",
                            defense_type="speculative_execution_control",
                            parameters={
                                "disable_speculative_execution": False,
                                "microcode_updates": True,
                                "branch_prediction_hardening": True
                            },
                            effectiveness=0.85
                        ))
        
        self.defense_strategies = {f"defense_{i}": defense for i, defense in enumerate(defenses)}
        return defenses
    
    async def get_hardware_security_status(self) -> Dict[str, Any]:
        """Get comprehensive hardware security status"""
        return {
            "inventory_count": len(self.hardware_inventory),
            "threat_intelligence_entries": len(self.threat_intelligence),
            "active_defenses": len(self.defense_strategies),
            "last_scan": datetime.now(timezone.utc).isoformat(),
            "overall_risk_level": self._calculate_overall_risk(),
            "components": {
                comp_id: {
                    "type": profile.component_type,
                    "trust_level": profile.trust_level,
                    "exposure_level": profile.attack_surface.get("exposure_level", 0)
                }
                for comp_id, profile in self.hardware_inventory.items()
            }
        }
    
    def _calculate_overall_risk(self) -> str:
        """Calculate overall hardware security risk level"""
        if not self.hardware_inventory:
            return "UNKNOWN"
        
        total_exposure = 0
        compromised_count = 0
        
        for profile in self.hardware_inventory.values():
            exposure = profile.attack_surface.get("exposure_level", 0)
            total_exposure += exposure
            
            if profile.trust_level == "COMPROMISED":
                compromised_count += 1
        
        avg_exposure = total_exposure / len(self.hardware_inventory)
        
        if compromised_count > 0 or avg_exposure > 0.8:
            return "CRITICAL"
        elif avg_exposure > 0.6:
            return "HIGH"
        elif avg_exposure > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

# Initialize the hardware security monitor
hardware_monitor = HardwareSecurityMonitor()

@app.post("/hardware/scan", response_model=Dict[str, HardwareProfile])
async def scan_hardware():
    """Scan and profile all hardware components"""
    logger.info("🔍 Starting comprehensive hardware scan...")
    profiles = await hardware_monitor.scan_hardware_inventory()
    logger.info(f"✅ Hardware scan complete: {len(profiles)} components identified")
    return profiles

@app.post("/hardware/attack", response_model=Dict[str, Any])
async def simulate_hardware_attack(attack: HardwareAttack):
    """Simulate hardware attack for security testing"""
    logger.info(f"🎯 Simulating {attack.attack_type} attack on {attack.target_component}")
    results = await hardware_monitor.simulate_hardware_attack(attack)
    logger.info(f"📊 Attack simulation complete: success={results.get('success', False)}")
    return results

@app.post("/hardware/defenses", response_model=List[HardwareDefense])
async def generate_adaptive_defenses():
    """Generate adaptive defense strategies"""
    logger.info("🛡️ Generating adaptive hardware defenses...")
    defenses = await hardware_monitor.generate_adaptive_defenses()
    logger.info(f"✅ Generated {len(defenses)} adaptive defenses")
    return defenses

@app.get("/hardware/status", response_model=Dict[str, Any])
async def get_hardware_security_status():
    """Get comprehensive hardware security status"""
    return await hardware_monitor.get_hardware_security_status()

@app.get("/hardware/threats", response_model=Dict[str, Any])
async def get_threat_intelligence():
    """Get hardware threat intelligence"""
    return {
        "threat_intelligence": hardware_monitor.threat_intelligence,
        "attack_patterns": hardware_monitor.attack_patterns,
        "adaptation_history": hardware_monitor.adaptation_history[-10:]  # Last 10 adaptations
    }

@app.post("/hardware/defend", response_model=Dict[str, Any])
async def activate_hardware_defense(defense_id: str):
    """Activate specific hardware defense"""
    if defense_id in hardware_monitor.defense_strategies:
        defense = hardware_monitor.defense_strategies[defense_id]
        logger.info(f"🛡️ Activating defense: {defense.defense_type} for {defense.component_type}")
        
        # Simulate defense activation
        return {
            "defense_activated": True,
            "defense_id": defense_id,
            "defense_type": defense.defense_type,
            "effectiveness": defense.effectiveness,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail=f"Defense {defense_id} not found")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "ok": True,
        "service": "hardware_security",
        "capabilities": [
            "hardware_profiling",
            "attack_simulation",
            "adaptive_defenses",
            "threat_intelligence"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "AEGIS‑C Hardware Security",
        "version": "1.0.0",
        "description": "Dynamic AI-driven hardware protection for ML/AI infrastructure",
        "endpoints": {
            "scan_hardware": "/hardware/scan",
            "simulate_attack": "/hardware/attack",
            "generate_defenses": "/hardware/defenses",
            "security_status": "/hardware/status",
            "threat_intel": "/hardware/threats",
            "activate_defense": "/hardware/defend",
            "health": "/health"
        }
    }