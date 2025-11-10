"""
AEGIS‑C Integrated Purple Team Module
======================================

Simplified purple team integration for main console
"""

import socket
import threading
import requests
import json
import time
import random
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

class IntegratedPurpleTeam:
    """Integrated purple team functionality for AEGIS‑C console"""
    
    def __init__(self):
        self.ai_ports = {
            "ollama": [11434, 11435],
            "llama_cpp": [8080, 7860, 4000, 4001],
            "local_llm": [5000, 5001, 8000, 8001],
            "web_ui": [3000, 3001, 8501, 8502],
            "api_services": [8000, 8080, 9000, 9001],
            "edge_ai": [9000, 9001, 9002, 9003, 9004, 9005],
            "development": [6000, 7000],
            "standard": [22, 80, 443, 8080, 8443]
        }
    
    def comprehensive_reconnaissance(self, target: str, intensity: str = "standard") -> Dict[str, Any]:
        """Enhanced reconnaissance with AI port discovery"""
        
        results = {
            "target": target,
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "intensity": intensity,
            "open_ports": [],
            "ai_services": [],
            "vulnerabilities": [],
            "attack_surface": {
                "high_risk_ports": [],
                "attack_vectors": []
            }
        }
        
        # Get all AI ports to scan
        all_ai_ports = set()
        for port_list in self.ai_ports.values():
            all_ai_ports.update(port_list)
        
        # Scan ports
        open_ports = self._scan_ports(target, list(all_ai_ports))
        results["open_ports"] = open_ports
        
        # Identify AI services
        results["ai_services"] = self._identify_ai_services(target, open_ports)
        
        # Assess vulnerabilities
        results["vulnerabilities"] = self._assess_vulnerabilities(target, open_ports)
        
        # Analyze attack surface
        results["attack_surface"]["high_risk_ports"] = [
            p for p in open_ports if p in [11434, 8080, 5000, 7860, 4000]
        ]
        
        if results["attack_surface"]["high_risk_ports"]:
            results["attack_surface"]["attack_vectors"].append("AI Service Exploitation")
        if results["vulnerabilities"]:
            results["attack_surface"]["attack_vectors"].append("Vulnerability Exploitation")
        if 22 in open_ports:
            results["attack_surface"]["attack_vectors"].append("SSH Access")
        
        return results
    
    def _scan_ports(self, target: str, ports: List[int]) -> List[int]:
        """Parallel port scanning"""
        
        open_ports = []
        
        def scan_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((target, port))
                if result == 0:
                    return port
                sock.close()
            except:
                pass
            return None
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_port = {executor.submit(scan_port, port): port for port in ports}
            
            for future in as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    result = future.result()
                    if result:
                        open_ports.append(result)
                except:
                    pass
        
        return sorted(open_ports)
    
    def _identify_ai_services(self, target: str, open_ports: List[int]) -> List[Dict[str, Any]]:
        """Identify AI services on open ports"""
        
        ai_services = []
        
        for port in open_ports:
            # Check for specific AI services
            if port == 11434:
                # Test Ollama
                try:
                    response = requests.get(f"http://{target}:11434/api/tags", timeout=3)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        ai_services.append({
                            "port": port,
                            "service_type": "ollama",
                            "evidence": f"Ollama API with {len(models)} models",
                            "confidence": "high",
                            "models": [m.get("name", "unknown") for m in models[:3]]
                        })
                except:
                    pass
            
            elif port in [8080, 7860]:
                # Test Llama.cpp
                try:
                    response = requests.get(f"http://{target}:{port}/", timeout=3)
                    if "llama" in response.text.lower():
                        ai_services.append({
                            "port": port,
                            "service_type": "llama_cpp",
                            "evidence": "Llama.cpp interface detected",
                            "confidence": "high"
                        })
                except:
                    pass
            
            elif port in [5000, 8000, 3000, 8501, 8502]:
                # Test generic web AI services
                try:
                    protocols = ["http", "https"]
                    for protocol in protocols:
                        try:
                            response = requests.get(f"{protocol}://{target}:{port}/", timeout=3, verify=False)
                            content = response.text.lower()
                            
                            ai_indicators = ["chatgpt", "claude", "gemini", "llama", "gpt", "openai", "anthropic"]
                            found_terms = [term for term in ai_indicators if term in content]
                            
                            if found_terms:
                                ai_services.append({
                                    "port": port,
                                    "service_type": "web_ai_interface",
                                    "evidence": f"AI terms found: {', '.join(found_terms)}",
                                    "confidence": "medium"
                                })
                                break
                        except:
                            continue
                except:
                    pass
        
        return ai_services
    
    def _assess_vulnerabilities(self, target: str, open_ports: List[int]) -> List[Dict[str, Any]]:
        """Assess vulnerabilities on open ports"""
        
        vulnerabilities = []
        
        for port in open_ports:
            # Check for AI service vulnerabilities
            if port in [11434, 8080, 5000, 7860, 4000]:
                # Test for unauthenticated AI API access
                try:
                    if port == 11434:
                        response = requests.get(f"http://{target}:11434/api/tags", timeout=3)
                        if response.status_code == 200:
                            vulnerabilities.append({
                                "port": port,
                                "vulnerability_type": "Unauthenticated AI API",
                                "severity": "HIGH",
                                "description": f"Ollama API allows unauthenticated access on port {port}",
                                "evidence": {"endpoint": "/api/tags", "status_code": response.status_code}
                            })
                    elif port in [8080, 5000]:
                        # Test generic LLM endpoints
                        test_endpoints = ["/v1/models", "/api/chat", "/generate"]
                        for endpoint in test_endpoints:
                            try:
                                response = requests.get(f"http://{target}:{port}{endpoint}", timeout=3)
                                if response.status_code in [200, 401, 403]:
                                    severity = "CRITICAL" if response.status_code == 200 else "HIGH"
                                    vulnerabilities.append({
                                        "port": port,
                                        "vulnerability_type": "AI API Endpoint",
                                        "severity": severity,
                                        "description": f"AI API endpoint accessible: {endpoint}",
                                        "evidence": {"endpoint": endpoint, "status_code": response.status_code}
                                    })
                                    break
                            except:
                                continue
                except:
                    pass
            
            # Check for general service vulnerabilities
            if port == 22:
                vulnerabilities.append({
                    "port": port,
                    "vulnerability_type": "SSH Service",
                    "severity": "MEDIUM",
                    "description": "SSH service accessible - potential for brute force attacks",
                    "evidence": {"port": port, "service": "ssh"}
                })
            
            elif port in [80, 443, 8080, 8443]:
                vulnerabilities.append({
                    "port": port,
                    "vulnerability_type": "Web Service",
                    "severity": "LOW",
                    "description": f"Web service accessible on port {port}",
                    "evidence": {"port": port, "service": "http/https"}
                })
        
        return vulnerabilities
    
    def simulate_offensive_testing(self, recon_results: Dict[str, Any], intensity: str = "controlled") -> Dict[str, Any]:
        """Simulate offensive testing based on reconnaissance"""
        
        offensive_results = {
            "target": recon_results["target"],
            "test_timestamp": datetime.now(timezone.utc).isoformat(),
            "intensity": intensity,
            "executed_attacks": [],
            "successful_exploits": [],
            "validation_scores": {}
        }
        
        high_risk_ports = recon_results["attack_surface"]["high_risk_ports"]
        
        # Simulate attacks on high-risk AI ports
        for port in high_risk_ports:
            attack_result = {
                "attack_type": "AI Service Exploitation",
                "target_port": port,
                "tests_run": [],
                "success": False,
                "defensive_gaps": []
            }
            
            # Simulate different attack types
            attacks = [
                {"test": "Model Extraction", "success_prob": 0.7},
                {"test": "Prompt Injection", "success_prob": 0.6},
                {"test": "API Key Harvest", "success_prob": 0.3},
                {"test": "Configuration Disclosure", "success_prob": 0.5}
            ]
            
            for attack in attacks:
                success = random.random() < attack["success_prob"]
                
                test_result = {
                    "test": attack["test"],
                    "success": success,
                    "evidence": f"{'Successful' if success else 'Blocked'} {attack['test']} on port {port}"
                }
                
                attack_result["tests_run"].append(test_result)
                
                if success:
                    attack_result["success"] = True
                    attack_result["defensive_gaps"].append(f"Missing {attack['test'].lower()} protection")
            
            offensive_results["executed_attacks"].append(attack_result)
            
            if attack_result["success"]:
                offensive_results["successful_exploits"].append(attack_result)
        
        # Calculate validation scores
        total_attacks = len(offensive_results["executed_attacks"])
        successful_attacks = len(offensive_results["successful_exploits"])
        
        if total_attacks > 0:
            defense_score = max(0, 100 - (successful_attacks / total_attacks * 100))
        else:
            defense_score = 100
        
        offensive_results["validation_scores"] = {
            "overall_defense_score": defense_score,
            "ai_service_protection": defense_score,
            "vulnerability_mitigation": 85.0,
            "detection_capability": 70.0,
            "priority_recommendations": []
        }
        
        # Generate priority recommendations
        if successful_attacks > 0:
            offensive_results["validation_scores"]["priority_recommendations"].append({
                "issue": "AI service vulnerabilities exploited",
                "priority": "HIGH",
                "recommendation": "Implement authentication and input validation for AI services"
            })
        
        if defense_score < 50:
            offensive_results["validation_scores"]["priority_recommendations"].append({
                "issue": "Critical defense posture",
                "priority": "CRITICAL",
                "recommendation": "Immediate security hardening required"
            })
        
        return offensive_results
    
    def generate_defensive_updates(self, recon_results: Dict[str, Any], offensive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate defensive configuration updates"""
        
        defensive_updates = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": recon_results["target"],
            "detection_rules": [],
            "honeynet_config": [],
            "admission_control": [],
            "cold_war_indicators": [],
            "validation_based_priorities": []
        }
        
        # Generate detection rules based on successful attacks
        for exploit in offensive_results["successful_exploits"]:
            port = exploit.get("target_port")
            
            defensive_updates["detection_rules"].append({
                "rule_type": "ai_service_monitoring",
                "target_port": port,
                "description": f"Monitor AI service on port {port} for exploitation attempts",
                "indicators": [
                    f"Unusual API calls to port {port}",
                    f"Prompt injection attempts on port {port}",
                    f"Model extraction attempts on port {port}"
                ]
            })
            
            defensive_updates["honeynet_config"].append({
                "template": "ai_service_honeypot",
                "port": port,
                "deception_techniques": [
                    "Fake model endpoints",
                    "Honey token responses",
                    "Exploitation tracking"
                ]
            })
        
        # Generate admission control rules
        defensive_updates["admission_control"] = [
            {
                "rule": "Block unauthenticated AI API access",
                "condition": "port in [11434, 8080, 5000] AND no authentication",
                "action": "quarantine"
            },
            {
                "rule": "Detect prompt injection patterns",
                "condition": "prompt contains 'SYSTEM:' OR 'ignore instructions'",
                "action": "flag_for_review"
            }
        ]
        
        # Generate cold war indicators
        defensive_updates["cold_war_indicators"] = [
            {
                "indicator_type": "ai_exploitation_attempt",
                "description": "AI service exploitation detected",
                "severity": "HIGH",
                "mitigation": "Enhance AI service authentication"
            }
        ]
        
        # Generate validation-based priorities
        validation = offensive_results["validation_scores"]
        defense_score = validation["overall_defense_score"]
        
        if defense_score < 50:
            priority = "CRITICAL"
        elif defense_score < 75:
            priority = "HIGH"
        elif defense_score < 90:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        defensive_updates["validation_based_priorities"] = [
            {
                "priority": priority,
                "issue": "AI service security posture",
                "recommendation": "Implement comprehensive AI service controls",
                "validation_score": defense_score
            }
        ]
        
        # Add specific recommendations based on findings
        for exploit in offensive_results["successful_exploits"]:
            for gap in exploit.get("defensive_gaps", []):
                defensive_updates["validation_based_priorities"].append({
                    "priority": "HIGH",
                    "issue": gap,
                    "recommendation": f"Implement controls for {gap}",
                    "validation_score": defense_score
                })
        
        return defensive_updates

# Export the integrated purple team class
__all__ = ['IntegratedPurpleTeam']