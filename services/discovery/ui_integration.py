"""
AEGIS‑C Discovery UI Integration
=================================

Flexible target discovery with web Gen AI assessment and console integration
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime, timezone
import subprocess
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FlexibleDiscoveryEngine:
    """Flexible discovery engine for multiple target types"""
    
    def __init__(self):
        self.target_types = {
            "edge_ai": {
                "description": "Edge AI device with common AI service ports",
                "port_ranges": ["8000-8010", "8080-8090", "9000-9010", "5000-5010"],
                "protocols": ["http", "https"],
                "endpoints": ["/v1/completions", "/v1/chat/completions", "/api/generate", "/api/models"]
            },
            "web_genai": {
                "description": "Web application with generative AI capabilities",
                "port_ranges": ["80", "443", "8080", "8443", "3000", "8000"],
                "protocols": ["http", "https"],
                "endpoints": ["/api/chat", "/api/complete", "/api/generate", "/api/ai", "/api/llm", "/chat", "/complete"],
                "js_patterns": ["openai", "anthropic", "google", "huggingface", "cohere", "api_key", "model"],
                "html_patterns": ["chatgpt", "claude", "gemini", "llm", "ai-chat", "ai-assistant"]
            },
            "api_service": {
                "description": "AI API service with REST endpoints",
                "port_ranges": ["8000-8050", "9000-9050", "5000-5050"],
                "protocols": ["http", "https"],
                "endpoints": ["/v1/", "/api/v1/", "/openai/", "/anthropic/", "/api/"]
            },
            "comprehensive": {
                "description": "Full-spectrum discovery across all AI service types",
                "port_ranges": ["1-1000", "8000-8050", "8080-8090", "9000-9050", "5000-5050"],
                "protocols": ["http", "https"],
                "endpoints": ["/api/", "/v1/", "/chat", "/generate", "/complete", "/models"]
            }
        }
        
    def discover_target(self, target_config):
        """Flexible discovery based on target type"""
        target_type = target_config.get("type", "comprehensive")
        target_host = target_config.get("host")
        custom_ports = target_config.get("custom_ports", "")
        
        # Get configuration for target type
        type_config = self.target_types.get(target_type, self.target_types["comprehensive"])
        
        # Use custom ports if provided
        if custom_ports:
            port_ranges = [custom_ports]
        else:
            port_ranges = type_config["port_ranges"]
        
        results = {
            "target_host": target_host,
            "target_type": target_type,
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "configuration": type_config,
            "findings": []
        }
        
        # Run discovery for each port range
        for port_range in port_ranges:
            try:
                # Call the existing discovery client
                from discovery_client import TargetDiscoveryClient
                client = TargetDiscoveryClient()
                
                scan_result = client.scan_target(target_host, port_range, timeout=3)
                
                # Filter results based on target type
                if target_type == "web_genai":
                    scan_result = self._enhance_web_genai_assessment(target_host, scan_result)
                
                results["findings"].append(scan_result)
                
            except Exception as e:
                st.error(f"Error scanning {port_range}: {e}")
        
        return results
    
    def _enhance_web_genai_assessment(self, target_host, scan_result):
        """Enhanced assessment for web Gen AI applications"""
        
        for port, service in scan_result["services"].items():
            if service.get("protocol") in ["http", "https"]:
                protocol = service["protocol"]
                base_url = f"{protocol}://{target_host}:{port}"
                
                # Check for web AI indicators
                web_indicators = self._check_web_ai_indicators(base_url)
                scan_result["ai_indicators"].extend(web_indicators)
                
                # Check for exposed API keys
                api_key_findings = self._check_exposed_api_keys(base_url)
                scan_result["vulnerabilities"].extend(api_key_findings)
                
                # Check for AI model identification
                model_findings = self._identify_ai_models(base_url)
                scan_result["ai_indicators"].extend(model_findings)
        
        return scan_result
    
    def _check_web_ai_indicators(self, base_url):
        """Check web application for AI service indicators"""
        indicators = []
        
        try:
            # Get main page
            response = requests.get(base_url, timeout=5, verify=False)
            content = response.text.lower()
            
            # Check for AI-related HTML patterns
            ai_patterns = ["chatgpt", "claude", "gemini", "openai", "anthropic", "cohere", "huggingface", "llm", "gpt"]
            found_patterns = [pattern for pattern in ai_patterns if pattern in content]
            
            if found_patterns:
                indicators.append({
                    "ai_type": "web_application",
                    "evidence": f"AI indicators found in page content: {', '.join(found_patterns)}",
                    "confidence": "medium"
                })
            
            # Check for AI-related JavaScript files
            if "application/javascript" in response.headers.get("Content-Type", ""):
                js_indicators = self._analyze_javascript_ai(content)
                indicators.extend(js_indicators)
                
        except Exception as e:
            pass
        
        return indicators
    
    def _analyze_javascript_ai(self, js_content):
        """Analyze JavaScript for AI service usage"""
        indicators = []
        
        ai_js_patterns = {
            "openai": ["openai", "gpt-", "chatgpt", "api.openai.com"],
            "anthropic": ["anthropic", "claude", "api.anthropic.com"],
            "google": ["google", "gemini", "generativelanguage.googleapis.com"],
            "cohere": ["cohere", "api.cohere.ai"],
            "huggingface": ["huggingface", "api-inference.huggingface.co"]
        }
        
        for ai_type, patterns in ai_js_patterns.items():
            if any(pattern in js_content.lower() for pattern in patterns):
                indicators.append({
                    "ai_type": f"javascript_{ai_type}",
                    "evidence": f"{ai_type} AI service detected in JavaScript",
                    "confidence": "high"
                })
        
        return indicators
    
    def _check_exposed_api_keys(self, base_url):
        """Check for exposed API keys in web application"""
        vulnerabilities = []
        
        # Common API key patterns
        api_key_patterns = [
            "sk-",  # OpenAI
            "sk-ant-",  # Anthropic
            "AIza",  # Google
            "hf_",  # HuggingFace
            "xoxb-",  # Slack (often used with AI bots)
        ]
        
        try:
            response = requests.get(base_url, timeout=5, verify=False)
            content = response.text
            
            for pattern in api_key_patterns:
                if pattern in content:
                    vulnerabilities.append({
                        "target_host": base_url.split("//")[1].split(":")[0],
                        "target_port": base_url.split(":")[-1].split("/")[0] if ":" in base_url else "443",
                        "vulnerability_type": "Exposed API Key",
                        "severity": "HIGH",
                        "description": f"Potential exposed API key with pattern: {pattern}",
                        "evidence": {"pattern": pattern, "url": base_url},
                        "discovered_at": datetime.now(timezone.utc).isoformat(),
                        "discovery_method": "content_analysis"
                    })
        
        except Exception:
            pass
        
        return vulnerabilities
    
    def _identify_ai_models(self, base_url):
        """Identify specific AI models used by web application"""
        indicators = []
        
        # Try common model identification endpoints
        model_endpoints = ["/api/models", "/v1/models", "/models", "/api/info"]
        
        for endpoint in model_endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=3, verify=False)
                if response.status_code == 200:
                    try:
                        model_data = response.json()
                        if "data" in model_data or "models" in model_data:
                            models = model_data.get("data", model_data.get("models", []))
                            if models:
                                model_names = [m.get("id", m.get("name", "unknown")) for m in models[:3]]
                                indicators.append({
                                    "ai_type": "identified_models",
                                    "evidence": f"Models identified: {', '.join(model_names)}",
                                    "confidence": "high",
                                    "models": model_names
                                })
                    except:
                        pass
        
            except Exception:
                pass
        
        return indicators

def discovery_ui():
    """Streamlit UI for flexible target discovery"""
    
    st.set_page_config(page_title="AEGIS‑C Discovery", layout="wide")
    st.title("🎯 AEGIS‑C Flexible Target Discovery")
    
    st.markdown("---")
    
    # Target Configuration Section
    st.subheader("🎯 Target Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_host = st.text_input(
            "Target Host/IP",
            placeholder="192.168.1.209 or example.com",
            help="Enter the target host or IP address to assess"
        )
    
    with col2:
        target_type = st.selectbox(
            "Target Type",
            options=list(FlexibleDiscoveryEngine().target_types.keys()),
            help="Select the type of target for optimized discovery"
        )
    
    # Show target type description
    engine = FlexibleDiscoveryEngine()
    st.info(engine.target_types[target_type]["description"])
    
    # Advanced Configuration
    with st.expander("🔧 Advanced Configuration"):
        custom_ports = st.text_input(
            "Custom Port Range",
            placeholder="8000-8010 or 80,443,8080",
            help="Override default port ranges with custom configuration"
        )
        
        timeout = st.slider("Scan Timeout (seconds)", 1, 10, 3)
        max_concurrent = st.slider("Max Concurrent Scans", 1, 20, 5)
    
    # Discovery Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔍 Start Discovery", type="primary"):
            if not target_host:
                st.error("Please enter a target host")
            else:
                # Store configuration in session state
                st.session_state.discovery_config = {
                    "host": target_host,
                    "type": target_type,
                    "custom_ports": custom_ports,
                    "timeout": timeout,
                    "max_concurrent": max_concurrent
                }
                st.session_state.discovery_running = True
                st.session_state.discovery_results = None
    
    with col2:
        if st.button("📊 View Previous Results"):
            st.session_state.show_results = True
    
    with col3:
        if st.button("🛡️ Generate Defensive Rules"):
            if st.session_state.get("discovery_results"):
                st.session_state.generate_rules = True
            else:
                st.warning("Run discovery first to generate rules")
    
    # Progress Section
    if st.session_state.get("discovery_running"):
        st.markdown("---")
        st.subheader("🔄 Discovery Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run discovery
        try:
            config = st.session_state.discovery_config
            status_text.text("Initializing discovery engine...")
            progress_bar.progress(10)
            
            results = engine.discover_target(config)
            progress_bar.progress(100)
            status_text.text("Discovery complete!")
            
            # Store results
            st.session_state.discovery_results = results
            st.session_state.discovery_running = False
            
            st.success("Discovery completed successfully!")
            
        except Exception as e:
            st.error(f"Discovery failed: {e}")
            st.session_state.discovery_running = False
    
    # Results Section
    if st.session_state.get("discovery_results"):
        st.markdown("---")
        st.subheader("📊 Discovery Results")
        
        results = st.session_state.discovery_results
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_ports = set()
        total_ai_indicators = []
        total_vulnerabilities = []
        
        for finding in results["findings"]:
            total_ports.update(finding.get("open_ports", []))
            total_ai_indicators.extend(finding.get("ai_indicators", []))
            total_vulnerabilities.extend(finding.get("vulnerabilities", []))
        
        with col1:
            st.metric("Open Ports", len(total_ports))
        
        with col2:
            st.metric("AI Indicators", len(total_ai_indicators))
        
        with col3:
            st.metric("Vulnerabilities", len(total_vulnerabilities))
        
        with col4:
            critical_vulns = len([v for v in total_vulnerabilities if v.get("severity") == "CRITICAL"])
            st.metric("Critical Issues", critical_vulns)
        
        # Detailed Findings
        tab1, tab2, tab3 = st.tabs(["🚨 Vulnerabilities", "🤖 AI Indicators", "🔧 Services"])
        
        with tab1:
            if total_vulnerabilities:
                for vuln in total_vulnerabilities:
                    severity_color = {
                        "CRITICAL": "🔴",
                        "HIGH": "🟠", 
                        "MEDIUM": "🟡",
                        "LOW": "🟢"
                    }
                    
                    st.markdown(f"""
                    **{severity_color.get(vuln.get('severity', 'LOW'), '🟢')} {vuln.get('vulnerability_type', 'Unknown')}**
                    - **Target**: {vuln.get('target_host', 'Unknown')}:{vuln.get('target_port', 'Unknown')}
                    - **Severity**: {vuln.get('severity', 'Unknown')}
                    - **Description**: {vuln.get('description', 'No description')}
                    - **Evidence**: {vuln.get('evidence', {})}
                    """)
                    st.markdown("---")
            else:
                st.success("No vulnerabilities found!")
        
        with tab2:
            if total_ai_indicators:
                for indicator in total_ai_indicators:
                    st.markdown(f"""
                    **🤖 {indicator.get('ai_type', 'Unknown')}**
                    - **Port**: {indicator.get('port', 'Unknown')}
                    - **Evidence**: {indicator.get('evidence', 'No evidence')}
                    - **Confidence**: {indicator.get('confidence', 'Unknown')}
                    """)
                    st.markdown("---")
            else:
                st.info("No AI indicators detected")
        
        with tab3:
            if total_ports:
                st.write("**Open Ports:**")
                st.write(", ".join(map(str, sorted(total_ports))))
                
                # Service details
                for finding in results["findings"]:
                    services = finding.get("services", {})
                    if services:
                        st.write("**Service Details:**")
                        for port, service in services.items():
                            st.markdown(f"""
                            **Port {port}**
                            - Protocol: {service.get('protocol', 'Unknown')}
                            - Server: {service.get('server', 'Unknown')}
                            - Status: {service.get('status_code', 'Unknown')}
                            """)
            else:
                st.info("No services detected")
    
    # Defensive Rules Generation
    if st.session_state.get("generate_rules") and st.session_state.get("discovery_results"):
        st.markdown("---")
        st.subheader("🛡️ Defensive Integration Rules")
        
        results = st.session_state.discovery_results
        
        # Generate rules based on findings
        total_ports = set()
        total_ai_indicators = []
        total_vulnerabilities = []
        
        for finding in results["findings"]:
            total_ports.update(finding.get("open_ports", []))
            total_ai_indicators.extend(finding.get("ai_indicators", []))
            total_vulnerabilities.extend(finding.get("vulnerabilities", []))
        
        # Detection Rules
        st.markdown("### 🔍 Detection Rules")
        
        if total_ports:
            st.code(f"""
# Add to AEGIS‑C Detector Configuration
DETECTION_RULES = {{
    "target_{results['target_host'].replace('.', '_')}": {{
        "ports": {sorted(total_ports)},
        "protocols": {list(set([s.get('protocol', 'unknown') for f in results['findings'] for s in f.get('services', {}).values()]))},
        "monitoring": "continuous"
    }}
}}
""")
        
        # Honeynet Configuration
        st.markdown("### 🍯 Honeynet Configuration")
        
        if total_ports:
            st.code(f"""
# Add to Honeynet Service Configuration
HONEYPOT_TARGETS = {{
    "target_{results['target_host'].replace('.', '_')}": {{
        "ports": {sorted(total_ports)},
        "templates": ["ai_api", "web_app", "ssh_service"],
        "canary_tokens": True
    }}
}}
""")
        
        # Admission Control Rules
        st.markdown("### 🛡️ Admission Control Rules")
        
        if total_vulnerabilities:
            vuln_rules = []
            for vuln in total_vulnerabilities:
                if vuln.get('severity') in ['CRITICAL', 'HIGH']:
                    vuln_rules.append(f"    # Block {vuln.get('vulnerability_type')} on port {vuln.get('target_port')}")
            
            st.code("\n".join(vuln_rules))
        
        # Cold War Defense Indicators
        st.markdown("### ❄️ Cold War Defense Indicators")
        
        if total_ai_indicators:
            st.code(f"""
# Add to Cold War Defense Configuration
AI_INDICATORS = {{
    "target_{results['target_host'].replace('.', '_')}": {json.dumps(total_ai_indicators, indent=6)}
}}
""")
        
        # Export functionality
        st.markdown("### 💾 Export Configuration")
        
        if st.button("📥 Export All Rules"):
            export_data = {
                "target": results["target_host"],
                "timestamp": results["scan_timestamp"],
                "detection_rules": {
                    "ports": list(total_ports),
                    "protocols": list(set([s.get('protocol', 'unknown') for f in results['findings'] for s in f.get('services', {}).values()]))
                },
                "honeynet_config": {
                    "ports": list(total_ports),
                    "templates": ["ai_api", "web_app", "ssh_service"]
                },
                "vulnerability_rules": total_vulnerabilities,
                "ai_indicators": total_ai_indicators
            }
            
            st.download_button(
                label="📥 Download Configuration",
                data=json.dumps(export_data, indent=2),
                file_name=f"aegis_c_defensive_rules_{results['target_host'].replace('.', '_')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    discovery_ui()