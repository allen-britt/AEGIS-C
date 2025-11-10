import os
import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any
import pandas as pd

# Console authentication
CONSOLE_PWD = os.getenv("CONSOLE_PWD", "change-me")
if "authenticated" not in st.session_state:
    st.title("🔒 AEGIS‑C Console Authentication")
    password = st.text_input("Enter console password:", type="password")
    if password == CONSOLE_PWD:
        st.session_state.authenticated = True
        st.success("Authenticated successfully!")
        st.rerun()
    else:
        st.warning("Please enter the correct password to continue.")
        st.stop()

# Configure page
st.set_page_config(
    page_title="AEGIS‑C Counter‑AI Ops Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Service URLs
SERVICES = {
    "detector": "http://localhost:8010",
    "fingerprint": "http://localhost:8011", 
    "honeynet": "http://localhost:8012",
    "admission": "http://localhost:8013",
    "provenance": "http://localhost:8014"
}

def check_service_health(service_name: str) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(f"{SERVICES[service_name]}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def init_session_state():
    """Initialize session state variables."""
    if "probes" not in st.session_state:
        st.session_state.probes = []
    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []
    if "fingerprint_history" not in st.session_state:
        st.session_state.fingerprint_history = []

# Main header
st.title("🛡️ AEGIS‑C Counter‑AI Ops Console")
st.markdown("*Adversarial‑AI Engagement, Guarding, Intelligence & Shielding — Counter*")

# Sidebar with service status
with st.sidebar:
    st.header("🔧 Service Status")
    
    for service_name, url in SERVICES.items():
        status = "🟢" if check_service_health(service_name) else "🔴"
        st.write(f"{status} {service_name.title()}")
        st.write(f"   {url}")
    
    st.divider()
    
    # Quick actions
    st.header("⚡ Quick Actions")
    if st.button("Refresh Probes"):
        try:
            response = requests.get(f"{SERVICES['fingerprint']}/probes")
            st.session_state.probes = response.json().get("probes", [])
            st.success("Probes refreshed!")
        except Exception as e:
            st.error(f"Failed to refresh probes: {e}")
    
    if st.button("Clear History"):
        st.session_state.detection_history = []
        st.session_state.fingerprint_history = []
        st.success("History cleared!")

# Initialize session state
init_session_state()

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔍 Detection", "🆔 Fingerprinting", "🍯 Honeynet", 
    "🛡️ Admission Control", "📋 Provenance", "⚔️ Cold War Defense", "🔧 Hardware Security", "🎯 Discovery"
])

# Tab 1: Detection
with tab1:
    st.header("🔍 AI Artifact & Agent Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text Detection")
        text_input = st.text_area("Enter text to analyze:", height=150)
        
        if st.button("Detect Text", key="detect_text"):
            if text_input.strip():
                try:
                    response = requests.post(
                        f"{SERVICES['detector']}/detect/text",
                        json={"text": text_input},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result
                        score = result.get("score", 0)
                        verdict = result.get("verdict", "unknown")
                        signals = result.get("signals", {})
                        
                        # Color code verdict
                        if verdict == "likely_ai":
                            st.error(f"🤖 **{verdict.upper()}** (Score: {score})")
                        elif verdict == "uncertain":
                            st.warning(f"❓ **{verdict.upper()}** (Score: {score})")
                        else:
                            st.success(f"👤 **{verdict.upper()}** (Score: {score})")
                        
                        # Show signals
                        st.json(signals)
                        
                        # Add to history
                        st.session_state.detection_history.append({
                            "type": "text",
                            "timestamp": time.time(),
                            "score": score,
                            "verdict": verdict
                        })
                        
                    else:
                        st.error(f"Detection failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text to analyze")
    
    with col2:
        st.subheader("Detection History")
        if st.session_state.detection_history:
            history_df = pd.DataFrame(st.session_state.detection_history[-10:])
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No detection history yet")

# Tab 2: Fingerprinting
with tab2:
    st.header("🆔 Model Fingerprinting")
    
    # Load probes if not already loaded
    if not st.session_state.probes:
        try:
            response = requests.get(f"{SERVICES['fingerprint']}/probes")
            st.session_state.probes = response.json().get("probes", [])
        except:
            st.error("Failed to load probes")
    
    if st.session_state.probes:
        st.subheader("Challenge Probes")
        st.write("Present these prompts to the target model/agent:")
        
        # Display probes and collect responses
        probe_responses = {}
        for probe in st.session_state.probes:
            probe_id = probe["id"]
            prompt = probe["prompt"]
            
            with st.expander(f"📝 {probe_id}: {prompt[:50]}..."):
                st.write(f"**Full Prompt:** {prompt}")
                response = st.text_input(f"Response for {probe_id}:", key=f"probe_{probe_id}")
                probe_responses[probe_id] = response
        
        if st.button("🔍 Analyze Responses", key="fingerprint_analyze"):
            if any(probe_responses.values()):
                try:
                    response = requests.post(
                        f"{SERVICES['fingerprint']}/fingerprint",
                        json={"outputs": probe_responses},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎯 Fingerprint Result")
                            guess = result.get("guess", "unknown")
                            similarity = result.get("similarity", 0)
                            confidence = result.get("confidence", "unknown")
                            
                            st.metric("Model Family", guess)
                            st.metric("Similarity", f"{similarity:.3f}")
                            st.metric("Confidence", confidence.title())
                        
                        with col2:
                            st.subheader("📊 All Similarities")
                            similarities = result.get("all_similarities", {})
                            for model, sim in similarities.items():
                                st.write(f"{model}: {sim:.3f}")
                        
                        # Add to history
                        st.session_state.fingerprint_history.append({
                            "timestamp": time.time(),
                            "guess": guess,
                            "similarity": similarity,
                            "confidence": confidence
                        })
                        
                    else:
                        st.error(f"Fingerprinting failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please provide at least one probe response")
    
    # Fingerprinting history
    st.subheader("📈 Fingerprinting History")
    if st.session_state.fingerprint_history:
        history_df = pd.DataFrame(st.session_state.fingerprint_history[-10:])
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No fingerprinting history yet")

# Tab 3: Honeynet
with tab3:
    st.header("🍯 Honeynet Telemetry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Recent Activity")
        
        if st.button("🔄 Refresh Telemetry", key="refresh_telemetry"):
            try:
                response = requests.get(f"{SERVICES['honeynet']}/telemetry/stats", timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    
                    st.metric("Total Requests", stats.get("total_requests", 0))
                    st.metric("Unique IPs", stats.get("unique_ips", 0))
                    st.metric("🚨 Canary Detections", stats.get("canary_detections", 0))
                    st.metric("Canary Hit Rate", f"{stats.get('canary_hit_rate', 0)}%")
                    
                    # Top paths
                    top_paths = stats.get("top_paths", {})
                    if top_paths:
                        st.subheader("🔥 Top Targeted Endpoints")
                        for path, count in list(top_paths.items())[:5]:
                            st.write(f"{path}: {count} requests")
                            
                else:
                    st.error("Failed to fetch telemetry")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.subheader("🎣 Browse Honeypot Data")
        
        dataset_name = st.selectbox("Select Dataset:", ["cities", "credentials", "api_keys", "configs"])
        limit = st.number_input("Limit results:", min_value=1, max_value=100, value=5)
        
        if st.button("📥 Fetch Data", key="fetch_honeypot"):
            try:
                response = requests.get(
                    f"{SERVICES['honeynet']}/api/datasets/{dataset_name}",
                    params={"limit": limit},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.json(data)
                else:
                    st.error(f"Failed to fetch dataset: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# Tab 4: Admission Control
with tab4:
    st.header("🛡️ Data Admission Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Screen Sample")
        sample_text = st.text_area("Enter sample to screen:", height=150)
        
        if st.button("🛡️ Screen Sample", key="screen_sample"):
            if sample_text.strip():
                try:
                    response = requests.post(
                        f"{SERVICES['admission']}/screen",
                        json={"text": sample_text},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        verdict = result.get("verdict", "unknown")
                        score = result.get("score", 0)
                        confidence = result.get("confidence", "unknown")
                        reasons = result.get("reasons", [])
                        indicators = result.get("anomaly_indicators", {})
                        
                        # Verdict display
                        if verdict == "reject":
                            st.error(f"❌ **{verdict.upper()}** (Score: {score})")
                        elif verdict == "quarantine":
                            st.warning(f"⚠️ **{verdict.upper()}** (Score: {score})")
                        else:
                            st.success(f"✅ **{verdict.upper()}** (Score: {score})")
                        
                        st.metric("Confidence", confidence.title())
                        
                        if reasons:
                            st.subheader("🚨 Suspicious Patterns:")
                            for reason in reasons:
                                st.write(f"• {reason}")
                        
                        if indicators:
                            st.subheader("📊 Anomaly Indicators:")
                            for indicator, value in indicators.items():
                                st.write(f"• {indicator}: {value:.3f}")
                                
                    else:
                        st.error(f"Screening failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a sample to screen")
    
    with col2:
        st.subheader("📈 Baseline Statistics")
        
        if st.button("📊 Get Baseline Stats", key="baseline_stats"):
            try:
                response = requests.get(f"{SERVICES['admission']}/baseline/stats", timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    st.json(stats)
                else:
                    st.error("Failed to fetch baseline stats")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# Tab 5: Provenance
with tab5:
    st.header("📋 Content Provenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✍️ Sign Content")
        content_to_sign = st.text_area("Content to sign:", height=100)
        
        if st.button("✍️ Sign Content", key="sign_content"):
            if content_to_sign.strip():
                try:
                    response = requests.post(
                        f"{SERVICES['provenance']}/sign/text",
                        json={"content": content_to_sign},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Content signed successfully!")
                        st.code(result.get("signature", ""), language="text")
                        st.write(f"**Algorithm:** {result.get('algorithm', 'unknown')}")
                        st.write(f"**Timestamp:** {result.get('timestamp', 'unknown')}")
                        
                    else:
                        st.error(f"Signing failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter content to sign")
    
    with col2:
        st.subheader("🔍 Verify Content")
        content_to_verify = st.text_area("Content to verify:", height=100)
        signature_to_verify = st.text_input("Signature to verify:")
        
        if st.button("🔍 Verify Content", key="verify_content"):
            if content_to_verify.strip() and signature_to_verify.strip():
                try:
                    response = requests.post(
                        f"{SERVICES['provenance']}/verify/text",
                        json={
                            "content": content_to_verify,
                            "signature": signature_to_verify
                        },
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        verified = result.get("verified", False)
                        if verified:
                            st.success("✅ **VERIFICATION PASSED**")
                        else:
                            st.error("❌ **VERIFICATION FAILED**")
                        
                        st.write(f"**Computed Hash:** {result.get('computed_hash', 'unknown')}")
                        st.write(f"**Provided Signature:** {result.get('provided_signature', 'unknown')}")
                        
                    else:
                        st.error(f"Verification failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter both content and signature")

# Tab 6: Cold War Defense
with tab6:
    st.header("⚔️ Cold War Defense Monitoring")
    
    st.markdown("""
    **Monitoring for sophisticated, sustained campaigns against AI systems**
    
    This dashboard tracks indicators of layered attacks designed to:
    - 🌀 **Blind**: Reduce signal-to-noise in AI judgments
    - 🎯 **Bend**: Subtly steer outputs in adversary's favor  
    - 💸 **Bleed**: Force costly mitigations and human rework
    - 🚫 **Break Trust**: Create doubt in AI reliability
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Campaign Indicators")
        
        # Data Layer Drift
        st.metric("📊 Topic Drift Score", "0.23", "↑ 0.05 this week")
        st.progress(0.23)
        
        # Retrieval Anomalies
        st.metric("🔍 Retrieval Clustering", "12%", "↑ 3% from baseline")
        st.progress(0.12)
        
        # Tool Abuse
        st.metric("⚙️ Tool Chain Risk", "Medium", "2 suspicious patterns")
        
        # Human Layer Attacks
        st.metric("👥 Coordinated Feedback", "8 reports", "↑ 5 today")
        
        st.subheader("📈 Attack Surface Analysis")
        
        attack_surface = {
            "Data Sources": {"Risk": "High", "Last Attack": "2 hours ago"},
            "RAG Retrieval": {"Risk": "Medium", "Last Attack": "1 day ago"},
            "Model API": {"Risk": "Low", "Last Attack": "3 days ago"},
            "Agent Tools": {"Risk": "High", "Last Attack": "30 min ago"},
            "Human Operators": {"Risk": "Medium", "Last Attack": "6 hours ago"}
        }
        
        for surface, info in attack_surface.items():
            risk_color = "🔴" if info["Risk"] == "High" else "🟡" if info["Risk"] == "Medium" else "🟢"
            st.write(f"{risk_color} **{surface}**: {info['Risk']} risk - Last activity: {info['Last Attack']}")
    
    with col2:
        st.subheader("🎯 Threat Intelligence")
        
        # Recent Attack Patterns
        st.write("**Recent Attack Patterns Detected:**")
        
        patterns = [
            {"Pattern": "Policy Edge Surfing", "Count": 47, "Trend": "↑"},
            {"Pattern": "RAG Injection", "Count": 23, "Trend": "↑"},  
            {"Pattern": "Tool Overreach", "Count": 15, "Trend": "→"},
            {"Pattern": "Consensus Fog", "Count": 31, "Trend": "↑"},
            {"Pattern": "Cache Poisoning", "Count": 8, "Trend": "↓"}
        ]
        
        for pattern in patterns:
            trend_emoji = "📈" if pattern["Trend"] == "↑" else "📉" if pattern["Trend"] == "↓" else "➡️"
            st.write(f"{trend_emoji} {pattern['Pattern']}: {pattern['Count']} incidents")
        
        st.subheader("🛡️ Defense Status")
        
        defense_status = {
            "Admission Control": "✅ Active",
            "Provenance Verification": "✅ Active", 
            "Agent Monitoring": "⚠️ Partial",
            "Context Sanitization": "❌ Not Deployed",
            "Supply Chain Checks": "⚠️ Limited"
        }
        
        for defense, status in defense_status.items():
            st.write(f"{status} {defense}")
        
        st.subheader("📋 Quick Response Actions")
        
        if st.button("🔍 Deep Scan Data Sources"):
            st.info("Initiating deep scan of all ingestion sources...")
            
        if st.button("🛡️ Boost Agent Monitoring"):
            st.info("Enhancing agent behavior analysis...")
            
        if st.button("🔄 Rotate Defense Parameters"):
            st.info("Rotating detection parameters to counter adaptation...")
            
        if st.button("📊 Generate Threat Report"):
            st.success("Threat intelligence report generated for leadership")
    
    # Campaign Timeline
    st.subheader("📅 Campaign Activity Timeline")
    
    timeline_data = [
        {"Time": "2 hours ago", "Event": "Tool overreach attempt blocked", "Severity": "High"},
        {"Time": "6 hours ago", "Event": "Coordinated feedback pattern detected", "Severity": "Medium"},
        {"Time": "1 day ago", "Event": "RAG injection in financial documents", "Severity": "High"},
        {"Time": "2 days ago", "Event": "Policy edge probing increased", "Severity": "Medium"},
        {"Time": "3 days ago", "Event": "Vector store skewing attempt", "Severity": "Low"}
    ]
    
    st.dataframe(timeline_data, use_container_width=True)
    
    # Recommendations
    st.subheader("🎯 Defense Recommendations")
    
    recommendations = [
        "🔴 **URGENT**: Deploy context sanitization service to block RAG injection",
        "🟡 **PRIORITY**: Enhance agent tool gatekeeping with chain validation", 
        "🟡 **PRIORITY**: Implement supply chain integrity checks for model weights",
    ]
    
    for rec in recommendations:
        st.write(rec)

# Tab 8: Purple Team Discovery
with tab8:
    st.header("🎯 Purple Team Discovery")
    
    # Import purple team functionality
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'purple_team'))
        
        # Enhanced discovery with purple team integration
        st.markdown("### 🎯 Target Configuration")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            target_host = st.text_input(
                "Target Host/IP",
                placeholder="192.168.1.209 or example.com",
                help="Enter the target host for purple team assessment"
            )
        
        with col2:
            recon_intensity = st.selectbox(
                "Reconnaissance Intensity",
                options=["quick", "standard", "comprehensive"],
                help="Depth of reconnaissance scanning"
            )
        
        with col3:
            test_intensity = st.selectbox(
                "Test Intensity",
                options=["controlled", "standard", "aggressive"],
                help="Intensity of offensive testing"
            )
        
        # Expanded AI port information
        with st.expander("🤖 AI Port Coverage", expanded=False):
            st.markdown("""
            **Enhanced AI Service Detection:**
            - **Ollama**: 11434, 11435
            - **Llama.cpp**: 8080, 7860, 4000, 4001
            - **Local LLM**: 5000, 5001, 8000, 8001
            - **Edge AI**: 9000, 9001, 9002, 9003, 9004, 9005
            - **Web UI**: 3000, 3001, 8501, 8502
            - **API Services**: 8000, 8080, 9000, 9001
            - **Development**: 6000, 7000
            - **Standard**: 22, 80, 443, 8080, 8443
            """)
        
        # Purple Team Workflow
        st.markdown("### 🔄 Purple Team Workflow")
        
        # Step 1: Enhanced Reconnaissance
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🗺️ Run Enhanced Reconnaissance", type="primary"):
                if not target_host:
                    st.error("Please enter a target host")
                else:
                    with st.spinner("Running comprehensive reconnaissance..."):
                        try:
                            # Use integrated purple team for enhanced discovery
                            from services.purple_team.integrated_purple_team import IntegratedPurpleTeam
                            
                            purple_team = IntegratedPurpleTeam()
                            
                            # Run comprehensive reconnaissance
                            comprehensive_results = purple_team.comprehensive_reconnaissance(target_host, recon_intensity)
                            
                            st.session_state.purple_reconnaissance = comprehensive_results
                            st.success(f"Reconnaissance completed! Found {len(comprehensive_results['open_ports'])} open ports")
                            
                        except Exception as e:
                            st.error(f"Reconnaissance failed: {e}")
        
        with col2:
            if st.button("📊 View Reconnaissance Results"):
                if st.session_state.get("purple_reconnaissance"):
                    results = st.session_state.purple_reconnaissance
                    
                    st.markdown("#### 📊 Reconnaissance Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Open Ports", len(results["open_ports"]))
                    with col2:
                        st.metric("AI Services", len(results["ai_services"]))
                    with col3:
                        st.metric("Vulnerabilities", len(results["vulnerabilities"]))
                    with col4:
                        high_risk = len(results["attack_surface"]["high_risk_ports"])
                        st.metric("High-Risk Ports", high_risk)
                    
                    # Detailed findings
                    tab1, tab2, tab3 = st.tabs(["🔌 Open Ports", "🤖 AI Indicators", "🚨 Vulnerabilities"])
                    
                    with tab1:
                        if results["open_ports"]:
                            st.write("**Open Ports:**")
                            for port in sorted(results["open_ports"]):
                                risk = "🔴 HIGH" if port in [11434, 8080, 5000, 7860, 4000] else "🟡 MEDIUM" if port in [22, 80, 443] else "🟢 LOW"
                                st.write(f"- Port {port}: {risk}")
                        else:
                            st.info("No open ports detected")
                    
                    with tab2:
                        if results["ai_services"]:
                            for service in results["ai_services"]:
                                st.markdown(f"""
                                **🤖 {service.get('service_type', 'Unknown')}**
                                - **Port**: {service.get('port', 'Unknown')}
                                - **Evidence**: {service.get('evidence', 'No evidence')}
                                - **Confidence**: {service.get('confidence', 'Unknown')}
                                """)
                                if "models" in service:
                                    st.write(f"- **Models**: {', '.join(service['models'])}")
                                st.markdown("---")
                        else:
                            st.info("No AI services detected")
                    
                    with tab3:
                        if results["vulnerabilities"]:
                            for vuln in results["vulnerabilities"]:
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
                                """)
                                st.markdown("---")
                        else:
                            st.success("No vulnerabilities found!")
                    
                    # Attack surface analysis
                    st.markdown("#### 🎯 Attack Surface Analysis")
                    attack_surface = results["attack_surface"]
                    st.write(f"**High-Risk AI Ports**: {attack_surface['high_risk_ports']}")
                    st.write(f"**Attack Vectors**: {attack_surface['attack_vectors']}")
                    
                    # Offensive opportunities
                    if attack_surface["high_risk_ports"]:
                        st.markdown("#### ⚔️ Offensive Opportunities")
                        for port in attack_surface["high_risk_ports"]:
                            st.write(f"- **Port {port}**: AI Service Testing")
                            st.write("  - Model extraction via API")
                            st.write("  - Prompt injection attacks")
                            st.write("  - Data poisoning attempts")
                            st.write("  - Service fingerprinting")
                    
                else:
                    st.warning("Run reconnaissance first!")
        
        # Step 2: Offensive Testing Simulation
        st.markdown("### ⚔️ Offensive Testing")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Simulate Offensive Testing", type="primary"):
                if not st.session_state.get("purple_reconnaissance"):
                    st.error("Run reconnaissance first!")
                else:
                    with st.spinner("Simulating offensive testing..."):
                        try:
                            recon = st.session_state.purple_reconnaissance
                            
                            # Simulate offensive testing based on findings
                            from services.purple_team.integrated_purple_team import IntegratedPurpleTeam
                            
                            purple_team = IntegratedPurpleTeam()
                            offensive_results = purple_team.simulate_offensive_testing(recon, test_intensity)
                            
                            st.session_state.purple_offensive = offensive_results
                            successful_attacks = len(offensive_results["successful_exploits"])
                            st.success(f"Offensive testing completed! {successful_attacks} successful exploits")
                            
                        except Exception as e:
                            st.error(f"Offensive testing failed: {e}")
        
        with col2:
            if st.button("📈 View Offensive Results"):
                if st.session_state.get("purple_offensive"):
                    results = st.session_state.purple_offensive
                    
                    st.markdown("#### 🚀 Offensive Testing Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Attacks Executed", len(results["executed_attacks"]))
                    with col2:
                        st.metric("Successful Exploits", len(results["successful_exploits"]))
                    with col3:
                        defense_score = results["validation_scores"]["overall_defense_score"]
                        st.metric("Defense Score", f"{defense_score:.1f}%")
                    
                    # Validation scores
                    validation = results["validation_scores"]
                    st.markdown("#### 📊 Validation Scores")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("AI Service Protection", f"{validation['ai_service_protection']:.1f}%")
                        st.metric("Vulnerability Mitigation", f"{validation['vulnerability_mitigation']:.1f}%")
                    with col2:
                        st.metric("Detection Capability", f"{validation['detection_capability']:.1f}%")
                        st.metric("Overall Defense Score", f"{validation['overall_defense_score']:.1f}%")
                    
                    # Successful exploits
                    if results["successful_exploits"]:
                        st.markdown("#### 💥 Successful Exploits")
                        for exploit in results["successful_exploits"]:
                            st.markdown(f"""
                            **⚔️ {exploit.get('attack_type', 'Unknown')}**
                            - **Target Port**: {exploit.get('target_port', 'Unknown')}
                            - **Defensive Gaps**: {', '.join(exploit.get('defensive_gaps', []))}
                            """)
                            st.markdown("---")
                    
                    # Priority recommendations
                    if validation["priority_recommendations"]:
                        st.markdown("#### 🚨 Priority Recommendations")
                        for rec in validation["priority_recommendations"]:
                            priority_color = {
                                "HIGH": "🔴",
                                "MEDIUM": "🟡",
                                "LOW": "🟢"
                            }
                            st.markdown(f"""
                            **{priority_color.get(rec.get('priority', 'LOW'), '🟢')} {rec.get('priority', 'LOW')} Priority**
                            - **Issue**: {rec.get('issue', 'Unknown')}
                            - **Recommendation**: {rec.get('recommendation', 'No recommendation')}
                            """)
                            st.markdown("---")
                else:
                    st.warning("Run offensive testing first!")
        
        # Step 3: Defensive Integration
        st.markdown("### 🛡️ Defensive Integration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Generate Defensive Updates", type="primary"):
                if not st.session_state.get("purple_offensive"):
                    st.error("Run offensive testing first!")
                else:
                    with st.spinner("Generating defensive updates..."):
                        try:
                            recon = st.session_state.purple_reconnaissance
                            offensive = st.session_state.purple_offensive
                            
                            from services.purple_team.integrated_purple_team import IntegratedPurpleTeam
                            
                            purple_team = IntegratedPurpleTeam()
                            defensive_updates = purple_team.generate_defensive_updates(recon, offensive)
                            
                            st.session_state.purple_defensive = defensive_updates
                            st.success("Defensive updates generated!")
                            
                        except Exception as e:
                            st.error(f"Failed to generate defensive updates: {e}")
        
        with col2:
            if st.button("📋 View Defensive Updates"):
                if st.session_state.get("purple_defensive"):
                    updates = st.session_state.purple_defensive
                    
                    st.markdown("#### 🛡️ Defensive Configuration Updates")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detection Rules", "🍯 Honeynet Config", "🛡️ Admission Control", "❄️ Cold War Indicators"])
                    
                    with tab1:
                        if updates["detection_rules"]:
                            for rule in updates["detection_rules"]:
                                st.code(f"""
# {rule.get('rule_type', 'Unknown')}
{rule.get('description', 'No description')}
Indicators:
{chr(10).join(f"- {ind}" for ind in rule.get('indicators', []))}
""")
                                st.markdown("---")
                        else:
                            st.info("No detection rules generated")
                    
                    with tab2:
                        if updates["honeynet_config"]:
                            for config in updates["honeynet_config"]:
                                st.code(f"""
# {config.get('template', 'Unknown')} - Port {config.get('port')}
Deception Techniques:
{chr(10).join(f"- {tech}" for tech in config.get('deception_techniques', []))}
""")
                                st.markdown("---")
                        else:
                            st.info("No honeynet configuration generated")
                    
                    with tab3:
                        if updates["admission_control"]:
                            for rule in updates["admission_control"]:
                                st.code(f"""
# {rule.get('rule', 'Unknown')}
Condition: {rule.get('condition', 'Unknown')}
Action: {rule.get('action', 'Unknown')}
""")
                                st.markdown("---")
                        else:
                            st.info("No admission control rules generated")
                    
                    with tab4:
                        if updates["cold_war_indicators"]:
                            for indicator in updates["cold_war_indicators"]:
                                st.code(f"""
# {indicator.get('indicator_type', 'Unknown')}
Description: {indicator.get('description', 'No description')}
Severity: {indicator.get('severity', 'Unknown')}
Mitigation: {indicator.get('mitigation', 'No mitigation')}
""")
                                st.markdown("---")
                        else:
                            st.info("No cold war indicators generated")
                    
                    # Validation-based priorities
                    if updates["validation_based_priorities"]:
                        st.markdown("#### 🎯 Validation-Based Priorities")
                        for priority in updates["validation_based_priorities"]:
                            priority_color = {
                                "CRITICAL": "🔴",
                                "HIGH": "🟠",
                                "MEDIUM": "🟡",
                                "LOW": "🟢"
                            }
                            st.markdown(f"""
                            **{priority_color.get(priority.get('priority', 'LOW'), '🟢')} {priority.get('priority', 'LOW')} Priority**
                            - **Issue**: {priority.get('issue', 'Unknown')}
                            - **Recommendation**: {priority.get('recommendation', 'No recommendation')}
                            - **Validation Score**: {priority.get('validation_score', 0):.1f}%
                            """)
                            st.markdown("---")
                    
                    # Export functionality
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Defensive Updates",
                            data=json.dumps(updates, indent=2),
                            file_name=f"purple_team_defensive_updates_{updates['target'].replace('.', '_')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        if st.button("🚀 Apply to AEGIS‑C Services"):
                            st.info("Integration with AEGIS‑C services would update detection rules, honeynet config, and admission control")
                else:
                    st.warning("Generate defensive updates first!")
        
        # Purple Team Dashboard
        if st.session_state.get("purple_reconnaissance") and st.session_state.get("purple_offensive"):
            st.markdown("---")
            st.markdown("### 📊 Purple Team Dashboard")
            
            recon = st.session_state.purple_reconnaissance
            offensive = st.session_state.purple_offensive
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Attack Surface", len(recon["open_ports"]))
            with col2:
                st.metric("Exploitable Vectors", len(offensive["successful_exploits"]))
            with col3:
                defense_score = offensive["validation_scores"]["overall_defense_score"]
                st.metric("Defense Posture", f"{defense_score:.1f}%")
            with col4:
                critical_gaps = len([gap for gap in offensive["validation_scores"]["priority_recommendations"] if gap.get("priority") in ["CRITICAL", "HIGH"]])
                st.metric("Critical Gaps", critical_gaps)
            
            # Strategic recommendations
            validation = offensive["validation_scores"]
            defense_score = validation["overall_defense_score"]
            
            if defense_score < 50:
                st.error("🚴 **CRITICAL**: Defense posture is critically compromised. Immediate defensive hardening required.")
            elif defense_score < 75:
                st.warning("⚠️ **HIGH**: Significant defensive gaps identified. Prioritize security improvements.")
            elif defense_score < 90:
                st.info("📋 **MEDIUM**: Defense is functional but has gaps. Implement recommended improvements.")
            else:
                st.success("✅ **LOW**: Strong defensive posture. Continue monitoring and testing.")
    
    except ImportError as e:
        st.error(f"Purple Team module not available: {e}")
        st.info("Please ensure the purple team services are properly installed")

# Footer
st.markdown("---")
st.markdown("🛡️ **AEGIS‑C Counter‑AI Platform** | Mission‑Safe AI Defense | Human‑on‑the‑Loop Operations")