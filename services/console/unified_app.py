#!/usr/bin/env python3
"""
AEGIS-C Unified Web Application
================================

Single interface for all AEGIS-C functionality:
- Brain Intelligence Demo
- Service Management & Monitoring  
- Detection & Analysis Tools
- Hardware Monitoring
- Purple Team Operations
- System Health Dashboard
"""

import streamlit as st
import requests
import json
import os
import time
import subprocess
import threading
from datetime import datetime
import pandas as pd

# Optional plotting imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Create dummy plotting functions
    class DummyPlotting:
        @staticmethod
        def line(*args, **kwargs):
            return None
        @staticmethod
        def pie(*args, **kwargs):
            return None
    
    px = DummyPlotting()

# Configuration
BRAIN_URL = os.getenv("BRAIN_URL", "http://localhost:8030")
SERVICES = {
    "brain": {"port": 8030, "name": "Brain Gateway", "icon": "🧠"},
    "detector": {"port": 8010, "name": "Detector", "icon": "🔍"},
    "fingerprint": {"port": 8011, "name": "Fingerprint", "icon": "👆"},
    "honeynet": {"port": 8012, "name": "Honeynet", "icon": "🎭"},
    "admission": {"port": 8013, "name": "Admission", "icon": "🛡️"},
    "provenance": {"port": 8014, "name": "Provenance", "icon": "🔐"},
    "coldwar": {"port": 8015, "name": "Cold War", "icon": "❄️"},
    "hardware": {"port": 8016, "name": "Hardware", "icon": "💻"},
    "discovery": {"port": 8017, "name": "Discovery", "icon": "🔎"},
    "intelligence": {"port": 8018, "name": "Intelligence", "icon": "📊"},
    "vuln_db": {"port": 8019, "name": "Vulnerability DB", "icon": "🗃️"}
}

def check_service_health(service_name, port):
    """Check if a service is healthy"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_service(service_name, port):
    """Start a service in background"""
    try:
        service_path = f"services/{service_name}/main.py"
        if os.path.exists(service_path):
            subprocess.Popen([
                "python", "-m", "uvicorn", "main:app", 
                f"--port={port}", "--host=localhost"
            ], cwd=f"services/{service_name}")
            return True
    except Exception as e:
        st.error(f"Failed to start {service_name}: {e}")
    return False

def assess_risk(signals_dict):
    """Assess risk using Brain Gateway"""
    try:
        signals = [{"name": k, "value": float(v)} for k, v in signals_dict.items()]
        payload = {
            "subject": "unified:demo",
            "kind": "artifact", 
            "signals": signals,
            "context": {"source": "unified_app"}
        }
        
        response = requests.post(f"{BRAIN_URL}/risk", json=payload, timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Risk assessment failed: {e}")
        return None

def decide_policy(risk_probability, options, subject_kind="artifact"):
    """Get policy decision from Brain Gateway"""
    try:
        payload = {
            "subject": "unified:demo",
            "kind": subject_kind,
            "risk": float(risk_probability),
            "options": options
        }
        
        response = requests.post(f"{BRAIN_URL}/policy", json=payload, timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Policy decision failed: {e}")
        return None

def detect_ai_text(text, case_id):
    """Detect AI-generated text"""
    try:
        payload = {"text": text, "case_id": case_id}
        response = requests.post("http://localhost:8010/detect", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"AI detection failed: {e}")
        return None

def get_hardware_status():
    """Get hardware monitoring status"""
    try:
        response = requests.get("http://localhost:8016/hardware/status", timeout=3)
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "Service unavailable", "gpu_count": 0}

def main():
    """Main unified application"""
    st.set_page_config(
        page_title="AEGIS-C Unified Platform",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .service-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .healthy { background-color: #d4edda; border-color: #c3e6cb; }
    .unhealthy { background-color: #f8d7da; border-color: #f5c6cb; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 12px 24px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🛡️ AEGIS-C Unified Platform</div>', unsafe_allow_html=True)
    st.markdown("Adaptive Counter-AI Intelligence Platform")
    
    # Sidebar Navigation
    st.sidebar.title("🎛️ Control Center")
    
    # Service Management Section
    st.sidebar.markdown("### 🚀 Service Management")
    
    # Auto-start services
    if st.sidebar.button("🔄 Auto-Start All Services", type="primary"):
        with st.sidebar:
            with st.spinner("Starting services..."):
                # Start brain first
                if not check_service_health("brain", 8030):
                    start_service("brain", 8030)
                    time.sleep(2)
                
                # Start other services
                for service_name, config in SERVICES.items():
                    if service_name != "brain":
                        if not check_service_health(service_name, config["port"]):
                            start_service(service_name, config["port"])
                            time.sleep(1)
                
                st.success("✅ Services started!")
                time.sleep(1)
                st.rerun()
    
    # Service Health Dashboard
    st.sidebar.markdown("### 📊 Service Health")
    
    health_data = []
    for service_name, config in SERVICES.items():
        is_healthy = check_service_health(service_name, config["port"])
        status_icon = "✅" if is_healthy else "❌"
        health_data.append({
            "Service": f"{config['icon']} {config['name']}",
            "Port": config["port"],
            "Status": status_icon,
            "Healthy": is_healthy
        })
    
    health_df = pd.DataFrame(health_data)
    
    # Display service status
    for _, row in health_df.iterrows():
        status_class = "healthy" if row["Healthy"] else "unhealthy"
        st.sidebar.markdown(f"""
        <div class="service-card {status_class}">
            <strong>{row['Service']}</strong><br>
            Port: {row['Port']} | Status: {row['Status']}
        </div>
        """, unsafe_allow_html=True)
    
    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧠 Intelligence", "🔍 Detection", "💻 Hardware", 
        "🎭 Deception", "🔎 Purple Team", "📈 Analytics"
    ])
    
    with tab1:
        st.header("🧠 Brain Intelligence Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Risk Assessment")
            
            # Signal inputs
            signal_cols = st.columns(3)
            with signal_cols[0]:
                ai_score = st.slider("AI Text Score", 0.0, 1.0, 0.3, 0.05)
                probe_sim = st.slider("Probe Similarity", 0.0, 1.0, 0.2, 0.05)
                canary_echo = st.slider("Canary Echo", 0.0, 5.0, 0.0, 0.5)
            
            with signal_cols[1]:
                rag_injection = st.slider("RAG Injection", 0.0, 1.0, 0.1, 0.05)
                ecc_delta = st.slider("ECC Delta", 0.0, 2.0, 0.0, 0.2)
                agent_anomaly = st.slider("Agent Anomaly", 0.0, 1.0, 0.0, 0.1)
            
            with signal_cols[2]:
                data_poison = st.slider("Data Poison Risk", 0.0, 1.0, 0.0, 0.1)
                threat_intel = st.slider("Threat Intel", 0.0, 10.0, 2.0, 0.5)
                hw_temp = st.slider("HW Temp Delta", 0.0, 50.0, 5.0, 2.0)
            
            if st.button("🔍 Assess Risk", type="primary"):
                signals = {
                    "ai_text_score": ai_score,
                    "probe_sim": probe_sim,
                    "canary_echo": canary_echo,
                    "rag_injection": rag_injection,
                    "ecc_delta": ecc_delta,
                    "agent_anomaly": agent_anomaly,
                    "data_poison_risk": data_poison,
                    "threat_intel_score": threat_intel / 10.0,
                    "hardware_temp": hw_temp / 50.0
                }
                
                with st.spinner("Calculating risk..."):
                    risk_result = assess_risk(signals)
                
                if risk_result:
                    st.session_state.last_risk = risk_result
        
        with col2:
            st.subheader("📊 Risk Results")
            
            if "last_risk" in st.session_state:
                risk = st.session_state.last_risk
                
                # Risk metrics
                prob = risk["probability"]
                level = risk["level"]
                
                # Color coding
                level_colors = {
                    "info": "🟩", "warn": "🟨", "high": "🟧", "critical": "🟥"
                }
                
                st.metric("Risk Probability", f"{prob:.3f}")
                st.metric("Risk Level", f"{level_colors.get(level, '⬜')} {level.upper()}")
                
                # Top features
                st.subheader("🎯 Top Features")
                for feature in risk["top_features"]:
                    contrib = feature["contribution"]
                    st.write(f"• {feature['feature']}: {contrib:+.3f}")
                
                # Policy recommendation
                if st.button("⚖️ Get Policy Action"):
                    options = ["observe", "raise_friction", "drain_node", "quarantine_host"]
                    policy_result = decide_policy(prob, options)
                    
                    if policy_result:
                        st.success(f"**Action:** {policy_result['action']}")
                        st.info(f"**Confidence:** {policy_result['confidence']:.3f}")
                        st.write(f"**Reason:** {policy_result['reason']}")
    
    with tab2:
        st.header("🔍 Detection & Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🤖 AI Content Detection")
            
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Paste content here to detect if it's AI-generated..."
            )
            
            case_id = st.text_input("Case ID:", value=f"detect-{int(time.time())}")
            
            if st.button("🔍 Analyze Content", type="primary") and text_input:
                with st.spinner("Analyzing content..."):
                    result = detect_ai_text(text_input, case_id)
                
                if result:
                    st.subheader("📊 Detection Results")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("AI Score", f"{result.get('ai_score', 0):.3f}")
                    with col_b:
                        st.metric("Confidence", f"{result.get('confidence', 0):.3f}")
                    with col_c:
                        verdict = "AI" if result.get('ai_score', 0) > 0.5 else "Human"
                        st.metric("Verdict", verdict)
                    
                    # Risk assessment
                    if "ai_score" in result:
                        st.subheader("🧠 Risk Assessment")
                        signals = {"ai_text_score": result["ai_score"]}
                        risk = assess_risk(signals)
                        
                        if risk:
                            st.write(f"**Risk Level:** {risk['level'].upper()} ({risk['probability']:.3f})")
                            
                            # Policy action
                            policy = decide_policy(risk["probability"], ["observe", "raise_friction"])
                            if policy:
                                st.write(f"**Recommended Action:** {policy['action']}")
        
        with col2:
            st.subheader("📈 Detection History")
            
            # Mock history data
            history_data = [
                {"time": "14:30", "case": "detect-001", "score": 0.85, "verdict": "AI"},
                {"time": "14:25", "case": "detect-002", "score": 0.23, "verdict": "Human"},
                {"time": "14:20", "case": "detect-003", "score": 0.91, "verdict": "AI"},
                {"time": "14:15", "case": "detect-004", "score": 0.12, "verdict": "Human"},
                {"time": "14:10", "case": "detect-005", "score": 0.67, "verdict": "AI"}
            ]
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
    
    with tab3:
        st.header("💻 Hardware Monitoring")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🖥️ System Status")
            
            # Get hardware status
            hw_status = get_hardware_status()
            
            # Mock hardware metrics
            st.subheader("📊 GPU Metrics")
            
            gpu_data = []
            for i in range(4):  # Mock 4 GPUs
                gpu_data.append({
                    "GPU": f"GPU {i}",
                    "Temperature": 65 + i * 3,
                    "Memory Usage": 45 + i * 10,
                    "ECC Errors": i * 2,
                    "Power": 250 + i * 20
                })
            
            gpu_df = pd.DataFrame(gpu_data)
            st.dataframe(gpu_df, use_container_width=True)
            
            # Hardware risk assessment
            st.subheader("🧠 Hardware Risk Analysis")
            
            ecc_input = st.slider("ECC Error Delta", 0.0, 5.0, 0.0, 0.5)
            temp_input = st.slider("Temperature Delta", 0.0, 30.0, 5.0, 2.0)
            
            if st.button("🔍 Assess Hardware Risk"):
                signals = {
                    "ecc_delta": ecc_input,
                    "hardware_temp": temp_input / 50.0,
                    "latency_ms": 0.1
                }
                
                risk = assess_risk(signals)
                if risk:
                    st.write(f"**Hardware Risk:** {risk['level'].upper()} ({risk['probability']:.3f})")
                    
                    # Hardware-specific policy
                    hw_options = ["observe", "throttle", "drain_node", "reset_gpu", "reattest"]
                    policy = decide_policy(risk["probability"], hw_options, "hardware")
                    
                    if policy:
                        st.success(f"**Hardware Action:** {policy['action']}")
                        st.info(f"**Reason:** {policy['reason']}")
        
        with col2:
            st.subheader("⚠️ Alerts")
            
            alerts = [
                {"time": "14:32", "type": "Temperature", "severity": "Warning", "gpu": "GPU 2"},
                {"time": "14:28", "type": "ECC Error", "severity": "Critical", "gpu": "GPU 1"},
                {"time": "14:15", "type": "Memory", "severity": "Info", "gpu": "GPU 3"}
            ]
            
            for alert in alerts:
                severity_color = {
                    "Critical": "🔴", "Warning": "🟡", "Info": "🔵"
                }.get(alert["severity"], "⚪")
                
                st.write(f"""
                **{severity_color} {alert['type']}**
                {alert['time']} - {alert['gpu']}<br>
                <small>{alert['severity']}</small>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.header("🎭 Deception & Honeynet")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🍯 Honeynet Configuration")
            
            # Session traits for personality selection
            st.write("**Configure Session Traits:**")
            
            trait_cols = st.columns(2)
            with trait_cols[0]:
                retry_rate = st.slider("Retry Rate (req/sec)", 0.0, 10.0, 1.5, 0.5)
                ua_len = st.slider("User Agent Length", 10, 200, 50, 10)
                depth = st.slider("Session Depth", 1, 10, 2, 1)
            
            with trait_cols[1]:
                cookie_support = st.checkbox("Cookie Support")
                js_support = st.checkbox("JavaScript Support")
                referer = st.checkbox("Referer Header")
            
            if st.button("🎭 Select Personality"):
                # Mock personality selection
                personalities = {
                    "fast_bot": {"template": "expensive_route", "throttle": 2000},
                    "browser_mimic": {"template": "dom_puzzle", "throttle": 500},
                    "api_client": {"template": "canary_dense", "throttle": 1000},
                    "stealthy": {"template": "fake_error", "throttle": 800}
                }
                
                # Simple classification logic
                if retry_rate > 5.0 and ua_len < 50:
                    selected = "fast_bot"
                elif ua_len > 100 and cookie_support:
                    selected = "browser_mimic"
                elif ua_len < 100 and not js_support:
                    selected = "api_client"
                else:
                    selected = "stealthy"
                
                personality = personalities[selected]
                
                st.success(f"**Selected Personality:** {selected}")
                st.info(f"**Template:** {personality['template']}")
                st.info(f"**Throttle:** {personality['throttle']}ms")
        
        with col2:
            st.subheader("🎯 Active Decoys")
            
            decoys = [
                {"name": "API Endpoint", "type": "REST API", "hits": 127},
                {"name": "Login Portal", "type": "Web Form", "hits": 89},
                {"name": "File Server", "type": "SMB", "hits": 45},
                {"name": "Database", "type": "MySQL", "hits": 23}
            ]
            
            for decoy in decoys:
                st.write(f"""
                **{decoy['name']}**
                Type: {decoy['type']}<br>
                Hits: {decoy['hits']}
                """, unsafe_allow_html=True)
    
    with tab5:
        st.header("🔎 Purple Team Operations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Offensive Testing")
            
            # Test configuration
            st.write("**Configure Test Parameters:**")
            
            target_url = st.text_input("Target URL:", "http://localhost:8012")
            test_type = st.selectbox("Test Type:", ["Probe Generation", "Pattern Matching", "Evasion Testing"])
            
            if st.button("🚀 Run Test", type="primary"):
                with st.spinner("Running offensive test..."):
                    # Mock test results
                    time.sleep(2)
                    
                    st.success("✅ Test Completed")
                    
                    # Results
                    st.subheader("📊 Test Results")
                    
                    results_cols = st.columns(3)
                    with results_cols[0]:
                        st.metric("Probes Sent", 47)
                    with results_cols[1]:
                        st.metric("Detections", 12)
                    with results_cols[2]:
                        st.metric("Evasion Rate", "74.5%")
                    
                    # Detailed results
                    st.write("**Detection Summary:**")
                    st.write("- AI-generated content detected: 8/12")
                    st.write("- Probe patterns identified: 15/20")
                    st.write("- Evasion techniques blocked: 9/15")
        
        with col2:
            st.subheader("📈 Test History")
            
            test_history = [
                {"time": "14:30", "type": "Probe Gen", "result": "Pass", "score": 85},
                {"time": "14:15", "type": "Pattern", "result": "Fail", "score": 45},
                {"time": "13:45", "type": "Evasion", "result": "Pass", "score": 92},
                {"time": "13:20", "type": "Probe Gen", "result": "Pass", "score": 78}
            ]
            
            for test in test_history:
                result_icon = "✅" if test["result"] == "Pass" else "❌"
                st.write(f"""
                **{result_icon} {test['type']}**
                {test['time']} - Score: {test['score']}%
                """, unsafe_allow_html=True)
    
    with tab6:
        st.header("📈 Analytics & Monitoring")
        
        # System overview metrics
        st.subheader("📊 System Overview")
        
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Active Services", f"{health_df['Healthy'].sum()}/{len(health_df)}")
        with metrics_cols[1]:
            st.metric("Total Detections", "1,247")
        with metrics_cols[2]:
            st.metric("Risk Score", "0.34")
        with metrics_cols[3]:
            st.metric("Uptime", "99.2%")
        
        # Charts
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            st.subheader("📈 Detection Trends")
            
            if PLOTLY_AVAILABLE:
                # Mock trend data
                trend_data = pd.DataFrame({
                    'Time': pd.date_range(start='2025-01-10 10:00', periods=24, freq='H'),
                    'Detections': [5, 8, 12, 7, 15, 9, 11, 6, 8, 10, 14, 9, 7, 12, 8, 6, 9, 11, 13, 8, 7, 10, 9, 6]
                })
                
                fig = px.line(trend_data, x='Time', y='Detections', title='Detection Volume Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Detection trend chart requires plotly. Install with: pip install plotly")
                # Show simple data table instead
                trend_data = pd.DataFrame({
                    'Hour': [f'{i:02d}:00' for i in range(24)],
                    'Detections': [5, 8, 12, 7, 15, 9, 11, 6, 8, 10, 14, 9, 7, 12, 8, 6, 9, 11, 13, 8, 7, 10, 9, 6]
                })
                st.dataframe(trend_data.head(10), use_container_width=True)
        
        with chart_cols[1]:
            st.subheader("🎯 Risk Distribution")
            
            if PLOTLY_AVAILABLE:
                risk_data = pd.DataFrame({
                    'Risk Level': ['Info', 'Warning', 'High', 'Critical'],
                    'Count': [45, 23, 12, 3]
                })
                
                fig = px.pie(risk_data, values='Count', names='Risk Level', title='Risk Level Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Risk distribution chart requires plotly. Install with: pip install plotly")
                # Show simple data table instead
                risk_data = pd.DataFrame({
                    'Risk Level': ['Info', 'Warning', 'High', 'Critical'],
                    'Count': [45, 23, 12, 3],
                    'Percentage': ['60.0%', '30.7%', '16.0%', '4.0%']
                })
                st.dataframe(risk_data, use_container_width=True)
        
        # Service performance table
        st.subheader("⚡ Service Performance")
        
        performance_data = []
        for _, row in health_df.iterrows():
            # Mock performance metrics
            performance_data.append({
                "Service": row["Service"],
                "Response Time": f"{50 + hash(row['Service']) % 100}ms",
                "Success Rate": f"{95 + hash(row['Service']) % 5}%",
                "Requests": f"{1000 + hash(row['Service']) % 5000}",
                "Status": "✅ Healthy" if row["Healthy"] else "❌ Unhealthy"
            })
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🛡️ AEGIS-C Unified Platform | Adaptive Counter-AI Intelligence | 
    Last Updated: {}
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
