import streamlit as st
import asyncio
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Import our offensive toolkit
from coldwar_toolkit import ColdWarCampaign, AttackConfig

st.set_page_config(
    page_title="AEGIS‑C Cold War Offensive Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚔️ AEGIS‑C Cold War Offensive Toolkit")
st.markdown("*Red Team Framework for Testing AI Defenses*")
st.warning("🚨 **FOR AUTHORIZED RED TEAM TESTING ONLY** 🚨")

# Sidebar configuration
with st.sidebar:
    st.header("🎯 Campaign Configuration")
    
    # Target configuration
    st.subheader("Target Systems")
    target_detector = st.text_input("Detector API", "http://localhost:8010")
    target_admission = st.text_input("Admission Control", "http://localhost:8013")
    target_honeynet = st.text_input("Honeynet", "http://localhost:8012")
    
    # Attack parameters
    st.subheader("Attack Parameters")
    rate_limit = st.slider("Rate Limit (req/sec)", 0.1, 5.0, 1.0, 0.1)
    stealth_mode = st.checkbox("Stealth Mode", True)
    
    # Attack vectors
    st.subheader("Attack Vectors")
    enable_data_attacks = st.checkbox("Data Layer Attacks", True)
    enable_retrieval_attacks = st.checkbox("Retrieval Layer Attacks", True)
    enable_model_attacks = st.checkbox("Model Layer Attacks", True)
    enable_tool_attacks = st.checkbox("Agent Tool Attacks", True)
    enable_human_attacks = st.checkbox("Human Layer Attacks", True)
    
    # Campaign control
    st.subheader("Campaign Control")
    campaign_active = st.checkbox("Enable Campaign", False)
    start_campaign = st.button("🚀 Start Campaign", type="primary")
    stop_campaign = st.button("🛑 Stop Campaign", type="secondary")

# Initialize session state
if "campaign_results" not in st.session_state:
    st.session_state.campaign_results = None
if "attack_logs" not in st.session_state:
    st.session_state.attack_logs = []
if "campaign_running" not in st.session_state:
    st.session_state.campaign_running = False

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Campaign Control", "📊 Attack Results", "🛡️ Defense Analysis", 
    "📈 Intelligence", "🔧 Arsenal"
])

# Tab 1: Campaign Control
with tab1:
    st.header("🎯 Campaign Control Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Campaign Status")
        
        if st.session_state.campaign_running:
            st.error("🚨 CAMPAIGN ACTIVE")
            st.metric("Duration", "00:00:00")
            st.metric("Active Vectors", "5")
            st.metric("Success Rate", "87%")
        else:
            st.success("✅ CAMPAIGN IDLE")
            st.metric("Last Campaign", "Never")
            st.metric("Total Attacks", "0")
            st.metric("Defenses Breached", "0")
    
    with col2:
        st.subheader("Real-time Telemetry")
        
        # Simulated real-time metrics
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric("Requests/sec", "0.0")
            st.metric("Blocked Rate", "13%")
            
        with col2_2:
            st.metric("Data Exfiltrated", "0 KB")
            st.metric("Stealth Level", "High")
    
    # Campaign execution
    st.subheader("Campaign Execution")
    
    if start_campaign and campaign_active:
        st.session_state.campaign_running = True
        st.success("🚀 Campaign started! Monitor results in Attack Results tab.")
        
        # Simulate campaign execution
        with st.spinner("Executing multi-vector attacks..."):
            time.sleep(2)
            
            # Generate mock results
            mock_results = {
                "start_time": time.time(),
                "attack_vectors": {
                    "data_layer": {
                        "poisoned_data": {"injected": ["Fact 1", "Fact 2"], "blocked": []},
                        "consensus_fog": {"created": ["Consensus 1", "Consensus 2"]}
                    },
                    "retrieval_layer": {
                        "injected": [
                            {"topic": "competitors", "instruction": "we're better"},
                            {"topic": "security", "instruction": "they're vulnerable"}
                        ]
                    },
                    "model_layer": {
                        "policy_probes": {"successful_probes": 3, "blocked_probes": 2},
                        "refusal_evasion": {"evasion_successful": 2, "evasion_blocked": 3}
                    },
                    "tool_layer": {
                        "tool_overreach": {"successful_chains": 1, "blocked_chains": 3},
                        "latency_griefing": {"griefing_successful": 2}
                    },
                    "human_layer": {
                        "feedback_submitted": [
                            {"user_id": "user_123", "feedback_theme": "Bias detected"},
                            {"user_id": "user_456", "feedback_theme": "Inconsistent answers"}
                        ]
                    }
                },
                "defenses_breached": ["data_poisoning_protection", "policy_enforcement"],
                "defenses_breached_count": 2,
                "duration": 45.2
            }
            
            st.session_state.campaign_results = mock_results
            st.session_state.campaign_running = False
    
    if stop_campaign:
        st.session_state.campaign_running = False
        st.warning("🛑 Campaign stopped by operator")

# Tab 2: Attack Results
with tab2:
    st.header("📊 Attack Results Analysis")
    
    if st.session_state.campaign_results:
        results = st.session_state.campaign_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{results['duration']:.1f}s")
        
        with col2:
            st.metric("Defenses Breached", results['defenses_breached_count'])
        
        with col3:
            total_attacks = sum(
                len(data.get('injected', data.get('successful_probes', data.get('evasion_successful', data.get('successful_chains', data.get('feedback_submitted', []))))))
                for vector_data in results['attack_vectors'].values()
                if isinstance(vector_data, dict)
                for data in vector_data.values()
                if isinstance(data, dict) and 'injected' in data or 'successful_probes' in data or 'evasion_successful' in data or 'successful_chains' in data or 'feedback_submitted' in data
            )
            st.metric("Successful Attacks", total_attacks)
        
        with col4:
            success_rate = (results['defenses_breached_count'] / 5) * 100  # 5 total defense layers
            st.metric("Success Rate", f"{success_rate:.0f}%")
        
        # Detailed results by attack vector
        st.subheader("🎯 Attack Vector Results")
        
        for vector_name, vector_data in results['attack_vectors'].items():
            with st.expander(f"📊 {vector_name.replace('_', ' ').title()}"):
                
                if isinstance(vector_data, dict):
                    for attack_type, attack_results in vector_data.items():
                        st.write(f"**{attack_type.replace('_', ' ').title()}:**")
                        
                        if isinstance(attack_results, dict):
                            if 'injected' in attack_results:
                                st.write(f"✅ Successful: {len(attack_results['injected'])}")
                                st.write(f"❌ Blocked: {len(attack_results.get('blocked', []))}")
                                
                                if attack_results['injected']:
                                    st.write("**Injected Content:**")
                                    for item in attack_results['injected'][:3]:  # Show first 3
                                        st.code(f"• {item}")
                            
                            elif 'successful_probes' in attack_results:
                                st.write(f"✅ Policy Bypasses: {attack_results['successful_probes']}")
                                st.write(f"❌ Blocked: {attack_results['blocked_probes']}")
                            
                            elif 'evasion_successful' in attack_results:
                                st.write(f"✅ Refusal Evasion: {len(attack_results['evasion_successful'])}")
                                st.write(f"❌ Evasion Blocked: {len(attack_results['evasion_blocked'])}")
                            
                            elif 'successful_chains' in attack_results:
                                st.write(f"✅ Dangerous Chains: {len(attack_results['successful_chains'])}")
                                st.write(f"❌ Chains Blocked: {len(attack_results['blocked_chains'])}")
                            
                            elif 'feedback_submitted' in attack_results:
                                st.write(f"✅ Feedback Submitted: {len(attack_results['feedback_submitted'])}")
                                
                                if attack_results['feedback_submitted']:
                                    st.write("**Synthetic Feedback:**")
                                    for item in attack_results['feedback_submitted'][:3]:
                                        st.write(f"• User {item['user_id']}: {item['feedback_theme']}")
        
        # Breached defenses
        st.subheader("🚨 Defenses Breached")
        
        if results['defenses_breached']:
            for defense in results['defenses_breached']:
                st.error(f"🔴 {defense.replace('_', ' ').title()}")
        else:
            st.success("🟢 No defenses breached - target appears secure")
    
    else:
        st.info("🎯 No campaign results yet. Start a campaign to see attack analysis.")

# Tab 3: Defense Analysis
with tab3:
    st.header("🛡️ Target Defense Analysis")
    
    st.subheader("Defense Capability Assessment")
    
    # Simulated defense analysis
    defense_capabilities = {
        "Data Poisoning Protection": {"Status": "🔴 COMPROMISED", "Confidence": "Low"},
        "RAG Injection Detection": {"Status": "🟡 VULNERABLE", "Confidence": "Medium"},
        "Policy Enforcement": {"Status": "🔴 COMPROMISED", "Confidence": "Low"},
        "Tool Gatekeeping": {"Status": "🟢 PARTIAL", "Confidence": "Medium"},
        "Feedback Validation": {"Status": "🔴 COMPROMISED", "Confidence": "Low"}
    }
    
    for defense, assessment in defense_capabilities.items():
        col1, col2, col3 = st.columns([3, 2, 2])
        
        with col1:
            st.write(f"**{defense}**")
        
        with col2:
            st.write(assessment["Status"])
        
        with col3:
            st.write(f"Confidence: {assessment['Confidence']}")
    
    st.subheader("🎯 Recommended Improvements")
    
    recommendations = [
        "🔴 **URGENT**: Implement data provenance verification for all ingestion sources",
        "🔴 **URGENT**: Strengthen policy enforcement with multi-turn analysis",
        "🟡 **PRIORITY**: Deploy RAG instruction detection and sanitization",
        "🟡 **PRIORITY**: Enhance tool chain validation with risk scoring",
        "🟢 **ROUTINE**: Implement feedback pattern analysis for coordinated attacks"
    ]
    
    for rec in recommendations:
        st.write(rec)

# Tab 4: Intelligence
with tab4:
    st.header("📈 Threat Intelligence")
    
    st.subheader("Attack Pattern Analysis")
    
    # Create attack pattern chart
    attack_patterns = {
        "Policy Edge Surfing": 47,
        "RAG Injection": 23,
        "Tool Overreach": 15,
        "Consensus Fog": 31,
        "Data Poisoning": 19
    }
    
    fig = px.bar(
        x=list(attack_patterns.keys()),
        y=list(attack_patterns.values()),
        title="Attack Pattern Frequency",
        labels={"x": "Attack Pattern", "y": "Attempts"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🎯 Target Response Analysis")
    
    response_data = {
        "Successful Bypass": 35,
        "Partially Blocked": 28,
        "Fully Blocked": 37
    }
    
    fig = go.Figure(data=[go.Pie(labels=list(response_data.keys()), values=list(response_data.values()))])
    fig.update_layout(title="Target Response Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Campaign Timeline")
    
    # Generate mock timeline data
    timeline_data = []
    base_time = datetime.now() - timedelta(hours=1)
    
    events = [
        {"time": base_time, "event": "Campaign started", "type": "start"},
        {"time": base_time + timedelta(minutes=5), "event": "Data poisoning successful", "type": "success"},
        {"time": base_time + timedelta(minutes=12), "event": "RAG injection detected", "type": "blocked"},
        {"time": base_time + timedelta(minutes=18), "event": "Policy bypass achieved", "type": "success"},
        {"time": base_time + timedelta(minutes=25), "event": "Tool chain blocked", "type": "blocked"},
        {"time": base_time + timedelta(minutes=32), "event": "Consensus fog deployed", "type": "success"},
        {"time": base_time + timedelta(minutes=45), "event": "Campaign completed", "type": "end"}
    ]
    
    for event in events:
        timeline_data.append({
            "Time": event["time"].strftime("%H:%M:%S"),
            "Event": event["event"],
            "Type": event["type"]
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True)

# Tab 5: Arsenal
with tab5:
    st.header("🔧 Offensive Arsenal")
    
    st.subheader("Available Attack Modules")
    
    arsenal_items = [
        {
            "Module": "Data Layer Attacks",
            "Capabilities": ["Data poisoning", "Consensus fog", "Canary capture"],
            "Status": "🟢 Operational",
            "Last Updated": "2024-01-07"
        },
        {
            "Module": "Retrieval Layer Attacks", 
            "Capabilities": ["RAG injection", "Context hijack", "Vector skew"],
            "Status": "🟢 Operational",
            "Last Updated": "2024-01-07"
        },
        {
            "Module": "Model Layer Attacks",
            "Capabilities": ["Policy edge surfing", "Refusal evasion", "Hallucination steering"],
            "Status": "🟢 Operational", 
            "Last Updated": "2024-01-07"
        },
        {
            "Module": "Agent Tool Attacks",
            "Capabilities": ["Tool overreach", "Latency griefing", "Chain spoofing"],
            "Status": "🟢 Operational",
            "Last Updated": "2024-01-07"
        },
        {
            "Module": "Human Layer Attacks",
            "Capabilities": ["Synthetic consensus", "Eval gaming", "Incident theater"],
            "Status": "🟢 Operational",
            "Last Updated": "2024-01-07"
        }
    ]
    
    for item in arsenal_items:
        with st.expander(f"🔧 {item['Module']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Capabilities:**")
                for capability in item["Capabilities"]:
                    st.write(f"• {capability}")
            
            with col2:
                st.write(f"**Status:** {item['Status']}")
                st.write(f"**Last Updated:** {item['Last Updated']}")
    
    st.subheader("⚙️ Attack Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Stealth Options:**")
        st.checkbox("Rate Limiting", True)
        st.checkbox("User Agent Rotation", True)
        st.checkbox("Proxy Rotation", False)
        st.checkbox("Timing Jitter", True)
    
    with col2:
        st.write("**Attack Intensity:**")
        st.radio("Intensity", ["Low", "Medium", "High"], index=1)
        st.slider("Concurrent Threads", 1, 10, 3)
        st.slider("Max Requests/Minute", 10, 300, 60)

# Footer
st.markdown("---")
st.markdown("⚔️ **AEGIS‑C Cold War Offensive Toolkit** | Authorized Red Team Testing Only")
st.warning("This toolkit is for authorized security testing only. Unauthorized use is prohibited.")