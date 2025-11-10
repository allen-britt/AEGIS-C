#!/usr/bin/env python3
"""
AEGIS‑C Brain Demo Panel
========================

Streamlit panel demonstrating Brain Gateway intelligence.
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime

# Configuration
BRAIN_URL = os.getenv("BRAIN_URL", "http://localhost:8030")

def check_brain_health():
    """Check if Brain Gateway is available"""
    try:
        response = requests.get(f"{BRAIN_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def assess_risk(signals_dict):
    """Assess risk using Brain Gateway"""
    try:
        signals = [{"name": k, "value": float(v)} for k, v in signals_dict.items()]
        payload = {
            "subject": "demo:streamlit",
            "kind": "artifact",
            "signals": signals,
            "context": {"source": "brain_demo"}
        }
        
        response = requests.post(f"{BRAIN_URL}/risk", json=payload, timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Risk assessment failed: {e}")
        return None

def decide_policy(risk_probability, options):
    """Get policy decision from Brain Gateway"""
    try:
        payload = {
            "subject": "demo:streamlit",
            "kind": "artifact",
            "risk": float(risk_probability),
            "options": options
        }
        
        response = requests.post(f"{BRAIN_URL}/policy", json=payload, timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Policy decision failed: {e}")
        return None

def run_brain_demo():
    """Run the Brain Gateway demo"""
    
    st.title("🧠 AEGIS‑C Brain Intelligence Demo")
    st.markdown("---")
    
    # Health check
    if not check_brain_health():
        st.error("❌ Brain Gateway is not available. Please ensure it's running.")
        st.info(f"Expected at: {BRAIN_URL}")
        return
    
    st.success("✅ Brain Gateway is connected and healthy")
    
    # Demo sections
    tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Policy Decision", "Signal Explorer"])
    
    with tab1:
        st.header("🎯 Adaptive Risk Assessment")
        st.markdown("Test different signal combinations to see how the Brain calculates risk probability.")
        
        # Signal inputs
        col1, col2 = st.columns(2)
        
        with col1:
            ai_score = st.slider("AI Text Score", 0.0, 1.0, 0.3, 0.05, help="AI-generated content probability")
            probe_sim = st.slider("Probe Similarity", 0.0, 1.0, 0.2, 0.05, help="Similarity to known probes")
            canary_echo = st.slider("Canary Echo", 0.0, 5.0, 0.0, 0.5, help="Canary token activation count")
        
        with col2:
            rag_injection = st.slider("RAG Injection", 0.0, 1.0, 0.1, 0.05, help="Injection attempt score")
            ecc_delta = st.slider("ECC Delta", 0.0, 2.0, 0.0, 0.2, help="Hardware ECC anomaly")
            latency_ms = st.slider("Latency (ms)", 0, 5000, 100, 50, help="Request latency")
        
        # Additional signals
        st.subheader("Additional Signals")
        col3, col4 = st.columns(2)
        
        with col3:
            agent_anomaly = st.slider("Agent Anomaly", 0.0, 1.0, 0.0, 0.1)
            data_poison_risk = st.slider("Data Poison Risk", 0.0, 1.0, 0.0, 0.1)
        
        with col4:
            threat_intel = st.slider("Threat Intel Score", 0.0, 10.0, 2.0, 0.5)
            hardware_temp = st.slider("Hardware Temp Delta", 0.0, 50.0, 5.0, 2.0)
        
        # Assess risk button
        if st.button("🔍 Assess Risk", type="primary"):
            signals = {
                "ai_text_score": ai_score,
                "probe_sim": probe_sim,
                "canary_echo": canary_echo,
                "rag_injection": rag_injection,
                "ecc_delta": ecc_delta,
                "latency_ms": latency_ms / 5000.0,  # Normalize
                "agent_anomaly": agent_anomaly,
                "data_poison_risk": data_poison_risk,
                "threat_intel_score": threat_intel / 10.0,  # Normalize
                "hardware_temp": hardware_temp / 50.0  # Normalize
            }
            
            with st.spinner("Assessing risk..."):
                risk_result = assess_risk(signals)
            
            if risk_result:
                # Display results
                st.markdown("### 📊 Risk Assessment Results")
                
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    prob = risk_result["probability"]
                    color = "🟢" if prob < 0.2 else "🟡" if prob < 0.5 else "🟠" if prob < 0.75 else "🔴"
                    st.metric("Risk Probability", f"{prob:.3f}", delta=None)
                
                with col6:
                    level = risk_result["level"]
                    level_color = {
                        "info": "🟩", "warn": "🟨", "high": "🟧", "critical": "🟥"
                    }.get(level, "⬜")
                    st.metric("Risk Level", f"{level_color} {level.upper()}")
                
                with col7:
                    timestamp = risk_result["timestamp"]
                    st.metric("Assessed At", timestamp[:19])
                
                # Top features
                st.subheader("🎯 Top Contributing Features")
                features = risk_result["top_features"]
                
                for i, feature in enumerate(features, 1):
                    contrib = feature["contribution"]
                    bar_color = "🔴" if abs(contrib) > 1.0 else "🟡" if abs(contrib) > 0.5 else "🟢"
                    st.write(f"{i}. {feature['feature']}: {bar_color} {contrib:+.3f}")
                
                # Store risk for policy decision
                st.session_state.last_risk = risk_result
    
    with tab2:
        st.header("⚖️ Adaptive Policy Decision")
        st.markdown("Based on the risk assessment, get an intelligent policy recommendation.")
        
        # Check if we have a risk assessment
        if "last_risk" not in st.session_state:
            st.info("👈 Please run a risk assessment first to get policy recommendations.")
        else:
            risk_result = st.session_state.last_risk
            risk_prob = risk_result["probability"]
            
            st.write(f"**Current Risk Probability:** {risk_prob:.3f} ({risk_result['level']})")
            
            # Policy options
            st.subheader("Available Policy Options")
            options = ["observe", "raise_friction", "drain_node", "reset_gpu", "quarantine_host"]
            selected_options = st.multiselect(
                "Select available options:",
                options,
                default=["observe", "raise_friction", "drain_node"]
            )
            
            if selected_options and st.button("🎯 Get Policy Recommendation", type="primary"):
                with st.spinner("Analyzing policy options..."):
                    policy_result = decide_policy(risk_prob, selected_options)
                
                if policy_result:
                    st.markdown("### 📋 Policy Recommendation")
                    
                    col8, col9, col10 = st.columns(3)
                    
                    with col8:
                        action = policy_result["action"]
                        action_icon = {
                            "observe": "👁️", "raise_friction": "⚠️", "drain_node": "🚫",
                            "reset_gpu": "🔄", "quarantine_host": "🔒"
                        }.get(action, "❓")
                        st.metric("Recommended Action", f"{action_icon} {action}")
                    
                    with col9:
                        conf = policy_result["confidence"]
                        conf_color = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                        st.metric("Confidence", f"{conf_color} {conf:.3f}")
                    
                    with col10:
                        timestamp = policy_result["timestamp"]
                        st.metric("Decided At", timestamp[:19])
                    
                    # Reasoning
                    st.subheader("🧠 Reasoning")
                    st.info(policy_result["reasoning"])
    
    with tab3:
        st.header("🔬 Signal Explorer")
        st.markdown("Explore how individual signals contribute to overall risk.")
        
        # Preset scenarios
        st.subheader("Preset Scenarios")
        
        scenarios = {
            "Normal Traffic": {
                "ai_text_score": 0.1, "probe_sim": 0.0, "canary_echo": 0.0,
                "rag_injection": 0.0, "ecc_delta": 0.0, "latency_ms": 100
            },
            "Suspicious Content": {
                "ai_text_score": 0.8, "probe_sim": 0.6, "canary_echo": 1.0,
                "rag_injection": 0.3, "ecc_delta": 0.0, "latency_ms": 200
            },
            "Hardware Anomaly": {
                "ai_text_score": 0.2, "probe_sim": 0.1, "canary_echo": 0.0,
                "rag_injection": 0.0, "ecc_delta": 1.5, "latency_ms": 1000
            },
            "Critical Threat": {
                "ai_text_score": 0.9, "probe_sim": 0.8, "canary_echo": 3.0,
                "rag_injection": 0.9, "ecc_delta": 1.2, "latency_ms": 2000
            }
        }
        
        selected_scenario = st.selectbox("Choose a scenario:", list(scenarios.keys()))
        
        if st.button(f"🎭 Load Scenario: {selected_scenario}"):
            scenario_signals = scenarios[selected_scenario]
            
            # Normalize signals
            normalized_signals = scenario_signals.copy()
            normalized_signals["latency_ms"] = scenario_signals["latency_ms"] / 5000.0
            
            with st.spinner(f"Analyzing {selected_scenario} scenario..."):
                risk_result = assess_risk(normalized_signals)
            
            if risk_result:
                st.markdown(f"### 📊 {selected_scenario} Risk Analysis")
                
                col11, col12 = st.columns(2)
                
                with col11:
                    prob = risk_result["probability"]
                    level = risk_result["level"]
                    st.metric("Risk Probability", f"{prob:.3f}")
                    st.metric("Risk Level", level.upper())
                
                with col12:
                    # Signal contributions
                    st.write("**Signal Contributions:**")
                    for feature in risk_result["top_features"]:
                        contrib = feature["contribution"]
                        st.write(f"• {feature['feature']}: {contrib:+.3f}")
                
                # Policy recommendation
                if st.button(f"🎯 Get Policy for {selected_scenario}"):
                    policy_result = decide_policy(prob, ["observe", "raise_friction", "drain_node", "quarantine_host"])
                    
                    if policy_result:
                        st.success(f"**Recommended:** {policy_result['action']} (confidence: {policy_result['confidence']:.3f})")
                        st.info(policy_result["reasoning"])
        
        # Signal impact analysis
        st.markdown("---")
        st.subheader("📈 Signal Impact Analysis")
        st.markdown("See how changing individual signals affects overall risk.")
        
        base_signals = {
            "ai_text_score": 0.3,
            "probe_sim": 0.2,
            "canary_echo": 0.0,
            "rag_injection": 0.1,
            "ecc_delta": 0.0,
            "latency_ms": 100
        }
        
        signal_to_vary = st.selectbox("Signal to analyze:", list(base_signals.keys()))
        
        if signal_to_vary:
            # Create range of values
            if signal_to_vary == "canary_echo":
                values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            elif signal_to_vary == "latency_ms":
                values = [50, 200, 500, 1000, 2000, 5000]
            else:
                values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            
            if st.button(f"📊 Analyze {signal_to_vary} Impact"):
                st.write(f"**Risk probability vs {signal_to_vary}:**")
                
                results = []
                for value in values:
                    test_signals = base_signals.copy()
                    test_signals[signal_to_vary] = value
                    
                    # Normalize for API
                    normalized = test_signals.copy()
                    normalized["latency_ms"] = test_signals["latency_ms"] / 5000.0
                    
                    risk_result = assess_risk(normalized)
                    if risk_result:
                        results.append((value, risk_result["probability"], risk_result["level"]))
                
                # Display results
                if results:
                    import pandas as pd
                    df = pd.DataFrame(results, columns=["Value", "Risk Probability", "Risk Level"])
                    st.dataframe(df, use_container_width=True)
                    
                    # Simple chart
                    st.line_chart(df.set_index("Value")["Risk Probability"])

if __name__ == "__main__":
    run_brain_demo()