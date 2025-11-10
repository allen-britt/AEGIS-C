# 🧠 AEGIS‑C Intelligence Integration

This document describes the clean, testable intelligence architecture that makes AEGIS‑C "smart."

## 🎯 What Makes AEGIS‑C Smart

### 1. **Brain Gateway** (`services/brain/main.py`)
- **Risk Assessment**: Fuses 10+ signals into probability with explainable features
- **Policy Decisions**: Chooses optimal actions based on risk and context
- **Fast & Reliable**: <100ms response time, deterministic fallback logic

### 2. **Universal Brain Client** (`services/common/brain_client.py`)
- **Simple Integration**: `assess()` and `decide()` functions for any service
- **Error Handling**: Graceful fallbacks when brain is unavailable
- **Convenience Functions**: Pre-built for common use cases

### 3. **Intelligent Service Wiring**
Every service now calls the brain for risk assessment and policy decisions:
- **Detector**: `ai_text_score` → risk → policy action
- **Hardware**: `ecc_delta` → risk → reset/drain decision
- **Honeynet**: Session traits → adaptive personality
- **Admission**: Content analysis → quarantine decision

## 🚀 Quick Start

### 1. Start the Brain Gateway
```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Start brain service
uvicorn services.brain.main:app --port 8030 --host 0.0.0.0

# Verify it's working
curl http://localhost:8030/health
```

### 2. Test Intelligence
```bash
# Run brain tests
bash tests/test_brain.sh

# Run full intelligence verification
bash verify-smart.sh
```

### 3. Start Full Platform
```bash
# Start brain first
docker-compose up brain -d

# Start other services (they'll connect to brain)
docker-compose up detector honeynet hardware

# Start console with brain demo
streamlit run services/console/brain_demo.py
```

## 📊 API Contracts

### Risk Assessment
```bash
POST /risk
{
  "subject": "artifact:123",
  "kind": "artifact", 
  "signals": [
    {"name": "ai_text_score", "value": 0.8},
    {"name": "canary_echo", "value": 2.0}
  ],
  "context": {"source": "detector"}
}

→ {
  "probability": 0.742,
  "level": "high",
  "top_features": [
    {"feature": "canary_echo", "contribution": 4.0},
    {"feature": "ai_text_score", "contribution": 0.96}
  ],
  "timestamp": "2025-01-10T18:30:00Z"
}
```

### Policy Decision
```bash
POST /policy
{
  "subject": "artifact:123",
  "kind": "artifact",
  "risk": 0.742,
  "options": ["observe", "raise_friction", "drain_node"]
}

→ {
  "action": "raise_friction",
  "confidence": 0.8,
  "reason": "High risk - applying friction",
  "timestamp": "2025-01-10T18:30:00Z"
}
```

## 🔧 Service Integration Examples

### Detector Service
```python
from services.common.brain_client import assess_artifact_risk, decide_artifact_action

# After scoring text
risk = assess_artifact_risk(case_id, ai_score)
if risk["probability"] >= 0.5:
    action = decide_artifact_action(case_id, risk["probability"])
    # Attach to alert payload
```

### Hardware Sentinel
```python
from services.common.brain_client import assess_hardware_risk, decide_hardware_action

# On ECC spike
risk = assess_hardware_risk(node_id, ecc_spike=1.5)
action = decide_hardware_action(node_id, risk["probability"])
# Execute reset/drain based on recommendation
```

## 🧪 Testing & Verification

### Acceptance Criteria
- ✅ **Brain Health**: `/risk` and `/policy` respond within 100ms
- ✅ **Detector Integration**: `ai_text_score ≥ 0.6` → `risk ≥ 0.6` → `action ≠ observe`
- ✅ **Hardware Integration**: ECC spike → `action = reset_gpu` or `drain_node`
- ✅ **Explainability**: Every risk assessment includes top contributing features
- ✅ **Consistency**: Same signals → same probability ±5%

### Automated Tests
```bash
# Brain gateway tests
bash tests/test_brain.sh

# Full intelligence verification
bash verify-smart.sh

# CI integration
.github/workflows/ci.yml  # Includes brain tests
```

## 🎛️ Console Demo Panel

The Streamlit console includes an interactive brain demo:

**Features:**
- 🎯 **Risk Assessment**: Adjust signal sliders and see real-time risk calculation
- ⚖️ **Policy Decisions**: Get intelligent action recommendations
- 🔬 **Signal Explorer**: Analyze individual signal impacts
- 🎭 **Preset Scenarios**: Test common threat patterns

**Access:**
```bash
streamlit run services/console/brain_demo.py
# Navigate to "Brain Intelligence Demo"
```

## 📡 Stub Services

For testing and development, deterministic stubs provide predictable responses:

| Service | Port | Endpoint | Purpose |
|---------|------|----------|---------|
| Probe Smith | 8021 | `/propose` | Generate probes from near misses |
| RAG Guard | 8022 | `/sanitize` | Content sanitization |
| Honeynet Personality | 8023 | `/policy` | Adaptive personality selection |
| Causal Explainer | 8024 | `/causal` | Root cause analysis |

**Start stubs:**
```bash
python services/probe_smith/stub.py &
python services/detector/rag_guard_stub.py &
python services/honeynet/personality_stub.py &
python services/brain/causal_stub.py &
```

## 🔄 Docker Integration

All services are wired to the brain in `docker-compose.yml`:

```yaml
brain:
  build: ./services/brain
  ports: ["8030:8030"]
  
detector:
  environment: ["BRAIN_URL=http://brain:8030"]
  depends_on: [brain]
  
# ... similar for all other services
```

## 📈 Intelligence Metrics

The brain tracks its own performance:
- **Response Times**: P50 <100ms target
- **Risk Accuracy**: Consistent probability calculations
- **Policy Effectiveness**: Action success rates
- **Learning Progress**: Corrections and improvements over time

Access via: `GET /secure/status` (from any service)

## 🎯 Next Steps

### Replace Stubs with Real Intelligence
1. **Probe Smith**: Replace stub with `services/probe_smith/main.py`
2. **RAG Guard**: Replace stub with `services/detector/rag_guard.py`  
3. **Causal Explainer**: Replace stub with `services/brain/causal.py`
4. **Active Learning**: Enable `services/brain/training_job.py`

### Enhanced Features
1. **Multi-Armed Bandits**: Replace policy rules with Thompson Sampling
2. **Feature Learning**: Update risk model from analyst corrections
3. **Causal Patterns**: Learn from incident history
4. **Hardware Intent**: Correlate anomalies with model impact

## 🏆 Success Indicators

AEGIS‑C is "smart" when:

- ✅ **Every alert** includes probability + feature attribution
- ✅ **Every action** includes reasoning + confidence score  
- ✅ **Services learn** from analyst corrections
- ✅ **Response times** stay under 100ms
- ✅ **False positives** decrease over time
- ✅ **Threat detection** improves with experience

---

**Result**: AEGIS‑C evolves from static tooling to an adaptive intelligence platform that learns, explains, and improves itself! 🧠✨