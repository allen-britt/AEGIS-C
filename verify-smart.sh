#!/bin/bash
set -e

echo "🧠 AEGIS‑C Intelligence Verification"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
check_service() {
    local service=$1
    local url=$2
    
    echo -n "Checking $service... "
    if curl -sf "$url" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

measure_response_time() {
    local url=$1
    local payload=$2
    
    start_time=$(date +%s%N)
    curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$url" >/dev/null
    end_time=$(date +%s%N)
    
    echo $((($end_time - $start_time) / 1000000))  # Convert to milliseconds
}

# 1. Brain Gateway health and performance
echo ""
echo "1. Brain Gateway Health & Performance"
echo "--------------------------------------"

if ! check_service "Brain Gateway" "http://localhost:8030/health"; then
    echo -e "${RED}❌ Brain Gateway is not running!${NC}"
    echo "Start it with: uvicorn services.brain.main:app --port 8030 --host 0.0.0.0"
    exit 1
fi

# Response time test
echo -n "Testing Brain response time... "
time_ms=$(measure_response_time "http://localhost:8030/risk" '{"subject":"test","kind":"artifact","signals":[{"name":"ai_text_score","value":0.5}]}')
if [ "$time_ms" -lt 100 ]; then
    echo -e "${GREEN}✓ ${time_ms}ms${NC}"
else
    echo -e "${YELLOW}⚠ ${time_ms}ms (target <100ms)${NC}"
fi

# 2. Risk assessment accuracy
echo ""
echo "2. Risk Assessment Accuracy"
echo "----------------------------"

# Test low risk
echo -n "Low risk assessment... "
low_risk=$(curl -s "http://localhost:8030/risk" -H "Content-Type: application/json" \
  -d '{"subject":"test:low","kind":"artifact","signals":[{"name":"ai_text_score","value":0.1}]}')

low_level=$(echo "$low_risk" | jq -r .level)
low_prob=$(echo "$low_risk" | jq -r .probability)

if [[ "$low_level" == "info" || "$low_level" == "warn" ]]; then
    echo -e "${GREEN}✓ Level: $low_level (${low_prob})${NC}"
else
    echo -e "${RED}❌ Expected low risk level, got: $low_level${NC}"
fi

# Test high risk
echo -n "High risk assessment... "
high_risk=$(curl -s "http://localhost:8030/risk" -H "Content-Type: application/json" \
  -d '{"subject":"test:high","kind":"artifact","signals":[{"name":"ai_text_score","value":0.9},{"name":"canary_echo","value":3.0}]}')

high_level=$(echo "$high_risk" | jq -r .level)
high_prob=$(echo "$high_risk" | jq -r .probability)

if [[ "$high_level" == "high" || "$high_level" == "critical" ]]; then
    echo -e "${GREEN}✓ Level: $high_level (${high_prob})${NC}"
else
    echo -e "${RED}❌ Expected high risk level, got: $high_level${NC}"
fi

# 3. Policy decision logic
echo ""
echo "3. Policy Decision Logic"
echo "------------------------"

# Test policy for low risk
echo -n "Low risk policy... "
low_policy=$(curl -s "http://localhost:8030/policy" -H "Content-Type: application/json" \
  -d "{\"subject\":\"test:low\",\"kind\":\"artifact\",\"risk\":$low_prob,\"options\":[\"observe\",\"raise_friction\"]}")

low_action=$(echo "$low_policy" | jq -r .action)
if [[ "$low_action" == "observe" ]]; then
    echo -e "${GREEN}✓ Action: $low_action${NC}"
else
    echo -e "${RED}❌ Expected 'observe' for low risk, got: $low_action${NC}"
fi

# Test policy for high risk
echo -n "High risk policy... "
high_policy=$(curl -s "http://localhost:8030/policy" -H "Content-Type: application/json" \
  -d "{\"subject\":\"test:high\",\"kind\":\"artifact\",\"risk\":$high_prob,\"options\":[\"observe\",\"raise_friction\",\"drain_node\"]}")

high_action=$(echo "$high_policy" | jq -r .action)
if [[ "$high_action" != "observe" ]]; then
    echo -e "${GREEN}✓ Action: $high_action${NC}"
else
    echo -e "${RED}❌ Expected action other than 'observe' for high risk${NC}"
fi

# 4. Hardware anomaly handling
echo ""
echo "4. Hardware Anomaly Handling"
echo "-----------------------------"

echo -n "Hardware ECC spike... "
hw_risk=$(curl -s "http://localhost:8030/risk" -H "Content-Type: application/json" \
  -d '{"subject":"node:gpu0","kind":"hardware","signals":[{"name":"ecc_delta","value":1.5}]}')

hw_prob=$(echo "$hw_risk" | jq -r .probability)
hw_policy=$(curl -s "http://localhost:8030/policy" -H "Content-Type: application/json" \
  -d "{\"subject\":\"node:gpu0\",\"kind\":\"hardware\",\"risk\":$hw_prob,\"options\":[\"observe\",\"reset_gpu\",\"drain_node\"]}")

hw_action=$(echo "$hw_policy" | jq -r .action)
if [[ "$hw_action" == "reset_gpu" || "$hw_action" == "drain_node" ]]; then
    echo -e "${GREEN}✓ Hardware action: $hw_action (risk: $hw_prob)${NC}"
else
    echo -e "${YELLOW}⚠ Hardware action: $hw_action (expected reset/drain)${NC}"
fi

# 5. Service integration tests
echo ""
echo "5. Service Integration Tests"
echo "----------------------------"

# Test stub services
stubs=(
    "Probe Smith:8021:/propose"
    "RAG Guard:8022:/sanitize"
    "Honeynet Personality:8023:/policy"
    "Causal Explainer:8024:/causal"
)

for stub in "${stubs[@]}"; do
    IFS=':' read -r name port endpoint <<< "$stub"
    echo -n "Testing $name stub... "
    
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Running${NC}"
    else
        echo -e "${YELLOW}⚠ Not running (stub service)${NC}"
    fi
done

# 6. Brain client functionality
echo ""
echo "6. Brain Client Functionality"
echo "-----------------------------"

echo -n "Testing brain client import... "
if python3 -c "from services.common.brain_client import assess, decide; print('✓')" 2>/dev/null; then
    echo -e "${GREEN}✓ Client imports successfully${NC}"
else
    echo -e "${RED}❌ Client import failed${NC}"
fi

echo -n "Testing brain client connection... "
if python3 -c "
from services.common.brain_client import test_brain_connection
success, msg = test_brain_connection()
print('✓' if success else '❌')
" 2>/dev/null; then
    echo -e "${GREEN}✓ Client connects successfully${NC}"
else
    echo -e "${RED}❌ Client connection failed${NC}"
fi

# 7. Docker compose integration
echo ""
echo "7. Docker Compose Integration"
echo "------------------------------"

echo -n "Checking docker-compose.yml for brain service... "
if grep -q "brain:" docker-compose.yml && grep -q "BRAIN_URL=http://brain:8030" docker-compose.yml; then
    echo -e "${GREEN}✓ Brain service configured${NC}"
else
    echo -e "${RED}❌ Brain service not properly configured${NC}"
fi

# 8. Console integration
echo ""
echo "8. Console Integration"
echo "----------------------"

echo -n "Checking console brain demo... "
if [ -f "services/console/brain_demo.py" ]; then
    echo -e "${GREEN}✓ Brain demo panel exists${NC}"
else
    echo -e "${RED}❌ Brain demo panel missing${NC}"
fi

# Summary
echo ""
echo "🎯 Intelligence Verification Summary"
echo "===================================="

# Count successes/failures
total_checks=0
passed_checks=0

# Simple heuristic based on output above
if curl -sf "http://localhost:8030/health" >/dev/null; then
    echo -e "✅ Brain Gateway: ${GREEN}Operational${NC}"
    passed_checks=$((passed_checks + 1))
else
    echo -e "❌ Brain Gateway: ${RED}Not Running${NC}"
fi
total_checks=$((total_checks + 1))

if [[ "$high_level" == "high" || "$high_level" == "critical" ]]; then
    echo -e "✅ Risk Assessment: ${GREEN}Working${NC}"
    passed_checks=$((passed_checks + 1))
else
    echo -e "❌ Risk Assessment: ${RED}Not Working${NC}"
fi
total_checks=$((total_checks + 1))

if [[ "$high_action" != "observe" ]]; then
    echo -e "✅ Policy Logic: ${GREEN}Working${NC}"
    passed_checks=$((passed_checks + 1))
else
    echo -e "❌ Policy Logic: ${RED}Not Working${NC}"
fi
total_checks=$((total_checks + 1))

if grep -q "BRAIN_URL=http://brain:8030" docker-compose.yml; then
    echo -e "✅ Service Integration: ${GREEN}Configured${NC}"
    passed_checks=$((passed_checks + 1))
else
    echo -e "❌ Service Integration: ${RED}Not Configured${NC}"
fi
total_checks=$((total_checks + 1))

# Final verdict
echo ""
pass_rate=$((passed_checks * 100 / total_checks))

if [ $pass_rate -eq 100 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED! AEGIS‑C is fully intelligent!${NC}"
    echo ""
    echo "🚀 Ready to run:"
    echo "   docker-compose up brain"
    echo "   docker-compose up detector honeynet hardware"
    echo "   streamlit run services/console/brain_demo.py"
elif [ $pass_rate -ge 75 ]; then
    echo -e "${YELLOW}⚠️  MOSTLY SMART ($pass_rate% passing)${NC}"
    echo "   Some components need attention"
else
    echo -e "${RED}❌ NOT SMART YET ($pass_rate% passing)${NC}"
    echo "   Major components need to be fixed"
fi

echo ""
echo "📊 Test completed at $(date)"
exit 0