#!/bin/bash
set -e

echo "Testing AEGIS‑C Brain Gateway..."

# Health check
echo "1. Health check..."
curl -sf localhost:8030/health >/dev/null
echo "✓ Health check passed"

# Test low risk scenario
echo ""
echo "2. Low risk assessment..."
LOW_RISK=$(curl -s localhost:8030/risk -H 'content-type: application/json' \
  -d '{"subject":"test:low","kind":"artifact","signals":[{"name":"ai_text_score","value":0.1}]}')

echo "Low risk result:"
echo "$LOW_RISK" | jq .

# Verify low risk classification
LOW_LEVEL=$(echo "$LOW_RISK" | jq -r .level)
if [[ "$LOW_LEVEL" != "info" && "$LOW_LEVEL" != "warn" ]]; then
    echo "❌ Expected low risk level 'info' or 'warn', got '$LOW_LEVEL'"
    exit 1
fi
echo "✓ Low risk classification correct"

# Test high risk hardware scenario
echo ""
echo "3. High risk hardware assessment..."
HW_RISK=$(curl -s localhost:8030/risk -H 'content-type: application/json' \
  -d '{"subject":"node:gpu0","kind":"hardware","signals":[{"name":"ecc_delta","value":1.2},{"name":"hardware_temp","value":0.8}]}')

echo "Hardware risk result:"
echo "$HW_RISK" | jq .

# Verify high risk probability
HW_PROB=$(echo "$HW_RISK" | jq -r .probability)
if (( $(echo "$HW_PROB < 0.7" | bc -l) )); then
    echo "❌ Expected high risk probability >= 0.7, got $HW_PROB"
    exit 1
fi
echo "✓ High risk probability correct"

# Test policy decision for high risk hardware
echo ""
echo "4. Policy decision for high risk hardware..."
HW_POLICY=$(curl -s localhost:8030/policy -H 'content-type: application/json' \
  -d "{\"subject\":\"node:gpu0\",\"kind\":\"hardware\",\"risk\":$HW_PROB,\"options\":[\"observe\",\"drain_node\",\"reset_gpu\",\"reattest\"]}")

echo "Hardware policy result:"
echo "$HW_POLICY" | jq .

# Verify appropriate action for high risk
HW_ACTION=$(echo "$HW_POLICY" | jq -r .action)
if [[ "$HW_ACTION" == "observe" ]]; then
    echo "❌ Expected action other than 'observe' for high risk, got '$HW_ACTION'"
    exit 1
fi
echo "✓ Hardware policy action appropriate: $HW_ACTION"

# Test policy decision for low risk
echo ""
echo "5. Policy decision for low risk..."
LOW_PROB=$(echo "$LOW_RISK" | jq -r .probability)
LOW_POLICY=$(curl -s localhost:8030/policy -H 'content-type: application/json' \
  -d "{\"subject\":\"test:low\",\"kind\":\"artifact\",\"risk\":$LOW_PROB,\"options\":[\"observe\",\"raise_friction\"]}")

echo "Low risk policy result:"
echo "$LOW_POLICY" | jq .

# Verify observe action for low risk
LOW_ACTION=$(echo "$LOW_POLICY" | jq -r .action)
if [[ "$LOW_ACTION" != "observe" ]]; then
    echo "❌ Expected 'observe' action for low risk, got '$LOW_ACTION'"
    exit 1
fi
echo "✓ Low risk policy action correct: $LOW_ACTION"

# Test response time (should be fast)
echo ""
echo "6. Response time test..."
START_TIME=$(date +%s%N)
curl -s localhost:8030/risk -H 'content-type: application/json' \
  -d '{"subject":"perf:test","kind":"artifact","signals":[{"name":"ai_text_score","value":0.5}]}' >/dev/null
END_TIME=$(date +%s%N)
RESPONSE_TIME=$((($END_TIME - $START_TIME) / 1000000))  # Convert to milliseconds

echo "Response time: ${RESPONSE_TIME}ms"
if [[ $RESPONSE_TIME -gt 100 ]]; then
    echo "❌ Response time > 100ms: ${RESPONSE_TIME}ms"
    exit 1
fi
echo "✓ Response time acceptable"

# Test error handling
echo ""
echo "7. Error handling test..."
# Test invalid request
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" localhost:8030/risk -H 'content-type: application/json' \
  -d '{"subject":"test","kind":"artifact"}')  # Missing signals

if [[ "$HTTP_CODE" != "400" ]]; then
    echo "❌ Expected HTTP 400 for missing signals, got $HTTP_CODE"
    exit 1
fi
echo "✓ Error handling correct"

echo ""
echo "🎉 All Brain Gateway tests passed!"
echo "✓ Service is healthy and responding correctly"
echo "✓ Risk assessment working with proper classification"
echo "✓ Policy decisions appropriate for risk levels"
echo "✓ Performance within acceptable limits"
echo "✓ Error handling working properly"

exit 0