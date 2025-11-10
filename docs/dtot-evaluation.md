# AEGIS‑C Developmental & Operational Test (DT/OT) Plan

## 1. Purpose
This document defines developmental testing (DT) and operational testing (OT) for AEGIS‑C. It specifies scenarios, evaluation criteria, pass/fail metrics, required tooling, and documentation outputs to certify readiness for production deployment under the mission Rules of Engagement.

## 2. Test Environments
| Environment | Description | Objectives |
|-------------|-------------|------------|
| DEV | Local developer machines or CI runners | Unit/integration tests per service |
| DT | Controlled staging environment mirroring production data flows (synthetic only) | Validate end-to-end workflows, tune thresholds |
| OT | Operator-run exercise using mission-realistic traffic, red-team campaigns, hardware telemetry | Demonstrate readiness, confirm SOPs, evaluate human-in-the-loop controls |

## 3. Test Roles
- **Test Director (TD):** Approves plans, ensures compliance with ROE.
- **Data Owner:** Provides sanitized datasets and verifies provenance.
- **Operators:** Execute console workflows, sign findings.
- **Red Team:** Runs Cold War offensive toolkit in OT.
- **Hardware Lead:** Operates hardware sentinel scenarios.

## 4. Developmental Testing (DT)
### 4.1 Service-Level Tests
| Service | Test | Procedure | Pass Criteria |
|---------|------|-----------|---------------|
| Detector | API health | `curl http://localhost:8010/health` | `ok=true` |
| Detector | Text AI detection | Post sample AI text; expect `verdict=likely_ai` threshold ≥ 0.6 | ≥ 0.6 score |
| Detector | Human baseline | Post human text; expect `verdict=likely_human` | ≤ 0.4 score |
| Fingerprint | Probe retrieval | GET `/probes`; ensure non-empty list | Status 200 + ≥ 4 probes |
| Fingerprint | Vector scoring | Submit exact matches; expect similarity ≥ 0.8 | Similarity ≥ 0.8, guess correct |
| Admission | Poison sample | Post `INJECT_TRIGGER` payload; expect `verdict=quarantine` | Score > 1.0 |
| Honeynet | Telemetry | GET dataset; ensure `CNRY-` tokens present in response | Response contains canary |
| Provenance | Sign/Verify | Upload sample file; verify computed hash equals signature | `verified=true` |
| Cold War Defense | Health & analytics | `curl http://localhost:8015/health`; run mock request | `ok=true`; endpoints return 200 |
| Hardware Sentinel | Scan | POST `/hardware/scan`; inventory includes CPU + GPU entries | ≥ 1 component, trust level reported |

### 4.2 Integration Tests
1. **`python test_platform.py`** – ensures detector, fingerprint, honeynet, admission, provenance respond correctly. **Pass:** script exits 0 and prints `ALL TESTS PASSED`.
2. **Cold War Campaign Dry Run:** Use offensive toolkit with stealth mode; confirm coldwar service records `defenses_breached_count` and console shows alert. **Pass:** Attack results available with ≥1 metric populated.
3. **Hardware Attack Simulation:** POST `/hardware/attack` targeting GPU memory exhaustion intensity 0.7. **Pass:** Response includes `success=true`, triggered defenses list, and recommendations.

### 4.3 Performance Benchmarks
- Detector response time ≤ 200 ms for text payloads (95th percentile).
- Admission control batch (100 samples) completes in ≤ 5 s.
- Hardware scan completes in ≤ 10 s on single GPU system.

## 5. Operational Testing (OT)
### 5.1 Scenario Matrix
| Scenario | Description | Primary Metrics | Pass Criteria |
|----------|-------------|-----------------|---------------|
| OT‑1 Detection | Synthetic adversary campaign hitting gateway, detector, admission | Detection precision, alert latency | ≥0.85 precision, alerts <30s latency |
| OT‑2 Honeynet Engagement | Red team uses offensive toolkit high-intensity mode | Session depth, canary triggers | ≥3 requests/session, ≥1 canary triggered |
| OT‑3 Data Pipeline Defense | Inject mix of clean/poison samples into admission queue | Quarantine rate | ≥90% of poisons quarantined, ≤5% false positives |
| OT‑4 Provenance Enforcement | Signed vs tampered assets ingested | Verification accuracy | 100% of signed pass, 100% tampered fail |
| OT‑5 Hardware Resilience | Induce simulated GPU thermal attack + ECC spike | Risk score reaction, policy activation | Risk ≥0.7 within 60s, adaptive defense triggered |
| OT‑6 Console Operations | Operators triage cases, approve actions | SOP compliance, audit trail completeness | All actions logged with dual review, 0 unauthorized actions |

### 5.2 Evaluation Procedures
1. **Pre-Test Checklist:** Ensure ROE compliance, sanitize datasets, verify logging.
2. **Execution:** Run each scenario sequentially; TD records metrics using console exports and service logs.
3. **Post-Test Review:** Compare metrics to pass criteria; collect operator feedback; file deviation reports.

## 6. Tooling & Data Requirements
- `scripts/load_test.py` (planned) for load generation.
- Offensive toolkit at `offensive/simple_dashboard.py` for red-team scenarios.
- Synthetic corpora for admission control (clean + poison sets).
- Hardware telemetry baseline (export from `/hardware/scan`).

## 7. Documentation Outputs
- DT Test Report: results matrix, issues, remediation plan.
- OT Test Report: scenario outcomes, operator feedback, readiness recommendation.
- Configuration baselines: service versions, thresholds, hardware inventory.
- Updated runbook with lessons learned.

## 8. Exit Criteria
AEGIS‑C moves to production only when:
- All DT and OT pass criteria met.
- No high-severity open issues; medium issues have mitigation plan.
- ROE compliance attestation signed by TD and mission owner.
- Hardware sentinel policies validated and active.

## 9. Change Management
Any future capability changes require revision of this DT/OT plan, regression testing using `test_platform.py`, and re-certification of affected components.
