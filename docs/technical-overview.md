# AEGIS‑C Technical Overview

## 1. Purpose
AEGIS‑C (Adversarial‑AI Engagement, Guarding, Intelligence & Shielding — Counter) provides a lawful, mission‑safe counter‑AI platform. The system detects adversary AI activity, applies friction, diverts hostile agents, protects our own pipelines, and enforces provenance with human oversight.

## 2. Rules of Engagement
- **Defensive only** – observe, label, verify, deceive with synthetic content; no destructive actions.
- **Privacy first** – log only operationally necessary metadata; redact by default.
- **Human‑on‑the‑loop** – no autonomous operational steps without signed provenance and dual review.

## 3. Reference Architecture
```
                           ┌───────────────────────────────────┐
                           │           Counter‑AI UI           │
                           │  (Ops Console / Reports / API)    │
                           └───────────────┬───────────────────┘
                                           │
                          Alerts/Findings  │  Tasks/Approvals
                                           │
┌──────────────────────────┼────────────────┼─────────────────────────┐
│ Ingress Layer            │ Analysis Layer │ Protection/Deception    │
│ (Gate & Sense)           │ (Detect/Attrib)│ (Deter/Degrade/Learn)   │
├──────────────────────────┼────────────────┼─────────────────────────┤
│ ■ Gateway & Friction     │ ■ Artifact      │ ■ Honeynet API + KB     │
│   (CAPTCHA, canaries,    │   Detector      │   (fake open data,      │
│   moving‑target DOM)     │   (text/img/    │   instrumented)         │
│                          │    audio/code)  │ ■ Canary propagation    │
│ ■ Provenance Verifier    │ ■ Agent/Bot     │   & beacon checks       │
│   (C2PA, sig)            │   Profiler      │ ■ Moving‑target layout  │
│                          │   (timing,      │   templates             │
│ ■ Probe Orchestrator     │    lat/entropy) │                         │
│   (challenges)           │ ■ Fingerprinting│ ■ Data Admission Control │
│                          │   (challenge    │   (guardrails for our   │
│                          │   response sim) │   training/RAG)         │
├──────────────────────────┴────────────────┴─────────────────────────┤
│                       Data & Telemetry Services                      │
│   Event Bus (NATS) • Feature Store • Object Store • Graph DB (Neo4j) │
│   • Postgres (case data) • Redis (caching) • MinIO (artifacts)       │
└───────────────────────────────────────────────────────────────────────┘
```

## 4. Service Catalog
| Service | Port | Purpose | Key Tech |
|---------|------|---------|----------|
| `gateway` | 8000 (planned) | Reverse proxy, friction, canaries | FastAPI + middleware |
| `detector` | 8010 | Text & agent heuristics, AI artifact detection | FastAPI, heuristics, ML |
| `fingerprint` | 8011 | Challenge probes, similarity scoring | FastAPI, NumPy |
| `honeynet` | 8012 | Deception APIs with telemetry | FastAPI |
| `admission` | 8013 | Data poisoning guard, quarantine workflow | FastAPI, scikit‑learn |
| `provenance` | 8014 | C2PA signing & verification (stub) | FastAPI, hashlib |
| `coldwar` | 8015 | Campaign analytics & defense intelligence | FastAPI |
| `hardware` | 8016 | Hardware security sentinel (GPU/CPU) | FastAPI, GPUtil, psutil |
| `console` | 8501 | Streamlit operations console | Streamlit |
| `offensive toolkit` | 8502 | Streamlit red‑team simulator | Streamlit |
| `tests/test_platform.py` | — | Health & regression testing | Python unittest |

## 5. Data Flows
1. **Ingress** – Operators feed artifacts or agents arrive via gateway. Provenance checks ensure authenticity prior to analysis.
2. **Detection & Attribution** – Detector scores inputs; fingerprint service challenges suspect models; coldwar service correlates campaigns.
3. **Protection & Deception** – Admission control quarantines suspect data; honeynet diverts hostile agents while logging telemetry.
4. **Hardware Sentinel** – Hardware service inventories GPUs/CPUs, monitors telemetry, simulates attacks, and generates adaptive defenses.
5. **Offensive Simulation** – Offensive toolkit executes controlled multi‑vector campaigns against local services for readiness testing.
6. **Console** – Presents health, alerts, campaign analytics, provenance results, hardware posture, and operator playbooks.

## 6. Deployment & Operations
- **Local Development** – Use `start.bat`/`start.sh` or run each service via `uvicorn`. Streamlit console & offensive toolkit launched separately.
- **Containerization** – `docker-compose.yml` provisions Postgres, Redis, Neo4j, MinIO, NATS for production parity.
- **Environments** – DEV (component testing), DT (developmental testing), OT (operational testing), PROD (controlled ops). DT/OT test plan is documented separately.
- **Monitoring** – Each FastAPI service exposes `/health`. Coldwar campaign telemetry and hardware sentinel metrics feed the console.

## 7. Security Controls
- **Network Segmentation** – Expose services internally; gateway front‑doors external traffic.
- **Access Control** – Streamlit console behind VPN or SSO in production; provenance signatures required before execution.
- **Logging** – Honeynet logs requests with canary tokens; coldwar & hardware services log attack/defense results; auditing defined in `docs/runbook.md`.
- **Hardware Policies** – Enforce ECC, monitor thermal/memory anomalies, support adaptive defenses (quotas, throttling, resets).

## 8. Compliance & Governance
- Aligns with mission ROE; any deception content remains synthetic.
- Provenance enforcement ensures human approval prior to operational output.
- DT/OT criteria ensure readiness before deployment (see `docs/dtot-evaluation.md`).

## 9. Extensibility Roadmap
- Replace provenance stub with full C2PA integration and HSM key storage.
- Expand detector to multimodal ML models and ensemble scoring.
- Integrate Next.js console with RBAC and case management.
- Build automated hardware risk scorer with Bayesian model and bandit policy engine.

## 10. Document Map
- `docs/technical-overview.md` (this file)
- `docs/threat-model.md`
- `docs/api-contracts.md`
- `docs/runbook.md`
- `docs/dtot-evaluation.md` (DT/OT plan)
- `README.md` (quick start)
