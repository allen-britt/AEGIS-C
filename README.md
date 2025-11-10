# AEGIS‑C (Adversarial‑AI Engagement, Guarding, Intelligence & Shielding — Counter)

AEGIS‑C is a lawful counter‑AI platform that detects adversary AI activity, applies defensive friction, diverts hostile agents, protects mission data pipelines, and enforces provenance with human oversight.

## Mission Rules of Engagement (ROE)
- Defensive monitoring, labeling, provenance verification, and deception with synthetic content only
- No unauthorized access, disruption, or sabotage
- Privacy first: redact PII by default
- Human‑on‑the‑loop: operational actions require signed provenance and dual review

## Architecture at a Glance
- **Ingress & Friction:** gateway (planned), provenance verifier, challenge probes
- **Analysis:** Detector (8010), Fingerprinting (8011), Cold War Defense analytics (8015)
- **Intelligence:** Intelligence Service (8018), Vulnerability Database (8019) with real-time threat feeds
- **Protection & Deception:** Admission control (8013), Honeynet (8012), Hardware sentinel (8016)
- **Purple Team:** Discovery service (8017) for reconnaissance and offensive simulation
- **Operators:** Streamlit console (8501) and offensive toolkit (8502) for red-team validation
- **Backends:** docker-compose provides Postgres, Redis, Neo4j, MinIO, NATS for production parity
- **Threat Intelligence:** NVD, Exploit-DB, GitHub Advisory integration with AI-specific vulnerability tracking
- Detailed design: `docs/technical-overview.md` & `docs/threat-model.md`

## Quick Start (Local)
```powershell
# From repo root (Windows)
python -m venv venv_windows
venv_windows\Scripts\pip install -r requirements.txt  # if consolidated file exists

# Start core FastAPI services (each in new terminal or via provided scripts)
venv_windows\Scripts\python -m uvicorn services.detector.main:app --port 8010 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.fingerprint.main:app --port 8011 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.honeynet.main:app --port 8012 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.admission.main:app --port 8013 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.provenance.main:app --port 8014 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.coldwar.main:app --port 8015 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.hardware.main:app --port 8016 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.discovery.main:app --port 8017 --host 0.0.0.0

# Start intelligence services (NEW!)
venv_windows\Scripts\python -m uvicorn services.intelligence.main:app --port 8018 --host 0.0.0.0
venv_windows\Scripts\python -m uvicorn services.vuln_db.main:app --port 8019 --host 0.0.0.0

# Launch operator console
venv_windows\Scripts\python -m streamlit run services\console\app.py --server.port 8501 --server.address 0.0.0.0

# Optional: launch offensive toolkit
venv_windows\Scripts\python -m streamlit run offensive\simple_dashboard.py --server.port 8502 --server.address 0.0.0.0
```
Scripts `start.bat` / `stop.bat` and `start.sh` / `stop.sh` orchestrate services automatically.

## Verification
```powershell
# Run end-to-end service checks
venv_windows\Scripts\python test_platform.py

# Test intelligence integration (NEW!)
venv_windows\Scripts\python test_intelligence_integration.py
```
Expected output includes `ALL TESTS PASSED` when detector/fingerprint/honeynet/admission/provenance services are healthy.

## 🧠 Intelligence Integration (NEW!)
AEGIS‑C now features real-time threat intelligence integration:

### **Threat Intelligence Sources**
- **NVD (National Vulnerability Database)**: Real CVE data with AI-specific filtering
- **Exploit-DB**: Known exploits and attack techniques
- **GitHub Security Advisories**: Python package vulnerabilities affecting AI/ML libraries
- **Custom Threat Feeds**: AI-specific attack patterns and indicators

### **Enhanced Detection Capabilities**
- **Vulnerability Correlation**: Cross-reference detected content with known CVEs
- **Threat Pattern Recognition**: Identify prompt injection, data poisoning, model extraction attempts
- **Risk Scoring**: Dynamic scoring based on current threat landscape
- **Automated Recommendations**: Security actions based on detected threats

### **Intelligence Services**
- **Intelligence Service (8018)**: Collects and processes threat data
- **Vulnerability Database (8019)**: Manages CVE and exploit information
- **Real-time Updates**: Automatic threat feed updates every 30-60 minutes

## Key Services & Ports
| Service | Port | Description |
|---------|------|-------------|
| `services/detector` | 8010 | AI artifact & agent detection (text heuristics + HTTP profiling) |
| `services/fingerprint` | 8011 | Challenge probes + similarity scoring |
| `services/honeynet` | 8012 | Deception APIs with canary telemetry |
| `services/admission` | 8013 | Data poisoning guard & quarantine |
| `services/provenance` | 8014 | C2PA signing/verification stub |
| `services/coldwar` | 8015 | Campaign analytics & defense intelligence |
| `services/hardware` | 8016 | Hardware security sentinel (inventory, attack sim, adaptive defenses) |
| `services/discovery` | 8017 | Purple Team reconnaissance and offensive simulation |
| `services/intelligence` | 8018 | **NEW!** Threat intelligence collection and analysis |
| `services/vuln_db` | 8019 | **NEW!** Vulnerability database with CVE/exploit tracking |
| `services/console` | 8501 | Streamlit counter‑AI operations console |
| `offensive/simple_dashboard.py` | 8502 | Red-team campaign simulator |

## Documentation Map
- `docs/technical-overview.md` – architecture, workflows, security controls
- `docs/threat-model.md` – adversary analysis and mitigations
- `docs/api-contracts.md` – service endpoint specifications
- `docs/runbook.md` – operator SOPs and escalation paths
- `docs/dtot-evaluation.md` – developmental & operational test plan

## Next Steps / Roadmap
- Implement gateway service with CAPTCHAs and moving-target layouts
- Upgrade provenance service to full C2PA integration with HSM
- Extend detector to multimodal AI artifacts and ML ensembles
- Add RBAC-enabled Next.js console and case management backend

For mission onboarding, review ROE, run DT/OT tests per `docs/dtot-evaluation.md`, and track hardware posture via the console’s Hardware Security tab.
