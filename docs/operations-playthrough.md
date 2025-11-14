# AEGIS-C Operations Playthrough

A practical, end-to-end walkthrough that takes you from a cold start to a fully operational AEGIS-C platform. Follow these steps sequentially to verify the installation, launch every service, exercise the unified console, and shut everything down gracefully.

---

## 1. Environment Preparation

### 1.1 Clone and enter the repo
```powershell
cd C:\path\to\workspace
git clone git@github.com:allen-britt/AEGIS-C.git
cd AEGIS-C
```
> **Expectation:** Repository downloads and the working tree is clean (`git status` shows no changes).

### 1.2 Optional: Create a Python virtual environment (recommended)
```powershell
python -m venv aegis-env
./aegis-env/Scripts/Activate.ps1
```
> **Expectation:** Prompt shows `(aegis-env)` prefix. `python --version` returns the interpreter inside the venv.

### 1.3 Install Python dependencies once
```powershell
pip install -r requirements.txt
```
> **Expectation:** Ends without errors. Key packages such as `fastapi`, `uvicorn`, `streamlit`, `requests`, and `structlog` are installed.

---

## 2. Single-Script Launch (Recommended)

### 2.1 Start everything with the unified launcher
```powershell
python launch.py
```
> **Behind the scenes:**
> - Validates core Python dependencies and reports optional ones (Plotly, pandas, pynvml).
> - Skips Docker infrastructure automatically if Docker Desktop is not available.
> - Starts each microservice (Brain, Detector, Honeynet, Hardware, etc.) and waits for `/health` to return HTTP 200.
> - Launches the unified Streamlit UI on `http://localhost:8500`.

> **Expectation:**
> - Terminal prints the ASCII banner, followed by `✅` status lines while services start.
> - Final summary shows all services with `✅` for healthy ones.
> - Browser accessible at [http://localhost:8500](http://localhost:8500).

### 2.2 Unified UI first look
- Navigate to **Service Health** in the sidebar: Brain, Detector, and others should show green cards.
- Under **🧠 Intelligence**, click **Assess Risk** with default sliders. Expect a probability ~0.30 and a level label.
- Click **⚖️ Get Policy Action** to see the Brain Gateway recommend an action (e.g., `observe` or `raise_friction`).

> **Expectation:** API calls succeed; any missing service is flagged as `❌` with actionable error text.

---

## 3. Manual Service Playthrough (Advanced)

If you want to reproduce what the launcher does under the hood, follow these steps in separate PowerShell windows (or background jobs):

### 3.1 Start the Brain Gateway first
```powershell
python -m uvicorn services.brain.main:app --port 8030 --host localhost
```
> **Expectation:** Console logs show `Application startup complete.` `http://localhost:8030/health` returns JSON with `{ "ok": true }`.

### 3.2 Start critical detection services
```powershell
python -m uvicorn services.detector.main:app --port 8010 --host localhost
python -m uvicorn services.fingerprint.main:app --port 8011 --host localhost
python -m uvicorn services.honeynet.main:app --port 8012 --host localhost
python -m uvicorn services.admission.main:app --port 8013 --host localhost
python -m uvicorn services.hardware.main:app --port 8016 --host localhost
```
> **Expectation:** Each process binds its port without error. Hitting `/health` on each port returns `200` within a couple of seconds.

### 3.3 Launch the unified console manually
```powershell
python -m streamlit run services\console\unified_app.py --server.port 8500 --server.address localhost
```
> **Expectation:** Streamlit logs a URL (`Local URL: http://localhost:8500`). Opening it shows the same UI as the automated launcher.

---

## 4. Verification & Smoke Tests

### 4.1 Brain Gateway smoke test
```powershell
bash tests/test_brain.sh
```
> **Expectation:** Script prints green checkmarks for health, low-risk, and high-risk scenarios. Fails fast if the Brain Gateway is unreachable.

### 4.2 Intelligence verification sweep
```powershell
bash verify-smart.sh
```
> **Expectation:** Runs a longer integration sequence: risk/policy calls, hardware anomaly checks, stubbed services, and console integration. All sections should report `PASS`.

> **Tip:** On Windows, run via Git Bash or WSL. Alternatively, translate commands with PowerShell `Invoke-RestMethod` equivalents as documented in `INTELLIGENCE.md`.

---

## 5. Operational Tips

- **Monitoring:** Keep the launcher terminal open. It prints a heartbeat every 30 s listing any unhealthy services.
- **Optional dependencies:**
  - `pip install plotly pandas` enables interactive charts in the Analytics tab.
  - `pip install pynvml` activates real GPU telemetry instead of mock data.
- **API key security:** Default `API_KEY=changeme-dev`. Set secure values via environment variables before production trials.
- **Docker Services:** If Docker Desktop is running, re-run `python launch.py` to automatically start PostgreSQL, Redis, and Neo4j via `docker-compose up -d`.

---

## 6. Shutdown Procedure

1. In the launcher terminal, press **Ctrl+C** once.
2. Wait for `🛑 Stopping all services...` messages to show each service terminated cleanly.
3. Close any remaining Streamlit windows or terminals.

> **Expectation:** No lingering Python or Streamlit processes remain (`Get-Process python` returns empty).

---

## 7. Troubleshooting Checklist

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `ModuleNotFoundError: structlog` | Requirements not installed in current interpreter | `pip install -r requirements.txt` (ensure venv active) |
| Streamlit shows blank or “Cannot connect” | Service not started / wrong host binding | Verify Brain (8030) and other ports respond on `localhost` |
| Launcher prints `⚠️  Docker not available` | Docker Desktop not running or not installed | Start Docker Desktop or ignore (infra auto-skipped) |
| Port 8500 already in use | Another Streamlit instance running | `Stop-Process -Name streamlit` then relaunch |

---

You now have a repeatable operations runbook for bringing AEGIS-C online, validating the intelligence core, and standing the platform down. Refer back here whenever you need a quick refresher or want to hand off to another operator.
