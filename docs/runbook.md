# AEGIS‑C Operator Runbook

## Mission Rules of Engagement (ROE)

- **Strictly Defensive**: Detection, labeling, provenance verification only
- **No Unauthorized Access**: No disruption or sabotage of external systems
- **Privacy First**: No PII collection beyond legitimate logs; redact by default
- **Human‑on‑the‑Loop**: No automated operational actions without dual review

## Quick Reference

### Service Endpoints
- **Detector**: `localhost:8010` - AI artifact and agent detection
- **Fingerprinting**: `localhost:8011` - Model fingerprinting
- **Honeynet**: `localhost:8012` - Fake APIs with telemetry
- **Admission Control**: `localhost:8013` - Data poisoning prevention
- **Provenance**: `localhost:8014` - Content signing/verification
- **Console**: `localhost:8501` - Operations dashboard

### Triage Workflow

#### 1. Incoming Artifact Analysis
```
1. Paste/upload content to Console → Detection tab
2. Review detection score and signals
3. If AI‑generated (score ≥ 0.6):
   - Run fingerprinting probes
   - Check provenance if available
   - Create case file
4. If hostile agent suspected:
   - Route to honeynet for observation
   - Monitor for canary activations
```

#### 2. Data Admission Control
```
1. Bulk screen new datasets via Admission Control tab
2. Review quarantined samples manually
3. Approve/deny based on:
   - Anomaly score (> 0.7 = auto reject)
   - Suspicious patterns detected
   - Business context
4. Document decision rationale
```

#### 3. Provenance Verification
```
1. Check inbound content badges:
   - 🟢 Green = Verified provenance
   - 🟡 Yellow = Unknown provenance  
   - 🔴 Red = Verification failed
2. For 🔴 content:
   - Investigate source
   - Check for tampering
   - Consider quarantine
```

## Alert Levels

| Level | Score | Action | Timeline |
|-------|-------|--------|----------|
| 🟢 LOW | < 0.4 | Monitor | Routine |
| 🟡 MEDIUM | 0.4-0.6 | Investigate | Within 24h |
| 🟠 HIGH | 0.6-0.8 | Priority Review | Within 4h |
| 🔴 CRITICAL | > 0.8 | Immediate Action | < 1h |

## Decision Matrix

### Detection Verdicts
- **likely_ai** (≥ 0.7): Flag for review, fingerprint, consider quarantine
- **uncertain** (0.4-0.6): Additional analysis required
- **likely_human** (< 0.4): Proceed normally

### Admission Control
- **accept** (< 0.4): Clear for production use
- **quarantine** (0.4-0.7): Manual review required
- **reject** (> 0.7): Block and investigate

### Canary Detection
- **Canary Hit**: Immediate investigation
- **Pattern Match**: Review source IP and behavior
- **False Positive**: Update canary patterns

## Standard Operating Procedures

### SOP 1: AI Artifact Detection
1. **Input**: Text, code, or media content
2. **Processing**: Run through detector service
3. **Analysis**: Review score, signals, patterns
4. **Decision**: Accept/flag/quarantine based on matrix
5. **Documentation**: Log decision with rationale

### SOP 2: Model Fingerprinting
1. **Preparation**: Load current probe set
2. **Execution**: Present probes to target model
3. **Collection**: Record all responses
4. **Analysis**: Run fingerprinting service
5. **Attribution**: Document model family and confidence

### SOP 3: Honeynet Monitoring
1. **Setup**: Deploy fake endpoints with canaries
2. **Monitoring**: Watch telemetry for unusual activity
3. **Detection**: Alert on canary activations
4. **Analysis**: Profile attacker TTPs
5. **Response**: Document and update defenses

### SOP 4: Data Admission Control
1. **Screening**: Run all new data through admission control
2. **Review**: Manually inspect quarantined items
3. **Decision**: Approve clean data, reject suspicious
4. **Documentation**: Maintain audit trail of all decisions

### SOP 5: Provenance Verification
1. **Signing**: Sign all outbound content
2. **Verification**: Check all inbound content signatures
3. **Validation**: Confirm content integrity and source
4. **Action**: Flag or reject invalid provenance

## Incident Response

### Type 1: AI Poisoning Attempt
1. **Detection**: Admission control flags suspicious data
2. **Containment**: Quarantine affected datasets
3. **Analysis**: Identify attack vectors and patterns
4. **Recovery**: Clean datasets and update filters
5. **Prevention**: Update detection rules and baselines

### Type 2: Model Exfiltration
1. **Detection**: Canary token activation in honeynet
2. **Tracking**: Monitor source IP and behavior patterns
3. **Analysis**: Profile attacker capabilities and intent
4. **Defense**: Update honeypots and canary placement
5. **Reporting**: Document incident for threat intelligence

### Type 3: Provenance Tampering
1. **Detection**: Verification failure on signed content
2. **Investigation**: Analyze tampering method and extent
3. **Impact**: Assess affected content and systems
4. **Recovery**: Re-sign affected content
5. **Hardening**: Update signing processes and validation

## Metrics and KPIs

### Detection Performance
- **True Positive Rate**: ≥ 85% AI artifacts correctly identified
- **False Positive Rate**: ≤ 15% human content flagged as AI
- **Mean Time to Detect**: ≤ 90 seconds per artifact

### Admission Control
- **Poisoning Detection Rate**: ≥ 90% malicious samples caught
- **False Quarantine Rate**: ≤ 10% clean samples quarantined
- **Processing Speed**: ≤ 5 seconds per sample

### Honeynet Effectiveness
- **Canary Detection Rate**: 100% canary activations identified
- **Session Depth**: ≥ 3 requests per automated bot
- **Telemetry Coverage**: 100% requests logged and analyzed

### Provenance Operations
- **Signing Success Rate**: 100% outbound content signed
- **Verification Accuracy**: 100% signature validation
- **Tamper Detection**: 100% modifications identified

## Maintenance Tasks

### Daily
- Review all high-priority alerts
- Check service health and performance
- Update detection baselines with new data

### Weekly
- Analyze detection trends and patterns
- Update honeypot canaries (rotate tokens)
- Review and update probe sets
- Generate performance reports

### Monthly
- Comprehensive model retraining
- Update threat intelligence feeds
- Review and update ROE documentation
- Conduct security audit of all services

## Troubleshooting

### Common Issues

#### Service Unavailable
1. Check service health endpoints
2. Review service logs for errors
3. Restart affected services if needed
4. Verify network connectivity

#### False Positives
1. Review detection thresholds
2. Update baseline data
3. Fine-tune algorithm parameters
4. Document patterns for future reference

#### Performance Issues
1. Monitor resource utilization
2. Check database query performance
3. Review ML model inference times
4. Scale services if needed

#### Canary Fatigue
1. Rotate canary token patterns
2. Update honeypot data structures
3. Introduce new deception techniques
4. Monitor attacker adaptation

## Escalation Procedures

### Level 1: Operator
- Handle routine alerts and analysis
- Document all decisions and actions
- Escalate complex incidents to Level 2

### Level 2: Security Engineer  
- Investigate complex attacks
- Update detection algorithms
- Coordinate with threat intelligence

### Level 3: Management
- Approve major operational changes
- Handle compliance and legal issues
- Coordinate with external partners

## Contact Information

- **Security Operations**: soc@aegis-c.local
- **Engineering Support**: eng@aegis-c.local  
- **Management**: leadership@aegis-c.local
- **Emergency**: emergency@aegis-c.local

---

**Last Updated**: 2024-01-01
**Version**: 1.0.0
**Classification**: Internal Use Only