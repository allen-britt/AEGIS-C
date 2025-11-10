# AEGIS‑C Runbook

## Operational Procedures

### Service Management

#### Starting Services
```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d detector

# Start with custom environment
docker-compose --env-file .env.production up -d
```

#### Stopping Services
```bash
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop detector

# Force stop and remove volumes
docker-compose down -v
```

#### Service Health Checks
```bash
# Check all service health
docker-compose ps

# Check specific service logs
docker-compose logs -f detector

# Check service health endpoint
curl http://localhost:8010/health
```

### Monitoring and Alerting

#### Key Metrics to Monitor
- Service uptime and health status
- Request rates and response times
- Error rates and types
- Resource utilization (CPU, memory, GPU)
- Detection scores and verdicts
- Threat intelligence updates

#### Alert Thresholds
- Service health check failures: Alert immediately
- Error rate > 5%: Alert within 5 minutes
- Response time > 2 seconds: Alert within 10 minutes
- GPU temperature > 85°C: Alert immediately
- Detection score > 0.8: Alert immediately

#### Log Analysis
```bash
# View structured logs
docker-compose logs -f detector | jq '.'

# Filter by severity
docker-compose logs detector | grep 'ERROR'

# Monitor authentication failures
docker-compose logs detector | grep 'Unauthorized'
```

### Incident Response

#### Level 1: Service Degradation
**Symptoms**:
- Increased response times
- Elevated error rates
- Partial service unavailability

**Actions**:
1. Check service health endpoints
2. Review recent deployments
3. Check resource utilization
4. Restart affected services if needed

```bash
# Restart service
docker-compose restart detector

# Scale service if needed
docker-compose up -d --scale detector=2
```

#### Level 2: Security Incident
**Symptoms**:
- Authentication failures
- Suspicious request patterns
- High detection scores
- Data quarantine triggers

**Actions**:
1. Review security logs
2. Check threat intelligence feeds
3. Validate data integrity
4. Consider service isolation

```bash
# Check recent security events
curl http://localhost:8015/events?severity=high

# Review quarantined data
curl http://localhost:8013/quarantine/list
```

#### Level 3: Critical Infrastructure
**Symptoms**:
- Multiple service failures
- Data corruption suspected
- Infrastructure compromise
- Widespread authentication failures

**Actions**:
1. Activate incident response team
2. Isolate affected systems
3. Initiate disaster recovery
4. Preserve forensic evidence

```bash
# Emergency shutdown
docker-compose down

# Preserve logs for forensics
docker-compose logs > incident_logs_$(date +%Y%m%d_%H%M%S).log
```

### Maintenance Procedures

#### Routine Maintenance
**Daily**:
- Check service health status
- Review error logs
- Monitor resource utilization
- Verify threat intelligence updates

**Weekly**:
- Rotate API keys
- Update threat intelligence feeds
- Clean up old logs and temporary data
- Review performance metrics

**Monthly**:
- Update dependencies
- Review and update detection rules
- Backup configuration and data
- Security audit and penetration testing

#### Service Updates
```bash
# Update service without downtime
docker-compose pull detector
docker-compose up -d --no-deps detector

# Rolling update for multiple instances
docker-compose up -d --no-deps --scale detector=2 detector
docker-compose up -d --no-deps --scale detector=1 detector
```

#### Database Maintenance
```bash
# PostgreSQL maintenance
docker-compose exec postgres psql -U postgres -c "VACUUM ANALYZE;"

# Redis maintenance
docker-compose exec redis redis-cli FLUSHDB

# Backup databases
docker-compose exec postgres pg_dump -U postgres > backup_$(date +%Y%m%d).sql
```

### Troubleshooting

#### Common Issues

**Service Won't Start**
```bash
# Check service logs
docker-compose logs detector

# Check port conflicts
netstat -tulpn | grep 8010

# Validate configuration
docker-compose config
```

**High Memory Usage**
```bash
# Check memory usage
docker stats

# Restart service to clear memory
docker-compose restart detector

# Scale up resources
docker-compose up -d --scale detector=2
```

**Authentication Failures**
```bash
# Verify API key
curl -H "x-api-key: changeme-dev" http://localhost:8010/secure/ping

# Check environment variables
docker-compose exec detector env | grep API_KEY
```

**GPU Issues**
```bash
# Check GPU status
curl http://localhost:8016/status

# Monitor GPU metrics
watch -n 5 'curl -s http://localhost:8016/metrics | jq .'

# Execute hardware policy
curl -X POST "http://localhost:8016/policy/execute?action=reset&target_gpu=0"
```

### Performance Tuning

#### Service Optimization
- Adjust worker processes based on CPU cores
- Configure appropriate timeouts and retries
- Optimize database connection pools
- Enable caching for frequently accessed data

#### Resource Scaling
```bash
# Scale services horizontally
docker-compose up -d --scale detector=3 --scale fingerprint=2

# Monitor scaling effectiveness
docker-compose logs -f | grep "worker"
```

#### Database Optimization
```bash
# PostgreSQL tuning
docker-compose exec postgres psql -U postgres -c "ALTER SYSTEM SET shared_buffers = '256MB';"
docker-compose restart postgres

# Redis optimization
docker-compose exec redis redis-cli CONFIG SET maxmemory 512mb
```

### Security Operations

#### Access Control
```bash
# Rotate API keys
export API_KEY=$(openssl rand -hex 16)
docker-compose up -d

# Update console password
export CONSOLE_PWD=$(openssl rand -base64 12)
docker-compose up -d console
```

#### Security Monitoring
```bash
# Monitor authentication attempts
docker-compose logs | grep "401\|403"

# Check for suspicious patterns
curl http://localhost:8010/metrics | grep detection_scores

# Review threat intelligence
curl http://localhost:8018/analysis/ai-threats
```

### Backup and Recovery

#### Data Backup
```bash
# Backup all data volumes
docker run --rm -v aegis-c_pg:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres_backup_$(date +%Y%m%d).tar.gz -C /data .
docker run --rm -v aegis-c_redis:/data -v $(pwd):/backup ubuntu tar czf /backup/redis_backup_$(date +%Y%m%d).tar.gz -C /data .
```

#### Service Recovery
```bash
# Restore from backup
docker-compose down
docker run --rm -v aegis-c_pg:/data -v $(pwd):/backup ubuntu tar xzf /backup/postgres_backup_20240115.tar.gz -C /data
docker-compose up -d
```

### Escalation Procedures

#### When to Escalate
- Multiple services down simultaneously
- Security breach confirmed
- Data corruption suspected
- Performance degradation > 50%
- SLA violations imminent

#### Escalation Contacts
- **Level 1**: On-call engineer (immediate)
- **Level 2**: Engineering team lead (15 minutes)
- **Level 3**: Security team (30 minutes)
- **Level 4**: Management (1 hour)

#### Escalation Checklist
- [ ] Document incident start time
- [ ] Assess impact scope
- [ ] Notify appropriate stakeholders
- [ ] Implement containment measures
- [ ] Preserve forensic evidence
- [ ] Initiate recovery procedures
- [ ] Post-incident review

*This runbook will be updated regularly based on operational experience and incident data.*