# Quantum Trader Kubernetes Deployment

**EPIC-K8S-001**: Kubernetes Infrastructure v2.0  
**Status**: ✅ Phase 1 Complete (Manifests & Structure)

---

## 📋 Overview

This directory contains Kubernetes manifests for deploying Quantum Trader v2.0 as a microservices architecture.

**Architecture**: 8 microservices + 2 infrastructure services (Redis, Postgres)

---

## 🏗️ Structure

```
deploy/k8s/
├── namespace.yaml                    # quantum-trader namespace
├── config/
│   └── configmap-appsettings.yaml   # App configuration (non-secret)
├── secrets/
│   └── secret-exchanges-example.yaml # Placeholder for API keys
├── redis/
│   ├── redis-deployment.yaml         # Redis cache (EventBus/PolicyStore)
│   └── redis-service.yaml
├── postgres/
│   ├── postgres-deployment.yaml      # PostgreSQL database
│   └── postgres-service.yaml
├── services/
│   ├── ai-engine-deployment.yaml     # AI Engine (port 8001)
│   ├── ai-engine-service.yaml
│   ├── execution-deployment.yaml     # Execution Service (port 8002)
│   ├── execution-service.yaml
│   ├── risk-safety-deployment.yaml   # Risk & Safety (port 8003)
│   ├── risk-safety-service.yaml
│   ├── portfolio-deployment.yaml     # Portfolio Intelligence (port 8004)
│   ├── portfolio-service.yaml
│   ├── rl-training-deployment.yaml   # RL Training (port 8006)
│   ├── rl-training-service.yaml
│   ├── monitoring-deployment.yaml    # Monitoring & Health (port 8005)
│   ├── monitoring-service.yaml
│   ├── federation-ai-deployment.yaml # Federation AI v3 (port 8007)
│   ├── federation-ai-service.yaml
│   ├── gateway-deployment.yaml       # API Gateway (port 8000)
│   └── gateway-service.yaml
└── ingress/
    └── ingress-gateway.yaml          # NGINX Ingress (quantum.local)
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites

- **Kubernetes cluster**: [kind](https://kind.sigs.k8s.io/), [minikube](https://minikube.sigs.k8s.io/), or [Docker Desktop with K8s](https://docs.docker.com/desktop/kubernetes/)
- **kubectl**: [Install kubectl](https://kubernetes.io/docs/tasks/tools/)
- **NGINX Ingress Controller**: [Installation guide](https://kubernetes.github.io/ingress-nginx/deploy/)

### Step 1: Create Local Cluster (kind example)

```bash
# Create cluster with ingress support
cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: quantum-trader
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
EOF

# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
```

### Step 2: Configure Secrets

**⚠️ IMPORTANT**: Copy example secrets and fill with real values:

```bash
# Copy example to real secret file
cp deploy/k8s/secrets/secret-exchanges-example.yaml deploy/k8s/secrets/secret-exchanges.yaml

# Edit with your real API keys
# NEVER commit secret-exchanges.yaml to Git!
code deploy/k8s/secrets/secret-exchanges.yaml
```

**Add to .gitignore**:
```bash
echo "deploy/k8s/secrets/secret-exchanges.yaml" >> .gitignore
```

### Step 3: Deploy Infrastructure

```bash
# 1. Create namespace
kubectl apply -f deploy/k8s/namespace.yaml

# 2. Apply ConfigMap
kubectl apply -f deploy/k8s/config/configmap-appsettings.yaml

# 3. Apply Secrets (after filling real values!)
kubectl apply -f deploy/k8s/secrets/secret-exchanges.yaml

# 4. Deploy Redis & Postgres
kubectl apply -f deploy/k8s/redis/
kubectl apply -f deploy/k8s/postgres/

# Wait for infrastructure to be ready
kubectl wait --for=condition=ready pod -l app=redis -n quantum-trader --timeout=60s
kubectl wait --for=condition=ready pod -l app=postgres -n quantum-trader --timeout=60s
```

### Step 4: Deploy Services

```bash
# Deploy all microservices
kubectl apply -f deploy/k8s/services/

# Wait for services to be ready (may take 2-3 minutes)
kubectl wait --for=condition=ready pod -l tier=backend -n quantum-trader --timeout=300s
```

### Step 5: Deploy Ingress

```bash
# Apply Ingress
kubectl apply -f deploy/k8s/ingress/

# Add to /etc/hosts (or C:\Windows\System32\drivers\etc\hosts on Windows)
echo "127.0.0.1 quantum.local" | sudo tee -a /etc/hosts
```

### Step 6: Verify Deployment

```bash
# Check all pods
kubectl get pods -n quantum-trader

# Check services
kubectl get svc -n quantum-trader

# Check ingress
kubectl get ingress -n quantum-trader

# Test API
curl http://quantum.local/health
```

---

## 🔍 Monitoring & Debugging

### View Logs

```bash
# AI Engine logs
kubectl logs -f deployment/ai-engine -n quantum-trader

# Execution Service logs
kubectl logs -f deployment/execution -n quantum-trader

# All backend services
kubectl logs -f -l tier=backend -n quantum-trader --max-log-requests=8
```

### Pod Status

```bash
# Get pod details
kubectl get pods -n quantum-trader -o wide

# Describe pod (troubleshooting)
kubectl describe pod <pod-name> -n quantum-trader

# Execute commands in pod
kubectl exec -it <pod-name> -n quantum-trader -- /bin/bash
```

### Port Forwarding (Bypass Ingress)

```bash
# Access AI Engine directly
kubectl port-forward svc/ai-engine-service 8001:8001 -n quantum-trader

# Access Gateway directly
kubectl port-forward svc/gateway-service 8000:8000 -n quantum-trader

# Access Redis
kubectl port-forward svc/redis 6379:6379 -n quantum-trader
```

---

## 🧪 Testing

### Dry-Run Validation

```bash
# Validate all manifests without applying
kubectl apply --dry-run=client -f deploy/k8s/namespace.yaml
kubectl apply --dry-run=client -f deploy/k8s/config/
kubectl apply --dry-run=client -f deploy/k8s/redis/
kubectl apply --dry-run=client -f deploy/k8s/postgres/
kubectl apply --dry-run=client -f deploy/k8s/services/
kubectl apply --dry-run=client -f deploy/k8s/ingress/
```

### Health Checks

```bash
# Check all health endpoints
for svc in ai-engine execution risk-safety portfolio rl-training monitoring federation-ai gateway; do
  echo "Checking $svc..."
  kubectl exec -n quantum-trader deployment/$svc -- curl -s http://localhost:800X/health || echo "FAILED"
done
```

---

## 🔧 Configuration

### ConfigMap (config/configmap-appsettings.yaml)

Non-sensitive configuration:
- Service discovery (Redis, Postgres URLs)
- Trading parameters (confidence thresholds, position sizing)
- AI model configuration
- Monitoring intervals

### Secrets (secrets/secret-exchanges.yaml)

Sensitive data:
- Exchange API keys (Binance, Bybit, OKX)
- Database credentials
- Admin tokens

**⚠️ Security**: 
- Never commit secrets to Git
- Use separate secrets per environment (dev/staging/prod)
- Consider using [External Secrets Operator](https://external-secrets.io/) for production

---

## 📊 Resource Requirements

### Minimum Cluster Resources

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit | Replicas |
|-----------|-------------|-----------|----------------|--------------|----------|
| AI Engine | 500m | 2000m | 1Gi | 2Gi | 2 |
| Execution | 250m | 1000m | 512Mi | 1Gi | 2 |
| Risk/Safety | 250m | 1000m | 512Mi | 1Gi | 2 |
| Portfolio | 250m | 1000m | 512Mi | 1Gi | 1 |
| RL Training | 500m | 2000m | 1Gi | 2Gi | 1 |
| Monitoring | 100m | 500m | 256Mi | 512Mi | 1 |
| Federation AI | 250m | 1000m | 512Mi | 1Gi | 1 |
| Gateway | 250m | 1000m | 512Mi | 1Gi | 2 |
| Redis | 100m | 500m | 256Mi | 512Mi | 1 |
| Postgres | 250m | 1000m | 512Mi | 1Gi | 1 |

**Total Minimum**: ~3 CPU cores, ~8 GB RAM

---

## 🗄️ Persistent Storage

### PersistentVolumeClaims

- `models-pvc` (10Gi): AI model files (ReadWriteMany)
- `logs-pvc` (5Gi): Application logs (ReadWriteMany)
- `data-pvc` (20Gi): Training data & backtest results (ReadWriteMany)
- `redis-pvc` (5Gi): Redis persistence (ReadWriteOnce)
- `postgres-pvc` (20Gi): Database storage (ReadWriteOnce)

**Note**: ReadWriteMany requires NFS or cloud provider storage (EFS, Azure Files, GCE PD)

---

## 🔄 Scaling

### Manual Scaling

```bash
# Scale AI Engine to 3 replicas
kubectl scale deployment ai-engine -n quantum-trader --replicas=3

# Scale Execution Service to 4 replicas
kubectl scale deployment execution -n quantum-trader --replicas=4
```

### Auto-Scaling (Future: EPIC-K8S-002)

Horizontal Pod Autoscaler (HPA) will be configured in Phase 2:
```yaml
# Example HPA (not yet implemented)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-engine-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-engine
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 🚨 Troubleshooting

### Pod CrashLoopBackOff

```bash
# Check pod events
kubectl describe pod <pod-name> -n quantum-trader

# Check logs from crashed container
kubectl logs <pod-name> -n quantum-trader --previous

# Common causes:
# 1. Missing secrets (check secret-exchanges.yaml)
# 2. Redis/Postgres not ready (check dependencies)
# 3. Image pull errors (check imagePullPolicy)
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints -n quantum-trader

# Check ingress
kubectl describe ingress quantum-trader-ingress -n quantum-trader

# Check NGINX Ingress logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

### Database Connection Issues

```bash
# Test Postgres connection
kubectl exec -it deployment/postgres -n quantum-trader -- psql -U quantum -d quantum_trader

# Test Redis connection
kubectl exec -it deployment/redis -n quantum-trader -- redis-cli ping
```

---

## 🔐 Security Best Practices

1. **Secrets Management**:
   - Use Kubernetes Secrets (current)
   - Upgrade to [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets) (Phase 2)
   - Integrate with [HashiCorp Vault](https://www.vaultproject.io/) (Production)

2. **Network Policies**:
   - Restrict pod-to-pod communication
   - Allow only necessary service access
   - Deny external access to infrastructure (Redis, Postgres)

3. **RBAC**:
   - Create service accounts per microservice
   - Limit permissions (principle of least privilege)
   - Audit access logs

4. **Image Security**:
   - Use minimal base images (alpine)
   - Scan images for vulnerabilities ([Trivy](https://github.com/aquasecurity/trivy))
   - Sign images with [Cosign](https://github.com/sigstore/cosign)

---

## 📝 Next Steps (EPIC-K8S-002)

See `EPIC_K8S_001_COMPLETION.md` for full roadmap:

1. **High Availability (HA)**:
   - Redis Sentinel (3 replicas)
   - Postgres Replication (primary + 2 replicas)
   - Multi-AZ deployment

2. **Auto-Scaling**:
   - Horizontal Pod Autoscaler (HPA)
   - Vertical Pod Autoscaler (VPA)
   - Cluster Autoscaler

3. **Observability**:
   - Prometheus + Grafana stack
   - Distributed tracing (Jaeger)
   - Centralized logging (ELK/Loki)

4. **CI/CD Integration**:
   - GitHub Actions → Build Docker images
   - Automated deployments (ArgoCD/Flux)
   - GitOps workflow

5. **Multi-Region**:
   - Global load balancing
   - Data replication
   - Disaster recovery

---

## 📚 References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [kind Quick Start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

---

**Status**: ✅ Ready for local dev/staging deployment  
**Version**: v2.0 (EPIC-K8S-001)  
**Last Updated**: 2024-12-04
