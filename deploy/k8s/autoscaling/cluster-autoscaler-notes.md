# Cluster Autoscaler — Quantum Trader v2.0

**EPIC-K8S-002** | Production infrastructure hardening

## Overview

Cluster Autoscaler automatically adjusts Kubernetes cluster size based on pod resource requirements and PodDisruptionBudgets.

---

## Requirements

### Prerequisites

1. **Cloud Provider Support**
   - AWS: EKS with Auto Scaling Groups
   - GCP: GKE with node pools
   - Azure: AKS with VM Scale Sets

2. **IAM Permissions**
   - Ability to create/delete nodes
   - Read access to node groups
   - Tagging permissions

3. **Node Groups**
   - Minimum 3 nodes per group (for HA)
   - Proper instance types (compute-optimized for trading)
   - Network configuration (VPC, subnets)

---

## Node Group Configuration

### General Purpose Node Group

**Purpose**: Microservices (AI Engine, Execution, Risk, Portfolio)

```yaml
NodeGroup: quantum-trader-general
InstanceType: t3.medium (2 vCPU, 4 GB RAM)
MinSize: 3
MaxSize: 10
DesiredCapacity: 3
Labels:
  workload: general
  tier: microservices
Taints: None
```

### Stateful Node Group

**Purpose**: Redis Sentinel, Postgres HA

```yaml
NodeGroup: quantum-trader-stateful
InstanceType: r6i.large (2 vCPU, 16 GB RAM)
MinSize: 3
MaxSize: 5
DesiredCapacity: 3
Labels:
  workload: stateful
  tier: infrastructure
Taints:
  - key: stateful
    value: "true"
    effect: NoSchedule
```

### High-Performance Node Group (Optional)

**Purpose**: RL Training, Model Inference

```yaml
NodeGroup: quantum-trader-gpu
InstanceType: g5.xlarge (4 vCPU, 16 GB RAM, 1 GPU)
MinSize: 0
MaxSize: 3
DesiredCapacity: 0
Labels:
  workload: ml
  tier: training
  accelerator: gpu
Taints:
  - key: nvidia.com/gpu
    value: "true"
    effect: NoSchedule
```

---

## Cluster Autoscaler Deployment

### Minimal Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - name: cluster-autoscaler
        image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
        command:
        - ./cluster-autoscaler
        - --v=4
        - --cloud-provider=aws  # or gcp, azure
        - --skip-nodes-with-local-storage=false
        - --skip-nodes-with-system-pods=false
        - --balance-similar-node-groups
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/quantum-trader
        resources:
          requests:
            cpu: 100m
            memory: 300Mi
          limits:
            cpu: 200m
            memory: 500Mi
```

---

## Safe Eviction Policies

### PodDisruptionBudget Integration

Cluster Autoscaler **respects** PodDisruptionBudgets when scaling down:

- Redis Sentinel: `maxUnavailable: 1` (always keep 2/3 nodes)
- Postgres HA: `maxUnavailable: 1` (always keep 1/2 nodes)
- Microservices: `minAvailable: 1` (always keep at least 1 replica)

**See**: `deploy/k8s/pdb/` for all PDB definitions

### Scale-Down Configuration

```yaml
# Cluster Autoscaler ConfigMap
scale-down-enabled: true
scale-down-delay-after-add: 10m
scale-down-delay-after-delete: 10s
scale-down-delay-after-failure: 3m
scale-down-unneeded-time: 10m
scale-down-utilization-threshold: 0.5
```

**Recommendation**: Set `scale-down-unneeded-time: 10m` to avoid thrashing during volatile trading hours.

---

## Node Labels for Workload Placement

### Label Scheme

| Label | Values | Purpose |
|-------|--------|---------|
| `workload` | general / stateful / ml | Workload type |
| `tier` | microservices / infrastructure / training | Service tier |
| `zone` | us-east-1a / us-east-1b / us-east-1c | Availability zone |
| `instance-type` | t3.medium / r6i.large / g5.xlarge | Instance size |

### NodeSelector Examples

**AI Engine** (general workload):
```yaml
nodeSelector:
  workload: general
  tier: microservices
```

**Redis Sentinel** (stateful workload):
```yaml
nodeSelector:
  workload: stateful
  tier: infrastructure
tolerations:
- key: stateful
  operator: Equal
  value: "true"
  effect: NoSchedule
```

---

## Monitoring & Validation

### Metrics to Monitor

- `cluster_autoscaler_scaled_up_nodes_total`
- `cluster_autoscaler_scaled_down_nodes_total`
- `cluster_autoscaler_unschedulable_pods_count`
- `cluster_autoscaler_nodes_count` (by node group)

### Validation Commands

```bash
# Check autoscaler logs
kubectl logs -n kube-system -l app=cluster-autoscaler --tail=50

# Check node count
kubectl get nodes --show-labels

# Check pending pods (triggers scale-up)
kubectl get pods --all-namespaces --field-selector=status.phase=Pending

# Describe autoscaler status
kubectl describe configmap -n kube-system cluster-autoscaler-status
```

---

## Best Practices

### DO:
- Use PodDisruptionBudgets for all stateful services
- Set proper resource requests (triggers scale-up)
- Use node affinity for stateful workloads
- Monitor autoscaler metrics in Grafana
- Test scale-down during low-traffic hours

### DON'T:
- Scale down below minimum required nodes (3 for HA)
- Use autoscaler without PDBs (causes service disruption)
- Set resource requests too low (causes overcommitment)
- Ignore pending pods (indicates capacity issues)
- Mix workload types on same node group

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pods stuck in Pending | Check node capacity, resource requests, and autoscaler logs |
| Nodes not scaling down | Verify PDBs not blocking eviction, check local storage |
| Frequent scale-up/down | Increase `scale-down-unneeded-time` to 15-20m |
| Stateful pods evicted | Add node taints + tolerations, use `maxUnavailable: 1` |

---

## Related Documentation

- **PodDisruptionBudgets**: `deploy/k8s/pdb/`
- **HPA Configuration**: `deploy/k8s/autoscaling/`
- **Node Resource Limits**: `deploy/k8s/services/`

---

**Updated**: December 4, 2025 | **EPIC-K8S-002** | Quantum Trader v2.0
