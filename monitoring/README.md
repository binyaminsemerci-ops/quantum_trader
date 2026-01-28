Reconcile Close Monitoring

Overview
- Grafana dashboard: monitoring/grafana/reconcile_close_dashboard.json
- Prometheus alert rules: monitoring/prometheus/rules/reconcile_close.rules.yml

Import Instructions
- Grafana: Import JSON via Dashboards → Import, select the file, set datasource to Prometheus.
- Prometheus: Add the rules file under your `rule_files` in prometheus.yml, then reload Prometheus.

Key Metrics
- p34_reconcile_close_published_total: plans published by P3.4
- reconcile_close_consumed_total: plans consumed by Apply Layer
- reconcile_close_executed_total{status}: success/rate_limit/error statuses
- reconcile_close_rejected_total{reason}: duplicate/guardrails
- reconcile_close_guardrail_fail_total{rule}: invariant violations
- quantum_apply_dedupe_hits_total: dedup hit count
- quantum_apply_last_success_epoch: last successful execution timestamp

Alerts
- GuardrailFailures: any guardrail failure in 5m
- ExecutionLag: no successful execution in >15m
- PublishedVsExecutedDrop: executed/published ratio <50% over 10m
- DedupSpike: >10 dedup hits over 10m
