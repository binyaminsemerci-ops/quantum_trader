#!/bin/bash
# P1.2: Add Safety Telemetry scrape job to Prometheus

cat >> /etc/prometheus/prometheus.yml << 'EOF'

  # P1.2: Safety Telemetry Exporter
  - job_name: quantum_safety_telemetry
    static_configs:
      - targets: [localhost:9105]
        labels:
          service: safety-telemetry
          component: quantum-trader
          environment: production
    scrape_interval: 15s
    scrape_timeout: 10s
EOF

echo "✅ Scrape job added to prometheus.yml"
