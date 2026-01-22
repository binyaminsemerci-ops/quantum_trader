#!/bin/bash
# Add rule_files section after global section

sed -i '/^global:/a \
\
rule_files:\
  - "/etc/prometheus/rules/*.yml"' /etc/prometheus/prometheus.yml

echo "✅ Added rule_files section"
