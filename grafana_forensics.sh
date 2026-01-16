#!/bin/bash
set -euo pipefail

TS="$(date -Is | tr ':' '-')"
OUT="/var/log/quantum/grafana_forensics_${TS}.log"
mkdir -p /var/log/quantum

exec > >(tee -a "$OUT") 2>&1

echo "=== META ==="
date -Is
hostnamectl || true
timedatectl || true
uname -a

echo
echo "=== 1) GRAFANA SERVICE + VERSION ==="
systemctl status grafana-server --no-pager -l || true
systemctl show grafana-server --property=ActiveState,SubState,ExecStart,User,Group,FragmentPath,EnvironmentFiles --no-pager || true
command -v grafana-server || true
grafana-server -v 2>/dev/null || true
dpkg -l | grep -i grafana || true

echo
echo "=== 2) LISTENERS / PORTS ==="
ss -lntp | egrep ":3000|:3100|:9090|:9100|:9093|:80|:443" || true

echo
echo "=== 3) GRAFANA HEALTH + ROOT ==="
curl -sS -m 3 -D- http://127.0.0.1:3000/api/health || true
echo
curl -sS -m 3 -D- http://127.0.0.1:3000/ | head -60 || true

echo
echo "=== 4) grafana.ini key values ==="
if [ -f /etc/grafana/grafana.ini ]; then
  egrep -n "^\s*(\[|http_port|domain|root_url|serve_from_sub_path|protocol|http_addr|enforce_domain|router_logging|log|paths|provisioning|auth)" /etc/grafana/grafana.ini || true
fi
echo "--- /etc/default/grafana-server ---"
sed -n '1,200p' /etc/default/grafana-server 2>/dev/null || true

echo
echo "=== 5) PROVISIONING (datasources + dashboards) ==="
find /etc/grafana/provisioning -maxdepth 4 -type f -print 2>/dev/null || true
echo
echo "--- datasources yaml content ---"
find /etc/grafana/provisioning/datasources -type f -print -exec sed -n '1,200p' {} \; 2>/dev/null || true
echo
echo "--- dashboards yaml content ---"
find /etc/grafana/provisioning/dashboards -type f -print -exec sed -n '1,240p' {} \; 2>/dev/null || true

echo
echo "=== 6) SQLITE DB (datasources + dashboards) ==="
if [ -f /var/lib/grafana/grafana.db ]; then
  ls -lh /var/lib/grafana/grafana.db || true
  sqlite3 /var/lib/grafana/grafana.db "select id,name,type,url,access,is_default,uid from data_source;" || true
  echo
  sqlite3 /var/lib/grafana/grafana.db "select id,uid,title,slug from dashboard order by id desc limit 50;" || true
fi

echo
echo "=== 7) DEPENDENCIES (loki/prometheus/nginx/caddy/apache/node_exporter) ==="
systemctl list-unit-files | egrep -i "loki|promtail|prometheus|alertmanager|node_exporter|nginx|caddy|apache2" || true
echo
systemctl list-units --all | egrep -i "loki|promtail|prometheus|alertmanager|node_exporter|nginx|caddy|apache2" || true

echo
echo "=== 8) LOCAL DEP HEALTH ==="
curl -sS -m 2 http://127.0.0.1:3100/ready || true
echo
curl -sS -m 2 http://127.0.0.1:9090/-/ready || true
echo
curl -sS -m 2 http://127.0.0.1:9100/metrics | head -30 || true

echo
echo "=== 9) GRAFANA ERROR EVIDENCE (last 24h) ==="
journalctl -u grafana-server --since "24 hours ago" --no-pager | egrep -i "status=400|downstream|tsdb.loki|framesLength|status=404|/api/quantum/ensemble/metrics|error|fail|panic" | tail -260 || true

echo
echo "=== 10) DASHBOARD JSON SEARCH FOR BAD URLS ==="
find /var/lib/grafana -type f -name "*.json" -o -name "*.yaml" -o -name "*.yml" 2>/dev/null \
 | head -200 \
 | while read -r f; do
    egrep -nH "localhost:|127\.0\.0\.1:|/api/quantum/ensemble/metrics|:3100|:9090|loki|prometheus" "$f" 2>/dev/null | head -40 || true
   done

echo
echo "=== 11) NGINX/CADDY ROUTING CHECK (if present) ==="
if systemctl is-active --quiet nginx; then
  nginx -T 2>/dev/null | egrep -n "server_name|listen|location|proxy_pass|grafana" | head -220 || true
fi
if systemctl is-active --quiet caddy; then
  caddy list-modules 2>/dev/null | head -60 || true
  caddy adapt --config /etc/caddy/Caddyfile 2>/dev/null | head -120 || true
fi

echo
echo "=== DONE ==="
echo "OUTPUT_LOG=$OUT"
