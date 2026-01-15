#!/bin/bash
set -e

# Backup nginx config
cp /etc/nginx/sites-enabled/app.quantumfond.com /etc/nginx/sites-enabled/app.quantumfond.com.bak

# Add diagnostics API location before /grafana/ location
sed -i '/location \/grafana\/ {/i\    # Quantum Core Diagnostics API\n    location /grafana/api/quantum/ {\n        proxy_pass http://127.0.0.1:8000/grafana/api/quantum/;\n        proxy_http_version 1.1;\n        proxy_set_header Host $host;\n        proxy_set_header X-Real-IP $remote_addr;\n        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n        proxy_set_header X-Forwarded-Proto $scheme;\n    }\n' /etc/nginx/sites-enabled/app.quantumfond.com

nginx -t && systemctl reload nginx
echo "✅ Nginx updated and reloaded"
grep -A 8 "Quantum Core Diagnostics" /etc/nginx/sites-enabled/app.quantumfond.com
