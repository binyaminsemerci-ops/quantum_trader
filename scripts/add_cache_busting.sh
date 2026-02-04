#!/bin/bash

# Add cache-busting for dashboard JS/CSS files

CONFIG_FILE="/etc/nginx/sites-enabled/app.quantumfond.com"

# Backup original
cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

# Check if cache-busting already exists
if grep -q "no-store, no-cache" "$CONFIG_FILE"; then
    echo "Cache-busting headers already configured"
    exit 0
fi

# Add cache-busting location block before the main location /
sed -i '/root \/root\/quantum_trader\/dashboard_v4\/frontend\/dist;/a\
\
    # Disable cache for JS/CSS bundles to ensure users get latest version\
    location ~* \\.(js|css)$ {\
        expires -1;\
        add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";\
        try_files $uri =404;\
    }' "$CONFIG_FILE"

# Test nginx config
if nginx -t; then
    echo "✅ Nginx config valid - reloading..."
    systemctl reload nginx
    echo "✅ Cache-busting enabled!"
else
    echo "❌ Nginx config error - restoring backup"
    mv "$CONFIG_FILE.backup" "$CONFIG_FILE"
    exit 1
fi
