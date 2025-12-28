# Instructions for updating nginx config on VPS
# Run these commands on VPS:

# 1. Backup existing config
sudo cp /etc/nginx/sites-available/quantumfond.conf /etc/nginx/sites-available/quantumfond.conf.backup

# 2. Add RL Dashboard proxy BEFORE the /api/ location block
# Insert this block after "index index.html;" and before "location /api/":

    # RL Dashboard API (RL Intelligence page)
    location /api/rl-dashboard/ {
        proxy_pass http://127.0.0.1:8027/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

# 3. Test nginx config
sudo nginx -t

# 4. Reload nginx
sudo systemctl reload nginx

# Full command for automated deployment:
cat > /tmp/rl_proxy.conf << 'EOF'
    # RL Dashboard API (RL Intelligence page)
    location /api/rl-dashboard/ {
        proxy_pass http://127.0.0.1:8027/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
EOF

# Insert after "index index.html;" in the app.quantumfond.com server block
sudo sed -i '/server_name app.quantumfond.com;/,/location \/api\// { /index index.html;/r /tmp/rl_proxy.conf' /etc/nginx/sites-available/quantumfond.conf
sudo nginx -t && sudo systemctl reload nginx
