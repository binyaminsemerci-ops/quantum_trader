#!/bin/bash
# Verify Nginx Fix and Health Status
# Created: 2025-12-24

echo "🔧 VERIFYING NGINX FIX AND HEALTH STATUS"
echo "=================================================================="

# Test 1: Check nginx config is valid
echo ""
echo "📊 Test 1: Validating nginx configuration..."
echo "---------------------------------------------"
docker exec quantum_nginx nginx -t 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Nginx config is valid"
else
    echo "❌ Nginx config has errors"
fi

# Test 2: Check upstream backend definition
echo ""
echo "📊 Test 2: Checking backend upstream configuration..."
echo "------------------------------------------------------"
docker exec quantum_nginx grep -A 3 "upstream backend" /etc/nginx/nginx.conf

if [ $? -eq 0 ]; then
    echo "✅ Backend upstream found"
else
    echo "❌ Backend upstream NOT found"
fi

# Test 3: Check /health proxy_pass
echo ""
echo "📊 Test 3: Checking /health location proxy configuration..."
echo "-----------------------------------------------------------"
docker exec quantum_nginx grep -A 5 "location /health" /etc/nginx/nginx.conf | grep proxy_pass

PROXY_BACKEND=$(docker exec quantum_nginx grep -A 5 "location /health" /etc/nginx/nginx.conf | grep -c "proxy_pass http://backend")

if [ "$PROXY_BACKEND" -gt 0 ]; then
    echo "✅ /health proxies to backend upstream"
else
    echo "❌ /health does NOT proxy to backend"
fi

# Test 4: Test backend health endpoint directly
echo ""
echo "📊 Test 4: Testing backend health endpoint directly..."
echo "-------------------------------------------------------"
BACKEND_HEALTH=$(curl -s http://localhost:8000/health 2>&1)

if echo "$BACKEND_HEALTH" | grep -q "healthy"; then
    echo "✅ Backend /health returns healthy"
    echo "   Response: $BACKEND_HEALTH"
else
    echo "⚠️  Backend health response: $BACKEND_HEALTH"
fi

# Test 5: Test nginx proxy to backend health
echo ""
echo "📊 Test 5: Testing nginx proxy to /health..."
echo "--------------------------------------------"
NGINX_HEALTH_HTTP=$(curl -s http://localhost:80/health 2>&1)

if echo "$NGINX_HEALTH_HTTP" | grep -q "healthy"; then
    echo "✅ HTTP nginx /health works"
else
    echo "⚠️  HTTP nginx /health response: $NGINX_HEALTH_HTTP"
fi

# Test 6: Test HTTPS health endpoint
echo ""
echo "📊 Test 6: Testing HTTPS health endpoint..."
echo "-------------------------------------------"
NGINX_HEALTH_HTTPS=$(curl -k -s https://localhost:443/health 2>&1)

if echo "$NGINX_HEALTH_HTTPS" | grep -q "healthy"; then
    echo "✅ HTTPS nginx /health works"
else
    echo "⚠️  HTTPS nginx /health response: $NGINX_HEALTH_HTTPS"
fi

# Test 7: Check nginx container health status
echo ""
echo "📊 Test 7: Checking nginx container health status..."
echo "-----------------------------------------------------"
NGINX_STATUS=$(docker inspect quantum_nginx --format='{{.State.Health.Status}}' 2>&1)

echo "   Container health status: $NGINX_STATUS"

if [ "$NGINX_STATUS" == "healthy" ]; then
    echo "✅ Nginx container is HEALTHY"
elif [ "$NGINX_STATUS" == "starting" ]; then
    echo "⏳ Nginx container is STARTING (wait 30s for healthcheck)"
else
    echo "❌ Nginx container status: $NGINX_STATUS"
fi

# Test 8: Check recent nginx errors
echo ""
echo "📊 Test 8: Checking recent nginx error logs..."
echo "-----------------------------------------------"
docker logs --tail 50 quantum_nginx 2>&1 | grep -i error | tail -5

if [ $? -ne 0 ]; then
    echo "✅ No recent errors in nginx logs"
fi

# Test 9: Check nginx access logs for health checks
echo ""
echo "📊 Test 9: Recent health check attempts in access logs..."
echo "----------------------------------------------------------"
docker logs --tail 20 quantum_nginx 2>&1 | grep "/health" | tail -5

# Test 10: Show last healthcheck result
echo ""
echo "📊 Test 10: Last healthcheck result..."
echo "--------------------------------------"
docker inspect quantum_nginx --format='{{json .State.Health}}' | jq '.Log[-1]' 2>/dev/null

echo ""
echo "=================================================================="
echo "🏁 VERIFICATION COMPLETE"
echo ""
echo "📋 Summary:"
echo "  - Nginx config valid: $(docker exec quantum_nginx nginx -t 2>&1 | grep -c 'successful')"
echo "  - Backend upstream exists: $PROXY_BACKEND"
echo "  - Backend /health direct: $(if echo "$BACKEND_HEALTH" | grep -q 'healthy'; then echo 'WORKS ✅'; else echo 'ISSUE ⚠️'; fi)"
echo "  - Nginx /health HTTP: $(if echo "$NGINX_HEALTH_HTTP" | grep -q 'healthy'; then echo 'WORKS ✅'; else echo 'ISSUE ⚠️'; fi)"
echo "  - Nginx /health HTTPS: $(if echo "$NGINX_HEALTH_HTTPS" | grep -q 'healthy'; then echo 'WORKS ✅'; else echo 'ISSUE ⚠️'; fi)"
echo "  - Container status: $NGINX_STATUS"
echo ""

if [ "$NGINX_STATUS" == "healthy" ]; then
    echo "🎉 VERDICT: Nginx is HEALTHY and working correctly!"
elif [ "$NGINX_STATUS" == "starting" ]; then
    echo "⏳ VERDICT: Wait 30 seconds for healthcheck to complete"
else
    echo "❌ VERDICT: Nginx needs additional troubleshooting"
    echo "   Check: docker logs quantum_nginx"
fi
