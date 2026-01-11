#!/bin/bash
# Test OpenTelemetry tracing setup
# Verifies: Jaeger running, OTLP endpoints, microservice instrumentation

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== OpenTelemetry Tracing Verification ==="
echo ""

# 1. Check Jaeger service
echo "1. Checking Jaeger service..."
if systemctl is-active --quiet jaeger; then
    echo -e "${GREEN}✅ jaeger.service is running${NC}"
else
    echo -e "${RED}❌ jaeger.service is NOT running${NC}"
    echo "   Start with: sudo systemctl start jaeger"
    exit 1
fi

# 2. Check Jaeger ports
echo ""
echo "2. Checking Jaeger ports..."
PORTS=(16686 4317 4318)
for port in "${PORTS[@]}"; do
    if ss -tlnp | grep -q ":$port"; then
        echo -e "${GREEN}✅ Port $port is listening${NC}"
    else
        echo -e "${RED}❌ Port $port NOT listening${NC}"
    fi
done

# 3. Check Jaeger UI
echo ""
echo "3. Checking Jaeger UI..."
if curl -sf http://localhost:16686 > /dev/null; then
    echo -e "${GREEN}✅ Jaeger UI responding at http://localhost:16686${NC}"
else
    echo -e "${RED}❌ Jaeger UI not responding${NC}"
fi

# 4. Check microservice tracing (ai-engine)
echo ""
echo "4. Checking microservice instrumentation..."
if systemctl is-active --quiet quantum-ai-engine; then
    echo -e "${GREEN}✅ quantum-ai-engine.service is running${NC}"
    
    # Check for OpenTelemetry in logs
    if journalctl -u quantum-ai-engine --since "5 minutes ago" | grep -qi "opentelemetry\|tracing"; then
        echo -e "${GREEN}✅ OpenTelemetry traces found in logs${NC}"
    else
        echo -e "${YELLOW}⚠️  No recent OpenTelemetry logs (may be normal if no traffic)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  quantum-ai-engine.service not running${NC}"
fi

# 5. Check OTLP endpoint configuration
echo ""
echo "5. Checking .env configuration..."
ENV_FILE="/home/qt/quantum_trader/.env"
if [ -f "$ENV_FILE" ]; then
    if grep -q "OTLP_ENDPOINT.*localhost:4317" "$ENV_FILE"; then
        echo -e "${GREEN}✅ OTLP_ENDPOINT correctly set to localhost:4317${NC}"
    elif grep -q "OTLP_ENDPOINT.*jaeger:4317" "$ENV_FILE"; then
        echo -e "${RED}❌ OTLP_ENDPOINT uses Docker hostname 'jaeger' - should be 'localhost'${NC}"
        echo "   Fix with: sed -i 's|http://jaeger:4317|http://localhost:4317|g' $ENV_FILE"
    else
        echo -e "${YELLOW}⚠️  OTLP_ENDPOINT not found in .env${NC}"
    fi
    
    if grep -q "ENABLE_TRACING=true" "$ENV_FILE"; then
        echo -e "${GREEN}✅ ENABLE_TRACING=true${NC}"
    else
        echo -e "${YELLOW}⚠️  ENABLE_TRACING not set to true${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found at $ENV_FILE${NC}"
fi

# 6. Test OTLP gRPC endpoint
echo ""
echo "6. Testing OTLP gRPC endpoint..."
if nc -zv localhost 4317 2>&1 | grep -q "succeeded"; then
    echo -e "${GREEN}✅ OTLP gRPC (4317) is reachable${NC}"
else
    echo -e "${RED}❌ OTLP gRPC (4317) NOT reachable${NC}"
fi

# Summary
echo ""
echo "=== Summary ==="
echo ""
echo "🌐 Access Jaeger UI via SSH tunnel:"
echo "   ssh -L 16686:localhost:16686 root@46.224.116.254"
echo "   Then open: http://localhost:16686"
echo ""
echo "📊 View Jaeger logs:"
echo "   journalctl -u jaeger -f"
echo ""
echo "🔍 Search for traces in microservice logs:"
echo "   journalctl -u quantum-ai-engine -f | grep -i tracing"
echo ""
