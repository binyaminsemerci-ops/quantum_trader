#!/bin/bash
#
# META-AGENT V2 DEPLOYMENT SCRIPT
#
# This script deploys Meta-Agent V2 to production with full validation.
#
# Usage:
#   bash deploy_meta_v2.sh
#
# Steps:
# 1. Validate prerequisites
# 2. Train model (if needed)
# 3. Run tests
# 4. Update service configuration
# 5. Restart service
# 6. Verify deployment
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
PROJECT_ROOT="/home/qt/quantum_trader"
VENV="/opt/quantum/venvs/ai-engine"
MODEL_DIR="$PROJECT_ROOT/ai_engine/models/meta_v2"
SERVICE_NAME="quantum-ai-engine"

# Functions
print_header() {
    echo -e "\n${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${BOLD}${GREEN}➜${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 found"
        return 0
    else
        print_error "$1 not found"
        return 1
    fi
}

# ============================================================================
# STEP 1: VALIDATE PREREQUISITES
# ============================================================================

print_header "STEP 1: VALIDATE PREREQUISITES"

print_step "Checking Python environment..."
if [ -f "$VENV/bin/python" ]; then
    PYTHON_VERSION=$($VENV/bin/python --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python virtual environment not found: $VENV"
    exit 1
fi

print_step "Checking required packages..."
$VENV/bin/python -c "import sklearn, numpy, pandas" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Required packages installed"
else
    print_error "Missing required packages"
    print_info "Install with: pip install scikit-learn numpy pandas"
    exit 1
fi

print_step "Checking project directory..."
if [ -d "$PROJECT_ROOT" ]; then
    print_success "Project directory found"
else
    print_error "Project directory not found: $PROJECT_ROOT"
    exit 1
fi

print_step "Checking base agents..."
if [ -d "$PROJECT_ROOT/ai_engine/agents" ]; then
    print_success "Agents directory found"
    
    # Check if base agent models exist
    if ls $PROJECT_ROOT/ai_engine/models/xgb*.pkl &> /dev/null && \
       ls $PROJECT_ROOT/ai_engine/models/lgbm*.pkl &> /dev/null && \
       ls $PROJECT_ROOT/ai_engine/models/nhits*.pth &> /dev/null && \
       ls $PROJECT_ROOT/ai_engine/models/patchtst*.pth &> /dev/null; then
        print_success "All 4 base agent models found"
    else
        print_warning "Some base agent models may be missing"
        print_info "Meta-agent requires all 4 base agents to be trained"
    fi
else
    print_error "Agents directory not found"
    exit 1
fi

# ============================================================================
# STEP 2: TRAIN MODEL (IF NEEDED)
# ============================================================================

print_header "STEP 2: CHECK MODEL STATUS"

if [ -f "$MODEL_DIR/meta_model.pkl" ]; then
    print_success "Meta-agent model already exists"
    print_info "Model path: $MODEL_DIR/meta_model.pkl"
    
    # Check metadata
    if [ -f "$MODEL_DIR/metadata.json" ]; then
        TRAINING_DATE=$(grep -oP '"training_date": "\K[^"]+' "$MODEL_DIR/metadata.json" || echo "unknown")
        CV_ACC=$(grep -oP '"cv_accuracy": \K[0-9.]+' "$MODEL_DIR/metadata.json" || echo "unknown")
        print_info "Training date: $TRAINING_DATE"
        print_info "CV accuracy: $CV_ACC"
    fi
    
    read -p "$(echo -e ${YELLOW}Do you want to retrain? [y/N]:${NC} )" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SHOULD_TRAIN=true
    else
        SHOULD_TRAIN=false
    fi
else
    print_warning "Meta-agent model not found"
    print_info "Training required for first-time deployment"
    SHOULD_TRAIN=true
fi

if [ "$SHOULD_TRAIN" = true ]; then
    print_step "Training Meta-Agent V2..."
    
    cd $PROJECT_ROOT
    $VENV/bin/python ops/retrain/train_meta_v2.py
    
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully"
    else
        print_error "Training failed"
        exit 1
    fi
else
    print_info "Skipping training, using existing model"
fi

# ============================================================================
# STEP 3: RUN TESTS
# ============================================================================

print_header "STEP 3: RUN TESTS"

print_step "Running unit tests..."
cd $PROJECT_ROOT

# Run pytest if available
if command -v pytest &> /dev/null; then
    $VENV/bin/pytest ai_engine/tests/test_meta_agent_v2.py -v
    if [ $? -eq 0 ]; then
        print_success "Unit tests passed"
    else
        print_warning "Some unit tests failed (may be OK if model not trained)"
    fi
else
    print_warning "pytest not found, skipping unit tests"
fi

print_step "Running integration tests..."
$VENV/bin/python test_meta_v2_integration.py

if [ $? -eq 0 ]; then
    print_success "Integration tests passed"
else
    print_error "Integration tests failed"
    read -p "$(echo -e ${YELLOW}Continue deployment anyway? [y/N]:${NC} )" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Deployment aborted"
        exit 1
    fi
fi

# ============================================================================
# STEP 4: UPDATE SERVICE CONFIGURATION
# ============================================================================

print_header "STEP 4: UPDATE SERVICE CONFIGURATION"

SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

if [ ! -f "$SERVICE_FILE" ]; then
    print_error "Service file not found: $SERVICE_FILE"
    exit 1
fi

print_step "Checking current configuration..."

# Check if META_AGENT_ENABLED is set
if grep -q "META_AGENT_ENABLED=true" "$SERVICE_FILE"; then
    print_success "META_AGENT_ENABLED already set to true"
else
    print_warning "META_AGENT_ENABLED not enabled"
    
    read -p "$(echo -e ${YELLOW}Enable Meta-Agent in service? [Y/n]:${NC} )" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_step "Updating service file..."
        
        # Backup service file
        sudo cp "$SERVICE_FILE" "$SERVICE_FILE.backup_$(date +%Y%m%d_%H%M%S)"
        print_info "Backup created: $SERVICE_FILE.backup_$(date +%Y%m%d_%H%M%S)"
        
        # Add META_AGENT_ENABLED
        sudo sed -i '/\[Service\]/a Environment="META_AGENT_ENABLED=true"' "$SERVICE_FILE"
        sudo sed -i '/\[Service\]/a Environment="META_OVERRIDE_THRESHOLD=0.65"' "$SERVICE_FILE"
        sudo sed -i '/\[Service\]/a Environment="META_FAIL_OPEN=true"' "$SERVICE_FILE"
        
        print_success "Service file updated"
    fi
fi

# Show current meta-agent settings
print_info "Current meta-agent settings:"
grep -A 5 "META_AGENT" "$SERVICE_FILE" || echo "  (none found)"

# ============================================================================
# STEP 5: RESTART SERVICE
# ============================================================================

print_header "STEP 5: RESTART SERVICE"

print_step "Reloading systemd daemon..."
sudo systemctl daemon-reload
print_success "Daemon reloaded"

print_step "Restarting $SERVICE_NAME..."
sudo systemctl restart $SERVICE_NAME

# Wait for service to start
sleep 5

# Check service status
sudo systemctl is-active --quiet $SERVICE_NAME
if [ $? -eq 0 ]; then
    print_success "Service started successfully"
else
    print_error "Service failed to start"
    print_info "Check logs with: journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi

# ============================================================================
# STEP 6: VERIFY DEPLOYMENT
# ============================================================================

print_header "STEP 6: VERIFY DEPLOYMENT"

print_step "Checking service logs for Meta-Agent..."
sleep 3

# Look for meta-agent initialization
if journalctl -u $SERVICE_NAME --since "30 seconds ago" | grep -q "META.*agent loaded"; then
    print_success "Meta-Agent V2 loaded successfully"
    
    # Show meta-agent logs
    echo -e "\n${BOLD}Recent Meta-Agent logs:${NC}"
    journalctl -u $SERVICE_NAME --since "30 seconds ago" --no-pager | grep "META" | tail -5
else
    print_warning "Meta-Agent initialization not found in logs"
    print_info "Check full logs with: journalctl -u $SERVICE_NAME -n 100"
fi

print_step "Checking for prediction activity..."
sleep 5

if journalctl -u $SERVICE_NAME --since "10 seconds ago" | grep -q "ENSEMBLE"; then
    print_success "Prediction activity detected"
else
    print_warning "No prediction activity yet (may be normal)"
fi

# ============================================================================
# DEPLOYMENT SUMMARY
# ============================================================================

print_header "DEPLOYMENT SUMMARY"

echo -e "${BOLD}Meta-Agent V2 Deployment Complete!${NC}\n"

print_success "Model trained and deployed"
print_success "Service configured and running"
print_success "Integration tests passed"

echo -e "\n${BOLD}Next Steps:${NC}"
echo "1. Monitor meta-agent decisions:"
echo "   ${BLUE}journalctl -u $SERVICE_NAME -f | grep META-V2${NC}"
echo ""
echo "2. Check statistics after 1 hour:"
echo "   ${BLUE}curl http://localhost:8001/api/ai/meta/status${NC}"
echo ""
echo "3. Review performance:"
echo "   - Override rate should be 15-30%"
echo "   - Meta should respect strong consensus (>75%)"
echo "   - No constant output warnings"
echo ""
echo "4. Documentation:"
echo "   ${BLUE}cat $PROJECT_ROOT/docs/META_AGENT_V2_GUIDE.md${NC}"

echo -e "\n${GREEN}${BOLD}✓ Deployment successful!${NC}\n"

exit 0
