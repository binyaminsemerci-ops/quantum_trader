#!/bin/bash
################################################################################
# QUANTUM TRADER - Full Ensemble v5 Deployment Script
# Description: Train all models and deploy to production
# Usage: ./deploy_ensemble_v5.sh [lightgbm|patchtst|nhits|all|validate]
################################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Paths
PROJECT_DIR="/home/qt/quantum_trader"
VENV="/opt/quantum/venvs/ai-engine"
MODELS_SRC="$PROJECT_DIR/ai_engine/models"
MODELS_PROD="/opt/quantum/ai_engine/models"

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source "$VENV/bin/activate"

# Navigate to project
cd "$PROJECT_DIR"

################################################################################
# FUNCTIONS
################################################################################

train_lightgbm() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    TRAINING LIGHTGBM V5                          ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    
    python3 ops/retrain/train_lightgbm_v5.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ LightGBM v5 training complete!${NC}"
        
        echo -e "${BLUE}📦 Deploying LightGBM v5 to production...${NC}"
        sudo cp "$MODELS_SRC"/lightgbm_v*_v5*.pkl "$MODELS_PROD/"
        sudo chown qt:qt "$MODELS_PROD"/lightgbm_v*_v5*
        
        echo -e "${GREEN}✅ LightGBM v5 deployed!${NC}"
    else
        echo -e "${RED}❌ LightGBM training failed!${NC}"
        exit 1
    fi
}

train_patchtst() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    TRAINING PATCHTST V5                          ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Check PyTorch
    python3 -c "import torch" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  PyTorch not found, installing...${NC}"
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
    
    python3 ops/retrain/train_patchtst_v5.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ PatchTST v5 training complete!${NC}"
        
        echo -e "${BLUE}📦 Deploying PatchTST v5 to production...${NC}"
        sudo cp "$MODELS_SRC"/patchtst_v*_v5* "$MODELS_PROD/"
        sudo chown qt:qt "$MODELS_PROD"/patchtst_v*_v5*
        
        echo -e "${GREEN}✅ PatchTST v5 deployed!${NC}"
    else
        echo -e "${RED}❌ PatchTST training failed!${NC}"
        exit 1
    fi
}

train_nhits() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    TRAINING N-HITS V5                            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    
    python3 ops/retrain/train_nhits_v5.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ N-HiTS v5 training complete!${NC}"
        
        echo -e "${BLUE}📦 Deploying N-HiTS v5 to production...${NC}"
        sudo cp "$MODELS_SRC"/nhits_v*_v5* "$MODELS_PROD/"
        sudo chown qt:qt "$MODELS_PROD"/nhits_v*_v5*
        
        echo -e "${GREEN}✅ N-HiTS v5 deployed!${NC}"
    else
        echo -e "${RED}❌ N-HiTS training failed!${NC}"
        exit 1
    fi
}

restart_service() {
    echo ""
    echo -e "${BLUE}🔄 Restarting AI Engine service...${NC}"
    sudo systemctl restart quantum-ai-engine.service
    
    echo -e "${BLUE}⏳ Waiting for service to start...${NC}"
    sleep 5
    
    # Check status
    if systemctl is-active --quiet quantum-ai-engine.service; then
        echo -e "${GREEN}✅ Service restarted successfully!${NC}"
        
        echo ""
        echo -e "${BLUE}📊 Checking agent logs...${NC}"
        journalctl -u quantum-ai-engine.service --since "10 seconds ago" | \
            grep -E "Agent.*Loaded|XGB-Agent|LGBM-Agent|PatchTST-Agent|NHiTS-Agent" || \
            echo -e "${YELLOW}⚠️  No agent logs found yet${NC}"
    else
        echo -e "${RED}❌ Service failed to start!${NC}"
        journalctl -u quantum-ai-engine.service --since "30 seconds ago" | tail -20
        exit 1
    fi
}

run_validation() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    ENSEMBLE VALIDATION                           ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    
    python3 ops/validation/ensemble_validate_v5.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Validation complete!${NC}"
        
        if [ -f "ops/validation/ensemble_v5_validation_report.json" ]; then
            echo -e "${BLUE}📄 Validation report:${NC}"
            cat ops/validation/ensemble_v5_validation_report.json | jq '.'
        fi
    else
        echo -e "${RED}❌ Validation failed!${NC}"
        exit 1
    fi
}

show_status() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║    CURRENT ENSEMBLE STATUS                       ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${BLUE}📁 Production Models:${NC}"
    ls -lh "$MODELS_PROD"/*_v5* 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
    
    echo ""
    echo -e "${BLUE}🔧 Service Status:${NC}"
    systemctl status quantum-ai-engine.service --no-pager | head -5
    
    echo ""
    echo -e "${BLUE}📊 Recent Agent Activity (last 1 minute):${NC}"
    journalctl -u quantum-ai-engine.service --since "1 minute ago" | \
        grep -E "Agent.*→" | tail -10 || \
        echo -e "${YELLOW}   No recent activity${NC}"
}

################################################################################
# MAIN
################################################################################

case "${1:-help}" in
    lightgbm)
        echo -e "${GREEN}Starting LightGBM v5 training...${NC}"
        train_lightgbm
        restart_service
        ;;
    
    patchtst)
        echo -e "${GREEN}Starting PatchTST v5 training...${NC}"
        train_patchtst
        restart_service
        ;;
    
    nhits)
        echo -e "${GREEN}Starting N-HiTS v5 training...${NC}"
        train_nhits
        restart_service
        ;;
    
    all)
        echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  FULL ENSEMBLE V5 TRAINING (ALL 3 MODELS)        ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}This will train:${NC}"
        echo -e "${YELLOW}  1. LightGBM v5 (~10 min)${NC}"
        echo -e "${YELLOW}  2. PatchTST v5 (~20 min)${NC}"
        echo -e "${YELLOW}  3. N-HiTS v5 (~20 min)${NC}"
        echo -e "${YELLOW}Total estimated time: 50-60 minutes${NC}"
        echo ""
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            train_lightgbm
            train_patchtst
            train_nhits
            restart_service
            run_validation
        else
            echo -e "${YELLOW}Aborted.${NC}"
            exit 0
        fi
        ;;
    
    validate)
        echo -e "${GREEN}Running ensemble validation...${NC}"
        run_validation
        ;;
    
    status)
        show_status
        ;;
    
    restart)
        restart_service
        ;;
    
    help|*)
        echo ""
        echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║  QUANTUM TRADER - ENSEMBLE V5 DEPLOYMENT         ║${NC}"
        echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  lightgbm   - Train and deploy LightGBM v5"
        echo "  patchtst   - Train and deploy PatchTST v5"
        echo "  nhits      - Train and deploy N-HiTS v5"
        echo "  all        - Train all 3 models sequentially"
        echo "  validate   - Run ensemble validation"
        echo "  status     - Show current ensemble status"
        echo "  restart    - Restart AI Engine service"
        echo "  help       - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 lightgbm     # Train LightGBM only"
        echo "  $0 all          # Train all models"
        echo "  $0 validate     # Check ensemble health"
        echo ""
        ;;
esac

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
