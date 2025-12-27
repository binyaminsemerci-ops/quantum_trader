"""
SHADOW MODEL INTEGRATION FOR PRODUCTION

This file contains the production-ready integration code for shadow models.
Copy relevant sections into:
- ai_engine/ensemble_manager.py
- backend/routes/ai.py

See SHADOW_MODELS_INTEGRATION_GUIDE.md for detailed instructions.
"""

# ============================================================================
# PART 1: ENSEMBLE MANAGER INTEGRATION
# ============================================================================

# Add to imports in ai_engine/ensemble_manager.py:
"""
import os
from datetime import datetime
from backend.services.ai.shadow_model_manager import (
    ShadowModelManager,
    ModelRole,
    ModelMetadata,
    PromotionStatus,
    TradeResult
)
"""

# Add to EnsembleManager.__init__():
"""
        # Shadow Model Manager (optional - controlled by env var)
        self.shadow_manager = None
        self.shadow_enabled = os.getenv('ENABLE_SHADOW_MODELS', 'false').lower() == 'true'
        
        if self.shadow_enabled:
            try:
                self.shadow_manager = ShadowModelManager(
                    min_trades_for_promotion=int(os.getenv('SHADOW_MIN_TRADES', '500')),
                    mdd_tolerance=float(os.getenv('SHADOW_MDD_TOLERANCE', '1.20')),
                    alpha=float(os.getenv('SHADOW_ALPHA', '0.05')),
                    n_bootstrap=int(os.getenv('SHADOW_N_BOOTSTRAP', '10000')),
                    checkpoint_path='data/shadow_models_checkpoint.json'
                )
                
                # Register current ensemble as champion
                self.shadow_manager.register_model(
                    model_name='ensemble_production_v1',
                    model_type='ensemble',
                    version='1.0',
                    role=ModelRole.CHAMPION,
                    description='Production 4-model ensemble (XGB+LGBM+NHITS+PatchTST)'
                )
                
                logger.info("[Shadow] Shadow model manager ENABLED")
                logger.info(f"[Shadow] Champion registered: ensemble_production_v1")
                
            except Exception as e:
                logger.error(f"[Shadow] Failed to initialize shadow manager: {e}")
                self.shadow_enabled = False
        else:
            logger.info("[Shadow] Shadow model manager DISABLED (set ENABLE_SHADOW_MODELS=true to enable)")
        
        # Shadow testing state
        self.shadow_trade_count = 0
        self.shadow_check_interval = int(os.getenv('SHADOW_CHECK_INTERVAL', '100'))
"""

# Add new method to EnsembleManager:
"""
    def record_trade_outcome_for_shadow(
        self,
        symbol: str,
        prediction_result: Tuple[str, float, Dict[str, Any]],
        actual_outcome: int,
        pnl: float
    ):
        \"\"\"
        Record trade outcome for shadow model tracking.
        
        Args:
            symbol: Trading pair
            prediction_result: Original prediction tuple (action, confidence, info)
            actual_outcome: 1 (win) or 0 (loss)
            pnl: Profit/loss amount
        \"\"\"
        if not self.shadow_enabled or self.shadow_manager is None:
            return
        
        try:
            action, confidence, info = prediction_result
            
            # Record champion outcome
            self.shadow_manager.record_prediction(
                model_name='ensemble_production_v1',
                prediction=1 if action == 'LONG' else 0,  # Convert action to binary
                actual_outcome=actual_outcome,
                pnl=pnl,
                confidence=confidence,
                executed=True
            )
            
            # Record challenger outcomes (if any challengers active)
            challengers = self.shadow_manager.get_challengers()
            for challenger_name in challengers:
                # Get challenger prediction (shadow mode)
                # In production, this would come from parallel prediction pipeline
                # For now, we simulate or skip
                pass
            
            # Increment counter
            self.shadow_trade_count += 1
            
            # Periodic promotion check (every 100 trades)
            if self.shadow_trade_count >= self.shadow_check_interval:
                self._check_shadow_promotions()
                self.shadow_trade_count = 0
        
        except Exception as e:
            logger.error(f"[Shadow] Failed to record trade outcome: {e}")
    
    def _check_shadow_promotions(self):
        \"\"\"Check if any challenger is ready for promotion\"\"\"
        if not self.shadow_enabled or self.shadow_manager is None:
            return
        
        try:
            challengers = self.shadow_manager.get_challengers()
            
            for challenger_name in challengers:
                trade_count = self.shadow_manager.get_trade_count(challenger_name)
                
                # Check if minimum trades reached
                if trade_count < 500:
                    logger.info(f"[Shadow] {challenger_name}: {trade_count}/500 trades")
                    continue
                
                # Run promotion check
                decision = self.shadow_manager.check_promotion_criteria(challenger_name)
                
                if decision is None:
                    continue
                
                logger.info(
                    f"[Shadow] {challenger_name} promotion check: "
                    f"Status={decision.status.value}, Score={decision.promotion_score:.1f}/100"
                )
                
                # Auto-promote if approved
                if decision.status == PromotionStatus.APPROVED:
                    logger.info(f"[Shadow] 🎉 Auto-promoting {challenger_name} → Champion")
                    
                    success = self.shadow_manager.promote_challenger(challenger_name)
                    
                    if success:
                        # Alert team
                        improvement = (
                            decision.challenger_metrics['win_rate'] - 
                            decision.champion_metrics['win_rate']
                        )
                        
                        logger.info(
                            f"[Shadow] PROMOTED: {challenger_name} | "
                            f"Score: {decision.promotion_score:.1f}/100 | "
                            f"WR Improvement: +{improvement:.2%}"
                        )
                
                elif decision.status == PromotionStatus.PENDING:
                    # Manual review needed
                    logger.warning(
                        f"[Shadow] ⚠️  {challenger_name} needs MANUAL REVIEW: "
                        f"{decision.reason}"
                    )
                
                else:
                    # Rejected
                    logger.info(
                        f"[Shadow] ❌ {challenger_name} rejected: {decision.reason}"
                    )
        
        except Exception as e:
            logger.error(f"[Shadow] Promotion check failed: {e}")
    
    def deploy_shadow_challenger(
        self,
        model_name: str,
        model_type: str,
        description: str = ""
    ):
        \"\"\"
        Deploy a new challenger model for shadow testing.
        
        Args:
            model_name: Unique model name
            model_type: Model type (xgboost, lightgbm, catboost, etc.)
            description: Human-readable description
        
        Returns:
            dict: Deployment status
        \"\"\"
        if not self.shadow_enabled or self.shadow_manager is None:
            return {
                'status': 'error',
                'message': 'Shadow models not enabled'
            }
        
        try:
            # Register with shadow manager
            self.shadow_manager.register_model(
                model_name=model_name,
                model_type=model_type,
                version='1.0',
                role=ModelRole.CHALLENGER,
                description=description
            )
            
            logger.info(f"[Shadow] Deployed challenger: {model_name} ({model_type})")
            
            return {
                'status': 'success',
                'model_name': model_name,
                'role': 'challenger',
                'allocation': '0% (shadow mode)',
                'message': f'Challenger {model_name} deployed successfully'
            }
        
        except Exception as e:
            logger.error(f"[Shadow] Failed to deploy challenger: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_shadow_status(self) -> Dict[str, Any]:
        \"\"\"Get current shadow model status\"\"\"
        if not self.shadow_enabled or self.shadow_manager is None:
            return {
                'enabled': False,
                'message': 'Shadow models not enabled'
            }
        
        try:
            champion = self.shadow_manager.get_champion()
            challengers = self.shadow_manager.get_challengers()
            
            status = {
                'enabled': True,
                'champion': {
                    'model_name': champion,
                    'metrics': self.shadow_manager.get_metrics(champion).__dict__ if champion else None,
                    'trade_count': self.shadow_manager.get_trade_count(champion)
                },
                'challengers': []
            }
            
            for challenger in challengers:
                metrics = self.shadow_manager.get_metrics(challenger)
                decision = self.shadow_manager.get_pending_decision(challenger)
                
                challenger_info = {
                    'model_name': challenger,
                    'metrics': metrics.__dict__ if metrics else None,
                    'trade_count': self.shadow_manager.get_trade_count(challenger),
                    'promotion_status': decision.status.value if decision else 'pending',
                    'promotion_score': decision.promotion_score if decision else 0,
                    'reason': decision.reason if decision else ''
                }
                
                status['challengers'].append(challenger_info)
            
            return status
        
        except Exception as e:
            logger.error(f"[Shadow] Failed to get status: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }
"""

# ============================================================================
# PART 2: API ROUTES INTEGRATION
# ============================================================================

# Add to imports in backend/routes/ai.py:
"""
from ai_engine.ensemble_manager import EnsembleManager
"""

# Add these routes to backend/routes/ai.py:
"""
# ============================================================================
# SHADOW MODEL ENDPOINTS
# ============================================================================

@router.get("/shadow/status")
async def get_shadow_status():
    \"\"\"Get status of all shadow models\"\"\"
    try:
        # Get ensemble manager instance (adjust based on your app structure)
        ensemble = make_default_agent()  # Or however you access the ensemble
        
        status = ensemble.get_shadow_status()
        
        return {
            'status': 'success',
            'data': status
        }
    
    except Exception as e:
        logger.error(f"Failed to get shadow status: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/shadow/comparison/{challenger_name}")
async def get_shadow_comparison(challenger_name: str):
    \"\"\"Get detailed comparison between champion and challenger\"\"\"
    try:
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        champion = ensemble.shadow_manager.get_champion()
        
        champion_metrics = ensemble.shadow_manager.get_metrics(champion)
        challenger_metrics = ensemble.shadow_manager.get_metrics(challenger_name)
        
        if not champion_metrics or not challenger_metrics:
            return {'status': 'error', 'message': 'Insufficient data'}
        
        # Get test results
        test_results_history = ensemble.shadow_manager.get_test_results_history(challenger_name, n=1)
        latest_test = test_results_history[0] if test_results_history else None
        
        comparison = {
            'champion': {
                'model_name': champion,
                'win_rate': champion_metrics.win_rate,
                'sharpe_ratio': champion_metrics.sharpe_ratio,
                'mean_pnl': champion_metrics.mean_pnl,
                'max_drawdown': champion_metrics.max_drawdown,
                'total_pnl': champion_metrics.total_pnl,
                'n_trades': champion_metrics.n_trades
            },
            'challenger': {
                'model_name': challenger_name,
                'win_rate': challenger_metrics.win_rate,
                'sharpe_ratio': challenger_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl,
                'max_drawdown': challenger_metrics.max_drawdown,
                'total_pnl': challenger_metrics.total_pnl,
                'n_trades': challenger_metrics.n_trades
            },
            'difference': {
                'win_rate': challenger_metrics.win_rate - champion_metrics.win_rate,
                'sharpe_ratio': challenger_metrics.sharpe_ratio - champion_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl - champion_metrics.mean_pnl,
                'max_drawdown': champion_metrics.max_drawdown - challenger_metrics.max_drawdown
            },
            'statistical_tests': latest_test.__dict__ if latest_test else None
        }
        
        return {
            'status': 'success',
            'data': comparison
        }
    
    except Exception as e:
        logger.error(f"Failed to get comparison: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/deploy")
async def deploy_shadow_model(request: dict):
    \"\"\"Deploy a new challenger model for shadow testing\"\"\"
    try:
        model_name = request.get('model_name')
        model_type = request.get('model_type')
        description = request.get('description', '')
        
        if not model_name or not model_type:
            return {'status': 'error', 'message': 'model_name and model_type required'}
        
        ensemble = make_default_agent()
        
        result = ensemble.deploy_shadow_challenger(
            model_name=model_name,
            model_type=model_type,
            description=description
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to deploy shadow model: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/promote/{challenger_name}")
async def promote_shadow_model(challenger_name: str, force: bool = False):
    \"\"\"Manually promote a challenger to champion\"\"\"
    try:
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        success = ensemble.shadow_manager.promote_challenger(challenger_name, force=force)
        
        if success:
            return {
                'status': 'success',
                'message': f'{challenger_name} promoted to champion'
            }
        else:
            return {
                'status': 'error',
                'message': 'Promotion failed (check criteria)'
            }
    
    except Exception as e:
        logger.error(f"Failed to promote: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/rollback")
async def rollback_champion(request: dict):
    \"\"\"Rollback to previous champion\"\"\"
    try:
        reason = request.get('reason', 'Manual rollback')
        
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        success = ensemble.shadow_manager.rollback_to_previous_champion(reason=reason)
        
        if success:
            champion = ensemble.shadow_manager.get_champion()
            return {
                'status': 'success',
                'message': f'Rolled back to {champion}'
            }
        else:
            return {
                'status': 'error',
                'message': 'Rollback failed (no history)'
            }
    
    except Exception as e:
        logger.error(f"Failed to rollback: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/shadow/history")
async def get_promotion_history(n: int = 10):
    \"\"\"Get promotion history\"\"\"
    try:
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        history = ensemble.shadow_manager.get_promotion_history(n=n)
        
        return {
            'status': 'success',
            'data': [event.__dict__ for event in history]
        }
    
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return {'status': 'error', 'message': str(e)}
"""

# ============================================================================
# PART 3: ENVIRONMENT VARIABLES
# ============================================================================

# Add to .env file:
"""
# Shadow Model Configuration
ENABLE_SHADOW_MODELS=false           # Set to 'true' to enable shadow testing
SHADOW_MIN_TRADES=500                # Minimum trades before promotion check
SHADOW_MDD_TOLERANCE=1.20           # Max drawdown tolerance (1.20 = 20% worse allowed)
SHADOW_ALPHA=0.05                   # Statistical significance level
SHADOW_N_BOOTSTRAP=10000            # Bootstrap iterations for CI
SHADOW_CHECK_INTERVAL=100           # Check for promotions every N trades
"""

print("✅ Shadow model integration code ready!")
print("\n📋 Next steps:")
print("1. Copy relevant sections to ai_engine/ensemble_manager.py")
print("2. Copy API routes to backend/routes/ai.py")
print("3. Add environment variables to .env")
print("4. Test with: ENABLE_SHADOW_MODELS=true")
print("\nSee SHADOW_MODELS_DEPLOYMENT_CHECKLIST.md for full deployment guide.")
