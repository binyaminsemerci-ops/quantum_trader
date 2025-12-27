"""Integration example showing CLM with real Quantum Trader components.

This demonstrates how to wire up the CLM with concrete implementations
for production use in Quantum Trader's backend.
"""

import logging
from datetime import datetime, timedelta, timezone

from backend.services.ai.continuous_learning_manager import (
    ContinuousLearningManager,
    ModelType,
    RetrainTrigger,
)
from backend.services.clm_implementations import (
    BinanceDataClient,
    QuantumFeatureEngineer,
    QuantumModelTrainer,
    QuantumModelEvaluator,
    QuantumShadowTester,
    SQLModelRegistry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run CLM integration example with real components."""
    
    print("=" * 80)
    print("CONTINUOUS LEARNING MANAGER - QUANTUM TRADER INTEGRATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # STEP 1: Initialize all concrete implementations
    # ========================================================================
    
    print("Step 1: Initializing CLM components...")
    print("-" * 80)
    
    # Data client (Binance API)
    data_client = BinanceDataClient(
        symbol="BTCUSDT",
        interval="1h"
    )
    print("✓ BinanceDataClient initialized (BTCUSDT, 1h)")
    
    # Feature engineer (Quantum Trader's feature_engineer module)
    feature_engineer = QuantumFeatureEngineer(use_advanced=True)
    print("✓ QuantumFeatureEngineer initialized (100+ features)")
    
    # Model trainer (XGBoost, LightGBM, N-HiTS, PatchTST)
    trainer = QuantumModelTrainer()
    print("✓ QuantumModelTrainer initialized (4 model types)")
    
    # Model evaluator (comprehensive metrics)
    evaluator = QuantumModelEvaluator(feature_engineer=feature_engineer)
    print("✓ QuantumModelEvaluator initialized")
    
    # Shadow tester (live parallel testing)
    shadow_tester = QuantumShadowTester(
        data_client=data_client,
        feature_engineer=feature_engineer
    )
    print("✓ QuantumShadowTester initialized")
    
    # Model registry (SQL-backed with file storage)
    registry = SQLModelRegistry()
    print("✓ SQLModelRegistry initialized (SQLite + disk storage)")
    print()
    
    # ========================================================================
    # STEP 2: Create CLM instance
    # ========================================================================
    
    print("Step 2: Creating Continuous Learning Manager...")
    print("-" * 80)
    
    clm = ContinuousLearningManager(
        data_client=data_client,
        feature_engineer=feature_engineer,
        trainer=trainer,
        evaluator=evaluator,
        shadow_tester=shadow_tester,
        registry=registry,
        retrain_interval_days=7,
        shadow_test_hours=24,
        min_improvement_threshold=0.02,  # 2% improvement required
        training_lookback_days=90,
    )
    print("✓ CLM initialized with production settings")
    print()
    
    # ========================================================================
    # STEP 3: Check current model status
    # ========================================================================
    
    print("Step 3: Checking current model status...")
    print("-" * 80)
    
    status = clm.get_model_status_summary()
    
    if status:
        for model_type, info in status.items():
            active_ver = info.get("active_version", "none")
            trained = info.get("active_trained_at", "never")
            metrics = info.get("active_metrics", {})
            
            print(f"  {model_type:15s} | {active_ver:20s} | Trained: {trained}")
            
            if metrics:
                rmse = metrics.get("rmse", "N/A")
                dir_acc = metrics.get("directional_accuracy", "N/A")
                print(f"                    | RMSE: {rmse:.6f} | Dir Acc: {dir_acc:.2%}")
    else:
        print("  No active models found (first run)")
    print()
    
    # ========================================================================
    # STEP 4: Check retraining triggers
    # ========================================================================
    
    print("Step 4: Checking retraining triggers...")
    print("-" * 80)
    
    triggers = clm.check_if_retrain_needed()
    
    if triggers:
        triggered_models = [
            model_type for model_type, trigger in triggers.items() if trigger
        ]
        
        if triggered_models:
            print(f"  {len(triggered_models)} model(s) need retraining:")
            for model_type in triggered_models:
                trigger = triggers[model_type]
                print(f"    • {model_type.value:15s} → {trigger.value}")
        else:
            print("  No models need retraining at this time")
    else:
        print("  All models are up-to-date")
    print()
    
    # ========================================================================
    # STEP 5: Run full retraining cycle (DEMO MODE - XGBOOST ONLY)
    # ========================================================================
    
    print("Step 5: Running retraining cycle (demo: XGBoost only)...")
    print("-" * 80)
    print("  WARNING: This will fetch real data from Binance and train a real model")
    print("  Estimated time: 2-5 minutes")
    print()
    
    # Confirm before proceeding
    import os
    if os.getenv("CLM_AUTO_DEMO") != "1":
        response = input("  Proceed with demo? (yes/no): ").strip().lower()
        if response != "yes":
            print("  Demo cancelled. Exiting.")
            return
    
    try:
        # Run cycle for XGBoost only (faster for demo)
        report = clm.run_full_cycle(
            models=[ModelType.XGBOOST],
            force=True  # Force retraining even if not triggered
        )
        
        print()
        print("✓ Retraining cycle complete!")
        print()
        
        # ====================================================================
        # STEP 6: Display results
        # ====================================================================
        
        print("Step 6: Retraining Results")
        print("-" * 80)
        print()
        
        print("RETRAINING REPORT")
        print("=" * 80)
        print(f"Trigger:       {report.trigger.value if report.trigger else 'manual'}")
        print(f"Started:       {report.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Duration:      {report.total_duration_seconds:.1f}s")
        print(f"Trained:       {len(report.models_trained)} models")
        print(f"Promoted:      {len(report.promoted_models)} models")
        print(f"Failed:        {len(report.failed_models)} models")
        print()
        
        # Trained models
        if report.models_trained:
            print("TRAINED MODELS:")
            print("-" * 80)
            for model_type in report.models_trained:
                print(f"  ✓ {model_type.value}")
            print()
        
        # Evaluations
        if report.evaluations:
            print("EVALUATION RESULTS:")
            print("-" * 80)
            for model_type, eval_result in report.evaluations.items():
                print(f"  {model_type.value}:")
                print(f"    RMSE:              {eval_result.rmse:.6f}")
                print(f"    MAE:               {eval_result.mae:.6f}")
                print(f"    Dir Accuracy:      {eval_result.directional_accuracy:.2%}")
                print(f"    vs Active RMSE:    {eval_result.vs_active_rmse_delta:+.6f}")
                print(f"    vs Active Dir Acc: {eval_result.vs_active_direction_delta:+.2%}")
                
                # Determine if better
                is_better = (
                    eval_result.vs_active_rmse_delta < 0 or 
                    eval_result.vs_active_direction_delta > 0.02
                )
                print(f"    Better than active: {'YES ✓' if is_better else 'NO ✗'}")
                print()
        
        # Shadow tests
        if report.shadow_tests:
            print("SHADOW TEST RESULTS:")
            print("-" * 80)
            for model_type, shadow_result in report.shadow_tests.items():
                print(f"  {model_type.value}:")
                print(f"    Live predictions:  {shadow_result.live_predictions}")
                print(f"    Candidate MAE:     {shadow_result.candidate_mae:.6f}")
                print(f"    Active MAE:        {shadow_result.active_mae:.6f}")
                print(f"    Candidate Dir Acc: {shadow_result.candidate_direction_acc:.2%}")
                print(f"    Active Dir Acc:    {shadow_result.active_direction_acc:.2%}")
                print(f"    Recommendation:    {'PROMOTE ✓' if shadow_result.recommend_promotion else 'KEEP ACTIVE ✗'}")
                print(f"    Reason:            {shadow_result.reason}")
                print()
        
        # Promoted models
        if report.promoted_models:
            print("PROMOTED MODELS:")
            print("-" * 80)
            for model_type in report.promoted_models:
                print(f"  ✓ {model_type.value} → ACTIVE")
            print()
        else:
            print("PROMOTED MODELS:")
            print("-" * 80)
            print("  None (no models met promotion criteria)")
            print()
        
        # Failed models
        if report.failed_models:
            print("FAILED MODELS:")
            print("-" * 80)
            for model_type in report.failed_models:
                print(f"  ✗ {model_type.value}")
            print()
        
        # ====================================================================
        # STEP 7: Show updated model status
        # ====================================================================
        
        print("Step 7: Updated Model Status")
        print("-" * 80)
        
        updated_status = clm.get_model_status_summary()
        
        if updated_status:
            for model_type, info in updated_status.items():
                active_ver = info.get("active_version", "none")
                trained = info.get("active_trained_at", "never")
                metrics = info.get("active_metrics", {})
                
                print(f"  {model_type:15s} | {active_ver:20s}")
                print(f"                    | Trained: {trained}")
                
                if metrics:
                    rmse = metrics.get("rmse", 0)
                    dir_acc = metrics.get("directional_accuracy", 0)
                    print(f"                    | RMSE: {rmse:.6f} | Dir Acc: {dir_acc:.2%}")
        print()
        
        print("=" * 80)
        print("INTEGRATION DEMO COMPLETE")
        print("=" * 80)
        print()
        print("Next Steps:")
        print("  1. Check backend/data/clm_registry.db for model metadata")
        print("  2. Check ai_engine/models/clm/ for saved model files")
        print("  3. Schedule CLM to run periodically (see CLM_README.md)")
        print("  4. Add FastAPI endpoints for manual triggers and status")
        print()
        
    except Exception as e:
        logger.error(f"Retraining cycle failed: {e}", exc_info=True)
        print()
        print("✗ Retraining cycle failed. Check logs for details.")
        print()


if __name__ == "__main__":
    main()
