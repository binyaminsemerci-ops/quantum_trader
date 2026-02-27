"""
TEST ENDPOINT FOR HYBRID AGENT
===============================

Test the Hybrid Agent (TFT + XGBoost) before full deployment.
Safe validation endpoint that doesn't affect live trading.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import logging
import os

router = APIRouter(prefix="/api/test/hybrid", tags=["test-hybrid"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def hybrid_health_check() -> Dict:
    """Check if Hybrid Agent can be loaded and is functional."""
    try:
        from ai_engine.agents.hybrid_agent import HybridAgent
        
        agent = HybridAgent()
        
        return {
            "status": "healthy",
            "mode": agent.mode,
            "tft_loaded": agent.tft_loaded,
            "xgb_loaded": agent.xgb_loaded,
            "weights": {
                "tft": agent.tft_weight,
                "xgb": agent.xgb_weight,
                "agreement_bonus": agent.agreement_bonus
            },
            "message": f"Hybrid Agent operational in {agent.mode} mode"
        }
    except Exception as e:
        logger.error(f"Hybrid Agent health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid Agent error: {str(e)}")


@router.post("/predict")
async def test_hybrid_prediction(symbol: str, features: Optional[Dict] = None) -> Dict:
    """Test prediction with Hybrid Agent for a single symbol.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        features: Optional feature dict. If not provided, fetches live data.
    
    Returns:
        Prediction results with both model outputs and combined result
    """
    try:
        from ai_engine.agents.hybrid_agent import HybridAgent
        
        agent = HybridAgent()
        
        # If no features provided, fetch live data
        if not features:
            from backend.routes.external_data import binance_ohlcv
            
            candles = await binance_ohlcv(symbol, limit=240)
            if not candles or 'candles' not in candles:
                raise HTTPException(status_code=404, detail=f"Cannot fetch data for {symbol}")
            
            # Compute features from latest candle
            from ai_engine.feature_engineer import compute_all_indicators
            import pandas as pd
            
            df = pd.DataFrame(candles['candles'])
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise HTTPException(status_code=500, detail=f"Missing column: {col}")
            
            # Rename to standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            df_features = compute_all_indicators(df, use_advanced=False)
            
            feature_cols = [
                'Close', 'Volume', 'EMA_10', 'EMA_50', 'RSI_14',
                'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower',
                'ATR', 'volume_sma_20', 'price_change_pct', 'high_low_range'
            ]
            
            # Check if all features are available
            missing = [col for col in feature_cols if col not in df_features.columns]
            if missing:
                raise HTTPException(status_code=500, detail=f"Missing features: {missing}")
            
            last_row = df_features.iloc[-1]
            features = {col: float(last_row[col]) for col in feature_cols}
        
        # Get prediction
        action, confidence = agent.predict_direction(features)
        
        # Get individual model predictions for comparison
        tft_action, tft_conf = "HOLD", 0.5
        xgb_action, xgb_conf = "HOLD", 0.5
        
        if agent.tft_loaded:
            tft_action, tft_conf = agent.tft_agent.predict_single(features)
        
        if agent.xgb_loaded:
            xgb_action, xgb_conf = agent.xgb_agent.predict_direction(features)
        
        return {
            "symbol": symbol,
            "hybrid_prediction": {
                "action": action,
                "confidence": confidence
            },
            "tft_prediction": {
                "action": tft_action,
                "confidence": tft_conf,
                "loaded": agent.tft_loaded
            },
            "xgb_prediction": {
                "action": xgb_action,
                "confidence": xgb_conf,
                "loaded": agent.xgb_loaded
            },
            "agreement": tft_action == xgb_action,
            "mode": agent.mode,
            "weights": {
                "tft": agent.tft_weight,
                "xgb": agent.xgb_weight
            }
        }
        
    except Exception as e:
        logger.error(f"Hybrid prediction test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_models(symbols: str = "BTCUSDT,ETHUSDT,BNBUSDT") -> Dict:
    """Compare predictions from XGBoost, TFT, and Hybrid for multiple symbols.
    
    Args:
        symbols: Comma-separated list of symbols
    
    Returns:
        Comparison of all three models
    """
    try:
        from ai_engine.agents.xgb_agent import XGBAgent
        from ai_engine.agents.tft_agent import TFTAgent
        from ai_engine.agents.hybrid_agent import HybridAgent
        from backend.routes.external_data import binance_ohlcv
        from ai_engine.feature_engineer import compute_all_indicators
        import pandas as pd
        
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Initialize all agents
        xgb_agent = XGBAgent()
        tft_agent = TFTAgent()
        hybrid_agent = HybridAgent()
        
        tft_agent.load_model()
        
        results = []
        
        for symbol in symbol_list[:5]:  # Limit to 5 symbols
            try:
                # Fetch data
                candles = await binance_ohlcv(symbol, limit=240)
                if not candles or 'candles' not in candles:
                    continue
                
                df = pd.DataFrame(candles['candles'])
                
                # Rename columns
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low', 
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                df_features = compute_all_indicators(df, use_advanced=False)
                
                feature_cols = [
                    'Close', 'Volume', 'EMA_10', 'EMA_50', 'RSI_14',
                    'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower',
                    'ATR', 'volume_sma_20', 'price_change_pct', 'high_low_range'
                ]
                
                last_row = df_features.iloc[-1]
                features = {col: float(last_row[col]) for col in feature_cols}
                
                # Feed history to TFT
                for idx, row in df_features.tail(120).iterrows():
                    hist_features = {col: row[col] for col in feature_cols}
                    tft_agent.add_to_history(symbol, hist_features)
                
                # Get predictions
                xgb_action, xgb_conf = xgb_agent.predict_direction(features)
                tft_action, tft_conf, tft_meta = tft_agent.predict(symbol, features)
                hybrid_action, hybrid_conf = hybrid_agent.predict_direction(features)
                
                results.append({
                    "symbol": symbol,
                    "price": features['Close'],
                    "xgb": {"action": xgb_action, "confidence": xgb_conf},
                    "tft": {
                        "action": tft_action, 
                        "confidence": tft_conf,
                        "rr_ratio": tft_meta.get('risk_reward_ratio', 0)
                    },
                    "hybrid": {"action": hybrid_action, "confidence": hybrid_conf},
                    "agreement": {
                        "all_agree": (xgb_action == tft_action == hybrid_action),
                        "tft_xgb_agree": (tft_action == xgb_action),
                        "hybrid_tft_agree": (hybrid_action == tft_action)
                    }
                })
                
            except Exception as e:
                logger.warning(f"Failed to compare {symbol}: {e}")
                continue
        
        # Summary statistics
        total = len(results)
        all_agree = sum(1 for r in results if r['agreement']['all_agree'])
        tft_xgb_agree = sum(1 for r in results if r['agreement']['tft_xgb_agree'])
        
        return {
            "results": results,
            "summary": {
                "total_symbols": total,
                "all_models_agree": all_agree,
                "all_agree_pct": (all_agree / total * 100) if total > 0 else 0,
                "tft_xgb_agree": tft_xgb_agree,
                "tft_xgb_agree_pct": (tft_xgb_agree / total * 100) if total > 0 else 0
            },
            "models": {
                "xgb": "XGBoost Agent",
                "tft": "Temporal Fusion Transformer",
                "hybrid": "TFT + XGBoost Ensemble"
            }
        }
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_hybrid_config() -> Dict:
    """Get current AI model configuration and environment settings."""
    
    ai_model = os.getenv('AI_MODEL', 'hybrid')
    
    return {
        "current_mode": ai_model,
        "available_modes": {
            "xgb": "XGBoost only (fast, proven)",
            "tft": "Temporal Fusion Transformer only (temporal patterns)",
            "hybrid": "TFT + XGBoost ensemble (best performance)"
        },
        "instructions": {
            "change_mode": "Set environment variable: AI_MODEL=xgb|tft|hybrid",
            "windows": "setx AI_MODEL hybrid",
            "linux": "export AI_MODEL=hybrid",
            "docker": "Add to docker-compose.yml: environment: - AI_MODEL=hybrid"
        },
        "recommendation": "Use 'hybrid' for best results (60% TFT, 40% XGBoost)"
    }
