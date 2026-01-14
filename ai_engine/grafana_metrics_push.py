#!/usr/bin/env python3
"""
Grafana Metrics Push Module
Sends ensemble metrics directly to Grafana endpoint as JSON
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


class GrafanaMetricsPusher:
    """
    Push ensemble metrics directly to Grafana HTTP endpoint.
    Non-blocking async implementation.
    """
    
    def __init__(self, grafana_url: str = "https://app.quantumfond.com/grafana/api/quantum/ensemble/metrics"):
        self.grafana_url = grafana_url
        self.enabled = True
        self.timeout = aiohttp.ClientTimeout(total=2)  # 2s timeout
        
    async def push_metrics(self, 
                          symbol: str,
                          action: str,
                          confidence: float,
                          info: Dict[str, Any],
                          active_predictions: Dict[str, Any],
                          weights: Dict[str, float]) -> bool:
        """
        Push ensemble prediction metrics to Grafana.
        
        Args:
            symbol: Trading pair
            action: BUY/SELL/HOLD
            confidence: Final ensemble confidence
            info: Ensemble metadata
            active_predictions: Active model predictions
            weights: Model weights
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        # Build metrics payload
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "consensus": info.get("consensus", 0),
            "active_models": list(active_predictions.keys()),
            "inactive_models": list(info.get("inactive_models", [])),
            "weights": weights,
            "model_predictions": {},
            "metadata": {
                "meta_override": info.get("meta_override", False),
                "meta_score": info.get("meta_score"),
                "governer_status": info.get("governer_status"),
                "effective_leverage": info.get("effective_leverage")
            }
        }
        
        # Add individual model predictions
        for model_name, pred in active_predictions.items():
            if isinstance(pred, dict):
                metrics["model_predictions"][model_name] = {
                    "action": pred.get("action"),
                    "confidence": pred.get("confidence"),
                    "version": pred.get("version")
                }
            else:
                # Tuple format
                metrics["model_predictions"][model_name] = {
                    "action": pred[0],
                    "confidence": pred[1],
                    "version": str(pred[2]) if len(pred) > 2 else "unknown"
                }
        
        # Push to Grafana
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.grafana_url,
                    json=metrics,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.debug(f"[GRAFANA] Metrics pushed: {symbol} {action}")
                        return True
                    else:
                        logger.warning(f"[GRAFANA] Push failed: HTTP {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.warning("[GRAFANA] Push timeout (2s) - continuing without Grafana")
            return False
        except Exception as e:
            logger.warning(f"[GRAFANA] Push error: {e} - continuing without Grafana")
            return False
    
    def push_sync(self, *args, **kwargs) -> bool:
        """
        Synchronous wrapper for push_metrics.
        Creates event loop if needed.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create task
                asyncio.create_task(self.push_metrics(*args, **kwargs))
                return True
            else:
                # Run in new loop
                return asyncio.run(self.push_metrics(*args, **kwargs))
        except Exception as e:
            logger.warning(f"[GRAFANA] Sync push failed: {e}")
            return False
