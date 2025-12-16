"""
AI-HFOS INTEGRATION LAYER
==========================

Connects AI Hedgefund Operating System with all subsystems.

This module handles:
- Data collection from all subsystems
- Directive distribution to subsystems
- Real-time coordination
- Event-driven updates
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from .ai_hedgefund_os import AIHedgeFundOS, AIHFOSOutput, SystemRiskMode

logger = logging.getLogger(__name__)


class AIHFOSIntegration:
    """
    Integration layer between AI-HFOS and all subsystems.
    
    Responsibilities:
    - Collect data from subsystems
    - Feed data to AI-HFOS
    - Distribute directives back to subsystems
    - Monitor system state
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        update_interval_seconds: int = 60
    ):
        self.data_dir = Path(data_dir)
        self.update_interval = update_interval_seconds
        
        # Initialize AI-HFOS
        self.hfos = AIHedgeFundOS(data_dir=str(self.data_dir))
        
        # State tracking
        self.last_output: Optional[AIHFOSOutput] = None
        self.running = False
        
        logger.info("[AI-HFOS Integration] Initialized")
    
    # ========================================================================
    # DATA COLLECTION
    # ========================================================================
    
    async def collect_universe_data(self) -> Dict[str, Any]:
        """Collect data from Universe OS."""
        try:
            universe_file = self.data_dir.parent / "universe_selector_output.json"
            if universe_file.exists():
                with open(universe_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[AI-HFOS] Error loading universe data: {e}")
        
        return {
            "data_confidence": "UNKNOWN",
            "current_universe": {"symbol_count": 0},
            "classifications": {}
        }
    
    async def collect_risk_data(self) -> Dict[str, Any]:
        """Collect data from Risk OS."""
        # TODO: Integrate with actual Risk Manager
        return {
            "emergency_brake_active": False,
            "daily_dd_pct": 0.0,
            "open_dd_pct": 0.0,
            "max_daily_dd_pct": 5.0,
            "risk_profile": "NORMAL"
        }
    
    async def collect_positions_data(self) -> Dict[str, Any]:
        """Collect data from Position Intelligence Layer."""
        # TODO: Integrate with actual PIL
        return {
            "position_count": 0,
            "toxic_count": 0,
            "winner_count": 0,
            "positions": []
        }
    
    async def collect_execution_data(self) -> Dict[str, Any]:
        """Collect data from Execution Layer."""
        # TODO: Integrate with actual Execution Layer
        return {
            "avg_slippage_bps": 0.0,
            "avg_spread_bps": 0.0,
            "fill_rate": 1.0,
            "recent_trades": []
        }
    
    async def collect_model_performance(self) -> Dict[str, Any]:
        """Collect data from Model Supervisor."""
        # TODO: Integrate with actual Model Supervisor
        return {
            "ensemble_accuracy": 0.0,
            "degraded_models": [],
            "model_weights": {}
        }
    
    async def collect_self_healing_report(self) -> Dict[str, Any]:
        """Collect data from Self-Healing System."""
        try:
            healing_file = self.data_dir / "self_healing_report.json"
            if healing_file.exists():
                with open(healing_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[AI-HFOS] Error loading self-healing report: {e}")
        
        return {
            "overall_status": "UNKNOWN",
            "subsystems": []
        }
    
    async def collect_pal_report(self) -> Dict[str, Any]:
        """Collect data from Profit Amplification Layer."""
        try:
            pal_file = self.data_dir / "profit_amplification_report.json"
            if pal_file.exists():
                with open(pal_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[AI-HFOS] Error loading PAL report: {e}")
        
        return {
            "amplification_candidates": [],
            "recommendations": []
        }
    
    async def collect_orchestrator_policy(self) -> Dict[str, Any]:
        """Collect data from Orchestrator Policy."""
        # TODO: Integrate with actual Orchestrator
        return {
            "regime": "UNKNOWN",
            "risk_profile": "NORMAL",
            "exit_mode": "FAST_TP"
        }
    
    async def collect_all_data(self) -> Dict[str, Dict[str, Any]]:
        """Collect data from all subsystems in parallel."""
        logger.info("[AI-HFOS] Collecting data from all subsystems...")
        
        results = await asyncio.gather(
            self.collect_universe_data(),
            self.collect_risk_data(),
            self.collect_positions_data(),
            self.collect_execution_data(),
            self.collect_model_performance(),
            self.collect_self_healing_report(),
            self.collect_pal_report(),
            self.collect_orchestrator_policy(),
            return_exceptions=True
        )
        
        return {
            "universe_data": results[0] if not isinstance(results[0], Exception) else {},
            "risk_data": results[1] if not isinstance(results[1], Exception) else {},
            "positions_data": results[2] if not isinstance(results[2], Exception) else {},
            "execution_data": results[3] if not isinstance(results[3], Exception) else {},
            "model_performance": results[4] if not isinstance(results[4], Exception) else {},
            "self_healing_report": results[5] if not isinstance(results[5], Exception) else {},
            "pal_report": results[6] if not isinstance(results[6], Exception) else {},
            "orchestrator_policy": results[7] if not isinstance(results[7], Exception) else {}
        }
    
    # ========================================================================
    # DIRECTIVE DISTRIBUTION
    # ========================================================================
    
    async def apply_global_directives(self, output: AIHFOSOutput):
        """Apply global directives to all subsystems."""
        directives = output.global_directives
        
        logger.info(f"[AI-HFOS] Applying global directives:")
        logger.info(f"  - Allow new trades: {directives.allow_new_trades}")
        logger.info(f"  - Allow new positions: {directives.allow_new_positions}")
        logger.info(f"  - Scale position sizes: {directives.scale_position_sizes:.1%}")
        
        # TODO: Implement actual directive application
        # Example:
        # await risk_manager.set_allow_new_trades(directives.allow_new_trades)
        # await portfolio_balancer.set_position_size_scale(directives.scale_position_sizes)
    
    async def apply_universe_directives(self, output: AIHFOSOutput):
        """Apply universe directives."""
        directives = output.universe_directives
        
        logger.info(f"[AI-HFOS] Applying universe directives:")
        logger.info(f"  - Universe mode: {directives.universe_mode}")
        
        # TODO: Implement actual directive application
        # await universe_os.set_mode(directives.universe_mode)
    
    async def apply_execution_directives(self, output: AIHFOSOutput):
        """Apply execution directives."""
        directives = output.execution_directives
        
        logger.info(f"[AI-HFOS] Applying execution directives:")
        logger.info(f"  - Order type preference: {directives.order_type_preference}")
        logger.info(f"  - Max slippage: {directives.max_slippage_bps} bps")
        
        # TODO: Implement actual directive application
        # await execution_layer.set_preferences(directives)
    
    async def apply_portfolio_directives(self, output: AIHFOSOutput):
        """Apply portfolio directives."""
        directives = output.portfolio_directives
        
        logger.info(f"[AI-HFOS] Applying portfolio directives:")
        logger.info(f"  - Reduce exposure: {directives.reduce_exposure_pct:.0f}%")
        
        # TODO: Implement actual directive application
        # await portfolio_balancer.set_exposure_reduction(directives.reduce_exposure_pct)
    
    async def apply_model_directives(self, output: AIHFOSOutput):
        """Apply model directives."""
        directives = output.model_directives
        
        logger.info(f"[AI-HFOS] Applying model directives:")
        logger.info(f"  - Conservative predictions: {directives.use_conservative_predictions}")
        
        # TODO: Implement actual directive application
        # await model_supervisor.set_conservative_mode(directives.use_conservative_predictions)
    
    async def execute_emergency_actions(self, output: AIHFOSOutput):
        """Execute emergency actions."""
        if not output.emergency_actions:
            return
        
        logger.warning(f"[AI-HFOS] ⚠️  {len(output.emergency_actions)} EMERGENCY ACTIONS REQUIRED")
        
        for action in sorted(output.emergency_actions, key=lambda a: a.priority):
            logger.critical(
                f"[AI-HFOS] EMERGENCY: {action.action_type} on {action.target} - "
                f"{action.rationale}"
            )
            
            # TODO: Implement emergency action execution
            # if action.action_type == "CLOSE_ALL_POSITIONS":
            #     await execution_layer.close_all_positions()
            # elif action.action_type == "PAUSE_NEW_TRADES":
            #     await risk_manager.pause_trading()
    
    async def process_amplification_opportunities(self, output: AIHFOSOutput):
        """Process amplification opportunities."""
        if not output.amplification_opportunities:
            return
        
        logger.info(f"[AI-HFOS] 📈 {len(output.amplification_opportunities)} amplification opportunities")
        
        for opp in output.amplification_opportunities:
            logger.info(
                f"[AI-HFOS] Opportunity: {opp.symbol} {opp.action} - "
                f"Expected +{opp.expected_r_increase:.2f}R ({opp.confidence:.0f}% confidence)"
            )
            
            # TODO: Implement amplification execution
            # await execution_layer.execute_amplification(opp)
    
    async def apply_all_directives(self, output: AIHFOSOutput):
        """Apply all directives from AI-HFOS output."""
        logger.info("[AI-HFOS] Applying directives to all subsystems...")
        
        await asyncio.gather(
            self.apply_global_directives(output),
            self.apply_universe_directives(output),
            self.apply_execution_directives(output),
            self.apply_portfolio_directives(output),
            self.apply_model_directives(output),
            self.execute_emergency_actions(output),
            self.process_amplification_opportunities(output),
            return_exceptions=True
        )
        
        logger.info("[AI-HFOS] All directives applied")
    
    # ========================================================================
    # COORDINATION LOOP
    # ========================================================================
    
    async def run_coordination_cycle(self):
        """Run one complete coordination cycle."""
        try:
            # 1. Collect data from all subsystems
            all_data = await self.collect_all_data()
            
            # 2. Run AI-HFOS analysis
            output = self.hfos.run_coordination_cycle(
                universe_data=all_data["universe_data"],
                risk_data=all_data["risk_data"],
                positions_data=all_data["positions_data"],
                execution_data=all_data["execution_data"],
                model_performance=all_data["model_performance"],
                self_healing_report=all_data["self_healing_report"],
                pal_report=all_data["pal_report"],
                orchestrator_policy=all_data["orchestrator_policy"]
            )
            
            # 3. Apply directives
            await self.apply_all_directives(output)
            
            # 4. Store output
            self.last_output = output
            
            # 5. Log summary
            logger.info("=" * 80)
            logger.info("[AI-HFOS] COORDINATION CYCLE COMPLETE")
            logger.info("=" * 80)
            logger.info(output.summary)
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"[AI-HFOS] Error in coordination cycle: {e}", exc_info=True)
    
    async def run_continuous(self):
        """Run continuous coordination loop."""
        self.running = True
        logger.info(f"[AI-HFOS] Starting continuous coordination (interval: {self.update_interval}s)")
        
        while self.running:
            await self.run_coordination_cycle()
            await asyncio.sleep(self.update_interval)
    
    async def start(self):
        """Start the coordination loop as a background task."""
        if not self.running:
            self._task = asyncio.create_task(self.run_continuous())
            logger.info("[AI-HFOS] Coordination loop started as background task")
    
    def stop(self):
        """Stop continuous coordination."""
        logger.info("[AI-HFOS] Stopping coordination loop")
        self.running = False
        if hasattr(self, '_task') and self._task:
            self._task.cancel()
    
    # ========================================================================
    # QUERY INTERFACE
    # ========================================================================
    
    def get_current_risk_mode(self) -> Optional[SystemRiskMode]:
        """Get current system risk mode."""
        if self.last_output:
            return self.last_output.system_risk_mode
        return None
    
    def get_current_directives(self) -> Optional[Dict[str, Any]]:
        """Get current directives."""
        if not self.last_output:
            return None
        
        return {
            "global": self.last_output.global_directives,
            "universe": self.last_output.universe_directives,
            "execution": self.last_output.execution_directives,
            "portfolio": self.last_output.portfolio_directives,
            "model": self.last_output.model_directives
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.last_output:
            return {"status": "not_initialized"}
        
        return {
            "risk_mode": self.last_output.system_risk_mode.value,
            "health": self.last_output.system_health.value,
            "timestamp": self.last_output.timestamp,
            "emergency_actions": len(self.last_output.emergency_actions),
            "conflicts": len(self.last_output.detected_conflicts),
            "amplification_opportunities": len(self.last_output.amplification_opportunities)
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of AI-HFOS Integration."""
    print("=" * 80)
    print("AI-HFOS INTEGRATION - Example Usage")
    print("=" * 80)
    print()
    
    # Initialize integration
    integration = AIHFOSIntegration(
        data_dir="./data",
        update_interval_seconds=60
    )
    
    # Run one coordination cycle
    print("[EXAMPLE] Running single coordination cycle...")
    await integration.run_coordination_cycle()
    
    print()
    print("[EXAMPLE] System status:")
    status = integration.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print()
    print("✅ Example complete")
    print()
    print("To run continuous coordination:")
    print("  await integration.run_continuous()")


if __name__ == "__main__":
    asyncio.run(main())
